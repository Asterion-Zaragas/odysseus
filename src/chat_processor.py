# src/chat_processor.py
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from src.chat_helpers import extract_urls
from src.youtube_handler import is_youtube_url
from src.search import comprehensive_web_search, fetch_webpage_content
from src.prompt_security import UNTRUSTED_CONTEXT_POLICY, untrusted_context_message
from services.memory.retrieval import retrieve as memory_retrieve
from services.memory.tier_scoring import TIER_CORE

logger = logging.getLogger(__name__)


# Straight and typographic quote pairs a model may wrap its reply in.
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


def _strip_wrapping_quotes(text: str) -> str:
    """Drop quote characters that wrap the whole query.

    "Reply ONLY with the query" reliably produces a quoted reply from some
    models (observed: ``"current weather"``). Passed through verbatim those
    quotes become an exact-phrase operator on most engines, which silently
    narrows the result set to near-nothing. Only balanced pairs enclosing the
    *entire* string are removed, so a genuine phrase search the user typed
    themselves (``berlin "hidden gems"``) is left alone.
    """
    text = text.strip()
    while len(text) >= 2:
        for opener, closer in _QUOTE_PAIRS:
            if text[0] == opener and text[-1] == closer:
                # Balanced only: an inner quote means the marks are load-bearing.
                if closer not in text[1:-1]:
                    text = text[1:-1].strip()
                    break
        else:
            break
    return text


def _clean_search_query(query: str, max_len: int = 200) -> str:
    """Strip fenced code blocks from a search query while preserving inline
    code text.

    This is a focused, defensive cleanup for the *final* web-search query
    selected in ``build_context_preface`` (issue #4547): regardless of whether
    the query came from the LLM-generated path (#4557) or the first-line
    fallback, residual fenced / inline markdown should not leak into the search
    call. Rather than using regex (which is brittle and strips inline code
    text like ``git reset`` from the query), we render the query to HTML via
    ``markdown`` and parse it with ``BeautifulSoup`` so that:

    * ``<pre>`` blocks (fenced / indented code) are removed entirely.
    * ``<code>`` elements (inline code) are preserved as plain text.

    Both libraries are already project dependencies. The result is whitespace
    collapsed and truncated to ``max_len``; an all-code input collapses to an
    empty string, which the caller treats as "no query".
    """
    import markdown as _md
    from bs4 import BeautifulSoup as _BS

    html = _md.markdown(query, extensions=["fenced_code"])
    soup = _BS(html, "html.parser")

    # Remove fenced / indented code blocks.
    for pre in soup.find_all("pre"):
        pre.decompose()

    # Preserve inline code by unwrapping <code> to text.
    for code in soup.find_all("code"):
        code.replace_with(code.get_text())

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = _strip_wrapping_quotes(text)
    return text[:max_len]


# Vocabulary from the query-extraction system prompt. If it comes back out of
# the model, the model restated its instructions instead of following them.
_QUERY_ECHO_MARKERS = (
    "search query",
    "user message:",
    "reply only",
    "goal:",
    "constraint:",
)


def _is_plausible_search_query(generated: str, message: str) -> bool:
    """Return True if ``generated`` looks like an extracted query, not an echo.

    The extraction step in ``build_context_preface`` asks a model to distil a
    search query from the user's message, and previously accepted any non-empty
    reply. Weaker instruct models routinely paraphrase the task instead
    ("User message: "..." * Goal: Extract a concise search query. * Constraint:
    Reply ONLY"), which was then truncated to 150 chars and sent to SearXNG
    verbatim — producing results unrelated to anything the user asked, and
    long enough that some engines answered 403.

    Two cheap signals, either of which sends us back to the first-line
    fallback: the reply is longer than the message it supposedly summarises,
    or it contains the extraction prompt's own vocabulary.
    """
    q = generated.strip()
    if not q:
        return False
    # A distilled query is never longer than its source. The floor keeps short
    # messages ("berlin?") from rejecting a reasonable expansion.
    if len(q) > max(len(message), 80):
        return False
    lowered = q.lower()
    return not any(marker in lowered for marker in _QUERY_ECHO_MARKERS)


class ChatProcessor:
    def __init__(self, memory_manager, personal_docs_manager, memory_vector=None, skills_manager=None):
        self.memory_manager = memory_manager
        self.personal_docs_manager = personal_docs_manager
        self.memory_vector = memory_vector
        self.skills_manager = skills_manager

    # Minimum similarity score for RAG results to be injected
    RAG_SIMILARITY_THRESHOLD = 0.35
    MEMORY_CONTEXT_LIMIT = 5
    PINNED_MEMORY_LIMIT = MEMORY_CONTEXT_LIMIT

    def _is_core_memory(self, memory: Dict[str, Any]) -> bool:
        """Return whether a pinned/core memory is safe to keep globally available."""
        tags = memory.get("tags") or []
        if "identity" in tags or "contact" in tags:
            return True
        text = (memory.get("text") or "").lower()
        return any(marker in text for marker in (
            "my name is",
            "name is",
            "call me",
            "i am ",
            "i'm ",
            "email",
            "phone",
            "address",
        ))

    async def _select_core_memories(
        self, message: str, core: list, *, owner: Optional[str], effort: str,
    ) -> list:
        """Keep core/pinned memories high-priority without injecting all of them.

        Pinned/tier-0 used to mean "always send every one of these to the
        model". That bloats every request and leaks unrelated personal
        context into tasks that don't need it. Only a small set of core
        identity/contact memories is always available; the rest must match
        the current request (via the staged retrieval pipeline), but are
        preferred ahead of ordinary, non-core memories.
        """
        if not core:
            return []

        def _recent_first(memory: Dict[str, Any]) -> int:
            try:
                return int(memory.get("timestamp") or 0)
            except Exception:
                return 0

        identity = sorted(
            [m for m in core if self._is_core_memory(m)],
            key=_recent_first,
            reverse=True,
        )[:self.PINNED_MEMORY_LIMIT]

        identity_ids = {m.get("id") for m in identity if m.get("id")}
        contextual_candidates = [
            m for m in core
            if not (m.get("id") and m.get("id") in identity_ids)
        ]
        remaining_slots = max(self.PINNED_MEMORY_LIMIT - len(identity), 0)
        contextual = []
        if remaining_slots and contextual_candidates:
            result = await memory_retrieve(
                message, contextual_candidates, effort=effort, memory_vector=self.memory_vector,
                owner=owner, k=remaining_slots, interactive=True,
            )
            contextual = result["memories"]

        selected = []
        seen = set()
        for memory in [*identity, *contextual]:
            key = memory.get("id") or memory.get("text")
            if key in seen:
                continue
            seen.add(key)
            selected.append(memory)
        return selected[:self.PINNED_MEMORY_LIMIT]

    async def build_context_preface(
        self,
        message: str,
        session: Any,
        use_web: bool = False,
        use_rag: bool = True,
        use_memory: bool = True,
        time_filter: Optional[str] = None,
        preset_system_prompt: Optional[str] = None,
        owner: Optional[str] = None,
        character_name: Optional[str] = None,
        agent_mode: bool = False,
        incognito: bool = False,
        use_skills: bool = True,
        memory_effort: Optional[str] = None,
        use_memory_context_doc: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]]]:
        """Build the context preface for LLM calls.

        Returns:
            Tuple of (preface messages, rag_sources list)

        Note on KV-cache friendliness: the ``system``-role messages assembled
        here are later concatenated into a single system message and sent as
        the very first thing in the payload (see ``llm_core``'s "consolidate
        system messages" step). Local OpenAI-compatible backends (llama.cpp /
        LM Studio) key their KV cache off the byte-identical token prefix, so
        *anything* that changes turn-to-turn — timestamps, retrieved snippets,
        per-turn counts — must NOT be folded into a system message here. Such
        content belongs in a separate ``user``/context message appended near
        the end of the array (see ``current_datetime_context_message`` and
        ``untrusted_context_message`` callers in ``build_chat_context``),
        which keeps the static system prefix byte-identical across turns of
        the same session and lets the backend reuse its cached prefix.
        """
        preface = []
        rag_sources = []

        # Add preset system prompt if specified
        if preset_system_prompt:
            preface.append({
                "role": "system",
                "content": preset_system_prompt
            })
        preface.append({
            "role": "system",
            "content": UNTRUSTED_CONTEXT_POLICY,
        })

        # Memory: core (pinned + tier-0, always included — optionally replaced
        # by the context-doc render) + rest (staged retrieval when relevant).
        self._last_used_memories = []  # track what was injected
        if use_memory:
            from src.settings import get_setting

            if use_memory_context_doc is None:
                use_memory_context_doc = bool(get_setting("memory_context_doc_injection", False))
            if not memory_effort:
                memory_effort = get_setting("memory_retrieval_effort", "medium")

            mem_entries = self.memory_manager.load(owner=owner)
            core = [
                m for m in mem_entries
                if m.get("pinned") or int(m.get("tier") if m.get("tier") is not None else 2) == TIER_CORE
            ]
            core_ids = {m.get("id") for m in core}
            rest = [m for m in mem_entries if m.get("id") not in core_ids]

            _used_ids: list = []
            if use_memory_context_doc:
                from services.memory.memory_context import MemoryContext

                doc_text = MemoryContext(owner).render_markdown()
                preface.append(untrusted_context_message("saved memory: context document", doc_text))
                # Pinned entries are considered for individual injection
                # regardless of the toggle (capped/relevance-filtered below so
                # a long pinned list doesn't bloat every request); non-pinned
                # tier-0 entries are covered by the doc's core-facts section
                # above, so they're suppressed here to avoid duplication.
                pinned_only = [m for m in core if m.get("pinned")]
                selected_pinned = await self._select_core_memories(
                    message, pinned_only, owner=owner, effort=memory_effort,
                )
                if selected_pinned:
                    pinned_text = "\n- ".join([m["text"] for m in selected_pinned])
                    preface.append(untrusted_context_message(
                        "saved memory: pinned user facts",
                        f"Core facts about the user:\n- {pinned_text}",
                    ))
                    for m in selected_pinned:
                        self._last_used_memories.append({"text": m["text"], "tags": m.get("tags") or [], "type": "pinned"})
                        if m.get("id"):
                            _used_ids.append(m["id"])
            elif core:
                selected_core = await self._select_core_memories(
                    message, core, owner=owner, effort=memory_effort,
                )
                if selected_core:
                    core_text = "\n- ".join([m["text"] for m in selected_core])
                    preface.append(untrusted_context_message(
                        "saved memory: pinned user facts",
                        f"Core facts about the user:\n- {core_text}",
                    ))
                    for m in selected_core:
                        self._last_used_memories.append({
                            "text": m["text"], "tags": m.get("tags") or [],
                            "type": "pinned" if m.get("pinned") else "core",
                        })
                        if m.get("id"):
                            _used_ids.append(m["id"])

            if rest:
                recall_k = int(get_setting("memory_recall_k", 3) or 3)
                result = await memory_retrieve(
                    message, rest, effort=memory_effort, memory_vector=self.memory_vector,
                    owner=owner, k=recall_k, interactive=True,
                )
                relevant = result["memories"]
                if relevant:
                    ext_text = "\n".join([f"- {m['text']}" for m in relevant])
                    preface.append(untrusted_context_message(
                        "saved memory: retrieved context",
                        (
                            "Memory context. Do not reference unless the user asks "
                            f"about these topics.\n{ext_text}"
                        ),
                    ))
                    for m in relevant:
                        self._last_used_memories.append({"text": m["text"], "tags": m.get("tags") or [], "type": "recalled"})
                        if m.get("id"):
                            _used_ids.append(m["id"])

            # Bump usage counters for the memories that were actually injected.
            if _used_ids and hasattr(self.memory_manager, "increment_uses"):
                try:
                    self.memory_manager.increment_uses(_used_ids)
                except Exception as _e:
                    logger.warning("Failed to increment memory uses: %s", _e)

            # (skills index injection moved out — see below; only fires in
            # agent mode so chat mode and incognito stay clean.)

        # RAG: search if enabled and rag_manager available, inject only above threshold
        if use_rag:
            try:
                rag_manager = getattr(self.personal_docs_manager, 'rag_manager', None)
                if rag_manager:
                    results = rag_manager.search(message, k=5, owner=owner)
                    # Filter by similarity threshold
                    relevant = [r for r in results if r.get("similarity", 0) >= self.RAG_SIMILARITY_THRESHOLD]
                    if relevant:
                        logger.info(f"RAG: {len(relevant)}/{len(results)} results above threshold {self.RAG_SIMILARITY_THRESHOLD}")
                        rag_sources = [
                            {
                                "filename": r["metadata"].get("filename", r["metadata"].get("source", "unknown")),
                                "snippet": r["document"][:200],
                                "similarity": round(r.get("similarity", 0), 3)
                            }
                            for r in relevant
                        ]
                        rag_content = "Relevant documents:\n\n" + "\n\n---\n\n".join(
                            f"[{s['filename']}]\n{r['document']}" for s, r in zip(rag_sources, relevant)
                        )
                        if len(rag_content) > 10000:
                            rag_content = rag_content[:10000] + "\n[Truncated]"
                        preface.append(untrusted_context_message("retrieved documents", rag_content))
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Add web search if enabled
        web_sources = []
        if use_web:
            try:
                from src.llm_core import llm_call

                t_url, t_model, t_headers = session.endpoint_url, session.model, session.headers

                # Default fallback is the first non-empty line of the original user message
                fallback_query = next((line.strip() for line in message.split("\n") if line.strip()), "")
                search_query = fallback_query

                try:
                    generated_query = llm_call(
                        t_url,
                        t_model,
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Extract a concise search query from the user's message. "
                                    "Reply ONLY with the query."
                                ),
                            },
                            {"role": "user", "content": message},
                        ],
                        headers=t_headers,
                        temperature=0.1,
                        max_tokens=50,
                        timeout=15,
                    ).strip()

                    if not generated_query:
                        # LLM returned an empty or whitespace-only query -> fall back to original query
                        logger.warning("LLM generated an empty search query, using fallback.")
                    elif _is_plausible_search_query(generated_query, message):
                        # LLM successfully generated a usable query -> use it
                        search_query = generated_query
                    else:
                        # LLM restated its instructions instead of extracting a
                        # query; searching that returns nonsense, so fall back.
                        logger.warning(
                            "LLM query extraction echoed the prompt instead of a query (%r), using fallback.",
                            generated_query[:120],
                        )
                except Exception as e:
                    # LLM failed (exception/error) -> fall back to original user query
                    logger.warning(f"Failed to generate search query via LLM, using fallback: {e}")

                search_query = " ".join(search_query.split())
                if len(search_query) > 150:
                    search_query = search_query[:150].strip()

                # Defensive cleanup of the final selected query (interim fix
                # for #4547): strip any residual fenced/inline markdown so that
                # neither the generated query nor the first-line fallback leaks
                # fences or backticks into the search call. No-op on clean
                # generated queries; collapses to "" when the query is all code.
                search_query = _clean_search_query(search_query, max_len=150)

                if search_query:
                    # Execute web search using the final selected query
                    web_context, web_sources = comprehensive_web_search(
                        search_query, time_filter=time_filter, return_sources=True
                    )
                    preface.append(untrusted_context_message("web search results", web_context))
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                preface.append({"role": "system", "content": "Web search encountered an error and could not retrieve results."})

        # Process non-YouTube URLs in message (YouTube handled by preprocess_message)
        # Skip auto-fetch for long pastes (the user already pasted the content —
        # fetching every embedded link buries the actual question under
        # hundreds of KB of duplicate page HTML and confuses the model) or for
        # link-heavy pastes (>3 URLs typically means it's a boilerplate-laden
        # blog post, not a "summarize this URL" request).
        urls = extract_urls(message)
        non_yt_urls = [u for u in urls if not is_youtube_url(u)]
        skip_url_fetch = len(message) > 2000 or len(non_yt_urls) > 3
        if not skip_url_fetch:
            for url in non_yt_urls:
                result = fetch_webpage_content(url)
                if result.get('success'):
                    content = result.get('content', '')[:10000]
                    preface.append(untrusted_context_message(
                        f"web page: {url}",
                        f"Content from {url}:\n\n{content}",
                    ))

        # Skills index — progressive disclosure. Only injected when the
        # model has the `manage_skills` tool available (agent_mode), and
        # never in incognito mode (the user has explicitly opted out of
        # context retention this turn). In plain chat mode the model can't
        # call the tool anyway, so the index would be noise.
        if agent_mode and not incognito and use_skills and self.skills_manager:
            try:
                idx = self.skills_manager.index_for(owner=owner)
            except Exception as e:
                logger.debug(f"Skills index unavailable: {e}")
                idx = []
            if idx:
                by_cat: Dict[str, list] = {}
                for s in idx:
                    by_cat.setdefault(s.get("category") or "general", []).append(s)
                lines = ["[Available skills — call manage_skills(action='view', name='...') to load one when relevant]"]
                for cat in sorted(by_cat):
                    lines.append(f"  {cat}:")
                    for s in sorted(by_cat[cat], key=lambda x: x["name"]):
                        desc = s.get("description") or ""
                        lines.append(f"    - {s['name']}: {desc}" if desc else f"    - {s['name']}")
                preface.append(untrusted_context_message("available skills index", "\n".join(lines)))

        return preface, rag_sources, web_sources
