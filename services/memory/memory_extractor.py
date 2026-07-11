"""
memory_extractor.py

Background auto-extraction of facts from chat conversations.
After each LLM response, this module sends the last few messages to the LLM
asking it to extract memorable facts, then stores them in both memory.json
and the FAISS vector index.

Full deduplication/consolidation, tag normalization, tier rescoring, and
archive expiry are the nightly curator's job now (services/memory/
memory_curator.py); this module only runs the cheap post-extraction triage
trigger (see AUDIT_INTERVAL below).
"""

import hashlib
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _fingerprint_entries(entries) -> str:
    """Stable hash of an owner's memories — order-independent, depends only
    on id+text+tags. Any add/edit/delete/retag invalidates it. Deliberately
    excludes uses/last_used_at: retrieval bumps those constantly and must not
    defeat the audit short-circuit."""
    items = sorted(
        (str(e.get("id", "")), e.get("text", ""), ",".join(sorted(e.get("tags") or [])))
        for e in _memory_dicts(entries)
    )
    h = hashlib.sha256()
    for triple in items:
        h.update(("\x1f".join(triple) + "\x1e").encode("utf-8"))
    return h.hexdigest()


def _memory_dicts(entries):
    for entry in entries or []:
        if isinstance(entry, dict):
            yield entry


EXTRACT_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Analyze the conversation and extract ONLY "
    "durable personal facts about the user that would be useful across many future conversations.\n\n"
    "Good examples: name, job title, city, family members, long-term projects, strong preferences.\n"
    "Bad examples: what they asked about today, temporary moods, generic statements, "
    "things the assistant said, one-off tasks, opinions on the current topic.\n\n"
    "Rules:\n"
    "- MAX 2 facts per conversation — only the most important\n"
    "- Only extract facts the USER stated or clearly implied\n"
    "- Each fact must be a single short sentence (under 15 words)\n"
    "- If a fact is similar to something likely already known, skip it\n"
    "- If nothing durable was revealed, return []\n\n"
    "Return a JSON array of objects with 'text' and 'category' fields.\n"
    "Categories: 'identity', 'preference', 'fact', 'contact', 'project', 'goal'\n\n"
    "Return ONLY valid JSON, no markdown fences."
)

# How many recent messages to include for extraction
CONTEXT_WINDOW = 6

AUDIT_INTERVAL = 5  # run the cheap triage pass every N new memories added
_extractions_since_audit = 0


def _message_text(message) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p).strip()
    return ""


def _message_role(message) -> str:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, dict):
        role = message.get("role")
    return str(role or "").lower()


def _clean_memory_value(value: str, max_len: int = 80) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" .,!?:;\"'`“”‘’")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    if not value or len(value) > max_len:
        return ""
    if re.search(r"https?://|@|[{}<>]", value):
        return ""
    return value


def _fallback_memory_candidates(messages) -> list[dict]:
    """Extract obvious durable facts without relying on the LLM.

    This is deliberately narrow. The LLM remains the main extractor, but
    simple identity/preference/goal statements should not silently vanish just
    because the background model judged them too conversational.
    """
    candidates = []
    seen = set()

    def add(text: str, category: str):
        text = _clean_memory_value(text, 120)
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append({"text": text, "category": category})

    for msg in messages:
        if _message_role(msg) != "user":
            continue
        text = _message_text(msg)
        if not text:
            continue

        m = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z0-9 .'\-]{1,50})\b", text, re.I)
        if m:
            name = _clean_memory_value(m.group(1), 50)
            if name:
                add(f"User's name is {name}.", "identity")

        m = re.search(r"\bcall me\s+([A-Za-z][A-Za-z0-9 .'\-]{1,50})\b", text, re.I)
        if m:
            name = _clean_memory_value(m.group(1), 50)
            if name:
                add(f"User wants to be called {name}.", "identity")

        m = re.search(r"\bi (?:live in|am from|'m from)\s+([^.!?\n]{2,80})", text, re.I)
        if m:
            place = _clean_memory_value(m.group(1), 80)
            if place:
                add(f"User lives in {place}.", "identity")

        m = re.search(r"\bi (prefer|like|love|hate|do not like|don't like)\s+([^.!?\n]{4,100})", text, re.I)
        if m:
            preference = _clean_memory_value(m.group(2), 100)
            if preference:
                # The same pattern catches likes and dislikes; keep the stored
                # sentiment faithful instead of recording every match as a
                # preference ("I hate cilantro" must not become "User prefers
                # cilantro").
                verb = m.group(1).lower()
                if verb in ("hate", "do not like", "don't like"):
                    add(f"User dislikes {preference}.", "preference")
                else:
                    add(f"User prefers {preference}.", "preference")

        m = re.search(
            r"\bi (?:(?:want|would like|plan|hope) to|wanna) "
            r"(?:go|travel|move|visit) to\s+([^.!?\n]{2,80})",
            text,
            re.I,
        )
        if m:
            destination = _clean_memory_value(m.group(1), 80)
            if destination:
                add(f"User wants to visit {destination}.", "goal")

    return candidates[:2]


def _is_text_duplicate(new_text: str, existing: list, threshold: float = 0.6) -> bool:
    """Check if new_text is too similar to any existing memory (Jaccard similarity)."""
    new_tokens = set(new_text.lower().split())
    if not new_tokens:
        return False
    for entry in _memory_dicts(existing):
        old_tokens = set(entry.get("text", "").lower().split())
        if not old_tokens:
            continue
        intersection = new_tokens & old_tokens
        union = new_tokens | old_tokens
        if len(intersection) / len(union) >= threshold:
            return True
    return False


def _parse_extraction_json(raw: str) -> list:
    """Parse the extraction LLM's reply into a list of facts, tolerating
    reasoning-model noise.

    The model emits <think>…</think> (and sometimes a prose preamble or a
    ```json fence) AROUND the JSON array; without stripping it, json.loads
    bombs and the run silently yields "0 candidates". Pure str -> list (no
    LLM/network); returns [] on any parse failure instead of raising.
    """
    text = (raw or "").strip()
    try:
        from src.text_helpers import strip_think as _strip_think
        text = _strip_think(text, prose=True, prompt_echo=True).strip()
    except Exception:
        pass
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # JSON may still be embedded in surrounding commentary (leading prose or
    # trailing remarks like "[...] Done!") — slice from the first '[' to the
    # last ']' whenever both exist. Slice unconditionally: a reply that starts
    # with '[' can still carry trailing commentary that breaks json.loads.
    _start = text.find("[")
    _end = text.rfind("]")
    if 0 <= _start < _end:
        text = text[_start : _end + 1]

    try:
        facts = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Memory extraction returned non-JSON: %r", (raw or "")[:120])
        return []
    except Exception:
        logger.debug("Memory extraction returned non-JSON: %r", (raw or "")[:120])
        return []
    return facts if isinstance(facts, list) else []


async def extract_and_store(
    session,
    memory_manager,
    memory_vector,
    endpoint_url: str,
    model: str,
    headers: Optional[dict] = None,
):
    """Extract facts from recent conversation and store them.

    Designed to run as a background task (asyncio.create_task).
    Errors are logged, never raised.
    """
    if not endpoint_url or not model:
        logger.debug("[memory-extract] No model or URL provided, skipping")
        return

    try:
        from src.llm_core import llm_call_async
        from services.memory.memory_tagger import tag_memory, apply_tags

        # Get last N messages from session
        messages = session.get_context_messages()
        recent = messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages

        if len(recent) < 2:
            return  # Need at least a user message and assistant response

        # Strip media (images/audio) from messages — background memory extraction
        # only needs the text. The VL-generated descriptions are already in the
        # text content of the messages. This avoids sending image tokens to
        # non-vision models and prevents accidental "vision grounding" triggers.
        stripped_recent = []
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Filter out multimodal blocks that aren't text
                text_only = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if not text_only and content:
                    continue
                content = text_only
            stripped_recent.append({"role": role, "content": content})

        if not stripped_recent:
            return

        fallback_facts = _fallback_memory_candidates(stripped_recent)

        # Flatten the window into a SINGLE user message instead of appending the
        # raw alternating role messages. Passed as raw chat messages, the model
        # treats the window as a conversation to CONTINUE rather than a transcript
        # to ANALYZE, so it reliably extracts nothing — typically returning `[]`
        # (and, depending on the input, sometimes an empty or <think>-only
        # completion when the window ends on an assistant turn). This was the real
        # cause of auto-memory logging "0 candidates" on every run. Reframing it as
        # one "analyze this transcript, return the JSON array" user message makes
        # the model actually extract. Controlled repro on this model: 0/6 trials
        # with the old structure vs 6/6 with this one. The skill extractor flattens
        # for the same reason.
        def _flatten_msg(m):
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(
                    b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            return f"{m.get('role', '?')}: {c}"

        transcript = "\n\n".join(_flatten_msg(m) for m in stripped_recent)
        extraction_messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Conversation to analyze:\n\n" + transcript
                + "\n\nReturn the JSON array of durable facts now (or [] if none)."
            )},
        ]

        facts = []
        try:
            raw = await llm_call_async(
                endpoint_url,
                model,
                extraction_messages,
                temperature=0.1,
                # A reasoning model spends most of its budget on <think> tokens
                # BEFORE emitting the JSON, so the old 500 truncated the response
                # before any JSON appeared → every run logged "0 candidates". The
                # audit path hit the same wall and raised to 16384; extraction's
                # output (a short facts list) is small, so an ample ceiling is
                # enough once thinking has room.
                max_tokens=4096,
                headers=headers,
            )

            # Parse JSON, tolerating reasoning-model noise (<think> blocks, a
            # ```json fence, and leading/trailing commentary). See
            # _parse_extraction_json — returns [] rather than raising.
            facts = _parse_extraction_json(raw)
        except Exception as e:
            logger.warning(f"LLM memory extraction failed; using fallback candidates if available: {e}")

        if not isinstance(facts, list):
            facts = []

        if fallback_facts:
            facts = list(facts) + fallback_facts

        if not facts:
            logger.info("Auto memory extraction ran: 0 candidates")
            return

        # Get owner from session
        _owner = getattr(session, 'owner', None)

        existing = memory_manager.load_all()
        added = 0
        new_entries = []

        for fact in facts:
            # The extraction prompt still speaks "category" (rework lands with
            # the distiller upgrade); its value becomes the entry's first tag.
            if isinstance(fact, str):
                fact_text = fact
                category = "fact"
            elif isinstance(fact, dict):
                fact_text = fact.get("text", "").strip()
                category = fact.get("category", "fact")
            else:
                continue

            if not fact_text or len(fact_text) < 5:
                continue

            # Dedup: check vector similarity first (fast), then exact text match.
            # A runtime embedding/ChromaDB failure (backend OOM, model evicted,
            # remote endpoint down) must not abort the whole batch — fall through
            # to the text/fuzzy dedup below instead of losing every validated
            # fact extracted this session. (`.healthy` is only set at init, so
            # it does not catch failures that develop later.)
            if memory_vector and memory_vector.healthy:
                try:
                    existing_id = memory_vector.find_similar(fact_text, threshold=0.72)
                except Exception as e:
                    logger.warning(f"Memory dedup (vector) unavailable, using text fallback: {e}")
                    existing_id = None
                if existing_id:
                    # The vector store is a single shared collection with no
                    # owner metadata, so find_similar can return ANOTHER
                    # tenant's memory. Only treat it as a duplicate when the
                    # match is this user's own (or a legacy unowned) memory —
                    # otherwise the user's freshly-extracted fact would be
                    # silently dropped. Mirror the owner predicate used by the
                    # text dedup below; cross-tenant/stale matches fall through.
                    _match = next((e for e in existing if e.get("id") == existing_id), None)
                    if _match is not None and (_match.get("owner") == _owner or _match.get("owner") is None):
                        logger.debug(f"Memory dedup (vector): '{fact_text[:50]}' matches {existing_id}")
                        continue

            # Text dedup fallback: exact match + fuzzy similarity
            user_existing = [e for e in existing if e.get("owner") == _owner or e.get("owner") is None] if _owner else existing
            if memory_manager.find_duplicates(fact_text, user_existing):
                continue
            # Fuzzy text similarity check (catches rephrased duplicates when vector index is unavailable)
            if _is_text_duplicate(fact_text, user_existing):
                logger.debug(f"Memory dedup (fuzzy): '{fact_text[:50]}' too similar to existing")
                continue

            entry = memory_manager.add_entry(fact_text, source="auto", category=category, owner=_owner)
            tag_result = await tag_memory(fact_text, owner=_owner)
            apply_tags(entry, tag_result)
            # Auto-pin identity facts (name, job, location) — core context.
            # Goes away with the distiller upgrade: generality → tier 0 takes over.
            if "identity" in entry["tags"]:
                entry["pinned"] = True
            if hasattr(session, "session_id"):
                entry["session_id"] = session.session_id
            elif hasattr(session, "name"):
                entry["session_id"] = session.name

            # `existing` doubles as the intra-batch dedup corpus; `new_entries`
            # is what actually gets persisted (re-loaded under the lock below,
            # so entries added elsewhere during this batch aren't clobbered).
            existing.append(entry)
            new_entries.append(entry)

            # Add to vector index. The JSON store (saved below) is the source of
            # truth and the keyword path can still retrieve this entry, so a vector
            # write failure must not drop the fact or abort the remaining batch.
            if memory_vector and memory_vector.healthy:
                try:
                    memory_vector.add(entry["id"], fact_text)
                except Exception as e:
                    logger.warning(f"Memory vector add failed for {entry['id']}: {e}")

            added += 1

        if added > 0:
            with memory_manager.lock:
                fresh = memory_manager.load_all()
                fresh.extend(new_entries)
                memory_manager.save(fresh)
            try:
                from src.event_bus import fire_event
                for _ in range(added):
                    fire_event("memory_added", _owner)
            except Exception:
                logger.debug("memory_added event dispatch failed", exc_info=True)
            logger.info(f"Auto-extracted {added} memories from session")

            global _extractions_since_audit
            _extractions_since_audit += added
            if _extractions_since_audit >= AUDIT_INTERVAL:
                _extractions_since_audit = 0
                logger.info("Triage threshold reached, running cheap memory triage")
                from services.memory.memory_curator import triage_new_entries
                await triage_new_entries(memory_manager, owner=_owner)
        else:
            logger.info("Auto memory extraction ran: 0 added")

    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")

