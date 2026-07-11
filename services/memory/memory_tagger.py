"""
memory_tagger.py

On-demand small-model ("memory fast" role) call that assigns tags and a
generality score to a single new memory at write time, plus the pure helper
that turns that result into the entry's initial tier.

Every memory write path calls `tag_memory()` then `apply_tags()`: the
extractor's store loop, the inline "remember:" chat command, the memory
page's add/update routes, and the `manage_memory` builtin + MCP tools. Kept
as a standalone module rather than folded into the extractor (concept doc
resolved decision #1) because the latter three paths don't go through
extraction at all.

Registry-constrained: the tagger is asked to prefer tags already in the
owner's tag registry (services/memory/memory_context.py); anything it
proposes outside the registry — including everything during cold start,
when the registry is still empty — lands in `provisional_tags` until a
curator run (Phase 4) promotes or merges it.

Never blocks or fails a write: any LLM/parse failure returns
`FALLBACK_RESULT` and the curator fixes untagged entries up overnight.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FALLBACK_RESULT: Dict = {"tags": [], "provisional_tags": [], "generality": None}

TAGGER_SYSTEM_PROMPT = (
    "You tag a single memory entry for a personal memory system. Given the "
    "memory text, an optional context hint, and the current tag registry, "
    "return STRICT JSON only:\n"
    '{"tags": ["..."], "new_tags": ["..."], "generality": 0-3}\n\n'
    "Rules:\n"
    "- \"tags\": 0-10 tags chosen from the REGISTRY below that apply. Prefer "
    "existing registry tags over inventing new ones.\n"
    "- \"new_tags\": tags you believe are needed but are NOT in the registry "
    "(e.g. a new person/place/project). Use the same kebab-case style; "
    "structured tags use person:/place:/org:/project: prefixes, e.g. "
    "\"person:sven\".\n"
    "- \"generality\": how broadly useful this fact is across future "
    "conversations. 3 = identity-level (name, job, home city, close "
    "relations). 2 = stable preference/relationship. 1 = project- or "
    "time-bound context. 0 = conversation-specific trivia. HIGHER means "
    "MORE durable/general.\n"
    "- tags + new_tags combined must not exceed 10.\n"
    "- Return ONLY the JSON object. No markdown fences, no commentary."
)

TAGGER_MAX_TOKENS = 512


def _parse_tagger_json(raw: str) -> Optional[Dict]:
    """Tolerant JSON-OBJECT parse for the tagger's reply.

    Mirrors memory_extractor._parse_extraction_json's tolerance for
    reasoning-model noise (<think> blocks, code fences, surrounding prose)
    but for a single `{...}` object instead of a `[...]` array. Pure
    str -> dict|None, never raises.
    """
    text = (raw or "").strip()
    try:
        from src.text_helpers import strip_think
        text = strip_think(text, prose=True, prompt_echo=True).strip()
    except Exception:
        pass
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        logger.debug("Memory tagger returned non-JSON: %r", (raw or "")[:120])
        return None
    return obj if isinstance(obj, dict) else None


async def tag_memory(
    text: str,
    context_hint: Optional[str] = None,
    owner: Optional[str] = None,
    *,
    interactive: bool = False,
) -> Dict:
    """Tag one memory. Never raises — worst case returns `FALLBACK_RESULT`.

    `interactive=True` must be passed by every caller running inline inside
    a tracked HTTP/chat request (the memory page's add/update routes, the
    inline "remember:" command, the manage_memory tool, the MCP server) —
    otherwise the call deadlocks itself against interactive_gate's
    foreground-quiet wait (src/task_endpoint.py:memory_llm_call_async), the
    same reasoning as the retrieval hot path in the concept doc. Background
    callers (the extraction store loop) keep the default `False` so tagging
    still respects "don't compete with the user's own local-model turn".

    Returns `{"tags": [...], "provisional_tags": [...], "generality": int|None}`.
    """
    text = (text or "").strip()
    if not text:
        return dict(FALLBACK_RESULT)

    try:
        from src.memory import normalize_tags
        from src.settings import get_setting
        from src.task_endpoint import memory_llm_call_async
        from services.memory.memory_context import MemoryContext

        ctx = MemoryContext(owner)
        registry_names = ctx.registry_names()
        cap = get_setting("memory_tag_registry_cap", 50)
        registry_excerpt = ctx.registry_excerpt(max_tags=cap)

        user_content = f"Memory: {text}\n"
        if context_hint:
            user_content += f"Context: {context_hint}\n"
        user_content += f"\nRegistry:\n{registry_excerpt}"

        messages = [
            {"role": "system", "content": TAGGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = await memory_llm_call_async(
            "fast",
            messages,
            owner=owner,
            interactive=interactive,
            temperature=0.1,
            max_tokens=TAGGER_MAX_TOKENS,
        )
        parsed = _parse_tagger_json(raw)
        if parsed is None:
            return dict(FALLBACK_RESULT)

        proposed = normalize_tags(list(parsed.get("tags") or []) + list(parsed.get("new_tags") or []))

        # Cold start (empty registry): nothing to validate against, so every
        # proposal is provisional until the first curator run bootstraps the
        # registry — matches the concept doc's cold-start contract exactly.
        if registry_names:
            tags = [t for t in proposed if t in registry_names]
            provisional = [t for t in proposed if t not in registry_names]
        else:
            tags = []
            provisional = proposed

        generality = parsed.get("generality")
        try:
            generality = max(0, min(3, int(generality)))
        except (TypeError, ValueError):
            generality = None

        return {"tags": tags, "provisional_tags": provisional, "generality": generality}
    except Exception as e:
        logger.warning("Memory tagger failed for owner=%r: %s", owner, e)
        return dict(FALLBACK_RESULT)


def apply_tags(entry: Dict, result: Dict, user_tags: Optional[List[str]] = None) -> Dict:
    """Apply a tag_memory() result to a freshly-built (or edited) entry and
    compute its tier. Mutates and returns `entry`.

    `user_tags`, when non-empty, wins outright — the memory page's "tags are
    directly user-editable" contract — and the tagger's tag proposals are
    dropped (its generality is still used). Otherwise the tagger's tags are
    MERGED with whatever tags were already seeded on the entry (e.g. the
    deprecated `category` param folded in by `add_entry`), so a tagger
    failure or cold start never leaves an entry with fewer tags than before
    this call — `FALLBACK_RESULT`'s empty tag lists are a safe no-op here.

    `generality` only overwrites the entry's existing value when the result
    actually has one — a transient tagger failure on an edit must not erase
    a generality a previous successful tag/curator run already set.
    """
    from src.memory import normalize_tags
    from services.memory.tier_scoring import initial_tier, tier_score

    if user_tags:
        entry["tags"] = normalize_tags(user_tags)
        entry["provisional_tags"] = []
    else:
        seeded = entry.get("tags") or []
        entry["tags"] = normalize_tags(list(seeded) + list(result.get("tags") or []))
        entry["provisional_tags"] = normalize_tags(result.get("provisional_tags") or [])

    if result.get("generality") is not None:
        entry["generality"] = result.get("generality")

    entry["tier"] = initial_tier(tier_score(entry))
    return entry
