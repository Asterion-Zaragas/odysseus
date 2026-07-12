"""
retrieval.py

Staged memory retrieval pipeline (memory upgrade Phase 6): tag-aware,
tier-weighted, effort-scoped. Shared by the chat preface
(``ChatProcessor.build_context_preface``) and the ``retrieve_memory_context``
agent tool, so both get identical ranking behavior.

Stages:
  A. facets (effort != low) — one *memory-fast* call turning the user
     message into ``{keywords, entities: [{type, name}], tag_guesses}``.
     Hard 2s timeout; a timeout or parse failure silently degrades the
     whole call to effort "low" (never blocks a chat turn on a slow model).
  B. candidates (pure code) — tag filter (facet tag_guesses/entities
     intersected with each entry's tags; an empty intersection means no
     entry matched, so the filter is skipped rather than emptying the
     pool) scoped to the tiers the effort level allows, then BM25 + vector
     hybrid scoring (ported from the pre-Phase-6
     ``ChatProcessor._hybrid_retrieve``) with a tier-weight multiplier and
     facet keywords folded into the query tokens.
  C. verify (effort high only) — one *memory-fast* call over the top ~20
     stage-B candidates that returns the ids of the truly relevant ones
     (<=5, further capped to the caller's ``k``). A parse/call failure
     falls back to the stage-B top-k; an explicit "none of these are
     relevant" answer (valid empty list) is honored as-is, not treated as
     a failure.
"""

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from services.memory.tier_scoring import TIER_ARCHIVE, TIER_CORE, TIER_DURABLE, TIER_SITUATIONAL

logger = logging.getLogger(__name__)

EFFORT_LEVELS = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"

# Max tier a given effort level's candidate search includes. Tier 0 (core)
# is handled outside this module (always-inject, like pinned) — callers that
# want it in the ranked pool too (e.g. the agent tool re-querying broadly)
# simply pass tier-0 entries in; this cap still applies to them.
_TIER_CAP = {"low": TIER_DURABLE, "medium": TIER_SITUATIONAL, "high": TIER_ARCHIVE}

# Ranking multiplier by tier — core/durable facts rank slightly above
# situational/archive ones of otherwise-equal keyword/vector relevance.
_TIER_WEIGHT = {TIER_CORE: 1.2, TIER_DURABLE: 1.0, TIER_SITUATIONAL: 0.85, TIER_ARCHIVE: 0.7}

_FACET_TIMEOUT_SECONDS = 2.0
_VERIFY_TIMEOUT_SECONDS = 8.0
_VERIFY_POOL_SIZE = 20

FACET_SYSTEM_PROMPT = (
    "Extract search facets from a user's chat message for a personal memory "
    "system. Return STRICT JSON only:\n"
    '{"keywords": ["..."], "entities": [{"type": "person|place|org|project", "name": "..."}], "tag_guesses": ["..."]}\n\n'
    "Rules:\n"
    "- \"keywords\": up to 6 salient content words/phrases from the message.\n"
    "- \"entities\": named people/places/organizations/projects mentioned or "
    "clearly implied by the message.\n"
    "- \"tag_guesses\": up to 6 tag-shaped guesses in kebab-case; structured "
    "tags use person:/place:/org:/project: prefixes, e.g. \"person:sven\".\n"
    "- Return ONLY the JSON object. No markdown fences, no commentary."
)

VERIFY_SYSTEM_PROMPT = (
    "You are given a user's chat message and a list of candidate memories "
    "(id: text). Return STRICT JSON only: {\"relevant_ids\": [\"...\"]} — at "
    "most 5 ids of the candidates that are ACTUALLY relevant to answering or "
    "informing a response to the message. Prefer fewer, precise picks over "
    "padding to 5; an empty list is a valid answer when nothing is relevant. "
    "Return ONLY the JSON object. No markdown fences, no commentary."
)

# ── Stopwords & tokenizer (ported from chat_processor._hybrid_retrieve) ──

_STOPWORDS = frozenset(
    "a an the is am are was were be been being have has had do does did "
    "will would shall should can could may might must need ought dare "
    "i me my mine we us our ours you your yours he him his she her hers "
    "it its they them their theirs this that these those "
    "and but or nor not no so if then else than too also very "
    "in on at to for of by with from up out about into over after "
    "what when where which who whom how why all each every some any "
    "just very really actually like well also still already even "
    "oh ok okay yes yeah hey hi hello thanks thank please sorry "
    "much more most own other another such only same here there "
    "because while during before until since through between both "
    "few many several some none nothing something anything everything "
    "get got make made go going went been come came take took "
    "know think want let say tell give see look find way thing "
    "don doesn didn won wouldn couldn shouldn wasn weren isn aren haven hasn "
    "don't doesn't didn't won't wouldn't couldn't shouldn't "
    "it's i'm i've i'll i'd you're you've you'll he's she's we're we've they're they've "
    "that's there's here's what's who's how's let's can't".split()
)


def _content_tokens(text: str) -> list:
    """Extract meaningful content words: no stopwords, min 3 chars, lowercase."""
    words = re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _normalize_effort(effort: Optional[str]) -> str:
    e = (effort or "").strip().lower()
    return e if e in EFFORT_LEVELS else DEFAULT_EFFORT


def _parse_json_object(raw: str) -> Optional[Dict]:
    """Tolerant JSON-object parse shared shape with memory_tagger's parser:
    strips <think> blocks / code fences / surrounding prose. Never raises."""
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
        logger.debug("Memory retrieval stage returned non-JSON: %r", (raw or "")[:120])
        return None
    return obj if isinstance(obj, dict) else None


async def _extract_facets(message: str, owner: Optional[str], interactive: bool) -> Optional[Dict]:
    """Stage A. Returns None on timeout/parse failure (caller degrades to low)."""
    try:
        from src.task_endpoint import memory_llm_call_async

        messages = [
            {"role": "system", "content": FACET_SYSTEM_PROMPT},
            {"role": "user", "content": message[:2000]},
        ]
        raw = await asyncio.wait_for(
            memory_llm_call_async(
                "fast", messages, owner=owner, interactive=interactive,
                temperature=0.1, max_tokens=300,
            ),
            timeout=_FACET_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.debug("Memory facet extraction failed/timed out for owner=%r: %s", owner, e)
        return None

    parsed = _parse_json_object(raw)
    if parsed is None:
        return None
    return {
        "keywords": [str(k) for k in (parsed.get("keywords") or []) if str(k).strip()][:10],
        "entities": [e for e in (parsed.get("entities") or []) if isinstance(e, dict)][:10],
        "tag_guesses": [str(t) for t in (parsed.get("tag_guesses") or []) if str(t).strip()][:10],
    }


def _facet_tags(facets: Optional[Dict]) -> set:
    """Normalize a facet result's tag_guesses + entities into a tag set."""
    if not facets:
        return set()
    from src.memory import normalize_tag

    tags = set()
    for t in (facets.get("tag_guesses") or []):
        nt = normalize_tag(t)
        if nt:
            tags.add(nt)
    for ent in (facets.get("entities") or []):
        etype = str(ent.get("type") or "").strip().lower()
        ename = str(ent.get("name") or "").strip()
        if etype in ("person", "place", "org", "project") and ename:
            nt = normalize_tag(f"{etype}:{ename}")
            if nt:
                tags.add(nt)
    return tags


def _stage_b_rank(
    message: str,
    entries: List[Dict],
    effort: str,
    facets: Optional[Dict],
    memory_vector=None,
    pool_k: int = 20,
) -> List[Dict]:
    """Pure-code candidate scoring. Returns entries sorted best-first
    (already gated by minimum relevance), no cap applied — callers slice."""
    if not entries or not message.strip():
        return []

    tier_cap = _TIER_CAP[effort]
    pool = [e for e in entries if int(e.get("tier") if e.get("tier") is not None else 2) <= tier_cap]
    if not pool:
        return []

    facet_tags = _facet_tags(facets)
    if facet_tags:
        filtered = [e for e in pool if facet_tags & set(e.get("tags") or [])]
        if filtered:
            pool = filtered
        # else: empty intersection -> no filter, keep the full tier-scoped pool.

    facet_keywords = [str(k) for k in ((facets or {}).get("keywords") or [])]
    query_tokens = _content_tokens(message)
    for kw in facet_keywords:
        query_tokens.extend(_content_tokens(kw))

    now = time.time()
    N = len(pool)
    doc_freq = Counter()
    mem_token_cache: Dict[str, set] = {}
    for mem in pool:
        toks = set(_content_tokens(mem.get("text", "")))
        mem_token_cache[mem["id"]] = toks
        for t in toks:
            doc_freq[t] += 1

    def _bm25_score(query_toks, mem_id):
        mem_toks = mem_token_cache.get(mem_id, set())
        if not mem_toks or not query_toks:
            return 0.0
        score = 0.0
        mem_len = len(mem_toks)
        avg_len = max(sum(len(v) for v in mem_token_cache.values()) / N, 1)
        k1, b = 1.5, 0.75
        for qt in query_toks:
            if qt not in mem_toks:
                continue
            df = doc_freq.get(qt, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (1 * (k1 + 1)) / (1 + k1 * (1 - b + b * mem_len / avg_len))
            score += idf * tf_norm
        return score

    has_vector = memory_vector is not None and getattr(memory_vector, "healthy", False)
    vector_scores: Dict[str, float] = {}
    if has_vector:
        try:
            results = memory_vector.search(message, k=min(max(pool_k, 5) * 2, 30))
            pool_by_id = {m["id"]: m for m in pool}
            for r in results:
                if r["memory_id"] in pool_by_id:
                    vector_scores[r["memory_id"]] = max(r["score"], 0.0)
        except Exception as e:
            logger.debug("Memory vector search failed: %s", e)

    scored = []
    msg_lower = message.lower()
    for mem in pool:
        mid = mem["id"]
        vs = vector_scores.get(mid, 0.0)
        kw = _bm25_score(query_tokens, mid)
        kw_norm = min(kw / 6.0, 1.0) if kw > 0 else 0.0

        # Tag-aware boost for identity/contact/preference queries (ported
        # from the pre-Phase-6 _hybrid_retrieve's tag_boost).
        tags = mem.get("tags") or []
        mem_lower = mem.get("text", "").lower()
        tag_boost = 1.0
        if any(w in msg_lower for w in ["name", "who am i", "my name"]):
            if "identity" in tags or any(w in mem_lower for w in ["name is", "i am", "called"]):
                tag_boost = 1.4
        elif any(w in msg_lower for w in ["phone", "email", "address", "contact"]):
            if "contact" in tags or "@" in mem_lower:
                tag_boost = 1.3
        elif any(w in msg_lower for w in ["like", "prefer", "favorite"]):
            if "preference" in tags:
                tag_boost = 1.2
        kw_norm = min(kw_norm * tag_boost, 1.0)

        ts = mem.get("timestamp", 0)
        days_old = max((now - ts) / 86400, 0)
        recency = 1.0 / (1.0 + days_old * 0.05)

        if has_vector:
            if vs < 0.20 and kw_norm < 0.08:
                continue
            final = (0.55 * vs) + (0.40 * kw_norm) + (0.05 * recency)
        else:
            if kw_norm < 0.08:
                continue
            final = (0.95 * kw_norm) + (0.05 * recency)

        if final <= 0.12:
            continue

        tier = int(mem.get("tier") if mem.get("tier") is not None else 2)
        final *= _TIER_WEIGHT.get(tier, 1.0)
        scored.append((final, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored]


async def _verify(
    message: str, candidates: List[Dict], owner: Optional[str], interactive: bool
) -> Optional[List[Dict]]:
    """Stage C. Returns None on parse/call failure (caller falls back to
    the stage-B top-k); an explicit empty `relevant_ids` is a legitimate
    answer and is returned as `[]`, not treated as failure."""
    if not candidates:
        return []

    lines = [f'{c["id"]}: {c.get("text", "")[:300]}' for c in candidates if c.get("id")]
    user_content = f"User message: {message[:1000]}\n\nCandidates:\n" + "\n".join(lines)

    try:
        from src.task_endpoint import memory_llm_call_async

        messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content[:6000]},
        ]
        raw = await asyncio.wait_for(
            memory_llm_call_async(
                "fast", messages, owner=owner, interactive=interactive,
                temperature=0.1, max_tokens=200,
            ),
            timeout=_VERIFY_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.debug("Memory verify stage failed for owner=%r: %s", owner, e)
        return None

    parsed = _parse_json_object(raw)
    if parsed is None:
        return None

    ids = [str(i) for i in (parsed.get("relevant_ids") or []) if str(i).strip()]
    by_id = {c["id"]: c for c in candidates if c.get("id")}
    return [by_id[i] for i in ids if i in by_id][:5]


async def retrieve(
    message: str,
    entries: List[Dict],
    *,
    effort: str = DEFAULT_EFFORT,
    memory_vector=None,
    owner: Optional[str] = None,
    k: int = 5,
    interactive: bool = False,
) -> Dict[str, Any]:
    """Staged retrieval over a candidate pool of memory entries.

    `entries` is the pool to search — callers decide what's in it (e.g. the
    chat preface excludes pinned/tier-0 entries it already injects
    unconditionally; the ``retrieve_memory_context`` tool passes everything).

    Returns ``{"memories": [...], "facets": dict|None, "effort_used": str}``.
    `effort_used` differs from the requested `effort` only when stage A
    degraded low from a facet timeout/parse failure.
    """
    effort = _normalize_effort(effort)
    message = (message or "").strip()
    if not message or not entries:
        return {"memories": [], "facets": None, "effort_used": effort}

    facets = None
    if effort != "low":
        facets = await _extract_facets(message, owner=owner, interactive=interactive)
        if facets is None:
            effort = "low"

    if effort == "high":
        ranked = _stage_b_rank(message, entries, effort, facets, memory_vector, pool_k=_VERIFY_POOL_SIZE)
        top_pool = ranked[:_VERIFY_POOL_SIZE]
        verified = await _verify(message, top_pool, owner=owner, interactive=interactive)
        # Verify itself returns at most 5 (its own prompt-level cap on "how
        # many are truly relevant"); this further caps to the caller's k so
        # a k=3 chat-preface call doesn't inject 5 memories just because
        # effort=high, same as the low/medium branch below already does.
        memories = (verified if verified is not None else ranked)[:k]
    else:
        ranked = _stage_b_rank(message, entries, effort, facets, memory_vector, pool_k=k)
        memories = ranked[:k]

    return {"memories": memories, "facets": facets, "effort_used": effort}
