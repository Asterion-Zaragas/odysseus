"""
memory_curator.py

Nightly, per-owner curator agent ("memory smart" role): the second half of
the memory upgrade's agentic lifecycle (the tagger, in memory_tagger.py,
handles on-demand tagging at write time). The curator never touches the
hot path — it dedupes/merges, normalizes tags, rescores tiers, expires
stale archive entries, and rebuilds the per-owner context document
(services/memory/memory_context.py).

Batching discipline: the curator NEVER puts a whole store in one prompt —
every LLM sub-pass runs many small sequential calls of at most
`memory_curator_batch` entries (see `cluster_batches`), clustered by
dominant tag so duplicates/synonyms land in the same batch. A failing batch
is skipped, never aborts the run (`_triage_batch` / `_dedupe_batch` /
`_propose_tag_merges` all degrade to "no-op for this batch" on any
LLM/parse failure).

Two-phase, auditable destructiveness: the curator can only *demote* a
memory to the archive tier (pure code, `_run_rescore_pass`); actual
deletion only happens in `_run_expire_pass` for archive-tier entries that
have been idle past `memory_archive_expiry_days`, are not pinned, and
carry none of the `memory_protected_tags`. Every destructive or
tier-changing action is appended to a per-owner JSONL changelog
(`data/memory_curation_log/<owner-slug>.jsonl`); `undo_expire` reverses an
expiry by re-inserting its full snapshot.

Owner convention (curator-local, deliberately different from
`MemoryManager.load(owner=...)`): `owner=None` here always means "the
ownerless/legacy bucket" (`_entries_for_owner`), never "all entries" —
`MemoryManager.load(None)` returns everything unfiltered, which would make
a nightly per-owner loop double-curate every real owner's data under the
legacy pass too.
"""

import copy
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from src.constants import DATA_DIR
from services.memory.memory_context import _owner_slug
from services.memory.memory_extractor import _fingerprint_entries
from services.memory.tier_scoring import (
    TIER_ARCHIVE,
    TIER_CORE,
    next_tier,
    tier_score,
)

logger = logging.getLogger(__name__)

CURATION_STATE_DIR = os.path.join(DATA_DIR, "memory_curation_state")
CURATION_LOG_DIR = os.path.join(DATA_DIR, "memory_curation_log")

# Pass order for a full curation run. Checkpointing is at pass granularity:
# a crash/shutdown mid-run resumes at the next NOT-yet-completed pass rather
# than mid-batch — batches are already small (memory_curator_batch entries),
# so redoing one pass's batches on resume is cheap, and cross-batch effects
# (e.g. a duplicate pair split across batches) are eventually caught once
# tag normalization pulls them into the same cluster on a later run.
PASS_ORDER = ("triage", "dedupe", "tag_normalize", "rescore", "expire", "context_doc")

DEFAULT_PROTECTED_TAGS = ("contact", "identity")


# ---- owner bucketing ----

def _entries_for_owner(all_entries: List[Dict], owner: Optional[str]) -> List[Dict]:
    """Curator-local owner filter: `owner=None`/`""` means strictly the
    ownerless bucket, never "everything" (unlike MemoryManager.load)."""
    if owner:
        return [e for e in all_entries if e.get("owner") == owner]
    return [e for e in all_entries if not e.get("owner")]


def list_owners(memory_manager) -> List[Optional[str]]:
    """Distinct curation targets: every real owner value present in the
    store, plus `None` once if any ownerless/legacy entries exist."""
    all_entries = memory_manager.load_all()
    owners = sorted({e.get("owner") for e in all_entries if e.get("owner")})
    if any(not e.get("owner") for e in all_entries):
        owners.append(None)
    return owners


# ---- checkpoint (resumability) ----

def _state_path(owner: Optional[str]) -> str:
    return os.path.join(CURATION_STATE_DIR, f"{_owner_slug(owner)}.json")


def _load_checkpoint(owner: Optional[str]) -> Dict:
    try:
        with open(_state_path(owner), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_checkpoint(owner: Optional[str], last_completed_pass: str) -> None:
    os.makedirs(CURATION_STATE_DIR, exist_ok=True)
    path = _state_path(owner)
    tmp = path + ".tmp"
    payload = {"last_completed_pass": last_completed_pass, "updated_at": time.time()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _clear_checkpoint(owner: Optional[str]) -> None:
    try:
        os.remove(_state_path(owner))
    except FileNotFoundError:
        pass


# ---- changelog ----

def _log_path(owner: Optional[str]) -> str:
    return os.path.join(CURATION_LOG_DIR, f"{_owner_slug(owner)}.jsonl")


def _log_action(owner: Optional[str], action: str, before=None, after=None, dry_run: bool = False) -> None:
    os.makedirs(CURATION_LOG_DIR, exist_ok=True)
    line = {"ts": time.time(), "action": action, "before": before, "after": after, "dry_run": dry_run}
    try:
        with open(_log_path(owner), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        logger.warning("Could not write curation log for owner=%r: %s", owner, e)


def read_curation_log(owner: Optional[str], limit: int = 100, include_dry_run: bool = True) -> List[Dict]:
    """`include_dry_run=False` hides `dry_run:true` lines — a dry run logs
    every proposed action for debugging, but those actions were never
    actually applied, so a caller presenting this as "what the curator did"
    (the Phase 7 curator panel) must exclude them or it shows phantom
    merges/expiries and could offer to "undo" one. Defaults to True to keep
    existing callers/tests (which want the full raw log) unchanged."""
    path = _log_path(owner)
    if not os.path.exists(path):
        return []
    lines: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.warning("Could not read curation log for owner=%r: %s", owner, e)
        return []
    if not include_dry_run:
        lines = [l for l in lines if not l.get("dry_run")]
    return lines[-limit:] if limit else lines


def undo_expire(memory_manager, memory_vector, owner: Optional[str], memory_id: str) -> bool:
    """Reverse an `expire` action by re-inserting its full-entry snapshot.

    Returns True if a snapshot was found and the entry is now present
    (either just restored, or it was already there — a safe no-op).
    """
    snapshot = None
    for rec in reversed(read_curation_log(owner, limit=0, include_dry_run=False)):
        before = rec.get("before")
        if rec.get("action") == "expire" and isinstance(before, dict) and before.get("id") == memory_id:
            snapshot = before
            break
    if snapshot is None:
        return False

    with memory_manager.lock:
        entries = memory_manager.load_all()
        if any(e.get("id") == memory_id for e in entries):
            return True
        entries.append(copy.deepcopy(snapshot))
        memory_manager.save(entries)

    if memory_vector is not None and getattr(memory_vector, "healthy", False):
        try:
            memory_vector.add(snapshot["id"], snapshot.get("text", ""))
        except Exception as e:
            logger.warning("undo_expire: vector re-add failed for %s: %s", memory_id, e)

    _log_action(owner, "undo_expire", before=None, after=snapshot)
    return True


# ---- batching ----

def cluster_batches(entries: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split entries into batches of at most `batch_size`, clustered by
    dominant (most-frequent-in-this-set) tag first so duplicates/synonyms
    land together; leftovers with no cluster batch by recency. Every entry
    appears in exactly one batch."""
    batch_size = max(1, int(batch_size or 1))
    assigned: Set[str] = set()
    batches: List[List[Dict]] = []

    tag_counts: Dict[str, int] = {}
    for e in entries:
        for t in (e.get("tags") or []):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    for tag in sorted(tag_counts, key=lambda t: (-tag_counts[t], t)):
        cluster = [e for e in entries if e.get("id") not in assigned and tag in (e.get("tags") or [])]
        for i in range(0, len(cluster), batch_size):
            chunk = cluster[i:i + batch_size]
            if not chunk:
                continue
            batches.append(chunk)
            assigned.update(e["id"] for e in chunk)

    leftovers = [e for e in entries if e.get("id") not in assigned]
    leftovers.sort(key=lambda e: e.get("timestamp", 0))
    for i in range(0, len(leftovers), batch_size):
        chunk = leftovers[i:i + batch_size]
        if chunk:
            batches.append(chunk)

    return batches


# ---- tolerant JSON array parsing (mirrors memory_extractor's tolerance for
# reasoning-model noise; kept local rather than imported — see
# memory_tagger's identical precedent for its object-shaped counterpart) ----

def _parse_json_array(raw: str) -> Optional[List]:
    text = (raw or "").strip()
    try:
        from src.text_helpers import strip_think
        text = strip_think(text, prose=True, prompt_echo=True).strip()
    except Exception:
        pass
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        logger.debug("Curator got non-JSON reply: %r", (raw or "")[:160])
        return None
    return parsed if isinstance(parsed, list) else None


# ---- Pass 1: triage ----

TRIAGE_SYSTEM_PROMPT = (
    "You triage a batch of personal memory entries for a memory system. For "
    "each entry decide:\n"
    "1. \"generality\": 0-3 — how broadly useful the fact is across future "
    "conversations. 3 = identity-level (name, job, home city, close "
    "relations). 2 = stable preference/relationship. 1 = project- or "
    "time-bound context. 0 = conversation-specific trivia. HIGHER means MORE "
    "durable/general. Only asked when the entry's generality is not already set.\n"
    "2. \"promote_tags\": which of the entry's provisional_tags are genuinely "
    "useful going forward and should become real tags. Provisional tags you "
    "do NOT list are dropped as noise.\n\n"
    "Input is a JSON array of {id, text, tags, provisional_tags, generality}. "
    "Return a JSON array of {id, generality, promote_tags} — exactly one "
    "object per input entry, using the SAME id. Return ONLY valid JSON, no "
    "markdown fences, no commentary."
)

TRIAGE_MAX_TOKENS = 2048


def _needs_triage(entry: Dict) -> bool:
    return entry.get("generality") is None or bool(entry.get("provisional_tags"))


async def _triage_batch(batch: List[Dict], owner: Optional[str], dry_run: bool) -> List[Tuple[str, Dict, Dict]]:
    """Mutates entries in `batch` in place; returns a list of
    (action, before_snapshot, after_snapshot) tuples for the changelog."""
    from src.memory import normalize_tags
    from src.task_endpoint import memory_llm_call_async

    payload = [
        {
            "id": e["id"],
            "text": e.get("text", ""),
            "tags": e.get("tags") or [],
            "provisional_tags": e.get("provisional_tags") or [],
            "generality": e.get("generality"),
        }
        for e in batch
    ]
    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, temperature=0.1, max_tokens=TRIAGE_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning("Curator triage batch failed for owner=%r: %s", owner, e)
        return []

    results = _parse_json_array(raw)
    if not results:
        return []

    by_id = {e["id"]: e for e in batch}
    actions: List[Tuple[str, Dict, Dict]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        entry = by_id.get(item.get("id"))
        if entry is None:
            continue
        before = copy.deepcopy(entry)

        if entry.get("generality") is None:
            generality = item.get("generality")
            try:
                entry["generality"] = max(0, min(3, int(generality)))
            except (TypeError, ValueError):
                pass

        provisional = set(entry.get("provisional_tags") or [])
        promote = [t for t in normalize_tags(item.get("promote_tags") or []) if t in provisional]
        if promote:
            entry["tags"] = normalize_tags(list(entry.get("tags") or []) + promote)
        entry["provisional_tags"] = []

        if entry != before:
            actions.append(("retag", before, copy.deepcopy(entry)))
    return actions


async def _run_triage_pass(entries: List[Dict], owner: Optional[str], batch_size: int, dry_run: bool) -> List[Dict]:
    pending = [e for e in entries if _needs_triage(e)]
    for batch in cluster_batches(pending, batch_size):
        actions = await _triage_batch(batch, owner, dry_run)
        for action, before, after in actions:
            _log_action(owner, action, before=before, after=after, dry_run=dry_run)
    return entries


# ---- Pass 2: dedupe / merge ----

DEDUPE_SYSTEM_PROMPT = (
    "You are a memory database curator. Be CONSERVATIVE: remove only TRUE "
    "duplicates and clearly useless entries. Every distinct fact must survive. "
    "When in doubt, KEEP the entry. Return the cleaned list.\n\n"
    "Rules:\n"
    "1. MERGE only entries that state the SAME fact in different words. If you "
    "are not sure two entries are the same fact, KEEP BOTH.\n"
    "   Merge: 'User's name is Sam' + 'The user is called Sam' -> one.\n"
    "   Do NOT merge related-but-distinct facts: 'Likes Python' and 'Uses "
    "Python at work' are DIFFERENT — keep both.\n"
    "2. REMOVE only entries that are genuinely worthless: about what the AI did "
    "(not the user), empty, or meaningless. Do NOT drop a real fact just "
    "because it seems minor or niche.\n"
    "3. Keep the original wording. Only lightly trim obvious redundancy — do "
    "NOT aggressively rewrite or shorten.\n"
    "4. Preserve the 'id' of the entry you keep when merging.\n"
    "5. Never invent facts. When unsure, KEEP.\n"
    "6. Keep each entry's 'tags' list unless merging (then union the tags).\n\n"
    "Return a JSON array of objects with fields: id, text, tags.\n"
    "Return ONLY valid JSON, no markdown fences."
)

DEDUPE_MAX_TOKENS = 4096

# A conservative dedupe pass should never cut a batch by more than half —
# below this batch size the check is skipped (too small a sample to judge).
_UNSAFE_REMOVAL_MIN_BATCH = 8
_UNSAFE_REMOVAL_RATIO = 0.5


async def _dedupe_batch(batch: List[Dict], owner: Optional[str]) -> Tuple[List[Dict], List[Tuple[str, list, list]]]:
    """Returns (surviving_entries, changelog_actions). Never raises; any
    failure or unsafe-looking result returns the batch unchanged."""
    if len(batch) < 2:
        return list(batch), []

    from src.memory import normalize_tags
    from src.task_endpoint import memory_llm_call_async

    payload = [{"id": e["id"], "text": e.get("text", ""), "tags": e.get("tags") or []} for e in batch]
    messages = [
        {"role": "system", "content": DEDUPE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, temperature=0.1, max_tokens=DEDUPE_MAX_TOKENS, timeout=120,
        )
    except Exception as e:
        logger.warning("Curator dedupe batch failed for owner=%r: %s", owner, e)
        return list(batch), []

    cleaned = _parse_json_array(raw)
    if cleaned is None:
        return list(batch), []

    originals = {e["id"]: e for e in batch}
    final: List[Dict] = []
    for item in cleaned:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        original = originals.get(mid)
        if original is None:
            continue
        new_text = str(item.get("text") or "").strip()
        if not new_text:
            continue
        entry = copy.deepcopy(original)
        entry["text"] = new_text
        if item.get("tags"):
            entry["tags"] = normalize_tags(item["tags"])
        final.append(entry)

    before_count, after_count = len(batch), len(final)
    if before_count >= _UNSAFE_REMOVAL_MIN_BATCH and after_count < before_count * _UNSAFE_REMOVAL_RATIO:
        logger.warning(
            "Curator dedupe batch would cut %s -> %s (>%.0f%% removed) — refusing, owner=%r",
            before_count, after_count, _UNSAFE_REMOVAL_RATIO * 100, owner,
        )
        return list(batch), []

    actions: List[Tuple[str, list, list]] = []
    if after_count < before_count:
        kept_ids = {e["id"] for e in final}
        removed = [e for e in batch if e["id"] not in kept_ids]
        actions.append(("merge", removed, final))
    return final, actions


async def _run_dedupe_pass(entries: List[Dict], owner: Optional[str], batch_size: int, dry_run: bool) -> List[Dict]:
    survivors: List[Dict] = []
    for batch in cluster_batches(entries, batch_size):
        final, actions = await _dedupe_batch(batch, owner)
        survivors.extend(final)
        for action, before, after in actions:
            _log_action(owner, action, before=before, after=after, dry_run=dry_run)
    return survivors


# ---- Pass 3: tag normalization ----

TAG_MERGE_SYSTEM_PROMPT = (
    "You maintain a tag registry for a personal memory system. Given a list "
    "of tags with usage counts, identify tags that are SYNONYMS or near-"
    "duplicates of a more common tag (e.g. \"job\" and \"work\", \"office\" "
    "and \"workplace\"). Be conservative: only merge when you are confident "
    "they mean the same thing.\n\n"
    "Return a JSON array of {\"from\": \"...\", \"into\": \"...\"} — 'from' "
    "merges INTO 'into' (prefer the more common/canonical spelling as "
    "'into'). Return [] if no merges are warranted. Return ONLY valid JSON, "
    "no markdown fences."
)

TAG_MERGE_MAX_TOKENS = 1024


def _tag_counts(entries: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in entries:
        for t in (e.get("tags") or []):
            counts[t] = counts.get(t, 0) + 1
    return counts


async def _propose_tag_merges(tag_counts: Dict[str, int], owner: Optional[str]) -> List[Tuple[str, str]]:
    if len(tag_counts) < 2:
        return []

    from src.memory import normalize_tag
    from src.task_endpoint import memory_llm_call_async

    payload = [{"tag": t, "count": c} for t, c in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    messages = [
        {"role": "system", "content": TAG_MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, temperature=0.1, max_tokens=TAG_MERGE_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning("Curator tag-merge proposal failed for owner=%r: %s", owner, e)
        return []

    parsed = _parse_json_array(raw)
    if not parsed:
        return []

    merges = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        frm = normalize_tag(item.get("from") or "")
        into = normalize_tag(item.get("into") or "")
        if frm and into and frm != into and frm in tag_counts:
            merges.append((frm, into))
    return merges


def _apply_tag_merges(entries: List[Dict], merges: List[Tuple[str, str]], protected: Set[str]) -> List[Tuple[str, str]]:
    """Rewrites tags in place; returns the merges actually applied (skips
    any that would merge away a protected tag)."""
    from src.memory import normalize_tags

    applied = []
    for frm, into in merges:
        if frm in protected:
            continue
        changed = False
        for e in entries:
            tags = e.get("tags") or []
            if frm in tags:
                e["tags"] = normalize_tags([into if t == frm else t for t in tags])
                changed = True
        if changed:
            applied.append((frm, into))
    return applied


def build_registry(entries: List[Dict], cap: int, protected: Set[str], existing_registry: Optional[List[Dict]] = None) -> List[Dict]:
    """Rebuild the tag registry from live tag counts, capped at `cap` —
    protected tags are always kept (never counted against the cap); the
    remaining budget goes to the highest-count tags. Lower-count tags simply
    fall out of the registry (they remain valid on entries, just no longer
    "featured" in tagger/facet prompts) rather than being force-merged,
    which would risk rewriting live entries without LLM confirmation."""
    counts = _tag_counts(entries)
    existing_by_name = {t.get("name"): t for t in (existing_registry or []) if t.get("name")}

    protected_names = sorted(protected, key=lambda n: (-counts.get(n, 0), n))
    other_names = sorted((n for n in counts if n not in protected), key=lambda n: (-counts[n], n))
    budget = max(cap - len(protected_names), 0)
    kept_names = protected_names + other_names[:budget]

    registry = []
    for name in kept_names:
        prior = existing_by_name.get(name, {})
        registry.append({
            "name": name,
            "description": prior.get("description", ""),
            "aliases": prior.get("aliases", []),
            "count": counts.get(name, 0),
            "protected": name in protected,
            "provisional": False,
        })
    registry.sort(key=lambda t: (-t["count"], t["name"]))
    return registry


async def _run_tag_normalize_pass(
    entries: List[Dict], owner: Optional[str], cap: int, protected: Set[str], dry_run: bool,
) -> List[Dict]:
    counts = _tag_counts(entries)
    merges = await _propose_tag_merges(counts, owner)
    applied = _apply_tag_merges(entries, merges, protected)
    for frm, into in applied:
        _log_action(owner, "retag", before={"tag": frm}, after={"tag": into}, dry_run=dry_run)
    return entries


# ---- Pass 4: re-scoring (pure code) ----

def _run_rescore_pass(entries: List[Dict], owner: Optional[str], dry_run: bool) -> List[Dict]:
    for e in entries:
        score = tier_score(e)
        current = e.get("tier", 2)
        new_tier = next_tier(current, score)
        if new_tier != current:
            action = "promote" if new_tier < current else "demote"
            _log_action(
                owner, action,
                before={"id": e.get("id"), "tier": current},
                after={"id": e.get("id"), "tier": new_tier},
                dry_run=dry_run,
            )
            e["tier"] = new_tier
    return entries


# ---- Pass 5: expiry (pure code) ----

def _run_expire_pass(
    entries: List[Dict], owner: Optional[str], protected: Set[str], expiry_days: float, dry_run: bool,
) -> List[Dict]:
    now = time.time()
    survivors: List[Dict] = []
    for e in entries:
        if e.get("tier") != TIER_ARCHIVE or e.get("pinned"):
            survivors.append(e)
            continue
        if protected & set(e.get("tags") or []):
            survivors.append(e)
            continue
        last_used = e.get("last_used_at") or e.get("timestamp") or 0
        age_days = (now - last_used) / 86400.0 if last_used else float("inf")
        if age_days >= expiry_days:
            _log_action(owner, "expire", before=copy.deepcopy(e), after=None, dry_run=dry_run)
        else:
            survivors.append(e)
    return survivors


# ---- Pass 6: context document rebuild ----

CONTEXT_SUMMARY_PROMPT = (
    "Summarize the following personal facts into a short list of core facts "
    "about the user, removing duplicates and near-duplicates. Return a JSON "
    "array of strings, one fact per entry. Return ONLY valid JSON, no "
    "markdown fences, no commentary."
)

CONTEXT_SUMMARY_MAX_TOKENS = 1024
_MAX_CORE_FACTS = 30


async def _summarize_core_facts(texts: List[str], owner: Optional[str]) -> List[str]:
    texts = list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))
    if not texts:
        return []

    from src.task_endpoint import memory_llm_call_async

    messages = [
        {"role": "system", "content": CONTEXT_SUMMARY_PROMPT},
        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, temperature=0.1, max_tokens=CONTEXT_SUMMARY_MAX_TOKENS,
        )
        parsed = _parse_json_array(raw)
        if parsed:
            out = [str(x).strip() for x in parsed if str(x).strip()]
            if out:
                return out[:_MAX_CORE_FACTS]
    except Exception as e:
        logger.warning("Curator core-facts summary failed for owner=%r: %s", owner, e)
    return texts[:_MAX_CORE_FACTS]


async def _run_context_doc_pass(
    owner: Optional[str], entries: List[Dict], cap: int, protected: Set[str], dry_run: bool,
) -> Dict:
    from services.memory.memory_context import MemoryContext

    ctx = MemoryContext(owner)
    doc = ctx.load()
    registry = build_registry(entries, cap, protected, existing_registry=doc.get("tag_registry"))

    core_pinned = [e for e in entries if e.get("tier") == TIER_CORE or e.get("pinned")]
    core_facts = await _summarize_core_facts([e.get("text", "") for e in core_pinned], owner)

    stats = {
        "total": len(entries),
        "by_tier": {str(t): sum(1 for e in entries if e.get("tier") == t) for t in (0, 1, 2, 3)},
        "provisional_tag_count": sum(1 for e in entries if e.get("provisional_tags")),
    }
    new_doc = {"version": doc.get("version", 1), "core_facts": core_facts, "tag_registry": registry, "stats": stats}
    if not dry_run:
        ctx.save(new_doc)
    return new_doc


# ---- orchestrator ----

async def curate(memory_manager, memory_vector, owner: Optional[str] = None, dry_run: bool = False) -> Dict:
    """Run the full curator pipeline for one owner.

    Never raises: a per-batch LLM failure degrades that batch to a no-op
    (see `_triage_batch` / `_dedupe_batch` / `_propose_tag_merges`); this
    function itself can still raise on genuine programming errors, but no
    LLM/network failure should propagate past it.
    """
    from src.settings import get_setting

    all_entries = memory_manager.load_all()
    existing = _entries_for_owner(all_entries, owner)
    own_ids = {e["id"] for e in existing}
    before_count = len(existing)

    if before_count == 0:
        return {"status": "empty", "before": 0, "after": 0, "already_tidy": True}

    if not dry_run:
        current_fp = _fingerprint_entries(existing)
        state_file = _tidy_state_path(memory_manager)
        last_fp = _load_tidy_state(state_file).get(owner or "", {}).get("fingerprint")
        if last_fp == current_fp and not _load_checkpoint(owner):
            return {"status": "unchanged", "before": before_count, "after": before_count, "already_tidy": True}

    batch_size = max(1, int(get_setting("memory_curator_batch", 25) or 25))
    cap = max(1, int(get_setting("memory_tag_registry_cap", 50) or 50))
    expiry_days = max(1, int(get_setting("memory_archive_expiry_days", 90) or 90))
    protected = {
        t.strip().lower()
        for t in (get_setting("memory_protected_tags", ",".join(DEFAULT_PROTECTED_TAGS)) or "").split(",")
        if t.strip()
    } or set(DEFAULT_PROTECTED_TAGS)

    checkpoint = _load_checkpoint(owner) if not dry_run else {}
    last_completed = checkpoint.get("last_completed_pass")
    start_idx = (PASS_ORDER.index(last_completed) + 1) if last_completed in PASS_ORDER else 0

    def _persist(current_entries: List[Dict]) -> None:
        """Merge this owner's working set back into the live store (preserving
        other owners and any entries added concurrently during the run) and
        save. Reloads fresh under the lock every time so concurrent writers
        aren't clobbered."""
        with memory_manager.lock:
            fresh_all = memory_manager.load_all()
            others = [e for e in fresh_all if e["id"] not in own_ids]
            memory_manager.save(current_entries + others)

    entries = [copy.deepcopy(e) for e in existing]
    for idx in range(start_idx, len(PASS_ORDER)):
        pass_name = PASS_ORDER[idx]
        if pass_name == "triage":
            entries = await _run_triage_pass(entries, owner, batch_size, dry_run)
        elif pass_name == "dedupe":
            entries = await _run_dedupe_pass(entries, owner, batch_size, dry_run)
        elif pass_name == "tag_normalize":
            entries = await _run_tag_normalize_pass(entries, owner, cap, protected, dry_run)
        elif pass_name == "rescore":
            entries = _run_rescore_pass(entries, owner, dry_run)
        elif pass_name == "expire":
            entries = _run_expire_pass(entries, owner, protected, expiry_days, dry_run)
        elif pass_name == "context_doc":
            await _run_context_doc_pass(owner, entries, cap, protected, dry_run)
        if not dry_run:
            # Write-ahead ordering: a completed pass's mutations MUST be on
            # disk before the checkpoint marks it done. curate() only ever
            # persisted at the very end before this, so a crash in a later
            # pass would advance the checkpoint (per-pass) yet lose every
            # in-memory mutation — the resumed run then reloads the
            # un-mutated store and SKIPS the "completed" pass, stranding
            # entries (e.g. untriaged, generality=None, which the tags-only
            # fingerprint short-circuit could then make permanent).
            _persist(entries)
            _save_checkpoint(owner, pass_name)

    after_count = len(entries)

    if not dry_run:
        _clear_checkpoint(owner)
        with memory_manager.lock:
            fresh_all = memory_manager.load_all()
            others = [e for e in fresh_all if e["id"] not in own_ids]
        if memory_vector is not None and getattr(memory_vector, "healthy", False):
            try:
                memory_vector.rebuild(entries + others)
            except Exception as e:
                logger.warning("Curator vector rebuild failed for owner=%r: %s", owner, e)
        _save_tidy_state(_tidy_state_path(memory_manager), owner, _fingerprint_entries(entries))

    return {
        "status": "dry_run" if dry_run else "done",
        "before": before_count,
        "after": after_count,
        "already_tidy": False,
    }


async def triage_new_entries(memory_manager, owner: Optional[str] = None) -> Dict:
    """Cheap post-extraction pass fired from the AUDIT_INTERVAL trigger in
    memory_extractor.extract_and_store: only resolves generality/provisional
    tags for entries needing triage. Dedupe, tag normalization, rescoring,
    expiry, and the context-doc rebuild stay nightly-only (full `curate`)."""
    from src.settings import get_setting

    all_entries = memory_manager.load_all()
    existing = _entries_for_owner(all_entries, owner)
    pending = [copy.deepcopy(e) for e in existing if _needs_triage(e)]
    if not pending:
        return {"triaged": 0}

    batch_size = max(1, int(get_setting("memory_curator_batch", 25) or 25))
    pending_ids = {e["id"] for e in pending}
    for batch in cluster_batches(pending, batch_size):
        actions = await _triage_batch(batch, owner, dry_run=False)
        for action, before, after in actions:
            _log_action(owner, action, before=before, after=after, dry_run=False)

    with memory_manager.lock:
        fresh_all = memory_manager.load_all()
        others = [e for e in fresh_all if e["id"] not in pending_ids]
        memory_manager.save(pending + others)

    return {"triaged": len(pending)}


# ---- fingerprint short-circuit state (mirrors the pre-Phase-4 tidy state
# sidecar so a second nightly run with nothing changed skips the LLM
# entirely; owned here now since memory_extractor no longer runs a
# whole-store audit) ----

def _tidy_state_path(memory_manager) -> str:
    return os.path.join(os.path.dirname(memory_manager.memory_file), "memory_tidy_state.json")


def _load_tidy_state(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_tidy_state(path: str, owner: Optional[str], fingerprint: str) -> None:
    state = _load_tidy_state(path)
    state[owner or ""] = {"fingerprint": fingerprint}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        logger.warning("Could not persist curator tidy fingerprint: %s", e)
