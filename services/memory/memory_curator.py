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

Resilience escalation for the two entry-batched passes (triage, dedupe;
`_run_pass_resilient`): since `cluster_batches` is fully deterministic given
unchanged content, a batch that fails would otherwise fail *identically*
every single future run. Three cheapest-first layers:
  1. **In-run retry** — the same batch content is retried a couple of times
     (`non_json_reply` and `llm_error` only) before being treated as failed;
     catches ordinary model flakiness at ~zero cost.
  2. **Bisection** — a batch that still fails with `non_json_reply` (and
     ONLY that reason — see below) is split in half and each half retried/
     bisected recursively down to single entries. Smaller prompts are both
     less likely to trip a weak model's format-following AND, if one entry's
     content really is the problem, isolates it instead of leaving the
     whole batch un-triaged/un-deduped.
  3. **Quarantine** — if a single entry alone still fails the same pass
     across `memory_curator_quarantine_after` separate `curate()` runs
     (data/memory_curation_failures/<owner>.json), it's excluded from that
     pass's batches until a human clears it (Curator tab) — a backstop so a
     chronically-bad entry doesn't burn LLM calls forever while blocking
     itself from ever being triaged/deduped.
Deliberately NOT bisected/quarantined: dedupe's `>50% removed` safety-guard
refusal (`unsafe_removal_refused`) is a considered judgment the model made
correctly, not a parse/format failure — shrinking the batch until it drops
below `_UNSAFE_REMOVAL_MIN_BATCH` would silently defeat the guard rather than
fix anything, so that reason is never retried, bisected, or quarantined; the
whole batch is left unchanged exactly as before this feature existed.
`llm_error` (the call itself raised — timeout/network/API error) IS retried
(transient failures are exactly what a retry is for) but never bisected — a
dead/slow endpoint fails at any batch size, so shrinking it just multiplies
calls against something that isn't coming back this run.

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
CURATION_FAILURES_DIR = os.path.join(DATA_DIR, "memory_curation_failures")

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


# ---- resilience: per-entry-per-pass failure tracking + quarantine ----
#
# Sidecar (mirrors the checkpoint/tidy-state sidecar pattern) rather than a
# field on the entry itself: keeps MemoryManager's schema and
# _fingerprint_entries (id+text+tags only) untouched, and keeps this bookkeeping
# out of exports/API responses that read entries directly.
# Shape: {"<entry_id>": {"<pass_name>": {"count": int, "last_error": str,
#                                          "last_ts": float, "quarantined": bool}}}

def _failure_state_path(owner: Optional[str]) -> str:
    return os.path.join(CURATION_FAILURES_DIR, f"{_owner_slug(owner)}.json")


def _load_failure_state(owner: Optional[str]) -> Dict:
    try:
        with open(_failure_state_path(owner), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_failure_state(owner: Optional[str], state: Dict) -> None:
    os.makedirs(CURATION_FAILURES_DIR, exist_ok=True)
    path = _failure_state_path(owner)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _quarantined_ids(state: Dict, pass_name: str) -> Set[str]:
    return {eid for eid, passes in state.items() if (passes.get(pass_name) or {}).get("quarantined")}


def _quarantine_threshold() -> int:
    from src.settings import get_setting
    return max(1, int(get_setting("memory_curator_quarantine_after", 3) or 3))


def _handle_quarantine_candidate(owner: Optional[str], pass_name: str, entry: Dict, reason: str, dry_run: bool) -> None:
    """Called on the terminal single-entry failure of a bisectable reason —
    bumps that entry's failure count for this pass and flips `quarantined`
    once the threshold is crossed. Dry runs preview (log a `quarantine` line
    with the projected count) without ever writing the sidecar, matching how
    every other pass already logs proposed-but-unapplied actions."""
    entry_id = entry.get("id")
    if entry_id is None:
        return
    threshold = _quarantine_threshold()

    if dry_run:
        state = _load_failure_state(owner)
        prior = ((state.get(entry_id) or {}).get(pass_name) or {}).get("count", 0)
        _log_action(
            owner, "quarantine",
            before=None,
            after={"id": entry_id, "pass": pass_name, "count": prior + 1, "threshold": threshold, "error": reason},
            dry_run=True,
        )
        return

    state = _load_failure_state(owner)
    rec = state.setdefault(entry_id, {}).setdefault(pass_name, {"count": 0, "quarantined": False})
    rec["count"] = rec.get("count", 0) + 1
    rec["last_error"] = reason
    rec["last_ts"] = time.time()
    newly_quarantined = rec["count"] >= threshold and not rec.get("quarantined")
    if newly_quarantined:
        rec["quarantined"] = True
    _save_failure_state(owner, state)
    if newly_quarantined:
        _log_action(
            owner, "quarantine",
            before=None,
            after={"id": entry_id, "pass": pass_name, "count": rec["count"], "error": reason},
            dry_run=False,
        )


def clear_quarantine(memory_manager, owner: Optional[str], entry_id: str, pass_name: Optional[str] = None) -> bool:
    """Manual recovery (Curator tab "clear"/"retry" button): un-quarantines
    one pass, or every pass, for a given entry so the next curate() run
    includes it again. Returns True iff something was actually cleared.

    Also invalidates the saved tidy fingerprint: un-quarantining doesn't
    change the entry's id/text/tags (what the fingerprint tracks), so
    without this the next curate() call would still see a matching
    fingerprint and short-circuit on `already_tidy`, never actually
    retrying the entry we just un-quarantined."""
    state = _load_failure_state(owner)
    passes = state.get(entry_id)
    if not passes:
        return False
    names = [pass_name] if pass_name else list(passes.keys())
    cleared = []
    for name in names:
        rec = passes.get(name)
        if rec and rec.get("quarantined"):
            rec["quarantined"] = False
            rec["count"] = 0
            cleared.append(name)
    if not cleared:
        return False
    _save_failure_state(owner, state)
    for name in cleared:
        _log_action(owner, "unquarantine", before=None, after={"id": entry_id, "pass": name})
    _invalidate_tidy_state(memory_manager, owner)
    return True


def list_quarantined(owner: Optional[str]) -> List[Dict]:
    """Every currently-quarantined (entry, pass) pair for an owner, for the
    Curator tab's manual-review list."""
    state = _load_failure_state(owner)
    out = []
    for entry_id, passes in state.items():
        for pass_name, rec in passes.items():
            if rec.get("quarantined"):
                out.append({
                    "id": entry_id, "pass": pass_name, "count": rec.get("count", 0),
                    "last_error": rec.get("last_error"), "last_ts": rec.get("last_ts"),
                })
    return out


# In-run retry attempts beyond the first, same batch/content — cheap, catches
# ordinary model flakiness. Bisection only kicks in once these are exhausted.
_BATCH_RETRY_ATTEMPTS = 2
_BISECT_MIN_SIZE = 1
# Only a parse failure is plausibly content-correlated enough to bisect; see
# module docstring for why llm_error and unsafe_removal_refused are excluded.
_BISECTABLE_REASONS = {"non_json_reply"}


def _classify_reason(local_failures: List[str]) -> Optional[str]:
    """`local_failures` entries look like '<pass>:llm_error:<detail>',
    '<pass>:non_json_reply', or '<pass>:unsafe_removal_refused' (the only
    strings `_triage_batch`/`_dedupe_batch` ever append) — returns just the
    category, or None if nothing was appended (a legitimate no-op result)."""
    if not local_failures:
        return None
    last = local_failures[-1]
    for cat in ("non_json_reply", "unsafe_removal_refused", "llm_error"):
        if cat in last:
            return cat
    return "unknown"


async def _run_pass_resilient(
    batch: List[Dict], call_batch, combine, owner: Optional[str], pass_name: str,
    dry_run: bool, master_failures: Optional[List[str]],
):
    """Runs `call_batch(sub_batch, local_failures) -> value` over `batch`,
    escalating retry -> bisect -> quarantine on a persistent failure (see
    module docstring). `combine(value, value) -> value` merges two bisected
    halves back together (list concatenation for triage's actions, tuple-wise
    concatenation for dedupe's (survivors, actions)).

    Exactly one entry is appended to `master_failures` per unresolved leaf
    (not per retry attempt) — this is what still gates `curate()`'s
    tidy-fingerprint save."""
    attempts = 1 + _BATCH_RETRY_ATTEMPTS
    value = None
    reason = None
    for _ in range(attempts):
        local: List[str] = []
        value = await call_batch(batch, local)
        reason = _classify_reason(local)
        if reason is None:
            return value
        if reason == "unsafe_removal_refused":
            break  # a considered judgment, not a transient/content glitch — never retried

    if reason in _BISECTABLE_REASONS and len(batch) > _BISECT_MIN_SIZE:
        mid = len(batch) // 2
        left = await _run_pass_resilient(batch[:mid], call_batch, combine, owner, pass_name, dry_run, master_failures)
        right = await _run_pass_resilient(batch[mid:], call_batch, combine, owner, pass_name, dry_run, master_failures)
        return combine(left, right)

    if master_failures is not None:
        master_failures.append(f"{pass_name}:{reason}")
    if reason in _BISECTABLE_REASONS and len(batch) == _BISECT_MIN_SIZE:
        _handle_quarantine_candidate(owner, pass_name, batch[0], reason, dry_run)
    return value


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


def _parse_json_object(raw: str) -> Optional[Dict]:
    """Object-shaped counterpart of `_parse_json_array` (mirrors
    `services/memory/retrieval.py`'s tolerance). Used by the dedupe pass, whose
    reply is a `{"merges": [...], "remove": [...]}` object."""
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
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        logger.debug("Curator got non-JSON object reply: %r", (raw or "")[:160])
        return None
    return parsed if isinstance(parsed, dict) else None


# JSON-mode constraint sent on every curator LLM pass. Asks the serving backend
# to restrict output to valid JSON, which eliminates the prose/prompt-echo/
# markdown-fence failures a weak local "smart" model produces that the tolerant
# parsers above can't recover from. `json_object` (not a strict schema) for broad
# local-backend support — llama.cpp/vLLM/Ollama all permit JSON *arrays* under it;
# `llm_call_async` degrades to prompt-only if a backend rejects the field.
_JSON_MODE = {"type": "json_object"}


def _user_message(payload: object, reminder: str) -> Dict:
    """User turn = the JSON payload plus a trailing output-format reminder. Small
    models weight the final user message most, so the "return only JSON" nudge
    lands harder here than in the system prompt — belt-and-suspenders with
    `_JSON_MODE`, and the only steer left when a backend can't honor it."""
    return {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False) + "\n\n" + reminder,
    }


# ---- Pass 1: triage ----

TRIAGE_SYSTEM_PROMPT = (
    "You triage personal memory entries for a memory system. For EVERY input "
    "entry, return one object with:\n"
    "1. \"generality\": integer 0-3 — how broadly useful the fact is across "
    "future conversations. 3 = identity-level (name, job, home city, close "
    "relations). 2 = stable preference/relationship. 1 = project- or time-bound "
    "context. 0 = conversation-specific trivia. HIGHER = MORE durable/general. "
    "Always return a value for every entry.\n"
    "2. \"promote_tags\": from the entry's provisional_tags, list ONLY the ones "
    "genuinely useful going forward (these are kept). Any provisional tag you do "
    "not list is discarded. Use [] to discard all of them.\n\n"
    "Input: a JSON array of {id, text, tags, provisional_tags, generality}.\n"
    "Output: a JSON array of {id, generality, promote_tags} — EXACTLY one object "
    "per input entry, reusing the SAME id.\n\n"
    "Example input:\n"
    "[{\"id\": \"m1\", \"text\": \"User's name is Sam\", \"tags\": [\"identity\"], "
    "\"provisional_tags\": [\"person-sam\", \"stray-typo\"], \"generality\": null},\n"
    " {\"id\": \"m2\", \"text\": \"Prefers tabs over spaces\", \"tags\": [], "
    "\"provisional_tags\": [], \"generality\": null}]\n"
    "Example output:\n"
    "[{\"id\": \"m1\", \"generality\": 3, \"promote_tags\": [\"person-sam\"]},\n"
    " {\"id\": \"m2\", \"generality\": 2, \"promote_tags\": []}]\n\n"
    "Return ONLY valid JSON, no markdown fences, no commentary."
)

TRIAGE_MAX_TOKENS = 2048


def _needs_triage(entry: Dict) -> bool:
    return entry.get("generality") is None or bool(entry.get("provisional_tags"))


async def _triage_batch(
    batch: List[Dict], owner: Optional[str], dry_run: bool, interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[Tuple[str, Dict, Dict]]:
    """Mutates entries in `batch` in place; returns a list of
    (action, before_snapshot, after_snapshot) tuples for the changelog.

    Appends a short description to `failures` (if given) whenever the LLM
    call raises or the reply doesn't parse as JSON — as opposed to a
    legitimate empty/no-op reply — so the caller can tell "nothing needed
    doing" apart from "we never got a usable answer" (see `curate()`'s
    tidy-fingerprint gating)."""
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
        _user_message(payload, "Return ONLY the JSON array, nothing else."),
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, interactive=interactive,
            temperature=0.1, max_tokens=TRIAGE_MAX_TOKENS,
            response_format=_JSON_MODE,
        )
    except Exception as e:
        logger.warning("Curator triage batch failed for owner=%r: %s", owner, e)
        if failures is not None:
            failures.append(f"triage:llm_error:{e}")
        return []

    results = _parse_json_array(raw)
    if results is None:
        if failures is not None:
            failures.append("triage:non_json_reply")
        return []
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


async def _run_triage_pass(
    entries: List[Dict], owner: Optional[str], batch_size: int, dry_run: bool, interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[Dict]:
    quarantined = _quarantined_ids(_load_failure_state(owner), "triage")
    pending = [e for e in entries if _needs_triage(e) and e.get("id") not in quarantined]

    async def call_batch(sub_batch: List[Dict], local_failures: List[str]) -> List[Tuple[str, Dict, Dict]]:
        return await _triage_batch(sub_batch, owner, dry_run, interactive, local_failures)

    for batch in cluster_batches(pending, batch_size):
        actions = await _run_pass_resilient(batch, call_batch, lambda a, b: a + b, owner, "triage", dry_run, failures)
        for action, before, after in actions:
            _log_action(owner, action, before=before, after=after, dry_run=dry_run)
    return entries


# ---- Pass 2: dedupe / merge ----

DEDUPE_SYSTEM_PROMPT = (
    "You are a memory database curator. Be CONSERVATIVE: identify only TRUE "
    "duplicates and genuinely worthless entries. Every distinct fact must "
    "survive. When in doubt, KEEP (merge nothing, remove nothing).\n\n"
    "You are given a JSON array of {id, text, tags}. You do NOT rewrite entry "
    "text — you only report which entries to MERGE or REMOVE, by id.\n\n"
    "Rules:\n"
    "1. MERGE entries that state the SAME fact in different words. Report each "
    "group as {\"keep\": \"<id to keep>\", \"drop\": [\"<id>\", ...]}. The "
    "dropped entries are deleted and their tags folded into the kept entry.\n"
    "   Merge: 'User's name is Sam' + 'The user is called Sam' -> keep one, drop "
    "the other.\n"
    "   Do NOT merge related-but-distinct facts: 'Likes Python' vs 'Uses Python "
    "at work' are DIFFERENT — keep both, merge neither.\n"
    "2. REMOVE only entries that are genuinely worthless: about what the AI did "
    "(not the user), empty, or meaningless. List their ids in \"remove\". Do NOT "
    "remove a real fact just because it seems minor or niche.\n"
    "3. Never invent facts or ids — use only ids present in the input. When "
    "unsure, leave an entry out of BOTH lists (it survives unchanged).\n\n"
    "Return a JSON object: {\"merges\": [{\"keep\": \"...\", \"drop\": "
    "[\"...\"]}], \"remove\": [\"...\"]}. Use empty arrays when nothing applies.\n"
    "Return ONLY valid JSON, no markdown fences."
)

DEDUPE_MAX_TOKENS = 4096

# A conservative dedupe pass should never cut a batch by more than half —
# below this batch size the check is skipped (too small a sample to judge).
_UNSAFE_REMOVAL_MIN_BATCH = 8
_UNSAFE_REMOVAL_RATIO = 0.5


async def _dedupe_batch(
    batch: List[Dict], owner: Optional[str], interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Tuple[str, list, list]]]:
    """Returns (surviving_entries, changelog_actions). Never raises; any
    failure or unsafe-looking result returns the batch unchanged.

    The model returns MERGE/REMOVE instructions by id — it never rewrites entry
    text (that keeps output small and un-truncatable, and makes text-mangling
    impossible; wording "optimization" is a deliberately separate future pass,
    not dedupe's job). This code applies them: `drop`ped entries are deleted and
    their tags unioned onto the kept entry, `remove`d entries are deleted, and
    every entry named in neither list survives verbatim.

    Appends to `failures` (if given) on an LLM/parse failure OR when the
    `>50% removed` safety guard refuses the result — both leave this
    batch's entries un-deduped, so both should block the caller from
    marking the store "tidy" (see `curate()`)."""
    if len(batch) < 2:
        return list(batch), []

    from src.memory import normalize_tags
    from src.task_endpoint import memory_llm_call_async

    payload = [{"id": e["id"], "text": e.get("text", ""), "tags": e.get("tags") or []} for e in batch]
    messages = [
        {"role": "system", "content": DEDUPE_SYSTEM_PROMPT},
        _user_message(payload, "Return ONLY the JSON object, nothing else."),
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, interactive=interactive,
            temperature=0.1, max_tokens=DEDUPE_MAX_TOKENS, timeout=120,
            response_format=_JSON_MODE,
        )
    except Exception as e:
        logger.warning("Curator dedupe batch failed for owner=%r: %s", owner, e)
        if failures is not None:
            failures.append(f"dedupe:llm_error:{e}")
        return list(batch), []

    result = _parse_json_object(raw)
    if result is None:
        if failures is not None:
            failures.append("dedupe:non_json_reply")
        return list(batch), []

    by_id = {e["id"]: e for e in batch}

    # Collect validated merge groups (keep_entry, [drop_entries]); each id can be
    # dropped at most once and never drops the entry it would merge into.
    merge_groups: List[Tuple[Dict, List[Dict]]] = []
    dropped_ids: Set[str] = set()
    for group in (result.get("merges") or []):
        if not isinstance(group, dict):
            continue
        keep = by_id.get(group.get("keep"))
        if keep is None:
            continue
        drops: List[Dict] = []
        for did in (group.get("drop") or []):
            entry = by_id.get(did)
            if entry is None or did == keep["id"] or did in dropped_ids:
                continue
            drops.append(entry)
            dropped_ids.add(did)
        if drops:
            merge_groups.append((keep, drops))

    remove_ids: Set[str] = {
        rid for rid in (result.get("remove") or [])
        if rid in by_id and rid not in dropped_ids
    }

    before_count = len(batch)
    removed_total = len(dropped_ids) + len(remove_ids)
    if before_count >= _UNSAFE_REMOVAL_MIN_BATCH and (before_count - removed_total) < before_count * _UNSAFE_REMOVAL_RATIO:
        logger.warning(
            "Curator dedupe batch would cut %s -> %s (>%.0f%% removed) — refusing, owner=%r",
            before_count, before_count - removed_total, _UNSAFE_REMOVAL_RATIO * 100, owner,
        )
        if failures is not None:
            failures.append("dedupe:unsafe_removal_refused")
        return list(batch), []

    # Apply merges: union dropped entries' tags onto the kept entry.
    actions: List[Tuple[str, list, list]] = []
    keep_updates: Dict[str, Dict] = {}
    for keep, drops in merge_groups:
        merged = copy.deepcopy(keep_updates.get(keep["id"], keep))
        union = list(merged.get("tags") or [])
        for entry in drops:
            union += entry.get("tags") or []
        merged["tags"] = normalize_tags(union)
        keep_updates[keep["id"]] = merged
        actions.append(("merge", [copy.deepcopy(d) for d in drops], [copy.deepcopy(merged)]))

    survivors: List[Dict] = []
    for e in batch:
        if e["id"] in dropped_ids or e["id"] in remove_ids:
            continue
        survivors.append(keep_updates.get(e["id"], e))

    # Worthless removals are logged as `expire` so they reuse the existing
    # single-snapshot undo path (before=full entry, after=None).
    for rid in remove_ids:
        actions.append(("expire", copy.deepcopy(by_id[rid]), None))

    return survivors, actions


async def _run_dedupe_pass(
    entries: List[Dict], owner: Optional[str], batch_size: int, dry_run: bool, interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[Dict]:
    quarantined = _quarantined_ids(_load_failure_state(owner), "dedupe")
    exempt = [e for e in entries if e.get("id") in quarantined]
    eligible = [e for e in entries if e.get("id") not in quarantined]

    async def call_batch(sub_batch: List[Dict], local_failures: List[str]) -> Tuple[List[Dict], List[Tuple[str, list, list]]]:
        return await _dedupe_batch(sub_batch, owner, interactive, local_failures)

    def combine(a, b):
        return (a[0] + b[0], a[1] + b[1])

    survivors: List[Dict] = []
    for batch in cluster_batches(eligible, batch_size):
        final, actions = await _run_pass_resilient(batch, call_batch, combine, owner, "dedupe", dry_run, failures)
        survivors.extend(final)
        for action, before, after in actions:
            _log_action(owner, action, before=before, after=after, dry_run=dry_run)
    # Quarantined entries were never sent to the LLM, but dedupe's output IS
    # the new surviving set — they must still come back, or they'd silently
    # vanish from the store (unlike triage, which returns all `entries`
    # regardless and just skips mutating the quarantined ones).
    return survivors + exempt


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


async def _propose_tag_merges(
    tag_counts: Dict[str, int], owner: Optional[str], interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    if len(tag_counts) < 2:
        return []

    from src.memory import normalize_tag
    from src.task_endpoint import memory_llm_call_async

    payload = [{"tag": t, "count": c} for t, c in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    messages = [
        {"role": "system", "content": TAG_MERGE_SYSTEM_PROMPT},
        _user_message(payload, "Return ONLY the JSON array, nothing else."),
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, interactive=interactive,
            temperature=0.1, max_tokens=TAG_MERGE_MAX_TOKENS,
            response_format=_JSON_MODE,
        )
    except Exception as e:
        logger.warning("Curator tag-merge proposal failed for owner=%r: %s", owner, e)
        if failures is not None:
            failures.append(f"tag_merge:llm_error:{e}")
        return []

    parsed = _parse_json_array(raw)
    if parsed is None:
        if failures is not None:
            failures.append("tag_merge:non_json_reply")
        return []
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
    entries: List[Dict], owner: Optional[str], cap: int, protected: Set[str], dry_run: bool, interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[Dict]:
    counts = _tag_counts(entries)
    merges = await _propose_tag_merges(counts, owner, interactive, failures)
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

_MAX_CORE_FACTS = 30

CONTEXT_SUMMARY_PROMPT = (
    "Summarize the following personal facts into a short list of core facts "
    "about the user, removing duplicates and near-duplicates. Return AT MOST "
    f"{_MAX_CORE_FACTS} facts, keeping the most durable/identity-level ones. "
    "Return a JSON array of strings, one fact per entry. Return ONLY valid JSON, "
    "no markdown fences, no commentary."
)

CONTEXT_SUMMARY_MAX_TOKENS = 1024


async def _summarize_core_facts(
    texts: List[str], owner: Optional[str], interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> List[str]:
    texts = list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))
    if not texts:
        return []

    from src.task_endpoint import memory_llm_call_async

    messages = [
        {"role": "system", "content": CONTEXT_SUMMARY_PROMPT},
        _user_message(texts, "Return ONLY the JSON array of strings, nothing else."),
    ]
    try:
        raw = await memory_llm_call_async(
            "smart", messages, owner=owner, interactive=interactive,
            temperature=0.1, max_tokens=CONTEXT_SUMMARY_MAX_TOKENS,
            response_format=_JSON_MODE,
        )
        parsed = _parse_json_array(raw)
        if parsed is None:
            if failures is not None:
                failures.append("context_doc:non_json_reply")
        elif parsed:
            out = [str(x).strip() for x in parsed if str(x).strip()]
            if out:
                return out[:_MAX_CORE_FACTS]
    except Exception as e:
        logger.warning("Curator core-facts summary failed for owner=%r: %s", owner, e)
        if failures is not None:
            failures.append(f"context_doc:llm_error:{e}")
    return texts[:_MAX_CORE_FACTS]


async def _run_context_doc_pass(
    owner: Optional[str], entries: List[Dict], cap: int, protected: Set[str], dry_run: bool, interactive: bool = False,
    failures: Optional[List[str]] = None,
) -> Dict:
    from services.memory.memory_context import MemoryContext

    ctx = MemoryContext(owner)
    doc = ctx.load()
    registry = build_registry(entries, cap, protected, existing_registry=doc.get("tag_registry"))

    core_pinned = [e for e in entries if e.get("tier") == TIER_CORE or e.get("pinned")]
    core_facts = await _summarize_core_facts([e.get("text", "") for e in core_pinned], owner, interactive, failures)

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

async def curate(
    memory_manager, memory_vector, owner: Optional[str] = None, dry_run: bool = False, interactive: bool = False,
    force: bool = False,
) -> Dict:
    """Run the full curator pipeline for one owner.

    Never raises: a per-batch LLM failure degrades that batch to a no-op
    (see `_triage_batch` / `_dedupe_batch` / `_propose_tag_merges`); this
    function itself can still raise on genuine programming errors, but no
    LLM/network failure should propagate past it. Every such failure (and
    the `>50% removed` dedupe safety refusal) is recorded in a `failures`
    list threaded through the passes; if that list is non-empty at the end
    of a live run, the tidy-fingerprint save is skipped so a future run
    retries the affected batches instead of being permanently short-
    circuited by `already_tidy` (see the end of this function).

    `interactive` selects the LLM foreground-gate mode for every memory-smart
    call the passes make. The nightly loop runs genuinely backgrounded
    (`interactive=False`, waits for foreground quiet). The manual
    `/api/memory/audit` route runs `curate()` INLINE inside its own tracked
    HTTP request, so it MUST pass `interactive=True`: with `False` the gate
    would wait for `_ACTIVE_REQUESTS == 0`, which includes the audit request
    itself, deadlocking the run (button stuck on "Running…", no LLM call ever
    issued).

    `force` bypasses only the tidy-fingerprint short-circuit below (the
    Curator tab's "Force run" button) — it does not affect checkpoint-resume
    or the dedupe safety guard.
    """
    from src.settings import get_setting

    all_entries = memory_manager.load_all()
    existing = _entries_for_owner(all_entries, owner)
    own_ids = {e["id"] for e in existing}
    before_count = len(existing)

    if before_count == 0:
        return {"status": "empty", "before": 0, "after": 0, "already_tidy": True}

    if not dry_run and not force:
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

    failures: List[str] = []
    entries = [copy.deepcopy(e) for e in existing]
    for idx in range(start_idx, len(PASS_ORDER)):
        pass_name = PASS_ORDER[idx]
        if pass_name == "triage":
            entries = await _run_triage_pass(entries, owner, batch_size, dry_run, interactive, failures)
        elif pass_name == "dedupe":
            entries = await _run_dedupe_pass(entries, owner, batch_size, dry_run, interactive, failures)
        elif pass_name == "tag_normalize":
            entries = await _run_tag_normalize_pass(entries, owner, cap, protected, dry_run, interactive, failures)
        elif pass_name == "rescore":
            entries = _run_rescore_pass(entries, owner, dry_run)
        elif pass_name == "expire":
            entries = _run_expire_pass(entries, owner, protected, expiry_days, dry_run)
        elif pass_name == "context_doc":
            await _run_context_doc_pass(owner, entries, cap, protected, dry_run, interactive, failures)
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
        if failures:
            logger.info(
                "Curator run for owner=%r had %d batch failure(s); skipping tidy "
                "fingerprint save so a future run retries: %s",
                owner, len(failures), failures[:5],
            )
        else:
            _save_tidy_state(_tidy_state_path(memory_manager), owner, _fingerprint_entries(entries))

    return {
        "status": "dry_run" if dry_run else "done",
        "before": before_count,
        "after": after_count,
        "already_tidy": False,
        "had_failures": bool(failures),
        "failure_count": len(failures),
    }


async def triage_new_entries(memory_manager, owner: Optional[str] = None) -> Dict:
    """Cheap post-extraction pass fired from the AUDIT_INTERVAL trigger in
    memory_extractor.extract_and_store: only resolves generality/provisional
    tags for entries needing triage. Dedupe, tag normalization, rescoring,
    expiry, and the context-doc rebuild stay nightly-only (full `curate`)."""
    from src.settings import get_setting

    all_entries = memory_manager.load_all()
    existing = _entries_for_owner(all_entries, owner)
    quarantined = _quarantined_ids(_load_failure_state(owner), "triage")
    pending = [copy.deepcopy(e) for e in existing if _needs_triage(e) and e.get("id") not in quarantined]
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


def _invalidate_tidy_state(memory_manager, owner: Optional[str]) -> None:
    """Drop a saved tidy fingerprint so the next `curate()` call doesn't
    short-circuit on `already_tidy`. Needed by `clear_quarantine`: unlike
    every other curator mutation, un-quarantining an entry changes nothing
    the fingerprint tracks (id+text+tags) — the entry's content never
    changed, only the sidecar's `quarantined` flag did — so without this the
    fingerprint would still match and the un-quarantined entry would never
    actually get retried."""
    path = _tidy_state_path(memory_manager)
    state = _load_tidy_state(path)
    key = owner or ""
    if key not in state:
        return
    del state[key]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        logger.warning("Could not invalidate curator tidy fingerprint: %s", e)
