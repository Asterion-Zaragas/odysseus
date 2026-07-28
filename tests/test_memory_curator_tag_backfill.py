"""tag_backfill: the curator pass that proposes registry-aware tags for
entries already sitting in the store (import-time legacy tags, or a
write-time tagger failure) — nothing else ever revisits an entry once it's
written. See .AGENT_CONTEXT/plans/2026-07-28-curator-tag-backfill.md.

Covers services/memory/memory_curator.py's `_tag_backfill_batch` /
`_run_tag_backfill_pass` / `_backfilled_ids` / `_mark_backfilled`, plus the
pass wired into the full `curate()` pipeline at the front of `PASS_ORDER`.
"""

import json
import tempfile

import pytest

from services.memory import memory_context as mc
from services.memory import memory_curator as cur
from src.memory import MemoryManager


@pytest.fixture()
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setattr(cur, "CURATION_FAILURES_DIR", str(tmp_path / "failures"))
    monkeypatch.setattr(cur, "TAG_BACKFILL_STATE_DIR", str(tmp_path / "tag_backfill_state"))
    monkeypatch.setattr(mc, "CONTEXT_DIR", str(tmp_path / "context"))
    return tmp_path


def _seed_registry(owner, names):
    """Bootstrap a non-empty tag registry so proposed tags can land as real
    (not provisional) — mirrors how a prior curator run's tag_normalize/
    context_doc pass would have populated it."""
    ctx = mc.MemoryContext(owner)
    doc = ctx.load()
    doc["tag_registry"] = [
        {"name": n, "description": "", "aliases": [], "count": 5, "protected": False, "provisional": False}
        for n in names
    ]
    ctx.save(doc)


# ── sidecar ──

def test_backfilled_ids_round_trip(_isolated_dirs):
    assert cur._backfilled_ids("alice") == set()
    cur._mark_backfilled("alice", {"a", "b"})
    assert cur._backfilled_ids("alice") == {"a", "b"}
    cur._mark_backfilled("alice", {"c"})
    assert cur._backfilled_ids("alice") == {"a", "b", "c"}


def test_mark_backfilled_noop_on_empty_ids(_isolated_dirs):
    cur._mark_backfilled("alice", set())
    assert cur._backfilled_ids("alice") == set()


def test_needs_tag_backfill_gate():
    entry = {"id": "x"}
    assert cur._needs_tag_backfill(entry, set()) is True
    assert cur._needs_tag_backfill(entry, {"x"}) is False


# ── _tag_backfill_batch: additive merge semantics ──

async def test_tag_backfill_batch_adds_registry_tags_without_losing_legacy_tag(_isolated_dirs, monkeypatch):
    _seed_registry("alice", ["identity", "person:sam"])
    entry = {"id": "m1", "text": "User's name is Sam", "tags": ["identity"], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "tags": ["person:sam"], "new_tags": []}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    actions, scanned = await cur._tag_backfill_batch([entry], "alice", cap=50)

    assert scanned == {"m1"}
    assert entry["tags"] == ["identity", "person:sam"]  # legacy tag survives, new one added
    assert len(actions) == 1
    action, before, after = actions[0]
    assert action == "tag_backfill"
    assert before["tags"] == ["identity"]
    assert after["tags"] == ["identity", "person:sam"]


async def test_tag_backfill_batch_cold_start_lands_provisional(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": "User's name is Sam", "tags": ["identity"], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "tags": [], "new_tags": ["person:sam"]}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    actions, scanned = await cur._tag_backfill_batch([entry], "alice", cap=50)

    assert entry["tags"] == ["identity"]  # nothing promoted to real tags
    assert entry["provisional_tags"] == ["person:sam"]
    assert scanned == {"m1"}


async def test_tag_backfill_batch_never_drops_existing_tags(_isolated_dirs, monkeypatch):
    _seed_registry("alice", ["identity"])
    entry = {"id": "m1", "text": "x", "tags": ["identity", "legacy-flat-cat"], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "tags": [], "new_tags": []}])  # nothing to add

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    actions, scanned = await cur._tag_backfill_batch([entry], "alice", cap=50)

    assert entry["tags"] == ["identity", "legacy-flat-cat"]
    assert actions == []  # no-op: entry unchanged
    assert scanned == {"m1"}  # but still counted as scanned


async def test_tag_backfill_batch_omitted_entry_still_scanned(_isolated_dirs, monkeypatch):
    """The model may omit an entry from its reply entirely when it has
    nothing to add — that's a valid outcome, not a failure, so the entry is
    still marked scanned."""
    e1 = {"id": "m1", "text": "a", "tags": [], "provisional_tags": [], "generality": 2}
    e2 = {"id": "m2", "text": "b", "tags": [], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "tags": [], "new_tags": ["foo"]}])  # m2 omitted

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    actions, scanned = await cur._tag_backfill_batch([e1, e2], "alice", cap=50)

    assert scanned == {"m1", "m2"}
    assert e2["tags"] == [] and e2["provisional_tags"] == []


async def test_tag_backfill_batch_non_json_reply_scans_nothing(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": "x", "tags": [], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return "not json at all"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    failures = []
    actions, scanned = await cur._tag_backfill_batch([entry], "alice", cap=50, failures=failures)

    assert actions == []
    assert scanned == set()
    assert failures == ["tag_backfill:non_json_reply"]


# ── _run_tag_backfill_pass: gating + sidecar persistence ──

async def test_run_tag_backfill_pass_skips_already_backfilled(_isolated_dirs, monkeypatch):
    e1 = {"id": "m1", "text": "a", "tags": [], "provisional_tags": [], "generality": 2}
    cur._mark_backfilled("alice", {"m1"})

    calls = {"n": 0}

    async def fake_llm(role, messages, owner=None, **kwargs):
        calls["n"] += 1
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    await cur._run_tag_backfill_pass([e1], "alice", batch_size=25, cap=50, dry_run=False)

    assert calls["n"] == 0  # already-scanned entry never sent to the LLM


async def test_run_tag_backfill_pass_marks_sidecar_only_on_success_not_dry_run(_isolated_dirs, monkeypatch):
    e1 = {"id": "m1", "text": "a", "tags": [], "provisional_tags": [], "generality": 2}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    await cur._run_tag_backfill_pass([e1], "alice", batch_size=25, cap=50, dry_run=True)
    assert cur._backfilled_ids("alice") == set()  # dry run never persists the sidecar

    await cur._run_tag_backfill_pass([e1], "alice", batch_size=25, cap=50, dry_run=False)
    assert cur._backfilled_ids("alice") == {"m1"}


async def test_run_tag_backfill_pass_excludes_quarantined_entries(_isolated_dirs, monkeypatch):
    e1 = {"id": "m1", "text": "a", "tags": [], "provisional_tags": [], "generality": 2}
    for _ in range(cur._quarantine_threshold()):
        cur._handle_quarantine_candidate("alice", "tag_backfill", e1, "non_json_reply", dry_run=False)
    quarantined = cur.list_quarantined("alice")
    assert len(quarantined) == 1
    assert quarantined[0]["id"] == "m1"
    assert quarantined[0]["pass"] == "tag_backfill"

    calls = {"n": 0}

    async def fake_llm(role, messages, owner=None, **kwargs):
        calls["n"] += 1
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    await cur._run_tag_backfill_pass([e1], "alice", batch_size=25, cap=50, dry_run=False)
    assert calls["n"] == 0


# ── resilience: bisection + quarantine reuse the shared machinery ──

async def test_tag_backfill_bisects_and_quarantines_poison_entry(_isolated_dirs, monkeypatch):
    poison = {"id": "poison", "text": "bad", "tags": [], "provisional_tags": [], "generality": 2}
    fine = [{"id": str(i), "text": f"fact {i}", "tags": [], "provisional_tags": [], "generality": 2} for i in range(4)]
    entries = fine + [poison]

    async def fake_llm(role, messages, owner=None, **kwargs):
        content = messages[-1]["content"]
        if '"poison"' in content:
            return "the model rambles instead of JSON"
        payload = json.loads(content.split("\n\n")[0])
        return json.dumps([{"id": e["id"], "tags": [], "new_tags": []} for e in payload["entries"]])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    threshold = cur._quarantine_threshold()
    for _ in range(threshold):
        failures = []
        await cur._run_tag_backfill_pass(entries, "alice", batch_size=25, cap=50, dry_run=False, failures=failures)
        assert failures  # the poison leaf always fails

    # The 4 fine entries got scanned (and marked) on the very first run despite
    # sharing a batch with the poison entry — bisection isolated it.
    assert {"0", "1", "2", "3"} <= cur._backfilled_ids("alice")
    assert "poison" not in cur._backfilled_ids("alice")
    quarantined = cur.list_quarantined("alice")
    assert len(quarantined) == 1 and quarantined[0]["id"] == "poison" and quarantined[0]["pass"] == "tag_backfill"


# ── full curate() pipeline integration ──

async def test_curate_full_pipeline_backfills_legacy_tagged_entries(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User's name is Sam", owner="alice")
        e1["tags"] = ["identity"]
        e1["generality"] = 2  # already triaged: this run's only job is to backfill tags
        e2 = mgr.add_entry("User prefers dark mode", owner="alice")
        e2["tags"] = ["preference"]
        e2["generality"] = 2
        mgr.save([e1, e2])

        async def fake_llm(role, messages, owner=None, **kwargs):
            sys = messages[0]["content"]
            if "REGISTRY" in sys and "new_tags" in sys:  # tag_backfill
                payload = json.loads(messages[-1]["content"].split("\n\n")[0])
                return json.dumps([
                    {"id": item["id"], "tags": [], "new_tags": [f"person:sam" if "Sam" in item["text"] else "dev"]}
                    for item in payload["entries"]
                ])
            if '"from"' in sys and '"into"' in sys:  # tag_merge — checked before the
                return "[]"                          # "duplicate"/"MERGE" substring below
            if "duplicate" in sys.lower() or "MERGE" in sys:  # dedupe: keep both
                return json.dumps({"merges": [], "remove": []})
            return "[]"  # triage / context-facts summary

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert result["status"] == "done"
        assert result["had_failures"] is False

        stored = {e["id"]: e for e in mgr.load(owner="alice")}
        # Cold-start registry: proposed tags land provisional, but the
        # original legacy tag is never touched.
        assert stored[e1["id"]]["tags"] == ["identity"]
        assert "person:sam" in stored[e1["id"]]["provisional_tags"]
        assert stored[e2["id"]]["tags"] == ["preference"]
        assert "dev" in stored[e2["id"]]["provisional_tags"]

        actions = [r["action"] for r in cur.read_curation_log("alice", include_dry_run=False)]
        assert "tag_backfill" in actions

        assert cur._backfilled_ids("alice") == {e1["id"], e2["id"]}


async def test_curate_tag_backfill_runs_once_ever(_isolated_dirs, monkeypatch):
    """After a first successful sweep, a second curate() run must not send
    already-backfilled entries through tag_backfill again — the sidecar
    marker persists across runs (not just within a single dry-run preview),
    keeping steady-state cost near zero."""
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = 2
        mgr.save([e1])

        backfill_calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            sys = messages[0]["content"]
            if "REGISTRY" in sys and "new_tags" in sys:
                backfill_calls["n"] += 1
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        await cur.curate(mgr, None, owner="alice", dry_run=False, force=True)
        assert backfill_calls["n"] == 1

        await cur.curate(mgr, None, owner="alice", dry_run=False, force=True)
        assert backfill_calls["n"] == 1  # no repeat call on the second sweep
