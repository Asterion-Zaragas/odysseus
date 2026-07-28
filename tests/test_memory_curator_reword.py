"""reword: the curator pass that lightly improves entry wording (grammar,
redundancy, clarity) — the only curator pass that rewrites `text` itself,
so it is off by default and guarded primarily by deterministic code (a
similarity floor + a numbers/dates-preserved check), not a second LLM call.
See .AGENT_CONTEXT/plans/2026-07-28-curator-reword-pass.md.

Covers src/memory_vector.py's `text_similarity`, plus
services/memory/memory_curator.py's `_numbers_preserved` / `_needs_reword` /
`_reword_batch` / `_run_reword_pass` / `_reworded_ids` / `_mark_reworded` /
`undo_reword`, and the pass wired into the full `curate()` pipeline.
"""

import json
import tempfile

import pytest

from services.memory import memory_context as mc
from services.memory import memory_curator as cur
from src.memory import MemoryManager
from src.memory_vector import MemoryVectorStore


@pytest.fixture()
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setattr(cur, "CURATION_FAILURES_DIR", str(tmp_path / "failures"))
    monkeypatch.setattr(cur, "TAG_BACKFILL_STATE_DIR", str(tmp_path / "tag_backfill_state"))
    monkeypatch.setattr(cur, "REWORD_STATE_DIR", str(tmp_path / "reword_state"))
    monkeypatch.setattr(mc, "CONTEXT_DIR", str(tmp_path / "context"))
    return tmp_path


def _enable_reword(monkeypatch, judge=False):
    settings = {"memory_curator_reword_enabled": True, "memory_curator_reword_judge_enabled": judge}
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: settings.get(key, default))


class _FakeVectorStore:
    """Minimal double: fixed similarity score, tracks calls and rebuilds."""

    healthy = True

    def __init__(self, similarity=0.95):
        self._similarity = similarity
        self.similarity_calls = []
        self.rebuild_calls = 0

    def text_similarity(self, a, b):
        self.similarity_calls.append((a, b))
        return self._similarity

    def rebuild(self, memories):
        self.rebuild_calls += 1


_LONG_TEXT = "The user's rent is $1450/mo and it is due on the 1st of every month."
_LONG_TEXT_REWORDED = "The user pays $1450/mo in rent, due on the 1st of each month."


# ── MemoryVectorStore.text_similarity ──

def test_text_similarity_high_for_near_identical(monkeypatch):
    store = MemoryVectorStore.__new__(MemoryVectorStore)
    store._healthy = True
    monkeypatch.setattr(store, "_embed", lambda texts: [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]])
    sim = store.text_similarity("a", "b")
    assert sim is not None and sim > 0.99


def test_text_similarity_low_for_unrelated(monkeypatch):
    store = MemoryVectorStore.__new__(MemoryVectorStore)
    store._healthy = True
    monkeypatch.setattr(store, "_embed", lambda texts: [[1.0, 0.0], [0.0, 1.0]])
    assert store.text_similarity("a", "b") == pytest.approx(0.0)


def test_text_similarity_none_when_unhealthy():
    store = MemoryVectorStore.__new__(MemoryVectorStore)
    store._healthy = False
    assert store.text_similarity("a", "b") is None


def test_text_similarity_none_on_embed_failure(monkeypatch):
    store = MemoryVectorStore.__new__(MemoryVectorStore)
    store._healthy = True

    def boom(texts):
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(store, "_embed", boom)
    assert store.text_similarity("a", "b") is None


# ── _numbers_preserved ──

def test_numbers_preserved_true_when_kept():
    assert cur._numbers_preserved("rent is $1450/mo", "rent costs $1450 per month") is True


def test_numbers_preserved_false_when_number_changed():
    assert cur._numbers_preserved("rent is $1450/mo", "rent is $1400/mo") is False


def test_numbers_preserved_false_when_number_dropped():
    assert cur._numbers_preserved("born on 1990-05-02", "born a while ago") is False


# ── _needs_reword gating ──

def test_needs_reword_gate_length_tier_and_done_ids():
    long_entry = {"id": "a", "text": _LONG_TEXT, "tier": 1}
    assert cur._needs_reword(long_entry, set()) is True
    assert cur._needs_reword(long_entry, {"a"}) is False

    short_entry = {"id": "b", "text": "short", "tier": 1}
    assert cur._needs_reword(short_entry, set()) is False

    archive_entry = {"id": "c", "text": _LONG_TEXT, "tier": cur.TIER_ARCHIVE}
    assert cur._needs_reword(archive_entry, set()) is False


# ── _reword_batch: guards ──

async def test_reword_batch_applies_when_guards_pass(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, False, False, False)

    assert entry["text"] == _LONG_TEXT_REWORDED
    assert scanned == {"m1"}
    assert len(actions) == 1
    action, before, after = actions[0]
    assert action == "reword"
    assert before["text"] == _LONG_TEXT
    assert after["text"] == _LONG_TEXT_REWORDED


async def test_reword_batch_refuses_below_similarity_floor(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.5)  # below _REWORD_MIN_SIMILARITY

    actions, scanned = await cur._reword_batch([entry], "alice", vector, False, False, False)

    assert entry["text"] == _LONG_TEXT  # unchanged
    assert scanned == set()  # not marked done — reconsidered next run
    assert len(actions) == 1
    action, before, after = actions[0]
    assert action == "reword_refused"
    assert before["id"] == "m1"
    assert "similarity" in before["reason"]
    assert after is None


async def test_reword_batch_refuses_when_number_dropped(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        mangled = _LONG_TEXT_REWORDED.replace("1450", "1400")
        return json.dumps([{"id": "m1", "text": mangled}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.99)  # similarity alone would pass

    actions, scanned = await cur._reword_batch([entry], "alice", vector, False, False, False)

    assert entry["text"] == _LONG_TEXT
    assert scanned == set()
    action, before, after = actions[0]
    assert action == "reword_refused"
    assert "number" in before["reason"]


async def test_reword_batch_refuses_when_vector_store_unavailable(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    actions, scanned = await cur._reword_batch([entry], "alice", None, False, False, False)

    assert entry["text"] == _LONG_TEXT
    assert scanned == set()
    action, before, after = actions[0]
    assert action == "reword_refused"
    assert "vector store" in before["reason"]


async def test_reword_batch_no_proposal_still_scanned(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return "[]"  # nothing to improve

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, False, False, False)

    assert entry["text"] == _LONG_TEXT
    assert scanned == {"m1"}  # evaluated, nothing to add — still marked done
    assert actions == []


async def test_reword_batch_non_json_reply_scans_nothing(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        return "not json at all"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    failures = []
    actions, scanned = await cur._reword_batch([entry], "alice", _FakeVectorStore(), False, False, False, failures)

    assert actions == []
    assert scanned == set()
    assert failures == ["reword:non_json_reply"]


# ── _reword_batch: optional judge ──

async def test_reword_batch_judge_disabled_never_invoked(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}
    judge_calls = {"n": 0}

    async def fake_llm(role, messages, owner=None, **kwargs):
        if "verifying" in messages[0]["content"]:
            judge_calls["n"] += 1
            return json.dumps([{"id": "m1", "preserved": True}])
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, False, False, False)

    assert judge_calls["n"] == 0
    assert entry["text"] == _LONG_TEXT_REWORDED
    assert scanned == {"m1"}


async def test_reword_batch_judge_enabled_false_answer_refuses(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        if "verifying" in messages[0]["content"]:
            return json.dumps([{"id": "m1", "preserved": False}])
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, True, False, False)

    assert entry["text"] == _LONG_TEXT
    assert scanned == set()
    action, before, after = actions[0]
    assert action == "reword_refused"
    assert "judge" in before["reason"]


async def test_reword_batch_judge_parse_failure_fails_closed(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        if "verifying" in messages[0]["content"]:
            return "the judge rambles instead of JSON"
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, True, False, False)

    assert entry["text"] == _LONG_TEXT  # refused, not applied
    assert scanned == set()
    assert actions[0][0] == "reword_refused"


async def test_reword_batch_judge_exception_fails_closed(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT}

    async def fake_llm(role, messages, owner=None, **kwargs):
        if "verifying" in messages[0]["content"]:
            raise RuntimeError("endpoint down")
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.97)

    actions, scanned = await cur._reword_batch([entry], "alice", vector, True, False, False)

    assert entry["text"] == _LONG_TEXT
    assert actions[0][0] == "reword_refused"


# ── _run_reword_pass: two-gate opt-in ──

async def test_run_reword_pass_disabled_by_default_makes_no_llm_call(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT, "tier": 1}
    calls = {"n": 0}

    async def fake_llm(role, messages, owner=None, **kwargs):
        calls["n"] += 1
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    # No _enable_reword() call: memory_curator_reword_enabled defaults False.
    result = await cur._run_reword_pass([entry], "alice", 25, False, _FakeVectorStore())

    assert calls["n"] == 0
    assert result[0]["text"] == _LONG_TEXT
    assert cur._reworded_ids("alice") == set()


async def test_run_reword_pass_excludes_quarantined_entries(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT, "tier": 1}
    for _ in range(cur._quarantine_threshold()):
        cur._handle_quarantine_candidate("alice", "reword", entry, "reword_refused", dry_run=False)
    assert any(q["id"] == "m1" and q["pass"] == "reword" for q in cur.list_quarantined("alice"))

    _enable_reword(monkeypatch)
    calls = {"n": 0}

    async def fake_llm(role, messages, owner=None, **kwargs):
        calls["n"] += 1
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    await cur._run_reword_pass([entry], "alice", 25, False, _FakeVectorStore())
    assert calls["n"] == 0


async def test_run_reword_pass_dry_run_never_persists_sidecar(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT, "tier": 1}
    _enable_reword(monkeypatch)

    async def fake_llm(role, messages, owner=None, **kwargs):
        return "[]"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    await cur._run_reword_pass([entry], "alice", 25, True, _FakeVectorStore())
    assert cur._reworded_ids("alice") == set()

    await cur._run_reword_pass([entry], "alice", 25, False, _FakeVectorStore())
    assert cur._reworded_ids("alice") == {"m1"}


# ── quarantine after repeated refusals ──

async def test_reword_quarantines_after_repeated_refusals_and_clears(_isolated_dirs, monkeypatch):
    entry = {"id": "m1", "text": _LONG_TEXT, "tier": 1}
    _enable_reword(monkeypatch)

    async def fake_llm(role, messages, owner=None, **kwargs):
        return json.dumps([{"id": "m1", "text": _LONG_TEXT_REWORDED}])

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    vector = _FakeVectorStore(similarity=0.5)  # always refused

    threshold = cur._quarantine_threshold()
    for _ in range(threshold):
        await cur._run_reword_pass([entry], "alice", 25, False, vector)

    quarantined = cur.list_quarantined("alice")
    assert len(quarantined) == 1
    assert quarantined[0]["id"] == "m1"
    assert quarantined[0]["pass"] == "reword"

    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        mgr.save([entry])
        assert cur.clear_quarantine(mgr, "alice", "m1", "reword") is True
    assert cur.list_quarantined("alice") == []


# ── undo_reword ──

async def test_undo_reword_restores_pre_reword_text(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry(_LONG_TEXT, owner="alice")
        mgr.save([e1])

        cur._log_action(
            "alice", "reword",
            before={"id": e1["id"], "text": _LONG_TEXT},
            after={"id": e1["id"], "text": _LONG_TEXT_REWORDED},
        )
        entries = mgr.load_all()
        entries[0]["text"] = _LONG_TEXT_REWORDED
        mgr.save(entries)

        assert cur.undo_reword(mgr, "alice", e1["id"]) is True
        restored = {e["id"]: e for e in mgr.load(owner="alice")}
        assert restored[e1["id"]]["text"] == _LONG_TEXT

        # Already-undone / never-changed is a safe no-op.
        assert cur.undo_reword(mgr, "alice", e1["id"]) is True


async def test_undo_reword_no_snapshot_returns_false(_isolated_dirs):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        assert cur.undo_reword(mgr, "alice", "nonexistent") is False


# ── full curate() pipeline integration ──

async def test_curate_full_pipeline_rewords_entry_when_enabled(_isolated_dirs, monkeypatch):
    _enable_reword(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry(_LONG_TEXT, owner="alice")
        e1["generality"] = 2
        mgr.save([e1])

        async def fake_llm(role, messages, owner=None, **kwargs):
            sys = messages[0]["content"]
            if "lightly improve the wording" in sys:
                return json.dumps([{"id": e1["id"], "text": _LONG_TEXT_REWORDED}])
            if "duplicate" in sys.lower() or "MERGE" in sys:
                return json.dumps({"merges": [], "remove": []})
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
        vector = _FakeVectorStore(similarity=0.97)

        result = await cur.curate(mgr, vector, owner="alice", dry_run=False)
        assert result["status"] == "done"
        assert result["had_failures"] is False

        stored = {e["id"]: e for e in mgr.load(owner="alice")}
        assert stored[e1["id"]]["text"] == _LONG_TEXT_REWORDED
        assert vector.rebuild_calls == 1

        actions = [r["action"] for r in cur.read_curation_log("alice", include_dry_run=False)]
        assert "reword" in actions
        assert cur._reworded_ids("alice") == {e1["id"]}


async def test_curate_reword_disabled_by_default_makes_zero_reword_calls(_isolated_dirs, monkeypatch):
    """memory_curator_dry_run soak mode alone (without also flipping
    memory_curator_reword_enabled) must result in zero reword proposals
    being even attempted, not just zero applied."""
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry(_LONG_TEXT, owner="alice")
        e1["generality"] = 2
        mgr.save([e1])

        reword_calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            sys = messages[0]["content"]
            if "lightly improve the wording" in sys:
                reword_calls["n"] += 1
                return json.dumps([{"id": e1["id"], "text": _LONG_TEXT_REWORDED}])
            if "duplicate" in sys.lower() or "MERGE" in sys:
                return json.dumps({"merges": [], "remove": []})
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, _FakeVectorStore(), owner="alice", dry_run=False)
        assert result["status"] == "done"
        assert reword_calls["n"] == 0
        stored = {e["id"]: e for e in mgr.load(owner="alice")}
        assert stored[e1["id"]]["text"] == _LONG_TEXT
