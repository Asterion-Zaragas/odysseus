"""Phase 4 of the memory upgrade: the nightly curator agent.

Covers services/memory/memory_curator.py: batch clustering, registry cap
enforcement, tier rescoring/expiry (pure code), the changelog + undo
round-trip, checkpoint resumability, and the full `curate()` pipeline
end-to-end against mocked "memory smart" LLM calls.
"""

import tempfile
import time

import pytest

from services.memory import memory_curator as cur
from src.memory import MemoryManager


# ── cluster_batches ──

def _entry(id_, tags=None, ts=0):
    return {"id": id_, "text": f"text-{id_}", "tags": tags or [], "timestamp": ts}


def test_cluster_batches_no_entry_in_two_batches_and_size_cap_honored():
    entries = [_entry(str(i), tags=["work"] if i % 2 == 0 else ["hobby"]) for i in range(10)]
    batches = cur.cluster_batches(entries, batch_size=3)

    seen = []
    for b in batches:
        assert len(b) <= 3
        seen.extend(e["id"] for e in b)
    assert sorted(seen) == sorted(e["id"] for e in entries)
    assert len(seen) == len(set(seen))


def test_cluster_batches_groups_by_dominant_tag_first():
    entries = [_entry("a", tags=["work"]), _entry("b", tags=["work"]), _entry("c", tags=["hobby"])]
    batches = cur.cluster_batches(entries, batch_size=10)
    # "work" (count 2) is more common than "hobby" (count 1) -> its cluster batch comes first.
    assert batches[0][0]["tags"] == ["work"] or batches[0][1]["tags"] == ["work"]
    assert {e["id"] for e in batches[0]} == {"a", "b"}


def test_cluster_batches_untagged_leftovers_batch_by_recency():
    entries = [_entry("new", ts=200), _entry("old", ts=100)]
    batches = cur.cluster_batches(entries, batch_size=10)
    assert [e["id"] for e in batches[0]] == ["old", "new"]


# ── owner bucketing ──

def test_entries_for_owner_none_means_ownerless_only():
    all_entries = [{"id": "1", "owner": "alice"}, {"id": "2"}, {"id": "3", "owner": None}]
    assert {e["id"] for e in cur._entries_for_owner(all_entries, None)} == {"2", "3"}
    assert {e["id"] for e in cur._entries_for_owner(all_entries, "alice")} == {"1"}


def test_list_owners_includes_legacy_bucket_once():
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        mgr.save([
            {**mgr.add_entry("a", owner="alice")},
            {**mgr.add_entry("b", owner="bob")},
            {**mgr.add_entry("c")},
        ])
        owners = cur.list_owners(mgr)
        assert owners == ["alice", "bob", None]


# ── registry cap enforcement ──

def test_build_registry_caps_and_keeps_protected_regardless_of_rank():
    entries = [_entry(str(i), tags=[f"tag{i}"]) for i in range(5)]
    entries.append(_entry("p", tags=["contact"]))  # lowest count (1), but protected
    registry = cur.build_registry(entries, cap=3, protected={"contact"})

    names = {t["name"] for t in registry}
    assert "contact" in names
    assert len(registry) == 3
    assert all(t["protected"] for t in registry if t["name"] == "contact")


def test_build_registry_sorted_by_count_desc():
    entries = [_entry("1", tags=["a"]), _entry("2", tags=["a"]), _entry("3", tags=["b"])]
    registry = cur.build_registry(entries, cap=10, protected=set())
    assert [t["name"] for t in registry] == ["a", "b"]
    assert registry[0]["count"] == 2


# ── rescore pass (pure code) ──

def test_rescore_pass_demotes_low_score_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    entry = {"id": "x", "tier": 0, "generality": 0, "uses": 0, "last_used_at": 0, "timestamp": 0}
    out = cur._run_rescore_pass([entry], owner="alice", dry_run=False)
    assert out[0]["tier"] > 0  # demoted at least one step
    log = cur.read_curation_log("alice")
    assert any(rec["action"] == "demote" for rec in log)


def test_rescore_pass_dry_run_still_logs_but_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    entry = {"id": "x", "tier": 0, "generality": 0, "uses": 0, "last_used_at": 0, "timestamp": 0}
    cur._run_rescore_pass([entry], owner="alice", dry_run=True)
    log = cur.read_curation_log("alice")
    assert log and log[-1]["dry_run"] is True


# ── expiry pass (pure code) ──

def test_expire_pass_removes_stale_archive_only():
    now = time.time()
    stale = {"id": "stale", "tier": cur.TIER_ARCHIVE, "pinned": False, "tags": [], "last_used_at": now - 200 * 86400}
    fresh = {"id": "fresh", "tier": cur.TIER_ARCHIVE, "pinned": False, "tags": [], "last_used_at": now}
    survivors = cur._run_expire_pass([stale, fresh], owner=None, protected=set(), expiry_days=90, dry_run=False)
    assert {e["id"] for e in survivors} == {"fresh"}


def test_expire_pass_pinned_and_protected_are_immune():
    now = time.time()
    pinned = {"id": "pinned", "tier": cur.TIER_ARCHIVE, "pinned": True, "tags": [], "last_used_at": now - 200 * 86400}
    protected_tag = {"id": "prot", "tier": cur.TIER_ARCHIVE, "pinned": False, "tags": ["contact"], "last_used_at": now - 200 * 86400}
    survivors = cur._run_expire_pass(
        [pinned, protected_tag], owner=None, protected={"contact"}, expiry_days=90, dry_run=False,
    )
    assert {e["id"] for e in survivors} == {"pinned", "prot"}


def test_expire_pass_only_touches_archive_tier():
    now = time.time()
    core = {"id": "core", "tier": cur.TIER_CORE, "pinned": False, "tags": [], "last_used_at": now - 500 * 86400}
    survivors = cur._run_expire_pass([core], owner=None, protected=set(), expiry_days=90, dry_run=False)
    assert {e["id"] for e in survivors} == {"core"}


# ── changelog + undo ──

def test_log_action_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    cur._log_action("alice", "demote", before={"id": "1", "tier": 1}, after={"id": "1", "tier": 2})
    log = cur.read_curation_log("alice")
    assert len(log) == 1
    assert log[0]["action"] == "demote"


def test_undo_expire_reinserts_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        snapshot = {"id": "gone", "text": "expired fact", "tags": [], "owner": "alice"}
        cur._log_action("alice", "expire", before=snapshot, after=None)

        assert mgr.load(owner="alice") == []
        ok = cur.undo_expire(mgr, None, "alice", "gone")
        assert ok is True
        restored = mgr.load(owner="alice")
        assert len(restored) == 1 and restored[0]["text"] == "expired fact"


def test_undo_expire_no_snapshot_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        assert cur.undo_expire(mgr, None, "alice", "never-existed") is False


# ── checkpoint ──

def test_checkpoint_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_STATE_DIR", str(tmp_path))
    assert cur._load_checkpoint("alice") == {}
    cur._save_checkpoint("alice", "triage")
    assert cur._load_checkpoint("alice")["last_completed_pass"] == "triage"
    cur._clear_checkpoint("alice")
    assert cur._load_checkpoint("alice") == {}


# ── full pipeline (mocked LLM) ──

@pytest.fixture()
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path / "log"))
    return tmp_path


async def test_curate_empty_store_short_circuits(_isolated_dirs):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert result["status"] == "empty"


async def test_curate_full_pipeline_runs_all_passes(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = None
        e1["provisional_tags"] = ["drinks"]
        mgr.save([e1])

        async def fake_llm(role, messages, owner=None, **kwargs):
            content = messages[-1]["content"]
            if "promote_tags" in messages[0]["content"]:
                return f'[{{"id": "{e1["id"]}", "generality": 2, "promote_tags": ["drinks"]}}]'
            if "from" in messages[0]["content"] and "into" in messages[0]["content"]:
                return "[]"
            if "duplicate" in messages[0]["content"].lower() or "MERGE" in messages[0]["content"]:
                return f'[{{"id": "{e1["id"]}", "text": "User likes tea", "tags": ["drinks"]}}]'
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert result["status"] == "done"
        stored = mgr.load(owner="alice")
        assert len(stored) == 1
        assert stored[0]["generality"] == 2
        assert "drinks" in stored[0]["tags"]
        assert stored[0]["provisional_tags"] == []

        from services.memory.memory_context import MemoryContext
        doc = MemoryContext("alice").load()
        assert doc["tag_registry"]  # rebuilt


async def test_curate_threads_interactive_flag_to_smart_calls(_isolated_dirs, monkeypatch):
    """Regression for the manual-curator deadlock (bugs/2026-07-13-curator-
    manual-run-deadlock.md): the /api/memory/audit route runs curate() INLINE
    inside its own tracked HTTP request, so it passes interactive=True and every
    memory-smart call must inherit it (interactive=False there waits for the
    request count to hit zero — i.e. waits on the audit request itself, and
    deadlocks). Verify curate() threads its interactive flag through every pass
    to memory_llm_call_async, for both True (manual route) and False (nightly)."""
    import json

    for flag in (True, False):
        with tempfile.TemporaryDirectory() as d:
            mgr = MemoryManager(d)
            e1 = mgr.add_entry("User likes tea", owner="alice")
            e1["generality"] = None
            e1["provisional_tags"] = ["morning"]
            e1["tags"] = ["drinks", "tea"]
            e2 = mgr.add_entry("User enjoys coffee", owner="alice")
            e2["generality"] = None
            e2["tags"] = ["drinks", "coffee"]
            e2["tier"] = 0  # core → triggers the context-doc summary call too
            mgr.save([e1, e2])

            seen = []

            async def fake_llm(role, messages, owner=None, interactive=False, **kwargs):
                seen.append(interactive)
                sys = messages[0]["content"]
                if "promote_tags" in sys:            # triage
                    return "[]"
                if '"from"' in sys and '"into"' in sys:  # tag-merge
                    return "[]"
                if "duplicate" in sys.lower() or "MERGE" in sys:  # dedupe: keep both
                    return json.dumps(
                        [{"id": e["id"], "text": e["text"], "tags": e.get("tags", [])} for e in (e1, e2)]
                    )
                return "[]"                          # context-facts summary etc.

            monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

            # dry_run=True runs the full real pipeline (all LLM calls) without
            # persisting — exactly the "Preview" path, and it skips the
            # fingerprint short-circuit so the passes always execute.
            await cur.curate(mgr, None, owner="alice", dry_run=True, interactive=flag)

            assert seen, "expected at least one memory-smart call"
            assert all(v is flag for v in seen), f"expected every call interactive={flag}, got {seen}"


async def test_curate_second_run_short_circuits_on_fingerprint(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = 2
        mgr.save([e1])

        calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            calls["n"] += 1
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        first = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert first["status"] == "done"
        calls_after_first = calls["n"]

        second = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert second["already_tidy"] is True
        assert calls["n"] == calls_after_first  # no new LLM calls


async def test_curate_dry_run_does_not_mutate_store_or_checkpoint(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        mgr.save([e1])

        async def fake_llm(role, messages, owner=None, **kwargs):
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        before = mgr.load(owner="alice")
        result = await cur.curate(mgr, None, owner="alice", dry_run=True)
        after = mgr.load(owner="alice")

        assert result["status"] == "dry_run"
        assert before == after
        assert cur._load_checkpoint("alice") == {}


async def test_curate_dedupe_unsafe_removal_refused(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        entries = [mgr.add_entry(f"fact {i}", owner="alice") for i in range(10)]
        for e in entries:
            e["generality"] = 2  # already triaged, skip straight through triage pass
        mgr.save(entries)

        async def fake_llm(role, messages, owner=None, **kwargs):
            # Dedupe pass returns only 2 of 10 -> over 50% cut, must be refused.
            if role == "smart" and isinstance(messages[-1]["content"], str) and "fact 0" in messages[-1]["content"]:
                import json as _json
                payload = _json.loads(messages[-1]["content"])
                if len(payload) >= 8:
                    return _json.dumps([{"id": payload[0]["id"], "text": payload[0]["text"], "tags": []}])
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        stored = mgr.load(owner="alice")
        assert len(stored) == 10  # unsafe removal refused, nothing lost


async def test_curate_resumes_after_checkpoint_skips_completed_passes(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = 2
        mgr.save([e1])

        cur._save_checkpoint("alice", "tag_normalize")  # pretend triage+dedupe already ran

        calls = []

        async def fake_llm(role, messages, owner=None, **kwargs):
            calls.append(messages[0]["content"][:40])
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert result["status"] == "done"
        # triage/dedupe prompts must not have been re-sent.
        assert not any("triage" in c.lower() or "MERGE" in c for c in calls)


async def test_curate_persists_each_pass_so_a_later_crash_keeps_earlier_work(_isolated_dirs, monkeypatch):
    """A completed pass's mutations must be on disk before its checkpoint is
    written — otherwise a crash in a later pass would advance the checkpoint
    yet lose the in-memory work, and the resumed run (which skips completed
    passes and reloads the un-mutated store) would strand the entry."""
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = None
        e1["provisional_tags"] = ["drinks"]
        mgr.save([e1])

        async def fake_llm(role, messages, owner=None, **kwargs):
            if "promote_tags" in messages[0]["content"]:
                return f'[{{"id": "{e1["id"]}", "generality": 2, "promote_tags": ["drinks"]}}]'
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        # Simulate a crash in the (later) rescore pass.
        def _boom(*a, **k):
            raise RuntimeError("simulated crash mid-run")

        monkeypatch.setattr(cur, "_run_rescore_pass", _boom)

        with pytest.raises(RuntimeError):
            await cur.curate(mgr, None, owner="alice", dry_run=False)

        # Triage ran and was persisted before the crash: generality is set and
        # the provisional tag was promoted, despite curate() never reaching its
        # final save. The checkpoint sits at the last pass that completed, so a
        # resume skips triage/dedupe without losing their work.
        stored = mgr.load(owner="alice")
        assert stored[0]["generality"] == 2
        assert "drinks" in stored[0]["tags"]
        assert stored[0]["provisional_tags"] == []
        assert cur._load_checkpoint("alice")["last_completed_pass"] == "tag_normalize"


# ── triage_new_entries (cheap extractor-trigger path) ──

async def test_triage_new_entries_only_touches_pending(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        done = mgr.add_entry("already triaged", owner="alice")
        done["generality"] = 3
        pending = mgr.add_entry("needs triage", owner="alice")
        pending["generality"] = None
        mgr.save([done, pending])

        async def fake_llm(role, messages, owner=None, **kwargs):
            return f'[{{"id": "{pending["id"]}", "generality": 1, "promote_tags": []}}]'

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.triage_new_entries(mgr, owner="alice")
        assert result["triaged"] == 1

        stored = {e["id"]: e for e in mgr.load(owner="alice")}
        assert stored[done["id"]]["generality"] == 3  # untouched
        assert stored[pending["id"]]["generality"] == 1
