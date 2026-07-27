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
    monkeypatch.setattr(cur, "CURATION_FAILURES_DIR", str(tmp_path / "failures"))
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
        assert first["had_failures"] is False  # legitimate "[]" replies aren't failures
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
        # The refused batch left its entries un-deduped, same as a real
        # failure would — must block the tidy fingerprint save so a future
        # run retries it instead of accepting it as permanently "clean".
        assert result["had_failures"] is True


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


# ── failure-aware fingerprint gating + force run ──

async def test_curate_non_json_reply_blocks_fingerprint_and_next_run_retries(_isolated_dirs, monkeypatch):
    """Regression for the "curator says already clean, but nothing was ever
    tagged" bug: if a batch's LLM reply never parses as JSON, that's a real
    failure (distinct from a legitimate "[]" reply) and must stop `curate()`
    from saving the tidy fingerprint — otherwise every future run
    (including the nightly loop) short-circuits on `already_tidy` forever
    without ever retrying the un-triaged entries."""
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = None
        e1["provisional_tags"] = ["drinks"]
        mgr.save([e1])

        calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            calls["n"] += 1
            if "promote_tags" in messages[0]["content"]:
                # Simulate the local model echoing its own instructions
                # instead of returning JSON (seen repeatedly in production).
                return "*   Input: A JSON array of memory entries...\n    *   Task:"
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        first = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert first["status"] == "done"
        assert first["had_failures"] is True
        assert first["failure_count"] >= 1
        # The entry was never actually triaged.
        stored = mgr.load(owner="alice")
        assert stored[0]["generality"] is None

        calls_after_first = calls["n"]
        second = await cur.curate(mgr, None, owner="alice", dry_run=False)
        # Must NOT short-circuit — the fingerprint was never saved.
        assert second.get("already_tidy") is not True
        assert calls["n"] > calls_after_first  # the triage batch was retried


async def test_curate_force_bypasses_tidy_fingerprint(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        # Already-triaged and sharing a tag, so every run's dedupe pass
        # clusters both into one real (>=2-entry) batch and calls the LLM,
        # regardless of whether triage has anything left to do — this is
        # what lets us observe whether `force` actually re-ran the passes.
        e1 = mgr.add_entry("User likes tea", owner="alice")
        e1["generality"] = 2
        e1["tags"] = ["drinks"]
        e2 = mgr.add_entry("User likes coffee", owner="alice")
        e2["generality"] = 2
        e2["tags"] = ["drinks"]
        mgr.save([e1, e2])

        calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            calls["n"] += 1
            sys = messages[0]["content"]
            if "duplicate" in sys.lower() or "MERGE" in sys:
                # Keep both entries unchanged (valid JSON, no removal).
                import json as _json
                payload = _json.loads(messages[-1]["content"])
                return _json.dumps(payload)
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        first = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert first["status"] == "done"
        assert first["had_failures"] is False
        calls_after_first = calls["n"]
        assert calls_after_first > 0  # sanity: the dedupe pass really ran

        # Without force: short-circuits, no new calls.
        second = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert second["already_tidy"] is True
        assert calls["n"] == calls_after_first

        # With force=True: bypasses the fingerprint check and actually runs.
        third = await cur.curate(mgr, None, owner="alice", dry_run=False, force=True)
        assert third["already_tidy"] is False
        assert calls["n"] > calls_after_first


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


# ── resilience: in-run retry, bisection, quarantine ──

async def test_run_pass_resilient_retries_before_giving_up(_isolated_dirs):
    calls = {"n": 0}

    async def call_batch(batch, local_failures):
        calls["n"] += 1
        if calls["n"] < 2:
            local_failures.append("triage:non_json_reply")
            return []
        return ["ok"]

    result = await cur._run_pass_resilient(
        [{"id": "a"}], call_batch, lambda a, b: a + b, owner="alice", pass_name="triage",
        dry_run=False, master_failures=None,
    )
    assert result == ["ok"]
    assert calls["n"] == 2  # succeeded on retry, no need to exhaust every attempt


async def test_run_pass_resilient_bisects_and_isolates_poison_entry(_isolated_dirs):
    poison_id = "poison"
    batch = [{"id": str(i)} for i in range(4)] + [{"id": poison_id}]

    async def call_batch(sub_batch, local_failures):
        if any(e["id"] == poison_id for e in sub_batch):
            local_failures.append("triage:non_json_reply")
            return []
        return [e["id"] for e in sub_batch]

    failures = []
    result = await cur._run_pass_resilient(
        batch, call_batch, lambda a, b: a + b, owner="alice", pass_name="triage",
        dry_run=False, master_failures=failures,
    )
    # Every non-poison id made it through; only the poison leaf is unresolved.
    assert set(result) == {"0", "1", "2", "3"}
    assert failures == ["triage:non_json_reply"]
    # One run only bumps the failure count once (below the default threshold
    # of 3) — it doesn't quarantine yet, but the leaf failure was tracked.
    state = cur._load_failure_state("alice")
    assert state[poison_id]["triage"]["count"] == 1
    assert cur.list_quarantined("alice") == []


async def test_run_pass_resilient_unsafe_removal_refused_never_retried_or_bisected(_isolated_dirs):
    calls = {"n": 0}

    async def call_batch(batch, local_failures):
        calls["n"] += 1
        local_failures.append("dedupe:unsafe_removal_refused")
        return list(batch), []

    batch = [{"id": str(i)} for i in range(10)]
    failures = []
    result = await cur._run_pass_resilient(
        batch, call_batch, lambda a, b: (a[0] + b[0], a[1] + b[1]), owner="alice", pass_name="dedupe",
        dry_run=False, master_failures=failures,
    )
    assert calls["n"] == 1  # never retried
    assert result[0] == batch  # batch unchanged, exactly like before this feature
    assert failures == ["dedupe:unsafe_removal_refused"]
    assert cur.list_quarantined("alice") == []  # this reason never quarantines


async def test_run_pass_resilient_llm_error_retried_but_never_bisected(_isolated_dirs):
    calls = {"n": 0}

    async def call_batch(batch, local_failures):
        calls["n"] += 1
        local_failures.append("triage:llm_error:boom")
        return []

    batch = [{"id": str(i)} for i in range(5)]
    failures = []
    result = await cur._run_pass_resilient(
        batch, call_batch, lambda a, b: a + b, owner="alice", pass_name="triage",
        dry_run=False, master_failures=failures,
    )
    assert calls["n"] == 3  # 1 + _BATCH_RETRY_ATTEMPTS, same 5-entry batch every time
    assert failures == ["triage:llm_error"]
    assert cur.list_quarantined("alice") == []  # llm_error is retryable but never quarantine-eligible


async def test_curate_triage_bisection_isolates_poison_entry_in_shared_batch(_isolated_dirs, monkeypatch):
    """A batch clustered by a shared tag (cluster_batches puts them together)
    where one entry's content always breaks the model's JSON output: the
    fine entries must still get triaged this run instead of the whole batch
    being left untouched."""
    import json

    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        fine_entries = [mgr.add_entry(f"fact {i}", owner="alice") for i in range(4)]
        for e in fine_entries:
            e["generality"] = None
            e["tags"] = ["shared"]
        poison = mgr.add_entry("poison fact", owner="alice")
        poison["generality"] = None
        poison["tags"] = ["shared"]
        mgr.save(fine_entries + [poison])

        async def fake_llm(role, messages, owner=None, **kwargs):
            sys = messages[0]["content"]
            if "promote_tags" in sys:  # triage
                payload = json.loads(messages[-1]["content"])
                if any(item["id"] == poison["id"] for item in payload):
                    return "the model rambles instead of returning JSON"
                return json.dumps([{"id": item["id"], "generality": 2, "promote_tags": []} for item in payload])
            if "duplicate" in sys.lower() or "MERGE" in sys:  # dedupe: keep everything unchanged
                payload = json.loads(messages[-1]["content"])
                return json.dumps(payload)
            return "[]"  # tag-merge / context-facts summary

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        result = await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert result["had_failures"] is True

        stored = {e["id"]: e for e in mgr.load(owner="alice")}
        assert len(stored) == 5  # nothing lost — dedupe kept everything unchanged
        for e in fine_entries:
            assert stored[e["id"]]["generality"] == 2
        assert stored[poison["id"]]["generality"] is None


async def test_quarantine_after_repeated_runs_excludes_entry_and_can_be_cleared(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("weird entry", owner="alice")
        e1["generality"] = None
        mgr.save([e1])

        calls = {"n": 0}

        async def fake_llm(role, messages, owner=None, **kwargs):
            calls["n"] += 1
            if "promote_tags" in messages[0]["content"]:
                return "not json at all"
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        # Default memory_curator_quarantine_after is 3 separate runs.
        for _ in range(3):
            result = await cur.curate(mgr, None, owner="alice", dry_run=False)
            assert result["had_failures"] is True

        quarantined = cur.list_quarantined("alice")
        assert len(quarantined) == 1
        assert quarantined[0]["id"] == e1["id"]
        assert quarantined[0]["pass"] == "triage"

        calls_before_4th = calls["n"]
        result4 = await cur.curate(mgr, None, owner="alice", dry_run=False)
        # Quarantined entry is excluded from triage batching entirely now —
        # no new LLM calls, and nothing left to fail.
        assert calls["n"] == calls_before_4th
        assert result4["had_failures"] is False

        # Manual clear -> the entry is fed to the curator again.
        assert cur.clear_quarantine(mgr, "alice", e1["id"], "triage") is True
        assert cur.list_quarantined("alice") == []
        await cur.curate(mgr, None, owner="alice", dry_run=False)
        assert calls["n"] > calls_before_4th


async def test_quarantine_dry_run_previews_without_persisting(_isolated_dirs, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        e1 = mgr.add_entry("weird entry", owner="alice")
        e1["generality"] = None
        mgr.save([e1])

        async def fake_llm(role, messages, owner=None, **kwargs):
            if "promote_tags" in messages[0]["content"]:
                return "not json"
            return "[]"

        monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

        for _ in range(5):  # well past the real threshold, but always dry_run
            result = await cur.curate(mgr, None, owner="alice", dry_run=True)
            assert result["status"] == "dry_run"

        assert cur.list_quarantined("alice") == []  # dry runs never persist quarantine state
        log = cur.read_curation_log("alice", include_dry_run=True)
        assert any(rec["action"] == "quarantine" and rec["dry_run"] for rec in log)
