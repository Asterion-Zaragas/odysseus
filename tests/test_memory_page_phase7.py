"""Phase 7 of the memory upgrade: memory page UI backend support.

Covers the two small backend pieces the Phase 7 frontend (tag chips, tier
badges, curator panel, context-doc viewer) relies on:

- `services/memory/memory_curator.read_curation_log`'s `include_dry_run`
  filter, and `undo_expire` no longer matching a dry-run-only snapshot — the
  Phase 1-6 audit flagged that a dry run logs proposed actions with a
  `dry_run: true` marker that the pre-Phase-7 log reader never filtered, so
  a curator panel built straight off the raw log would show phantom
  merge/expire actions and could offer to "undo" one that was never applied.
- `GET /api/memory/context-doc`, the new route backing the context-doc
  viewer.
"""

import tempfile

import pytest

from services.memory import memory_curator as cur
from services.memory import memory_context as mc
from src.memory import MemoryManager


# ── read_curation_log include_dry_run ──

def test_read_curation_log_default_includes_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    cur._log_action("alice", "demote", before={"id": "1"}, after={"id": "1"}, dry_run=True)
    log = cur.read_curation_log("alice")
    assert len(log) == 1  # existing (Phase 4) callers keep seeing everything


def test_read_curation_log_can_exclude_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    cur._log_action("alice", "demote", before={"id": "1"}, after={"id": "1"}, dry_run=True)
    cur._log_action("alice", "demote", before={"id": "2"}, after={"id": "2"}, dry_run=False)
    log = cur.read_curation_log("alice", include_dry_run=False)
    assert len(log) == 1
    assert log[0]["before"]["id"] == "2"


def test_undo_expire_ignores_dry_run_only_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        snapshot = {"id": "gone", "text": "expired fact", "tags": [], "owner": "alice"}
        cur._log_action("alice", "expire", before=snapshot, after=None, dry_run=True)

        assert cur.undo_expire(mgr, None, "alice", "gone") is False
        assert mgr.load(owner="alice") == []


def test_undo_expire_still_finds_real_expire_alongside_dry_run_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    with tempfile.TemporaryDirectory() as d:
        mgr = MemoryManager(d)
        snapshot = {"id": "gone", "text": "expired fact", "tags": [], "owner": "alice"}
        cur._log_action("alice", "expire", before=snapshot, after=None, dry_run=True)
        cur._log_action("alice", "expire", before=snapshot, after=None, dry_run=False)

        assert cur.undo_expire(mgr, None, "alice", "gone") is True
        restored = mgr.load(owner="alice")
        assert len(restored) == 1 and restored[0]["text"] == "expired fact"


# ── GET /api/memory/context-doc ──

@pytest.fixture()
def context_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "CONTEXT_DIR", str(tmp_path))
    return tmp_path


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request_for(user):
    from types import SimpleNamespace
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
    )


def test_context_doc_route_cold_start_is_empty(context_dir, monkeypatch, tmp_path):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    mem = MagicMock()
    router = mr.setup_memory_routes(mem, MagicMock())
    handler = _route(router, "/api/memory/context-doc", "GET")

    out = handler(request=_request_for("alice"))
    assert out["core_facts"] == []
    assert out["tag_registry"] == []
    assert "Memory context" in out["markdown"]


def test_context_doc_route_reflects_saved_context(context_dir, monkeypatch):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    ctx = mc.MemoryContext("alice")
    ctx.save({**mc._empty_context(), "core_facts": ["Lives in Berlin"],
              "tag_registry": [{"name": "identity", "count": 3, "protected": True}]})

    mem = MagicMock()
    router = mr.setup_memory_routes(mem, MagicMock())
    handler = _route(router, "/api/memory/context-doc", "GET")

    out = handler(request=_request_for("alice"))
    assert out["core_facts"] == ["Lives in Berlin"]
    assert out["tag_registry"][0]["name"] == "identity"
    assert "Lives in Berlin" in out["markdown"]


def test_context_doc_route_is_owner_scoped(context_dir, monkeypatch):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    mc.MemoryContext("alice").save({**mc._empty_context(), "core_facts": ["alice-only fact"]})

    monkeypatch.setattr(mr, "get_current_user", lambda request: "bob", raising=False)
    mem = MagicMock()
    router = mr.setup_memory_routes(mem, MagicMock())
    handler = _route(router, "/api/memory/context-doc", "GET")

    out = handler(request=_request_for("bob"))
    assert out["core_facts"] == []


# ── GET /api/memory/curation-log route excludes dry-run ──

def test_curation_log_route_excludes_dry_run(tmp_path, monkeypatch):
    import routes.memory_routes as mr
    from unittest.mock import MagicMock

    monkeypatch.setattr(cur, "CURATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    cur._log_action("alice", "merge", before={"id": "1"}, after=None, dry_run=True)
    cur._log_action("alice", "retag", before={"id": "2"}, after={"id": "2"}, dry_run=False)

    mem = MagicMock()
    router = mr.setup_memory_routes(mem, MagicMock())
    handler = _route(router, "/api/memory/curation-log", "GET")

    out = handler(request=_request_for("alice"), limit=100)
    assert len(out["log"]) == 1
    assert out["log"][0]["action"] == "retag"
