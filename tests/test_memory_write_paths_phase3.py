"""Phase 3 of the memory upgrade: tagger wiring across every memory write path.

Covers routes/memory/memory_routes.py (/add, PUT update), the manage_memory
builtin tool (src/ai_interaction.py), the inline "remember:" chat command
(src/chat_handler.py), and the MCP memory server (mcp_servers/memory_server.py)
— each should call the tagger with interactive=True (they all run inline
inside a tracked request) and apply its result via apply_tags.

Uses a real MemoryManager backed by a tmp_path store (not mocked) so
add_entry/apply_tags exercise their actual dict shapes; only the LLM call
inside tag_memory is faked.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.memory import MemoryManager
from src.request_models import MemoryAddRequest


def _fake_tag_memory(tags=None, provisional_tags=None, generality=1, capture=None):
    async def _tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        if capture is not None:
            capture.append({"text": text, "owner": owner, "interactive": interactive})
        return {"tags": list(tags or []), "provisional_tags": list(provisional_tags or []), "generality": generality}
    return _tag_memory


def _request(user="alice"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
    )


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _allow_memory_management(monkeypatch):
    monkeypatch.setattr("src.auth_helpers.require_privilege", lambda request, privilege: "alice")


# ── routes/memory/memory_routes.py: POST /add ──

@pytest.mark.asyncio
async def test_add_memory_route_applies_tagger_result(tmp_path, monkeypatch):
    import routes.memory.memory_routes as mr

    _allow_memory_management(monkeypatch)
    mm = MemoryManager(str(tmp_path))
    router = mr.setup_memory_routes(mm, MagicMock())
    add_memory = _route(router, "/api/memory/add", "POST")

    calls = []
    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2, capture=calls),
    )

    out = await add_memory(
        request=_request("alice"),
        memory_data=MemoryAddRequest(text="Works at Acme", source="user"),
    )
    assert out["ok"] is True

    saved = mm.load(owner="alice")
    assert len(saved) == 1
    assert saved[0]["tags"] == ["work"]
    assert saved[0]["generality"] == 2
    assert saved[0]["tier"] is not None
    assert calls == [{"text": "Works at Acme", "owner": "alice", "interactive": True}]


@pytest.mark.asyncio
async def test_add_memory_route_explicit_tags_override_tagger(tmp_path, monkeypatch):
    import routes.memory.memory_routes as mr

    _allow_memory_management(monkeypatch)
    mm = MemoryManager(str(tmp_path))
    router = mr.setup_memory_routes(mm, MagicMock())
    add_memory = _route(router, "/api/memory/add", "POST")

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["tagger-guess"], generality=1),
    )

    await add_memory(
        request=_request("alice"),
        memory_data=MemoryAddRequest(text="Prefers dark roast", tags=["preference"], source="user"),
    )

    saved = mm.load(owner="alice")
    assert saved[0]["tags"] == ["preference"]


@pytest.mark.asyncio
async def test_add_memory_route_saves_untagged_on_tagger_fallback(tmp_path, monkeypatch):
    """tag_memory's real contract is "never raise, worst case FALLBACK_RESULT"
    (see test_memory_context_tagger_phase3.py) — this checks the route still
    completes the write in that worst case instead of losing the memory."""
    import routes.memory.memory_routes as mr
    from services.memory.memory_tagger import FALLBACK_RESULT

    _allow_memory_management(monkeypatch)
    mm = MemoryManager(str(tmp_path))
    router = mr.setup_memory_routes(mm, MagicMock())
    add_memory = _route(router, "/api/memory/add", "POST")

    async def _fallback(*a, **k):
        return dict(FALLBACK_RESULT)

    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fallback)

    out = await add_memory(
        request=_request("alice"),
        memory_data=MemoryAddRequest(text="Still saved", source="user"),
    )
    assert out["ok"] is True
    saved = mm.load(owner="alice")
    assert len(saved) == 1
    assert saved[0]["text"] == "Still saved"
    assert saved[0]["tags"] == []
    assert saved[0]["generality"] is None


# ── routes/memory/memory_routes.py: PUT /{memory_id} ──

@pytest.mark.asyncio
async def test_update_memory_route_retags_on_plain_text_edit(tmp_path, monkeypatch):
    import routes.memory.memory_routes as mr

    mm = MemoryManager(str(tmp_path))
    entry = mm.add_entry("Old text", tags=["fact"], owner="alice")
    mm.save([entry])

    router = mr.setup_memory_routes(mm, MagicMock())
    update_memory = _route(router, "/api/memory/{memory_id}", "PUT")

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=1),
    )

    out = await update_memory(request=_request("alice"), memory_id=entry["id"], text="New text", tags=None, category=None)
    assert out["ok"] is True

    saved = mm.load(owner="alice")[0]
    assert saved["text"] == "New text"
    # merge semantics: old seeded tag survives alongside the tagger's new one
    assert set(saved["tags"]) == {"fact", "work"}
    assert saved["generality"] == 1


@pytest.mark.asyncio
async def test_update_memory_route_explicit_tags_override(tmp_path, monkeypatch):
    import routes.memory.memory_routes as mr

    mm = MemoryManager(str(tmp_path))
    entry = mm.add_entry("Old text", tags=["fact"], owner="alice")
    mm.save([entry])

    router = mr.setup_memory_routes(mm, MagicMock())
    update_memory = _route(router, "/api/memory/{memory_id}", "PUT")

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["tagger-guess"], generality=1),
    )

    await update_memory(
        request=_request("alice"), memory_id=entry["id"],
        text="New text", tags='["custom"]', category=None,
    )
    saved = mm.load(owner="alice")[0]
    assert saved["tags"] == ["custom"]


@pytest.mark.asyncio
async def test_update_memory_route_unknown_id_raises_404(tmp_path):
    import routes.memory.memory_routes as mr

    mm = MemoryManager(str(tmp_path))
    router = mr.setup_memory_routes(mm, MagicMock())
    update_memory = _route(router, "/api/memory/{memory_id}", "PUT")

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await update_memory(request=_request("alice"), memory_id="missing", text="x", tags=None, category=None)
    assert exc.value.status_code == 404


# ── manage_memory builtin tool (src/ai_interaction.py) ──

@pytest.fixture()
def ai_memory_manager(tmp_path):
    import src.ai_interaction as ai

    mm = MemoryManager(str(tmp_path))
    ai.set_memory_manager(mm, None)
    yield mm
    ai.set_memory_manager(None, None)


@pytest.mark.asyncio
async def test_do_manage_memory_add_calls_tagger(ai_memory_manager, monkeypatch):
    import src.ai_interaction as ai

    calls = []
    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2, capture=calls),
    )

    result = await ai.do_manage_memory("add\nWorks at Acme", owner="alice")
    assert "Memory added" in result["results"]
    saved = ai_memory_manager.load(owner="alice")
    assert saved[0]["tags"] == ["work"]
    assert calls == [{"text": "Works at Acme", "owner": "alice", "interactive": True}]


@pytest.mark.asyncio
async def test_do_manage_memory_edit_retags_from_new_text(ai_memory_manager, monkeypatch):
    # Phase 8 fix: the edit action used to only touch text/timestamp and
    # never re-tag, unlike the memory-page PUT /{id} route - tags/generality
    # could silently drift from an edited memory's actual content.
    import src.ai_interaction as ai

    entry = ai_memory_manager.add_entry("Old text", tags=["fact"], owner="alice")
    ai_memory_manager.save([entry])

    calls = []
    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2, capture=calls),
    )

    result = await ai.do_manage_memory(f"edit\n{entry['id']}\nNew text", owner="alice")
    assert result["results"] == "Memory updated: New text"

    saved = ai_memory_manager.load(owner="alice")[0]
    assert saved["text"] == "New text"
    # merge semantics: old seeded tag survives alongside the tagger's new one
    assert set(saved["tags"]) == {"fact", "work"}
    assert saved["generality"] == 2
    assert calls == [{"text": "New text", "owner": "alice", "interactive": True}]


@pytest.mark.asyncio
async def test_do_manage_memory_add_line3_tags_override(ai_memory_manager, monkeypatch):
    import src.ai_interaction as ai

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["tagger-guess"], generality=1),
    )

    await ai.do_manage_memory("add\nLikes tea\npreference,drink", owner="alice")
    saved = ai_memory_manager.load(owner="alice")
    assert saved[0]["tags"] == ["preference", "drink"]


# ── inline "remember:" chat command (src/chat_handler.py) ──

@pytest.mark.asyncio
async def test_inline_remember_command_tags_and_scopes_owner(tmp_path, monkeypatch):
    import sys
    from src.chat_handler import ChatHandler

    mm = MemoryManager(str(tmp_path))
    session_manager = MagicMock()
    handler = ChatHandler(session_manager, mm, None, None, None, None)

    session = SimpleNamespace(
        owner="alice", id="sess-1",
        add_message=MagicMock(),
    )

    calls = []
    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2, capture=calls),
    )
    # tests/conftest.py stubs src.database with a bare ModuleType (no
    # update_session_last_accessed) for lightweight collection — patch
    # whatever object is currently registered under that name rather than
    # assuming the real module, matching how the rest of the suite treats it.
    monkeypatch.setattr(sys.modules["src.database"], "update_session_last_accessed", lambda sid: None, raising=False)

    response = await handler.handle_memory_command(session, "remember: I work at Acme")
    assert response == "Saved to memory: I work at Acme"

    saved = mm.load(owner="alice")
    assert len(saved) == 1
    assert saved[0]["tags"] == ["work"]
    assert calls == [{"text": "I work at Acme", "owner": "alice", "interactive": True}]
    session_manager.save_sessions.assert_called_once()


@pytest.mark.asyncio
async def test_inline_remember_command_noop_for_non_command(tmp_path, monkeypatch):
    from src.chat_handler import ChatHandler

    mm = MemoryManager(str(tmp_path))
    handler = ChatHandler(MagicMock(), mm, None, None, None, None)
    session = SimpleNamespace(owner="alice", id="sess-1", add_message=MagicMock())

    def fail_if_called(*a, **k):
        raise AssertionError("must not tag a non-memory message")

    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", fail_if_called)

    response = await handler.handle_memory_command(session, "what's the weather?")
    assert response is None
    assert mm.load(owner="alice") == []


# ── MCP memory server (mcp_servers/memory_server.py) ──

@pytest.fixture()
def mcp_memory_manager(tmp_path):
    import mcp_servers.memory_server as ms

    mm = MemoryManager(str(tmp_path))
    ms._memory_manager = mm
    ms._memory_vector = None
    ms._initialized = True
    yield mm, ms
    ms._memory_manager = None
    ms._memory_vector = None
    ms._initialized = False


@pytest.mark.asyncio
async def test_mcp_memory_add_calls_tagger(mcp_memory_manager, monkeypatch):
    mm, ms = mcp_memory_manager
    monkeypatch.delenv("ODYSSEUS_MCP_MEMORY_OWNER", raising=False)
    monkeypatch.delenv("ODYSSEUS_MEMORY_OWNER", raising=False)

    calls = []
    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2, capture=calls),
    )

    result = await ms.call_tool("manage_memory", {"action": "add", "text": "Works at Acme"})
    assert "Memory added" in result[0].text

    saved = mm.load_all()
    assert len(saved) == 1
    assert saved[0]["tags"] == ["work"]
    assert calls == [{"text": "Works at Acme", "owner": None, "interactive": True}]


@pytest.mark.asyncio
async def test_mcp_memory_add_explicit_tags_override(mcp_memory_manager, monkeypatch):
    mm, ms = mcp_memory_manager
    monkeypatch.delenv("ODYSSEUS_MCP_MEMORY_OWNER", raising=False)
    monkeypatch.delenv("ODYSSEUS_MEMORY_OWNER", raising=False)

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["tagger-guess"], generality=1),
    )

    await ms.call_tool("manage_memory", {"action": "add", "text": "Prefers tea", "tags": ["preference"]})
    saved = mm.load_all()
    assert saved[0]["tags"] == ["preference"]


@pytest.mark.asyncio
async def test_mcp_memory_edit_retags_from_new_text(mcp_memory_manager, monkeypatch):
    # Phase 8 fix: mirrors the manage_memory builtin and PUT /{id} route -
    # the MCP edit action used to only touch text/timestamp and never re-tag.
    mm, ms = mcp_memory_manager
    monkeypatch.delenv("ODYSSEUS_MCP_MEMORY_OWNER", raising=False)
    monkeypatch.delenv("ODYSSEUS_MEMORY_OWNER", raising=False)

    entry = mm.add_entry("Old text", tags=["fact"])
    mm.save([entry])

    monkeypatch.setattr(
        "services.memory.memory_tagger.tag_memory",
        _fake_tag_memory(tags=["work"], generality=2),
    )

    result = await ms.call_tool("manage_memory", {"action": "edit", "memory_id": entry["id"][:8], "text": "New text"})
    assert result[0].text == "Memory updated: New text"

    saved = mm.load_all()[0]
    assert saved["text"] == "New text"
    assert set(saved["tags"]) == {"fact", "work"}
    assert saved["generality"] == 2
