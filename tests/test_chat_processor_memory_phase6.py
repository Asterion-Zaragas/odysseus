"""Phase 6 of the memory upgrade: ChatProcessor.build_context_preface's memory
section — core (pinned + tier-0) capped/relevance-gated injection vs.
context-doc toggle, per-turn effort/doc overrides falling back to settings,
and staged retrieval over the non-core "rest" pool.

Core injection is capped via _select_core_memories: identity/contact-tagged
entries are always available, but a pinned or tier-0 entry with no identity
signal must match the current message (via the staged retrieval pipeline,
same as "rest") to be injected — pinned/core no longer means "inject
everything, every turn," which used to bloat every request and leak
unrelated personal context into tasks that didn't need it.

build_context_preface is async as of this phase (it awaits
services.memory.retrieval.retrieve); these tests exercise that directly
rather than through routes/chat_helpers.py.
"""
from types import SimpleNamespace

from src.chat_processor import ChatProcessor


def _entry(id_, text, tier=2, pinned=False, tags=None):
    return {"id": id_, "text": text, "tier": tier, "pinned": pinned, "tags": tags or [], "timestamp": 0, "uses": 0}


class _FakeMemoryManager:
    def __init__(self, entries):
        self._entries = entries
        self.increment_uses_calls = []

    def load(self, owner=None):
        return list(self._entries)

    def increment_uses(self, ids):
        self.increment_uses_calls.append(list(ids))


def _session():
    return SimpleNamespace(endpoint_url="http://local", model="test", headers={})


def _joined(preface):
    return "\n".join(m.get("content") or "" for m in preface)


async def test_identity_core_always_injected_but_irrelevant_pinned_is_capped(monkeypatch):
    """Identity/contact-tagged core memories bypass relevance-gating entirely.

    A pinned or tier-0 entry with no identity signal is no longer injected
    unconditionally: it must clear the same staged-retrieval bar as an
    ordinary "rest" memory. Here the retrieval pipeline is faked to find
    nothing relevant to "hi", so the plain tier-0 fact is dropped while the
    identity-tagged pinned fact still comes through.
    """
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        return {"memories": [], "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([
        _entry("p", "User's name is Alex", tier=2, pinned=True, tags=["identity"]),
        _entry("t0", "Some tier-0 fact unrelated to anything", tier=0, pinned=False),
    ])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    preface, _, _ = await processor.build_context_preface(
        message="hi", session=_session(), use_web=False, use_rag=False, use_skills=False,
        memory_effort="low",
    )

    joined = _joined(preface)
    assert "User's name is Alex" in joined
    assert "Some tier-0 fact unrelated to anything" not in joined
    kinds = {m["text"]: m["type"] for m in processor._last_used_memories}
    assert kinds == {"User's name is Alex": "pinned"}


async def test_context_doc_toggle_suppresses_nonpinned_core_and_caps_irrelevant_pinned(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    monkeypatch.setattr(
        "services.memory.memory_context.MemoryContext.render_markdown",
        lambda self: "# Memory context\n\nMOCKED DOC BODY",
    )

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        return {"memories": [], "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([
        _entry("p", "User's name is Alex", tier=2, pinned=True, tags=["identity"]),
        _entry("t0", "Tier zero identity fact", tier=0, pinned=False),
    ])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    preface, _, _ = await processor.build_context_preface(
        message="hi", session=_session(), use_web=False, use_rag=False, use_skills=False,
        memory_effort="low", use_memory_context_doc=True,
    )

    joined = _joined(preface)
    assert "MOCKED DOC BODY" in joined
    assert "User's name is Alex" in joined
    # The non-pinned tier-0 entry is covered by the doc's core-facts section
    # instead of being injected individually (avoids duplication).
    assert "Tier zero identity fact" not in joined
    kinds = {m["text"]: m["type"] for m in processor._last_used_memories}
    assert kinds == {"User's name is Alex": "pinned"}


async def test_rest_entries_flow_through_staged_retrieve(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)
    captured = {}

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        captured.update(message=message, entries=entries, effort=effort, owner=owner, k=k, interactive=interactive)
        return {"memories": [entries[0]], "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([
        _entry("p", "Pinned fact", tier=2, pinned=True),
        _entry("r", "Recallable situational memory", tier=2, pinned=False),
    ])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    preface, _, _ = await processor.build_context_preface(
        message="what did I say about my situation", session=_session(),
        use_web=False, use_rag=False, use_skills=False,
        memory_effort="high", owner="alice",
    )

    assert captured["effort"] == "high"
    assert captured["owner"] == "alice"
    assert captured["k"] == 3
    assert captured["interactive"] is True
    assert [e["id"] for e in captured["entries"]] == ["r"]  # pinned excluded from the "rest" pool
    joined = _joined(preface)
    assert "Recallable situational memory" in joined
    kinds = {m["text"]: m["type"] for m in processor._last_used_memories}
    assert kinds["Recallable situational memory"] == "recalled"


async def test_preface_recall_k_honors_memory_recall_k_setting(monkeypatch):
    settings = {"memory_recall_k": 2}
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: settings.get(key, default))
    captured = {}

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        captured["k"] = k
        return {"memories": [], "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([_entry("r", "Recallable situational memory", tier=2)])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    await processor.build_context_preface(
        message="q", session=_session(), use_web=False, use_rag=False, use_skills=False,
        memory_effort="low",
    )

    assert captured["k"] == 2


async def test_effort_and_doc_toggle_default_from_settings_when_omitted(monkeypatch):
    settings = {"memory_retrieval_effort": "high", "memory_context_doc_injection": True}
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(
        "services.memory.memory_context.MemoryContext.render_markdown",
        lambda self: "DOC",
    )
    captured = {}

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        captured["effort"] = effort
        return {"memories": [], "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([_entry("r", "Something to search for", tier=2)])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    preface, _, _ = await processor.build_context_preface(
        message="q", session=_session(), use_web=False, use_rag=False, use_skills=False,
    )

    assert captured["effort"] == "high"
    assert "DOC" in _joined(preface)


async def test_increment_uses_covers_core_and_recalled_ids(monkeypatch):
    monkeypatch.setattr("src.settings.get_setting", lambda key, default=None: default)

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        return {"memories": entries, "facets": None, "effort_used": effort}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)

    mm = _FakeMemoryManager([
        _entry("p", "Pinned fact", tier=2, pinned=True),
        _entry("r", "Recallable memory", tier=2, pinned=False),
    ])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    await processor.build_context_preface(
        message="q", session=_session(), use_web=False, use_rag=False, use_skills=False,
        memory_effort="low",
    )

    assert mm.increment_uses_calls
    used_ids = set(mm.increment_uses_calls[0])
    assert used_ids == {"p", "r"}


async def test_use_memory_false_skips_memory_entirely(monkeypatch):
    called = {"hit": False}

    async def fake_retrieve(*a, **kw):
        called["hit"] = True
        return {"memories": [], "facets": None, "effort_used": "low"}

    monkeypatch.setattr("src.chat_processor.memory_retrieve", fake_retrieve)
    mm = _FakeMemoryManager([_entry("p", "Pinned fact", tier=2, pinned=True)])
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    preface, _, _ = await processor.build_context_preface(
        message="q", session=_session(), use_web=False, use_rag=False, use_skills=False,
        use_memory=False,
    )

    assert called["hit"] is False
    assert _joined(preface).count("Pinned fact") == 0
