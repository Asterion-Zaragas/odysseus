"""Phase 3/5 of the memory upgrade: the extractor's store loop calls the tagger.

Regression coverage for wiring tag_memory()/apply_tags() into
services/memory/memory_extractor.py's extract_and_store — every auto-
extracted fact should land with tagger-assigned tags/generality/tier.

Since Phase 5 the extractor runs on the memory-smart role
(src.task_endpoint.memory_llm_call_async) instead of calling the raw LLM
directly, and its own identity auto-pin is gone — a high-generality fact
now relies purely on the tagger's generality + tier_scoring to land in a
durable tier, exactly like every other fact.
"""

import asyncio
import tempfile

import pytest

import src.event_bus
from src.memory import MemoryManager
from services.memory import memory_extractor
from services.memory.memory_extractor import extract_and_store


@pytest.fixture(autouse=True)
def _isolated_audit_counter(monkeypatch):
    # `_extractions_since_audit` is a process-wide module global that
    # accumulates across every test file exercising extract_and_store in
    # this session. Without resetting it here, these tests' extra calls can
    # tip some *other* test's small in-memory store over AUDIT_INTERVAL and
    # trigger a real (fake-LLM-driven) audit it never expected — pin it to 0
    # so each test's own single fact never crosses the threshold, and so
    # these tests don't leave the shared counter polluted for others either
    # (monkeypatch restores the pre-test value on teardown).
    monkeypatch.setattr(memory_extractor, "_extractions_since_audit", 0)


class _FakeSession:
    owner = "alice"
    session_id = "sess-1"

    def get_context_messages(self):
        return [
            {"role": "user", "content": "Hi, a few things about me."},
            {"role": "assistant", "content": "Noted."},
        ]


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_extracted_facts_get_tagged(monkeypatch):
    facts_json = (
        '[{"text": "Alice lives in Lisbon", "durability": 0.8, '
        '"context_hint": "talking about where they live"}]'
    )

    async def _fake_memory_llm(role, messages, **kwargs):
        assert role == "smart"
        return facts_json

    calls = []

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        calls.append({"text": text, "context_hint": context_hint, "owner": owner, "interactive": interactive})
        return {"tags": ["place:lisbon"], "provisional_tags": [], "generality": 2}

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", _fake_memory_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    # No more legacy "category" seed tag on the LLM path — tags come purely
    # from the tagger now.
    assert entry["tags"] == ["place:lisbon"]
    assert entry["generality"] == 2
    assert entry["tier"] is not None
    # Background call, not inline in a tracked HTTP request -> gate stays on.
    # context_hint from the extraction JSON is threaded through to the tagger.
    assert calls == [{
        "text": "Alice lives in Lisbon",
        "context_hint": "talking about where they live",
        "owner": "alice",
        "interactive": False,
    }]


def test_high_generality_fact_lands_durable_tier_without_auto_pin(monkeypatch):
    facts_json = (
        '[{"text": "User is called Sam", "durability": 0.95, '
        '"context_hint": "introducing themselves"}]'
    )

    async def _fake_memory_llm(role, messages, **kwargs):
        assert role == "smart"
        return facts_json

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        assert context_hint == "introducing themselves"
        return {"tags": ["person:sam"], "provisional_tags": [], "generality": 3}

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", _fake_memory_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    assert entry["tags"] == ["person:sam"]
    # Phase 5 removes the extractor's identity auto-pin entirely — pinning
    # is now a purely manual user flag, never set by extraction.
    assert entry["pinned"] is False
    assert entry["generality"] == 3
    # A brand-new, zero-usage entry can't quite clear the CORE threshold
    # (that needs the full recency bonus, which decays the instant any real
    # time elapses) but generality 3 alone guarantees at least DURABLE —
    # see tier_scoring's own test_high_generality_never_falls_below_durable.
    assert entry["tier"] == 1


def test_tagger_failure_still_stores_fact_untagged(monkeypatch):
    facts_json = (
        '[{"text": "Alice likes tea", "durability": 0.7, '
        '"context_hint": "discussing drinks"}]'
    )

    async def _fake_memory_llm(role, messages, **kwargs):
        assert role == "smart"
        return facts_json

    from services.memory.memory_tagger import FALLBACK_RESULT

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        return dict(FALLBACK_RESULT)

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", _fake_memory_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    # Nothing lost: a tagger fallback just leaves the entry untagged (no
    # legacy category seed on the LLM path any more).
    assert entry["tags"] == []
    assert entry["generality"] is None
    assert entry["tier"] == 2


def test_low_durability_fact_is_dropped_before_storage(monkeypatch):
    facts_json = (
        '[{"text": "User is annoyed today", "durability": 0.2, '
        '"context_hint": "venting"}]'
    )

    async def _fake_memory_llm(role, messages, **kwargs):
        assert role == "smart"
        return facts_json

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", _fake_memory_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None))
        stored = mgr.load(owner="alice")

    # Below DURABILITY_THRESHOLD (0.6) -> dropped before it ever reaches the
    # tagger/store loop.
    assert stored == []
