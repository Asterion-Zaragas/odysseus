"""Phase 3 of the memory upgrade: the extractor's store loop calls the tagger.

Regression coverage for wiring tag_memory()/apply_tags() into
services/memory/memory_extractor.py's extract_and_store — every auto-
extracted fact should land with tagger-assigned tags/generality/tier, and
the existing identity auto-pin (still present until Phase 5 removes it)
must keep working on top of the tagger's tags.
"""

import asyncio
import tempfile

import pytest

import src.event_bus
import src.llm_core
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
    facts_json = '[{"text": "Alice lives in Lisbon", "category": "fact"}]'

    async def _fake_llm(url, model, messages, **kwargs):
        return facts_json

    calls = []

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        calls.append({"text": text, "owner": owner, "interactive": interactive})
        return {"tags": ["place:lisbon"], "provisional_tags": [], "generality": 2}

    monkeypatch.setattr(src.llm_core, "llm_call_async", _fake_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None, endpoint_url="http://x", model="m", headers=None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    # "fact" (from add_entry's category param) merges with the tagger's tag.
    assert set(entry["tags"]) == {"fact", "place:lisbon"}
    assert entry["generality"] == 2
    assert entry["tier"] is not None
    # Background call, not inline in a tracked HTTP request -> gate stays on.
    assert calls == [{"text": "Alice lives in Lisbon", "owner": "alice", "interactive": False}]


def test_identity_auto_pin_still_applies_after_tagging(monkeypatch):
    facts_json = '[{"text": "User is called Sam", "category": "identity"}]'

    async def _fake_llm(url, model, messages, **kwargs):
        return facts_json

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        return {"tags": ["person:sam"], "provisional_tags": [], "generality": 3}

    monkeypatch.setattr(src.llm_core, "llm_call_async", _fake_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None, endpoint_url="http://x", model="m", headers=None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    # "identity" (from add_entry's category) survives the merge, so the
    # existing auto-pin check (`"identity" in entry["tags"]`) still fires.
    assert "identity" in entry["tags"]
    assert entry["pinned"] is True
    assert entry["generality"] == 3
    # A brand-new, zero-usage entry can't quite clear the CORE threshold
    # (that needs the full recency bonus, which decays the instant any real
    # time elapses) but generality 3 alone guarantees at least DURABLE —
    # see tier_scoring's own test_high_generality_never_falls_below_durable.
    assert entry["tier"] == 1


def test_tagger_failure_still_stores_fact_untagged(monkeypatch):
    facts_json = '[{"text": "Alice likes tea", "category": "preference"}]'

    async def _fake_llm(url, model, messages, **kwargs):
        return facts_json

    from services.memory.memory_tagger import FALLBACK_RESULT

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        return dict(FALLBACK_RESULT)

    monkeypatch.setattr(src.llm_core, "llm_call_async", _fake_llm)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)

    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(extract_and_store(_FakeSession(), mgr, None, endpoint_url="http://x", model="m", headers=None))
        stored = mgr.load(owner="alice")

    assert len(stored) == 1
    entry = stored[0]
    # Nothing lost: the legacy category-derived tag survives a tagger fallback.
    assert entry["tags"] == ["preference"]
    assert entry["generality"] is None
    assert entry["tier"] == 2
