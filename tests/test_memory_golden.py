"""Golden-set regression suite (memory upgrade Phase 8).

Two independent golden sets live in tests/memory_golden/ (see harness.py's
docstring for the full rationale):

  - retrieval_cases.json: ~30 query -> expected-memory-id cases against a
    fixed ~25-entry fixture store, run directly through the pure-code
    `retrieval._stage_b_rank` (tier caps, tag filtering, BM25 ranking, tier
    weighting) - no LLM involved, so this runs in plain CI.
  - transcripts.json: ~20 synthetic extraction transcripts, each carrying a
    "golden" (hand-authored ground-truth) extraction result. The golden
    result is fed back in through a mocked memory-smart call so the code
    around it (durability threshold, dedup, storage) is exercised the same
    way tests/test_memory_extractor_tagger_phase3.py does, just across a
    broader, curated set of scenarios (transient vs durable, boundary
    durability, multi-fact turns, assistant-vs-user attribution).

Both sets double as fixtures for scripts/memory_golden_bench.py, which
replays them against a *real* configured model pair instead of canned
answers - that's the model-selection benchmark / prompt-regression tool the
implementation plan calls for; it is intentionally not a pytest test since
it needs a live endpoint.
"""

import asyncio
import json
import tempfile

import pytest

import src.event_bus
from services.memory import memory_extractor, retrieval
from src.memory import MemoryManager
from tests.memory_golden.harness import (
    load_fixture_store,
    load_retrieval_cases,
    load_transcripts,
    recall_at_k,
)

# ── Retrieval golden set (pure code, no mocking needed) ──

_ENTRIES = load_fixture_store()
_RETRIEVAL_CASES = load_retrieval_cases()


@pytest.mark.parametrize("case", _RETRIEVAL_CASES, ids=[c["id"] for c in _RETRIEVAL_CASES])
def test_retrieval_golden_case(case):
    ranked = retrieval._stage_b_rank(
        case["query"], _ENTRIES, case["effort"], facets=case["facets"], memory_vector=None, pool_k=20,
    )
    returned_ids = [e["id"] for e in ranked]

    missing = [eid for eid in case["expected_ids"] if eid not in returned_ids]
    assert not missing, (
        f"{case['id']}: expected id(s) {missing} not found in top results {returned_ids} "
        f"for query {case['query']!r} at effort={case['effort']}"
    )

    leaked = [eid for eid in case["must_not_include"] if eid in returned_ids]
    assert not leaked, (
        f"{case['id']}: id(s) {leaked} should have been excluded (tier cap / tag filter) "
        f"but appeared in {returned_ids} for query {case['query']!r} at effort={case['effort']}"
    )


def test_retrieval_golden_set_overall_recall_is_perfect():
    """Aggregate sanity check - if individual cases above ever get loosened
    (e.g. must_not_include dropped) this still catches a wholesale ranking
    regression across the set."""
    rows = []
    for case in _RETRIEVAL_CASES:
        ranked = retrieval._stage_b_rank(
            case["query"], _ENTRIES, case["effort"], facets=case["facets"], memory_vector=None, pool_k=20,
        )
        returned_ids = [e["id"] for e in ranked]
        rows.append({
            "recall": recall_at_k(returned_ids, case["expected_ids"]),
            "precision": 1.0,  # precision isn't meaningful here - see must_not_include checks above
        })
    from tests.memory_golden.harness import summarize_scores
    summary = summarize_scores(rows)
    assert summary["recall"] == 1.0, summary


# ── Extraction golden set (transcripts, canned-model-response mode) ──

_TRANSCRIPTS = load_transcripts()

# DURABILITY_THRESHOLD is a module constant the code filters on; import it
# rather than hardcoding 0.6 here so a future tuning pass can't silently
# desync this test from the code it's supposed to guard.
_THRESHOLD = memory_extractor.DURABILITY_THRESHOLD


def _expected_stored_texts(case):
    return {f["text"] for f in case["golden_facts"] if f["durability"] >= _THRESHOLD}


class _GoldenSession:
    def __init__(self, owner, messages):
        self.owner = owner
        self.session_id = f"golden-{owner}"
        self._messages = messages

    def get_context_messages(self):
        return self._messages


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _isolated_audit_counter(monkeypatch):
    # Shared module global across the whole test session - see the identical
    # fixture in test_memory_extractor_tagger_phase3.py for why this must be
    # pinned per-test.
    monkeypatch.setattr(memory_extractor, "_extractions_since_audit", 0)


@pytest.mark.parametrize("case", _TRANSCRIPTS, ids=[c["id"] for c in _TRANSCRIPTS])
def test_extraction_golden_case(case, monkeypatch):
    golden_json = json.dumps(case["golden_facts"])

    async def _fake_memory_llm(role, messages, **kwargs):
        assert role == "smart"
        return golden_json

    async def _fake_tag_memory(text, context_hint=None, owner=None, *, interactive=False):
        return {"tags": [], "provisional_tags": [], "generality": 1}

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", _fake_memory_llm)
    monkeypatch.setattr("services.memory.memory_tagger.tag_memory", _fake_tag_memory)
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)

    owner = case["id"]
    with tempfile.TemporaryDirectory() as data_dir:
        mgr = MemoryManager(data_dir)
        _run(memory_extractor.extract_and_store(_GoldenSession(owner, case["messages"]), mgr, None))
        stored_texts = {e["text"] for e in mgr.load(owner=owner)}

    assert stored_texts == _expected_stored_texts(case), (
        f"{case['id']} ({case['category']}): stored {stored_texts}, "
        f"expected {_expected_stored_texts(case)} (golden durabilities: "
        f"{[(f['text'], f['durability']) for f in case['golden_facts']]}, threshold={_THRESHOLD})"
    )
