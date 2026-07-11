"""Phase 6 of the memory upgrade: the staged retrieval pipeline.

Covers services/memory/retrieval.py (facet extraction with timeout/parse
degrade, tier-capped + tag-filtered + tier-weighted candidate ranking, the
verify stage's failure-vs-legitimate-empty distinction, and the retrieve()
orchestration), plus the retrieve_memory_context agent tool
(src/ai_interaction.py do_retrieve_memory_context).
"""

import asyncio
import json
import time

from src.memory import MemoryManager
from services.memory import retrieval


def _entry(id_, text, tier=2, tags=None, timestamp=None, uses=0, last_used_at=0, pinned=False):
    return {
        "id": id_,
        "text": text,
        "tier": tier,
        "tags": tags or [],
        # Fresh by default so recency doesn't zero out the relevance gate —
        # individual tests override this when recency itself is under test.
        "timestamp": time.time() if timestamp is None else timestamp,
        "uses": uses,
        "last_used_at": last_used_at,
        "pinned": pinned,
    }


# ── Stage A: facet extraction ──

async def test_extract_facets_parses_json(monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        assert role == "fast"
        return json.dumps({
            "keywords": ["pizza"],
            "entities": [{"type": "person", "name": "Sven"}],
            "tag_guesses": ["food"],
        })
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    facets = await retrieval._extract_facets("what does sven like to eat", owner="u1", interactive=True)

    assert facets["keywords"] == ["pizza"]
    assert facets["tag_guesses"] == ["food"]
    assert facets["entities"] == [{"type": "person", "name": "Sven"}]


async def test_extract_facets_degrades_on_timeout(monkeypatch):
    async def slow_llm(role, messages, **kwargs):
        await asyncio.sleep(10)
        return "{}"
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", slow_llm)
    monkeypatch.setattr(retrieval, "_FACET_TIMEOUT_SECONDS", 0.05)

    facets = await retrieval._extract_facets("hello", owner=None, interactive=True)

    assert facets is None


async def test_extract_facets_degrades_on_bad_json(monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        return "not JSON at all"
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    facets = await retrieval._extract_facets("hello", owner=None, interactive=True)

    assert facets is None


# ── Stage B: pure-code candidate ranking ──

def test_stage_b_rank_excludes_entries_above_the_effort_tier_cap():
    entries = [
        _entry("durable", "sven likes pizza and pasta", tier=1),
        _entry("archived", "sven likes pizza too but old", tier=3),
    ]
    ranked = retrieval._stage_b_rank("what does sven like pizza", entries, "low", None, None)
    ids = [e["id"] for e in ranked]
    assert "archived" not in ids
    assert "durable" in ids


def test_stage_b_rank_tag_filter_applies_when_it_matches():
    entries = [
        _entry("food", "sven likes pizza a lot", tier=2, tags=["food"]),
        _entry("hobby", "sven likes pizza too and hiking", tier=2, tags=["hobby"]),
    ]
    facets = {"tag_guesses": ["food"], "entities": [], "keywords": []}
    ranked = retrieval._stage_b_rank("sven pizza food", entries, "medium", facets, None)
    ids = [e["id"] for e in ranked]
    assert ids == ["food"]


def test_stage_b_rank_tag_filter_skipped_when_intersection_empty():
    entries = [_entry("a", "sven likes pizza a lot", tier=2, tags=["food"])]
    facets = {"tag_guesses": ["nonexistent-tag"], "entities": [], "keywords": []}
    ranked = retrieval._stage_b_rank("sven pizza", entries, "medium", facets, None)
    # No entry carries the guessed tag -> filter is skipped rather than emptying the pool.
    assert [e["id"] for e in ranked] == ["a"]


def test_stage_b_rank_entity_guess_folds_into_tag_filter():
    entries = [
        _entry("sven-fact", "loves hiking on weekends", tier=2, tags=["person:sven"]),
        _entry("other-fact", "loves hiking on weekends too", tier=2, tags=["person:mia"]),
    ]
    facets = {"tag_guesses": [], "entities": [{"type": "person", "name": "sven"}], "keywords": []}
    ranked = retrieval._stage_b_rank("hiking weekends", entries, "medium", facets, None)
    assert [e["id"] for e in ranked] == ["sven-fact"]


def test_stage_b_rank_tier_weight_breaks_ties():
    # Identical text -> identical raw BM25/recency; only the tier-weight
    # multiplier should separate them. Enough shared content tokens to clear
    # the minimum-relevance gate on a tiny (N=2) corpus.
    text = "sven loves long hiking trips every single weekend"
    entries = [
        _entry("core", text, tier=0),
        _entry("archive", text, tier=3),
    ]
    ranked = retrieval._stage_b_rank(text, entries, "high", None, None)
    ids = [e["id"] for e in ranked]
    assert ids.index("core") < ids.index("archive")


def test_stage_b_rank_returns_empty_for_blank_message():
    entries = [_entry("a", "some text", tier=1)]
    assert retrieval._stage_b_rank("", entries, "medium", None, None) == []


# ── Stage C: verify ──

async def test_verify_returns_only_llm_selected_ids(monkeypatch):
    candidates = [_entry("a", "sven likes pizza"), _entry("b", "unrelated fact")]

    async def fake_llm(role, messages, **kwargs):
        return json.dumps({"relevant_ids": ["a"]})
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    result = await retrieval._verify("what does sven like", candidates, owner=None, interactive=True)

    assert [r["id"] for r in result] == ["a"]


async def test_verify_returns_none_on_parse_failure_so_caller_can_fall_back(monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        return "garbage, not json"
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    result = await retrieval._verify("query", [_entry("a", "text")], owner=None, interactive=True)

    assert result is None


async def test_verify_empty_relevant_ids_is_legitimate_not_a_failure(monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        return json.dumps({"relevant_ids": []})
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    result = await retrieval._verify("query", [_entry("a", "text")], owner=None, interactive=True)

    assert result == []


async def test_verify_with_no_candidates_short_circuits_without_a_call(monkeypatch):
    called = {"hit": False}

    async def fake_llm(role, messages, **kwargs):
        called["hit"] = True
        return "{}"
    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)

    result = await retrieval._verify("query", [], owner=None, interactive=True)

    assert result == []
    assert called["hit"] is False


# ── retrieve() orchestration ──

async def test_retrieve_low_effort_never_calls_facets(monkeypatch):
    called = {"hit": False}

    async def fake_extract(*a, **kw):
        called["hit"] = True
        return None
    monkeypatch.setattr(retrieval, "_extract_facets", fake_extract)

    entries = [_entry("a", "sven likes pizza", tier=1)]
    result = await retrieval.retrieve("sven pizza", entries, effort="low")

    assert called["hit"] is False
    assert result["effort_used"] == "low"
    assert result["facets"] is None


async def test_retrieve_medium_degrades_to_low_when_facets_fail(monkeypatch):
    async def fake_extract(*a, **kw):
        return None
    monkeypatch.setattr(retrieval, "_extract_facets", fake_extract)

    entries = [_entry("a", "sven likes pizza", tier=1)]
    result = await retrieval.retrieve("sven pizza", entries, effort="medium")

    assert result["effort_used"] == "low"


async def test_retrieve_high_effort_runs_verify_over_top_candidates(monkeypatch):
    async def fake_extract(*a, **kw):
        return {"keywords": [], "entities": [], "tag_guesses": []}

    verify_calls = []

    async def fake_verify(message, candidates, owner, interactive):
        verify_calls.append(candidates)
        return candidates[:1]

    monkeypatch.setattr(retrieval, "_extract_facets", fake_extract)
    monkeypatch.setattr(retrieval, "_verify", fake_verify)

    text = "sven really loves pizza with extra cheese every friday night"
    entries = [_entry(str(i), text, tier=1) for i in range(3)]
    result = await retrieval.retrieve(text, entries, effort="high")

    assert len(verify_calls) == 1
    assert result["effort_used"] == "high"
    assert len(result["memories"]) == 1


async def test_retrieve_high_effort_falls_back_to_top5_when_verify_fails(monkeypatch):
    async def fake_extract(*a, **kw):
        return {"keywords": [], "entities": [], "tag_guesses": []}

    async def fake_verify(message, candidates, owner, interactive):
        return None  # parse/call failure

    monkeypatch.setattr(retrieval, "_extract_facets", fake_extract)
    monkeypatch.setattr(retrieval, "_verify", fake_verify)

    # A handful of distractor entries keep BM25's idf from collapsing (an
    # all-identical corpus makes the shared phrase look uninformative rather
    # than distinctive) while still giving >5 matching candidates so the
    # top-5 fallback cap is actually exercised.
    text = "sven really loves pizza with extra cheese every friday night"
    entries = [_entry(str(i), text, tier=1) for i in range(6)]
    entries += [_entry("distractor-1", "completely unrelated hobby notes", tier=1)]
    entries += [_entry("distractor-2", "another unconnected topic entirely", tier=1)]
    result = await retrieval.retrieve(text, entries, effort="high")

    assert len(result["memories"]) == 5


async def test_retrieve_normalizes_unknown_effort_to_medium(monkeypatch):
    async def fake_extract(*a, **kw):
        return {"keywords": [], "entities": [], "tag_guesses": []}
    monkeypatch.setattr(retrieval, "_extract_facets", fake_extract)

    result = await retrieval.retrieve("sven pizza", [_entry("a", "sven likes pizza", tier=1)], effort="ludicrous")

    assert result["effort_used"] == "medium"


async def test_retrieve_returns_empty_for_no_entries_or_blank_message():
    result = await retrieval.retrieve("", [_entry("a", "text")], effort="medium")
    assert result == {"memories": [], "facets": None, "effort_used": "medium"}

    result2 = await retrieval.retrieve("hello", [], effort="medium")
    assert result2["memories"] == []


# ── retrieve_memory_context agent tool ──

import pytest

import src.ai_interaction as ai


@pytest.fixture()
def ai_memory_manager(tmp_path):
    mm = MemoryManager(str(tmp_path))
    ai.set_memory_manager(mm, None)
    yield mm
    ai.set_memory_manager(None, None)


async def test_do_retrieve_memory_context_parses_fenced_query_and_effort(ai_memory_manager, monkeypatch):
    mm = ai_memory_manager
    entry = mm.add_entry("Sven likes pizza", tags=["food"], owner="alice")
    entry["tier"] = 1
    with mm.lock:
        mm.save([entry])

    captured = {}

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        captured.update(message=message, effort=effort, owner=owner, k=k, interactive=interactive)
        return {"memories": entries, "facets": None, "effort_used": effort}

    monkeypatch.setattr("services.memory.retrieval.retrieve", fake_retrieve)

    result = await ai.do_retrieve_memory_context("what does sven like\nhigh", owner="alice")

    assert captured["message"] == "what does sven like"
    assert captured["effort"] == "high"
    assert captured["owner"] == "alice"
    assert captured["interactive"] is True
    assert "pizza" in result["results"]


async def test_do_retrieve_memory_context_parses_json_args(ai_memory_manager, monkeypatch):
    mm = ai_memory_manager
    entry = mm.add_entry("Sven likes pizza", owner="alice")
    with mm.lock:
        mm.save([entry])

    captured = {}

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        captured.update(message=message, effort=effort)
        return {"memories": [], "facets": None, "effort_used": effort}

    monkeypatch.setattr("services.memory.retrieval.retrieve", fake_retrieve)

    result = await ai.do_retrieve_memory_context(
        json.dumps({"query": "food preferences", "effort": "low"}), owner="alice"
    )

    assert captured["message"] == "food preferences"
    assert captured["effort"] == "low"
    assert "No relevant memories" in result["results"]


async def test_do_retrieve_memory_context_requires_a_query(ai_memory_manager):
    result = await ai.do_retrieve_memory_context("", owner="alice")
    assert "error" in result


async def test_do_retrieve_memory_context_errors_without_memory_manager():
    ai.set_memory_manager(None, None)
    result = await ai.do_retrieve_memory_context("some query", owner="alice")
    assert "error" in result


async def test_do_retrieve_memory_context_bumps_usage_counters(ai_memory_manager, monkeypatch):
    mm = ai_memory_manager
    entry = mm.add_entry("Sven likes pizza", owner="alice")
    with mm.lock:
        mm.save([entry])

    async def fake_retrieve(message, entries, *, effort, memory_vector, owner, k, interactive):
        return {"memories": [entries[0]], "facets": None, "effort_used": effort}

    monkeypatch.setattr("services.memory.retrieval.retrieve", fake_retrieve)

    await ai.do_retrieve_memory_context("food", owner="alice")

    saved = mm.load(owner="alice")
    assert saved[0]["uses"] == 1
