"""Phase 3 of the memory upgrade: context document + tagger agent.

Covers services/memory/memory_context.py (per-owner context doc, cold-start
capable) and services/memory/memory_tagger.py (tag_memory + apply_tags:
registry-constrained tagging, generality clamping, tier computation, and the
"never fail the write" fallback contract).
"""

import pytest

from services.memory import memory_context as mc
from services.memory import memory_tagger as mt


# ── MemoryContext ──

@pytest.fixture()
def context_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "CONTEXT_DIR", str(tmp_path))
    return tmp_path


def test_owner_slug_sanitizes_and_defaults():
    assert mc._owner_slug(None) == "default"
    assert mc._owner_slug("alice") == "alice"
    # "/" (a path separator) is replaced; "." is an allowed char (kept as-is).
    assert mc._owner_slug("alice/../etc") == "alice_.._etc"
    assert "/" not in mc._owner_slug("../../etc/passwd")
    assert mc._owner_slug("a b!c") == "a_b_c"


def test_cold_start_load_returns_empty_context(context_dir):
    ctx = mc.MemoryContext("alice")
    data = ctx.load()
    assert data["core_facts"] == []
    assert data["tag_registry"] == []
    assert data["version"] == mc._SCHEMA_VERSION


def test_save_load_round_trips(context_dir):
    ctx = mc.MemoryContext("alice")
    payload = mc._empty_context()
    payload["core_facts"] = ["User's name is Sam"]
    payload["tag_registry"] = [{"name": "work", "count": 3}]
    ctx.save(payload)

    reloaded = mc.MemoryContext("alice").load()
    assert reloaded["core_facts"] == ["User's name is Sam"]
    assert reloaded["tag_registry"] == [{"name": "work", "count": 3}]
    assert reloaded["updated_at"] > 0


def test_different_owners_get_different_files(context_dir):
    mc.MemoryContext("alice").save({**mc._empty_context(), "core_facts": ["alice fact"]})
    mc.MemoryContext("bob").save({**mc._empty_context(), "core_facts": ["bob fact"]})

    assert mc.MemoryContext("alice").load()["core_facts"] == ["alice fact"]
    assert mc.MemoryContext("bob").load()["core_facts"] == ["bob fact"]


def test_registry_names_cold_start_is_empty_set(context_dir):
    assert mc.MemoryContext("alice").registry_names() == set()


def test_registry_names_reflects_saved_registry(context_dir):
    ctx = mc.MemoryContext("alice")
    ctx.save({**mc._empty_context(), "tag_registry": [{"name": "work"}, {"name": "hobby"}]})
    assert ctx.registry_names() == {"work", "hobby"}


def test_registry_excerpt_cold_start_message(context_dir):
    excerpt = mc.MemoryContext("alice").registry_excerpt()
    assert "no tags registered" in excerpt


def test_registry_excerpt_sorted_by_count_and_capped(context_dir):
    ctx = mc.MemoryContext("alice")
    ctx.save({**mc._empty_context(), "tag_registry": [
        {"name": "low", "count": 1},
        {"name": "high", "count": 9, "description": "top tag", "aliases": ["hi"]},
        {"name": "mid", "count": 5},
    ]})
    excerpt = ctx.registry_excerpt(max_tags=2)
    lines = excerpt.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("- high")
    assert "top tag" in lines[0]
    assert "aka hi" in lines[0]
    assert lines[1].startswith("- mid")


def test_core_facts_excerpt(context_dir):
    ctx = mc.MemoryContext("alice")
    assert "no core facts" in ctx.core_facts_excerpt()
    ctx.save({**mc._empty_context(), "core_facts": ["fact one", "fact two"]})
    excerpt = mc.MemoryContext("alice").core_facts_excerpt()
    assert excerpt == "- fact one\n- fact two"


def test_render_markdown_contains_sections(context_dir):
    ctx = mc.MemoryContext("alice")
    ctx.save({**mc._empty_context(), "core_facts": ["fact one"], "tag_registry": [{"name": "work", "count": 2}]})
    md = mc.MemoryContext("alice").render_markdown()
    assert "# Memory context" in md
    assert "## Core facts" in md
    assert "- fact one" in md
    assert "## Tags" in md
    assert "**work**" in md


# ── _parse_tagger_json ──

def test_parse_tagger_json_clean_object():
    assert mt._parse_tagger_json('{"tags": ["work"], "generality": 1}') == {"tags": ["work"], "generality": 1}


def test_parse_tagger_json_strips_code_fence():
    raw = '```json\n{"tags": ["work"], "generality": 1}\n```'
    assert mt._parse_tagger_json(raw) == {"tags": ["work"], "generality": 1}


def test_parse_tagger_json_strips_think_block():
    raw = '<think>reasoning about tags</think>\n{"tags": [], "generality": 0}'
    assert mt._parse_tagger_json(raw) == {"tags": [], "generality": 0}


def test_parse_tagger_json_surrounding_prose():
    raw = 'Sure, here is the tagging:\n{"tags": ["work"], "generality": 2}\nDone!'
    assert mt._parse_tagger_json(raw) == {"tags": ["work"], "generality": 2}


def test_parse_tagger_json_garbage_returns_none():
    assert mt._parse_tagger_json("not json at all") is None
    assert mt._parse_tagger_json("") is None
    assert mt._parse_tagger_json(None) is None


def test_parse_tagger_json_array_is_not_a_dict():
    assert mt._parse_tagger_json("[1, 2, 3]") is None


# ── tag_memory ──

@pytest.mark.asyncio
async def test_tag_memory_empty_text_short_circuits(monkeypatch):
    async def fail_if_called(*a, **k):
        raise AssertionError("must not call the LLM for empty text")

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fail_if_called)
    result = await mt.tag_memory("   ")
    assert result == mt.FALLBACK_RESULT


@pytest.mark.asyncio
async def test_tag_memory_llm_failure_returns_fallback(context_dir, monkeypatch):
    async def raise_error(*a, **k):
        raise RuntimeError("no endpoint")

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", raise_error)
    result = await mt.tag_memory("User likes tea", owner="alice")
    assert result == mt.FALLBACK_RESULT


@pytest.mark.asyncio
async def test_tag_memory_malformed_json_returns_fallback(context_dir, monkeypatch):
    async def bad_json(*a, **k):
        return "not json"

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", bad_json)
    result = await mt.tag_memory("User likes tea", owner="alice")
    assert result == mt.FALLBACK_RESULT


@pytest.mark.asyncio
async def test_tag_memory_cold_start_all_provisional(context_dir, monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        assert role == "fast"
        return '{"tags": ["work"], "new_tags": ["person:sven"], "generality": 1}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    result = await mt.tag_memory("Sven is my manager", owner="alice")
    # Registry is empty (cold start) -> everything provisional, per the
    # concept doc's cold-start contract, regardless of the tags/new_tags split.
    assert result["tags"] == []
    assert set(result["provisional_tags"]) == {"work", "person:sven"}
    assert result["generality"] == 1


@pytest.mark.asyncio
async def test_tag_memory_classifies_against_registry(context_dir, monkeypatch):
    mc.MemoryContext("alice").save({**mc._empty_context(), "tag_registry": [{"name": "work", "count": 5}]})

    async def fake_llm(role, messages, **kwargs):
        return '{"tags": ["work"], "new_tags": ["person:sven"], "generality": 2}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    result = await mt.tag_memory("Sven works with me", owner="alice")
    assert result["tags"] == ["work"]
    assert result["provisional_tags"] == ["person:sven"]
    assert result["generality"] == 2


@pytest.mark.asyncio
async def test_tag_memory_clamps_generality(context_dir, monkeypatch):
    async def fake_llm(role, messages, **kwargs):
        return '{"tags": [], "generality": 99}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    result = await mt.tag_memory("trivial fact", owner="alice")
    assert result["generality"] == 3

    async def fake_llm_negative(role, messages, **kwargs):
        return '{"tags": [], "generality": -5}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm_negative)
    result = await mt.tag_memory("trivial fact", owner="alice")
    assert result["generality"] == 0

    async def fake_llm_bad(role, messages, **kwargs):
        return '{"tags": [], "generality": "high"}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm_bad)
    result = await mt.tag_memory("trivial fact", owner="alice")
    assert result["generality"] is None


@pytest.mark.asyncio
async def test_tag_memory_forwards_interactive_and_role(context_dir, monkeypatch):
    captured = {}

    async def fake_llm(role, messages, **kwargs):
        captured["role"] = role
        captured.update(kwargs)
        return '{"tags": [], "generality": 0}'

    monkeypatch.setattr("src.task_endpoint.memory_llm_call_async", fake_llm)
    await mt.tag_memory("text", owner="alice", interactive=True)
    assert captured["role"] == "fast"
    assert captured["interactive"] is True
    assert captured["owner"] == "alice"


# ── apply_tags ──

def test_apply_tags_user_tags_win_outright():
    entry = {"tags": ["fact"], "generality": None, "uses": 0, "timestamp": 0, "last_used_at": 0}
    result = {"tags": ["ignored"], "provisional_tags": ["also-ignored"], "generality": 1}
    mt.apply_tags(entry, result, user_tags=["work", "person:sven"])
    assert entry["tags"] == ["work", "person:sven"]
    assert entry["provisional_tags"] == []
    assert entry["generality"] == 1


def test_apply_tags_merges_seeded_and_tagger_tags():
    entry = {"tags": ["fact"], "generality": None, "uses": 0, "timestamp": 0, "last_used_at": 0}
    result = {"tags": ["work"], "provisional_tags": ["person:sven"], "generality": 1}
    mt.apply_tags(entry, result, user_tags=None)
    assert set(entry["tags"]) == {"fact", "work"}
    assert entry["provisional_tags"] == ["person:sven"]
    assert entry["generality"] == 1


def test_apply_tags_fallback_result_preserves_seeded_tags():
    entry = {"tags": ["fact"], "generality": 2, "uses": 0, "timestamp": 0, "last_used_at": 0}
    mt.apply_tags(entry, dict(mt.FALLBACK_RESULT), user_tags=None)
    assert entry["tags"] == ["fact"]
    assert entry["provisional_tags"] == []
    # A failed re-tag must not erase a previously-set generality.
    assert entry["generality"] == 2


def test_apply_tags_computes_tier_from_generality():
    import time
    now = time.time()
    # A brand-new, zero-usage entry sits exactly ON the CORE/DURABLE boundary
    # for generality 3 (0.5 generality + 0.2 recency == 0.70), which is only
    # reproducible if tier_score's internal time.time() call lands at exactly
    # `now` — not guaranteed here since apply_tags doesn't take a `now`
    # override. Give it real usage/recency margin instead, so the assertion
    # is about apply_tags wiring into tier_scoring correctly, not a
    # microsecond race against a knife-edge boundary.
    entry = {
        "tags": [], "generality": None, "uses": 100,
        "timestamp": now - 60 * 86400, "last_used_at": now,
    }
    result = {"tags": [], "provisional_tags": [], "generality": 3}
    mt.apply_tags(entry, result, user_tags=None)
    assert entry["tier"] == 0  # heavily used, recently, identity-level -> core


def test_apply_tags_none_generality_defaults_to_situational_tier():
    entry = {"tags": [], "generality": None, "uses": 0, "timestamp": 0, "last_used_at": 0}
    mt.apply_tags(entry, dict(mt.FALLBACK_RESULT), user_tags=None)
    assert entry["tier"] == 2
