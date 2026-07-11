"""Phase 1 of the memory upgrade: tags schema, category migration, tier scoring.

Covers the invariants the later phases build on: lazy category->tags migration
(ids preserved so the vector index stays valid), tag normalization rules, the
new entry defaults, and the pure tier-scoring math with hysteresis.
"""

import json
import time

from src.memory import (
    MAX_TAGS,
    MemoryManager,
    compat_category,
    normalize_tag,
    normalize_tags,
)
from services.memory.tier_scoring import (
    TIER_ARCHIVE,
    TIER_CORE,
    TIER_DURABLE,
    TIER_SITUATIONAL,
    initial_tier,
    next_tier,
    recency_decay,
    tier_score,
    usage_frequency,
)
from services.memory.memory_extractor import _fingerprint_entries


# ── Tag normalization ──

def test_normalize_tag_kebab_cases():
    assert normalize_tag("Dress Code") == "dress-code"
    assert normalize_tag("  WORK  ") == "work"
    assert normalize_tag("snake_case_tag") == "snake-case-tag"
    assert normalize_tag("weird!!chars##") == "weirdchars"


def test_normalize_tag_prefixes():
    # Allowed prefixes keep their colon; anything else becomes a separator
    assert normalize_tag("Person: Sven") == "person:sven"
    assert normalize_tag("org:acme corp") == "org:acme-corp"
    assert normalize_tag("madeup:prefix") == "madeup-prefix"
    # A bare prefix with no value is meaningless
    assert normalize_tag("person:") is None


def test_normalize_tag_rejects_empty_and_oversized():
    assert normalize_tag("") is None
    assert normalize_tag("   ") is None
    assert normalize_tag(None) is None
    assert normalize_tag("!!!") is None
    assert normalize_tag("x" * 60) is None


def test_normalize_tags_dedupes_and_caps():
    assert normalize_tags(["Work", "work", " WORK "]) == ["work"]
    many = [f"tag-{i}" for i in range(MAX_TAGS + 5)]
    assert len(normalize_tags(many)) == MAX_TAGS
    # order-preserving, drops unusable entries
    assert normalize_tags(["b", "", "a", None, "b"]) == ["b", "a"]
    # a bare string is treated as one tag, not iterated by character
    assert normalize_tags("hobby") == ["hobby"]


# ── Schema migration ──

def test_validate_entries_migrates_category_to_tags(tmp_path):
    m = MemoryManager(str(tmp_path))
    out = m._validate_entries([
        {"id": "a", "text": "user likes tea", "category": "preference"},
        {"id": "b", "text": "user's name is Sam"},
    ])
    a, b = out
    assert a["tags"] == ["preference"]
    assert "category" not in a
    assert b["tags"] == []
    # ids preserved — the Chroma index keys off them
    assert [e["id"] for e in out] == ["a", "b"]


def test_validate_entries_new_defaults(tmp_path):
    m = MemoryManager(str(tmp_path))
    (entry,) = m._validate_entries([{"id": "a", "text": "t", "category": "fact"}])
    assert entry["tier"] == 2
    assert entry["generality"] is None  # marks "needs curator triage"
    assert entry["last_used_at"] == 0
    assert entry["pinned"] is False
    assert entry["provisional_tags"] == []
    assert entry["uses"] == 0


def test_migration_persists_via_save_roundtrip(tmp_path):
    # Simulate a pre-upgrade memory.json on disk, then load -> save -> reload.
    m = MemoryManager(str(tmp_path))
    old = [{"id": "a", "text": "works at Acme", "timestamp": 1, "source": "auto",
            "category": "fact", "uses": 3, "pinned": True}]
    with open(m.memory_file, "w", encoding="utf-8") as f:
        json.dump(old, f)

    loaded = m.load_all()
    m.save(loaded)
    with open(m.memory_file, encoding="utf-8") as f:
        on_disk = json.load(f)

    (entry,) = on_disk
    assert entry["tags"] == ["fact"]
    assert "category" not in entry
    assert entry["uses"] == 3 and entry["pinned"] is True  # metadata survives


def test_add_entry_tags_and_category_alias(tmp_path):
    m = MemoryManager(str(tmp_path))
    e1 = m.add_entry("t", tags=["Work", "person:Sven"])
    assert e1["tags"] == ["work", "person:sven"]
    e2 = m.add_entry("t", category="preference")
    assert e2["tags"] == ["preference"]
    e3 = m.add_entry("t", tags=["hobby"], category="preference")
    assert e3["tags"] == ["hobby", "preference"]


def test_compat_category_is_first_tag():
    assert compat_category({"tags": ["preference", "work"]}) == "preference"
    assert compat_category({"tags": []}) == "fact"
    assert compat_category({}) == "fact"


def test_increment_uses_sets_last_used_at(tmp_path):
    m = MemoryManager(str(tmp_path))
    entry = m.add_entry("t", tags=["fact"])
    m.save([entry])
    m.increment_uses([entry["id"]])
    (loaded,) = m.load_all()
    assert loaded["uses"] == 1
    assert loaded["last_used_at"] >= int(time.time()) - 5


# ── Audit fingerprint ──

def test_fingerprint_ignores_usage_but_not_tags():
    base = [{"id": "a", "text": "t", "tags": ["work"], "uses": 0, "last_used_at": 0}]
    bumped = [{**base[0], "uses": 9, "last_used_at": 123}]
    retagged = [{**base[0], "tags": ["work", "hobby"]}]
    # retrieval bumping counters must not defeat the audit short-circuit...
    assert _fingerprint_entries(base) == _fingerprint_entries(bumped)
    # ...but a retag must invalidate it
    assert _fingerprint_entries(base) != _fingerprint_entries(retagged)


# ── Tier scoring ──

def test_usage_frequency_saturates():
    assert usage_frequency(0, 30) == 0.0
    assert usage_frequency(4, 30) == 1.0
    assert usage_frequency(400, 30) == 1.0
    # young memories are floored to one month of age
    assert usage_frequency(2, 1) == 0.5


def test_recency_decay_half_life():
    now = time.time()
    assert recency_decay(now, now) == 1.0
    month_ago = now - 30 * 86400
    assert abs(recency_decay(month_ago, now) - 0.5) < 0.01
    assert recency_decay(0, now) == 0.0


def test_tier_score_untriaged_is_none():
    assert tier_score({"generality": None, "timestamp": time.time()}) is None


def test_tier_score_fresh_entries_by_generality():
    now = time.time()
    fresh = {"timestamp": now, "uses": 0, "last_used_at": 0}
    # generality dominates: fresh core fact reaches tier 0, trivia does not
    assert initial_tier(tier_score({**fresh, "generality": 3}, now)) == TIER_CORE
    assert initial_tier(tier_score({**fresh, "generality": 2}, now)) == TIER_DURABLE
    assert initial_tier(tier_score({**fresh, "generality": 1}, now)) == TIER_SITUATIONAL
    assert initial_tier(tier_score({**fresh, "generality": 0}, now)) == TIER_SITUATIONAL


def test_tier_score_low_generality_ages_into_archive():
    now = time.time()
    stale = {"timestamp": now - 200 * 86400, "uses": 0,
             "last_used_at": now - 200 * 86400, "generality": 0}
    assert initial_tier(tier_score(stale, now)) == TIER_ARCHIVE


def test_high_generality_never_falls_below_durable():
    # A core fact (generality 3) unused for a year still scores >= W_GENERALITY,
    # keeping it at least durable — asking the user's name again would be a bug.
    now = time.time()
    old = {"timestamp": now - 365 * 86400, "uses": 0,
           "last_used_at": now - 365 * 86400, "generality": 3}
    score = tier_score(old, now)
    assert initial_tier(score) <= TIER_DURABLE
    assert next_tier(TIER_DURABLE, score) == TIER_DURABLE


def test_next_tier_hysteresis_holds_boundary_scores():
    # 0.45 is the initial boundary between durable and situational: an entry
    # sitting there must keep whichever tier it already has (no flapping).
    assert initial_tier(0.45) == TIER_DURABLE
    assert next_tier(TIER_DURABLE, 0.45) == TIER_DURABLE
    assert next_tier(TIER_SITUATIONAL, 0.45) == TIER_SITUATIONAL
    # clear the promote threshold -> promoted; fall below demote -> demoted
    assert next_tier(TIER_SITUATIONAL, 0.55) == TIER_DURABLE
    assert next_tier(TIER_DURABLE, 0.35) == TIER_SITUATIONAL


def test_next_tier_moves_one_step_per_run():
    assert next_tier(TIER_ARCHIVE, 0.99) == TIER_SITUATIONAL
    assert next_tier(TIER_CORE, 0.0) == TIER_DURABLE


def test_next_tier_untriaged_and_bounds():
    assert next_tier(TIER_DURABLE, None) == TIER_DURABLE
    assert next_tier(-5, 0.99) == TIER_CORE
    assert next_tier(99, 0.0) == TIER_ARCHIVE
