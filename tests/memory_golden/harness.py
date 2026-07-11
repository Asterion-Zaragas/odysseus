"""
Golden-set loader + scoring helpers (memory upgrade Phase 8).

Three JSON fixtures live alongside this file:
  - fixture_store.json    ~25 memory entries spanning every tier/tag shape,
                           used as the fixed candidate pool for retrieval cases.
  - retrieval_cases.json  ~30 query -> expected-memory-id cases against the
                           fixture store, one per effort level.
  - transcripts.json      ~20 synthetic extraction transcripts, each carrying
                           a "golden" (i.e. hand-authored ground-truth) model
                           response alongside the conversation.

Every fixture is consumed two ways:
  1. **Deterministic regression** (`tests/test_memory_golden.py`, runs in plain
     CI, no LLM): the golden model response is fed back in through a mocked
     `memory_llm_call_async`, so the *code* (parsing, durability threshold,
     dedup, tier caps, tag filtering, BM25 ranking) is exercised end-to-end
     against realistic-shaped LLM output without needing a live model.
  2. **Live benchmark** (`scripts/memory_golden_bench.py`, opt-in, needs a
     configured endpoint): the real memory-fast/memory-smart roles are called
     over the same fixtures and scored against the golden answers — this is
     the model-selection benchmark (e.g. comparing a 4B vs 8B tagger) and the
     regression check for prompt edits that the implementation plan calls for.
"""

import json
import os
import time
from typing import Any, Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> Any:
    with open(os.path.join(_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixture_store_raw() -> List[Dict]:
    """The fixture entries exactly as authored (no defaults filled in)."""
    return _load("fixture_store.json")


def load_fixture_store(now: float = None) -> List[Dict]:
    """Fixture entries with the runtime fields retrieval/tier_scoring expect
    filled in with neutral defaults, so callers don't need MemoryManager to
    exercise them. `now` lets callers pin a timestamp for reproducible
    recency scoring; defaults to the current time (fresh entries)."""
    now = time.time() if now is None else now
    entries = []
    for raw in load_fixture_store_raw():
        entry = {
            "timestamp": now,
            "uses": 0,
            "last_used_at": 0,
            "provisional_tags": [],
            "owner": "golden-owner",
            "source": "auto",
        }
        entry.update(raw)
        entries.append(entry)
    return entries


def load_retrieval_cases() -> List[Dict]:
    return _load("retrieval_cases.json")


def load_transcripts() -> List[Dict]:
    return _load("transcripts.json")


def recall_at_k(returned_ids: List[str], expected_ids: List[str]) -> float:
    """Fraction of expected_ids present anywhere in returned_ids. 1.0 if
    expected_ids is empty (nothing to find, so trivially satisfied)."""
    if not expected_ids:
        return 1.0
    hit = sum(1 for eid in expected_ids if eid in returned_ids)
    return hit / len(expected_ids)


def precision_at_k(returned_ids: List[str], expected_ids: List[str]) -> float:
    """Fraction of returned_ids that were actually expected. 1.0 if nothing
    was returned (vacuously precise)."""
    if not returned_ids:
        return 1.0
    hit = sum(1 for rid in returned_ids if rid in expected_ids)
    return hit / len(returned_ids)


def summarize_scores(rows: List[Dict]) -> Dict[str, float]:
    """rows: [{"recall": float, "precision": float}, ...] -> averages."""
    if not rows:
        return {"recall": 1.0, "precision": 1.0, "n": 0}
    n = len(rows)
    return {
        "recall": sum(r["recall"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "n": n,
    }
