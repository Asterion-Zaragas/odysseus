#!/usr/bin/env python3
"""Memory-system golden-set benchmark (memory upgrade Phase 8).

Replays tests/memory_golden/{transcripts,retrieval_cases}.json against the
REAL configured memory-fast/memory-smart model roles (settings:
memory_fast_endpoint_id/-model, memory_smart_endpoint_id/-model, falling
back through the same task/utility/default chain the app uses) instead of
the canned answers tests/test_memory_golden.py uses. Two things this is for:

  1. Model-selection benchmark - point memory_fast/-smart at different
     models (e.g. a 4B vs an 8B tagger) and compare the scorecards below.
  2. Regression check for prompt edits - a change to EXTRACT_SYSTEM_PROMPT_
     TEMPLATE, TAGGER_SYSTEM_PROMPT, FACET_SYSTEM_PROMPT, or
     VERIFY_SYSTEM_PROMPT should be re-run against this before merging.

Needs a running app data dir (data/settings.json) with a model configured for
at least the memory-smart role (extraction) and ideally memory-fast too
(facets/verify; retrieval degrades to low without it). Talks to whatever
endpoint that resolves to - no mock, no sandbox: only run this against a
model you intend to actually call.

Usage:
    python scripts/memory_golden_bench.py                # both sets
    python scripts/memory_golden_bench.py --set retrieval
    python scripts/memory_golden_bench.py --set extraction --limit 5
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import MemoryManager, get_text_similarity
from tests.memory_golden.harness import (
    load_fixture_store,
    load_retrieval_cases,
    load_transcripts,
    precision_at_k,
    recall_at_k,
    summarize_scores,
)

_FUZZY_MATCH_THRESHOLD = 0.4  # Jaccard word overlap - live models paraphrase, so
                              # this is intentionally looser than an exact match.


class _BenchSession:
    def __init__(self, owner, messages):
        self.owner = owner
        self.session_id = f"bench-{owner}"
        self._messages = messages

    def get_context_messages(self):
        return self._messages


async def _bench_retrieval(limit=None):
    from services.memory import retrieval

    entries = load_fixture_store()
    cases = load_retrieval_cases()
    if limit:
        cases = cases[:limit]

    rows = []
    print(f"\n=== Retrieval golden set: {len(cases)} cases (live) ===")
    for case in cases:
        t0 = time.monotonic()
        result = await retrieval.retrieve(
            case["query"], entries, effort=case["effort"], memory_vector=None,
            owner="golden-bench", k=5, interactive=True,
        )
        elapsed = time.monotonic() - t0
        returned_ids = [m["id"] for m in result["memories"]]
        recall = recall_at_k(returned_ids, case["expected_ids"])
        leaked = [eid for eid in case["must_not_include"] if eid in returned_ids]
        degraded = result["effort_used"] != case["effort"]
        rows.append({"recall": recall, "precision": precision_at_k(returned_ids, case["expected_ids"])})
        flag = "OK" if recall == 1.0 and not leaked else "MISS"
        degrade_note = f" (degraded to {result['effort_used']})" if degraded else ""
        print(f"  [{flag:4s}] {case['id']:28s} {elapsed*1000:6.0f}ms recall={recall:.2f}"
              f"{'  LEAKED='+str(leaked) if leaked else ''}{degrade_note}")

    summary = summarize_scores(rows)
    print(f"\nOverall: recall={summary['recall']:.3f} precision={summary['precision']:.3f} (n={summary['n']})")


async def _bench_extraction(limit=None):
    from services.memory import memory_extractor

    cases = load_transcripts()
    if limit:
        cases = cases[:limit]

    print(f"\n=== Extraction golden set: {len(cases)} cases (live) ===")
    correct_decisions = 0
    fuzzy_hits = 0
    total_expected = 0
    for case in cases:
        expected_texts = {f["text"] for f in case["golden_facts"] if f["durability"] >= memory_extractor.DURABILITY_THRESHOLD}
        owner = f"bench-{case['id']}"
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as data_dir:
            mgr = MemoryManager(data_dir)
            await memory_extractor.extract_and_store(_BenchSession(owner, case["messages"]), mgr, None)
            stored = mgr.load(owner=owner)
        elapsed = time.monotonic() - t0
        stored_texts = [e["text"] for e in stored]

        decision_correct = bool(stored_texts) == bool(expected_texts)
        correct_decisions += int(decision_correct)

        hits = 0
        for exp in expected_texts:
            if any(get_text_similarity(exp, got) >= _FUZZY_MATCH_THRESHOLD for got in stored_texts):
                hits += 1
        fuzzy_hits += hits
        total_expected += len(expected_texts)

        flag = "OK" if decision_correct and hits == len(expected_texts) else "CHECK"
        print(f"  [{flag:5s}] {case['id']:28s} {elapsed*1000:6.0f}ms "
              f"expected={sorted(expected_texts)} stored={stored_texts}")

    n = len(cases)
    print(f"\nExtract/skip decision accuracy: {correct_decisions}/{n} "
          f"({100*correct_decisions/n:.0f}%)" if n else "\nNo cases run.")
    if total_expected:
        print(f"Fuzzy content match on expected facts: {fuzzy_hits}/{total_expected} "
              f"({100*fuzzy_hits/total_expected:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("retrieval", "extraction", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases of each set")
    args = parser.parse_args()

    async def _main():
        if args.set in ("retrieval", "both"):
            await _bench_retrieval(args.limit)
        if args.set in ("extraction", "both"):
            await _bench_extraction(args.limit)

    asyncio.run(_main())


if __name__ == "__main__":
    main()
