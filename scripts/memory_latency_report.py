#!/usr/bin/env python3
"""Memory preface-build latency report (memory upgrade Phase 8).

Times `ChatProcessor.build_context_preface`'s memory section at each retrieval
effort level (low/medium/high) against a fixed ~25-entry fixture store
(tests/memory_golden/fixture_store.json), using whatever memory-fast/-smart
model is actually configured (settings.py memory_fast_endpoint_id/-model
etc. - the same resolution chain the app uses). Web/RAG/skills injection are
disabled so the timed span is (as closely as this entry point allows) just
the memory retrieval work Phase 6 added: core-facts injection plus
`services.memory.retrieval.retrieve()` for the non-core pool.

This is a report, not a test: there is no pass/fail threshold here on
purpose (effort=high on a slow/cloud model can legitimately take seconds -
see the implementation plan's risk notes) - run it against your own model
pair and eyeball the numbers.

Usage:
    python scripts/memory_latency_report.py
    python scripts/memory_latency_report.py --repeats 10 --effort medium
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat_processor import ChatProcessor
from tests.memory_golden.harness import load_fixture_store

_SAMPLE_QUERIES = [
    "what is my name",
    "what does the user want to do in japan",
    "what chat tool does the user's team use",
    "did the user's old apartment lease end",
    "does the user have any dietary restrictions",
]


class _FixtureMemoryManager:
    """Read-only stand-in for MemoryManager over the golden fixture store -
    real save-path methods aren't needed for a preface build."""

    def __init__(self, entries):
        self._entries = entries

    def load(self, owner=None):
        return list(self._entries)

    def increment_uses(self, ids):
        pass


def _session():
    return SimpleNamespace(endpoint_url="http://localhost:8080", model="local", headers={})


def _percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


async def _time_effort(processor, effort, repeats):
    samples = []
    for i in range(repeats):
        query = _SAMPLE_QUERIES[i % len(_SAMPLE_QUERIES)]
        t0 = time.monotonic()
        await processor.build_context_preface(
            message=query, session=_session(), use_web=False, use_rag=False,
            use_skills=False, memory_effort=effort,
        )
        samples.append(time.monotonic() - t0)
    return samples


async def _main(efforts, repeats):
    entries = load_fixture_store()
    mm = _FixtureMemoryManager(entries)
    processor = ChatProcessor(memory_manager=mm, personal_docs_manager=SimpleNamespace(rag_manager=None))

    print(f"Preface-build latency over {repeats} call(s) each, {len(entries)}-entry fixture store\n")
    print(f"{'effort':8s} {'min':>8s} {'mean':>8s} {'p95':>8s} {'max':>8s}")
    for effort in efforts:
        samples_s = await _time_effort(processor, effort, repeats)
        samples_ms = [s * 1000 for s in samples_s]
        print(f"{effort:8s} {min(samples_ms):8.1f} {statistics.mean(samples_ms):8.1f} "
              f"{_percentile(samples_ms, 95):8.1f} {max(samples_ms):8.1f}   (ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effort", choices=("low", "medium", "high"), default=None,
                         help="only time one effort level (default: all three)")
    parser.add_argument("--repeats", type=int, default=5, help="calls per effort level (default 5)")
    args = parser.parse_args()

    efforts = [args.effort] if args.effort else ["low", "medium", "high"]
    asyncio.run(_main(efforts, args.repeats))


if __name__ == "__main__":
    main()
