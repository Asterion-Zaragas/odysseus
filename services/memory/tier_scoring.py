"""
tier_scoring.py

Pure math for memory tiers. No LLM, no IO — the only model-supplied input is
the entry's `generality` (0-3, judged at creation by the tagger and revisited
by the curator); everything else derives from usage counters and timestamps.

Tiers:
    0 core        — identity-level, always-relevant facts
    1 durable     — stable preferences, relationships, recurring context
    2 situational — project/time-bound, moderate generality
    3 archive     — conversation-specific residue, expiry candidates

Tier changes happen only in curator runs (never on the retrieval hot path) via
`next_tier`, which applies hysteresis and moves at most one tier per run so
entries can't flap. `initial_tier` maps a score straight to a tier for
brand-new entries. Pinning is a separate manual flag and never affects tiers.
"""

import math
import time
from typing import Dict, Optional

TIER_CORE = 0
TIER_DURABLE = 1
TIER_SITUATIONAL = 2
TIER_ARCHIVE = 3

TIER_NAMES = {
    TIER_CORE: "core",
    TIER_DURABLE: "durable",
    TIER_SITUATIONAL: "situational",
    TIER_ARCHIVE: "archive",
}

# Score = weighted sum of generality, usage frequency, and recency; all three
# components are normalized to [0, 1] so the score is too. Generality
# dominates on purpose: a core fact must stay high-tier even when rarely
# retrieved (asking the user's name every chat would be a bug, not a signal).
W_GENERALITY = 0.5
W_USAGE = 0.3
W_RECENCY = 0.2

# A memory injected ~this many times per 30 days saturates the usage signal.
_USAGE_SATURATION_PER_MONTH = 4.0

# Recency half-life: an unused memory loses half its recency signal per this
# many days.
_RECENCY_HALF_LIFE_DAYS = 30.0

# Score boundaries. initial_tier uses the plain boundary; next_tier uses the
# promote/demote pair around it (promote above, demote below) so an entry
# whose score hovers on a boundary keeps its tier instead of flapping nightly.
_INITIAL_BOUNDS = ((TIER_CORE, 0.70), (TIER_DURABLE, 0.45), (TIER_SITUATIONAL, 0.20))
_PROMOTE_INTO = {TIER_CORE: 0.75, TIER_DURABLE: 0.50, TIER_SITUATIONAL: 0.25}
_DEMOTE_BELOW = {TIER_DURABLE: 0.65, TIER_SITUATIONAL: 0.40, TIER_ARCHIVE: 0.15}


def usage_frequency(uses: int, age_days: float) -> float:
    """Normalized injections-per-month, saturating at _USAGE_SATURATION_PER_MONTH.

    Age is floored at one month so a memory used once on day 1 doesn't count
    as "30 uses/month".
    """
    uses = max(int(uses or 0), 0)
    months = max(float(age_days), 30.0) / 30.0
    return min((uses / months) / _USAGE_SATURATION_PER_MONTH, 1.0)


def recency_decay(last_used_at: float, now: Optional[float] = None) -> float:
    """Exponential decay on days since last use: 1.0 fresh, 0.5 per half-life."""
    now = time.time() if now is None else now
    if not last_used_at or last_used_at > now:
        return 0.0 if not last_used_at else 1.0
    days = (now - last_used_at) / 86400.0
    return math.pow(0.5, days / _RECENCY_HALF_LIFE_DAYS)


def tier_score(entry: Dict, now: Optional[float] = None) -> Optional[float]:
    """Relevance score in [0, 1], or None when the entry is untriaged
    (`generality` unset) — callers must then leave the tier alone.

    A never-used entry falls back to its creation timestamp for recency, so
    fresh memories start with full recency signal and age out naturally.
    """
    generality = entry.get("generality")
    if generality is None:
        return None
    now = time.time() if now is None else now

    g = min(max(float(generality), 0.0), 3.0) / 3.0
    created = float(entry.get("timestamp") or 0)
    age_days = max((now - created) / 86400.0, 0.0) if created else 0.0
    u = usage_frequency(entry.get("uses", 0), age_days)
    r = recency_decay(entry.get("last_used_at") or created, now)

    return W_GENERALITY * g + W_USAGE * u + W_RECENCY * r


def initial_tier(score: Optional[float]) -> int:
    """Tier for a brand-new entry, straight from its score (no hysteresis)."""
    if score is None:
        return TIER_SITUATIONAL
    for tier, bound in _INITIAL_BOUNDS:
        if score >= bound:
            return tier
    return TIER_ARCHIVE


def next_tier(current_tier: int, score: Optional[float]) -> int:
    """Curator-run tier update: hysteresis, at most one step per run.

    Promotion requires clearing the higher tier's promote threshold; demotion
    requires falling below the current tier's keep threshold. Scores between
    the two leave the tier unchanged.
    """
    current = min(max(int(current_tier), TIER_CORE), TIER_ARCHIVE)
    if score is None:
        return current

    above = current - 1
    if above >= TIER_CORE and score >= _PROMOTE_INTO[above]:
        return above

    below = current + 1
    if below <= TIER_ARCHIVE and score < _DEMOTE_BELOW[below]:
        return below

    return current
