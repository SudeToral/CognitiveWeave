"""Temporal scoring functions for CognitiveWeave.

Two independent concerns live here:

1. Ebbinghaus edge decay  — how fast a graph edge weakens over time.
2. Recency boost          — how much fresher nodes are preferred in retrieval.

Both are pure functions with no side effects, so they're trivially testable
and can be imported anywhere without pulling in storage dependencies.

Ebbinghaus model
----------------
The forgetting curve (Ebbinghaus, 1885) describes retention R as:

    R = e^(-t / S)

where t is elapsed time and S is the "stability" of the memory.
We apply this to edge weights:

    w_new = w_old × e^(-Δt / stability)

Stability starts at DEFAULT_STABILITY (days) for every new edge and
increases by STABILITY_BOOST each time the edge is accessed during
retrieval — implementing spaced-repetition: the more an edge is used,
the slower it decays.

Recency boost
-------------
When ranking retrieval results, newer nodes are preferred:

    boost = e^(-days_since_creation / halflife)

halflife=30 means a 30-day-old node scores ×0.37 relative to a new node.
This is applied *after* RRF fusion so it doesn't distort the rank signal,
only the final ordering.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

# --- Decay constants -------------------------------------------------------

DEFAULT_STABILITY: float = 7.0     # days — a new edge decays like a 1-week memory
STABILITY_BOOST: float = 2.0       # added to stability each time edge is retrieved
MIN_WEIGHT: float = 1e-6           # floor to avoid true zero (kept for prune threshold)

# --- Recency constants ------------------------------------------------------

DEFAULT_HALFLIFE: float = 30.0     # days — node at 30 days scores ×0.37 vs today


# ---------------------------------------------------------------------------
# Ebbinghaus decay
# ---------------------------------------------------------------------------

def ebbinghaus_decay(weight: float, delta_days: float, stability: float) -> float:
    """Apply Ebbinghaus forgetting curve to an edge weight.

    Args:
        weight:      Current edge weight (0, 1].
        delta_days:  Days elapsed since last decay application.
        stability:   Edge stability (higher = slower decay).

    Returns:
        New weight after decay. Always >= MIN_WEIGHT.
    """
    if delta_days <= 0:
        return weight
    if stability <= 0:
        stability = DEFAULT_STABILITY
    decayed = weight * math.exp(-delta_days / stability)
    return max(decayed, MIN_WEIGHT)


def new_stability(current_stability: float, boost: float = STABILITY_BOOST) -> float:
    """Increase stability when an edge is accessed (spaced repetition effect)."""
    return current_stability + boost


def days_since(ts: datetime) -> float:
    """Return fractional days between ts and now (UTC)."""
    now = datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max((now - ts).total_seconds() / 86_400, 0.0)


# ---------------------------------------------------------------------------
# Recency boost
# ---------------------------------------------------------------------------

def recency_boost(created_at: datetime, halflife_days: float = DEFAULT_HALFLIFE) -> float:
    """Score multiplier [0, 1] based on node age.

    A node created today returns 1.0.
    A node created `halflife_days` ago returns ~0.37 (1/e).

    Args:
        created_at:    UTC datetime when the node was first ingested.
        halflife_days: Controls decay speed. Larger = slower.

    Returns:
        Float in (0, 1].
    """
    delta = days_since(created_at)
    return math.exp(-delta / halflife_days)


def recency_boost_from_iso(iso_str: str | None, halflife_days: float = DEFAULT_HALFLIFE) -> float:
    """Convenience wrapper — parses ISO 8601 string from Neo4j/metadata.

    Returns 1.0 if iso_str is None or unparseable (no penalty for missing data).
    """
    if not iso_str:
        return 1.0
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return recency_boost(dt, halflife_days)
    except (ValueError, AttributeError):
        return 1.0
