"""Unit tests for temporal.py — pure math, no external deps."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from cognitiveweave.storage.temporal import (
    ebbinghaus_decay,
    new_stability,
    days_since,
    recency_boost,
    recency_boost_from_iso,
    DEFAULT_STABILITY,
    STABILITY_BOOST,
    MIN_WEIGHT,
    DEFAULT_HALFLIFE,
)


class TestEbbinghausDecay:
    def test_zero_delta_returns_same_weight(self):
        assert ebbinghaus_decay(1.0, delta_days=0, stability=7.0) == 1.0

    def test_negative_delta_returns_same_weight(self):
        assert ebbinghaus_decay(0.8, delta_days=-5, stability=7.0) == 0.8

    def test_one_halflife_halves_approximately(self):
        # After Δt = stability days, weight should be w × e^-1 ≈ w × 0.368
        w = ebbinghaus_decay(1.0, delta_days=7.0, stability=7.0)
        assert w == pytest.approx(math.exp(-1), rel=1e-6)

    def test_higher_stability_decays_slower(self):
        fast = ebbinghaus_decay(1.0, delta_days=7, stability=7.0)
        slow = ebbinghaus_decay(1.0, delta_days=7, stability=30.0)
        assert slow > fast

    def test_result_bounded_by_min_weight(self):
        # Extreme decay should not go below MIN_WEIGHT
        w = ebbinghaus_decay(0.001, delta_days=1000, stability=0.1)
        assert w >= MIN_WEIGHT

    def test_zero_stability_uses_default(self):
        # stability=0 would cause division by zero — should use DEFAULT_STABILITY
        w = ebbinghaus_decay(1.0, delta_days=7, stability=0)
        expected = ebbinghaus_decay(1.0, delta_days=7, stability=DEFAULT_STABILITY)
        assert w == pytest.approx(expected)

    def test_chained_decay_matches_single_call(self):
        # Two half-cycles should equal one full cycle
        w_double = ebbinghaus_decay(
            ebbinghaus_decay(1.0, delta_days=3.5, stability=7.0),
            delta_days=3.5,
            stability=7.0,
        )
        w_single = ebbinghaus_decay(1.0, delta_days=7.0, stability=7.0)
        assert w_double == pytest.approx(w_single, rel=1e-6)


class TestNewStability:
    def test_adds_boost(self):
        result = new_stability(7.0)
        assert result == pytest.approx(7.0 + STABILITY_BOOST)

    def test_custom_boost(self):
        result = new_stability(10.0, boost=5.0)
        assert result == pytest.approx(15.0)

    def test_stability_grows_with_repeated_access(self):
        s = DEFAULT_STABILITY
        for _ in range(10):
            s = new_stability(s)
        assert s > DEFAULT_STABILITY + 5 * STABILITY_BOOST


class TestDaysSince:
    def test_now_returns_near_zero(self):
        now = datetime.now(timezone.utc)
        assert days_since(now) < 0.001

    def test_one_day_ago(self):
        one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        assert days_since(one_day_ago) == pytest.approx(1.0, abs=0.01)

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime.utcnow() - timedelta(hours=24)
        assert days_since(naive) == pytest.approx(1.0, abs=0.01)

    def test_future_datetime_returns_zero(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        assert days_since(future) == 0.0


class TestRecencyBoost:
    def test_created_now_returns_one(self):
        now = datetime.now(timezone.utc)
        boost = recency_boost(now, halflife_days=30)
        assert boost == pytest.approx(1.0, abs=0.001)

    def test_halflife_age_returns_1_over_e(self):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        boost = recency_boost(old, halflife_days=30)
        assert boost == pytest.approx(1 / math.e, rel=0.01)

    def test_older_nodes_score_lower(self):
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        old = datetime.now(timezone.utc) - timedelta(days=60)
        assert recency_boost(recent) > recency_boost(old)

    def test_boost_always_in_zero_one(self):
        very_old = datetime.now(timezone.utc) - timedelta(days=3650)
        assert 0 < recency_boost(very_old) <= 1.0

    def test_larger_halflife_decays_slower(self):
        age = datetime.now(timezone.utc) - timedelta(days=30)
        short = recency_boost(age, halflife_days=15)
        long_ = recency_boost(age, halflife_days=60)
        assert long_ > short


class TestRecencyBoostFromIso:
    def test_none_returns_one(self):
        assert recency_boost_from_iso(None) == 1.0

    def test_empty_string_returns_one(self):
        assert recency_boost_from_iso("") == 1.0

    def test_invalid_string_returns_one(self):
        assert recency_boost_from_iso("not-a-date") == 1.0

    def test_valid_iso_parsed_correctly(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        boost = recency_boost_from_iso(ts, halflife_days=30)
        assert boost == pytest.approx(1 / math.e, rel=0.05)

    def test_z_suffix_handled(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        boost = recency_boost_from_iso(ts)
        assert boost == pytest.approx(1.0, abs=0.01)
