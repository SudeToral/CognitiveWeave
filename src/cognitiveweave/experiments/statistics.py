"""Statistical analysis for epistemic regulation experiments.

Provides:
  - Per-run summary statistics (mean, std, 95% CI)
  - Welch's t-test for between-condition comparison
  - Cohen's d effect size
  - Entropy time-series aggregation across runs
  - Linear trend detection (slope over cycles)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    """Summary statistics for a scalar outcome across N runs."""
    mean: float
    std: float
    ci_lower: float   # 95% confidence interval lower bound
    ci_upper: float   # 95% confidence interval upper bound
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f}  95% CI [{self.ci_lower:.3f}, {self.ci_upper:.3f}]  n={self.n}"


@dataclass
class ComparisonResult:
    """Result of a two-sample comparison between conditions A and B."""
    mean_a: float
    mean_b: float
    t_stat: float
    p_value: float
    cohen_d: float
    effect_size: str   # "negligible" | "small" | "medium" | "large"
    significant: bool  # p < 0.05

    def __str__(self) -> str:
        sig = "✓ significant" if self.significant else "✗ not significant"
        return (
            f"A={self.mean_a:.3f}  B={self.mean_b:.3f}  "
            f"t={self.t_stat:.3f}  p={self.p_value:.4f}  "
            f"d={self.cohen_d:.3f} ({self.effect_size})  {sig}"
        )


@dataclass
class TimeSeriesStats:
    """Aggregated entropy time series across multiple runs."""
    cycles: list[int]
    mean_entropy: list[float]
    std_entropy: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    # Linear trend: positive = growing, negative = collapsing
    slope: float
    slope_p_value: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_stats(values: Sequence[float]) -> RunStats:
    """Compute summary statistics for a list of scalar outcomes (one per run)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        m = float(arr[0]) if n == 1 else 0.0
        return RunStats(mean=m, std=0.0, ci_lower=m, ci_upper=m, n=n)

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    sem  = std / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * sem
    return RunStats(
        mean=mean, std=std,
        ci_lower=mean - margin, ci_upper=mean + margin,
        n=n,
    )


def welch_t_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> ComparisonResult:
    """Welch's t-test (unequal variances) + Cohen's d effect size."""
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)

    # Pooled Cohen's d
    std_a = float(np.std(a, ddof=1))
    std_b = float(np.std(b, ddof=1))
    pooled = np.sqrt((std_a**2 + std_b**2) / 2)
    d = float(abs(float(np.mean(a)) - float(np.mean(b))) / pooled) if pooled > 0 else 0.0

    if d < 0.20:
        effect = "negligible"
    elif d < 0.50:
        effect = "small"
    elif d < 0.80:
        effect = "medium"
    else:
        effect = "large"

    return ComparisonResult(
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        t_stat=float(t_stat),
        p_value=float(p_value),
        cohen_d=d,
        effect_size=effect,
        significant=bool(p_value < 0.05),
    )


def aggregate_time_series(
    entropy_per_run: list[list[float]],
) -> TimeSeriesStats:
    """Aggregate entropy[cycle] across multiple runs.

    Args:
        entropy_per_run: list of N lists, each of length T (cycles).
                         All lists must be the same length.

    Returns:
        TimeSeriesStats with mean/std/CI per cycle + linear trend.
    """
    arr = np.array(entropy_per_run, dtype=float)  # shape (N_runs, T_cycles)
    n_runs, n_cycles = arr.shape
    cycles = list(range(n_cycles))

    means = arr.mean(axis=0).tolist()
    stds  = arr.std(axis=0, ddof=1).tolist()

    t_crit = float(scipy_stats.t.ppf(0.975, df=n_runs - 1)) if n_runs > 1 else 1.96
    ci_lower = [m - t_crit * s / np.sqrt(n_runs) for m, s in zip(means, stds, strict=True)]
    ci_upper = [m + t_crit * s / np.sqrt(n_runs) for m, s in zip(means, stds, strict=True)]

    # Linear trend via OLS — guard against constant series (all zeros → NaN)
    import math
    if len(set(means)) < 2:
        slope, p = 0.0, 1.0
    else:
        _slope, _intercept, _r, _p, _se = scipy_stats.linregress(cycles, means)
        slope = float(_slope)
        p = float(_p) if not math.isnan(float(_p)) else 1.0

    return TimeSeriesStats(
        cycles=cycles,
        mean_entropy=means,
        std_entropy=stds,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        slope=slope,
        slope_p_value=p,
    )


def isolation_rate(cross_pollination_per_agent: dict[str, int]) -> float:
    """Fraction of agents with zero cross-pollination (completely isolated)."""
    if not cross_pollination_per_agent:
        return 0.0
    isolated = sum(1 for v in cross_pollination_per_agent.values() if v == 0)
    return isolated / len(cross_pollination_per_agent)
