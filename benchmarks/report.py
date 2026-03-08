"""Benchmark report formatter — prints results to stdout and optionally to JSON.

Usage:
    uv run python -m benchmarks.report
"""
from __future__ import annotations

import json
import sys
from typing import TextIO

from benchmarks.runner import BenchmarkReport


def print_table(reports: list[BenchmarkReport], out: TextIO = sys.stdout) -> None:
    """Print a markdown-compatible results table."""
    header = f"{'Strategy':<20} {'MRR@10':>8} {'NDCG@10':>9} {'Latency(ms)':>12}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for r in sorted(reports, key=lambda x: x.mean_mrr, reverse=True):
        out.write(
            f"{r.strategy:<20} {r.mean_mrr:>8.4f} {r.mean_ndcg:>9.4f} "
            f"{r.mean_latency_ms:>12.2f}\n"
        )
    out.write("\n")


def print_per_query(reports: list[BenchmarkReport], out: TextIO = sys.stdout) -> None:
    """Print per-query breakdown for each strategy."""
    for report in reports:
        out.write(f"Strategy: {report.strategy}\n")
        for qr in report.per_query:
            out.write(
                f"  {qr.query_id}  MRR={qr.mrr:.3f}  NDCG={qr.ndcg:.3f}  "
                f"retrieved={qr.retrieved_ids[:5]}\n"
            )
        out.write("\n")


def to_json(reports: list[BenchmarkReport]) -> str:
    data = [
        {
            "strategy": r.strategy,
            "mean_mrr": r.mean_mrr,
            "mean_ndcg": r.mean_ndcg,
            "mean_latency_ms": r.mean_latency_ms,
            "per_query": [
                {
                    "query_id": qr.query_id,
                    "mrr": qr.mrr,
                    "ndcg": qr.ndcg,
                    "latency_ms": qr.latency_ms,
                    "retrieved_ids": qr.retrieved_ids,
                }
                for qr in r.per_query
            ],
        }
        for r in reports
    ]
    return json.dumps(data, indent=2)


def improvement_over_baseline(
    reports: list[BenchmarkReport],
    baseline: str = "faiss_only",
    target: str = "hybrid_temporal",
) -> dict[str, float]:
    """Compute percentage improvement of target over baseline."""
    base = next((r for r in reports if r.strategy == baseline), None)
    tgt = next((r for r in reports if r.strategy == target), None)
    if not base or not tgt:
        return {}
    return {
        "mrr_improvement_pct": (tgt.mean_mrr - base.mean_mrr) / max(base.mean_mrr, 1e-9) * 100,
        "ndcg_improvement_pct": (tgt.mean_ndcg - base.mean_ndcg) / max(base.mean_ndcg, 1e-9) * 100,
        "latency_overhead_pct": (tgt.mean_latency_ms - base.mean_latency_ms) / max(base.mean_latency_ms, 1e-9) * 100,
    }
