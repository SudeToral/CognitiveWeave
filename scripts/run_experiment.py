"""CLI entry point for controlled A/B experiments.

Usage:
    # Run condition A (no regulation) — 10 runs × 20 cycles × 10 agents
    uv run python scripts/run_experiment.py --condition A

    # Run condition B (with regulation)
    uv run python scripts/run_experiment.py --condition B

    # Quick smoke test (2 runs × 5 cycles × 3 agents)
    uv run python scripts/run_experiment.py --condition A --n-runs 2 --n-cycles 5 --n-agents 3

    # Compare saved results
    uv run python scripts/run_experiment.py --compare experiments/runs/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")

from cognitiveweave.config.settings import Settings
from cognitiveweave.experiments.runner import (
    ConditionResult,
    ExperimentConfig,
    ExperimentRunner,
    compare_conditions,
)
from cognitiveweave.llm.ollama_client import OllamaClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CognitiveWeave experiment runner")
    sub = p.add_subparsers(dest="cmd")

    # ── run ────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Execute one experiment condition")
    run_p.add_argument("--condition",  default="A",   help="Condition label (A, B, C_bridge …)")
    run_p.add_argument("--regulation", action="store_true", help="Enable entropy regulation")
    run_p.add_argument("--n-runs",     type=int, default=10)
    run_p.add_argument("--n-cycles",   type=int, default=20)
    run_p.add_argument("--n-agents",        type=int, default=10)
    run_p.add_argument("--bandwidth",       type=int, default=5)
    run_p.add_argument("--smart-regulation", action="store_true",
                       help="Condition C: seed migration instead of bandwidth cut")
    run_p.add_argument("--output-dir", default="experiments/runs")

    # ── compare ────────────────────────────────────────────────────────────
    cmp_p = sub.add_parser("compare", help="Compare two saved condition summaries")
    cmp_p.add_argument("file_a", help="Path to condition A summary JSON")
    cmp_p.add_argument("file_b", help="Path to condition B summary JSON")

    # ── abc ────────────────────────────────────────────────────────────────
    ab_p = sub.add_parser("abc", help="Run A, B, C conditions then compare all")
    ab_p.add_argument("--n-runs",   type=int, default=10)
    ab_p.add_argument("--n-cycles", type=int, default=20)
    ab_p.add_argument("--n-agents", type=int, default=10)
    ab_p.add_argument("--output-dir", default="experiments/runs")

    # ── ab (legacy) ────────────────────────────────────────────────────────
    ab_p2 = sub.add_parser("ab", help="Run A and B conditions then compare")
    ab_p2.add_argument("--n-runs",   type=int, default=10)
    ab_p2.add_argument("--n-cycles", type=int, default=20)
    ab_p2.add_argument("--n-agents", type=int, default=10)
    ab_p2.add_argument("--output-dir", default="experiments/runs")

    return p


def make_runner(output_dir: str) -> ExperimentRunner:
    s = Settings()
    neo4j = Neo4jClient(s.neo4j)
    neo4j.connect()
    neo4j.create_constraints()

    faiss = FAISSIndex(s.faiss)
    faiss.load_model()
    from pathlib import Path as _P
    if _P(s.faiss.index_path).exists():
        faiss.load()
    else:
        faiss.build_index()

    ollama = OllamaClient(s.ollama)
    return ExperimentRunner(neo4j, faiss, ollama, output_dir=output_dir)


def print_result(result: ConditionResult) -> None:
    result.compute_statistics()
    print(f"\n{'═'*60}")
    print(f"  Condition {result.config.condition}  "
          f"(regulation={'ON' if result.config.regulation else 'OFF'})")
    print(f"  {result.config.n_runs} runs × {result.config.n_cycles} cycles × "
          f"{result.config.n_agents} agents")
    print(f"{'─'*60}")
    print(f"  Final entropy :  {result.final_entropy_stats}")
    print(f"  Isolation rate:  {result.isolation_stats}")
    print(f"  Cross-pol total: {result.cross_pol_stats}")
    if result.entropy_ts:
        slope = result.entropy_ts.slope
        p     = result.entropy_ts.slope_p_value
        trend = "↑ rising" if slope > 0 else "↓ falling"
        print(f"  Entropy trend :  slope={slope:.4f}  p={p:.4f}  {trend}")
    print(f"{'═'*60}\n")


def cmd_run(args: argparse.Namespace) -> None:
    smart = getattr(args, "smart_regulation", False)
    config = ExperimentConfig(
        condition=args.condition,
        regulation=args.regulation,
        smart_regulation=smart,
        n_runs=args.n_runs,
        n_cycles=args.n_cycles,
        n_agents=args.n_agents,
        bandwidth=args.bandwidth,
    )
    runner = make_runner(args.output_dir)
    result = runner.run(config)
    print_result(result)


def cmd_abc(args: argparse.Namespace) -> None:
    """Run all three conditions and print a full comparison table."""
    runner = make_runner(args.output_dir)

    configs = [
        ExperimentConfig("A", regulation=False, smart_regulation=False,
                         n_runs=args.n_runs, n_cycles=args.n_cycles, n_agents=args.n_agents),
        ExperimentConfig("B", regulation=True,  smart_regulation=False,
                         n_runs=args.n_runs, n_cycles=args.n_cycles, n_agents=args.n_agents),
        ExperimentConfig("C", regulation=True,  smart_regulation=True,
                         n_runs=args.n_runs, n_cycles=args.n_cycles, n_agents=args.n_agents),
    ]
    results = []
    for cfg in configs:
        print(f"\n{'▶'*3} Running Condition {cfg.condition} ...")
        r = runner.run(cfg)
        print_result(r)
        results.append(r)

    # Pairwise comparisons
    labels = [("A", "B"), ("A", "C"), ("B", "C")]
    for (la, lb), (ra, rb) in zip(labels, [
        (results[0], results[1]),
        (results[0], results[2]),
        (results[1], results[2]),
    ], strict=True):
        comparisons = compare_conditions(ra, rb)
        print(f"\n{'═'*60}")
        print(f"  {la} vs {lb} — Statistical Comparison")
        print(f"{'─'*60}")
        for metric, cmp in comparisons.items():
            print(f"  {metric:<22} {cmp}")
    print(f"{'═'*60}\n")


def cmd_ab(args: argparse.Namespace) -> None:
    runner = make_runner(args.output_dir)

    config_a = ExperimentConfig(
        condition="A", regulation=False,
        n_runs=args.n_runs, n_cycles=args.n_cycles, n_agents=args.n_agents,
    )
    config_b = ExperimentConfig(
        condition="B", regulation=True,
        n_runs=args.n_runs, n_cycles=args.n_cycles, n_agents=args.n_agents,
    )

    logger.info("Running condition A (baseline)...")
    result_a = runner.run(config_a)
    print_result(result_a)

    logger.info("Running condition B (regulated)...")
    result_b = runner.run(config_b)
    print_result(result_b)

    # Statistical comparison
    comparisons = compare_conditions(result_a, result_b)
    print(f"\n{'═'*60}")
    print("  A vs B — Statistical Comparison")
    print(f"{'─'*60}")
    for metric, cmp in comparisons.items():
        print(f"  {metric:<22} {cmp}")
    print(f"{'═'*60}\n")


def cmd_compare(args: argparse.Namespace) -> None:
    from cognitiveweave.experiments.statistics import welch_t_test, run_stats

    with open(args.file_a) as f:
        data_a = json.load(f)
    with open(args.file_b) as f:
        data_b = json.load(f)

    def extract_final_entropy(data: dict) -> list[float]:
        return [r["cycles"][-1]["entropy"] for r in data["runs"] if r["cycles"]]

    def extract_isolation(data: dict) -> list[float]:
        return [
            sum(c["isolation"] for c in r["cycles"]) / len(r["cycles"])
            for r in data["runs"] if r["cycles"]
        ]

    fe_a = extract_final_entropy(data_a)
    fe_b = extract_final_entropy(data_b)
    iso_a = extract_isolation(data_a)
    iso_b = extract_isolation(data_b)

    print(f"\nComparing {args.file_a} vs {args.file_b}\n")
    print(f"Final entropy   A: {run_stats(fe_a)}")
    print(f"Final entropy   B: {run_stats(fe_b)}")
    print(f"Test: {welch_t_test(fe_a, fe_b)}\n")
    print(f"Isolation rate  A: {run_stats(iso_a)}")
    print(f"Isolation rate  B: {run_stats(iso_b)}")
    print(f"Test: {welch_t_test(iso_a, iso_b)}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "abc":
        cmd_abc(args)
    elif args.cmd == "ab":
        cmd_ab(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
