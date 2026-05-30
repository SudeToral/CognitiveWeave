"""ExperimentRunner — controlled A/B experiment execution.

Runs multiple independent replications of an experiment condition
(regulation ON vs OFF) and collects per-cycle metrics for statistical
analysis. Each run starts from a clean experiment state.

Usage:
    config_a = ExperimentConfig(condition="A", regulation=False, n_runs=10, n_cycles=20)
    config_b = ExperimentConfig(condition="B", regulation=True,  n_runs=10, n_cycles=20)

    runner = ExperimentRunner(neo4j, faiss, ollama)
    result_a = runner.run(config_a)
    result_b = runner.run(config_b)

    comparison = compare_conditions(result_a, result_b)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cognitiveweave.experiments.agent import SocietyAgent
from cognitiveweave.experiments.epistemic_monitor import EpistemicMonitor, find_bridge_nodes
from cognitiveweave.experiments.observer import ExperimentObserver
from cognitiveweave.experiments.population import AgentPopulation
from cognitiveweave.experiments.statistics import (
    ComparisonResult,
    TimeSeriesStats,
    aggregate_time_series,
    isolation_rate,
    run_stats,
    welch_t_test,
)
from cognitiveweave.llm.ollama_client import OllamaClient
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10-Agent Pool
# ---------------------------------------------------------------------------
# Agents span four domain clusters in the knowledge graph.
# Seed positions are chosen to create a mix of bridge, peripheral,
# and isolated agents — matching the structural diversity described
# in the experimental design.

AGENT_POOL: list[dict[str, str]] = [
    # Bridge agents (seed sits between domain clusters)
    {"name": "A1", "interest": "forgetting curves and long-term memory retention",            "seed": "c2_ebbinghaus",        "color": "#fab387"},
    {"name": "A2", "interest": "spaced repetition and optimal learning schedules",            "seed": "c2_spacing",           "color": "#f9e2af"},

    # Neuroscience cluster
    {"name": "A3", "interest": "hippocampus role in memory consolidation during sleep",       "seed": "c3_hippocampus",       "color": "#89b4fa"},
    {"name": "A4", "interest": "prefrontal cortex and working memory capacity",               "seed": "c3_pfc",               "color": "#74c7ec"},
    {"name": "A5", "interest": "synaptic plasticity and LTP mechanisms",                      "seed": "c1_ltp",               "color": "#89dceb"},

    # Memory biology cluster
    {"name": "A6", "interest": "memory consolidation and protein synthesis",                  "seed": "c1_consolidation",     "color": "#a6e3a1"},
    {"name": "A7", "interest": "engram cells and memory trace formation",                     "seed": "c1_engram",            "color": "#94e2d5"},

    # AI/ML cluster (structurally isolated from neuroscience)
    {"name": "A8", "interest": "attention mechanisms in transformer architectures",           "seed": "c4_attention",         "color": "#cba6f7"},
    {"name": "A9", "interest": "retrieval augmented generation and knowledge bases",          "seed": "c4_knowledge_graph",   "color": "#f38ba8"},

    # Cognitive science cluster (peripheral)
    {"name": "A10", "interest": "metacognition and self-regulated learning strategies",       "seed": "c6_metacognition",     "color": "#eba0ac"},
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Full specification of one experiment condition."""
    condition: str              # "A" (baseline) | "B" (regulated) | "C" (smart)
    regulation: bool            # entropy feedback loop on/off
    smart_regulation: bool = False  # Condition C: seed migration instead of bandwidth cut
    n_runs: int      = 10       # independent replications
    n_cycles: int    = 20       # cycles per run
    n_agents: int    = 10       # how many agents from AGENT_POOL to use
    bandwidth: int   = 5        # base retrieval bandwidth per agent
    shared_faiss: bool = True   # agents share the vector index


@dataclass
class CycleRecord:
    """Metrics captured at the end of one cycle within one run."""
    cycle: int
    entropy: float
    regime: str
    cross_pollination: dict[str, int]   # agent → count
    node_counts: dict[str, int]
    avg_confidence: dict[str, float]
    isolation: float                     # fraction of agents with 0 cross-pol


@dataclass
class RunResult:
    """All cycles from a single run."""
    run_id: int
    config: ExperimentConfig
    cycles: list[CycleRecord] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def entropy_series(self) -> list[float]:
        return [c.entropy for c in self.cycles]

    @property
    def final_entropy(self) -> float:
        return self.cycles[-1].entropy if self.cycles else 0.0

    @property
    def mean_isolation(self) -> float:
        return float(np.mean([c.isolation for c in self.cycles])) if self.cycles else 0.0

    @property
    def total_cross_pollination(self) -> int:
        if not self.cycles:
            return 0
        last = self.cycles[-1].cross_pollination
        return sum(last.values())


@dataclass
class ConditionResult:
    """Aggregated results across all runs for one condition."""
    config: ExperimentConfig
    runs: list[RunResult] = field(default_factory=list)

    # Computed after all runs complete
    entropy_ts: TimeSeriesStats | None = None
    final_entropy_stats: Any = None      # RunStats
    isolation_stats: Any = None          # RunStats
    cross_pol_stats: Any = None          # RunStats

    def compute_statistics(self) -> None:
        """Aggregate metrics across runs."""
        if not self.runs:
            return

        # Pad/trim entropy series to same length
        min_len = min(len(r.entropy_series) for r in self.runs)
        series = [r.entropy_series[:min_len] for r in self.runs]

        self.entropy_ts         = aggregate_time_series(series)
        self.final_entropy_stats = run_stats([r.final_entropy    for r in self.runs])
        self.isolation_stats     = run_stats([r.mean_isolation    for r in self.runs])
        self.cross_pol_stats     = run_stats([r.total_cross_pollination for r in self.runs])

    def to_dict(self) -> dict:
        self.compute_statistics()
        return {
            "condition":     self.config.condition,
            "regulation":    self.config.regulation,
            "n_runs":        self.config.n_runs,
            "n_cycles":      self.config.n_cycles,
            "n_agents":      self.config.n_agents,
            "final_entropy": str(self.final_entropy_stats),
            "isolation":     str(self.isolation_stats),
            "cross_pol":     str(self.cross_pol_stats),
            "entropy_trend_slope":   self.entropy_ts.slope if self.entropy_ts else None,
            "entropy_trend_p":       self.entropy_ts.slope_p_value if self.entropy_ts else None,
            "runs": [
                {
                    "run_id":    r.run_id,
                    "wall_time": r.wall_time_s,
                    "cycles":    [
                        {
                            "cycle":            c.cycle,
                            "entropy":          c.entropy,
                            "regime":           c.regime,
                            "isolation":        c.isolation,
                            "cross_pollination": c.cross_pollination,
                        }
                        for c in r.cycles
                    ],
                }
                for r in self.runs
            ],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Executes controlled experiment conditions with full metric collection.

    Args:
        neo4j:   Shared Neo4j client.
        faiss:   Shared FAISS index (pre-loaded with base knowledge).
        ollama:  LLM client for belief synthesis.
        output_dir: Where to save JSON results after each run.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        faiss: FAISSIndex,
        ollama: OllamaClient,
        *,
        output_dir: str = "experiments/runs",
    ) -> None:
        self._neo4j = neo4j
        self._faiss = faiss
        self._ollama = ollama
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, config: ExperimentConfig) -> ConditionResult:
        """Run all replications for one condition. Saves results after each run."""
        logger.info(
            "Condition %s: %d runs × %d cycles × %d agents  regulation=%s",
            config.condition, config.n_runs, config.n_cycles,
            config.n_agents, config.regulation,
        )
        result = ConditionResult(config=config)

        for run_id in range(1, config.n_runs + 1):
            logger.info("  Run %d/%d ...", run_id, config.n_runs)
            t0 = time.monotonic()
            run = self._single_run(run_id, config)
            run.wall_time_s = time.monotonic() - t0
            result.runs.append(run)
            logger.info(
                "  Run %d done  final_entropy=%.3f  t=%.1fs",
                run_id, run.final_entropy, run.wall_time_s,
            )
            # Save incrementally — safe against crashes mid-experiment
            self._save_run(run, config)

        result.compute_statistics()
        self._save_condition(result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_run(self, run_id: int, config: ExperimentConfig) -> RunResult:
        """One isolated replication: clean state → agents → cycles → metrics."""
        # 1 — Clean previous experiment nodes from Neo4j
        observer = ExperimentObserver(self._neo4j)
        observer.cleanup()

        # 2 — Build agents from pool
        pool = AGENT_POOL[:config.n_agents]
        retriever = HybridRetriever(self._faiss, self._neo4j)
        agents = [
            SocietyAgent(
                name=p["name"],
                interest=p["interest"],
                retriever=retriever,
                neo4j=self._neo4j,
                ollama=self._ollama,
                bandwidth=config.bandwidth,
                seed_node_id=p["seed"],
                faiss=self._faiss if config.shared_faiss else None,
            )
            for p in pool
        ]

        # 3 — Bridge nodes (Condition C only)
        bridge_seeds: list[str] = []
        if config.smart_regulation:
            bridge_seeds = find_bridge_nodes(self._neo4j)
            logger.info("  Bridge nodes detected: %s", bridge_seeds)

        # Rebuild agents with bridge seeds if Condition C
        if config.smart_regulation:
            agents = [
                SocietyAgent(
                    name=p["name"],
                    interest=p["interest"],
                    retriever=retriever,
                    neo4j=self._neo4j,
                    ollama=self._ollama,
                    bandwidth=config.bandwidth,
                    seed_node_id=p["seed"],
                    faiss=self._faiss if config.shared_faiss else None,
                    bridge_seeds=bridge_seeds,
                )
                for p in pool
            ]

        # 4 — Build population
        # Condition A: monitor runs in measure-only mode (write_to_graph=False)
        # Condition B: monitor writes signal, agents adapt bandwidth
        # Condition C: monitor writes signal, agents migrate seed to bridge nodes
        measure_monitor = EpistemicMonitor(self._neo4j, write_to_graph=False)
        active_monitor  = EpistemicMonitor(self._neo4j, write_to_graph=True) \
                          if (config.regulation or config.smart_regulation) else None
        population = AgentPopulation(agents, monitor=active_monitor)

        # 4 — Run cycles
        run = RunResult(run_id=run_id, config=config)
        agent_names = [p["name"] for p in pool]

        for cycle_num in range(config.n_cycles):
            result_cycle = population.run_cycle(cycle_num)
            snap = observer.snapshot(cycle_num, agent_names)

            # Always measure entropy (even in Condition A)
            beliefs = {
                w.agent: w.belief
                for w in result_cycle.successful_writes if w.belief
            }
            iso = isolation_rate(snap.cross_reads)
            if beliefs:
                obs_state = measure_monitor.observe(
                    cycle_num, beliefs, self._faiss.encode
                )
                entropy = obs_state.entropy
                if config.regulation or config.smart_regulation:
                    regime = obs_state.regime
                    # Write isolation signal so agents can read it next cycle
                    if active_monitor is not None:
                        active_monitor._write_state_to_graph(obs_state, isolation_rate=iso)
                else:
                    regime = "unregulated"
            else:
                entropy = 0.0
                regime  = "unregulated"
                iso     = 0.0

            record = CycleRecord(
                cycle=cycle_num,
                entropy=entropy,
                regime=regime,
                cross_pollination=dict(snap.cross_reads),
                node_counts=dict(snap.node_counts),
                avg_confidence=dict(snap.avg_confidence),
                isolation=iso,
            )
            run.cycles.append(record)

        return run

    def _save_run(self, run: RunResult, config: ExperimentConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self._output_dir / f"{ts}_cond{config.condition}_run{run.run_id:02d}.json"
        data = {
            "condition": config.condition,
            "run_id": run.run_id,
            "wall_time_s": run.wall_time_s,
            "cycles": [
                {
                    "cycle": c.cycle, "entropy": c.entropy, "regime": c.regime,
                    "isolation": c.isolation,
                    "cross_pollination": c.cross_pollination,
                }
                for c in run.cycles
            ],
        }
        fname.write_text(json.dumps(data, indent=2))

    def _save_condition(self, result: ConditionResult) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self._output_dir / f"{ts}_condition_{result.config.condition}_summary.json"
        fname.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Saved condition summary → %s", fname)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_conditions(
    a: ConditionResult,
    b: ConditionResult,
) -> dict[str, ComparisonResult]:
    """Run statistical tests comparing two conditions on key outcomes.

    Returns a dict mapping metric name → ComparisonResult.
    """
    a.compute_statistics()
    b.compute_statistics()

    return {
        "final_entropy": welch_t_test(
            [r.final_entropy        for r in a.runs],
            [r.final_entropy        for r in b.runs],
        ),
        "mean_isolation": welch_t_test(
            [r.mean_isolation       for r in a.runs],
            [r.mean_isolation       for r in b.runs],
        ),
        "cross_pollination": welch_t_test(
            [r.total_cross_pollination for r in a.runs],
            [r.total_cross_pollination for r in b.runs],
        ),
    }
