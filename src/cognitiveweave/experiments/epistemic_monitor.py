"""EpistemicMonitor — population-level epistemic state computation.

Computes collective belief diversity after each experiment cycle and writes
the result to the shared Neo4j graph as a readable signal node. Agents read
this node at the start of their own cycle and adapt their behavior accordingly.

No agent calls this monitor directly. The AgentPopulation calls it once per
cycle, after all agents have written. Each agent then independently decides
how to respond to the entropy signal — no central authority coordinates them.

Entropy Metric
--------------
We measure diversity as the mean pairwise cosine distance between all agent
belief embeddings collected in the current cycle. This gives a value in [0, 1]:

  0.0 → all agents produced identical beliefs (total homogenization)
  0.5 → moderate diversity (healthy disagreement)
  1.0 → orthogonal beliefs (maximum divergence / chaos)

Thresholds (configurable):
  entropy < LOW  → homogenization risk → agents should explore
  entropy > HIGH → divergence risk     → agents should consolidate
  LOW ≤ entropy ≤ HIGH → healthy range, no adaptation pressure
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.telemetry import tracer

logger = logging.getLogger(__name__)

# Node id for the system state signal — agents query this
_STATE_NODE_ID = "system:epistemic_state"

# Default thresholds — can be overridden at construction time
DEFAULT_LOW_ENTROPY  = 0.15   # below → homogenization, agents explore
DEFAULT_HIGH_ENTROPY = 0.60   # above → divergence, agents consolidate


@dataclass
class EpistemicState:
    """Snapshot of population-level epistemic diversity at one cycle."""
    cycle: int
    entropy: float                        # mean pairwise cosine distance [0, 1]
    agent_count: int
    regime: str                           # "homogenizing" | "healthy" | "diverging"
    agent_entropies: dict[str, float] = field(default_factory=dict)  # per-agent drift


class EpistemicMonitor:
    """Computes and publishes epistemic state to the shared graph.

    Args:
        neo4j:        Shared Neo4j client.
        low_entropy:  Below this → homogenization warning.
        high_entropy: Above this → divergence warning.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        *,
        low_entropy: float = DEFAULT_LOW_ENTROPY,
        high_entropy: float = DEFAULT_HIGH_ENTROPY,
    ) -> None:
        self._neo4j = neo4j
        self._low = low_entropy
        self._high = high_entropy
        self._history: list[EpistemicState] = []

    # ------------------------------------------------------------------
    # Public API — called by AgentPopulation after each cycle
    # ------------------------------------------------------------------

    def observe(
        self,
        cycle: int,
        agent_beliefs: dict[str, str],   # agent_name → belief text
        faiss_encode_fn: Any,            # FAISSIndex.encode callable
    ) -> EpistemicState:
        """Compute entropy from current cycle's beliefs and publish to graph.

        Args:
            cycle:           Current cycle number.
            agent_beliefs:   Map of agent name → belief text written this cycle.
            faiss_encode_fn: Callable that returns embeddings for a list of texts.

        Returns:
            EpistemicState snapshot for this cycle.
        """
        with tracer.start_as_current_span("epistemic_monitor.observe") as span:
            span.set_attribute("cycle", cycle)
            span.set_attribute("agent_count", len(agent_beliefs))

            if len(agent_beliefs) < 2:
                state = EpistemicState(
                    cycle=cycle, entropy=0.0,
                    agent_count=len(agent_beliefs), regime="healthy",
                )
                self._history.append(state)
                return state

            names = list(agent_beliefs.keys())
            texts = [agent_beliefs[n] for n in names]
            embeddings = faiss_encode_fn(texts)  # shape (N, dim), L2-normalised

            entropy = self._mean_pairwise_distance(embeddings)
            regime  = self._classify(entropy)

            span.set_attribute("entropy", entropy)
            span.set_attribute("regime", regime)

            state = EpistemicState(
                cycle=cycle,
                entropy=entropy,
                agent_count=len(agent_beliefs),
                regime=regime,
            )
            self._history.append(state)
            self._write_to_graph(state)

            logger.info(
                "EpistemicMonitor cycle=%d entropy=%.3f regime=%s",
                cycle, entropy, regime,
            )
            return state

    def read_current_state(self) -> EpistemicState | None:
        """Return the most recently computed state (from history, no DB call)."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[EpistemicState]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mean_pairwise_distance(self, embeddings: np.ndarray) -> float:
        """Mean cosine distance across all agent pairs.

        Embeddings are L2-normalised → cosine similarity = dot product.
        Distance = 1 - similarity, clamped to [0, 1].
        """
        n = len(embeddings)
        if n < 2:
            return 0.0
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                dist = max(0.0, min(1.0, 1.0 - sim))
                distances.append(dist)
        return float(np.mean(distances))

    def _classify(self, entropy: float) -> str:
        if entropy < self._low:
            return "homogenizing"
        if entropy > self._high:
            return "diverging"
        return "healthy"

    def _write_to_graph(self, state: EpistemicState) -> None:
        """Upsert the system state node so agents can read it next cycle."""
        self._neo4j.upsert_node(
            _STATE_NODE_ID,
            content=f"System epistemic state: entropy={state.entropy:.3f} regime={state.regime}",
            metadata={
                "entropy":     state.entropy,
                "regime":      state.regime,
                "cycle":       state.cycle,
                "agent_count": state.agent_count,
                "cluster":     "system",
            },
        )


def read_epistemic_signal(neo4j: Neo4jClient) -> dict[str, Any]:
    """Read the current epistemic state from the graph.

    Called by SocietyAgent at the start of each cycle.
    Returns defaults if no state has been written yet.
    """
    node = neo4j.get_node(_STATE_NODE_ID)
    if node is None:
        return {"entropy": 0.5, "regime": "healthy"}
    return {
        "entropy": float(node.get("entropy", 0.5)),
        "regime":  str(node.get("regime", "healthy")),
        "cycle":   int(node.get("cycle", 0)),
    }
