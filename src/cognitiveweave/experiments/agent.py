"""SocietyAgent — a single participant in the belief dynamics experiment.

Each agent:
- Has a name, an interest topic, and a communication bandwidth limit.
- Each cycle: reads up to `bandwidth` nodes from the shared graph,
  synthesizes a belief via Ollama, and writes it back.
- Tags every write with its name, cycle number, and interest so the
  observer can reconstruct the belief history per agent.

Agents never call each other directly. All coordination happens through
the shared Neo4j graph — the same indirect communication channel that
makes emergent belief dynamics possible.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from cognitiveweave.experiments.epistemic_monitor import read_epistemic_signal
from cognitiveweave.llm.ollama_client import OllamaClient, _parse_json
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.telemetry import (
    agent_cycle_duration,
    cosine_distance,
    cross_pollination_counter,
    tracer,
)
from cognitiveweave.telemetry import (
    belief_drift as belief_drift_metric,
)

logger = logging.getLogger(__name__)

_SYNTHESIZE_SYSTEM = (
    "You are an agent forming a belief about a topic based on evidence. "
    "Given a list of knowledge fragments, synthesize ONE concise belief "
    "statement (1-2 sentences). Be direct — do not hedge unnecessarily. "
    "Respond ONLY with JSON: "
    "{\"belief\": <string>, \"confidence\": <float 0.0-1.0>}. "
    "No markdown, no extra text."
)


@dataclass
class AgentMemory:
    """Tracks what this agent has written to the shared graph."""
    node_ids: list[str] = field(default_factory=list)
    # cycle_num -> list of node ids written that cycle
    cycle_writes: dict[int, list[str]] = field(default_factory=dict)
    # last belief embedding vector — used to compute drift between cycles
    last_embedding: list[float] | None = None
    # cumulative drift across all cycles (sum of per-cycle cosine distances)
    total_drift: float = 0.0


@dataclass
class CycleWrite:
    agent: str
    cycle: int
    node_id: str | None
    belief: str
    confidence: float
    read_from: list[str]


class SocietyAgent:
    """A participant in the emergent epistemology experiment.

    Args:
        name:        Unique identifier for this agent (shown in UI).
        interest:    The topic this agent is curious about. Used as the
                     retrieval query each cycle.
        retriever:   Shared HybridRetriever (FAISS + Neo4j).
        neo4j:       Shared Neo4j client for writes.
        ollama:      Ollama client for belief synthesis.
        bandwidth:   Max nodes to read per cycle (simulates limited attention).
        seed_node_id: Optional starting node for graph traversal.
    """

    def __init__(
        self,
        name: str,
        interest: str,
        retriever: HybridRetriever,
        neo4j: Neo4jClient,
        ollama: OllamaClient,
        *,
        bandwidth: int = 5,
        seed_node_id: str | None = None,
        faiss: FAISSIndex | None = None,
        bridge_seeds: list[str] | None = None,
    ) -> None:
        self.name = name
        self.interest = interest
        self.bandwidth = bandwidth
        self._base_bandwidth = bandwidth
        self._seed = seed_node_id
        self._original_seed = seed_node_id
        self._retriever = retriever
        self._neo4j = neo4j
        self._ollama = ollama
        self._faiss = faiss
        self._bridge_seeds = bridge_seeds or []  # Condition C: bridge nodes to migrate to
        self._cycles_isolated = 0               # consecutive cycles with 0 cross-pol
        self.memory = AgentMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cycle(self, cycle_num: int) -> CycleWrite:
        """Run one experiment cycle. Returns a record of what was written."""
        t0 = time.monotonic()
        attrs = {"agent": self.name}

        # Read system entropy and adapt before executing
        self._adapt_to_epistemic_state(cycle_num)

        with tracer.start_as_current_span("agent.cycle") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("agent.interest", self.interest[:120])
            span.set_attribute("cycle.num", cycle_num)
            span.set_attribute("bandwidth", self.bandwidth)

            result = self._run_cycle(cycle_num, span)

        elapsed_ms = (time.monotonic() - t0) * 1000
        agent_cycle_duration.record(elapsed_ms, attrs)
        return result

    def _adapt_to_epistemic_state(self, cycle_num: int) -> None:
        """Read system entropy from graph and adjust behaviour.

        Standard regulation (Condition B):
          homogenizing → bandwidth ↑  (escape echo chamber)
          diverging     → bandwidth ↓  (consolidate)
          healthy       → reset

        Smart regulation (Condition C, activated when bridge_seeds are set):
          homogenizing → bandwidth ↑  (same)
          diverging + population isolated → migrate seed to bridge node
          diverging + cross-pol exists   → mild bandwidth ↓
          healthy       → reset seed + bandwidth
        """
        if cycle_num == 0:
            return

        signal = read_epistemic_signal(self._neo4j)
        regime         = signal.get("regime", "healthy")
        isolation_rate = float(signal.get("isolation_rate", 0.0))

        if regime == "homogenizing":
            self.bandwidth = min(self._base_bandwidth + 3, 15)
            logger.debug("Agent[%s] homogenizing → bandwidth=%d", self.name, self.bandwidth)

        elif regime == "diverging":
            if self._bridge_seeds and isolation_rate > 0.5:
                # Condition C: population is isolated AND diverging
                # → migrate to a bridge node to enable cross-pollination
                self._cycles_isolated += 1
                if self._cycles_isolated >= 2:   # wait 2 cycles before migrating
                    import random
                    new_seed = random.choice(self._bridge_seeds)
                    if new_seed != self._seed:
                        logger.debug(
                            "Agent[%s] isolated+diverging → seed %s → %s",
                            self.name, self._seed, new_seed,
                        )
                        self._seed = new_seed
                        self._cycles_isolated = 0
            else:
                # Standard Condition B: reduce bandwidth
                self.bandwidth = max(self._base_bandwidth - 2, 2)
                logger.debug("Agent[%s] diverging → bandwidth=%d", self.name, self.bandwidth)

        else:
            # healthy — reset to base
            self.bandwidth = self._base_bandwidth
            self._seed = self._original_seed   # return to home seed
            self._cycles_isolated = 0

    def _run_cycle(self, cycle_num: int, span: Any) -> CycleWrite:
        """Inner cycle logic — separated so the OTel span wraps cleanly."""
        results = self._retriever.retrieve(
            self.interest,
            seed_node_ids=[self._seed] if self._seed else None,
            top_k=self.bandwidth,
        )

        if not results:
            logger.warning("SocietyAgent[%s] cycle %d: no results", self.name, cycle_num)
            span.set_attribute("cycle.skipped", True)
            return CycleWrite(
                agent=self.name, cycle=cycle_num, node_id=None,
                belief="", confidence=0.0, read_from=[],
            )

        fragments = self._collect_content(results)
        if not fragments:
            return CycleWrite(
                agent=self.name, cycle=cycle_num, node_id=None,
                belief="", confidence=0.0, read_from=[r.id for r in results],
            )

        belief, confidence = self._synthesize(fragments, cycle_num)
        node_id = self._write(belief, confidence, cycle_num, [r.id for r in results])

        # --- Belief drift: cosine distance from previous cycle's belief ----
        attrs = {"agent": self.name}
        if self._faiss is not None:
            current_vec = self._faiss.encode([belief])[0].tolist()
            if self.memory.last_embedding is not None:
                drift = cosine_distance(self.memory.last_embedding, current_vec)
                self.memory.total_drift += drift
                belief_drift_metric.record(drift, attrs)
                span.set_attribute("belief.drift", drift)
                span.set_attribute("belief.total_drift", self.memory.total_drift)
                logger.debug("SocietyAgent[%s] cycle %d drift=%.4f", self.name, cycle_num, drift)
            self.memory.last_embedding = current_vec

        # --- Cross-pollination: count reads from other agents' nodes -------
        cross_reads = sum(
            1 for rid in [r.id for r in results]
            if rid.startswith("exp_") and not rid.startswith(f"exp_{self.name}_")
        )
        if cross_reads:
            cross_pollination_counter.add(cross_reads, attrs)
            span.set_attribute("cross_pollination.reads", cross_reads)

        span.set_attribute("belief.confidence", confidence)
        span.set_attribute("node_id", node_id or "")

        return CycleWrite(
            agent=self.name,
            cycle=cycle_num,
            node_id=node_id,
            belief=belief,
            confidence=confidence,
            read_from=[r.id for r in results],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_content(self, results: list[Any]) -> list[str]:
        fragments = []
        for r in results:
            node = self._neo4j.get_node(r.id)
            if node and node.get("content"):
                fragments.append(node["content"])
        return fragments

    def _synthesize(self, fragments: list[str], cycle_num: int) -> tuple[str, float]:
        """Ask Ollama to synthesize a belief from the retrieved fragments.

        Falls back to returning the first fragment verbatim if Ollama is
        unavailable — so the experiment can run without a local LLM.
        """
        evidence = "\n".join(f"- {f[:300]}" for f in fragments[:self.bandwidth])
        user_msg = (
            f"My interest: {self.interest}\n"
            f"Cycle: {cycle_num}\n\n"
            f"Evidence from shared knowledge base:\n{evidence}"
        )
        try:
            raw = self._ollama._chat(_SYNTHESIZE_SYSTEM, user_msg)
            parsed = _parse_json(raw)
            if parsed and "belief" in parsed:
                belief = str(parsed["belief"])[:400]
                confidence = float(parsed.get("confidence", 0.6))
                logger.info(
                    "SocietyAgent[%s] cycle %d: synthesized (conf=%.2f)",
                    self.name, cycle_num, confidence,
                )
                return belief, confidence
        except Exception as exc:
            logger.warning("SocietyAgent[%s] synthesis failed: %s", self.name, exc)

        # Fallback: use first fragment directly
        return fragments[0][:300], 0.4

    def _write(
        self,
        belief: str,
        confidence: float,
        cycle_num: int,
        result_ids: list[str],
    ) -> str:
        node_id = f"exp_{self.name}_c{cycle_num}_{str(uuid.uuid4())[:6]}"
        self._neo4j.upsert_node(
            node_id,
            content=belief,
            metadata={
                "source": self.name,
                "confidence": confidence,
                "cycle": cycle_num,
                "cluster": "experiment",
                "interest": self.interest,
            },
        )
        # Add to FAISS so future agents can find this node via semantic search
        if self._faiss is not None:
            self._faiss.add(node_id, belief, {
                "source": self.name, "cluster": "experiment", "cycle": cycle_num,
            })
        # Anchor edges: all retrieved nodes get DERIVED_FROM edges
        # Base nodes get weighted by rank; experiment nodes from other agents
        # get secondary edges (cross-pollination signal)
        for rank, rid in enumerate(result_ids):
            if rid.startswith("exp_"):
                # Cross-pollination: only other agents' experiment nodes
                if not rid.startswith(f"exp_{self.name}_"):
                    self._neo4j.upsert_edge(
                        node_id, rid,
                        relation="DERIVED_FROM",
                        weight=confidence * 0.8,
                    )
            else:
                # Base knowledge node — weight decays with rank
                rank_weight = confidence * (1.0 / (1.0 + rank * 0.3))
                self._neo4j.upsert_edge(
                    node_id, rid,
                    relation="DERIVED_FROM",
                    weight=rank_weight,
                )
        self.memory.node_ids.append(node_id)
        self.memory.cycle_writes.setdefault(cycle_num, []).append(node_id)
        return node_id
