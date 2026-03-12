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
import uuid
from dataclasses import dataclass, field
from typing import Any

from cognitiveweave.llm.ollama_client import OllamaClient, _parse_json
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient

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
    ) -> None:
        self.name = name
        self.interest = interest
        self.bandwidth = bandwidth
        self._seed = seed_node_id
        self._retriever = retriever
        self._neo4j = neo4j
        self._ollama = ollama
        self._faiss = faiss
        self.memory = AgentMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cycle(self, cycle_num: int) -> CycleWrite:
        """Run one experiment cycle. Returns a record of what was written."""
        results = self._retriever.retrieve(
            self.interest,
            seed_node_ids=[self._seed] if self._seed else None,
            top_k=self.bandwidth,
        )

        if not results:
            logger.warning("SocietyAgent[%s] cycle %d: no results", self.name, cycle_num)
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
        # Primary anchor: top-ranked result
        self._neo4j.upsert_edge(
            node_id, result_ids[0],
            relation="DERIVED_FROM",
            weight=confidence,
        )
        # Secondary edges: any other-agent experiment nodes in the read set
        # These are the cross-pollination links the observer tracks
        for rid in result_ids[1:]:
            if rid.startswith("exp_") and not rid.startswith(f"exp_{self.name}_"):
                self._neo4j.upsert_edge(
                    node_id, rid,
                    relation="DERIVED_FROM",
                    weight=confidence * 0.8,
                )
        self.memory.node_ids.append(node_id)
        self.memory.cycle_writes.setdefault(cycle_num, []).append(node_id)
        return node_id
