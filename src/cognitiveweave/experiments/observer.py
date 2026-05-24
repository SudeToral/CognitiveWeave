"""ExperimentObserver — snapshots and analyzes the shared belief graph.

At each cycle the observer queries Neo4j for the state of experiment nodes
and produces metrics that answer the core research questions:

1. Belief production: how much is each agent writing, and how confident?
2. Belief drift: are agent confidence scores converging or diverging?
3. Cross-pollination: are agents reading each other's writes
   (detected via DERIVED_FROM edges landing on experiment nodes)?
4. Consensus candidates: nodes that multiple agents have linked to,
   suggesting shared focal points are emerging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cognitiveweave.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class BeliefSnapshot:
    """State of the experiment graph at the end of one cycle."""
    cycle: int
    # agent_name -> total nodes written so far
    node_counts: dict[str, int] = field(default_factory=dict)
    # agent_name -> average confidence of all their nodes so far
    avg_confidence: dict[str, float] = field(default_factory=dict)
    # total experiment nodes in the graph
    total_nodes: int = 0
    # node_ids that have been linked to by ≥2 different agents (emerging consensus)
    consensus_anchors: list[str] = field(default_factory=list)
    # agent_name -> count of reads from another agent's experiment nodes
    cross_reads: dict[str, int] = field(default_factory=dict)


class ExperimentObserver:
    """Reads Neo4j to produce BeliefSnapshot objects for visualization.

    All queries are read-only and use the shared Neo4jClient.

    Args:
        neo4j:    The shared Neo4j client.
        database: Neo4j database name (default "neo4j").
    """

    def __init__(self, neo4j: Neo4jClient, database: str = "neo4j") -> None:
        self._neo4j = neo4j
        self._db = database

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, cycle: int, agent_names: list[str]) -> BeliefSnapshot:
        """Compute metrics for the current state of the experiment graph."""
        snap = BeliefSnapshot(cycle=cycle)

        for name in agent_names:
            counts, avg_conf = self._agent_stats(name)
            snap.node_counts[name] = counts
            snap.avg_confidence[name] = avg_conf

        snap.total_nodes = self._total_experiment_nodes()
        snap.consensus_anchors = self._consensus_anchors(agent_names)
        snap.cross_reads = self._cross_read_counts(agent_names)

        return snap

    # ------------------------------------------------------------------
    # Full node dump (for table rendering)
    # ------------------------------------------------------------------

    def get_all_experiment_nodes(self) -> list[dict[str, Any]]:
        """Return all experiment nodes ordered by cycle, for display."""
        query = """
        MATCH (n:KnowledgeNode {cluster: 'experiment'})
        RETURN n.id        AS id,
               n.content   AS content,
               n.source    AS source,
               n.confidence AS confidence,
               n.cycle     AS cycle,
               n.interest  AS interest
        ORDER BY n.cycle ASC
        """
        with self._neo4j._session() as s:
            return [dict(r) for r in s.run(query)]

    def has_experiment_data(self) -> bool:
        """Return True if any experiment nodes exist in the graph."""
        query = "MATCH (n:KnowledgeNode {cluster: 'experiment'}) RETURN count(n) AS c LIMIT 1"
        with self._neo4j._session() as s:
            record = s.run(query).single()
            return bool(record and record["c"] > 0)

    def rebuild_snapshots(self, agent_names: list[str]) -> list[BeliefSnapshot]:
        """Reconstruct per-cycle BeliefSnapshot history from existing Neo4j data."""
        query = "MATCH (n:KnowledgeNode {cluster: 'experiment'}) RETURN DISTINCT n.cycle AS c ORDER BY c ASC"
        with self._neo4j._session() as s:
            cycles = [r["c"] for r in s.run(query) if r["c"] is not None]
        return [self.snapshot(c, agent_names) for c in cycles]

    def get_agent_beliefs_by_cycle(self, agent_name: str) -> list[tuple[int, str]]:
        """Return [(cycle_num, belief_text), ...] ordered by cycle ascending."""
        query = """
        MATCH (n:KnowledgeNode {source: $src, cluster: 'experiment'})
        RETURN n.cycle AS cycle, n.content AS content
        ORDER BY n.cycle ASC
        """
        with self._neo4j._session() as s:
            return [(r["cycle"], r["content"]) for r in s.run(query, src=agent_name)
                    if r["content"]]

    def cleanup(self) -> int:
        """Delete all experiment nodes (for experiment reset)."""
        query = """
        MATCH (n:KnowledgeNode {cluster: 'experiment'})
        DETACH DELETE n
        RETURN count(n) AS deleted
        """
        with self._neo4j._session() as s:
            record = s.run(query).single()
            return record["deleted"] if record else 0

    # ------------------------------------------------------------------
    # Internal queries
    # ------------------------------------------------------------------

    def _agent_stats(self, agent_name: str) -> tuple[int, float]:
        query = """
        MATCH (n:KnowledgeNode {source: $src, cluster: 'experiment'})
        RETURN count(n) AS cnt,
               avg(toFloat(n.confidence)) AS avg_conf
        """
        with self._neo4j._session() as s:
            record = s.run(query, src=agent_name).single()
            if not record:
                return 0, 0.0
            cnt = record["cnt"] or 0
            avg_conf = record["avg_conf"] or 0.0
            return int(cnt), float(avg_conf)

    def _total_experiment_nodes(self) -> int:
        query = "MATCH (n:KnowledgeNode {cluster: 'experiment'}) RETURN count(n) AS c"
        with self._neo4j._session() as s:
            record = s.run(query).single()
            return int(record["c"]) if record else 0

    def _consensus_anchors(self, agent_names: list[str]) -> list[str]:
        """Find non-experiment nodes that ≥2 distinct agents have linked to.

        These are the 'focal points' that multiple agents independently
        treated as anchor knowledge — a signal of emerging consensus.
        """
        query = """
        MATCH (exp:KnowledgeNode {cluster: 'experiment'})-[:DERIVED_FROM]->(anchor:KnowledgeNode)
        WHERE anchor.cluster <> 'experiment'
        WITH anchor.id AS anchor_id, collect(DISTINCT exp.source) AS sources
        WHERE size(sources) >= 2
        RETURN anchor_id
        ORDER BY size(sources) DESC
        LIMIT 10
        """
        with self._neo4j._session() as s:
            return [r["anchor_id"] for r in s.run(query)]

    def _cross_read_counts(self, agent_names: list[str]) -> dict[str, int]:
        """Count how many times each agent anchored to another agent's experiment nodes.

        A cross-read happens when an agent's new write is DERIVED_FROM a node
        that was written by a different agent. This means the agent read and
        built on another agent's belief — the core signal of knowledge spreading.
        """
        query = """
        MATCH (writer:KnowledgeNode {cluster: 'experiment'})
              -[:DERIVED_FROM]->
              (anchor:KnowledgeNode {cluster: 'experiment'})
        WHERE writer.source <> anchor.source
        RETURN writer.source AS agent, count(*) AS cross_count
        """
        counts: dict[str, int] = {name: 0 for name in agent_names}
        with self._neo4j._session() as s:
            for record in s.run(query):
                agent = record["agent"]
                if agent in counts:
                    counts[agent] = int(record["cross_count"])
        return counts
