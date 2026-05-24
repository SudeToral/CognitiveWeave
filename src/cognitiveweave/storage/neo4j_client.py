from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase, Session

from cognitiveweave.config.settings import Neo4jSettings
from cognitiveweave.storage.temporal import DEFAULT_STABILITY, STABILITY_BOOST


class Neo4jClient:
    """Neo4j graph storage — nodes, edges, structural path queries."""

    def __init__(self, settings: Neo4jSettings) -> None:
        self._settings = settings
        self._driver: Driver | None = None

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        """Open a managed session. Raises RuntimeError if not connected."""
        if self._driver is None:
            raise RuntimeError("Neo4jClient not connected — call connect() first")
        with self._driver.session(database=self._settings.database) as session:
            yield session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._settings.uri,
            auth=(self._settings.user, self._settings.password),
        )
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jClient:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """Idempotent — safe to call on each startup."""
        queries = [
            "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:KnowledgeNode) REQUIRE n.id IS UNIQUE",
        ]
        with self._session() as s:
            for q in queries:
                s.run(q)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_id: str | None = None,
        *,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create or update a KnowledgeNode. Returns the node id."""
        if node_id is None:
            node_id = str(uuid.uuid4())
        props = {"content": content, **(metadata or {})}
        # ON CREATE SET stamps created_at once; subsequent upserts don't overwrite it.
        query = """
        MERGE (n:KnowledgeNode {id: $id})
        ON CREATE SET n.created_at = toString(datetime())
        SET n += $props
        RETURN n.id AS id
        """
        with self._session() as s:
            result = s.run(query, id=node_id, props=props)
            record = result.single()
            return record["id"] if record else node_id

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a directed edge.

        Temporal fields are stamped only on creation (ON CREATE SET):
        - last_decay_at: ISO timestamp used by Ebbinghaus decay formula.
        - stability:     Controls how fast the edge decays (higher = slower).
        """
        props = {"weight": weight, **(metadata or {})}
        query = f"""
        MATCH (a:KnowledgeNode {{id: $src}})
        MATCH (b:KnowledgeNode {{id: $tgt}})
        MERGE (a)-[r:{relation}]->(b)
        ON CREATE SET r.last_decay_at = toString(datetime()),
                      r.stability     = $stability
        SET r += $props
        """
        with self._session() as s:
            s.run(query, src=source_id, tgt=target_id, props=props, stability=DEFAULT_STABILITY)

    def delete_node(self, node_id: str) -> None:
        query = "MATCH (n:KnowledgeNode {id: $id}) DETACH DELETE n"
        with self._session() as s:
            s.run(query, id=node_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        query = "MATCH (n:KnowledgeNode {id: $id}) RETURN properties(n) AS props"
        with self._session() as s:
            record = s.run(query, id=node_id).single()
            return dict(record["props"]) if record else None

    def structural_search(
        self,
        node_ids: list[str],
        *,
        max_hops: int = 2,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """BFS from seed nodes. Returns scored candidates + boosts traversed edge stability.

        Score = (1 / hop_distance) * avg_edge_weight_along_path
        This makes temporal decay affect retrieval: edges that have decayed to low
        weights reduce the structural score of paths through them.
        """
        # Variable-length path bounds must be literals in Neo4j 5.x — interpolate max_hops.
        query = f"""
        UNWIND $ids AS seed
        MATCH path = (start:KnowledgeNode {{id: seed}})-[*1..{max_hops}]-(candidate:KnowledgeNode)
        WHERE candidate.id <> seed
        WITH candidate,
             length(path) AS hops,
             [r IN relationships(path) | coalesce(r.weight, 1.0)] AS weights
        WITH candidate,
             min(hops) AS min_hops,
             avg(reduce(s = 0.0, w IN weights | s + w) / size(weights)) AS avg_weight
        RETURN candidate.id AS id,
               properties(candidate) AS props,
               (1.0 / (min_hops + 1.0)) * avg_weight AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self._session() as s:
            result = s.run(query, ids=node_ids, limit=limit)
            rows = [
                {"id": r["id"], "score": r["score"], **r["props"]}
                for r in result
            ]

        if rows:
            self._boost_path_stability(node_ids, [r["id"] for r in rows])

        return rows

    def _boost_path_stability(self, seeds: list[str], found: list[str]) -> None:
        """Increase stability on edges between seeds and found nodes.

        Frequently-retrieved edges decay slower — spaced repetition effect.
        """
        query = """
        UNWIND $seeds AS seed
        UNWIND $found AS target
        MATCH (a:KnowledgeNode {id: seed})-[r]-(b:KnowledgeNode {id: target})
        SET r.stability = coalesce(r.stability, $default_stab) + $boost
        """
        with self._session() as s:
            s.run(query, seeds=seeds, found=found,
                  default_stab=DEFAULT_STABILITY, boost=STABILITY_BOOST)

    # ------------------------------------------------------------------
    # Temporal decay (Ebbinghaus)
    # ------------------------------------------------------------------

    def decay_edge_weights(self) -> int:
        """Apply Ebbinghaus decay to all edges. Returns updated count.

        Formula: w_new = w_old × exp(-Δt / stability)

        Δt       = days since last_decay_at (computed in Cypher via duration.inDays)
        stability = per-edge field; increases each time the edge is retrieved

        Edges without temporal fields are initialised first so this is
        safe to call on graphs created before temporal support was added.
        """
        init_query = """
        MATCH ()-[r]->()
        WHERE r.last_decay_at IS NULL
        SET r.last_decay_at = toString(datetime()),
            r.stability     = $default_stab
        """
        decay_query = """
        MATCH ()-[r]->()
        WITH r,
             duration.inDays(datetime(r.last_decay_at), datetime()).days AS delta_days
        SET r.weight = r.weight * exp(-toFloat(delta_days) / coalesce(r.stability, $default_stab)),
            r.last_decay_at = toString(datetime())
        RETURN count(r) AS updated
        """
        with self._session() as s:
            s.run(init_query, default_stab=DEFAULT_STABILITY)
            record = s.run(decay_query, default_stab=DEFAULT_STABILITY).single()
            return int(record["updated"]) if record else 0

    def prune_weak_edges(self, threshold: float = 0.1) -> int:
        """Delete edges whose weight has decayed below threshold."""
        query = """
        MATCH ()-[r]->()
        WHERE r.weight < $threshold
        DELETE r
        RETURN count(r) AS deleted
        """
        with self._session() as s:
            record = s.run(query, threshold=threshold).single()
            return int(record["deleted"]) if record else 0
