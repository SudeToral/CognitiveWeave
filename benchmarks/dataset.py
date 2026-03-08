"""Synthetic benchmark dataset for CognitiveWeave retrieval evaluation.

Each query has a known set of relevant node ids (ground truth). The dataset
covers the knowledge domains used in the project's own knowledge graph so
the embedding model is not disadvantaged by out-of-domain vocabulary.

BenchmarkQuery fields:
    query_id    Stable identifier.
    text        Natural language query string.
    relevant    Ordered list of node ids — first entry is most relevant.
    seed_ids    Node ids to use as BFS seeds for graph search.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkQuery:
    query_id: str
    text: str
    relevant: list[str]          # ground truth, ordered by relevance
    seed_ids: list[str] = field(default_factory=list)


@dataclass
class BenchmarkNode:
    node_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class BenchmarkDataset:
    nodes: list[BenchmarkNode]
    edges: list[BenchmarkEdge]
    queries: list[BenchmarkQuery]


def build_dataset() -> BenchmarkDataset:
    """Return a synthetic knowledge graph + query set.

    Graph covers memory science, graph algorithms, and distributed systems —
    the three core domains of CognitiveWeave's theoretical background.
    Queries are designed to test both semantic (FAISS) and structural
    (Neo4j BFS) retrieval paths.
    """
    nodes = [
        BenchmarkNode("n01", "Ebbinghaus forgetting curve describes exponential memory decay over time"),
        BenchmarkNode("n02", "Spaced repetition exploits the spacing effect to improve long-term retention"),
        BenchmarkNode("n03", "Memory consolidation transfers information from short-term to long-term memory"),
        BenchmarkNode("n04", "Synaptic plasticity is the ability of synapses to strengthen or weaken over time"),
        BenchmarkNode("n05", "Long-term potentiation is a sustained strengthening of synaptic transmission"),
        BenchmarkNode("n06", "Reciprocal rank fusion combines ranked lists from multiple retrieval systems"),
        BenchmarkNode("n07", "BM25 is a bag-of-words retrieval function based on term frequency statistics"),
        BenchmarkNode("n08", "Dense retrieval uses vector embeddings to find semantically similar documents"),
        BenchmarkNode("n09", "FAISS is a library for efficient similarity search over dense vectors"),
        BenchmarkNode("n10", "Approximate nearest neighbor search trades recall for speed at large scale"),
        BenchmarkNode("n11", "Neo4j is a graph database that stores nodes and edges with properties"),
        BenchmarkNode("n12", "Cypher is the declarative query language used by Neo4j"),
        BenchmarkNode("n13", "Graph traversal algorithms include BFS, DFS, and Dijkstra shortest path"),
        BenchmarkNode("n14", "Community detection in graphs identifies densely connected subgraphs"),
        BenchmarkNode("n15", "The Leiden algorithm improves upon Louvain for community detection quality"),
        BenchmarkNode("n16", "Redis supports publish-subscribe messaging for real-time event distribution"),
        BenchmarkNode("n17", "Distributed locks prevent concurrent writes to the same resource across nodes"),
        BenchmarkNode("n18", "CRDTs allow distributed systems to merge state without coordination"),
        BenchmarkNode("n19", "The CAP theorem states distributed systems can guarantee only two of consistency, availability, and partition tolerance"),
        BenchmarkNode("n20", "Event sourcing records state changes as an immutable sequence of events"),
    ]

    edges = [
        BenchmarkEdge("n01", "n02", "MOTIVATES"),
        BenchmarkEdge("n02", "n03", "ENHANCES"),
        BenchmarkEdge("n03", "n04", "DEPENDS_ON"),
        BenchmarkEdge("n04", "n05", "ENABLES"),
        BenchmarkEdge("n06", "n07", "FUSES_WITH"),
        BenchmarkEdge("n06", "n08", "FUSES_WITH"),
        BenchmarkEdge("n08", "n09", "IMPLEMENTED_BY"),
        BenchmarkEdge("n09", "n10", "USES"),
        BenchmarkEdge("n11", "n12", "USES"),
        BenchmarkEdge("n11", "n13", "SUPPORTS"),
        BenchmarkEdge("n13", "n14", "RELATED_TO"),
        BenchmarkEdge("n14", "n15", "IMPROVED_BY"),
        BenchmarkEdge("n16", "n17", "COMPLEMENTS"),
        BenchmarkEdge("n17", "n18", "CONTRASTS_WITH"),
        BenchmarkEdge("n18", "n19", "RELATED_TO"),
        BenchmarkEdge("n19", "n20", "MOTIVATES"),
    ]

    queries = [
        BenchmarkQuery(
            query_id="q01",
            text="how does human memory decay over time",
            relevant=["n01", "n02", "n03"],
            seed_ids=["n01"],
        ),
        BenchmarkQuery(
            query_id="q02",
            text="combining multiple search systems for better retrieval",
            relevant=["n06", "n07", "n08"],
            seed_ids=["n06"],
        ),
        BenchmarkQuery(
            query_id="q03",
            text="efficient vector similarity search at scale",
            relevant=["n09", "n10", "n08"],
            seed_ids=["n08"],
        ),
        BenchmarkQuery(
            query_id="q04",
            text="graph database query language and traversal",
            relevant=["n11", "n12", "n13"],
            seed_ids=["n11"],
        ),
        BenchmarkQuery(
            query_id="q05",
            text="detecting communities and clusters in graphs",
            relevant=["n14", "n15", "n13"],
            seed_ids=["n14"],
        ),
        BenchmarkQuery(
            query_id="q06",
            text="distributed coordination and concurrent write safety",
            relevant=["n17", "n16", "n18"],
            seed_ids=["n17"],
        ),
        BenchmarkQuery(
            query_id="q07",
            text="synaptic strengthening and neural plasticity",
            relevant=["n04", "n05", "n03"],
            seed_ids=["n04"],
        ),
        BenchmarkQuery(
            query_id="q08",
            text="event driven architecture and state management",
            relevant=["n20", "n16", "n19"],
            seed_ids=["n20"],
        ),
    ]

    return BenchmarkDataset(nodes=nodes, edges=edges, queries=queries)
