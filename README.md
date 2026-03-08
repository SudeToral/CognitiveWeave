# CognitiveWeave

A multi-agent epistemological memory infrastructure for language model systems. CognitiveWeave provides a shared, self-optimizing knowledge graph that multiple AI agents can read from and write to concurrently, with built-in mechanisms for temporal decay, belief reconciliation, and confidence-aware retrieval.

## Motivation

Static knowledge graphs accumulate stale and contradictory information over time. Two failure modes are common in production systems: graph topology rot, where edges between concepts persist indefinitely regardless of relevance, and belief divergence, where multiple agents write conflicting facts to the same nodes without resolution. CognitiveWeave addresses both by treating graph maintenance as a first-class concern handled by dedicated agents rather than an offline batch process.

The system is positioned as an open-source alternative to closed systems such as Zep/Graphiti (January 2025), which introduced temporal knowledge graphs for single-agent memory but does not expose topology optimization or multi-agent coordination primitives.

## Theoretical Background

**Ebbinghaus Forgetting Curve.** The core decay model derives from Hermann Ebbinghaus's 1885 empirical work on memory retention. His forgetting curve describes retention R as R = e^(-t/S), where t is elapsed time and S is the stability of the memory trace. CognitiveWeave applies this formula directly to graph edge weights: each edge carries a `stability` field that increases when the edge is traversed during retrieval (implementing spaced repetition) and a `last_decay_at` timestamp. The Curator Agent runs the decay formula inside Neo4j using Cypher's native `exp()` function, so the computation is performed at the storage layer rather than in application code.

**Reciprocal Rank Fusion.** Retrieval combines two independent ranking signals: dense vector similarity from FAISS and structural proximity from Neo4j BFS traversal. These are merged using Reciprocal Rank Fusion (Cormack et al., 2009) with the standard k=60 constant. RRF was chosen over score normalization because it is robust to the different score distributions produced by cosine similarity and hop-distance scoring — neither signal needs to be normalized before fusion.

**Recency Bias in Retrieval.** After RRF fusion, each result is multiplied by a recency boost factor computed as e^(-age/halflife), where age is days since the node was first ingested. This keeps the retrieval signal independent from the freshness signal: `rrf_score` reflects topical relevance, `temporal_score` reflects relevance weighted by freshness. Both values are exposed in retrieval results for downstream monitoring.

**Belief Reconciliation as Distributed Systems Consensus.** The ReconcilerAgent adapts the fork/merge pattern from distributed version control to the knowledge graph domain. When two agents write conflicting content to the same node, the agent forks both versions into isolated Redis namespaces, evaluates each using the local Ollama LLM, and writes the winner back under a distributed lock. This is conceptually related to CRDTs (Conflict-free Replicated Data Types) but uses LLM judgment rather than a mathematical merge function, which is appropriate when the conflict is semantic rather than structural.

## Architecture

The system is organized into four layers.

**Storage Layer.** Neo4j holds the graph structure. Nodes represent knowledge claims, edges represent semantic or causal relationships between them. Each edge carries temporal metadata (`weight`, `stability`, `last_decay_at`) that the decay system operates on. FAISS provides dense vector indexing using `all-MiniLM-L6-v2` embeddings (384 dimensions, inner product index with L2-normalized vectors, equivalent to cosine similarity). Redis serves two roles: it is the event bus for inter-agent communication via Pub/Sub and Streams, and it is the state cache for agent coordination data including forked belief versions and distributed locks. MinIO provides S3-compatible object storage for large artifacts.

**Retrieval Layer.** The `HybridRetriever` class executes FAISS search and Neo4j BFS in sequence, applies RRF fusion, and multiplies the resulting scores by the recency boost. The `retrieve()` method accepts optional `seed_node_ids` for structural expansion. When seeds are provided, Neo4j BFS runs from those nodes; when omitted, only the FAISS signal is used. This makes the same retrieval path usable for both pure semantic queries and graph-neighborhood exploration.

**Agent Layer.** Five agents communicate exclusively through the Redis event bus. No agent calls another agent's methods directly. Each agent runs as an asyncio task that blocks on a Redis Pub/Sub channel and offloads blocking work to a per-agent `ThreadPoolExecutor`.

**Protocol Layer.** The MCP (Model Context Protocol) server exposes four tools to external language model clients: `retrieve`, `upsert_knowledge`, `get_graph_stats`, and `trigger_decay`. This allows any MCP-compatible LLM to use CognitiveWeave as a memory backend without direct dependency on its internal modules.

## Components

**CuratorAgent** monitors graph health. It responds to `decay:trigger` events and can also run on a configurable timer via `run_periodic()`. On each cycle it applies the Ebbinghaus decay formula to all edges via a two-step Cypher query, first initializing edges that lack temporal fields, then computing `w_new = w_old * exp(-delta_days / stability)`, and then prunes edges whose weight has fallen below a configurable threshold.

**RetrieverAgent** wraps the `HybridRetriever` and exposes it over the event bus. It listens for `retrieve:request` events, runs the retrieval in a thread because sentence-transformers encoding is CPU-bound, and publishes results to a `reply_to` channel specified in the request. This allows other agents to issue retrieval requests and receive results on their own channels without blocking the bus.

**ReconcilerAgent** handles semantic conflicts. When it receives a `conflict:detected` event it stores both conflicting versions in Redis under isolated key prefixes, calls `OllamaClient.evaluate_conflict()` in a thread, acquires a distributed lock on the target node using Redis SETNX with TTL, writes the winning version to Neo4j, and publishes `conflict:resolved`. A `_heuristic_score()` method provides a fallback path when Ollama is unavailable.

**EpistemologistAgent** scores incoming knowledge for credibility. If an explicit confidence value is provided by the upstream caller, the LLM call is skipped. Otherwise, it calls `OllamaClient.score_belief()` and applies a heuristic fallback if the call fails. Nodes scoring below 0.35 are flagged with `uncertain=True` in Neo4j. When a new version of an existing node diverges in confidence by more than 0.50, the agent emits `conflict:detected` to trigger the Reconciler.

**MonitorAgent** aggregates operational metrics. It increments counters in Redis for each known event type and exposes a `render_prometheus()` method that produces Prometheus text exposition format output for the MCP server and a future HTTP endpoint.

**OllamaClient** is a synchronous `httpx` wrapper around the Ollama `/api/chat` endpoint. It is designed to be called from a `ThreadPoolExecutor` so async agent loops are never blocked by LLM latency. Both `score_belief()` and `evaluate_conflict()` return fallback values on network errors or unparseable responses rather than raising exceptions.

**RedisBus** provides three primitives: `publish()` writes to a Pub/Sub channel and appends to an audit Stream in a single pipeline; `subscribe()` runs a blocking async listener loop; `lock()` is an async context manager that implements distributed mutual exclusion using Redis `SET NX PX` and releases the lock only if the caller still holds it, verified by a stored UUID value.

## Key Design Decisions

**Cypher-native decay computation.** The Ebbinghaus formula runs inside Neo4j rather than in Python. This avoids loading all edge data into application memory and ensures the decay timestamp update is atomic with the weight update within the same transaction.

**Stability as a per-edge property.** Rather than using a global decay rate, each edge has its own stability value that increases when the edge is accessed during retrieval. Edges that are frequently traversed decay more slowly, implementing spaced repetition at the graph topology level. The graph naturally strengthens high-utility connections and weakens peripheral ones without explicit curation logic.

**RRF over learned fusion.** A learned re-ranking model would require training data and a separate serving infrastructure. RRF requires neither and has well-understood behavior: a document that ranks first in both lists will always outscore one that ranks first in only one. The k=60 constant follows the original Cormack et al. empirical evaluation.

**Explicit confidence bypasses LLM scoring.** When the caller provides a confidence value directly, the EpistemologistAgent skips the Ollama call. This allows high-throughput ingestion pipelines that have their own confidence signals to populate the graph without incurring LLM latency on every write.

**Fallback chain for LLM calls.** Both the ReconcilerAgent and EpistemologistAgent implement a three-level fallback: explicit value, LLM call, heuristic. This ensures agents remain operational when Ollama is restarting or overloaded. The heuristic is not intended to be accurate; it is intended to be predictable so agent behavior during degraded mode is understandable.

**Event bus isolation.** Agents do not call each other's methods. All coordination happens through Redis Pub/Sub events. Each agent can be restarted independently without affecting others, and new agents can be added by subscribing to existing channels without modifying existing agent code.

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| Graph storage | Neo4j 5.x | Nodes, edges, temporal decay |
| Vector index | FAISS (IndexFlatIP) | Dense semantic search |
| Embedding model | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| Event bus | Redis 7 Pub/Sub + Streams | Inter-agent signaling and audit log |
| Distributed lock | Redis SETNX + TTL | Concurrent write protection |
| LLM inference | Ollama (local) | Belief scoring and conflict resolution |
| Object storage | MinIO | S3-compatible artifact storage |
| Agent protocol | MCP (Model Context Protocol) | External LLM tool interface |
| Runtime | Python 3.12 + asyncio | Agent concurrency model |
| Package manager | uv | Dependency management |

## Project Status

Weeks 1 through 8 of the planned 10-week roadmap are complete. 110 unit tests pass with no external dependencies required.

Implemented: hybrid retrieval with RRF and recency boost, Ebbinghaus edge decay with per-edge stability tracking, five-agent system with Redis coordination, MCP server with four tools, Ollama integration for belief scoring and conflict resolution, distributed locking for concurrent graph writes.

Remaining: k3s deployment manifests, Prometheus and Grafana integration, end-to-end benchmark against GraphRAG baseline, ArXiv manuscript.

## References

Ebbinghaus, H. (1885). *Uber das Gedachtnis*. Duncker and Humblot.

Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR 2009*.

Edge, D., et al. (2024). From local to global: A GraphRAG approach to query-focused summarization. *EMNLP 2024*.

Rasmussen, D., et al. (2025). Zep: A temporal knowledge graph architecture for agent memory. *arXiv:2501.13956*.

Wozniak, P. A. (1990). Optimization of learning. Master's thesis, University of Technology, Poznan.
