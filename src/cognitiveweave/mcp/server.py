from __future__ import annotations

import json
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from cognitiveweave.bus.redis_bus import RedisBus, CH_RETRIEVER, CH_CURATOR, CH_MONITOR
from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever


def build_mcp_server(
    settings: Settings,
    neo4j: Neo4jClient,
    faiss: FAISSIndex,
    bus: RedisBus,
) -> Server:
    """Factory — wires storage + bus into an MCP Server instance.

    Exposed tools:
        retrieve           — hybrid RRF search
        upsert_knowledge   — add / update a knowledge node
        get_graph_stats    — node + edge counts from Neo4j
        trigger_decay      — ask CuratorAgent to run a decay cycle now
    """
    server = Server("cognitiveweave")
    retriever = HybridRetriever(faiss, neo4j)

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="retrieve",
                description=(
                    "Hybrid semantic + structural retrieval from the knowledge graph. "
                    "Returns ranked results with RRF scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query"},
                        "seed_node_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional node ids for BFS structural expansion",
                        },
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="upsert_knowledge",
                description="Add or update a knowledge node in the graph and vector index.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "Stable id (uuid recommended)"},
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="get_graph_stats",
                description="Return node count, edge count, and last decay cycle stats.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="trigger_decay",
                description=(
                    "Ask the CuratorAgent to run an immediate decay + prune cycle. "
                    "Returns immediately; result arrives asynchronously via bus."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent]:

        if name == "retrieve":
            results = retriever.retrieve(
                arguments["query"],
                seed_node_ids=arguments.get("seed_node_ids"),
                top_k=int(arguments.get("top_k", 10)),
            )
            payload = [
                {
                    "id": r.id,
                    "rrf_score": r.rrf_score,
                    "faiss_rank": r.faiss_rank,
                    "graph_rank": r.graph_rank,
                    **r.metadata,
                }
                for r in results
            ]
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        if name == "upsert_knowledge":
            node_id = neo4j.upsert_node(
                arguments.get("node_id"),
                content=arguments["content"],
                metadata={
                    **arguments.get("metadata", {}),
                    "confidence": arguments.get("confidence", 0.5),
                },
            )
            faiss.add(node_id, arguments["content"], arguments.get("metadata"))
            # Notify EpistemologistAgent
            await bus.publish(
                "cw:epistemologist",
                {
                    "type": "node:created",
                    "node_id": node_id,
                    "content": arguments["content"],
                    "confidence": arguments.get("confidence"),
                    "source": arguments.get("metadata", {}).get("source", "mcp"),
                },
            )
            return [types.TextContent(type="text", text=json.dumps({"node_id": node_id}))]

        if name == "get_graph_stats":
            node_count_result = neo4j._driver.session().run(
                "MATCH (n:KnowledgeNode) RETURN count(n) AS cnt"
            ).single()
            edge_count_result = neo4j._driver.session().run(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            ).single()
            curator_state = await bus.get_state("curator:last_cycle")
            stats = {
                "node_count": node_count_result["cnt"] if node_count_result else 0,
                "edge_count": edge_count_result["cnt"] if edge_count_result else 0,
                "faiss_vectors": len(faiss),
                "curator_last_cycle": curator_state,
            }
            return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

        if name == "trigger_decay":
            await bus.publish(CH_CURATOR, {"type": "decay:trigger"})
            return [types.TextContent(type="text", text='{"status": "decay triggered"}')]

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run_stdio(settings: Settings | None = None) -> None:
    """Entry point for `uv run cognitiveweave-mcp` — runs over stdio."""
    if settings is None:
        settings = Settings()

    neo4j = Neo4jClient(settings.neo4j)
    neo4j.connect()
    neo4j.create_constraints()

    faiss = FAISSIndex(settings.faiss)
    faiss.load_model()
    faiss.build_index()

    bus = RedisBus(settings.redis)
    await bus.connect()

    server = build_mcp_server(settings, neo4j, faiss, bus)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await bus.close()
        neo4j.close()
