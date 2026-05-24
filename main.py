"""CognitiveWeave — system entry point.

Starts all five agents and the MCP server.
Telemetry is initialised first so every subsequent component is instrumented.

Usage:
    uv run python main.py
"""
from __future__ import annotations

import asyncio
import logging

from cognitiveweave.telemetry import setup_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _run() -> None:
    from cognitiveweave.config.settings import Settings
    from cognitiveweave.storage.neo4j_client import Neo4jClient
    from cognitiveweave.storage.faiss_index import FAISSIndex
    from cognitiveweave.bus.redis_bus import RedisBus
    from cognitiveweave.agents.curator import CuratorAgent
    from cognitiveweave.agents.epistemologist import EpistemologistAgent
    from cognitiveweave.agents.reconciler import ReconcilerAgent
    from cognitiveweave.agents.retriever import RetrieverAgent
    from cognitiveweave.agents.monitor import MonitorAgent
    from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever
    from cognitiveweave.llm.ollama_client import OllamaClient

    settings = Settings()

    # Telemetry — must be first
    setup_telemetry(console_fallback=True)

    neo4j = Neo4jClient(settings.neo4j)
    neo4j.connect()
    neo4j.create_constraints()

    faiss = FAISSIndex(settings.faiss)
    faiss.load_model()
    faiss.load() if __import__("pathlib").Path(settings.faiss.index_path).exists() else faiss.build_index()

    bus = RedisBus(settings.redis)
    await bus.connect()

    ollama = OllamaClient(settings.ollama)
    retriever = HybridRetriever(faiss, neo4j)

    agents = [
        CuratorAgent(neo4j, bus),
        EpistemologistAgent(neo4j, bus, ollama),
        ReconcilerAgent(neo4j, bus, ollama),
        RetrieverAgent(retriever, bus),
        MonitorAgent(bus),
    ]

    logger.info("Starting %d agents…", len(agents))
    tasks = [agent.start() for agent in agents]

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down…")
    finally:
        for agent in agents:
            await agent.stop()
        await bus.close()
        neo4j.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
