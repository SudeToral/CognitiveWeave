from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from cognitiveweave.config.settings import RedisSettings

# Canonical channel names — agents import these instead of bare strings.
CH_CURATOR = "cw:curator"
CH_RETRIEVER = "cw:retriever"
CH_RECONCILER = "cw:reconciler"
CH_EPISTEMOLOGIST = "cw:epistemologist"
CH_MONITOR = "cw:monitor"

STREAM_KEY = "cw:stream:events"
STATE_PREFIX = "cw:state:"
LOCK_PREFIX = "cw:lock:"


class RedisBus:
    """Redis event bus — three primitives:

    1. Pub/Sub  — fire-and-forget signals between agents.
    2. Streams  — append-only audit log of every published event.
    3. Lock     — distributed mutex via SETNX+TTL for concurrent graph writes.
    """

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._redis: Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._redis = Redis(
            host=self._settings.host,
            port=self._settings.port,
            db=self._settings.db,
            decode_responses=True,
        )
        await self._redis.ping()

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def __aenter__(self) -> RedisBus:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Pub/Sub + Stream
    # ------------------------------------------------------------------

    async def publish(self, channel: str, event: dict[str, Any]) -> None:
        """Publish to Pub/Sub and append to the audit Stream atomically."""
        payload = json.dumps(event)
        pipe = self._redis.pipeline()
        pipe.publish(channel, payload)
        pipe.xadd(STREAM_KEY, {"channel": channel, "payload": payload}, maxlen=10_000)
        await pipe.execute()

    async def subscribe(
        self,
        channel: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Block-listen on channel; call handler for each message.

        Runs until the task is cancelled — intended to be wrapped in
        asyncio.create_task() by each agent.
        """
        pubsub: PubSub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                event = json.loads(message["data"])
                await handler(event)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    # ------------------------------------------------------------------
    # Distributed lock
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lock(self, key: str, ttl_ms: int = 5_000):
        """Acquire a distributed lock. Raises RuntimeError if already held.

        Usage:
            async with bus.lock("node:abc-123"):
                await neo4j_client.upsert_node(...)
        """
        lock_key = f"{LOCK_PREFIX}{key}"
        lock_val = str(uuid.uuid4())
        acquired = await self._redis.set(lock_key, lock_val, nx=True, px=ttl_ms)
        if not acquired:
            raise RuntimeError(f"Lock contention on {key!r}")
        try:
            yield
        finally:
            current = await self._redis.get(lock_key)
            if current == lock_val:
                await self._redis.delete(lock_key)

    # ------------------------------------------------------------------
    # Agent state cache
    # ------------------------------------------------------------------

    async def set_state(self, key: str, value: Any, ttl: int = 3_600) -> None:
        await self._redis.set(f"{STATE_PREFIX}{key}", json.dumps(value), ex=ttl)

    async def get_state(self, key: str) -> Any | None:
        raw = await self._redis.get(f"{STATE_PREFIX}{key}")
        return json.loads(raw) if raw else None

    async def increment_counter(self, key: str, by: int = 1) -> int:
        return await self._redis.incrby(f"{STATE_PREFIX}{key}", by)

    # ------------------------------------------------------------------
    # Stream read (for Monitor)
    # ------------------------------------------------------------------

    async def read_stream(self, count: int = 100, last_id: str = "0") -> list[dict[str, Any]]:
        """Read up to `count` entries from the audit stream starting after last_id."""
        entries = await self._redis.xread({STREAM_KEY: last_id}, count=count)
        if not entries:
            return []
        results = []
        for _stream, messages in entries:
            for msg_id, fields in messages:
                results.append({
                    "id": msg_id,
                    "channel": fields["channel"],
                    "event": json.loads(fields["payload"]),
                })
        return results
