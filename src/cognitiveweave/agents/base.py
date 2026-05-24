from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cognitiveweave.bus.redis_bus import RedisBus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all CognitiveWeave agents.

    Design decisions:
    - Each agent owns one asyncio Task that blocks on Redis Pub/Sub.
    - CPU-bound work is offloaded to a per-agent ThreadPoolExecutor so
      I/O-bound pub/sub processing is never blocked.
    - Errors in handle_event are caught and logged — a single bad event
      never kills the agent loop.
    """

    CHANNEL: str  # Subclass must declare

    def __init__(self, bus: RedisBus, *, max_workers: int = 2) -> None:
        self._bus = bus
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=self.__class__.__name__,
        )
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Schedule agent on the running event loop. Returns the task."""
        self._task = asyncio.create_task(
            self._run(), name=self.__class__.__name__
        )
        logger.info("%s started on channel %s", self.__class__.__name__, self.CHANNEL)
        return self._task

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._executor.shutdown(wait=False)
        logger.info("%s stopped", self.__class__.__name__)

    async def _run(self) -> None:
        await self._bus.subscribe(self.CHANNEL, self._dispatch)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, event: dict[str, Any]) -> None:
        try:
            await self.handle_event(event)
        except Exception:
            logger.exception(
                "%s: unhandled error processing event %s",
                self.__class__.__name__,
                event.get("type", "?"),
            )

    @abstractmethod
    async def handle_event(self, event: dict[str, Any]) -> None:
        """Process one event from the bus."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def run_in_thread(self, fn, *args: Any) -> Any:
        """Run a blocking/CPU-bound function in the agent's thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def emit(self, channel: str, event: dict[str, Any]) -> None:
        await self._bus.publish(channel, event)
