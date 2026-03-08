"""Unit tests for RedisBus — Redis client is fully mocked."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognitiveweave.config.settings import RedisSettings
from cognitiveweave.bus.redis_bus import RedisBus


@pytest.fixture()
def settings() -> RedisSettings:
    return RedisSettings(host="localhost", port=6379, db=0)


@pytest.fixture()
def mock_redis():
    r = AsyncMock()
    r.ping = AsyncMock()
    r.aclose = AsyncMock()
    r.set = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=None)
    r.delete = AsyncMock()
    r.incrby = AsyncMock(return_value=1)
    r.xread = AsyncMock(return_value=[])
    pipe = MagicMock()
    pipe.publish = MagicMock()
    pipe.xadd = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, "0-1"])
    r.pipeline = MagicMock(return_value=pipe)
    return r


@pytest.fixture()
def connected_bus(settings, mock_redis):
    """RedisBus with _redis injected — no real TCP connection."""
    b = RedisBus(settings)
    b._redis = mock_redis
    return b, mock_redis


class TestConnect:
    @pytest.mark.asyncio
    async def test_ping_called_on_connect(self, settings, mock_redis):
        with patch("cognitiveweave.bus.redis_bus.Redis", return_value=mock_redis):
            b = RedisBus(settings)
            await b.connect()
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, connected_bus):
        b, mock_redis = connected_bus
        await b.close()
        mock_redis.aclose.assert_called_once()
        assert b._redis is None


class TestPublish:
    @pytest.mark.asyncio
    async def test_publish_uses_pipeline(self, connected_bus):
        b, mock_redis = connected_bus
        await b.publish("cw:curator", {"type": "decay:trigger"})
        mock_redis.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_publish_serialises_event(self, connected_bus):
        b, mock_redis = connected_bus
        pipe = mock_redis.pipeline.return_value
        await b.publish("cw:test", {"type": "hello", "data": 42})
        pipe.publish.assert_called_once()
        channel, payload = pipe.publish.call_args[0]
        assert channel == "cw:test"
        assert json.loads(payload) == {"type": "hello", "data": 42}

    @pytest.mark.asyncio
    async def test_publish_appends_to_stream(self, connected_bus):
        b, mock_redis = connected_bus
        pipe = mock_redis.pipeline.return_value
        await b.publish("cw:curator", {"type": "decay:trigger"})
        pipe.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_executes_pipeline(self, connected_bus):
        b, mock_redis = connected_bus
        pipe = mock_redis.pipeline.return_value
        await b.publish("cw:test", {"type": "x"})
        pipe.execute.assert_called_once()


class TestLock:
    @pytest.mark.asyncio
    async def test_lock_acquires_via_setnx(self, connected_bus):
        b, mock_redis = connected_bus
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)
        async with b.lock("node:test"):
            call_kwargs = mock_redis.set.call_args[1]
            assert call_kwargs.get("nx") is True
            assert call_kwargs.get("px") is not None

    @pytest.mark.asyncio
    async def test_lock_raises_on_contention(self, connected_bus):
        b, mock_redis = connected_bus
        mock_redis.set = AsyncMock(return_value=None)  # SETNX failed
        with pytest.raises(RuntimeError, match="Lock contention"):
            async with b.lock("node:contested"):
                pass

    @pytest.mark.asyncio
    async def test_lock_releases_own_key(self, connected_bus):
        b, mock_redis = connected_bus
        captured = {}

        async def capture_set(key, val, **_):
            captured["key"] = key
            captured["val"] = val
            return True

        mock_redis.set.side_effect = capture_set
        mock_redis.get = AsyncMock(side_effect=lambda k: captured.get("val"))
        mock_redis.delete = AsyncMock()

        async with b.lock("release-test"):
            pass

        mock_redis.delete.assert_called_once_with(captured["key"])


class TestState:
    @pytest.mark.asyncio
    async def test_set_state_serialises_json(self, connected_bus):
        b, mock_redis = connected_bus
        await b.set_state("curator:stats", {"pruned": 5})
        args, kwargs = mock_redis.set.call_args
        assert "curator:stats" in args[0]
        assert json.loads(args[1]) == {"pruned": 5}
        assert kwargs.get("ex") == 3_600

    @pytest.mark.asyncio
    async def test_get_state_returns_none_for_missing_key(self, connected_bus):
        b, mock_redis = connected_bus
        mock_redis.get = AsyncMock(return_value=None)
        assert await b.get_state("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_state_deserialises_json(self, connected_bus):
        b, mock_redis = connected_bus
        mock_redis.get = AsyncMock(return_value=json.dumps({"key": "value"}))
        assert await b.get_state("existing") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_increment_counter(self, connected_bus):
        b, mock_redis = connected_bus
        mock_redis.incrby = AsyncMock(return_value=3)
        result = await b.increment_counter("cw_nodes_created_total")
        assert result == 3
        mock_redis.incrby.assert_called_once()
