"""Redis FIFO Queue and PubSub Broker implementation."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict
import redis.asyncio as aioredis
from music_dna_agent.src.domain.models import StepName
from music_dna_agent.src.domain.ports import QueuePort

QUEUE_NAME = "music_dna:jobs"
CHANNEL_PREFIX = "music_dna:progress:"


class RedisQueue(QueuePort):
    """Async Redis broker for job scheduling and live progress broadcasting."""

    def __init__(self, redis_url: str = "redis://hermes-music-dna-redis:6379/0"):
        self._redis_url = redis_url
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def enqueue_job(self, job_id: str) -> None:
        client = await self._get_client()
        await client.rpush(QUEUE_NAME, job_id)

    async def pop_job(self, timeout: int = 5) -> Optional[str]:
        client = await self._get_client()
        res = await client.blpop(QUEUE_NAME, timeout=timeout)
        if res:
            _, job_id = res
            return job_id
        return None

    async def publish_progress(self, job_id: str, step: StepName, pct: int, message: str) -> None:
        client = await self._get_client()
        channel = f"{CHANNEL_PREFIX}{job_id}"
        payload = {
            "job_id": job_id,
            "step": step.value,
            "pct": pct,
            "message": message,
        }
        await client.publish(channel, json.dumps(payload))

    async def subscribe_progress(self, job_id: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        client = await self._get_client()
        pubsub = client.pubsub()
        channel = f"{CHANNEL_PREFIX}{job_id}"
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await callback(data)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
