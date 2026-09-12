"""Dependency injection middleware for storage and queue instances."""
from typing import Any, Awaitable, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from music_dna_agent.src.domain.ports import QueuePort, StoragePort


class DependencyInjectionMiddleware(BaseMiddleware):
    """Injects storage and queue ports into handler data dict."""

    def __init__(self, storage: StoragePort, queue: QueuePort):
        self.storage = storage
        self.queue = queue

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        data["storage"] = self.storage
        data["queue"] = self.queue
        return await handler(event, data)
