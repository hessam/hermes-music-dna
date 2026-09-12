"""Owner-only security middleware for single-user bot isolation."""
from typing import Any, Awaitable, Callable, Dict, Set
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User


class OwnerOnlyMiddleware(BaseMiddleware):
    """Rejects incoming updates if the user is not in the whitelist."""

    def __init__(self, allowed_users: Set[int]):
        self.allowed_users = allowed_users

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user: User = data.get("event_from_user")
        if not user:
            return await handler(event, data)

        if self.allowed_users and user.id not in self.allowed_users:
            if hasattr(event, "answer"):
                await event.answer("⛔ Access restricted to the system owner.")
            return

        return await handler(event, data)
