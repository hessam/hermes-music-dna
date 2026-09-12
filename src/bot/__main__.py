"""Bot entrypoint: Sets up aiogram Dispatcher, middlewares, and long-polling."""
from __future__ import annotations

import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import RedisStorage
from music_dna_agent.src.bot.config import BotConfig
from music_dna_agent.src.bot.middleware.owner_only import OwnerOnlyMiddleware
from music_dna_agent.src.bot.middleware.db_session import DependencyInjectionMiddleware
from music_dna_agent.src.bot.routers import core, job, callbacks, status, vocal_router
from music_dna_agent.src.adapters.infra.sqlite_storage import SQLiteStorage
from music_dna_agent.src.adapters.infra.redis_queue import RedisQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("music_dna_bot")


async def main() -> None:
    config = BotConfig()
    if not config.bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set in environment or config!")
        sys.exit(1)

    logger.info("Initializing Music DNA SQLite Storage & Redis Queue...")
    storage = SQLiteStorage(db_path=config.db_path)
    await storage.init_db()

    queue = RedisQueue(redis_url=config.redis_url)
    fsm_storage = RedisStorage.from_url(config.redis_url)

    bot = Bot(token=config.bot_token)
    dp = Dispatcher(storage=fsm_storage)

    # 1. Register Global Middlewares
    dp.update.outer_middleware(OwnerOnlyMiddleware(allowed_users=config.allowed_users_set))
    dp.update.middleware(DependencyInjectionMiddleware(storage=storage, queue=queue))

    # 2. Register Feature Routers
    dp.include_router(core.router)
    dp.include_router(job.router)
    dp.include_router(callbacks.router)
    dp.include_router(status.router)
    dp.include_router(vocal_router.router)  # Standalone Vocal Bible Designer

    logger.info("Starting aiogram long-polling for owner IDs: %s", config.allowed_users_set)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, storage=storage, queue=queue)


if __name__ == "__main__":
    asyncio.run(main())
