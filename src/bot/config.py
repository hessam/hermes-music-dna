"""Bot configuration and environment variable settings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Set
from pydantic import BaseModel, Field


def load_dotenv(env_path: str = "/workspace/workspace/music_dna_agent/.env") -> None:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k not in os.environ:
                    os.environ[k] = v


class BotConfig(BaseModel):
    bot_token: str = Field(default="")
    allowed_user_ids: str = Field(default="92241363")
    redis_url: str = Field(default="redis://hermes-music-dna-redis:6379/0")
    db_path: str = Field(default="/workspace/jobs/music_dna.db")
    workspace_root: str = Field(default="/workspace/jobs")

    def __init__(self, **data):
        load_dotenv()
        super().__init__(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            allowed_user_ids=os.environ.get("ALLOWED_USER_IDS", "92241363,Telehessam"),
            redis_url=os.environ.get("REDIS_URL", "redis://hermes-music-dna-redis:6379/0"),
            db_path=os.environ.get("DB_PATH", "/workspace/jobs/music_dna.db"),
            workspace_root=os.environ.get("WORKSPACE_ROOT", "/workspace/jobs"),
            **data,
        )

    @property
    def allowed_users_set(self) -> Set[int]:
        res = set()
        for item in self.allowed_user_ids.split(","):
            item = item.strip()
            if item.isdigit():
                res.add(int(item))
        return res
