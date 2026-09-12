"""Status and history commands router."""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from music_dna_agent.src.domain.ports import StoragePort

router = Router(name="status_router")


@router.message(Command("history"))
async def cmd_history(message: Message, storage: StoragePort) -> None:
    jobs = await storage.list_jobs_by_user(message.from_user.id, limit=5)
    if not jobs:
        await message.answer("ℹ️ You have no previous analysis jobs.")
        return

    lines = ["📜 **Your Recent Guide Tracks:**\n"]
    for j in jobs:
        dna_info = f"({j.dna.key} {j.dna.scale}, {round(j.dna.bpm)} BPM)" if j.dna else ""
        lines.append(
            f"• **`{j.original_filename}`** — `{j.status.value}` {dna_info}\n"
            f"  ID: `{j.id[:8]}...` | Preset: `{j.quality.value}`"
        )

    await message.answer("\n".join(lines), parse_mode="Markdown")
