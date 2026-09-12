"""Background Pipeline Worker: Pops jobs from Redis, runs synthesis & multi-asset extraction."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import FSInputFile
from music_dna_agent.src.domain.models import JobStatus, QualityOption
from music_dna_agent.src.bot.config import BotConfig
from music_dna_agent.src.adapters.infra.sqlite_storage import SQLiteStorage
from music_dna_agent.src.adapters.infra.redis_queue import RedisQueue
from music_dna_agent.src.adapters.infra.filesystem import FilesystemManager
from music_dna_agent.src.adapters.generators.suno_adapter import SunoAdapter
from music_dna_agent.src.pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("music_dna_worker")


async def send_chunked_message(bot: Bot, chat_id: int, text: str, parse_mode: str = "Markdown") -> None:
    """Safely sends long markdown messages split into <= 4000 character chunks with plain-text fallback."""
    max_len = 3800
    chunks = []
    curr = text
    while curr:
        if len(curr) <= max_len:
            chunks.append(curr)
            break
        split_idx = curr.rfind("\n", 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        chunks.append(curr[:split_idx])
        curr = curr[split_idx:].lstrip("\n")

    for chunk in chunks:
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
        except TelegramBadRequest as tb_err:
            logger.warning("Markdown formatting rejected by Telegram (%s). Falling back to plain text.", tb_err)
            # Strip markdown backticks/asterisks and send plain text
            plain = chunk.replace("```", "").replace("`", "").replace("**", "").replace("*", "")
            await bot.send_message(chat_id=chat_id, text=plain)


async def main() -> None:
    config = BotConfig()
    storage = SQLiteStorage(db_path=config.db_path)
    await storage.init_db()

    queue = RedisQueue(redis_url=config.redis_url)
    fs_manager = FilesystemManager(root_dir=config.workspace_root)
    orchestrator = PipelineOrchestrator(storage=storage, queue=queue, fs_manager=fs_manager)

    bot = Bot(token=config.bot_token) if config.bot_token else None

    logger.info("🚀 Music DNA Background Worker started. Listening for jobs on %s...", config.redis_url)

    while True:
        try:
            job_id = await queue.pop_job(timeout=3)
            if not job_id:
                await asyncio.sleep(0.5)
                continue

            job = await storage.get_job(job_id)
            if not job:
                logger.warning("Job %s not found in SQLite storage. Skipping.", job_id)
                continue

            logger.info("Processing job %s for user %s (%s, Mode: %s)...", job.id, job.user_id, job.original_filename, job.quality.value)

            try:
                processed_job = await orchestrator.run(job)

                if bot and processed_job.chat_id:
                    job_dir = fs_manager.get_job_dir(processed_job.id)
                    voice_dir = os.path.join(job_dir, "voice")
                    analysis_dir = os.path.join(job_dir, "analysis")

                    # 1. Send Rich Summary Message
                    summary_md = SunoAdapter.format_summary_markdown(processed_job.dna) if processed_job.dna else "DNA analysis complete."
                    await send_chunked_message(bot, processed_job.chat_id, summary_md)

                    # 2. Send Clean Vocal Reference Audio
                    clean_ref_path = processed_job.output_manifest.get(
                        "isolated_vocals", os.path.join(voice_dir, "clean_vocal_reference.wav"))
                    if os.path.exists(clean_ref_path):
                        vocal_audio = FSInputFile(clean_ref_path, filename=f"CleanVocal_{processed_job.original_filename}.wav")
                        await bot.send_audio(
                            chat_id=processed_job.chat_id,
                            audio=vocal_audio,
                            caption="🎙️ Separated vocal stem — original separator output, without added vocal effects.",
                        )

                    # 3. Send Guide Track (if synthesized)
                    melody_path = processed_job.output_manifest.get("melody_guide", processed_job.guide_track_path)
                    if melody_path and os.path.exists(melody_path):
                        audio_file = FSInputFile(melody_path, filename=f"GuideTrack_{processed_job.original_filename}.wav")
                        await bot.send_audio(
                            chat_id=processed_job.chat_id,
                            audio=audio_file,
                            caption=f"🎹 Melody guide from extracted notes ({processed_job.original_filename})",
                        )

                    # 4. Send Analysis JSON Document
                    vocal_json_path = os.path.join(analysis_dir, "vocal_dna.json")
                    if os.path.exists(vocal_json_path):
                        doc_file = FSInputFile(vocal_json_path, filename="vocal_dna.json")
                        await bot.send_document(
                            chat_id=processed_job.chat_id,
                            document=doc_file,
                            caption="🧬 Full Acoustic Vocal DNA Profile (JSON)",
                        )

            except Exception as exc:
                logger.error("Job %s processing failed: %s", job.id, exc)
                if bot and job.chat_id:
                    await bot.send_message(
                        chat_id=job.chat_id,
                        text=f"❌ **Analysis Failed:** `{str(exc)}`\n\nPlease try uploading a clearer audio sample.",
                        parse_mode="Markdown",
                    )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Worker loop exception: %s", e)
            await asyncio.sleep(2)

    if bot:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
