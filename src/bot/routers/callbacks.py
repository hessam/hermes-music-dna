"""Callback query router: Handles quality selection and job triggers."""
import asyncio
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery
from music_dna_agent.src.domain.models import JobStatus, QualityOption
from music_dna_agent.src.domain.ports import QueuePort, StoragePort
from music_dna_agent.src.bot.states.job_states import JobFlow

router = Router(name="callback_router")


@router.callback_query(F.data.startswith("qual:"))
async def handle_quality_selection(
    query: CallbackQuery,
    bot: Bot,
    state: FSMContext,
    storage: StoragePort,
    queue: QueuePort,
) -> None:
    parts = query.data.split(":")
    if len(parts) != 3:
        await query.answer("Invalid request.")
        return

    _, job_id, quality_str = parts
    job = await storage.get_job(job_id)
    if not job:
        await query.answer("Job not found.")
        return

    job.quality = QualityOption(quality_str)
    job.status = JobStatus.QUEUED
    await storage.save_job(job)

    # Push to processing queue
    await queue.enqueue_job(job_id)
    await state.set_state(JobFlow.processing)

    await query.answer("Job queued!")
    await query.message.edit_text(
        f"⏳ **Analyzing Track DNA...**\n"
        f"• **Job ID:** `{job_id[:8]}...`\n"
        f"• **Preset:** `{job.quality.value}`\n"
        f"• **Status:** Queued (Step 1/4: Demucs Stems)\n\n"
        f"_This typically takes 45–90 seconds on our 4-core server..._",
        parse_mode="Markdown",
    )


@router.callback_query(F.data.startswith("cancel:"))
async def handle_cancel_selection(
    query: CallbackQuery,
    state: FSMContext,
    storage: StoragePort,
) -> None:
    parts = query.data.split(":")
    job_id = parts[1] if len(parts) > 1 else ""

    if job_id:
        job = await storage.get_job(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            await storage.save_job(job)

    await state.clear()
    await state.set_state(JobFlow.waiting_for_audio)
    await query.answer("Cancelled.")
    await query.message.edit_text("❌ Analysis cancelled. Upload a new track anytime.")
