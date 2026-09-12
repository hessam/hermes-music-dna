"""Job intake router: Receives audio files, downloads them, and initiates FSM."""
import os
import uuid
import logging
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Document, Message
from music_dna_agent.src.domain.models import Job, JobStatus, QualityOption
from music_dna_agent.src.domain.ports import StoragePort
from music_dna_agent.src.adapters.infra.filesystem import FilesystemManager
from music_dna_agent.src.bot.keyboards.quality import get_quality_keyboard
from music_dna_agent.src.bot.states.job_states import JobFlow

logger = logging.getLogger(__name__)
router = Router(name="job_router")
fs_manager = FilesystemManager()

ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/flac", "audio/aac"
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB Telegram standard API limit


@router.message(F.audio | F.voice | F.document)
async def handle_audio_upload(
    message: Message,
    bot: Bot,
    state: FSMContext,
    storage: StoragePort,
) -> None:
    # 1. Validate file format
    file_id = None
    file_name = "audio.wav"
    file_size = 0

    if message.audio:
        file_id = message.audio.file_id
        file_name = message.audio.file_name or f"audio_{message.audio.file_unique_id}.mp3"
        file_size = message.audio.file_size or 0
    elif message.voice:
        file_id = message.voice.file_id
        file_name = f"voice_{message.voice.file_unique_id}.ogg"
        file_size = message.voice.file_size or 0
    elif message.document:
        if message.document.mime_type not in ALLOWED_MIME_TYPES and not message.document.file_name.lower().endswith(
            (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")
        ):
            await message.answer("⚠️ Please upload a valid audio file (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`).")
            return
        file_id = message.document.file_id
        file_name = message.document.file_name or f"doc_{message.document.file_unique_id}.mp3"
        file_size = message.document.file_size or 0

    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        await message.answer(
            f"⚠️ **File is too large ({size_mb:.1f} MB)**.\n\n"
            f"Telegram Bot API limits bot downloads to 20 MB. Please upload a compressed audio file (e.g. MP3 or smaller WAV) under 20 MB.",
            parse_mode="Markdown",
        )
        return

    # 2. Prepare workspace & download file
    job_id = str(uuid.uuid4())
    job_dir = fs_manager.get_job_dir(job_id)
    dest_path = os.path.join(job_dir, file_name)

    status_msg = await message.answer("📥 **Downloading audio file...**", parse_mode="Markdown")

    try:
        file_info = await bot.get_file(file_id)
        await bot.download_file(file_info.file_path, dest_path)
    except Exception as e:
        logger.exception("Failed to download audio file from Telegram: %s", e)
        await status_msg.edit_text(
            f"❌ **Download failed:** Telegram reported: `{e}`\n"
            f"Please ensure the file is under 20 MB and try again.",
            parse_mode="Markdown",
        )
        return

    # 3. Create job in storage
    job = Job(
        id=job_id,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        original_filename=file_name,
        input_file_path=dest_path,
        quality=QualityOption.CRISP_30S,
        status=JobStatus.QUEUED,
    )
    await storage.save_job(job)

    # 4. Set FSM and prompt quality selection
    await state.set_state(JobFlow.confirming_options)
    await state.update_data(job_id=job_id, status_message_id=status_msg.message_id)

    kb = get_quality_keyboard(job_id)
    await status_msg.edit_text(
        f"🎵 **Audio Received:** `{file_name}`\n\n"
        f"Choose guide track synthesis quality:",
        reply_markup=kb,
        parse_mode="Markdown",
    )


@router.message(Command("reprompt"))
async def handle_reprompt(
    message: Message,
    storage: StoragePort,
) -> None:
    """Re-generate Suno prompts from the last analyzed song — no re-processing."""
    from music_dna_agent.src.adapters.generators.suno_adapter import SunoAdapter

    jobs = await storage.list_jobs_by_user(message.from_user.id, limit=10)
    last_with_dna = next(
        (j for j in jobs if j.dna is not None),
        None,
    )

    if not last_with_dna:
        await message.answer(
            "❌ No analyzed song found. Send me an audio file first and I'll extract its DNA.",
        )
        return

    fname = last_with_dna.original_filename or "unknown"
    await message.answer(
        f"🔄 Re-generating prompts from: `{fname}`...",
        parse_mode="Markdown",
    )

    try:
        report = SunoAdapter.format_summary_markdown(last_with_dna.dna)
        await message.answer(report, parse_mode="Markdown")
    except Exception as e:
        logger.exception("Reprompt failed: %s", e)
        await message.answer(f"❌ Failed to generate prompts: `{e}`", parse_mode="Markdown")

