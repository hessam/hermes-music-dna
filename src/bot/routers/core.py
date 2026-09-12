"""Core commands router (/start, /help, /cancel)."""
from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from music_dna_agent.src.bot.states.job_states import JobFlow

router = Router(name="core_router")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.set_state(JobFlow.waiting_for_audio)
    welcome_text = (
        "👋 **Welcome to Music DNA Studio!** 🧬🎹\n\n"
        "I extract the essential musical DNA from any song and synthesize a clean, "
        "high-definition guide track for Suno AI.\n\n"
        "**4-Step Synthesis Pipeline:**\n"
        "1️⃣ **Stem Isolation:** Demucs 4-stem vocal/bass/drums split\n"
        "2️⃣ **Pitch & Melody:** Basic Pitch neural transcription\n"
        "3️⃣ **Groove & Chords:** Librosa micro-timing & harmonic analysis\n"
        "4️⃣ **Master Synthesis:** FluidSynth virtual instruments + Suno seed bounce\n\n"
        "🚀 **How to use:**\n"
        "Simply send me any audio file (`.mp3`, `.wav`, `.m4a`, or voice message) to begin!"
    )
    await message.answer(welcome_text, parse_mode="Markdown")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    help_text = (
        "📖 **Music DNA Commands:**\n\n"
        "• `/start` — Restart bot dialog\n"
        "• `/history` — View your recent processed guide tracks\n"
        "• `/cancel` — Cancel ongoing operation\n\n"
        "💡 **Pro-Tip:** For the best Suno results, send a clean studio recording or acoustic demo."
    )
    await message.answer(help_text, parse_mode="Markdown")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Nothing to cancel.")
        return

    await state.clear()
    await state.set_state(JobFlow.waiting_for_audio)
    await message.answer("❌ Operation cancelled. Send a new audio file when you are ready.")
