"""Isolated Telegram Router for 4-Layer Dual-Vocalist Master Engine."""
from __future__ import annotations
import logging
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from music_dna_agent.src.vocal_designer.vocal_bible import BibleRegistry
from music_dna_agent.src.vocal_designer.input_interpreter import InputInterpreter, InterpretationResult
from music_dna_agent.src.vocal_designer.suno_prompt_builder import SunoPromptBuilder

logger = logging.getLogger(__name__)
router = Router(name="vocal_router")
interpreter = InputInterpreter()

_active_user_sessions: dict[int, tuple[InterpretationResult, str]] = {}


def get_steering_keyboard(singer_slug: str = "powerful-dreamer") -> InlineKeyboardMarkup:
    if singer_slug == "velvet-blade":
        buttons = [
            [
                InlineKeyboardButton(text="🗡️ The Blade (+Power)", callback_data="steer:power:up"),
                InlineKeyboardButton(text="🕯️ The Siren (+Intimacy)", callback_data="steer:intimacy:up"),
                InlineKeyboardButton(text="🔮 The Oracle (Ethereal)", callback_data="steer:oracle:up"),
            ],
            [
                InlineKeyboardButton(text="🌫️ The Shadow (Dark)", callback_data="steer:shadow:up"),
                InlineKeyboardButton(text="🎯 Precise Diction", callback_data="steer:diction:precise"),
                InlineKeyboardButton(text="🥁 Syncopated Lock", callback_data="steer:rhythm:syncopated"),
            ],
        ]
    else:
        buttons = [
            [
                InlineKeyboardButton(text="⚡ +Power", callback_data="steer:power:up"),
                InlineKeyboardButton(text="🌌 +Dreamy", callback_data="steer:dreamy:up"),
                InlineKeyboardButton(text="🔥 +Grit", callback_data="steer:texture:up"),
            ],
            [
                InlineKeyboardButton(text="💔 +Intimacy", callback_data="steer:intimacy:up"),
                InlineKeyboardButton(text="🎯 Precise Diction", callback_data="steer:diction:precise"),
                InlineKeyboardButton(text="🥁 Syncopated Lock", callback_data="steer:rhythm:syncopated"),
            ],
        ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


import re as _re

_CLOSE_MAP_TAGS = {"b": "</b>", "i": "</i>", "code": "</code>", "pre": "</pre>"}

def _close_open_tags(chunk: str) -> str:
    for tag in ["pre", "code", "b", "i"]:
        opens = len(_re.findall(f"<{tag}(?:\\s[^>]*)?>", chunk))
        closes = len(_re.findall(f"</{tag}>", chunk))
        for _ in range(opens - closes):
            chunk += _CLOSE_MAP_TAGS[tag]
    return chunk

def _split_html_chunks(text: str, max_chunk_len: int = 3800) -> list[str]:
    if len(text) <= max_chunk_len:
        return [text]
    parts = text.split("\n\n")
    chunks = []
    current: list[str] = []
    current_len = 0
    accumulated = ""
    for p in parts:
        open_pre = accumulated.count("<pre>")
        close_pre = accumulated.count("</pre>")
        in_pre = open_pre > close_pre
        if in_pre or (current_len + len(p) + 2 <= max_chunk_len):
            current.append(p)
            current_len += len(p) + 2
            accumulated += p + "\n\n"
        else:
            if current:
                chunks.append(_close_open_tags("\n\n".join(current)))
            current = [p]
            current_len = len(p)
            accumulated = p + "\n\n"
    if current:
        chunks.append(_close_open_tags("\n\n".join(current)))
    return chunks


async def _safe_send_html(status_msg: Message, full_html: str, kb: InlineKeyboardMarkup, original_msg: Message) -> None:
    chunks = _split_html_chunks(full_html)
    try:
        if len(chunks) == 1:
            await status_msg.edit_text(chunks[0], reply_markup=kb, parse_mode="HTML")
        else:
            await status_msg.edit_text(chunks[0], parse_mode="HTML")
            for i, ch in enumerate(chunks[1:]):
                is_last = (i == len(chunks) - 2)
                await original_msg.answer(ch, reply_markup=kb if is_last else None, parse_mode="HTML")
    except Exception as err:
        logger.warning("HTML send failed (%s), falling back to plain text...", err)
        plain = full_html.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "").replace("<pre>", "").replace("</pre>", "").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        plain_chunks = _split_html_chunks(plain)
        if len(plain_chunks) == 1:
            await status_msg.edit_text(plain_chunks[0], reply_markup=kb, parse_mode=None)
        else:
            await status_msg.edit_text(plain_chunks[0], parse_mode=None)
            for i, ch in enumerate(plain_chunks[1:]):
                is_last = (i == len(plain_chunks) - 2)
                await original_msg.answer(ch, reply_markup=kb if is_last else None, parse_mode=None)


@router.message(Command("vocal"))
async def handle_vocal_command(message: Message) -> None:
    query = message.text.partition(" ")[2].strip() if message.text else ""
    if not query:
        help_msg = (
            "🎙️ <b>The Dual-Vocalist Master Designer (GPT-5.6 Luna)</b>\n\n"
            "• <code>/vocal &lt;song description, lyrics, or style&gt;</code> (The Powerful Dreamer - Male)\n"
            "• <code>/vocal_female &lt;song description or lyrics&gt;</code> (The Velvet Blade - Female)\n\n"
            "<b>Character Limits:</b>\n"
            "• Style Box: Max 1000 characters\n"
            "• Lyrics Box: Max 4000 characters (includes identity preamble + custom verses)\n\n"
            "<b>Example:</b>\n"
            "<code>/vocal 1970s progressive folk rock, 105 BPM, overdriven flute, [Verse 1 | lyrics...]</code>"
        )
        await message.answer(help_msg, parse_mode="HTML")
        return

    status = await message.answer("🧠 <b>GPT-5.6 Luna directing 4-layer vocal specification & Suno blueprint...</b>", parse_mode="HTML")

    try:
        interp = await interpreter.interpret(query, force_singer="powerful-dreamer")
        singer_slug = interp.singer_slug
        _active_user_sessions[message.from_user.id] = (interp, singer_slug)

        bible = BibleRegistry.get(singer_slug)
        result = SunoPromptBuilder.build(bible, interp, query)
        report_html = SunoPromptBuilder.format_html(result, query)
        kb = get_steering_keyboard(singer_slug)

        await _safe_send_html(status, report_html, kb, message)
    except Exception as e:
        logger.exception("Vocal prompt generation failed: %s", e)
        await status.edit_text(f"❌ <b>Vocal Director error:</b> <code>{e}</code>", parse_mode="HTML")


@router.message(Command("vocal_female"))
async def handle_vocal_female_command(message: Message) -> None:
    query = message.text.partition(" ")[2].strip() if message.text else ""
    if not query:
        query = "atmospheric progressive rock, intimate dark vocal"

    status = await message.answer("🧠 <b>GPT-5.6 Luna directing The Velvet Blade (Female Vocalist)...</b>", parse_mode="HTML")

    try:
        interp = await interpreter.interpret(query, force_singer="velvet-blade")
        singer_slug = "velvet-blade"
        _active_user_sessions[message.from_user.id] = (interp, singer_slug)

        bible = BibleRegistry.get(singer_slug)
        result = SunoPromptBuilder.build(bible, interp, query)
        report_html = SunoPromptBuilder.format_html(result, query)
        kb = get_steering_keyboard(singer_slug)

        await _safe_send_html(status, report_html, kb, message)
    except Exception as e:
        logger.exception("Vocal female generation failed: %s", e)
        await status.edit_text(f"❌ <b>Vocal Director error:</b> <code>{e}</code>", parse_mode="HTML")


@router.callback_query(F.data.startswith("steer:"))
async def handle_vector_steer(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    param = parts[1]
    action = parts[2]

    user_id = callback.from_user.id
    session = _active_user_sessions.get(user_id)

    if not session:
        await callback.answer("Session expired. Please send a new /vocal command.")
        return

    interp, singer_slug = session
    bible = BibleRegistry.get(singer_slug)

    if singer_slug == "velvet-blade":
        if param == "power":
            interp.power = min(10.0, interp.power + 1.5)
            interp.primary_mode_id = "blade"
            await callback.answer("The Blade (+Power) engaged")
        elif param == "intimacy":
            interp.intimacy = min(10.0, interp.intimacy + 1.5)
            interp.primary_mode_id = "siren"
            await callback.answer("The Siren (+Intimacy) engaged")
        elif param == "oracle":
            interp.primary_mode_id = "oracle"
            await callback.answer("The Oracle (Ethereal) engaged")
        elif param == "shadow":
            interp.primary_mode_id = "shadow"
            await callback.answer("The Shadow (Dark) engaged")
        elif param == "diction":
            interp.hard_constraints.diction_descriptor = "exceptionally clear, precise, intimate articulation"
            await callback.answer("Precise diction confirmed")
        elif param == "rhythm":
            interp.rhythm_displacement = 9.0
            interp.hard_constraints.rhythmic_descriptor = "hypnotic syncopated phrasing"
            await callback.answer("Syncopated lock confirmed")
    else:
        if param == "power":
            interp.power = min(10.0, interp.power + 1.5)
            interp.primary_mode_id = "commander"
            await callback.answer(f"Power increased to {interp.power:.0f}/10")
        elif param == "dreamy":
            interp.intimacy = min(10.0, interp.intimacy + 1.5)
            interp.power = max(2.5, interp.power - 1.5)
            interp.primary_mode_id = "dreamer"
            await callback.answer(f"Dreamy Intimacy increased to {interp.intimacy:.0f}/10")
        elif param == "texture":
            interp.texture = min(10.0, interp.texture + 2.0)
            interp.primary_mode_id = "beast"
            await callback.answer(f"Texture/Grit increased to {interp.texture:.0f}/10")
        elif param == "intimacy":
            interp.intimacy = min(10.0, interp.intimacy + 2.0)
            interp.primary_mode_id = "wound"
            await callback.answer(f"Intimacy set to {interp.intimacy:.0f}/10")
        elif param == "diction":
            interp.hard_constraints.diction_descriptor = "exceptionally clear diction"
            await callback.answer("Clear diction confirmed")
        elif param == "rhythm":
            interp.rhythm_displacement = 9.0
            interp.hard_constraints.rhythmic_descriptor = "syncopated off-beat vocal phrasing"
            await callback.answer("Syncopated off-beat lock enabled")

    result = SunoPromptBuilder.build(bible, interp)
    report_html = SunoPromptBuilder.format_html(result, "Steered Vector")
    kb = get_steering_keyboard(singer_slug)

    if callback.message:
        try:
            await callback.message.edit_text(report_html, reply_markup=kb, parse_mode="HTML")
        except Exception:
            plain = report_html.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "").replace("<pre>", "").replace("</pre>", "").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            await callback.message.edit_text(plain, reply_markup=kb, parse_mode=None)


@router.message(Command("vocal_bible"))
async def handle_vocal_bible_info(message: Message) -> None:
    male_bible = BibleRegistry.get("powerful-dreamer")
    fem_bible = BibleRegistry.get("velvet-blade")

    info = (
        f"📖 <b>THE DUAL-VOCALIST MASTER BIBLE</b>\n\n"
        f"🎙️ <b>1. {male_bible.name.upper()} (MALE)</b>\n"
        f"• <b>Essence:</b> <i>Power held under control</i>\n"
        f"• <b>Voice:</b> <code>{male_bible.identity_card.voice_type}</code>\n"
        f"• <b>Core:</b> {male_bible.identity_card.vocal_core}\n"
        f"• <b>Dynamics:</b> {male_bible.identity_card.vocal_dynamics}\n"
        f"• <b>Modes:</b> Beast · Commander · Showman · Storyteller · Dreamer · Wound\n\n"
        f"🗡️ <b>2. {fem_bible.name.upper()} (FEMALE)</b>\n"
        f"• <b>Essence:</b> <i>Desire held under control (Contained danger)</i>\n"
        f"• <b>Voice:</b> <code>{fem_bible.identity_card.voice_type}</code>\n"
        f"• <b>Core:</b> {fem_bible.identity_card.vocal_core}\n"
        f"• <b>Dynamics:</b> {fem_bible.identity_card.vocal_dynamics}\n"
        f"• <b>Modes:</b> Siren · Shadow · Oracle · Blade · Flame · Confession\n\n"
        f"👉 Use <code>/vocal &lt;prompt&gt;</code> for Male, or <code>/vocal_female &lt;prompt&gt;</code> for Female."
    )
    await message.answer(info, parse_mode="HTML")
