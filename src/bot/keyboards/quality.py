"""Inline keyboard builders for quality and analysis mode selection."""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def get_quality_keyboard(job_id: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="🎯 Crisp 30s (Seed)",
            callback_data=f"qual:{job_id}:crisp_30s",
        ),
        InlineKeyboardButton(
            text="🎼 Full 60s (Deep Ref)",
            callback_data=f"qual:{job_id}:full_60s",
        ),
    )
    builder.row(
        InlineKeyboardButton(
            text="🎙️ Vocal DNA Clone (Suno Voice)",
            callback_data=f"qual:{job_id}:vocal_dna",
        )
    )
    builder.row(
        InlineKeyboardButton(
            text="❌ Cancel",
            callback_data=f"cancel:{job_id}",
        )
    )
    return builder.as_markup()
