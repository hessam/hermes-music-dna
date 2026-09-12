"""LLM-powered Input Parser & Music Producer driven by GPT-5.6 Luna with 4 Laws of Vocal Anchoring."""
from __future__ import annotations
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-5.6-luna"
FALLBACK_KEY = os.environ.get("OPENROUTER_API_KEY", "")

INLINE_CUES = {"breath", "whisper", "sigh", "pause", "gasp", "laughter", "humming", "throat clearing", "applause"}

# Mode ID aliases — maps any LLM hallucinated mode name to the canonical one
_MALE_MODE_ALIASES = {
    "storyteller": "storyteller", "wound": "wound", "beast": "beast",
    "commander": "commander", "showman": "showman", "dreamer": "dreamer",
    "intimate": "wound", "dark": "wound", "powerful": "beast", "epic": "commander",
}
_FEMALE_MODE_ALIASES = {
    "siren": "siren", "shadow": "shadow", "oracle": "oracle",
    "blade": "blade", "flame": "flame", "confession": "confession",
    "confessional": "confession", "dark": "shadow", "luminous": "flame",
    "intimate": "siren", "powerful": "blade", "epic": "flame",
    "storyteller": "siren", "wound": "confession", "dreamer": "oracle",
}

def _normalize_mode(singer_slug: str, mode_id: str) -> str:
    """Maps any LLM-hallucinated mode name to a valid canonical mode ID. Empty → first mode."""
    if not mode_id:
        return "siren" if ("velvet" in singer_slug or "female" in singer_slug) else "storyteller"
    if "velvet" in singer_slug or "female" in singer_slug:
        return _FEMALE_MODE_ALIASES.get(mode_id, list(_FEMALE_MODE_ALIASES.values())[0])
    return _MALE_MODE_ALIASES.get(mode_id, list(_MALE_MODE_ALIASES.values())[0])




def is_inline_cue(tag_content: str) -> bool:
    clean = tag_content.strip().lower()
    return clean in INLINE_CUES or clean.startswith("breath") or clean.startswith("whisper")


@dataclass
class SectionConstraint:
    section_name: str
    is_instrumental: bool = False
    vocal_token: Optional[str] = None          # 1-2 words e.g. "low-chest", "luminous mix"
    arrangement_token: Optional[str] = None    # 1-2 words e.g. "sparse acoustic", "organ swells"
    raw_tag: Optional[str] = None
    lyrics_lines: List[str] = field(default_factory=list)


@dataclass
class HardConstraints:
    genre: str
    tempo_bpm: str
    instruments: List[str] = field(default_factory=list)
    production: str = "vintage tape warmth"
    vocal_register: str = "high-baritone"
    vocal_core: str = "dense forward chest core"


@dataclass
class InterpretationResult:
    has_explicit_sections: bool
    sections: List[SectionConstraint]
    hard_constraints: HardConstraints
    singer_slug: str
    primary_mode_id: str
    secondary_mode_id: str
    power: float
    texture: float
    intimacy: float
    rhythm_displacement: float
    current_state_summary: str


SYSTEM_PROMPT = """You are a World-Class Music Producer and vocal director. Your ONLY job is to analyze the user's actual input — their lyrics, emotions, genre hints, and song structure — and deduce the correct musical world from SCRATCH.

CRITICAL RULE: NEVER copy values from this prompt's examples. ALL field values must be deduced entirely from the user's input.

THE 4 LAWS OF VOCAL ANCHORING:
1. EXPLICIT GENDER & REGISTER (set by Preferred Singer hint — do not override):
   - powerful-dreamer → vocal_register: "high-baritone", vocal_core: "dense forward chest core"
   - velvet-blade → vocal_register: "contralto", vocal_core: "dry dense low-chest"
2. DESCRIPTORS IN THE 4-7 SWEET SPOT: Keep tokens concise. Never stack adjectives.
3. CONCISE SECTION TAGS (2-3 WORDS MAX per token): Suno attention decays after 3-4 words.
   - vocal_token examples: "low-chest", "luminous mix", "intimate close-mic", "yearning build", "skyward release"
   - arrangement_token examples: "sparse piano", "heavy groove", "syncopated strings", "ambient pads"
4. SONG WORLD DEDUCTION — from the user's input, deduce:
   - genre: Decade + specific genre (e.g. "2000s Persian art pop", "1990s dream pop", "1980s new wave")
   - instruments: 3 REAL PHYSICAL instruments that match the song's era, culture, and emotional world
   - tempo_bpm: Exact BPM number matching the song's energy (slow ballad=60-80, mid=90-110, upbeat=120-140)
   - production: Preserve explicitly requested production; otherwise describe a suitable production without imposing an era or tape saturation.

DEDUCTION RULES:
- Explicit user genre, tempo, instrumentation and production instructions take priority over every deduction below. Do not rewrite supplied lyrics or invent lyrics for a description-only input.
- If the user provides Persian/Iranian lyrics → deduce a Persian or Middle Eastern influenced genre (e.g. "2000s Persian rock", "contemporary Persian folk", "Iranian art pop")
- If the user mentions a specific artist or style → match that exact world
- If no genre is mentioned, read the emotional texture of the lyrics to deduce (dark/intimate → dream pop / art rock; longing → folk/ballad; aggressive → hard rock)
- Sections must reflect the ACTUAL sections in the user's input, not a template
- vocal_token per section must match the EMOTIONAL CONTENT of that section (not a generic label)

PERFORMANCE MODES (pick primary + secondary based on emotional content):
powerful-dreamer modes: beast, commander, showman, storyteller, dreamer, wound
velvet-blade modes: siren, shadow, oracle, blade, flame, confession

OUTPUT STRICT JSON (no markdown, no explanation, raw JSON only):
{
  "singer_slug": "powerful-dreamer",
  "hard_constraints": {
    "genre": "<deduce from input>",
    "tempo_bpm": "<deduce from input> BPM",
    "instruments": ["<instrument 1>", "<instrument 2>", "<instrument 3>"],
    "production": "vintage tape warmth",
    "vocal_register": "<set by singer_slug>",
    "vocal_core": "<set by singer_slug>"
  },
  "sections": [
    {
      "section_name": "<exact section name from input>",
      "is_instrumental": false,
      "vocal_token": "<2 words max, matches this section emotion>",
      "arrangement_token": "<2 words max, real instruments>"
    }
  ],
  "primary_mode_id": "<mode from list above>",
  "secondary_mode_id": "<mode from list above>",
  "power": 5.0,
  "texture": 3.0,
  "intimacy": 8.5,
  "rhythm_displacement": 9.0,
  "current_state_summary": "<1 sentence describing deduced song world and emotional arc>"
}"""


def _clean_and_parse_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except Exception:
        pass

    repaired = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(repaired)
    except Exception as err:
        logger.error("JSON repair failed: %s\nRAW:\n%s", err, raw[:400])
        raise err


def extract_sections_and_lyrics(raw_text: str) -> List[Dict[str, Any]]:
    """Extracts true song sections while preserving inline cues like [Breath] in lyrics."""
    pattern = re.compile(r"\[([^\]]+)\]")
    matches = list(pattern.finditer(raw_text))

    section_matches = []
    for m in matches:
        tag_text = m.group(1).strip()
        if not is_inline_cue(tag_text):
            section_matches.append(m)

    if not section_matches:
        return []

    sections = []
    for i, match in enumerate(section_matches):
        full_tag = match.group(1).strip()
        start_pos = match.end()
        end_pos = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(raw_text)

        body = raw_text[start_pos:end_pos].strip()
        lines = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("///")]

        tag_parts = [p.strip() for p in full_tag.split("|") if p.strip()]
        sec_name = tag_parts[0] if tag_parts else f"Section {i+1}"

        sections.append({
            "section_name": sec_name,
            "raw_tag": full_tag,
            "tag_parts": tag_parts[1:] if len(tag_parts) > 1 else [],
            "lyrics_lines": lines,
        })
    return sections


class InputInterpreter:
    """Interprets raw text prompts dynamically using GPT-5.6 Luna via OpenRouter."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or FALLBACK_KEY

    async def interpret(self, user_input: str, force_singer: Optional[str] = None) -> InterpretationResult:
        if not user_input or len(user_input.strip()) < 3:
            raise ValueError("Empty or too-short prompt. Please provide lyrics, a genre, or a song description.")

        raw_parsed_sections = extract_sections_and_lyrics(user_input)

        try:
            res = await self._call_luna(user_input, force_singer)
            if res:
                if raw_parsed_sections:
                    # The input owns section order and lyrics; inference adds direction only.
                    remaining = list(res.sections)
                    merged_sections = []
                    for raw in raw_parsed_sections:
                        matched = next((section for section in remaining
                                        if section.section_name.casefold() == raw["section_name"].casefold()), None)
                        if matched is not None:
                            remaining.remove(matched)
                        merged_sections.append(SectionConstraint(
                            section_name=raw["section_name"],
                            is_instrumental=bool(matched and matched.is_instrumental and not raw["lyrics_lines"]),
                            vocal_token=matched.vocal_token if matched else None,
                            arrangement_token=matched.arrangement_token if matched else None,
                            raw_tag=raw["raw_tag"],
                            lyrics_lines=raw["lyrics_lines"],
                        ))
                    res.sections = merged_sections
                    res.has_explicit_sections = True

                if force_singer:
                    res.singer_slug = force_singer

                logger.info("GPT-5.6 Luna dynamic interpretation complete: singer=%s primary=%s secondary=%s", res.singer_slug, res.primary_mode_id, res.secondary_mode_id)
                return res
        except Exception as e:
            logger.error("GPT-5.6 Luna call failed: %s", e)
            raise RuntimeError(f"GPT-5.6 Luna inference error: {e}")

        raise RuntimeError("No response from GPT-5.6 Luna.")

    async def _call_luna(self, user_input: str, force_singer: Optional[str] = None) -> Optional[InterpretationResult]:
        import aiohttp
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/music-dna-agent",
            "X-Title": "Music DNA Vocal Designer (GPT-5.6 Luna)",
        }
        hint = f" Preferred Singer: {force_singer}." if force_singer else ""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Direct this vocal performance dynamically:{hint}\n\n{user_input}"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 3500,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("OpenRouter GPT-5.6 Luna returned %s: %s", resp.status, text)
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _clean_and_parse_json(content)

                raw_secs = parsed.get("sections", [])
                sections: List[SectionConstraint] = []
                for s in raw_secs:
                    sections.append(SectionConstraint(
                        section_name=s.get("section_name", "Section"),
                        is_instrumental=bool(s.get("is_instrumental", False)),
                        vocal_token=s.get("vocal_token"),
                        arrangement_token=s.get("arrangement_token"),
                        raw_tag=s.get("raw_tag"),
                        lyrics_lines=s.get("lyrics_lines", []),
                    ))

                hc_data = parsed.get("hard_constraints", {})
                _genre = hc_data.get("genre")
                _bpm = hc_data.get("tempo_bpm")
                _instruments = hc_data.get("instruments")
                if not _genre or not _bpm or not _instruments:
                    raise RuntimeError(
                        f"Luna returned incomplete hard_constraints — missing: "
                        f"{'genre ' if not _genre else ''}"
                        f"{'tempo_bpm ' if not _bpm else ''}"
                        f"{'instruments' if not _instruments else ''}. "
                        f"Raw: {hc_data}"
                    )
                hard_constraints = HardConstraints(
                    genre=_genre,
                    tempo_bpm=_bpm,
                    instruments=_instruments,
                    production=hc_data.get("production", "vintage tape warmth"),
                    vocal_register=hc_data.get("vocal_register", ""),
                    vocal_core=hc_data.get("vocal_core", ""),
                )

                singer = parsed.get("singer_slug", "powerful-dreamer").lower()
                if "velvet" in singer or "female" in singer:
                    singer = "velvet-blade"
                else:
                    singer = "powerful-dreamer"

                return InterpretationResult(
                    has_explicit_sections=parsed.get("has_explicit_sections", bool(sections)),
                    sections=sections,
                    hard_constraints=hard_constraints,
                    singer_slug=singer,
                    primary_mode_id=_normalize_mode(singer, parsed.get("primary_mode_id", "").lower()),
                    secondary_mode_id=_normalize_mode(singer, parsed.get("secondary_mode_id", "").lower()),
                    power=float(parsed.get("power", 5.0)),
                    texture=float(parsed.get("texture", 3.0)),
                    intimacy=float(parsed.get("intimacy", 5.0)),
                    rhythm_displacement=float(parsed.get("rhythm_displacement", 5.0)),
                    current_state_summary=parsed.get("current_state_summary", ""),
                )
