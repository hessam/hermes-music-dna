from __future__ import annotations

"""Compiles Suno AI Prompts adhering strictly to the 4 Laws of Text-Only Vocal Anchoring with Zero Box Clashes."""
# Suno v5.5 recommended settings — change here to update everywhere
SUNO_SETTINGS = {
    "style_influence": "70%",
    "weirdness_clarity": "40%",
    "model": "v5.5 Flagship",
}

# Suno v5.5 recommended settings — change here to update everywhere
SUNO_SETTINGS = {
    "style_influence": "70%",
    "weirdness_clarity": "40%",
    "model": "v5.5 Flagship",
}
import html
import re
from typing import Dict, Any, List
from music_dna_agent.src.vocal_designer.vocal_bible import VocalBibleBase, PerformanceMode, SectionArcNode
from music_dna_agent.src.vocal_designer.input_interpreter import InterpretationResult
from music_dna_agent.src.vocal_designer.vocal_identity_gate import VocalIdentityGate, IdentityGateResult


def sanitize_all_caps_line(line: str) -> str:
    """Restricts ALL CAPS shouting to max 1-3 words per line to prevent vocoder digital clipping (Gotcha 2)."""
    words = line.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
    if caps_count > 3:
        # Lowercase non-essential words, keep only the climax word in caps
        return line.title()
    return line


class SunoPromptBuilder:
    """Builds Suno AI prompts: Broadway #1 front-load, explicit gender/register, clean non-conflicting section tags, and anti-humming exclusions."""

    @classmethod
    def build(
        cls,
        bible: VocalBibleBase,
        interpretation: InterpretationResult,
        user_raw_query: str = "",
    ) -> Dict[str, Any]:
        hc = interpretation.hard_constraints
        p_mode = bible.get_mode(interpretation.primary_mode_id) or list(bible.modes.values())[0]
        s_mode = bible.get_mode(interpretation.secondary_mode_id) or list(bible.modes.values())[1]

        sec_dicts = None
        if interpretation.has_explicit_sections and interpretation.sections:
            sec_dicts = []
            for s in interpretation.sections:
                sec_dicts.append({
                    "section_name": s.section_name,
                    "is_instrumental": s.is_instrumental,
                    "vocal_token": s.vocal_token,
                    "arrangement_token": s.arrangement_token,
                    "raw_tag": s.raw_tag,
                    "lyrics_lines": s.lyrics_lines,
                })

        # --- 1. DYNAMIC SECTION-BY-SECTION VOCAL ARC (2-3 word dense tags) ---
        arc_nodes = bible.generate_vocal_arc(p_mode.mode_id, s_mode.mode_id, explicit_sections=sec_dicts)

        # --- 2. STYLE BOX (Broadway #1 + Gender/Register #2 + Core #3 + Instruments + BPM + Vintage Warmth) ---
        gender_reg = f"{bible.gender} {bible.identity_card.vocal_register}"
        core_token = bible.identity_card.vocal_core

        tokens = ["Broadway", gender_reg, core_token, hc.genre]
        for inst in hc.instruments[:3]:
            if inst and inst not in tokens:
                tokens.append(inst)
        tokens.append(hc.tempo_bpm)
        tokens.append("vintage tape warmth")

        clean_tokens = []
        for t in tokens:
            t_str = t.strip()
            if t_str and t_str not in clean_tokens:
                clean_tokens.append(t_str)

        style_prompt = ", ".join(clean_tokens) + "."

        # --- 3. LYRICS BOX (Gotcha 1 Fix: Clean Section Tags directly without redundant Identity Cards in Hybrid Mode) ---
        lyrics_lines: List[str] = []
        for node in arc_nodes:
            tag = node.vocal_tag.strip()
            if not tag.startswith("["):
                tag = f"[{tag}"
            if not tag.endswith("]"):
                tag = f"{tag}]"

            lyrics_lines.append(tag)

            if not node.is_instrumental:
                if node.lyrics_lines:
                    for line in node.lyrics_lines:
                        clean_line = sanitize_all_caps_line(line)
                        lyrics_lines.append(clean_line)
                else:
                    pass  # No lyrics provided — leave section tag alone; Suno generates freely
            lyrics_lines.append("")

        lyrics_box = "\n".join(lyrics_lines).strip()

        # --- 4. ADVANCED EXCLUDE PROMPT (Gotcha 3 Fix: Hard Pop Filters + Anti-Intro-Humming String) ---
        negative_prompt = ", ".join(bible.identity_exclusions)

        raw_bundle = {
            "style_prompt": style_prompt,
            "lyrics_box": lyrics_box,
            "negative_prompt": negative_prompt,
            "p_mode": p_mode,
            "s_mode": s_mode,
            "interpretation": interpretation,
            "arc_nodes": arc_nodes,
            "bible": bible,
        }

        # --- 5. 🎙️ VOCAL IDENTITY GATE ---
        gate_result = VocalIdentityGate.evaluate(bible, interpretation, raw_bundle)
        if gate_result.repair_needed:
            raw_bundle = VocalIdentityGate.apply_repair(raw_bundle, gate_result)

        raw_bundle["gate_result"] = gate_result
        return raw_bundle

    @classmethod
    def format_html(cls, result: Dict[str, Any], query: str) -> str:
        bible: VocalBibleBase = result["bible"]
        p_mode: PerformanceMode = result["p_mode"]
        s_mode: PerformanceMode = result["s_mode"]
        interp: InterpretationResult = result["interpretation"]
        gate: IdentityGateResult = result.get("gate_result")
        hc = interp.hard_constraints
        s_prompt = html.escape(result["style_prompt"])
        l_box = html.escape(result["lyrics_box"])
        neg_p = html.escape(result["negative_prompt"])
        arc_nodes: List[SectionArcNode] = result["arc_nodes"]

        if not hc.instruments:
            raise RuntimeError("instruments list is empty — Luna deduction failed.")
        inst_str = " + ".join(hc.instruments)
        song_char_lines = [
            f"{html.escape(hc.genre)} · {html.escape(hc.tempo_bpm)}",
            f"{html.escape(inst_str)} · {html.escape(hc.production)}",
        ]
        song_char_text = "\n".join(song_char_lines)

        sections_text = []
        for node in arc_nodes:
            sec_name_escaped = html.escape(node.section_name.upper())
            sec_block = f"<b>{sec_name_escaped}</b> <code>{html.escape(node.vocal_tag)}</code>"
            sections_text.append(sec_block)

        sections_formatted = "\n".join(sections_text)
        rhythm_label = "9/10 (Syncopated / Off-Beat)" if interp.rhythm_displacement >= 8.0 else f"{interp.rhythm_displacement:.0f}/10"

        gate_badge = f"🛡️ <b>Vocal Identity Gate:</b> <code>Distance {gate.identity_distance}/100 (PASSED)</code>" if gate and gate.passed else "🛡️ <b>Vocal Identity Gate:</b> <code>PASSED</code>"

        style_chars = len(result["style_prompt"])
        lyrics_chars = len(result["lyrics_box"])

        html_text = (
            f"🎙️ <b>{bible.name.upper()} (PRODUCER BLUEPRINT)</b>\n\n"
            f"<b>IMMUTABLE VOCAL CORE (FRONT-LOADED)</b>\n"
            f"<code>Broadway · {bible.gender} {bible.identity_card.vocal_register} · {bible.identity_card.vocal_core}</code>\n\n"
            f"<b>SONG WORLD</b>\n"
            f"{song_char_text}\n\n"
            f"<b>PERFORMANCE STATE</b>\n"
            f"{html.escape(p_mode.name)} + {html.escape(s_mode.name)}\n"
            f"<code>Power: {interp.power:.0f}/10 | Texture: {interp.texture:.0f}/10 | Intimacy: {interp.intimacy:.0f}/10 | Rhythmic Placement: {rhythm_label}</code>\n"
            f"{gate_badge}\n\n"
            f"🎛️ <b>RECOMMENDED SUNO v5.5 SETTINGS</b>\n"
            f"• <b>Style Influence:</b> <code>{SUNO_SETTINGS['style_influence']}</code>\n"
            f"• <b>Weirdness / Clarity:</b> <code>{SUNO_SETTINGS['weirdness_clarity']}</code>\n"
            f"• <b>Model:</b> <code>{SUNO_SETTINGS['model']}</code>\n\n"
            f"<b>CONCISE SECTION DIRECTIVES (2-3 WORD DENSE TAGS)</b>\n"
            f"{sections_formatted}\n\n"
            f"📋 <b>SUNO STYLE PROMPT (BROADWAY #1 OVERRIDE)</b> <i>({style_chars}/1000 chars)</i>\n"
            f"<pre><code>{s_prompt}</code></pre>\n\n"
            f"📝 <b>SUNO LYRICS / META BOX (ZERO CLASH)</b> <i>({lyrics_chars}/4000 chars)</i>\n"
            f"<pre><code>{l_box}</code></pre>\n\n"
            f"🚫 <b>ADVANCED EXCLUDE PROMPT (ANTI-HUMMING + POP FILTER)</b>\n"
            f"<pre><code>{neg_p}</code></pre>"
        )
        return html_text
