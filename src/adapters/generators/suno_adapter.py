"""Translation of physical acoustics to high-impact Suno AI prompt architecture.

Design Laws:
  1. No numeric tokens (Hz, ms, st) — Suno tokenizer cannot parse them
  2. Max 7 comma-separated clauses in the Style box (positional priority decay)
  3. Phonation vocabulary must match Suno's training distribution (evocative, not engineering)
  4. Negative prompt is conditional — never exclude a feature that IS the signature
  5. Section tags: max 3 words, no backslashes, pipe (|) optional
"""
from __future__ import annotations

from typing import Dict, Any, List
from music_dna_agent.src.domain.models import MusicalDNA, VocalMeasurements


# ---------------------------------------------------------------------------
# Acoustic Interpreter
# ---------------------------------------------------------------------------

class AcousticInterpreter:
    """Translates physical acoustic measurements into Suno-compatible vocabulary.

    All output tokens must be in Suno's training corpus — no engineering notation.
    """

    # -----------------------------------------------------------------------
    # Register & Voice Type
    # -----------------------------------------------------------------------
    @classmethod
    def get_register_and_type(cls, f0_median: float) -> Dict[str, str]:
        if f0_median < 110:
            return {
                "voice_type": "Deep Bass",
                "pitch_note": "A2",
                "vocal_range": "Low-chest register (A2)",
                "character": "commanding, resonant chest-dominant low end with profound subharmonic body"
            }
        elif f0_median < 145:
            return {
                "voice_type": "Low Baritone",
                "pitch_note": "C3",
                "vocal_range": "Chest register (C3)",
                "character": "warm, rich baritone timbre with centered chest resonance and grounded fundamental authority"
            }
        elif f0_median < 185:
            return {
                "voice_type": "High Baritone",
                "pitch_note": "F3",
                "vocal_range": "Mid register (F3)",
                "character": "flexible mid-register tone with fluid chest-to-mix transition"
            }
        elif f0_median < 250:
            return {
                "voice_type": "High Tenor",
                "pitch_note": "A3",
                "vocal_range": "Upper-chest to head mix (A3)",
                "character": "bright, cutting upper-mid voice with effortless forward projection"
            }
        elif f0_median < 350:
            return {
                "voice_type": "Alto",
                "pitch_note": "D4",
                "vocal_range": "Belt / head-mix register (D4)",
                "character": "soaring, clear vocal presence with focused upper-harmonic shine"
            }
        else:
            return {
                "voice_type": "Soprano",
                "pitch_note": "G4+",
                "vocal_range": "Head register (G4+)",
                "character": "ethereal, crystalline top register with delicate harmonic extension"
            }

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Praat Formants & Harmonicity Interpretation (1:1 Text-Only Voice Match)
    # -----------------------------------------------------------------------
    @classmethod
    def _classify_formants(cls, formants: Optional[Dict[str, float]], f0_median: float) -> Dict[str, str]:
        """Map Praat Burg Formants (F1-F4) into Suno vocal resonance tokens."""
        if not formants:
            return {
                "ring_token": "natural vocal resonance",
                "tract_desc": "balanced natural vocal tract resonance",
            }

        f1 = formants.get("f1", 600.0)
        f2 = formants.get("f2", 1500.0)
        f3 = formants.get("f3", 2800.0)

        # Singer's Formant / Ring (F3 cluster 2.6 - 3.4 kHz)
        is_female = f0_median > 175.0
        singers_formant_range = (2800.0, 3600.0) if is_female else (2500.0, 3300.0)
        has_singers_formant = singers_formant_range[0] <= f3 <= singers_formant_range[1]

        # Pharyngeal Twang vs Cavernous Throat (F2)
        twang_threshold = 2100.0 if is_female else 1750.0
        dark_threshold = 1400.0 if is_female else 1250.0

        if f2 > twang_threshold:
            tract_shape = "forward pharyngeal twang"
        elif f2 < dark_threshold:
            tract_shape = "dark cavernous chest resonance"
        else:
            tract_shape = "centered vocal tract resonance"

        # Jaw Drop / Belting (F1)
        if f1 > 700.0:
            jaw_desc = "open-throat belting acoustics"
        elif f1 < 450.0:
            jaw_desc = "covered intimate acoustics"
        else:
            jaw_desc = "balanced oral resonance"

        if has_singers_formant:
            ring_token = "singer's formant ring"
            tract_desc = f"{tract_shape} with penetrating singer's formant cut and {jaw_desc}"
        else:
            ring_token = tract_shape
            tract_desc = f"{tract_shape} with {jaw_desc}"

        return {
            "ring_token": ring_token,
            "tract_desc": tract_desc,
        }

    @classmethod
    def _classify_hnr(cls, hnr_db: Optional[float], is_pressed: bool) -> Dict[str, str]:
        """Map Praat Harmonics-to-Noise Ratio (dB) into Suno phonation tokens."""
        if hnr_db is None:
            return {"hnr_token": "", "hnr_desc": "natural harmonic balance"}

        if hnr_db < 6.0:
            if is_pressed:
                return {
                    "hnr_token": "raspy gravel distortion",
                    "hnr_desc": f"heavy subharmonic breakup and raspy overdrive ({hnr_db:.1f} dB HNR)",
                }
            return {
                "hnr_token": "husky breathy texture",
                "hnr_desc": f"pronounced airy breath turbulence and smoky edge ({hnr_db:.1f} dB HNR)",
            }
        elif hnr_db < 12.0:
            return {
                "hnr_token": "textured breath air",
                "hnr_desc": f"organic breath air with textured harmonic presence ({hnr_db:.1f} dB HNR)",
            }
        elif hnr_db > 18.0:
            return {
                "hnr_token": "crystal harmonic clarity",
                "hnr_desc": f"pure bell-like harmonic clarity with zero breath noise ({hnr_db:.1f} dB HNR)",
            }
        return {
            "hnr_token": "clean harmonic balance",
            "hnr_desc": f"even modal harmonic resonance ({hnr_db:.1f} dB HNR)",
        }

    # Phonation & Timbre (the critical path for accuracy)
    # -----------------------------------------------------------------------
    @classmethod
    def _classify_phonation(cls, m: VocalMeasurements) -> Dict[str, str]:
        """Classify vocal phonation from glottal perturbation metrics.

        Decision matrix (ordered by specificity):
          H1-H2 < 0        → pressed / over-driven (distorted, raspy belter)
          Shimmer > 18%     → heavy breathiness / falsetto-adjacent
          Shimmer > 14%     → breathy with texture
          Jitter > 4.5%     → unstable / character grit
          Jitter > 2.5%     → moderate organic roughness
          Subharmonic > 0.08→ vocal fry signature present
          H1-H2 > 8        → clean, airy, soft phonation
          else             → balanced, modal

        Returns evocative Suno-parseable tokens only (no Hz, no %).
        """
        h = m.h1_h2_db
        j = m.jitter_local_pct
        sh = m.shimmer_local_pct
        sub = m.subharmonic_energy_ratio

        # Pressed / over-driven belter (e.g. Axl Rose, Chris Cornell)
        if h < 0:
            base_phon = "raw aggressive belting"
            base_timbre = "raspy, gritty, aggressive"
            base_texture = "over-driven glottal attack with biting harmonic edge"
            grit_token = "raspy aggressive"
        # Airy / soft (e.g. Norah Jones, Nick Drake)
        elif h > 8.0 or sh > 18.0:
            base_phon = "soft breathy phonation"
            base_timbre = "warm, airy, intimate"
            base_texture = "gentle acoustic airflow with natural warmth"
            grit_token = ""
        # Balanced modal (middle ground)
        else:
            base_phon = "clear resonant phonation"
            base_timbre = "balanced, resonant, forward"
            base_texture = "focused acoustic resonance with full presence"
            grit_token = ""

        # Overlay grit / fry texture
        has_fry = sub > 0.07
        has_grit = j > 2.5 or sh > 14.0 or sub > 0.05

        if has_fry and h >= 0:  # fry without full press
            base_phon += " with natural vocal fry texture"
            base_timbre += ", gravelly fry undertone"
            base_texture += ", with raw vocal fry character"
            grit_token = "gravelly fry"
        elif has_grit and h >= 0:
            base_phon += " with organic vocal grit"
            base_timbre += ", textured character"
            base_texture += ", textured with natural breakup"
            grit_token = grit_token or "textured gritty"

        return {
            "phonation": base_phon,
            "timbre_tags": base_timbre,
            "texture": base_texture,
            "grit_token": grit_token,
            "has_fry": has_fry,
            "has_grit": has_grit,
            "is_pressed": h < 0,
        }

    @classmethod
    def _classify_space(cls, m: VocalMeasurements) -> str:
        """Map CPP prominence to Suno mic placement vocabulary."""
        if m.cpp_db > 14.0:
            return "close-mic presence"
        elif m.cpp_db < 9.5:
            return "ambient room mic"
        else:
            return "studio direct mic"

    @classmethod
    def _classify_vibrato(cls, m: VocalMeasurements) -> Dict[str, str]:
        """Map vibrato rate/depth to Suno vocabulary. No numeric Hz output."""
        if not m.vibrato_rate_hz or not m.vibrato_depth_semitones or m.vibrato_depth_semitones < 0.35:
            return {
                "tag": "straight-tone delivery",
                "desc": "straight-tone phrasing with subtle expressive tail",
            }
        r = m.vibrato_rate_hz
        d = m.vibrato_depth_semitones
        if r < 4.2:
            return {
                "tag": "slow expressive vibrato",
                "desc": f"slow blossoming vibrato on sustained phrases",
            }
        elif r > 6.5:
            return {
                "tag": "rapid tight vibrato",
                "desc": "rapid tight vibrato pulse on held notes",
            }
        else:
            if d > 1.2:
                return {
                    "tag": "wide natural vibrato",
                    "desc": "wide natural vibrato with expressive depth",
                }
            return {
                "tag": "natural vibrato",
                "desc": "controlled natural vibrato on sustained notes",
            }

    @classmethod
    def _classify_register_delivery(cls, registers: List[str], is_pressed: bool, has_grit: bool) -> str:
        """Map register list to a Suno-friendly delivery style tag."""
        reg_set = set(r.lower() for r in (registers or []))
        if is_pressed:
            if "head" in reg_set or "falsetto" in reg_set:
                return "screaming head register belting"
            return "aggressive chest belting"
        if "falsetto" in reg_set:
            return "soaring falsetto"
        if "head" in reg_set and "chest" in reg_set:
            return "dynamic chest-to-head mix"
        if "head" in reg_set:
            return "head voice projection"
        return "chest voice delivery"

    # -----------------------------------------------------------------------
    # Master interpreter
    # -----------------------------------------------------------------------
    @classmethod
    def interpret_timbre(cls, m: VocalMeasurements, f0_median: float) -> Dict[str, Any]:
        reg_info = cls.get_register_and_type(f0_median)
        phon = cls._classify_phonation(m)
        space = cls._classify_space(m)
        vib = cls._classify_vibrato(m)
        delivery = cls._classify_register_delivery(
            m.detected_registers,
            phon["is_pressed"],
            phon["has_grit"],
        )
        formant_info = cls._classify_formants(m.formants_hz, f0_median)
        hnr_info = cls._classify_hnr(m.hnr_db, phon["is_pressed"])

        # Fuse Praat HNR clarity/rasp into timbre tags if relevant
        timbre_tags = phon["timbre_tags"]
        if hnr_info["hnr_token"] and hnr_info["hnr_token"] not in timbre_tags:
            timbre_tags = f"{timbre_tags}, {hnr_info['hnr_token']}"

        return {
            # Register
            "voice_type": reg_info["voice_type"],
            "vocal_range": reg_info["vocal_range"],
            "character": reg_info["character"],
            "pitch_note": reg_info["pitch_note"],
            # Phonation & HNR
            "phonation": phon["phonation"],
            "timbre_tags": timbre_tags,
            "texture": phon["texture"],
            "grit_token": phon["grit_token"],
            "has_fry": phon["has_fry"],
            "has_grit": phon["has_grit"],
            "is_pressed": phon["is_pressed"],
            "hnr_token": hnr_info["hnr_token"],
            "hnr_desc": hnr_info["hnr_desc"],
            # Formants & Resonance
            "formant_ring": formant_info["ring_token"],
            "vocal_tract_desc": formant_info["tract_desc"],
            # Space
            "space_tag": space,
            # Vibrato
            "vibrato_tag": vib["tag"],
            "vibrato": vib["desc"],
            # Delivery
            "delivery": delivery,
        }


# ---------------------------------------------------------------------------
# Suno Adapter
# ---------------------------------------------------------------------------

class SunoAdapter:
    """Translates physical acoustics into Suno AI prompt architecture.

    Enforces:
      - Max 7 style tokens (positional priority law)
      - No numeric tokens (Hz, ms, %) in output prompts
      - Conditional negative prompt (never exclude the signature)
      - Short section tags (max 3 words, pipe separator)
    """

    MAX_STYLE_TOKENS = 7

    # -----------------------------------------------------------------------
    # Prompt generation
    # -----------------------------------------------------------------------
    @classmethod
    def generate_prompts(cls, dna: MusicalDNA) -> Dict[str, str]:
        bpm = round(dna.bpm)
        key_str = f"{dna.key} {dna.scale.capitalize()}"
        chords_str = " - ".join(dna.chord_progression[:4]) if dna.chord_progression else ""

        # Tempo and loudness do not identify genre, instrumentation or recording gear.
        energy = "follow the reference arrangement"
        tempo_key = f"{key_str}, {bpm} BPM"
        chord_token = chords_str
        production_tokens = ["preserve reference dynamics and ambience"]

        style_elements = [energy, tempo_key]
        if chord_token:
            style_elements.append(chord_token)
        style_elements.extend(production_tokens)

        # Slots 5-7 reserved for the triple-stack vocal identity
        vocal_compact_tags = []
        detailed_persona_text = ""
        vocal_chain_header = ""
        lyrics_box_template = ""
        negative_tokens: List[str] = []
        has_fry = False
        is_pressed = False

        if dna.vocal_dna and dna.vocal_dna.measurements:
            m = dna.vocal_dna.measurements
            f0_med = m.f0_distribution.get("median", 150.0)
            interp = AcousticInterpreter.interpret_timbre(m, f0_med)
            has_fry = interp["has_fry"]
            is_pressed = interp["is_pressed"]

            # Triple-Stack: Character | Delivery + Resonance | Texture + Harmonic Purity
            # (NO raw Hz notation in Style box — highest-fidelity Suno prompt tokens)
            vocal_compact_tags.append(f"lead {interp['voice_type'].lower()}")  # Slot 5
            delivery_slot = f"{interp['delivery']}, {interp['formant_ring']}" if interp["formant_ring"] else interp["delivery"]
            vocal_compact_tags.append(delivery_slot)                           # Slot 6
            vocal_compact_tags.append(interp["timbre_tags"])                   # Slot 7

            # Short Vocal Chain header (pipe syntax, ≤3 words per segment)
            chain_parts = [
                f"lead {interp['voice_type'].lower()}",
                interp["delivery"],
                interp["formant_ring"],
                interp["vibrato_tag"],
                interp["space_tag"],
            ]
            # Deduplicate and keep concise
            clean_chain = [p for p in chain_parts if p]
            vocal_chain_header = "[" + " | ".join(clean_chain) + "]"

            # Detailed Persona Description (human-readable paragraph, not in Style box)
            detailed_persona_text = (
                f"{vocal_chain_header}\n\n"
                f"A lead {interp['voice_type'].lower()} singer. "
                f"{interp['character'].capitalize()}.\n"
                f"- Phonation & Harmonicity: {interp['phonation']}, featuring {interp['texture']}. {interp['hnr_desc'][0].upper() + interp['hnr_desc'][1:] if interp['hnr_desc'] else ''}.\n"
                f"- Resonance & Vocal Tract: {interp['vocal_tract_desc'].capitalize()}.\n"
                f"- Mic & Spatial Placement: {interp['space_tag'].capitalize()}.\n"
                f"- Phrasing & Delivery: {interp['vibrato'].capitalize()}, with {interp['delivery']}."
            )

            # Section tags: ≤3 words, pipe separator, no backslash
            intro_tag = "[Intro]"
            verse_tag = "[Verse]"
            chorus_energy = "power chorus" if is_pressed else "soaring chorus"
            chorus_tag = f"[Chorus | {chorus_energy}]"
            bridge_tag = "[Bridge]"
            outro_tag = "[Outro]"

            lyrics_sections = [
                vocal_chain_header,
                f"\n{intro_tag}\n(instrumental opening...)",
                f"\n{verse_tag}\n(write your verse lyrics here...)",
                f"\n[Pre-Chorus]\n(write your pre-chorus here...)",
                f"\n{chorus_tag}\n(write your chorus hook here...)",
                f"\n{verse_tag}\n(write your second verse here...)",
                f"\n{chorus_tag}\n(repeat chorus with more energy...)",
                f"\n{bridge_tag}\n(write your bridge here...)",
                f"\n{chorus_tag}\n(final chorus climax...)",
                f"\n{outro_tag}\n(fade out...)\n[End]",
            ]
            lyrics_box_template = "\n".join(lyrics_sections)

        else:
            vocal_compact_tags.extend(["expressive lead vocal", "dynamic delivery", "clear resonant tone"])
            vocal_chain_header = "[lead vocal | dynamic delivery | studio mic]"
            detailed_persona_text = (
                f"{vocal_chain_header}\n\n"
                "A dynamic, expressive lead vocal performance with natural tonal balance, "
                "clear acoustic definition, musical phrasing, and authentic room space."
            )
            lyrics_box_template = (
                f"{vocal_chain_header}\n\n[Intro]\n\n[Verse]\n(your verse here...)\n\n"
                "[Chorus]\n(your chorus here...)\n\n[Outro]\n[End]"
            )

        # Trim style to MAX_STYLE_TOKENS (vocal tags fill last 3 slots)
        style_base = style_elements[:cls.MAX_STYLE_TOKENS - 3]
        style_prompt = ", ".join(style_base + vocal_compact_tags[:3])

        # --- Conditional Negative Prompt ---
        # Core universal exclusions (always safe)
        negative_tokens = [
            "digital pitch correction",
            "metallic reverb wash",
            "chipmunk harmonics",
            "muddy low-end",
        ]
        # Conditional: only exclude fry/grit if NOT present in the measured signal
        if not has_fry and not is_pressed:
            negative_tokens.append("vocal fry")
            negative_tokens.append("raspy distortion")
        # Conditional: only exclude autotune sheen if not a clean-voice singer
        if not is_pressed:
            negative_tokens.append("synthetic autotune sheen")

        negative_prompt = ", ".join(negative_tokens)

        return {
            "style_prompt": style_prompt,
            "vocal_prompt": ", ".join(vocal_compact_tags),
            "vocal_chain_header": vocal_chain_header,
            "detailed_persona_text": detailed_persona_text,
            "lyrics_box_template": lyrics_box_template,
            "negative_prompt": negative_prompt,
        }

    # -----------------------------------------------------------------------
    # Markdown report formatter
    # -----------------------------------------------------------------------
    @classmethod
    def format_summary_markdown(cls, dna: MusicalDNA) -> str:
        prompts = cls.generate_prompts(dna)
        chords_str = " -> ".join(dna.chord_progression[:8]) if dna.chord_progression else "N/A"
        sections_str = ", ".join([s.label for s in dna.sections]) if dna.sections else "Standard"

        md = (
            f"🎵 **Music DNA Extraction & Generation Blueprint**\n\n"
            f"• **Key & Scale:** `{dna.key} {dna.scale.capitalize()}`\n"
            f"• **Tempo & Groove:** `{round(dna.bpm, 1)} BPM` (Offset: `{round(dna.groove_offset_ms, 1)} ms`)\n"
            f"• **Song Structure:** `{sections_str}`\n"
            f"• **Harmonic Chords:** `{chords_str}`\n"
            f"• **Mix Headroom:** `{round(dna.lufs_integrated, 1)} LUFS` (Range: `{round(dna.lufs_range, 1)} dB`)\n\n"
        )

        if dna.vocal_dna and dna.vocal_dna.measurements:
            m = dna.vocal_dna.measurements
            f0_med = m.f0_distribution.get("median", 150.0)
            interp = AcousticInterpreter.interpret_timbre(m, f0_med)
            vib_str = f"{m.vibrato_rate_hz:.1f} Hz ({m.vibrato_depth_semitones:.1f} st)" if m.vibrato_rate_hz else "Subtle / Straight-tone"
            med_val = round(f0_med)

            md += (
                f"🎙️ **Vocal Persona Profile (Physical Measurements & Grounded Interpretation)**\n"
                f"• **Classified Voice:** `{interp['voice_type']}` (`{interp['vocal_range']}`)\n"
                f"• **Pitch Center (F0):** `{med_val} Hz ({interp['pitch_note']})` (Span: `{round(m.f0_range_semitones, 1)} st`)\n"
                f"• **Vibrato Dynamics:** `{vib_str}`\n"
                f"• **Glottal Perturbation:** `Jitter {m.jitter_local_pct:.2f}% | Shimmer {m.shimmer_local_pct:.2f}%`\n"
                f"• **Prominence (CPP):** `{m.cpp_db:.1f} dB`\n"
                f"• **Harmonic Tilt (H1-H2):** `{m.h1_h2_db:.1f} dB`\n"
                f"• **Subharmonic Ratio:** `{m.subharmonic_energy_ratio:.3f}`\n\n"
            )

        style_p = prompts["style_prompt"]
        persona_desc = prompts["detailed_persona_text"]
        lyrics_template = prompts["lyrics_box_template"]
        neg_p = prompts["negative_prompt"]

        md += (
            f"📋 **Suno Style Prompt (Copy-Paste into Suno 'Style of Music' box):**\n"
            f"```\n{style_p}\n```\n\n"
            f"🗣️ **Detailed Vocal Persona Description (Custom Model Card):**\n"
            f"```\n{persona_desc}\n```\n\n"
            f"📝 **Complete Song & Lyrics Template (Copy-Paste into Suno 'Lyrics' box):**\n"
            f"```\n{lyrics_template}\n```\n\n"
            f"🚫 **Negative Prompt (Paste into Exclude / Negative Box):**\n"
            f"```\n{neg_p}\n```"
        )

        return md
