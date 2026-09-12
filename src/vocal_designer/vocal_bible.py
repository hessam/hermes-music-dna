"""The 4-Layer Vocal Bible: The Powerful Dreamer & The Velvet Blade (Zero-Clash Hybrid Architecture)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


@dataclass
class VocalIdentityCard:
    """Layer 1: Immutable Vocal Specification Card (The Singer's permanent DNA)."""
    identity_name: str
    gender: str
    voice_type: str
    vocal_register: str
    vocal_core: str
    vocal_dynamics: str


@dataclass
class PerformanceMode:
    """Layer 2: Performance State (Ways the same person behaves)."""
    mode_id: str
    name: str
    default_power: float
    default_texture: float
    default_intimacy: float
    default_rhythm_displacement: float
    vocal_token: str
    section_label: str


@dataclass
class SectionArcNode:
    """Layer 3: Section-level Vocal & Instrumental Behavior (2-3 words max)."""
    section_name: str
    is_instrumental: bool
    vocal_token: str
    arrangement_token: str
    vocal_tag: str
    lyrics_lines: List[str] = field(default_factory=list)


class VocalBibleBase(ABC):
    slug: str
    name: str
    gender: str
    identity_card: VocalIdentityCard
    modes: Dict[str, PerformanceMode]
    identity_exclusions: List[str]

    @abstractmethod
    def get_mode(self, mode_id: str) -> Optional[PerformanceMode]:
        pass

    @abstractmethod
    def generate_vocal_arc(
        self,
        primary_mode_id: str,
        secondary_mode_id: str,
        explicit_sections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SectionArcNode]:
        pass


class PowerfulDreamerBible(VocalBibleBase):
    """The Powerful Dreamer (Male): Power held under control."""
    slug = "powerful-dreamer"
    name = "The Powerful Dreamer"
    gender = "male"

    identity_card = VocalIdentityCard(
        identity_name="The Powerful Dreamer",
        gender="male",
        voice_type="male high-baritone",
        vocal_register="high-baritone",
        vocal_core="dense forward chest core",
        vocal_dynamics="restrained power -> controlled expansion",
    )

    modes: Dict[str, PerformanceMode] = {
        "beast": PerformanceMode(mode_id="beast", name="The Beast", default_power=10.0, default_texture=8.0, default_intimacy=2.0, default_rhythm_displacement=3.0, vocal_token="aggressive belt", section_label="power climax"),
        "commander": PerformanceMode(mode_id="commander", name="The Commander", default_power=9.0, default_texture=5.0, default_intimacy=4.0, default_rhythm_displacement=4.0, vocal_token="commanding mix", section_label="anthem chorus"),
        "showman": PerformanceMode(mode_id="showman", name="The Showman", default_power=7.0, default_texture=4.0, default_intimacy=5.0, default_rhythm_displacement=5.0, vocal_token="theatrical delivery", section_label="melodic lift"),
        "storyteller": PerformanceMode(mode_id="storyteller", name="The Storyteller", default_power=5.0, default_texture=3.0, default_intimacy=8.5, default_rhythm_displacement=9.0, vocal_token="syncopated talk-singing", section_label="rhythmic narrative"),
        "dreamer": PerformanceMode(mode_id="dreamer", name="The Dreamer", default_power=3.0, default_texture=2.0, default_intimacy=9.0, default_rhythm_displacement=8.0, vocal_token="ethereal head-mix", section_label="floating atmosphere"),
        "wound": PerformanceMode(mode_id="wound", name="The Wound", default_power=3.0, default_texture=2.5, default_intimacy=9.5, default_rhythm_displacement=7.0, vocal_token="low-chest intimate", section_label="intimate vulnerability"),
    }

    # Core Exclusions + Gotcha 3 Anti-Intro-Humming Exclusions
    identity_exclusions: List[str] = [
        "pop vocal",
        "breathy indie",
        "whisper",
        "operatic",
        "screamer",
        "autotune",
        "reverb wash",
        "electronic beat",
        "oohs",
        "aahs",
        "yeahs",
        "humming intro",
        "hum",
        "humming",
        "vocalise",
        "scatting",
        "spoken intro",
    ]

    def get_mode(self, mode_id: str) -> Optional[PerformanceMode]:
        return self.modes.get(mode_id.lower())

    def generate_vocal_arc(
        self,
        primary_mode_id: str,
        secondary_mode_id: str,
        explicit_sections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SectionArcNode]:
        p_mode = self.get_mode(primary_mode_id) or self.modes["storyteller"]
        s_mode = self.get_mode(secondary_mode_id) or self.modes["wound"]

        if explicit_sections:
            nodes: List[SectionArcNode] = []
            for sec in explicit_sections:
                s_name = sec.get("section_name", "Section 1")
                s_lower = s_name.lower()
                is_instr = sec.get("is_instrumental", False) or any(k in s_lower for k in ["intro", "solo", "instrumental", "end"]) and not sec.get("lyrics_lines")

                v_tok = sec.get("vocal_token")
                a_tok = sec.get("arrangement_token")

                if is_instr:
                    arr = a_tok or p_mode.section_label
                    tag = f"[{s_name} | {arr}]"
                    nodes.append(SectionArcNode(
                        section_name=s_name,
                        is_instrumental=True,
                        vocal_token="instrumental",
                        arrangement_token=arr,
                        vocal_tag=tag,
                        lyrics_lines=[],
                    ))
                    continue

                if not v_tok:
                    v_tok = "low-chest" if "verse" in s_lower or "bridge" in s_lower else p_mode.vocal_token
                if not a_tok:
                    a_tok = p_mode.section_label if "verse" not in s_lower else "fingerpicked sparse"

                tag = f"[{s_name} | {v_tok} | {a_tok}]"
                nodes.append(SectionArcNode(
                    section_name=s_name,
                    is_instrumental=False,
                    vocal_token=v_tok,
                    arrangement_token=a_tok,
                    vocal_tag=tag,
                    lyrics_lines=sec.get("lyrics_lines", []),
                ))
            return nodes

        # No explicit sections provided — return a minimal arc using the deduced mode tokens only
        verse_tag = f"[Verse | {p_mode.vocal_token}]"
        chorus_tag = f"[Chorus | {s_mode.vocal_token}]"
        return [
            SectionArcNode(section_name="Verse", is_instrumental=False, vocal_token=p_mode.vocal_token, arrangement_token=p_mode.section_label, vocal_tag=verse_tag),
            SectionArcNode(section_name="Chorus", is_instrumental=False, vocal_token=s_mode.vocal_token, arrangement_token=s_mode.section_label, vocal_tag=chorus_tag),
        ]


class VelvetBladeBible(VocalBibleBase):
    """The Velvet Blade (Female): Contralto Mirage architecture."""
    slug = "velvet-blade"
    name = "The Velvet Blade"
    gender = "female"

    identity_card = VocalIdentityCard(
        identity_name="The Velvet Blade",
        gender="female",
        voice_type="female contralto",
        vocal_register="contralto",
        vocal_core="dry dense low-chest",
        vocal_dynamics="dark low-chest restraint -> clean luminous upper release",
    )

    modes: Dict[str, PerformanceMode] = {
        "siren": PerformanceMode(mode_id="siren", name="The Siren", default_power=3.0, default_texture=2.0, default_intimacy=10.0, default_rhythm_displacement=6.0, vocal_token="intimate close-mic", section_label="magnetic presence"),
        "shadow": PerformanceMode(mode_id="shadow", name="The Shadow", default_power=4.0, default_texture=3.0, default_intimacy=8.0, default_rhythm_displacement=7.0, vocal_token="dry dark-chest", section_label="atmospheric mystery"),
        "oracle": PerformanceMode(mode_id="oracle", name="The Oracle", default_power=4.0, default_texture=1.0, default_intimacy=6.0, default_rhythm_displacement=8.0, vocal_token="clean prophetic mix", section_label="clean head-voice"),
        "blade": PerformanceMode(mode_id="blade", name="The Blade", default_power=8.0, default_texture=6.0, default_intimacy=4.0, default_rhythm_displacement=7.0, vocal_token="knife-consonant attack", section_label="sharp rock attack"),
        "flame": PerformanceMode(mode_id="flame", name="The Flame", default_power=9.0, default_texture=4.0, default_intimacy=5.0, default_rhythm_displacement=4.0, vocal_token="luminous ringing release", section_label="luminous climax"),
        "confession": PerformanceMode(mode_id="confession", name="The Confession", default_power=2.0, default_texture=2.0, default_intimacy=10.0, default_rhythm_displacement=5.0, vocal_token="dry low-chest spoken", section_label="exposed vulnerability"),
    }

    # Core Exclusions + Gotcha 3 Anti-Intro-Humming Exclusions
    identity_exclusions: List[str] = [
        "pop soprano",
        "breathy indie",
        "whisper",
        "vocal fry",
        "Janis rasp",
        "autotune",
        "reverb wash",
        "electronic beat",
        "operatic",
        "diva belting",
        "oohs",
        "aahs",
        "yeahs",
        "humming intro",
        "hum",
        "humming",
        "vocalise",
        "scatting",
        "spoken intro",
    ]

    def get_mode(self, mode_id: str) -> Optional[PerformanceMode]:
        return self.modes.get(mode_id.lower())

    def generate_vocal_arc(
        self,
        primary_mode_id: str,
        secondary_mode_id: str,
        explicit_sections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SectionArcNode]:
        p_mode = self.get_mode(primary_mode_id) or self.modes["siren"]
        s_mode = self.get_mode(secondary_mode_id) or self.modes["shadow"]

        if explicit_sections:
            nodes: List[SectionArcNode] = []
            for sec in explicit_sections:
                s_name = sec.get("section_name", "Section 1")
                s_lower = s_name.lower()
                is_instr = sec.get("is_instrumental", False) or any(k in s_lower for k in ["intro", "solo", "instrumental", "end"]) and not sec.get("lyrics_lines")

                v_tok = sec.get("vocal_token")
                a_tok = sec.get("arrangement_token")

                if is_instr:
                    arr = a_tok or "atmospheric lead"
                    tag = f"[{s_name} | {arr}]"
                    nodes.append(SectionArcNode(
                        section_name=s_name,
                        is_instrumental=True,
                        vocal_token="instrumental",
                        arrangement_token=arr,
                        vocal_tag=tag,
                        lyrics_lines=[],
                    ))
                    continue

                if not v_tok:
                    v_tok = p_mode.vocal_token if "chorus" not in s_lower else s_mode.vocal_token
                if not a_tok:
                    a_tok = p_mode.section_label if "chorus" not in s_lower else s_mode.section_label

                tag = f"[{s_name} | {v_tok} | {a_tok}]"
                nodes.append(SectionArcNode(
                    section_name=s_name,
                    is_instrumental=False,
                    vocal_token=v_tok,
                    arrangement_token=a_tok,
                    vocal_tag=tag,
                    lyrics_lines=sec.get("lyrics_lines", []),
                ))
            return nodes

        # No explicit sections provided — return a minimal arc using the deduced mode tokens only
        verse_tag = f"[Verse | {p_mode.vocal_token}]"
        chorus_tag = f"[Chorus | {s_mode.vocal_token}]"
        return [
            SectionArcNode(section_name="Verse", is_instrumental=False, vocal_token=p_mode.vocal_token, arrangement_token=p_mode.section_label, vocal_tag=verse_tag),
            SectionArcNode(section_name="Chorus", is_instrumental=False, vocal_token=s_mode.vocal_token, arrangement_token=s_mode.section_label, vocal_tag=chorus_tag),
        ]


class BibleRegistry:
    _registry: Dict[str, VocalBibleBase] = {
        "powerful-dreamer": PowerfulDreamerBible(),
        "velvet-blade": VelvetBladeBible(),
    }

    @classmethod
    def get(cls, slug: str = "powerful-dreamer") -> VocalBibleBase:
        s = slug.lower().strip()
        if "velvet" in s or "female" in s:
            return cls._registry["velvet-blade"]
        return cls._registry.get(s, cls._registry["powerful-dreamer"])
