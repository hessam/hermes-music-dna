"""Phase 5: Section-Level Structural Analysis (Intro/Verse/Chorus/Bridge/Outro Segmentation)."""
from __future__ import annotations

import json
import os
import librosa
import numpy as np
from music_dna_agent.src.domain.models import Job, MusicalSection, SectionVocalProfile, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort


class StructureSegmenter(PipelineStepPort):
    """Segments track into verse/chorus structure to preserve dynamic performance contrast."""

    @property
    def name(self) -> StepName:
        return StepName.STRUCTURE_SEGMENTATION

    async def execute(self, job: Job, workspace_dir: str) -> str:
        input_audio = job.input_file_path
        if not os.path.exists(input_audio):
            input_audio = os.path.join(workspace_dir, "input_trimmed.wav")

        y, sr = librosa.load(input_audio, sr=22050, duration=180.0)
        total_duration = len(y) / sr

        # 1. Structural Boundary Detection (Novelty / Recurrence Matrix)
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            bounds = librosa.segment.agglomerative(chroma, k=min(6, max(2, int(total_duration / 25))))
            bound_times = list(librosa.frames_to_time(bounds, sr=sr))
        except Exception:
            # Fallback 30s uniform intervals
            bound_times = list(np.arange(0, total_duration, 30.0))

        if 0.0 not in bound_times:
            bound_times.insert(0, 0.0)
        if total_duration not in bound_times:
            bound_times.append(total_duration)
        bound_times = sorted(list(set(bound_times)))

        section_labels = ["intro", "verse_1", "chorus_1", "verse_2", "chorus_2", "bridge", "outro"]

        sections = []
        vocal_sections = []

        vocals_path = os.path.join(workspace_dir, "stems", "vocals.wav")
        y_voc, sr_voc = librosa.load(vocals_path, sr=22050) if os.path.exists(vocals_path) else (None, 22050)

        for i in range(len(bound_times) - 1):
            start_t = float(bound_times[i])
            end_t = float(bound_times[i + 1])
            if (end_t - start_t) < 5.0:
                continue

            lbl = section_labels[i] if i < len(section_labels) else f"section_{i+1}"
            
            # Segment audio chunk
            start_samp = int(start_t * sr)
            end_samp = int(end_t * sr)
            chunk = y[start_samp:end_samp]
            rms = librosa.feature.rms(y=chunk)[0]
            energy = float(np.mean(rms)) if len(rms) > 0 else 0.1

            sections.append(MusicalSection(
                label=lbl,
                start_s=start_t,
                end_s=end_t,
                bpm=120.0,
                key="C",
                scale="major",
                energy_intensity=min(1.0, energy * 4.0),
            ))

            # Vocal section profile
            if y_voc is not None:
                voc_chunk = y_voc[int(start_t * sr_voc):int(end_t * sr_voc)]
                if len(voc_chunk) > 1000:
                    voc_rms = librosa.feature.rms(y=voc_chunk)[0]
                    intensity_lufs = float(-24.0 + 20 * np.log10(np.mean(voc_rms) + 1e-6))
                    vocal_sections.append(SectionVocalProfile(
                        label=lbl,
                        start_s=start_t,
                        end_s=end_t,
                        f0_median_hz=140.0,
                        intensity_mean_lufs=intensity_lufs,
                        dynamic_range_db=12.0,
                        dominant_register="chest" if "verse" in lbl else "mixed",
                    ))

        if not sections:
            sections.append(MusicalSection(
                label="main_body",
                start_s=0.0,
                end_s=total_duration,
                bpm=120.0,
                key="C",
                scale="major",
                energy_intensity=0.5,
            ))

        if job.dna:
            job.dna.sections = sections
            if job.dna.vocal_dna:
                job.dna.vocal_dna.section_profiles = vocal_sections

        analysis_dir = os.path.join(workspace_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        sections_path = os.path.join(analysis_dir, "sections.json")

        with open(sections_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sections], f, indent=2)

        return sections_path
