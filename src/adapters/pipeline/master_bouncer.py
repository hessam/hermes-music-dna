"""Phase 8: Master Bounce, Output Packaging & Generator Artifact Assembly."""
from __future__ import annotations

import asyncio
import json
import os
from music_dna_agent.src.domain.models import Job, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort
from music_dna_agent.src.adapters.generators.suno_adapter import SunoAdapter


class MasterBouncer(PipelineStepPort):
    """Mixes clean synthesis layers and packages full multi-asset generative bundle."""

    @property
    def name(self) -> StepName:
        return StepName.MASTER_BOUNCE

    async def execute(self, job: Job, workspace_dir: str) -> str:
        synth_dir = os.path.join(workspace_dir, "synth")
        guide_dir = os.path.join(workspace_dir, "guide_track")
        prompts_dir = os.path.join(workspace_dir, "prompts")
        analysis_dir = os.path.join(workspace_dir, "analysis")
        voice_dir = os.path.join(workspace_dir, "voice")

        os.makedirs(guide_dir, exist_ok=True)
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)

        guide_wav_path = os.path.join(guide_dir, "guide_track.wav")
        guide_mp3_path = os.path.join(guide_dir, "guide_track.mp3")

        synth_lead = os.path.join(synth_dir, "lead.wav")
        synth_chords = os.path.join(synth_dir, "chords.wav")
        synth_bass = os.path.join(synth_dir, "bass.wav")
        synth_drums = os.path.join(synth_dir, "drums.wav")

        inputs = []
        filter_parts = []
        mix_labels = []
        idx = 0

        # Layer 1: Lead
        if os.path.exists(synth_lead) and os.path.getsize(synth_lead) > 2000:
            inputs.extend(["-i", synth_lead])
            filter_parts.append(f"[{idx}:a]volume=1.4[lead];")
            mix_labels.append("[lead]")
            idx += 1

        # Layer 2: Chords
        if os.path.exists(synth_chords) and os.path.getsize(synth_chords) > 2000:
            inputs.extend(["-i", synth_chords])
            filter_parts.append(f"[{idx}:a]volume=1.0[chords];")
            mix_labels.append("[chords]")
            idx += 1

        # Layer 3: Bass
        if os.path.exists(synth_bass) and os.path.getsize(synth_bass) > 2000:
            inputs.extend(["-i", synth_bass])
            filter_parts.append(f"[{idx}:a]volume=1.2[bass];")
            mix_labels.append("[bass]")
            idx += 1

        # Layer 4: Drums
        if os.path.exists(synth_drums) and os.path.getsize(synth_drums) > 2000:
            inputs.extend(["-i", synth_drums])
            filter_parts.append(f"[{idx}:a]volume=0.85[drums];")
            mix_labels.append("[drums]")
            idx += 1

        if mix_labels:
            mix_inputs = "".join(mix_labels)
            filter_str = (
                "".join(filter_parts)
                + f"{mix_inputs}amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=2,"
                f"dynaudnorm=f=150:g=15:m=10:p=0.95,"
                f"alimiter=limit=0.95:level=true[out]"
            )

            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", filter_str,
                "-map", "[out]",
                "-ar", "44100",
                "-ac", "2",
                guide_wav_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

            # MP3 copy
            if os.path.exists(guide_wav_path):
                mp3_cmd = ["ffmpeg", "-y", "-i", guide_wav_path, "-b:a", "320k", guide_mp3_path]
                mp3_proc = await asyncio.create_subprocess_exec(*mp3_cmd)
                await mp3_proc.communicate()

        # Generate Prompts via SunoAdapter
        if job.dna:
            prompts = SunoAdapter.generate_prompts(job.dna)
            style_p_path = os.path.join(prompts_dir, "suno_style_prompt.txt")
            vocal_p_path = os.path.join(prompts_dir, "suno_vocal_prompt.txt")
            neg_p_path = os.path.join(prompts_dir, "negative_prompt.txt")

            with open(style_p_path, "w", encoding="utf-8") as f:
                f.write(prompts["style_prompt"])
            with open(vocal_p_path, "w", encoding="utf-8") as f:
                f.write(prompts["vocal_prompt"])
            with open(neg_p_path, "w", encoding="utf-8") as f:
                f.write(prompts["negative_prompt"])

        # Populate output manifest
        clean_ref_path = os.path.join(voice_dir, "clean_vocal_reference.wav")
        manifest = {}
        if os.path.exists(clean_ref_path):
            manifest["clean_vocal_reference"] = clean_ref_path
        if os.path.exists(guide_wav_path):
            manifest["guide_track_wav"] = guide_wav_path
            manifest["guide_track_mp3"] = guide_mp3_path
            job.guide_track_path = guide_wav_path

        manifest["vocal_dna_json"] = os.path.join(analysis_dir, "vocal_dna.json")
        manifest["musical_dna_json"] = os.path.join(analysis_dir, "musical_dna.json")
        manifest["sections_json"] = os.path.join(analysis_dir, "sections.json")
        manifest["suno_style_prompt"] = os.path.join(prompts_dir, "suno_style_prompt.txt")
        manifest["suno_vocal_prompt"] = os.path.join(prompts_dir, "suno_vocal_prompt.txt")
        manifest["negative_prompt"] = os.path.join(prompts_dir, "negative_prompt.txt")

        job.output_manifest = manifest
        return guide_wav_path if os.path.exists(guide_wav_path) else clean_ref_path
