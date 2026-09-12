"""Step 1: High-Fidelity Stem Isolation using Mel-Band RoFormer with Demucs Fallback."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict
from music_dna_agent.src.domain.models import Job, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort


class DemucsStemIsolator(PipelineStepPort):
    """Isolates audio into vocals, bass, drums, and other stems using Mel-Band RoFormer and Demucs."""

    def __init__(self, model_name: str = "melband_roformer", segment_length: int = 7):
        self._model_name = model_name
        self._segment_length = segment_length

    @property
    def name(self) -> StepName:
        return StepName.STEM_ISOLATION

    async def execute(self, job: Job, workspace_dir: str) -> str:
        stems_dir = os.path.join(workspace_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)

        input_audio = job.input_file_path
        if not os.path.exists(input_audio):
            raise FileNotFoundError(f"Input audio file not found: {input_audio}")

        duration_limit = 35 if job.quality.value == "crisp_30s" else 70
        trimmed_input = os.path.join(workspace_dir, "input_trimmed.wav")

        trim_cmd = [
            "ffmpeg", "-y", "-i", input_audio,
            "-t", str(duration_limit),
            "-ar", "44100", "-ac", "2",
            trimmed_input
        ]
        proc = await asyncio.create_subprocess_exec(
            *trim_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        target_audio = trimmed_input if os.path.exists(trimmed_input) else input_audio

        # 1. Primary High-Fidelity Vocal Separation using Mel-Band RoFormer
        roformer_success = False
        try:
            from audio_separator.separator import Separator
            model_dir = "/tmp/audio-separator-models"
            sep = Separator(output_dir=stems_dir, output_format="WAV", model_file_dir=model_dir)
            sep.load_model("melband_roformer_big_beta4.ckpt")
            outputs = sep.separate(target_audio)
            
            for out_file in outputs:
                full_path = os.path.join(stems_dir, out_file)
                lower_name = out_file.lower()
                if "vocal" in lower_name:
                    shutil.copy2(full_path, os.path.join(stems_dir, "vocals.wav"))
                elif "instrument" in lower_name or "other" in lower_name:
                    shutil.copy2(full_path, os.path.join(stems_dir, "instrumental.wav"))
                    shutil.copy2(full_path, os.path.join(stems_dir, "other.wav"))
            
            if os.path.exists(os.path.join(stems_dir, "vocals.wav")):
                roformer_success = True
        except Exception as e:
            # Graceful fallback to Demucs if RoFormer encounters runtime issue
            pass

        # 2. Demucs execution (either for 4-stem breakdown or if RoFormer was bypassed)
        if not roformer_success or not os.path.exists(os.path.join(stems_dir, "drums.wav")):
            cmd = [
                "demucs",
                "--shifts=1",
                "--segment", str(self._segment_length),
                "-n", "htdemucs",
                "-o", stems_dir,
                target_audio,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            track_basename = Path(target_audio).stem
            demucs_out_dir = os.path.join(stems_dir, "htdemucs", track_basename)
            if not os.path.isdir(demucs_out_dir):
                candidates = list(Path(stems_dir).rglob("vocals.wav"))
                if candidates:
                    demucs_out_dir = str(candidates[0].parent)

            if os.path.isdir(demucs_out_dir):
                for stem_name in ["vocals", "bass", "drums", "other"]:
                    src = os.path.join(demucs_out_dir, f"{stem_name}.wav")
                    dest = os.path.join(stems_dir, f"{stem_name}.wav")
                    # Only overwrite vocals if RoFormer didn't run
                    if os.path.exists(src) and (stem_name != "vocals" or not roformer_success):
                        shutil.copy2(src, dest)

        return stems_dir
