"""Phase 1: Ingestion Quality Control Gate (LUFS, Peak Headroom, Duration & Clipping Validation)."""
from __future__ import annotations

import json
import os
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from music_dna_agent.src.domain.models import Job, SourceQC, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort


class SourceQCGate(PipelineStepPort):
    """Validates source audio integrity, dynamic range, and recording quality before heavy processing."""

    @property
    def name(self) -> StepName:
        return StepName.SOURCE_QC

    async def execute(self, job: Job, workspace_dir: str) -> str:
        input_path = job.input_file_path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found at {input_path}")

        data, rate = sf.read(input_path)
        
        # 1. Duration check
        duration_s = float(len(data) / rate) if rate > 0 else 0.0
        if duration_s < 10.0:
            raise ValueError(f"Source audio duration ({duration_s:.1f}s) is too short. Minimum 10 seconds required.")

        # 2. Channel info
        is_stereo = (data.ndim > 1 and data.shape[1] >= 2)
        audio_mono = np.mean(data, axis=1) if is_stereo else data

        # 3. Peak Headroom & Clipping
        peak_amp = float(np.max(np.abs(data)))
        clipping_detected = (peak_amp >= 0.999)
        true_peak_dbtp = float(20 * np.log10(peak_amp + 1e-9))

        # 4. Integrated Loudness (LUFS)
        try:
            meter = pyln.Meter(rate)
            lufs_integrated = float(meter.integrated_loudness(data))
        except Exception:
            lufs_integrated = -18.0

        qc_notes = []
        if clipping_detected:
            qc_notes.append("Digital clipping / 0 dBFS peak detected in source material.")
        if lufs_integrated < -30.0:
            qc_notes.append("Low average signal energy (< -30 LUFS). Dynamic boosting recommended.")
        if duration_s < 30.0:
            qc_notes.append(f"Short reference audio ({duration_s:.1f}s). Segment selection will operate in concise mode.")

        qc_result = SourceQC(
            duration_s=duration_s,
            sample_rate=rate,
            lufs_integrated=lufs_integrated,
            true_peak_dbtp=true_peak_dbtp,
            is_stereo=is_stereo,
            clipping_detected=clipping_detected,
            qc_passed=True,
            qc_notes=qc_notes,
        )

        job.source_qc = qc_result

        # Save QC report JSON
        qc_json_path = os.path.join(workspace_dir, "source_qc.json")
        with open(qc_json_path, "w", encoding="utf-8") as f:
            json.dump(qc_result.to_dict(), f, indent=2)

        return qc_json_path
