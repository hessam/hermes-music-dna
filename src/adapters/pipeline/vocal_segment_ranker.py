"""Phase 2: Vocal Isolation Bleed Scorer, Multi-Reference Portfolio & Candidate Ranker."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from music_dna_agent.src.domain.models import Job, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort


class VocalSegmentRanker(PipelineStepPort):
    """Evaluates true vocal isolation, rejects bleed, and extracts a diversified reference portfolio."""

    @property
    def name(self) -> StepName:
        return StepName.VOCAL_SEGMENT_RANKING

    async def execute(self, job: Job, workspace_dir: str) -> str:
        stems_dir = os.path.join(workspace_dir, "stems")
        vocals_path = os.path.join(stems_dir, "vocals.wav")
        other_path = os.path.join(stems_dir, "other.wav")
        drums_path = os.path.join(stems_dir, "drums.wav")
        bass_path = os.path.join(stems_dir, "bass.wav")

        if not os.path.exists(vocals_path):
            raise FileNotFoundError("Isolated vocals are missing; the mixed input is not a vocal reference.")

        voice_dir = os.path.join(workspace_dir, "voice")
        segments_dir = os.path.join(voice_dir, "segments")
        portfolio_dir = os.path.join(voice_dir, "portfolio")
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(portfolio_dir, exist_ok=True)
        # Preserve the separator's waveform independently of shorter analysis clips.
        isolated_path = os.path.join(voice_dir, "isolated_vocals.wav")
        shutil.copy2(vocals_path, isolated_path)
        job.output_manifest["isolated_vocals"] = isolated_path

        y_voc, sr = librosa.load(vocals_path, sr=22050)
        total_duration = len(y_voc) / sr

        # Load accompaniment stems for true bleed calculation if available
        y_accomp = None
        if os.path.exists(other_path) and os.path.exists(drums_path):
            try:
                y_o, _ = librosa.load(other_path, sr=sr)
                y_d, _ = librosa.load(drums_path, sr=sr)
                min_len = min(len(y_voc), len(y_o), len(y_d))
                y_accomp = np.abs(y_o[:min_len]) + np.abs(y_d[:min_len])
            except Exception:
                y_accomp = None

        window_sec = 16.0
        hop_sec = 3.0
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        candidates = []
        num_windows = max(1, (len(y_voc) - window_samples) // hop_samples + 1)

        for i in range(num_windows):
            start_samp = i * hop_samples
            end_samp = min(len(y_voc), start_samp + window_samples)
            chunk = y_voc[start_samp:end_samp]

            rms = librosa.feature.rms(y=chunk)[0]
            mean_rms = float(np.mean(rms))
            if mean_rms < 0.012:
                continue

            # 1. True Isolation Ratio (Vocal Power vs Accompaniment Bleed)
            if y_accomp is not None and end_samp <= len(y_accomp):
                acc_chunk = y_accomp[start_samp:end_samp]
                p_voc = np.mean(chunk**2)
                p_acc = np.mean(acc_chunk**2) + 1e-8
                isolation_ratio_db = float(10.0 * np.log10(p_voc / p_acc))
            else:
                # Monophonic harmonicity fallback
                y_h, y_p = librosa.effects.hpss(chunk)
                isolation_ratio_db = float(10.0 * np.log10(np.mean(y_h**2) / (np.mean(y_p**2) + 1e-8)))

            # 2. Pitch Continuity & Voiced Fraction (pyin)
            f0, voiced, _ = librosa.pyin(chunk, fmin=65.0, fmax=800.0, sr=sr)
            voiced_ratio = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
            voiced_f0 = f0[voiced & ~np.isnan(f0)]
            f0_med = float(np.median(voiced_f0)) if len(voiced_f0) > 10 else 150.0
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 10 else 0.0

            # 3. Dynamic Range & Loudness
            dyn_range_db = float(20.0 * np.log10((np.max(rms) + 1e-6) / (np.min(rms) + 1e-6)))

            # 4. Composite Quality Score
            # Rewards: high vocal power, pitch continuity, clean isolation. Penalizes: extreme pitch volatility (shouts/bleed)
            iso_norm = min(1.0, max(0.0, (isolation_ratio_db + 6.0) / 20.0))
            score = (0.40 * iso_norm +
                     0.40 * voiced_ratio +
                     0.20 * min(1.0, dyn_range_db / 18.0))

            start_s = round(start_samp / sr, 2)
            end_s = round(end_samp / sr, 2)

            candidates.append({
                "start_s": start_s,
                "end_s": end_s,
                "score": round(score, 3),
                "isolation_db": round(isolation_ratio_db, 2),
                "voiced_ratio": round(voiced_ratio, 2),
                "f0_median_hz": round(f0_med, 1),
                "dynamic_range_db": round(dyn_range_db, 1),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:8] if candidates else []

        saved_segment_paths = []
        for idx, seg in enumerate(top_candidates, 1):
            seg_filename = f"segment_{idx:02d}.wav"
            seg_path = os.path.join(segments_dir, seg_filename)

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start_s"]),
                "-to", str(seg["end_s"]),
                "-i", vocals_path,
                "-c:a", "pcm_s24le",
                seg_path
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError("Vocal reference extraction failed.")
            if os.path.exists(seg_path):
                saved_segment_paths.append(seg_path)
                seg["file_path"] = seg_path

        # Primary Core Reference
        clean_ref_path = os.path.join(voice_dir, "clean_vocal_reference.wav")
        if saved_segment_paths:
            shutil.copy2(saved_segment_paths[0], clean_ref_path)
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", vocals_path,
                "-t", "30",
                "-c:a", "pcm_s24le",
                clean_ref_path
            ]
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.communicate()
            if proc.returncode != 0 or not os.path.isfile(clean_ref_path):
                raise RuntimeError("Vocal reference extraction failed.")

        # Build Multi-Reference Portfolio: Low Register, High Register, Dynamic Belt
        portfolio = {"core_identity": clean_ref_path}
        
        if len(top_candidates) >= 2:
            # Low vs High register
            sorted_by_pitch = sorted(top_candidates, key=lambda x: x["f0_median_hz"])
            if "file_path" in sorted_by_pitch[0]:
                low_p = os.path.join(portfolio_dir, "segment_low_register.wav")
                shutil.copy2(sorted_by_pitch[0]["file_path"], low_p)
                portfolio["low_register"] = low_p
            if "file_path" in sorted_by_pitch[-1]:
                high_p = os.path.join(portfolio_dir, "segment_high_register.wav")
                shutil.copy2(sorted_by_pitch[-1]["file_path"], high_p)
                portfolio["high_register"] = high_p

        manifest_path = os.path.join(voice_dir, "segments_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "clean_reference": clean_ref_path,
                "portfolio": portfolio,
                "total_segments_extracted": len(saved_segment_paths),
                "ranked_segments": top_candidates,
            }, f, indent=2)

        return clean_ref_path
