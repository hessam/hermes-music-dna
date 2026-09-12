"""Phase 4: Multi-Segment Speaker Identity Embedding & MFCC Trajectory Statistics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from music_dna_agent.src.domain.models import Job, StepName, VocalIdentity
from music_dna_agent.src.domain.ports import PipelineStepPort
from music_dna_agent.src.adapters.pipeline.analysis_export import write_vocal_analysis


class VoiceEmbedder(PipelineStepPort):
    """Extracts multi-segment speaker centroid vectors (ECAPA-TDNN) and temporal MFCC trajectory statistics."""

    @property
    def name(self) -> StepName:
        return StepName.VOICE_EMBEDDING

    async def execute(self, job: Job, workspace_dir: str) -> str:
        voice_dir = os.path.join(workspace_dir, "voice")
        segments_dir = os.path.join(voice_dir, "segments")
        clean_ref_path = os.path.join(voice_dir, "clean_vocal_reference.wav")

        # 1. Gather all candidate audio segments for multi-segment analysis
        segment_files = []
        if os.path.isdir(segments_dir):
            for p in sorted(Path(segments_dir).glob("segment_*.wav")):
                segment_files.append(str(p))

        if not segment_files:
            if os.path.exists(clean_ref_path):
                segment_files = [clean_ref_path]
            else:
                segment_files = [job.input_file_path]

        # 2. Extract ECAPA-TDNN Embeddings per Segment
        embeddings_list = []
        embedded_paths = []
        model_name = "speechbrain/spkrec-ecapa-voxceleb"
        classifier = None

        try:
            from speechbrain.inference.speaker import EncoderClassifier
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
        except Exception:
            classifier = None
            model_name = "spectral-mel-192d"

        for seg_path in segment_files[:6]:  # evaluate up to top 6 segments
            try:
                y_seg, sr_seg = librosa.load(seg_path, sr=16000, duration=20.0)
                if len(y_seg) < 16000 * 2:  # skip shorter than 2s
                    continue
                if classifier is not None:
                    wav_tensor = torch.tensor(y_seg).unsqueeze(0)
                    with torch.inference_mode():
                        emb = classifier.encode_batch(wav_tensor).squeeze().cpu().numpy()
                    norm_val = np.linalg.norm(emb)
                    if np.isfinite(emb).all() and norm_val > 0:
                        embeddings_list.append(emb / norm_val)
                        embedded_paths.append(seg_path)
                else:
                    mel = librosa.feature.melspectrogram(y=y_seg, sr=sr_seg, n_mels=192)
                    mel_mean = np.mean(mel, axis=1)
                    norm_val = np.linalg.norm(mel_mean) + 1e-8
                    embeddings_list.append(mel_mean / norm_val)
                    embedded_paths.append(seg_path)
            except Exception:
                continue

        if not embeddings_list:
            # Fallback
            model_name = "spectral-mel-192d"
            y_fb, sr_fb = librosa.load(clean_ref_path if os.path.exists(clean_ref_path) else job.input_file_path, sr=16000, duration=20.0)
            mel = librosa.feature.melspectrogram(y=y_fb, sr=sr_fb, n_mels=192)
            mel_mean = np.mean(mel, axis=1)
            norm_val = np.linalg.norm(mel_mean) + 1e-8
            embeddings_list = [mel_mean / norm_val]
            embedded_paths = [clean_ref_path if os.path.exists(clean_ref_path) else job.input_file_path]

        # 3. Compute Centroid Vector & Pairwise Cosine Consistency
        emb_arr = np.array(embeddings_list)
        centroid = np.mean(emb_arr, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm

        pairwise_cosines = []
        if len(embeddings_list) > 1:
            for i in range(len(embeddings_list)):
                for j in range(i + 1, len(embeddings_list)):
                    cos_sim = float(np.dot(embeddings_list[i], embeddings_list[j]))
                    pairwise_cosines.append(cos_sim)

        mean_cosine = float(np.mean(pairwise_cosines)) if pairwise_cosines else 1.0
        std_cosine = float(np.std(pairwise_cosines)) if pairwise_cosines else 0.0

        # 4. Temporal MFCC Trajectory Statistics across clean reference
        y_ref, sr_ref = librosa.load(clean_ref_path if os.path.exists(clean_ref_path) else job.input_file_path, sr=16000, duration=30.0)
        mfcc = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_stack = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # 39 x T

        mfcc_stats = {
            "mean": [float(x) for x in np.mean(mfcc_stack, axis=1)],
            "std": [float(x) for x in np.std(mfcc_stack, axis=1)],
            "p25": [float(x) for x in np.percentile(mfcc_stack, 25, axis=1)],
            "p75": [float(x) for x in np.percentile(mfcc_stack, 75, axis=1)],
        }

        # A single segment or a spectral fallback cannot establish speaker identity.
        signal_reliability = (float(max(0.0, min(1.0, mean_cosine)))
                              if classifier is not None and model_name != "spectral-mel-192d"
                              and len(embeddings_list) > 1 else 0.0)

        identity = VocalIdentity(
            speaker_identity_embedding=[float(x) for x in centroid],
            segment_count=len(embeddings_list),
            mean_pairwise_cosine=mean_cosine,
            std_pairwise_cosine=std_cosine,
            mfcc_trajectory_stats=mfcc_stats,
            embedding_model=model_name,
            signal_reliability=signal_reliability,
        )

        if job.dna and job.dna.vocal_dna:
            job.dna.vocal_dna.identity = identity

        analysis_dir = os.path.join(workspace_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        embed_json_path = os.path.join(analysis_dir, "voice_embedding.json")

        with open(embed_json_path, "w", encoding="utf-8") as f:
            json.dump(identity.to_dict(), f, indent=2, allow_nan=False)

        if job.dna and job.dna.vocal_dna:
            write_vocal_analysis(job, analysis_dir, raw_identity={
                "embedding_model": model_name,
                "learned_speaker_embedding": model_name != "spectral-mel-192d",
                "verification_status": "not_evaluated_against_voice_vault",
                "sample_rate_hz": 16000,
                "segment_paths": embedded_paths,
                "segment_embeddings": [embedding.tolist() for embedding in embeddings_list],
                "centroid_embedding": centroid.tolist(),
                "pairwise_cosines": pairwise_cosines,
                "mfcc_trajectory_stats": mfcc_stats,
                "mfcc_frame_times_s": (np.arange(mfcc_stack.shape[1]) * 512 / sr_ref).tolist(),
                "mfcc_trajectory": mfcc_stack.tolist(),
            })

        return embed_json_path
