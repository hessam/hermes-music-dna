"""Voice Identity & Singer Verification Benchmark Suite."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Tuple
import numpy as np


class VoiceIdentityEvaluator:
    """Evaluates speaker embedding stability across songs, registers, and production styles."""

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @classmethod
    def evaluate_corpus(
        cls,
        singer_embeddings: Dict[str, Dict[str, List[float]]],  # {singer_id: {song_id: embedding}}
        output_dir: str,
    ) -> Dict[str, any]:
        """Runs complete same-singer vs different-singer verification evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        
        same_singer_scores = []
        diff_singer_scores = []
        pair_records = []

        singers = list(singer_embeddings.keys())

        # 1. Compute all pairwise comparisons
        for i, s1 in enumerate(singers):
            songs_s1 = singer_embeddings[s1]
            song_keys_1 = list(songs_s1.keys())

            # A. Same singer (Intra-class)
            for j in range(len(song_keys_1)):
                for k in range(j + 1, len(song_keys_1)):
                    k1, k2 = song_keys_1[j], song_keys_1[k]
                    sim = cls.cosine_similarity(songs_s1[k1], songs_s1[k2])
                    same_singer_scores.append(sim)
                    pair_records.append({
                        "type": "same_singer",
                        "singer_1": s1,
                        "song_1": k1,
                        "singer_2": s1,
                        "song_2": k2,
                        "similarity": round(sim, 4),
                    })

            # B. Different singer (Inter-class)
            for j in range(i + 1, len(singers)):
                s2 = singers[j]
                songs_s2 = singer_embeddings[s2]
                for k1, emb1 in songs_s1.items():
                    for k2, emb2 in songs_s2.items():
                        sim = cls.cosine_similarity(emb1, emb2)
                        diff_singer_scores.append(sim)
                        pair_records.append({
                            "type": "diff_singer",
                            "singer_1": s1,
                            "song_1": k1,
                            "singer_2": s2,
                            "song_2": k2,
                            "similarity": round(sim, 4),
                        })

        same_arr = np.array(same_singer_scores) if same_singer_scores else np.array([0.0])
        diff_arr = np.array(diff_singer_scores) if diff_singer_scores else np.array([0.0])

        mu_same, std_same = float(np.mean(same_arr)), float(np.std(same_arr))
        mu_diff, std_diff = float(np.mean(diff_arr)), float(np.std(diff_arr))

        # Separation metric (d-prime)
        d_prime = float((mu_same - mu_diff) / math.sqrt(0.5 * (std_same**2 + std_diff**2) + 1e-8))

        # Equal Error Rate (EER) approximation
        thresholds = np.linspace(-0.2, 1.0, 500)
        min_eer_diff = 1.0
        eer = 0.5
        eer_thresh = 0.5

        for th in thresholds:
            far = np.mean(diff_arr >= th) if len(diff_arr) > 0 else 0.0  # False Acceptance
            frr = np.mean(same_arr < th) if len(same_arr) > 0 else 0.0   # False Rejection
            diff = abs(far - frr)
            if diff < min_eer_diff:
                min_eer_diff = diff
                eer = float(0.5 * (far + frr))
                eer_thresh = float(th)

        # 2. Export Pairs CSV
        csv_path = os.path.join(output_dir, "pairwise_similarities.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["type", "singer_1", "song_1", "singer_2", "song_2", "similarity"])
            writer.writeheader()
            writer.writerows(pair_records)

        # 3. Export Summary JSON
        metrics = {
            "total_singers": len(singers),
            "same_singer_pairs": len(same_singer_scores),
            "different_singer_pairs": len(diff_singer_scores),
            "same_singer_similarity_mean": round(mu_same, 4),
            "same_singer_similarity_std": round(std_same, 4),
            "different_singer_similarity_mean": round(mu_diff, 4),
            "different_singer_similarity_std": round(std_diff, 4),
            "d_prime_separation": round(d_prime, 3),
            "equal_error_rate_eer": round(eer, 4),
            "optimal_threshold": round(eer_thresh, 4),
            "verification_confidence_grade": "A" if d_prime > 2.5 else ("B" if d_prime > 1.5 else "C"),
        }

        json_path = os.path.join(output_dir, "voice_identity_benchmark.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return metrics
