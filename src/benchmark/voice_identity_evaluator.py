"""Voice Identity & Singer Verification Benchmark Suite."""
from __future__ import annotations

from array import array
import csv
import json
import math
import os
import shutil
import tempfile
from typing import Dict, List, Tuple
import numpy as np


class VoiceIdentityEvaluator:
    """Evaluates speaker embedding stability across songs, registers, and production styles."""

    @staticmethod
    def _equal_error_rate(same: np.ndarray, different: np.ndarray) -> Tuple[float, float]:
        """Search the original threshold grid without rescanning scores per threshold."""
        thresholds = np.linspace(-0.2, 1.0, 500)
        def counts_below(scores: np.ndarray) -> Tuple[np.ndarray, int]:
            bins = np.zeros(len(thresholds) + 1, dtype=np.int64)
            valid_count = len(scores)
            for start in range(0, len(scores), 65536):
                chunk = scores[start:start + 65536]
                # NaNs compare false but remain in the rate denominator. NumPy
                # places them beyond the last threshold, alongside +infinity.
                valid_count -= int(np.count_nonzero(np.isnan(chunk)))
                positions = np.searchsorted(thresholds, chunk, side="right")
                bins += np.bincount(positions, minlength=len(bins))
            return np.cumsum(bins)[:-1], valid_count

        same_below, _ = counts_below(same)
        different_below, different_valid = counts_below(different)
        frr = (same_below / len(same)
               if len(same) else np.zeros_like(thresholds))
        far = ((different_valid - different_below) / len(different)
               if len(different) else np.zeros_like(thresholds))
        differences = np.abs(far - frr)
        index = int(np.argmin(differences))  # First threshold wins ties.
        if differences[index] < 1.0:
            return float(0.5 * (far[index] + frr[index])), float(thresholds[index])
        return 0.5, 0.5

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
        
        same_singer_scores = array("d")
        diff_singer_scores = array("d")

        singers = list(singer_embeddings.keys())

        # Stage CSV rows on disk instead of retaining one dictionary per pair.
        with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as pair_csv:
            writer = csv.DictWriter(pair_csv, fieldnames=["type", "singer_1", "song_1", "singer_2", "song_2", "similarity"])
            writer.writeheader()
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
                        writer.writerow({
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
                            writer.writerow({
                                "type": "diff_singer",
                                "singer_1": s1,
                                "song_1": k1,
                                "singer_2": s2,
                                "song_2": k2,
                                "similarity": round(sim, 4),
                            })

            same_arr = np.asarray(same_singer_scores) if same_singer_scores else np.array([0.0])
            diff_arr = np.asarray(diff_singer_scores) if diff_singer_scores else np.array([0.0])

            mu_same, std_same = float(np.mean(same_arr)), float(np.std(same_arr))
            mu_diff, std_diff = float(np.mean(diff_arr)), float(np.std(diff_arr))

            # Separation metric (d-prime)
            d_prime = float((mu_same - mu_diff) / math.sqrt(0.5 * (std_same**2 + std_diff**2) + 1e-8))

            # Equal Error Rate (EER) approximation
            eer, eer_thresh = cls._equal_error_rate(same_arr, diff_arr)

            # Publish only after scoring succeeds, preserving existing files on failure.
            csv_path = os.path.join(output_dir, "pairwise_similarities.csv")
            pair_csv.seek(0)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                shutil.copyfileobj(pair_csv, f)

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
