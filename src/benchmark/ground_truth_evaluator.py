"""Music DNA & Ground-Truth Reconstruction Benchmark Suite."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple
import numpy as np


class GroundTruthMusicEvaluator:
    """Evaluates extracted musical facts against annotated ground truth."""

    @staticmethod
    def evaluate_tempo(extracted_bpm: float, ground_truth_bpm: float) -> Dict[str, float]:
        abs_err = abs(extracted_bpm - ground_truth_bpm)
        # Check octave tempo errors (half or double tempo)
        half_err = abs((extracted_bpm * 2) - ground_truth_bpm)
        double_err = abs((extracted_bpm / 2) - ground_truth_bpm)
        effective_err = min(abs_err, half_err, double_err)
        ape = (effective_err / ground_truth_bpm) * 100.0 if ground_truth_bpm > 0 else 0.0

        return {
            "extracted_bpm": round(extracted_bpm, 1),
            "ground_truth_bpm": round(ground_truth_bpm, 1),
            "absolute_error_bpm": round(effective_err, 2),
            "percentage_error_pct": round(ape, 2),
            "is_within_4_bpm": bool(effective_err <= 4.0),
        }

    @staticmethod
    def evaluate_key(extracted_key: str, extracted_scale: str, gt_key: str, gt_scale: str) -> Dict[str, any]:
        exact_match = (extracted_key.upper() == gt_key.upper() and extracted_scale.lower() == gt_scale.lower())
        
        # Circle of fifths / relative minor tolerance
        is_tonic_match = (extracted_key.upper() == gt_key.upper())
        
        return {
            "extracted": f"{extracted_key} {extracted_scale}",
            "ground_truth": f"{gt_key} {gt_scale}",
            "exact_match": exact_match,
            "tonic_match": is_tonic_match,
        }

    @staticmethod
    def evaluate_section_boundaries(
        extracted_boundaries: List[float],
        gt_boundaries: List[float],
        tolerance_s: float = 3.0,
    ) -> Dict[str, float]:
        """Calculates Precision, Recall, and F1 score for structural boundary detection."""
        if not extracted_boundaries or not gt_boundaries:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        true_positives = 0
        matched_gt = set()

        for eb in extracted_boundaries:
            for i, gb in enumerate(gt_boundaries):
                if i not in matched_gt and abs(eb - gb) <= tolerance_s:
                    true_positives += 1
                    matched_gt.add(i)
                    break

        precision = true_positives / len(extracted_boundaries) if extracted_boundaries else 0.0
        recall = true_positives / len(gt_boundaries) if gt_boundaries else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "tolerance_seconds": tolerance_s,
            "true_positives": true_positives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        }

    @classmethod
    def evaluate_track_batch(
        cls,
        evaluations: List[Dict[str, any]],
        output_dir: str,
    ) -> Dict[str, any]:
        os.makedirs(output_dir, exist_ok=True)

        bpm_errors = [e["tempo"]["percentage_error_pct"] for e in evaluations if "tempo" in e]
        key_matches = [1.0 if e["key"]["exact_match"] else 0.0 for e in evaluations if "key" in e]
        f1_scores = [e["structure"]["f1_score"] for e in evaluations if "structure" in e]

        summary = {
            "total_tracks_evaluated": len(evaluations),
            "mean_bpm_error_pct": round(float(np.mean(bpm_errors)), 2) if bpm_errors else 0.0,
            "tempo_accuracy_within_4bpm_pct": round(float(np.mean([1.0 if e["tempo"]["is_within_4_bpm"] else 0.0 for e in evaluations])) * 100, 1) if evaluations else 0.0,
            "key_exact_accuracy_pct": round(float(np.mean(key_matches)) * 100, 1) if key_matches else 0.0,
            "mean_section_boundary_f1": round(float(np.mean(f1_scores)), 3) if f1_scores else 0.0,
        }

        with open(os.path.join(output_dir, "music_reconstruction_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "tracks": evaluations}, f, indent=2)

        return summary
