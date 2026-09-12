"""Mathematical Confidence & Reliability Engine (Decoupled Measurements vs Reliability)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np


class MeasuredConfidenceEngine:
    """Computes mathematically derived reliability scores from physical measurements & cluster stability."""

    @staticmethod
    def calculate_voice_metrics(
        measurements,  # VocalMeasurements
        identity,      # VocalIdentity
        voiced_ratio: float,
        isolation_db: float,
    ) -> Dict[str, float]:
        # 1. Identity Consistency: derived directly from multi-segment pairwise cosine correlation
        intra_consistency = float(identity.mean_pairwise_cosine)
        iso_factor = min(1.0, max(0.0, (isolation_db + 8.0) / 24.0))
        voiced_factor = min(1.0, max(0.0, voiced_ratio))
        
        identity_reliability = (0.50 * intra_consistency + 0.30 * voiced_factor + 0.20 * iso_factor) * 100.0

        # 2. Measurement Quality: derived from CPP and lack of excessive glottal instability
        cpp_factor = min(1.0, max(0.0, measurements.cpp_db / 18.0))
        jitter_penalty = max(0.0, (measurements.jitter_local_pct - 0.5) / 3.0)
        shimmer_penalty = max(0.0, (measurements.shimmer_local_pct - 2.0) / 10.0)
        measurement_quality = max(0.0, (0.6 * cpp_factor + 0.4 * iso_factor) - (0.15 * jitter_penalty + 0.15 * shimmer_penalty)) * 100.0

        # 3. Register & Pitch Coverage
        reg_count = len(measurements.detected_registers)
        register_cov = min(100.0, (reg_count / 3.0) * 100.0)
        pitch_cov = min(100.0, (measurements.f0_range_semitones / 24.0) * 100.0)

        # 4. Contamination floor: from subharmonic ratio and bleed floor
        contam = min(100.0, max(0.0, (1.0 - iso_factor) * 50.0 + (measurements.subharmonic_energy_ratio * 40.0)))

        return {
            "identity_consistency_pct": round(identity_reliability, 1),
            "measurement_quality_pct": round(measurement_quality, 1),
            "register_coverage_pct": round(register_cov, 1),
            "pitch_coverage_pct": round(pitch_cov, 1),
            "contamination_risk_pct": round(contam, 1),
        }

    @staticmethod
    def calculate_music_metrics(
        bpm_confidence: float,
        key_strength: float,
        chord_progression: List[str],
        sections: list,
        lufs_range: float,
    ) -> Dict[str, float]:
        tempo_conf = min(100.0, max(0.0, bpm_confidence * 100.0))
        key_conf = min(100.0, max(0.0, key_strength * 100.0))
        harmony_conf = 85.0 if len(chord_progression) >= 3 else (50.0 if chord_progression else 20.0)
        struct_conf = 90.0 if len(sections) >= 3 else (65.0 if sections else 30.0)
        prod_conf = min(100.0, max(40.0, (lufs_range / 12.0) * 80.0 + 20.0))

        return {
            "tempo_confidence_pct": round(tempo_conf, 1),
            "key_confidence_pct": round(key_conf, 1),
            "harmony_confidence_pct": round(harmony_conf, 1),
            "structure_confidence_pct": round(struct_conf, 1),
            "production_confidence_pct": round(prod_conf, 1),
        }
