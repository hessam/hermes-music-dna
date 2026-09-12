"""Unit tests for the Benchmark validation subsystem."""
import os
import pytest
from music_dna_agent.src.benchmark.voice_identity_evaluator import VoiceIdentityEvaluator
from music_dna_agent.src.benchmark.ground_truth_evaluator import GroundTruthMusicEvaluator
from music_dna_agent.src.domain.confidence_calculator import MeasuredConfidenceEngine
from music_dna_agent.src.domain.models import VocalMeasurements, VocalIdentity


def test_voice_identity_evaluator_eer(tmp_path):
    singer_embeddings = {
        "singer_A": {
            "song_1": [1.0, 0.0, 0.0],
            "song_2": [0.95, 0.05, 0.0],
        },
        "singer_B": {
            "song_1": [0.0, 1.0, 0.0],
            "song_2": [0.0, 0.95, 0.05],
        }
    }
    metrics = VoiceIdentityEvaluator.evaluate_corpus(singer_embeddings, str(tmp_path))
    assert metrics["same_singer_similarity_mean"] > 0.9
    assert metrics["different_singer_similarity_mean"] < 0.1
    assert metrics["d_prime_separation"] > 5.0


def test_measured_confidence_engine():
    m = VocalMeasurements(
        f0_distribution={"median": 130.0},
        f0_range_semitones=14.0,
        vibrato_rate_hz=5.5,
        vibrato_depth_semitones=0.8,
        vibrato_onset_median_ms=300.0,
        vibrato_onset_p25_ms=200.0,
        vibrato_onset_p75_ms=400.0,
        cpp_db=16.0,
        h1_h2_db=3.5,
        jitter_local_pct=0.6,
        shimmer_local_pct=2.5,
        subharmonic_energy_ratio=0.04,
        detected_registers=["chest", "mixed"],
    )
    ident = VocalIdentity(
        speaker_identity_embedding=[0.2] * 192,
        segment_count=4,
        mean_pairwise_cosine=0.92,
        std_pairwise_cosine=0.03,
        mfcc_trajectory_stats={},
        embedding_model="speechbrain/spkrec-ecapa-voxceleb",
        signal_reliability=0.92,
    )
    res = MeasuredConfidenceEngine.calculate_voice_metrics(m, ident, voiced_ratio=0.85, isolation_db=14.0)
    assert 0.0 <= res["identity_consistency_pct"] <= 100.0
    assert 0.0 <= res["measurement_quality_pct"] <= 100.0
    assert res["contamination_risk_pct"] < 40.0
