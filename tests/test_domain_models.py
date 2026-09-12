"""Unit tests for domain models, state transitions, and model-independent representation."""
import pytest
from music_dna_agent.src.domain.models import (
    Job,
    JobStatus,
    MusicalDNA,
    NoteEvent,
    QualityOption,
    StepName,
    StepStatus,
    VocalMeasurements,
    VocalDNA,
    VocalIdentity,
)


def test_job_initialization():
    job = Job(
        user_id=12345,
        chat_id=12345,
        original_filename="sample.mp3",
        input_file_path="/workspace/sample.mp3",
        quality=QualityOption.CRISP_30S,
    )
    assert job.status == JobStatus.QUEUED
    assert len(job.steps) == len(StepName)


def test_model_independent_musical_dna_serialization():
    dna = MusicalDNA(
        bpm=128.0,
        bpm_confidence=0.95,
        tempo_stability=0.98,
        swing_ratio=None,
        groove_offset_ms=4.2,
        key="A",
        scale="minor",
        chord_progression=["Am", "F", "C", "G"],
        chord_extensions=[],
        harmonic_rhythm=1.0,
        melody_notes=[
            NoteEvent(pitch=69, start_time=0.0, end_time=0.5, velocity=80, pitch_name="A4")
        ],
        vocal_dna=VocalDNA(
            measurements=VocalMeasurements(
                f0_distribution={"p5": 100.0, "p25": 120.0, "median": 140.0, "p75": 160.0, "p95": 190.0},
                f0_range_semitones=11.2,
                vibrato_rate_hz=5.4,
                vibrato_depth_semitones=0.8,
                vibrato_onset_median_ms=280.0,
                vibrato_onset_p25_ms=180.0,
                vibrato_onset_p75_ms=390.0,
                cpp_db=14.2,
                h1_h2_db=4.1,
                jitter_local_pct=0.75,
                shimmer_local_pct=3.1,
                subharmonic_energy_ratio=0.045,
                detected_registers=["chest", "mixed"],
            ),
            identity=VocalIdentity(
                speaker_identity_embedding=[0.05] * 192,
                segment_count=4,
                mean_pairwise_cosine=0.92,
                std_pairwise_cosine=0.03,
                mfcc_trajectory_stats={"mean": [1.2] * 39, "std": [0.3] * 39, "p25": [0.9] * 39, "p75": [1.4] * 39},
                embedding_model="speechbrain/spkrec-ecapa-voxceleb",
                signal_reliability=0.92,
            ),
        ),
    )

    data = dna.to_dict()
    assert data["tempo"]["bpm"] == 128.0
    assert data["harmony"]["key"] == "A"
    assert data["vocal_dna"]["measurements"]["vibrato_rate_hz"] == 5.4
    assert len(data["vocal_dna"]["identity"]["speaker_identity_embedding"]) == 192
    assert "suggested_suno_prompt" not in data
