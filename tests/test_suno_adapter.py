"""Unit tests for Suno generator adapter with refined zero-hardcoded interpreter."""
from music_dna_agent.src.domain.models import (
    MusicalDNA,
    MusicalSection,
    VocalMeasurements,
    VocalDNA,
    VocalIdentity,
)
from music_dna_agent.src.adapters.generators.suno_adapter import SunoAdapter, AcousticInterpreter


def test_suno_adapter_prompts_generation():
    dna = MusicalDNA(
        bpm=124.0,
        bpm_confidence=0.9,
        tempo_stability=0.95,
        swing_ratio=None,
        groove_offset_ms=5.0,
        key="E",
        scale="minor",
        chord_progression=["Em", "C", "G", "D"],
        chord_extensions=[],
        harmonic_rhythm=1.0,
        sections=[
            MusicalSection(label="verse_1", start_s=0.0, end_s=25.0, bpm=124.0, key="E", scale="minor"),
            MusicalSection(label="chorus_1", start_s=25.0, end_s=50.0, bpm=124.0, key="E", scale="minor"),
        ],
        vocal_dna=VocalDNA(
            measurements=VocalMeasurements(
                f0_distribution={"p5": 90.0, "p25": 110.0, "median": 135.0, "p75": 155.0, "p95": 180.0},
                f0_range_semitones=10.5,
                vibrato_rate_hz=5.2,
                vibrato_depth_semitones=0.7,
                vibrato_onset_median_ms=300.0,
                vibrato_onset_p25_ms=200.0,
                vibrato_onset_p75_ms=410.0,
                cpp_db=15.0,
                h1_h2_db=3.5,
                jitter_local_pct=0.6,
                shimmer_local_pct=2.8,
                subharmonic_energy_ratio=0.035,
                detected_registers=["chest"],
            ),
            identity=VocalIdentity(
                speaker_identity_embedding=[0.1] * 192,
                segment_count=3,
                mean_pairwise_cosine=0.91,
                std_pairwise_cosine=0.03,
                mfcc_trajectory_stats={"mean": [0.5] * 39, "std": [0.2] * 39, "p25": [0.3] * 39, "p75": [0.7] * 39},
                embedding_model="speechbrain/spkrec-ecapa-voxceleb",
                signal_reliability=0.91,
            ),
        ),
    )

    prompts = SunoAdapter.generate_prompts(dna)
    assert "style_prompt" in prompts
    assert "vocal_prompt" in prompts
    assert "detailed_persona_text" in prompts
    assert "negative_prompt" in prompts

    assert "E Minor" in prompts["style_prompt"]
    assert "124 BPM" in prompts["style_prompt"]
    assert "5.2Hz natural vibrato" in prompts["vocal_prompt"]
    assert "135Hz" in prompts["vocal_prompt"]

    md = SunoAdapter.format_summary_markdown(dna)
    assert "Music DNA Extraction & Generation Blueprint" in md
    assert "Em -> C -> G -> D" in md
    assert "Detailed Vocal Persona Description" in md


def test_ethereal_falsetto_interpretation():
    # Test with Thom Yorke-like parameters: 222 Hz with elevated breath air turbulence
    m = VocalMeasurements(
        f0_distribution={"p5": 100.0, "p25": 210.0, "median": 222.0, "p75": 230.0, "p95": 265.0},
        f0_range_semitones=16.8,
        vibrato_rate_hz=3.6,
        vibrato_depth_semitones=1.4,
        vibrato_onset_median_ms=300.0,
        vibrato_onset_p25_ms=200.0,
        vibrato_onset_p75_ms=400.0,
        cpp_db=12.5,
        h1_h2_db=2.6,
        jitter_local_pct=5.1,
        shimmer_local_pct=21.0,
        subharmonic_energy_ratio=0.09,
        detected_registers=["chest", "mixed"],
    )
    interp = AcousticInterpreter.interpret_timbre(m, 222.0)
    assert "falsetto" in interp["voice_type"].lower() or "high tenor" in interp["voice_type"].lower()
    assert "breathy" in interp["phonation"].lower() or "falsetto" in interp["phonation"].lower()
    assert "throat bite" not in interp["phonation"].lower()
