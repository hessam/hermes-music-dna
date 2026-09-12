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
    assert "low baritone" in prompts["vocal_prompt"].lower()
    assert "chest voice" in prompts["vocal_prompt"].lower()

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


def test_praat_formants_and_hnr_suno_interpretation():
    """Verify Praat F1-F4 formants and HNR are translated into 1:1 Suno vocal tokens."""
    m = VocalMeasurements(
        f0_distribution={"p5": 110.0, "p25": 140.0, "median": 165.0, "p75": 190.0, "p95": 220.0},
        f0_range_semitones=12.0,
        vibrato_rate_hz=5.5,
        vibrato_depth_semitones=0.8,
        vibrato_onset_median_ms=300.0,
        vibrato_onset_p25_ms=200.0,
        vibrato_onset_p75_ms=400.0,
        cpp_db=15.0,
        h1_h2_db=1.0,
        jitter_local_pct=0.8,
        shimmer_local_pct=3.0,
        subharmonic_energy_ratio=0.03,
        detected_registers=["chest", "mixed"],
        formants_hz={"f1": 850.0, "f2": 1850.0, "f3": 3050.0, "f4": 4100.0},
        hnr_db=3.5,
    )
    interp = AcousticInterpreter.interpret_timbre(m, 165.0)

    # Singer's formant ring (F3 = 3050 Hz) & forward twang (F2 = 1850 Hz)
    assert "singer's formant" in interp["formant_ring"].lower() or "twang" in interp["formant_ring"].lower()
    assert "singer's formant" in interp["vocal_tract_desc"].lower()
    assert "twang" in interp["vocal_tract_desc"].lower()

    # Low HNR (3.5 dB) translated to raspy/husky breath texture
    assert "husky" in interp["hnr_token"].lower() or "breath" in interp["hnr_token"].lower() or "raspy" in interp["hnr_token"].lower()

    dna = MusicalDNA(
        bpm=130.0,
        bpm_confidence=0.9,
        tempo_stability=0.95,
        swing_ratio=None,
        groove_offset_ms=0.0,
        key="A",
        scale="minor",
        chord_progression=["Am", "F", "C", "G"],
        chord_extensions=[],
        harmonic_rhythm=1.0,
        sections=[MusicalSection(label="verse_1", start_s=0.0, end_s=30.0, bpm=130.0, key="A", scale="minor")],
        vocal_dna=VocalDNA(
            measurements=m,
            identity=VocalIdentity(
                speaker_identity_embedding=[0.05] * 192,
                segment_count=2,
                mean_pairwise_cosine=0.92,
                std_pairwise_cosine=0.02,
                mfcc_trajectory_stats={"mean": [0.0] * 39, "std": [0.1] * 39, "p25": [0.0] * 39, "p75": [0.1] * 39},
                embedding_model="speechbrain/spkrec-ecapa-voxceleb",
                signal_reliability=0.92,
            ),
        ),
    )
    prompts = SunoAdapter.generate_prompts(dna)
    assert "singer's formant ring" in prompts["vocal_prompt"].lower() or "twang" in prompts["vocal_prompt"].lower()
    assert "Resonance & Vocal Tract" in prompts["detailed_persona_text"]
    assert "hnr" in prompts["detailed_persona_text"].lower()
