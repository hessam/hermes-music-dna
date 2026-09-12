"""Domain models and value objects for Music DNA Agent (Model-Independent IR)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class JobStatus(str, Enum):
    QUEUED = "queued"
    CHECKING_SOURCE_QC = "checking_source_qc"
    ISOLATING_STEMS = "isolating_stems"
    SEGMENTING_STRUCTURE = "segmenting_structure"
    RANKING_VOCAL_SEGMENTS = "ranking_vocal_segments"
    ANALYZING_ESSENTIA = "analyzing_essentia"
    EXTRACTING_PITCH = "extracting_pitch"
    EXTRACTING_GROOVE = "extracting_groove"
    ANALYZING_VOCAL_DNA = "analyzing_vocal_dna"
    EXTRACTING_VOICE_EMBEDDINGS = "extracting_voice_embeddings"
    RENDERING_SYNTH = "rendering_synth"
    MASTERING = "mastering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepName(str, Enum):
    SOURCE_QC = "source_qc"
    STEM_ISOLATION = "stem_isolation"
    STRUCTURE_SEGMENTATION = "structure_segmentation"
    VOCAL_SEGMENT_RANKING = "vocal_segment_ranking"
    ESSENTIA_ANALYSIS = "essentia_analysis"
    PITCH_EXTRACTION = "pitch_extraction"
    GROOVE_EXTRACTION = "groove_extraction"
    VOCAL_DNA_ANALYSIS = "vocal_dna_analysis"
    VOICE_EMBEDDING = "voice_embedding"
    SYNTH_SYNTHESIS = "synth_synthesis"
    MASTER_BOUNCE = "master_bounce"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class QualityOption(str, Enum):
    CRISP_30S = "crisp_30s"    # 30-second condensed seed
    FULL_60S = "full_60s"      # 60-second full reference
    VOCAL_DNA = "vocal_dna"    # Deep Vocal DNA profiling & high-signal acapella


@dataclass
class SourceQC:
    duration_s: float
    sample_rate: int
    lufs_integrated: float
    true_peak_dbtp: float
    is_stereo: bool
    clipping_detected: bool
    qc_passed: bool
    qc_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 2),
            "sample_rate": self.sample_rate,
            "lufs_integrated": round(self.lufs_integrated, 2),
            "true_peak_dbtp": round(self.true_peak_dbtp, 2),
            "is_stereo": self.is_stereo,
            "clipping_detected": self.clipping_detected,
            "qc_passed": self.qc_passed,
            "qc_notes": self.qc_notes,
        }


@dataclass
class NoteEvent:
    pitch: int          # MIDI note number (0-127)
    start_time: float   # seconds
    end_time: float     # seconds
    velocity: int       # 0-127
    pitch_name: str     # e.g. "C4", "F#3"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch": self.pitch,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "velocity": self.velocity,
            "pitch_name": self.pitch_name,
        }


@dataclass
class VocalMeasurements:
    """Raw, physically-grounded continuous acoustic measurements without subjective labels."""
    f0_distribution: Dict[str, float]               # P5, P25, median, P75, P95 in Hz
    f0_range_semitones: float                       # Dynamic vocal range in semitones
    vibrato_rate_hz: Optional[float]                # Modulation rate in 3.5-8.5 Hz band
    vibrato_depth_semitones: Optional[float]        # Modulation depth (+/- semitones)
    vibrato_onset_median_ms: Optional[float]        # Measured note-onset latency median
    vibrato_onset_p25_ms: Optional[float]           # 25th percentile onset latency
    vibrato_onset_p75_ms: Optional[float]           # 75th percentile onset latency
    cpp_db: float                                   # Cepstral Peak Prominence (dB)
    h1_h2_db: float                                 # Frame-interpolated H1 - H2 harmonic energy tilt (dB)
    jitter_local_pct: float                         # Cycle-to-cycle frequency perturbation (%)
    shimmer_local_pct: float                        # Cycle-to-cycle amplitude perturbation (%)
    subharmonic_energy_ratio: float                 # Period-doubling energy ratio (F0/2, 3F0/2 vs F0)
    detected_registers: List[str]                   # Discovered acoustic registers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f0_distribution": {k: round(v, 2) for k, v in self.f0_distribution.items()},
            "f0_range_semitones": round(self.f0_range_semitones, 2),
            "vibrato_rate_hz": round(self.vibrato_rate_hz, 2) if self.vibrato_rate_hz else None,
            "vibrato_depth_semitones": round(self.vibrato_depth_semitones, 2) if self.vibrato_depth_semitones else None,
            "vibrato_onset_ms": {
                "median": round(self.vibrato_onset_median_ms, 1) if self.vibrato_onset_median_ms else None,
                "p25": round(self.vibrato_onset_p25_ms, 1) if self.vibrato_onset_p25_ms else None,
                "p75": round(self.vibrato_onset_p75_ms, 1) if self.vibrato_onset_p75_ms else None,
            },
            "cpp_db": round(self.cpp_db, 2),
            "h1_h2_db": round(self.h1_h2_db, 2),
            "jitter_local_pct": round(self.jitter_local_pct, 3),
            "shimmer_local_pct": round(self.shimmer_local_pct, 3),
            "subharmonic_energy_ratio": round(self.subharmonic_energy_ratio, 4),
            "detected_registers": self.detected_registers,
        }


@dataclass
class VocalIdentity:
    """Multi-segment learned speaker identity representation & temporal feature statistics."""
    speaker_identity_embedding: List[float]         # 192-dim L2-normalized centroid vector
    segment_count: int                              # Number of reference segments evaluated
    mean_pairwise_cosine: float                     # Intra-speaker embedding consistency mean
    std_pairwise_cosine: float                      # Intra-speaker embedding consistency std
    mfcc_trajectory_stats: Dict[str, List[float]]   # 39-dim statistics: mean, std, p25, p75
    embedding_model: str                            # e.g. "speechbrain/spkrec-ecapa-voxceleb"
    signal_reliability: float                       # Derived from pairwise cosine consistency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_identity_embedding": [round(x, 5) for x in self.speaker_identity_embedding],
            "segment_count": self.segment_count,
            "mean_pairwise_cosine": round(self.mean_pairwise_cosine, 4),
            "std_pairwise_cosine": round(self.std_pairwise_cosine, 4),
            "mfcc_trajectory_stats": {
                k: [round(v, 4) for v in vals] for k, vals in self.mfcc_trajectory_stats.items()
            },
            "embedding_model": self.embedding_model,
            "signal_reliability": round(self.signal_reliability, 3),
        }


@dataclass
class SectionVocalProfile:
    """Dynamic vocal character for a specific song section."""
    label: str
    start_s: float
    end_s: float
    f0_median_hz: float
    intensity_mean_lufs: float
    dynamic_range_db: float
    dominant_register: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "f0_median_hz": round(self.f0_median_hz, 1),
            "intensity_mean_lufs": round(self.intensity_mean_lufs, 1),
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "dominant_register": self.dominant_register,
        }


@dataclass
class VocalDNA:
    """Complete Model-Independent Vocal Representation with Multi-Reference Portfolio."""
    measurements: VocalMeasurements
    identity: VocalIdentity
    section_profiles: List[SectionVocalProfile] = field(default_factory=list)
    reference_portfolio: Dict[str, str] = field(default_factory=dict)
    clean_reference_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurements": self.measurements.to_dict(),
            "identity": self.identity.to_dict(),
            "section_profiles": [s.to_dict() for s in self.section_profiles],
            "reference_portfolio": self.reference_portfolio,
            "clean_reference_path": self.clean_reference_path,
        }


@dataclass
class MusicalSection:
    label: str
    start_s: float
    end_s: float
    bpm: float
    key: str
    scale: str
    chord_progression: List[str] = field(default_factory=list)
    energy_intensity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "bpm": round(self.bpm, 1),
            "key": self.key,
            "scale": self.scale,
            "chord_progression": self.chord_progression,
            "energy_intensity": round(self.energy_intensity, 2),
        }


@dataclass
class MusicalDNA:
    """Model-Independent Intermediate Representation (IR) of complete composition."""
    bpm: float
    bpm_confidence: float
    tempo_stability: float
    swing_ratio: Optional[float]
    groove_offset_ms: float

    key: str
    scale: str
    chord_progression: List[str]
    chord_extensions: List[str]
    harmonic_rhythm: float

    sections: List[MusicalSection] = field(default_factory=list)
    dominant_instruments: List[str] = field(default_factory=list)
    arrangement_density: float = 0.0
    melody_notes: List[NoteEvent] = field(default_factory=list)

    lufs_integrated: float = -14.0
    lufs_range: float = 6.0
    stereo_width: float = 0.5
    bass_to_mid_ratio: float = 1.0
    spectral_balance: Dict[str, float] = field(default_factory=dict)

    vocal_dna: Optional[VocalDNA] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "tempo": {
                "bpm": round(self.bpm, 2),
                "bpm_confidence": round(self.bpm_confidence, 2),
                "tempo_stability": round(self.tempo_stability, 3),
                "swing_ratio": round(self.swing_ratio, 2) if self.swing_ratio else None,
                "groove_offset_ms": round(self.groove_offset_ms, 2),
            },
            "harmony": {
                "key": self.key,
                "scale": self.scale,
                "chord_progression": self.chord_progression,
                "chord_extensions": self.chord_extensions,
                "harmonic_rhythm": round(self.harmonic_rhythm, 2),
            },
            "arrangement": {
                "sections": [s.to_dict() for s in self.sections],
                "dominant_instruments": self.dominant_instruments,
                "arrangement_density": round(self.arrangement_density, 2),
                "melody_notes_count": len(self.melody_notes),
            },
            "mix": {
                "lufs_integrated": round(self.lufs_integrated, 2),
                "lufs_range": round(self.lufs_range, 2),
                "stereo_width": round(self.stereo_width, 2),
                "bass_to_mid_ratio": round(self.bass_to_mid_ratio, 2),
                "spectral_balance": {k: round(v, 3) for k, v in self.spectral_balance.items()},
            }
        }
        if self.vocal_dna:
            res["vocal_dna"] = self.vocal_dna.to_dict()
        return res


@dataclass
class JobStep:
    name: StepName
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int = 0
    chat_id: int = 0
    original_filename: str = ""
    input_file_path: str = ""
    quality: QualityOption = QualityOption.CRISP_30S
    custom_style_prompt: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    steps: Dict[StepName, JobStep] = field(default_factory=dict)
    source_qc: Optional[SourceQC] = None
    dna: Optional[MusicalDNA] = None
    output_manifest: Dict[str, str] = field(default_factory=dict)
    guide_track_path: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.steps:
            for step_name in StepName:
                self.steps[step_name] = JobStep(name=step_name)

    def mark_step_start(self, step_name: StepName) -> None:
        if step_name in self.steps:
            self.steps[step_name].status = StepStatus.RUNNING
            self.steps[step_name].started_at = time.time()
        self.updated_at = time.time()

    def mark_step_done(self, step_name: StepName, output_path: Optional[str] = None) -> None:
        if step_name in self.steps:
            self.steps[step_name].status = StepStatus.DONE
            self.steps[step_name].finished_at = time.time()
            self.steps[step_name].output_path = output_path
        self.updated_at = time.time()

    def mark_step_failed(self, step_name: StepName, error: str) -> None:
        if step_name in self.steps:
            self.steps[step_name].status = StepStatus.FAILED
            self.steps[step_name].finished_at = time.time()
            self.steps[step_name].error = error
        self.status = JobStatus.FAILED
        self.error_message = error
        self.updated_at = time.time()
