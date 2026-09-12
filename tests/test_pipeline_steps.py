"""Unit tests for pipeline adapters: SourceQC, SegmentRanker, VocalDNAAnalyzer, and EssentiaAnalyzer."""
import os
import pytest
import numpy as np
import soundfile as sf
from music_dna_agent.src.domain.models import Job, QualityOption, StepName, StepStatus
from music_dna_agent.src.adapters.pipeline.source_qc import SourceQCGate
from music_dna_agent.src.adapters.pipeline.vocal_segment_ranker import VocalSegmentRanker
from music_dna_agent.src.adapters.pipeline.vocal_dna_analyzer import VocalDNAAnalyzer
from music_dna_agent.src.adapters.pipeline.essentia_analyzer import EssentiaAnalyzer


@pytest.fixture
def synthetic_audio_path(tmp_path):
    sr = 22050
    duration = 15.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    vib = 3.0 * np.sin(2 * np.pi * 5.2 * t)
    sig = 0.7 * np.sin(2 * np.pi * (130.0 + vib) * t)
    
    file_path = str(tmp_path / "synthetic_test.wav")
    sf.write(file_path, sig, sr)
    return file_path


@pytest.mark.asyncio
async def test_source_qc_gate(synthetic_audio_path, tmp_path):
    job = Job(
        original_filename="test.wav",
        input_file_path=synthetic_audio_path,
        quality=QualityOption.CRISP_30S,
    )
    gate = SourceQCGate()
    out_json = await gate.execute(job, str(tmp_path))
    assert os.path.exists(out_json)
    assert job.source_qc.qc_passed is True


@pytest.mark.asyncio
async def test_vocal_dna_analyzer(synthetic_audio_path, tmp_path):
    job = Job(
        original_filename="test.wav",
        input_file_path=synthetic_audio_path,
        quality=QualityOption.VOCAL_DNA,
    )
    analyzer = VocalDNAAnalyzer()
    out_json = await analyzer.execute(job, str(tmp_path))
    assert os.path.exists(out_json)
    assert job.dna.vocal_dna is not None
    m = job.dna.vocal_dna.measurements
    assert 100.0 <= m.f0_distribution["median"] <= 160.0
    assert m.cpp_db > 0.0
    assert "chest" in m.detected_registers or "mixed" in m.detected_registers


@pytest.mark.asyncio
async def test_essentia_analyzer(synthetic_audio_path, tmp_path):
    job = Job(
        original_filename="test.wav",
        input_file_path=synthetic_audio_path,
        quality=QualityOption.FULL_60S,
    )
    analyzer = EssentiaAnalyzer()
    out_json = await analyzer.execute(job, str(tmp_path))
    assert os.path.exists(out_json)
    assert job.dna.key is not None
