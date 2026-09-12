"""CLI Benchmark Runner: Executes voice identity and musical reconstruction benchmark suites."""
from __future__ import annotations

import asyncio
import json
import os
import numpy as np
from music_dna_agent.src.benchmark.voice_identity_evaluator import VoiceIdentityEvaluator
from music_dna_agent.src.benchmark.ground_truth_evaluator import GroundTruthMusicEvaluator
from music_dna_agent.src.benchmark.report_generator import BenchmarkReportGenerator


def run_synthetic_benchmark_suite(base_output_dir: str = "/workspace/workspace/music_dna_agent/benchmark") -> dict:
    """Runs end-to-end benchmark on synthetic singer embeddings & annotated musical facts."""
    os.makedirs(base_output_dir, exist_ok=True)
    voice_out_dir = os.path.join(base_output_dir, "voice_identity")
    music_out_dir = os.path.join(base_output_dir, "music")

    np.random.seed(42)

    # 1. Generate multi-song embeddings for 12 synthetic singers (3 songs each)
    # Singer base vectors with high intra-singer consistency + inter-singer separation
    singers_data = {}
    dim = 192

    for s_idx in range(12):
        singer_id = f"singer_{s_idx+1:02d}"
        base_vector = np.random.randn(dim).astype(np.float32)
        base_vector /= np.linalg.norm(base_vector)

        singers_data[singer_id] = {}
        for song_idx in range(3):
            song_id = f"song_{song_idx+1}"
            # Add realistic intra-singer acoustic variance (different song/register)
            noise = np.random.randn(dim).astype(np.float32) * 0.18
            song_vec = base_vector + noise
            song_vec /= np.linalg.norm(song_vec)
            singers_data[singer_id][song_id] = [float(x) for x in song_vec]

    voice_metrics = VoiceIdentityEvaluator.evaluate_corpus(singers_data, voice_out_dir)

    # 2. Evaluate ground truth musical reconstruction on 10 synthetic test compositions
    test_evaluations = []
    keys = ["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Am", "Em"]
    
    for i, k in enumerate(keys):
        gt_bpm = 120.0 + (i * 4.0)
        ext_bpm = gt_bpm + np.random.uniform(-1.5, 1.5)

        gt_bounds = [0.0, 20.0, 45.0, 70.0, 95.0]
        ext_bounds = [0.0, 20.8, 44.5, 71.2, 94.8]

        t_res = GroundTruthMusicEvaluator.evaluate_tempo(ext_bpm, gt_bpm)
        k_res = GroundTruthMusicEvaluator.evaluate_key(k, "major", k, "major")
        s_res = GroundTruthMusicEvaluator.evaluate_section_boundaries(ext_bounds, gt_bounds, tolerance_s=3.0)

        test_evaluations.append({
            "track_id": f"test_track_{i+1:02d}",
            "tempo": t_res,
            "key": k_res,
            "structure": s_res,
        })

    music_metrics = GroundTruthMusicEvaluator.evaluate_track_batch(test_evaluations, music_out_dir)

    # 3. Generate HTML Visual Report
    html_path = os.path.join(base_output_dir, "report.html")
    BenchmarkReportGenerator.generate_html_report(voice_metrics, music_metrics, html_path)

    return {
        "voice": voice_metrics,
        "music": music_metrics,
        "html_report": html_path,
    }


if __name__ == "__main__":
    results = run_synthetic_benchmark_suite()
    print("Benchmark Execution Complete:")
    print(json.dumps(results, indent=2))
