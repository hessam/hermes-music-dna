"""HTML & SVG Benchmark Visualizer and Metric Report Generator."""
from __future__ import annotations

import json
import os
from typing import Dict


class BenchmarkReportGenerator:
    """Generates visual dashboard report for voice cloning & music reconstruction benchmarks."""

    @classmethod
    def generate_html_report(
        cls,
        voice_metrics: Dict[str, any],
        music_metrics: Dict[str, any],
        output_path: str,
    ) -> str:
        d_prime = voice_metrics.get("d_prime_separation", 0.0)
        eer = voice_metrics.get("equal_error_rate_eer", 0.0) * 100.0
        same_mu = voice_metrics.get("same_singer_similarity_mean", 0.0)
        diff_mu = voice_metrics.get("different_singer_similarity_mean", 0.0)
        grade = voice_metrics.get("verification_confidence_grade", "N/A")

        tempo_acc = music_metrics.get("tempo_accuracy_within_4bpm_pct", 0.0)
        key_acc = music_metrics.get("key_exact_accuracy_pct", 0.0)
        struct_f1 = music_metrics.get("mean_section_boundary_f1", 0.0) * 100.0

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Music DNA & Voice Cloning Verification Benchmark</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ font-size: 28px; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }}
        p.subtitle {{ color: #94a3b8; font-size: 15px; margin-top: 0; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; }}
        .card h2 {{ font-size: 16px; font-weight: 600; color: #cbd5e1; margin-top: 0; text-transform: uppercase; letter-spacing: 0.05em; }}
        .metric-row {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .metric-label {{ color: #94a3b8; font-size: 14px; }}
        .metric-val {{ font-size: 18px; font-weight: 700; font-family: ui-monospace, monospace; color: #f1f5f9; }}
        .val-highlight {{ color: #38bdf8; }}
        .val-good {{ color: #4ade80; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700; background: #0369a1; color: #e0f2fe; }}
        .viz-box {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin-bottom: 30px; }}
        .bar-container {{ margin: 16px 0; }}
        .bar-label {{ font-size: 13px; color: #94a3b8; margin-bottom: 6px; display: flex; justify-content: space-between; }}
        .bar-track {{ height: 12px; background: #334155; border-radius: 6px; overflow: hidden; }}
        .bar-fill-same {{ height: 100%; background: #38bdf8; width: {min(100, max(0, int(same_mu * 100)))}%; }}
        .bar-fill-diff {{ height: 100%; background: #f43f5e; width: {min(100, max(0, int(diff_mu * 100)))}%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Music DNA & Voice Identity Benchmark</h1>
        <p class="subtitle">Empirical Verification Suite: Speaker Separation, Metric Stability & Reconstruction Fidelity</p>
        
        <div class="grid">
            <div class="card">
                <h2>🎙️ Voice Identity Metrics</h2>
                <div class="metric-row">
                    <span class="metric-label">Verification Separation (d')</span>
                    <span class="metric-val val-highlight">{d_prime:.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Equal Error Rate (EER)</span>
                    <span class="metric-val val-good">{eer:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Same-Singer Similarity</span>
                    <span class="metric-val">{same_mu:.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Different-Singer Similarity</span>
                    <span class="metric-val">{diff_mu:.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Verification Grade</span>
                    <span class="badge">GRADE {grade}</span>
                </div>
            </div>

            <div class="card">
                <h2>🎵 Music Reconstruction Fidelity</h2>
                <div class="metric-row">
                    <span class="metric-label">Tempo Accuracy (≤ 4 BPM)</span>
                    <span class="metric-val val-good">{tempo_acc:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Key & Mode Accuracy</span>
                    <span class="metric-val val-highlight">{key_acc:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Structural Boundary F1</span>
                    <span class="metric-val">{struct_f1:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Tracks Evaluated</span>
                    <span class="metric-val">{music_metrics.get("total_tracks_evaluated", 0)}</span>
                </div>
            </div>
        </div>

        <div class="viz-box">
            <h2>📊 Embedding Similarity Separation (Cosine Distance)</h2>
            <div class="bar-container">
                <div class="bar-label"><span>Same Singer Pair Similarity (&mu; = {same_mu:.3f})</span><span>{same_mu*100:.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill-same"></div></div>
            </div>
            <div class="bar-container">
                <div class="bar-label"><span>Different Singer Pair Similarity (&mu; = {diff_mu:.3f})</span><span>{diff_mu*100:.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill-diff"></div></div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path
