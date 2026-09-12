"""Harmonic analysis & Chord Progression extraction using Librosa."""
from __future__ import annotations

import os
from typing import List, Tuple
import numpy as np
from music_dna_agent.src.domain.models import Job


CHORD_TEMPLATES = {
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Cm": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "C#": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "C#m": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "D": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "Dm": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "D#": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "D#m": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "E": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    "Em": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "F": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "Fm": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "F#": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "F#m": [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "Gm": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "G#": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "G#m": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    "A": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "Am": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "A#": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "A#m": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "B": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    "Bm": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ChordAnalyzer:
    """Analyzes harmonic chroma vectors and recognizes chord progressions & key."""

    @staticmethod
    def analyze(audio_path: str) -> Tuple[str, str, List[str]]:
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, duration=60.0)
        # Extract harmonic component
        y_harmonic, _ = librosa.effects.hpss(y)

        # Chroma CQT for precise pitch resolution
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=1024)

        # Compute key and scale
        chroma_sum = np.sum(chroma, axis=1)
        root_idx = int(np.argmax(chroma_sum))
        key = PITCH_CLASSES[root_idx]

        # Simple major vs minor estimation based on 3rd interval
        minor_third_idx = (root_idx + 3) % 12
        major_third_idx = (root_idx + 4) % 12
        scale = "Minor" if chroma_sum[minor_third_idx] > chroma_sum[major_third_idx] else "Major"

        # Frame-by-frame chord matching (segment by bar/beat)
        frame_chords: List[str] = []
        hop_step = max(1, chroma.shape[1] // 16)  # 16 time buckets

        for col in range(0, chroma.shape[1], hop_step):
            vec = chroma[:, col : col + hop_step].mean(axis=1)
            vec = vec / (np.linalg.norm(vec) + 1e-6)

            best_chord = "C"
            best_sim = -1.0
            for chord_name, template in CHORD_TEMPLATES.items():
                t_vec = np.array(template, dtype=float)
                t_vec = t_vec / np.linalg.norm(t_vec)
                sim = float(np.dot(vec, t_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_chord = chord_name

            if not frame_chords or frame_chords[-1] != best_chord:
                frame_chords.append(best_chord)

        return key, scale, frame_chords[:8]
