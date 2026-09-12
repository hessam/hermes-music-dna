"""Guardrail test: Ensure no banned subjective/era/mic hallucination strings exist in pipeline analyzer code."""
import os
import re

BANNED_PATTERNS = [
    r"Neumann\s+U47",
    r"Shure\s+SM7B",
    r"1980s\s+Montreal",
    r"1970s\s+Outlaw",
    r"1960s\s+British",
    r"High\s+Rock\s+Tenor",
    r"Sub-Bass\s+/\s+Deep\s+Low\s+Baritone",
    r"Lyrical\s+Baritone",
]


def test_no_banned_heuristic_strings_in_analyzers():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analyzers = [
        os.path.join(base_dir, "src", "adapters", "pipeline", "vocal_dna_analyzer.py"),
        os.path.join(base_dir, "src", "adapters", "pipeline", "groove_extractor.py"),
        os.path.join(base_dir, "src", "adapters", "pipeline", "source_qc.py"),
        os.path.join(base_dir, "src", "adapters", "pipeline", "voice_embedder.py"),
        os.path.join(base_dir, "src", "adapters", "pipeline", "structure_segmenter.py"),
    ]

    for file_path in analyzers:
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            for pattern in BANNED_PATTERNS:
                assert not re.search(pattern, content, re.IGNORECASE), (
                    f"Banned subjective hallucination pattern {pattern} found in {file_path}"
                )
