"""Publish the latest vocal identity without discarding raw analysis fields."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

from music_dna_agent.src.domain.models import Job


def write_vocal_analysis(
    job: Job, analysis_dir: str, *, raw_identity: Optional[Dict[str, Any]] = None,
    raw_acoustics: Optional[Dict[str, Any]] = None,
) -> str:
    if not job.dna or not job.dna.vocal_dna:
        raise ValueError("Vocal analysis is unavailable.")
    path = Path(analysis_dir) / "vocal_dna.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(job.dna.vocal_dna.to_dict())
    if raw_identity is not None:
        payload["raw_identity"] = raw_identity
    if raw_acoustics is not None:
        payload["raw_acoustics"] = raw_acoustics
    # Validate before touching the published artifact.
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".json", delete=False) as handle:
        temporary_path = handle.name
        handle.write(encoded)
    try:
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return str(path)
