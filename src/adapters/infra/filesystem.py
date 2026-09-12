"""Filesystem Workspace Manager for jobs and audio assets."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional


class FilesystemManager:
    """Manages workspace folder layout and file hashing for audio jobs."""

    def __init__(self, root_dir: str = "/workspace/jobs"):
        self._root_dir = root_dir
        os.makedirs(self._root_dir, exist_ok=True)

    def get_job_dir(self, job_id: str) -> str:
        job_dir = os.path.join(self._root_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def calculate_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def cleanup_intermediate_stems(self, job_id: str) -> None:
        """Purge heavy raw stem files after master bounce to preserve disk space."""
        stems_dir = os.path.join(self._root_dir, job_id, "stems")
        if os.path.exists(stems_dir):
            shutil.rmtree(stems_dir, ignore_errors=True)
