"""Domain port interfaces (Hexagonal Architecture contracts)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from music_dna_agent.src.domain.models import Job, MusicalDNA, StepName


class StoragePort(ABC):
    """Abstract persistence interface for jobs and DNA analysis."""

    @abstractmethod
    async def save_job(self, job: Job) -> None:
        """Create or update a job state."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a job by unique ID."""
        pass

    @abstractmethod
    async def list_jobs_by_user(self, user_id: int, limit: int = 10) -> List[Job]:
        """List historical jobs for a user."""
        pass

    @abstractmethod
    async def save_dna(self, job_id: str, dna: MusicalDNA) -> None:
        """Save extracted musical DNA."""
        pass

    @abstractmethod
    async def get_cached_dna_by_hash(self, file_hash: str) -> Optional[MusicalDNA]:
        """Retrieve cached DNA by audio content SHA256 hash."""
        pass


class QueuePort(ABC):
    """Abstract job queue and event broker interface."""

    @abstractmethod
    async def enqueue_job(self, job_id: str) -> None:
        """Push a job ID into the FIFO processing queue."""
        pass

    @abstractmethod
    async def publish_progress(self, job_id: str, step: StepName, pct: int, message: str) -> None:
        """Publish a real-time progress update event."""
        pass

    @abstractmethod
    async def subscribe_progress(self, job_id: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Subscribe to live progress events for a given job."""
        pass


class PipelineStepPort(ABC):
    """Abstract contract for an individual pipeline step."""

    @property
    @abstractmethod
    def name(self) -> StepName:
        """Return the step identifier."""
        pass

    @abstractmethod
    async def execute(self, job: Job, workspace_dir: str) -> str:
        """Execute the step and return primary artifact output path."""
        pass
