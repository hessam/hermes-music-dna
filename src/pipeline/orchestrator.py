"""Pipeline Orchestrator: Complete Multi-Step Architecture for Music DNA & Vocal Cloning."""
from __future__ import annotations

import logging
import os
import traceback
from typing import Dict, List
from music_dna_agent.src.domain.models import Job, JobStatus, QualityOption, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort, QueuePort, StoragePort
from music_dna_agent.src.adapters.infra.filesystem import FilesystemManager
from music_dna_agent.src.adapters.pipeline.source_qc import SourceQCGate
from music_dna_agent.src.adapters.pipeline.stem_isolator import DemucsStemIsolator
from music_dna_agent.src.adapters.pipeline.structure_segmenter import StructureSegmenter
from music_dna_agent.src.adapters.pipeline.vocal_segment_ranker import VocalSegmentRanker
from music_dna_agent.src.adapters.pipeline.essentia_analyzer import EssentiaAnalyzer
from music_dna_agent.src.adapters.pipeline.pitch_extractor import BasicPitchExtractor
from music_dna_agent.src.adapters.pipeline.groove_extractor import LibrosaGrooveExtractor
from music_dna_agent.src.adapters.pipeline.vocal_dna_analyzer import VocalDNAAnalyzer
from music_dna_agent.src.adapters.pipeline.voice_embedder import VoiceEmbedder
from music_dna_agent.src.adapters.pipeline.synth_renderer import FluidSynthRenderer
from music_dna_agent.src.adapters.pipeline.master_bouncer import MasterBouncer

logger = logging.getLogger(__name__)

STEP_TO_JOB_STATUS: Dict[StepName, JobStatus] = {
    StepName.SOURCE_QC: JobStatus.CHECKING_SOURCE_QC,
    StepName.STEM_ISOLATION: JobStatus.ISOLATING_STEMS,
    StepName.STRUCTURE_SEGMENTATION: JobStatus.SEGMENTING_STRUCTURE,
    StepName.VOCAL_SEGMENT_RANKING: JobStatus.RANKING_VOCAL_SEGMENTS,
    StepName.ESSENTIA_ANALYSIS: JobStatus.ANALYZING_ESSENTIA,
    StepName.PITCH_EXTRACTION: JobStatus.EXTRACTING_PITCH,
    StepName.GROOVE_EXTRACTION: JobStatus.EXTRACTING_GROOVE,
    StepName.VOCAL_DNA_ANALYSIS: JobStatus.ANALYZING_VOCAL_DNA,
    StepName.VOICE_EMBEDDING: JobStatus.EXTRACTING_VOICE_EMBEDDINGS,
    StepName.SYNTH_SYNTHESIS: JobStatus.RENDERING_SYNTH,
    StepName.MASTER_BOUNCE: JobStatus.MASTERING,
}


class PipelineOrchestrator:
    """Executes the complete Music DNA analysis, Vocal DNA cloning, and guide track synthesis."""

    def __init__(
        self,
        storage: StoragePort,
        queue: QueuePort,
        fs_manager: FilesystemManager,
    ):
        self._storage = storage
        self._queue = queue
        self._fs_manager = fs_manager

    def _get_pipeline_steps(self, quality: QualityOption) -> List[PipelineStepPort]:
        if quality == QualityOption.VOCAL_DNA:
            return [
                SourceQCGate(),
                DemucsStemIsolator(),
                VocalSegmentRanker(),
                VocalDNAAnalyzer(),
                VoiceEmbedder(),
                MasterBouncer(),
            ]
        # Full Guide Track & Complete Composition DNA Pipeline
        return [
            SourceQCGate(),
            DemucsStemIsolator(),
            StructureSegmenter(),
            VocalSegmentRanker(),
            EssentiaAnalyzer(),
            BasicPitchExtractor(),
            LibrosaGrooveExtractor(),
            VocalDNAAnalyzer(),
            VoiceEmbedder(),
            FluidSynthRenderer(),
            MasterBouncer(),
        ]

    async def run(self, job: Job) -> Job:
        workspace_dir = self._fs_manager.get_job_dir(job.id)
        logger.info("Starting pipeline for job %s (Mode: %s) in %s", job.id, job.quality.value, workspace_dir)

        steps = self._get_pipeline_steps(job.quality)

        step_weights = {
            StepName.SOURCE_QC: (5, 10, "Validating source audio & dynamic headroom..."),
            StepName.STEM_ISOLATION: (10, 35, "Isolating vocal & instrumental stems (Demucs)..."),
            StepName.STRUCTURE_SEGMENTATION: (35, 45, "Segmenting verse/chorus structure..."),
            StepName.VOCAL_SEGMENT_RANKING: (45, 55, "Scoring & ranking cleanest vocal segments..."),
            StepName.ESSENTIA_ANALYSIS: (55, 65, "Extracting tonality & Essentia descriptors..."),
            StepName.PITCH_EXTRACTION: (65, 75, "Transcribing multi-stem pitch & riffs (Basic Pitch)..."),
            StepName.GROOVE_EXTRACTION: (75, 80, "Analyzing microtiming & groove offset..."),
            StepName.VOCAL_DNA_ANALYSIS: (80, 88, "Measuring physical vocal acoustics (CPP, Jitter)..."),
            StepName.VOICE_EMBEDDING: (88, 92, "Generating learned speaker vector (ECAPA-TDNN)..."),
            StepName.SYNTH_SYNTHESIS: (92, 97, "Synthesizing clean virtual instruments (FluidR3)..."),
            StepName.MASTER_BOUNCE: (97, 100, "Mastering & compiling multi-asset bundle..."),
        }

        try:
            for step in steps:
                step_name = step.name
                start_pct, end_pct, message = step_weights.get(step_name, (0, 100, f"Running {step_name.value}..."))

                # 1. Update status
                job.mark_step_start(step_name)
                job.status = STEP_TO_JOB_STATUS.get(step_name, JobStatus.ISOLATING_STEMS)
                await self._storage.save_job(job)
                await self._queue.publish_progress(job.id, step_name, start_pct, message)

                # 2. Execute Step
                output_path = await step.execute(job, workspace_dir)

                # 3. Mark step done
                job.mark_step_done(step_name, output_path=output_path)
                await self._storage.save_job(job)
                await self._queue.publish_progress(job.id, step_name, end_pct, f"Completed {step_name.value}")

            # All steps passed
            job.status = JobStatus.COMPLETED
            await self._storage.save_job(job)
            await self._queue.publish_progress(job.id, steps[-1].name, 100, "Analysis complete!")

            # Cleanup heavy raw stems if full bounce was performed
            if job.quality != QualityOption.VOCAL_DNA:
                self._fs_manager.cleanup_intermediate_stems(job.id)

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error("Pipeline failure for job %s: %s", job.id, error_trace)
            failed_step = step_name if step_name in locals() else StepName.SOURCE_QC
            job.mark_step_failed(failed_step, str(e))
            await self._storage.save_job(job)
            await self._queue.publish_progress(job.id, failed_step, 0, f"Failed: {str(e)}")
            raise e

        return job
