"""SQLite persistent storage adapter with thread-safe async transactions."""
from __future__ import annotations

import json
import os
from typing import List, Optional
import aiosqlite
from music_dna_agent.src.domain.models import (
    Job,
    JobStatus,
    JobStep,
    MusicalDNA,
    MusicalSection,
    QualityOption,
    StepName,
    StepStatus,
    VocalDNA,
    VocalIdentity,
    VocalMeasurements,
    SectionVocalProfile,
)
from music_dna_agent.src.domain.ports import StoragePort


def _deserialize_dna(dna_json_str: Optional[str]) -> Optional[MusicalDNA]:
    if not dna_json_str:
        return None
    try:
        data = json.loads(dna_json_str)
        tempo = data.get("tempo", {})
        harmony = data.get("harmony", {})
        mix = data.get("mix", {})
        arrangement = data.get("arrangement", {})

        sections = []
        for s in arrangement.get("sections", []):
            sections.append(
                MusicalSection(
                    label=s.get("label", "Section"),
                    start_s=s.get("start_s", 0.0),
                    end_s=s.get("end_s", 0.0),
                    bpm=s.get("bpm", 120.0),
                    key=s.get("key", "C"),
                    scale=s.get("scale", "major"),
                    chord_progression=s.get("chord_progression", []),
                    energy_intensity=s.get("energy_intensity", 0.0),
                )
            )

        vocal_dna = None
        v_data = data.get("vocal_dna")
        if v_data and "measurements" in v_data:
            m_data = v_data["measurements"]
            von = m_data.get("vibrato_onset_ms", {})
            measurements = VocalMeasurements(
                f0_distribution=m_data.get("f0_distribution", {}),
                f0_range_semitones=m_data.get("f0_range_semitones", 12.0),
                vibrato_rate_hz=m_data.get("vibrato_rate_hz"),
                vibrato_depth_semitones=m_data.get("vibrato_depth_semitones"),
                vibrato_onset_median_ms=von.get("median") if isinstance(von, dict) else m_data.get("vibrato_onset_median_ms"),
                vibrato_onset_p25_ms=von.get("p25") if isinstance(von, dict) else m_data.get("vibrato_onset_p25_ms"),
                vibrato_onset_p75_ms=von.get("p75") if isinstance(von, dict) else m_data.get("vibrato_onset_p75_ms"),
                cpp_db=m_data.get("cpp_db", 12.0),
                h1_h2_db=m_data.get("h1_h2_db", 0.0),
                jitter_local_pct=m_data.get("jitter_local_pct", 1.0),
                shimmer_local_pct=m_data.get("shimmer_local_pct", 5.0),
                subharmonic_energy_ratio=m_data.get("subharmonic_energy_ratio", 0.0),
                detected_registers=m_data.get("detected_registers", []),
            )

            id_data = v_data.get("identity", {})
            identity = VocalIdentity(
                speaker_identity_embedding=id_data.get("speaker_identity_embedding", []),
                segment_count=id_data.get("segment_count", 0),
                mean_pairwise_cosine=id_data.get("mean_pairwise_cosine", 0.0),
                std_pairwise_cosine=id_data.get("std_pairwise_cosine", 0.0),
                mfcc_trajectory_stats=id_data.get("mfcc_trajectory_stats", {}),
                embedding_model=id_data.get("embedding_model", ""),
                signal_reliability=id_data.get("signal_reliability", 0.0),
            )

            sec_profiles = []
            for sp in v_data.get("section_profiles", []):
                sec_profiles.append(
                    SectionVocalProfile(
                        label=sp.get("label", ""),
                        start_s=sp.get("start_s", 0.0),
                        end_s=sp.get("end_s", 0.0),
                        f0_median_hz=sp.get("f0_median_hz", 150.0),
                        intensity_mean_lufs=sp.get("intensity_mean_lufs", -14.0),
                        dynamic_range_db=sp.get("dynamic_range_db", 0.0),
                        dominant_register=sp.get("dominant_register", "chest"),
                    )
                )

            vocal_dna = VocalDNA(
                measurements=measurements,
                identity=identity,
                section_profiles=sec_profiles,
                reference_portfolio=v_data.get("reference_portfolio", {}),
                clean_reference_path=v_data.get("clean_reference_path"),
            )

        return MusicalDNA(
            bpm=tempo.get("bpm", data.get("bpm", 120.0)),
            bpm_confidence=tempo.get("bpm_confidence", 0.8),
            tempo_stability=tempo.get("tempo_stability", 0.9),
            swing_ratio=tempo.get("swing_ratio"),
            groove_offset_ms=tempo.get("groove_offset_ms", data.get("groove_offset_ms", 0.0)),
            key=harmony.get("key", data.get("key", "C")),
            scale=harmony.get("scale", data.get("scale", "major")),
            chord_progression=harmony.get("chord_progression", data.get("chord_progression", [])),
            chord_extensions=harmony.get("chord_extensions", []),
            harmonic_rhythm=harmony.get("harmonic_rhythm", 1.0),
            sections=sections,
            dominant_instruments=arrangement.get("dominant_instruments", []),
            arrangement_density=arrangement.get("arrangement_density", 0.0),
            lufs_integrated=mix.get("lufs_integrated", -14.0),
            lufs_range=mix.get("lufs_range", 6.0),
            stereo_width=mix.get("stereo_width", 0.6),
            bass_to_mid_ratio=mix.get("bass_to_mid_ratio", 1.0),
            spectral_balance=mix.get("spectral_balance", {}),
            vocal_dna=vocal_dna,
        )
    except Exception as e:
        return None


class SQLiteStorage(StoragePort):
    """Stores jobs, step checkpoints, and analyzed DNA in a local SQLite database."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    async def init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    original_filename TEXT,
                    input_file_path TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    custom_style_prompt TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    guide_track_path TEXT,
                    error_message TEXT,
                    dna_json TEXT
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS job_steps (
                    job_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    output_path TEXT,
                    error TEXT,
                    PRIMARY KEY (job_id, step_name),
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS dna_cache (
                    file_hash TEXT PRIMARY KEY,
                    dna_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.commit()

    async def save_job(self, job: Job) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            dna_json = json.dumps(job.dna.to_dict()) if job.dna else None
            await db.execute(
                """
                INSERT INTO jobs (id, user_id, chat_id, original_filename, input_file_path, quality, custom_style_prompt, status, created_at, updated_at, guide_track_path, error_message, dna_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    guide_track_path=excluded.guide_track_path,
                    error_message=excluded.error_message,
                    dna_json=excluded.dna_json
                """,
                (
                    job.id,
                    job.user_id,
                    job.chat_id,
                    job.original_filename,
                    job.input_file_path,
                    job.quality.value,
                    job.custom_style_prompt,
                    job.status.value,
                    job.created_at,
                    job.updated_at,
                    job.guide_track_path,
                    job.error_message,
                    dna_json,
                ),
            )
            for step_name, step in job.steps.items():
                await db.execute(
                    """
                    INSERT INTO job_steps (job_id, step_name, status, started_at, finished_at, output_path, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id, step_name) DO UPDATE SET
                        status=excluded.status,
                        started_at=excluded.started_at,
                        finished_at=excluded.finished_at,
                        output_path=excluded.output_path,
                        error=excluded.error
                    """,
                    (
                        job.id,
                        step_name.value,
                        step.status.value,
                        step.started_at,
                        step.finished_at,
                        step.output_path,
                        step.error,
                    ),
                )
            await db.commit()

    async def get_job(self, job_id: str) -> Optional[Job]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                dna = _deserialize_dna(row["dna_json"])

                job = Job(
                    id=row["id"],
                    user_id=row["user_id"],
                    chat_id=row["chat_id"],
                    original_filename=row["original_filename"],
                    input_file_path=row["input_file_path"],
                    quality=QualityOption(row["quality"]),
                    custom_style_prompt=row["custom_style_prompt"],
                    status=JobStatus(row["status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    guide_track_path=row["guide_track_path"],
                    error_message=row["error_message"],
                    dna=dna,
                )

                async with db.execute("SELECT * FROM job_steps WHERE job_id = ?", (job_id,)) as s_cursor:
                    async for s_row in s_cursor:
                        try:
                            s_name = StepName(s_row["step_name"])
                            job.steps[s_name] = JobStep(
                                name=s_name,
                                status=StepStatus(s_row["status"]),
                                started_at=s_row["started_at"],
                                finished_at=s_row["finished_at"],
                                output_path=s_row["output_path"],
                                error=s_row["error"],
                            )
                        except ValueError:
                            pass
                return job

    async def list_jobs_by_user(self, user_id: int, limit: int = 10) -> List[Job]:
        jobs = []
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ) as cursor:
                async for row in cursor:
                    job = await self.get_job(row["id"])
                    if job:
                        jobs.append(job)
        return jobs

    async def save_dna(self, job_id: str, dna: MusicalDNA) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE jobs SET dna_json = ? WHERE id = ?",
                (json.dumps(dna.to_dict()), job_id),
            )
            await db.commit()

    async def get_cached_dna_by_hash(self, file_hash: str) -> Optional[MusicalDNA]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT dna_json FROM dna_cache WHERE file_hash = ?", (file_hash,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return _deserialize_dna(row["dna_json"])
        return None
