"""Step 2: Multi-Stem Neural Pitch & Riff Transcription using Basic Pitch."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List
from basic_pitch.inference import ICASSP_2022_MODEL_PATH, predict_and_save
import pretty_midi
import soundfile as sf
import numpy as np
from music_dna_agent.src.domain.models import Job, NoteEvent, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort


class BasicPitchExtractor(PipelineStepPort):
    """Transcribes Vocals, Instrumental (Other), and Bass stems into exact polyphonic MIDIs."""

    def __init__(self, onset_threshold: float = 0.45, frame_threshold: float = 0.25):
        self._onset_threshold = onset_threshold
        self._frame_threshold = frame_threshold

    @property
    def name(self) -> StepName:
        return StepName.PITCH_EXTRACTION

    def _has_audio_content(self, wav_path: str, threshold: float = 0.005) -> bool:
        if not os.path.exists(wav_path):
            return False
        try:
            data, _ = sf.read(wav_path, frames=44100 * 10)
            return float(np.max(np.abs(data))) > threshold
        except Exception:
            return False

    async def execute(self, job: Job, workspace_dir: str) -> str:
        stems_dir = os.path.join(workspace_dir, "stems")
        midi_dir = os.path.join(workspace_dir, "midi")
        os.makedirs(midi_dir, exist_ok=True)

        vocals_wav = os.path.join(stems_dir, "vocals.wav")
        other_wav = os.path.join(stems_dir, "other.wav")
        bass_wav = os.path.join(stems_dir, "bass.wav")

        targets = []
        if self._has_audio_content(vocals_wav):
            targets.append(("vocals", vocals_wav))
        if self._has_audio_content(other_wav):
            targets.append(("other", other_wav))
        if self._has_audio_content(bass_wav):
            targets.append(("bass", bass_wav))

        if not targets:
            targets.append(("other", job.input_file_path))

        all_notes: List[NoteEvent] = []

        for stem_type, wav_path in targets:
            temp_out_dir = os.path.join(midi_dir, f"tmp_{stem_type}")
            os.makedirs(temp_out_dir, exist_ok=True)

            # Transcribe stem
            predict_and_save(
                audio_path_list=[wav_path],
                output_directory=temp_out_dir,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=True,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=self._onset_threshold,
                frame_threshold=self._frame_threshold,
                minimum_note_length=50,  # ms
                minimum_frequency=30.0 if stem_type == "bass" else 65.0,
                maximum_frequency=500.0 if stem_type == "bass" else 2500.0,
            )

            # Move and rename output MIDI
            base_name = Path(wav_path).stem
            generated_mid = os.path.join(temp_out_dir, f"{base_name}_basic_pitch.mid")
            target_mid = os.path.join(midi_dir, f"{stem_type}.mid")

            if os.path.exists(generated_mid):
                os.replace(generated_mid, target_mid)
                pm = pretty_midi.PrettyMIDI(target_mid)
                for inst in pm.instruments:
                    for note in inst.notes:
                        all_notes.append(
                            NoteEvent(
                                pitch=note.pitch,
                                start_time=note.start,
                                end_time=note.end,
                                velocity=note.velocity,
                                pitch_name=pretty_midi.note_number_to_name(note.pitch),
                            )
                        )

        if job.dna:
            job.dna.melody_notes = all_notes

        return midi_dir
