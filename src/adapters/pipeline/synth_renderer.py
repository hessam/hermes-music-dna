"""Step 4: 100% Pure Virtual Instrument Synthesis via FluidR3 SoundFont."""
from __future__ import annotations

import asyncio
import os
import shutil
from typing import Optional
import pretty_midi
import soundfile as sf
from music_dna_agent.src.domain.models import Job, StepName
from music_dna_agent.src.domain.ports import PipelineStepPort

SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


class FluidSynthRenderer(PipelineStepPort):
    """Synthesizes extracted musical DNA into 100% clean virtual instruments:

    1. Lead Line: Crisp Lead Synth (Program 81)
    2. Chords/Atmosphere: Smooth Warm Pad / Grand Piano (Program 89 / 0)
    3. Bass & Groove: Tight Synth Bass (Program 38)
    4. Percussion Grid: Clean Click / Hi-Hat & Kick (Channel 9 MIDI Drums)
    """

    @property
    def name(self) -> StepName:
        return StepName.SYNTH_SYNTHESIS

    async def execute(self, job: Job, workspace_dir: str) -> str:
        synth_dir = os.path.join(workspace_dir, "synth")
        midi_dir = os.path.join(workspace_dir, "midi")
        os.makedirs(synth_dir, exist_ok=True)

        vocals_mid = os.path.join(midi_dir, "vocals.mid")
        other_mid = os.path.join(midi_dir, "other.mid")
        bass_mid = os.path.join(midi_dir, "bass.mid")

        tasks = []

        # 1. Lead Line (Vocals MIDI or Highest Melody line from Other)
        lead_mid_path = os.path.join(synth_dir, "lead.mid")
        lead_wav_path = os.path.join(synth_dir, "lead.wav")
        has_vocal_melody = self._has_notes(vocals_mid)
        if has_vocal_melody:
            self._format_instrument(vocals_mid, lead_mid_path, program=0)
            tasks.append(self._render_fluidsynth(lead_mid_path, lead_wav_path))
        elif self._has_notes(other_mid):
            self._format_instrument(other_mid, lead_mid_path, program=0)
            tasks.append(self._render_fluidsynth(lead_mid_path, lead_wav_path))
        else:
            raise RuntimeError("No playable melodic MIDI notes were extracted.")

        # 2. Chords & Atmosphere (Piano / Warm Pad from Other MIDI)
        chords_mid_path = os.path.join(synth_dir, "chords.mid")
        chords_wav_path = os.path.join(synth_dir, "chords.wav")
        if has_vocal_melody and self._has_notes(other_mid):
            self._format_instrument(other_mid, chords_mid_path, program=0)  # Acoustic Grand Piano
            tasks.append(self._render_fluidsynth(chords_mid_path, chords_wav_path))

        # 3. Bassline (Synth Bass from Bass MIDI)
        bass_mid_path = os.path.join(synth_dir, "bass.mid")
        bass_wav_path = os.path.join(synth_dir, "bass.wav")
        if self._has_notes(bass_mid):
            self._format_instrument(bass_mid, bass_mid_path, program=38)  # Synth Bass 1
            tasks.append(self._render_fluidsynth(bass_mid_path, bass_wav_path))

        # A synthetic fixed beat grid does not represent the source's rhythm.

        if tasks:
            await asyncio.gather(*tasks)

        if os.path.isfile(lead_wav_path):
            guide_dir = os.path.join(workspace_dir, "guide")
            os.makedirs(guide_dir, exist_ok=True)
            melody_path = os.path.join(guide_dir, "melody_guide.wav")
            shutil.copy2(lead_wav_path, melody_path)
            job.output_manifest["melody_guide"] = melody_path
            job.output_manifest["melody_midi"] = lead_mid_path

        return synth_dir

    @staticmethod
    def _has_notes(midi_path: str) -> bool:
        if not os.path.isfile(midi_path):
            return False
        pm = pretty_midi.PrettyMIDI(midi_path)
        return any(note.end > note.start and note.velocity > 0
                   for instrument in pm.instruments if not instrument.is_drum
                   for note in instrument.notes)

    def _format_instrument(self, src_midi: str, dst_midi: str, program: int) -> None:
        pm = pretty_midi.PrettyMIDI(src_midi)
        pm.instruments = [inst for inst in pm.instruments if not inst.is_drum]
        for inst in pm.instruments:
            inst.is_drum = False
            inst.program = program
        pm.write(dst_midi)

    def _build_clean_drum_grid(self, bpm: float, duration: float, out_path: str) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        drums = pretty_midi.Instrument(program=0, is_drum=True, name="Reference Click")

        sec_per_beat = 60.0 / bpm
        sec_per_8th = sec_per_beat / 2.0
        cur_t = 0.0
        beat_idx = 0

        while cur_t < duration:
            # Kick on beat 1 and 3 (MIDI 36)
            if beat_idx % 4 == 0 or beat_idx % 4 == 2:
                drums.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=cur_t, end=cur_t + 0.1))
            # Snare/Side-stick on beat 2 and 4 (MIDI 38)
            elif beat_idx % 4 == 1 or beat_idx % 4 == 3:
                drums.notes.append(pretty_midi.Note(velocity=85, pitch=38, start=cur_t, end=cur_t + 0.1))

            # Closed Hi-Hat click on every 8th note (MIDI 42)
            drums.notes.append(pretty_midi.Note(velocity=70, pitch=42, start=cur_t, end=cur_t + 0.05))
            drums.notes.append(pretty_midi.Note(velocity=60, pitch=42, start=cur_t + sec_per_8th, end=cur_t + sec_per_8th + 0.05))

            cur_t += sec_per_beat
            beat_idx += 1

        pm.instruments.append(drums)
        pm.write(out_path)

    async def _render_fluidsynth(self, midi_path: str, wav_path: str) -> None:
        if os.path.exists(SOUNDFONT_PATH):
            cmd = [
                "fluidsynth",
                "-ni",
                "-F", wav_path,
                "-r", "44100",
                "-g", "0.5",
                SOUNDFONT_PATH,
                midi_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                return
            raise RuntimeError("FluidSynth failed to render the melodic guide.")

        # Fallback pure synth
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio_data = pm.synthesize(fs=44100)
        if len(audio_data) == 0:
            raise RuntimeError("MIDI synthesis produced an empty guide.")
        sf.write(wav_path, audio_data, 44100)
