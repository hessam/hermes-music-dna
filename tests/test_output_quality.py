"""Focused artifact contract tests; external inference is replaced at its boundary."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch
import tempfile
import types
from contextlib import nullcontext
from unittest.mock import Mock

# The checkout name differs from its deployed package name.
ROOT = Path(__file__).resolve().parents[1]
if "music_dna_agent" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "music_dna_agent", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["music_dna_agent"] = package
    spec.loader.exec_module(package)

from music_dna_agent.src.vocal_designer.input_interpreter import (
    HardConstraints, InputInterpreter, InterpretationResult, SectionConstraint,
)
from music_dna_agent.src.vocal_designer.suno_prompt_builder import SunoPromptBuilder
from music_dna_agent.src.vocal_designer.vocal_bible import BibleRegistry


def interpretation():
    return InterpretationResult(
        True, [SectionConstraint("Verse", lyrics_lines=["I WILL KEEP THESE WORDS"])],
        HardConstraints("minimal techno", "124 BPM", ["drum machine", "synth bass"],
                        "dry digital production"),
        "powerful-dreamer", "storyteller", "dreamer", 5, 3, 8, 5, "test",
    )


def load_adapter(name, dependencies):
    spec = importlib.util.spec_from_file_location(
        f"quality_{name}", ROOT / "src" / "adapters" / "pipeline" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class BlueprintQualityTests(unittest.IsolatedAsyncioTestCase):
    def test_preserves_song_world_lyrics_and_steering(self):
        data = interpretation()
        data.hard_constraints.diction_descriptor = "precise diction"
        data.hard_constraints.rhythmic_descriptor = "syncopated phrasing"
        result = SunoPromptBuilder.build(BibleRegistry.get(), data)
        style = result["style_prompt"]
        for phrase in ("minimal techno", "dry digital production", "precise diction", "syncopated phrasing"):
            self.assertIn(phrase, style)
        self.assertNotIn("Broadway", style)
        self.assertIn("I WILL KEEP THESE WORDS", result["lyrics_box"])
        self.assertNotIn("electronic beat", result["negative_prompt"])

    async def test_repeated_sections_keep_each_original_lyric(self):
        interpreter = InputInterpreter(api_key="test")
        data = interpretation()
        data.sections = [SectionConstraint("Chorus"), SectionConstraint("Chorus")]
        interpreter._call_luna = AsyncMock(return_value=data)
        result = await interpreter.interpret("[Chorus]\nfirst lyric\n[Chorus]\nsecond lyric")
        self.assertEqual([s.lyrics_lines for s in result.sections], [["first lyric"], ["second lyric"]])

    async def test_model_cannot_drop_or_reorder_user_sections(self):
        interpreter = InputInterpreter(api_key="test")
        data = interpretation()
        data.sections = [SectionConstraint("Chorus", vocal_token="bold")]
        interpreter._call_luna = AsyncMock(return_value=data)
        result = await interpreter.interpret("[Verse]\nhello\n[Chorus]\nworld\n[Outro]\ngoodbye")
        self.assertEqual([s.section_name for s in result.sections], ["Verse", "Chorus", "Outro"])
        self.assertEqual([s.lyrics_lines for s in result.sections], [["hello"], ["world"], ["goodbye"]])

    def test_overlong_lyrics_fail_without_truncation(self):
        data = interpretation()
        data.sections[0].lyrics_lines = ["x" * 4001]
        with self.assertRaisesRegex(ValueError, "4000"):
            SunoPromptBuilder.build(BibleRegistry.get(), data)
        self.assertEqual(len(data.sections[0].lyrics_lines[0]), 4001)

    def test_failed_gate_is_never_rendered_as_passed(self):
        from music_dna_agent.src.vocal_designer.vocal_identity_gate import IdentityGateResult
        result = SunoPromptBuilder.build(BibleRegistry.get(), interpretation())
        result["gate_result"] = IdentityGateResult(False, 80, [], True)
        report = SunoPromptBuilder.format_html(result, "test")
        self.assertNotIn("PASSED", report)


class SeparationQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_trim_cannot_report_success(self):
        from music_dna_agent.src.adapters.pipeline.stem_isolator import DemucsStemIsolator
        from music_dna_agent.src.domain.models import Job
        process = types.SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b"", b"bad input")))
        with tempfile.TemporaryDirectory() as output:
            source = Path(output) / "input.wav"
            source.write_bytes(b"input")
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
                with self.assertRaisesRegex(RuntimeError, "trim"):
                    await DemucsStemIsolator().execute(Job(input_file_path=str(source)), output)

    async def test_missing_isolated_vocals_never_uses_mix(self):
        VocalSegmentRanker = load_adapter("vocal_segment_ranker", {
            "librosa": types.ModuleType("librosa"), "soundfile": types.ModuleType("soundfile"),
        }).VocalSegmentRanker
        from music_dna_agent.src.domain.models import Job
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(FileNotFoundError, "vocal"):
                await VocalSegmentRanker().execute(Job(input_file_path="mixed.wav"), output)


class AnalysisExportTests(unittest.TestCase):
    def test_audio_prompt_does_not_invent_genre_from_tempo(self):
        from music_dna_agent.src.domain.models import MusicalDNA
        from music_dna_agent.src.adapters.generators.suno_adapter import SunoAdapter
        dna = MusicalDNA(140, 0.9, 0.9, None, 0.0, "C", "major", [], [], 1.0)
        prompt = SunoAdapter.generate_prompts(dna)["style_prompt"]
        self.assertIn("140 BPM", prompt)
        self.assertNotIn("rock", prompt)
        self.assertNotIn("analog", prompt)

    def test_final_identity_replaces_placeholder_and_preserves_raw_pitch(self):
        import json
        from music_dna_agent.src.adapters.pipeline.analysis_export import write_vocal_analysis
        vocal = types.SimpleNamespace(to_dict=lambda: {
            "identity": {"speaker_identity_embedding": [0.123456789], "embedding_model": "actual-model"},
            "measurements": {"cpp_db": 12.0},
        })
        job = types.SimpleNamespace(dna=types.SimpleNamespace(vocal_dna=vocal))
        with tempfile.TemporaryDirectory() as output:
            path = Path(output) / "vocal_dna.json"
            path.write_text(json.dumps({"identity": {"embedding_model": "pending"},
                                       "raw_acoustics": {"f0_hz": [120.1, None]}}))
            write_vocal_analysis(job, output, raw_identity={"segment_embeddings": [[0.123456789]]})
            result = json.loads(path.read_text())
        self.assertEqual(result["identity"]["embedding_model"], "actual-model")
        self.assertEqual(result["raw_acoustics"]["f0_hz"], [120.1, None])
        self.assertEqual(result["raw_identity"]["segment_embeddings"], [[0.123456789]])


class EmbeddingQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_learned_encoder_is_labelled_as_spectral_fallback(self):
        import json
        import numpy as np
        audio = types.SimpleNamespace(
            load=lambda *a, **k: (np.ones(48000, dtype=np.float32), 16000),
            feature=types.SimpleNamespace(
                melspectrogram=lambda **k: np.ones((192, 10)),
                mfcc=lambda **k: np.ones((13, 10)),
                delta=lambda values, **k: np.zeros_like(values),
            ),
        )
        torch = types.SimpleNamespace(tensor=Mock(), inference_mode=nullcontext)
        encoder = types.SimpleNamespace(encode_batch=Mock(side_effect=RuntimeError("inference failed")))
        speaker = types.ModuleType("speechbrain.inference.speaker")
        speaker.EncoderClassifier = types.SimpleNamespace(from_hparams=lambda **k: encoder)
        dependencies = {"librosa": audio, "torch": torch,
                        "soundfile": types.ModuleType("soundfile"),
                        "speechbrain.inference.speaker": speaker}
        module = load_adapter("voice_embedder", dependencies)
        vocal = types.SimpleNamespace(identity=None)
        vocal.to_dict = lambda: {"identity": vocal.identity.to_dict()}
        job = types.SimpleNamespace(input_file_path="reference.wav", dna=types.SimpleNamespace(vocal_dna=vocal))
        with tempfile.TemporaryDirectory() as output, patch.dict(sys.modules, dependencies):
            await module.VoiceEmbedder().execute(job, output)
            exported = json.loads((Path(output) / "analysis" / "vocal_dna.json").read_text())
        encoder.encode_batch.assert_called_once()
        self.assertEqual(exported["identity"]["embedding_model"], "spectral-mel-192d")
        self.assertEqual(exported["identity"]["signal_reliability"], 0.0)
        self.assertFalse(exported["raw_identity"]["learned_speaker_embedding"])
        self.assertEqual(len(exported["raw_identity"]["segment_embeddings"][0]), 192)


class MelodyQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_playable_midi_produces_lead_without_invented_drums(self):
        pretty = types.ModuleType("pretty_midi")
        note = types.SimpleNamespace(pitch=60, start=0.25, end=0.75, velocity=80)
        pretty.PrettyMIDI = lambda *args, **kwargs: types.SimpleNamespace(
            instruments=[types.SimpleNamespace(is_drum=False, notes=[note])])
        FluidSynthRenderer = load_adapter("synth_renderer", {
            "pretty_midi": pretty, "soundfile": types.ModuleType("soundfile"),
        }).FluidSynthRenderer
        from music_dna_agent.src.domain.models import Job
        with tempfile.TemporaryDirectory() as output:
            midi = Path(output) / "midi"
            midi.mkdir()
            (midi / "vocals.mid").write_bytes(b"small valid fixture")
            renderer = FluidSynthRenderer()
            with patch.object(renderer, "_format_instrument") as formatter, \
                 patch.object(renderer, "_render_fluidsynth", new_callable=AsyncMock) as render, \
                 patch.object(renderer, "_build_clean_drum_grid") as drums:
                await renderer.execute(Job(), output)
            self.assertEqual(render.await_count, 1)
            self.assertEqual(Path(render.await_args.args[1]).name, "lead.wav")
            self.assertEqual(formatter.call_args.kwargs["program"], 0)
            drums.assert_not_called()


if __name__ == "__main__":
    unittest.main()
