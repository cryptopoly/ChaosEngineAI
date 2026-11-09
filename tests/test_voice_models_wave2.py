"""Wave-2 speech models (D-70, FU-149): Chatterbox-Turbo, VoxCPM2, Chatterbox Multilingual V3, Orpheus 3B.

Catalog rows (pins, companions, per-row licence), readiness with shards and companions, sound tags per model, the
VoxCPM2 leading-paren guard, request mapping for the new fields, the cloning rules on every model that can clone
(permission, watermark, AI tag, audit), the worker's companion redirect and per-family calls with fake models.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy  # noqa: F401 - imported before any sys.modules patch, so it is never loaded twice

from backend_service import voice_clone, voice_clone_safety, voice_env, voice_library, voice_speak, voice_text
from backend_service import voice_worker_models as wm
from backend_service.catalog import voice_models

WAVE2 = ["chatterbox-turbo", "voxcpm2", "chatterbox-mtl", "orpheus-3b"]
CLONERS = [k for k in WAVE2 if voice_models.voice_model(k)["capabilities"]["clone"]]


def make_wav(seconds: float = 0.2, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(seconds * rate))
    return buffer.getvalue()


def every_row_listed():
    """Offer every wave-2 row (a CI box has no measured numbers to gate on)."""
    rows = [dict(row, listed=True) for row in voice_models.VOICE_MODELS]
    return mock.patch.object(voice_models, "listed_models", return_value=rows)


def installed(*keys: str):
    repos = {voice_models.voice_model(k)["repo"] for k in keys}
    return mock.patch.object(voice_clone, "snapshot_dir", side_effect=lambda row: Path("/snap") / row["key"] if row["repo"] in repos else None)


class Machine:
    def __enter__(self):
        self.patches = [
            mock.patch.object(voice_clone, "available_here", return_value=True),
            mock.patch.object(voice_env, "is_installed", return_value=True),
            mock.patch.object(voice_clone, "companion_paths", side_effect=lambda row: {c["repo"]: f"/cache/{c['repo']}" for c in row.get("companions", []) if c.get("role") == "loader"}),
        ]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *_exc):
        for patch in reversed(self.patches):
            patch.stop()


class CatalogTests(unittest.TestCase):
    def test_rows_are_pinned_with_pinned_companions(self) -> None:
        for key in WAVE2:
            row = voice_models.voice_model(key)
            self.assertRegex(row["revision"], r"^[0-9a-f]{40}$", key)
            self.assertTrue(row["repo"].startswith("mlx-community/"))
            self.assertEqual(row["lanes"], ["mlx"])
            for companion in row["companions"]:
                self.assertRegex(companion["revision"], r"^[0-9a-f]{40}$")
                self.assertTrue(companion["files"])
                self.assertIn(companion["role"], ("loader", "defaultVoice"))
            self.assertAlmostEqual(row["downloadGb"], row["sizeGb"] + sum(c["sizeGb"] for c in row["companions"]), places=2)
        loaders = {c["repo"] for k in WAVE2 for c in voice_models.voice_model(k)["companions"] if c["role"] == "loader"}
        self.assertEqual(loaders, set(wm.COMPANION_IDS))

    def test_each_row_shows_its_upstream_licence(self) -> None:
        ids = {k: voice_models.voice_model(k)["licence"]["id"] for k in WAVE2}
        self.assertEqual(ids, {"chatterbox-turbo": "mit", "voxcpm2": "apache-2.0", "chatterbox-mtl": "mit", "orpheus-3b": "apache-2.0"})
        orpheus = voice_models.voice_model("orpheus-3b")["licence"]
        self.assertIn("Llama 3.2", orpheus["note"])
        self.assertIn("meta-llama/Llama-3.2-3B-Instruct", orpheus["baseUrl"])
        for row in voice_models.VOICE_MODELS:
            self.assertIn(row["licence"]["id"], ("mit", "apache-2.0"))  # permissive only
            self.assertTrue(row["licence"]["url"].startswith("https://huggingface.co/"))

    def test_families_rates_and_controls(self) -> None:
        turbo, vox, mtl, orpheus = (voice_models.voice_model(k) for k in WAVE2)
        self.assertEqual((turbo["sampleRate"], vox["sampleRate"], mtl["sampleRate"], orpheus["sampleRate"]), (24_000, 48_000, 24_000, 24_000))
        self.assertEqual(len(turbo["controls"]["tags"]), 9)
        self.assertEqual(len(turbo["controls"]["styleTags"]), 10)
        self.assertEqual(len(orpheus["controls"]["tags"]), 8)
        self.assertEqual(len(mtl["controls"]["languages"]), 23)
        self.assertEqual(mtl["controls"]["expression"]["default"], 0.5)
        self.assertTrue(vox["capabilities"]["design"] and vox["capabilities"]["clone"])
        self.assertFalse(orpheus["capabilities"]["clone"])
        self.assertEqual(turbo["minReferenceS"], 5.0)
        self.assertLess(orpheus["chunkChars"], 400)

    def test_listed_rows_are_measured_and_unmeasured_rows_stay_out(self) -> None:
        for row in voice_models.VOICE_MODELS:
            measured = (row.get("estimates") or {}).get("basis") == "measured"
            self.assertEqual(row["listed"], measured, row["key"])
        listed = {r["key"] for r in voice_models.listed_models()}
        self.assertTrue(set(voice_models.CLONE_MODELS[i]["key"] for i in range(len(voice_models.CLONE_MODELS))) <= listed)

    def test_default_clone_model_is_one_constant(self) -> None:
        self.assertEqual([r["key"] for r in voice_models.VOICE_MODELS if r["default"]], [voice_models.DEFAULT_CLONE_MODEL])
        self.assertEqual(voice_models.clone_model("")["key"], voice_models.DEFAULT_CLONE_MODEL)


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self._patch = mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_cache_dir",
                                 side_effect=lambda repo: self.cache / repo.replace("/", "--"))
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _folder(self, repo: str, revision: str, files: dict[str, str]) -> Path:
        path = self.cache / repo.replace("/", "--") / "snapshots" / revision
        path.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (path / name).write_text(content)
        return path

    def test_sharded_weights_count_when_every_shard_is_there(self) -> None:
        index = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
        row = {"repo": "org/sharded", "revision": "r1", "companions": []}
        folder = self._folder("org/sharded", "r1", {"config.json": "{}", "model.safetensors.index.json": index,
                                                    "model-00001-of-00002.safetensors": "x"})
        self.assertIsNone(voice_clone.snapshot_dir(row))
        (folder / "model-00002-of-00002.safetensors").write_text("x")
        self.assertEqual(voice_clone.snapshot_dir(row), folder)

    def test_a_missing_companion_means_not_installed(self) -> None:
        row = voice_models.voice_model("orpheus-3b")
        model = self._folder(row["repo"], row["revision"], {"config.json": "{}", "model.safetensors": "x"})
        self.assertIsNone(voice_clone.snapshot_dir(row))
        for companion in row["companions"]:
            self._folder(companion["repo"], companion["revision"], {name: "x" for name in companion["files"]})
        self.assertEqual(voice_clone.snapshot_dir(row), model)
        self.assertEqual(set(voice_clone.companion_paths(row)), {c["repo"] for c in row["companions"]})


class TagTests(unittest.TestCase):
    def test_apply_tags_per_style(self) -> None:
        turbo = frozenset(voice_models.TURBO_TAGS + voice_models.TURBO_STYLE_TAGS)
        orpheus = frozenset(voice_models.ORPHEUS_TAGS)
        self.assertEqual(voice_text.apply_tags("Ha [laugh] yes [whispering] ok.", turbo, "keep"), ("Ha [laugh] yes [whispering] ok.", False))
        # A tag Turbo does not know is removed, never read aloud.
        self.assertEqual(voice_text.apply_tags("Oh [yawn] fine.", turbo, "keep"), ("Oh fine.", True))
        self.assertEqual(voice_text.apply_tags("Hi [laugh] and [sniff] bye.", orpheus, "angle"), ("Hi <laugh> and <sniffle> bye.", False))
        self.assertEqual(voice_text.apply_tags("Well [Clear Throat] hello.", frozenset(), "strip"), ("Well hello.", True))
        # Other brackets are not tags.
        self.assertEqual(voice_text.apply_tags("See note [1] and [sic].", frozenset(), "strip"), ("See note [1] and [sic].", False))

    def test_a_tag_stays_with_its_sentence(self) -> None:
        pieces = voice_text.split_sentences(voice_text.clean_for_speech("That is so funny! [laugh]\n\nNext line here."))
        self.assertEqual(voice_text.keep_tags_with_sentences(pieces), ["That is so funny! [laugh]", "Next line here."])
        self.assertEqual(voice_text.keep_tags_with_sentences(["[sigh].", "Long day."]), ["[sigh] Long day."])

    def test_orpheus_tags_survive_cleaning_as_angle_brackets(self) -> None:
        # clean_for_speech removes anything shaped like an HTML tag, so the angle form is written after it.
        self.assertEqual(voice_text.clean_for_speech("Hi <laugh> there."), "Hi there.")
        plan = voice_speak.Plan("preset", "orpheus:tara", "orpheus-3b", "Orpheus", "mlx-audio", "/s", {},
                                family="orpheus", chunk_chars=250, tag_style="angle", tags=frozenset(voice_models.ORPHEUS_TAGS))
        sentences, notes = voice_speak.prepare_text("Hi **there** [laugh]. That was [sniff] sad.", plan)
        self.assertEqual(sentences, ["Hi there <laugh>.", "That was <sniffle> sad."])
        self.assertEqual(notes, [])

    def test_models_without_tags_drop_them_with_a_note(self) -> None:
        plan = voice_speak.Plan("preset", "qwen3:Ryan", "k", "Qwen", "mlx-audio", "/s", {}, family="qwen3-tts")
        sentences, notes = voice_speak.prepare_text("Hello [laugh] world.", plan)
        self.assertEqual(sentences, ["Hello world."])
        self.assertEqual(notes, [voice_speak.TAGS_SKIPPED])
        kokoro = voice_speak.build_plan("af_heart")
        self.assertEqual(voice_speak.prepare_text("Oh [sigh] well.", kokoro)[0], ["Oh well."])

    def test_voxcpm2_never_gets_a_leading_paren(self) -> None:
        self.assertEqual(voice_text.guard_leading_paren("(code block omitted)."), "code block omitted.")
        self.assertEqual(voice_text.guard_leading_paren("(aside) the rest"), "aside the rest")
        self.assertEqual(voice_text.guard_leading_paren("No paren (here)."), "No paren (here).")
        plan = voice_speak.Plan("design", "design", "voxcpm2", "VoxCPM2", "mlx-audio", "/s", {}, family="voxcpm2")
        sentences, _ = voice_speak.prepare_text("Look:\n\n```py\nx = 1\n```\n\n(Quietly) we go on.", plan)
        self.assertFalse(any(s.lstrip().startswith("(") for s in sentences), sentences)

    def test_orpheus_chunks_are_short_enough_for_its_token_budget(self) -> None:
        row = voice_models.voice_model("orpheus-3b")
        plan = voice_speak.Plan("preset", "orpheus:tara", row["key"], row["name"], "mlx-audio", "/s", {},
                                family="orpheus", chunk_chars=row["chunkChars"])
        text = " ".join(["This sentence keeps going with many words and no full stop"] * 20) + "."
        self.assertTrue(all(len(s) <= row["chunkChars"] for s in voice_speak.prepare_text(text, plan)[0]))


class PlanTests(unittest.TestCase):
    def test_turbo_orpheus_and_voxcpm2_design_plans(self) -> None:
        with every_row_listed(), Machine(), installed(*WAVE2, "qwen3-tts-design-1.7b"):
            turbo = voice_speak.build_plan("chatterbox-turbo", style="angry")
            orpheus = voice_speak.build_plan("orpheus:leo")
            design = voice_speak.build_plan("design", style="A calm older man", model_key="voxcpm2")
            qwen_design = voice_speak.build_plan("design", style="A calm older man")
        self.assertEqual(turbo.extra["family"], "chatterbox-turbo")
        self.assertIn("mlx-community/S3TokenizerV2", turbo.extra["companions"])
        self.assertTrue(turbo.notes)  # style ignored, with a note
        self.assertEqual((orpheus.extra["speaker"], orpheus.extra["maxTokens"], orpheus.tag_style), ("leo", 2400, "angle"))
        self.assertEqual(set(orpheus.extra["companions"]), {"mlx-community/snac_24khz", "mlx-community/orpheus-3b-0.1-ft-bf16"})
        self.assertEqual(design.extra, {"family": "voxcpm2", "companions": {}, "mode": "design", "instruct": "A calm older man"})
        self.assertEqual(qwen_design.model_key, "qwen3-tts-design-1.7b")  # the first installed designer
        for plan in (turbo, orpheus, design, qwen_design):
            self.assertFalse({"refAudioPath", "promptAudioPath", "continueAudioPath"} & set(plan.extra))

    def test_unknown_orpheus_voice_and_wrong_model(self) -> None:
        with every_row_listed(), Machine(), installed(*WAVE2):
            with self.assertRaisesRegex(ValueError, "Unknown voice"):
                voice_speak.build_plan("orpheus:nobody")
            with self.assertRaisesRegex(ValueError, "cannot speak this voice"):
                voice_speak.build_plan("design", style="x", model_key="chatterbox-turbo")

    def test_wave2_voices_are_hidden_until_the_row_is_listed(self) -> None:
        with Machine(), installed(*WAVE2), mock.patch.object(voice_models, "listed_models",
                                                             return_value=[r for r in voice_models.VOICE_MODELS if r["family"] == "qwen3-tts"]):
            ids = {v["id"] for v in voice_speak.list_voices(include_clones=False)}
            with self.assertRaisesRegex(ValueError, "Unknown voice"):
                voice_speak.build_plan("orpheus:tara")
        self.assertFalse(any(i.startswith("orpheus:") or i == "chatterbox-turbo" for i in ids))

    def test_listed_voices_carry_group_family_and_models(self) -> None:
        with every_row_listed(), Machine(), installed(*WAVE2), mock.patch.object(voice_env, "watermark_installed", return_value=True), \
                mock.patch.object(voice_clone, "list_voices", return_value=[{"id": "1234567890_deadbeef", "name": "Me", "consent": {}, "durationS": 8.0}]):
            voices = {v["id"]: v for v in voice_speak.list_voices()}
        self.assertEqual(voices["chatterbox-turbo"]["group"], "natural")
        self.assertEqual(len([v for v in voices if v.startswith("orpheus:")]), 8)
        self.assertEqual(voices["orpheus:tara"]["family"], "orpheus")
        self.assertIn("voxcpm2", voices["design"]["models"])
        clone_models = voices["clone:1234567890_deadbeef"]["models"]
        self.assertTrue({"voxcpm2", "chatterbox-turbo", "chatterbox-mtl"} <= set(clone_models))
        self.assertNotIn("orpheus-3b", clone_models)
        self.assertEqual(voices["af_heart"]["group"], "quick")

    def test_new_fields_default_to_nothing_and_are_validated(self) -> None:
        with mock.patch.object(voice_speak, "render") as render, mock.patch.object(voice_speak, "build_plan") as build:
            render.return_value = voice_speak.Rendered(make_wav(), 0.2, 0.1, 1, [])
            voice_speak.speak("Hi", "af_heart", origin="ui")
        build.assert_called_once_with("af_heart", "", "", language="auto", expression=0.5, guidance=0.5)
        for bad in ({"language": "english"}, {"expression": 1.5}, {"guidance": -0.1}):
            with self.assertRaises(ValueError):
                voice_speak.speak("Hi", "af_heart", origin="ui", **bad)
        self.assertEqual(voice_speak.chatterbox_language("auto"), "en")
        self.assertEqual(voice_speak.chatterbox_language("de-AT"), "de")
        with self.assertRaisesRegex(ValueError, "Chatterbox speaks"):
            voice_speak.chatterbox_language("cy")

    def test_qwen_presets_take_a_language_name(self) -> None:
        with every_row_listed(), Machine(), installed("qwen3-tts-custom-1.7b"):
            self.assertEqual(voice_speak.build_plan("qwen3:Ryan").extra["language"], "auto")
            self.assertEqual(voice_speak.build_plan("qwen3:Ryan", language="de").extra["language"], "german")
            self.assertEqual(voice_speak.build_plan("qwen3:Ryan", language="cy").extra["language"], "auto")

    def test_voxcpm2_design_continues_from_its_own_first_chunk(self) -> None:
        plan = voice_speak.Plan("design", "design", "voxcpm2", "VoxCPM2", "mlx-audio", "/s",
                                {"family": "voxcpm2", "mode": "design", "instruct": "A calm man"}, family="voxcpm2")
        worker = mock.Mock()
        worker.request.return_value = {"wavB64": base64.b64encode(make_wav(rate=48_000)).decode(), "seconds": 0.1}
        with mock.patch("backend_service.voice_client.get_worker", return_value=worker):
            rendered = voice_speak.render(plan, "First sentence here. Second one now. Third and last.")
        calls = [c.kwargs for c in worker.request.call_args_list]
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["instruct"], "A calm man")
        for later in calls[1:]:
            self.assertNotIn("instruct", later)
            self.assertEqual((later["mode"], later["continueText"]), ("continue", "First sentence here."))
            self.assertTrue(later["continueAudioPath"].endswith("first.wav"))
        self.assertEqual(voice_speak.wav_info(rendered.wav)[2], 48_000)  # joined at the file's own rate


class CloneRuleTests(unittest.TestCase):
    """Every model that can clone gets the same marks and permission rules (D-47, D-70)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        self._patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", self.data)
        self._patch.start()
        sample = self.data / "sample.m4a"
        sample.write_bytes(b"sample-audio")
        self.voice = voice_clone.create_voice(
            speaker_name="Dan", basis="own", name="Mine", sample_audio=sample, sample_ext=".m4a", sample_text=None,
            permission_confirmed=True, origin="ui", transcribe=mock.Mock(return_value={"text": "Hello, my voice.", "durationS": 8.0}),
        )

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _speak(self, key: str, *, origin: str = "ui", permission: bool = False, style: str = "", **kw):
        worker = mock.Mock()
        rate = voice_models.voice_model(key)["sampleRate"]
        worker.request.return_value = {"wavB64": base64.b64encode(make_wav(rate=rate)).decode(), "seconds": 0.1}
        rows = [dict(r, listed=True) for r in voice_models.VOICE_MODELS if r["capabilities"]["clone"]]
        with mock.patch.object(voice_clone, "CLONE_MODELS", rows), \
             mock.patch.object(voice_models, "CLONE_MODELS", rows), \
             mock.patch.object(voice_clone, "snapshot_dir", return_value=self.data), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True), \
             mock.patch("backend_service.voice_client.get_worker", return_value=worker), \
             mock.patch.object(voice_clone_safety, "apply_watermark", return_value=True) as mark:
            item = voice_clone.speak(self.voice["id"], "Hello there [laugh].", key, origin=origin,
                                     permission_confirmed=permission, style=style, **kw)
        return item, worker.request.call_args.kwargs, mark

    def test_every_cloning_model_watermarks_tags_saves_and_audits(self) -> None:
        self.assertEqual(set(CLONERS), {"chatterbox-turbo", "voxcpm2", "chatterbox-mtl"})
        for key in CLONERS:
            with self.subTest(key=key):
                row = voice_models.voice_model(key)
                item, sent, mark = self._speak(key)
                mark.assert_called_once()
                self.assertEqual(sent["family"], row["family"])
                self.assertTrue(sent["refAudioPath"])
                tag = voice_clone_safety.read_wav_tag(voice_library.audio_path(item["id"]))
                self.assertTrue(tag["machine_generated"] and tag["clonedVoice"])
                self.assertEqual(tag["model"], row["repo"])
                self.assertTrue(item["clonedVoice"])
                audit = [json.loads(l) for l in voice_clone_safety.audit_path().read_text().splitlines()]
                self.assertEqual(audit[-1]["event"], "speak")
                self.assertEqual(audit[-1]["model"], key)

    def test_agents_must_confirm_permission_on_every_cloning_model(self) -> None:
        for key in CLONERS:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "permission_confirmed"):
                self._speak(key, origin="agent")
            item, _sent, mark = self._speak(key, origin="agent", permission=True)
            mark.assert_called_once()

    def test_voxcpm2_modes_and_chatterbox_controls(self) -> None:
        _item, exact, _ = self._speak("voxcpm2")
        self.assertEqual((exact["mode"], exact["refText"]), ("exact", "Hello, my voice."))
        self.assertNotIn("instruct", exact)
        item, steered, _ = self._speak("voxcpm2", style="cheerful and quick")
        self.assertEqual((steered["mode"], steered["instruct"]), ("steered", "cheerful and quick"))
        self.assertIn(voice_clone.STEERED_NOTE, item["notes"])
        _item, mtl, _ = self._speak("chatterbox-mtl", language="de", expression=0.8, guidance=0.0)
        self.assertEqual((mtl["language"], mtl["expression"], mtl["guidance"]), ("de", 0.8, 0.0))
        self.assertNotIn("defaultVoicePath", mtl)
        item, turbo, _ = self._speak("chatterbox-turbo", style="angry")
        self.assertIn("[laugh]", turbo["text"])  # Turbo keeps its tags on the clone path too
        self.assertIn(voice_clone.NO_STYLE_NOTE, item["notes"])

    def test_turbo_needs_five_seconds_of_sample(self) -> None:
        library_item = voice_library.get(self.voice["id"])
        with mock.patch.object(voice_clone, "get_voice", return_value={**library_item, "durationS": 4.0}):
            with self.assertRaisesRegex(ValueError, "at least 5 s"):
                self._speak("chatterbox-turbo")
            self._speak("voxcpm2")  # 3 s minimum

    def test_orpheus_cannot_clone(self) -> None:
        with self.assertRaises(ValueError):
            voice_models.clone_model("orpheus-3b")


class FakeModel:
    """Records generate() calls; ``_conds`` mimics Chatterbox's stored voice."""

    sample_rate = 24_000

    def __init__(self, conds=None):
        self._conds = conds
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        if "ref_audio" in kwargs:
            self._conds = "cloned"  # what Turbo's prepare_conditionals does
        return [types.SimpleNamespace(audio=[0.0, 0.1, 0.0])]


def decode(path, rate):
    return [0.0] * 10


class WorkerCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._seed = mock.patch.object(wm, "_seed")
        self._seed.start()

    def tearDown(self) -> None:
        self._seed.stop()

    def test_turbo_arguments_and_the_builtin_voice_is_restored_after_cloning(self) -> None:
        model = FakeModel(conds="builtin")
        wm.synthesize(model, {"family": "chatterbox-turbo", "text": "Hi [laugh]."}, decode)
        self.assertEqual(model.calls[0], {"text": "Hi [laugh].", "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.2,
                                          "split_pattern": None, "max_tokens": 800})
        wm.synthesize(model, {"family": "chatterbox-turbo", "text": "Hi.", "refAudioPath": "/clip.wav"}, decode)
        self.assertEqual(model.calls[1]["sample_rate"], 24_000)
        self.assertEqual(model._conds, "builtin")  # never the cloned voice for the next plain request

    def test_chatterbox_sends_language_expression_guidance(self) -> None:
        mx = types.ModuleType("mlx.core")
        mx.array = lambda value: ("mx", value)
        with mock.patch.dict(sys.modules, {"mlx": types.ModuleType("mlx"), "mlx.core": mx}):
            model = FakeModel(conds=None)
            with self.assertRaisesRegex(ValueError, "voice sample"):
                wm.synthesize(model, {"family": "chatterbox", "text": "Hallo."}, decode)
            wm.synthesize(model, {"family": "chatterbox", "text": "Hallo.", "language": "de", "expression": 0.7,
                                  "guidance": 0.3, "refAudioPath": "/clip.wav"}, decode)
        call = model.calls[-1]
        self.assertEqual((call["lang_code"], call["exaggeration"], call["cfg_weight"]), ("de", 0.7, 0.3))
        self.assertEqual(call["ref_audio"][0], "mx")
        self.assertIsNone(model._conds)

    def test_voxcpm2_modes(self) -> None:
        model = FakeModel()
        model.sample_rate = 48_000
        wm.synthesize(model, {"family": "voxcpm2", "text": "Hi.", "instruct": "A calm woman"}, decode)
        wm.synthesize(model, {"family": "voxcpm2", "text": "Hi.", "refAudioPath": "/c.wav", "refText": "Hello", "mode": "exact"}, decode)
        wm.synthesize(model, {"family": "voxcpm2", "text": "Hi.", "refAudioPath": "/c.wav", "mode": "steered", "instruct": "cheerful"}, decode)
        wm.synthesize(model, {"family": "voxcpm2", "text": "Hi.", "continueAudioPath": "/first.wav", "continueText": "First."}, decode)
        design, exact, steered, cont = model.calls
        self.assertEqual((design["instruct"], design["inference_timesteps"], design["cfg_value"]), ("A calm woman", 10, 2.0))
        self.assertEqual(exact["prompt_text"], "Hello")
        self.assertIn("prompt_audio", exact)
        self.assertNotIn("prompt_audio", steered)
        self.assertEqual(steered["instruct"], "cheerful")
        self.assertEqual(cont["prompt_text"], "First.")
        self.assertNotIn("instruct", cont)
        with self.assertRaises(ValueError):
            wm.synthesize(model, {"family": "voxcpm2", "text": "Hi."}, decode)

    def test_orpheus_voice_budget_and_no_cloning(self) -> None:
        model = FakeModel()
        wm.synthesize(model, {"family": "orpheus", "text": "Hi <laugh>.", "speaker": "zoe", "maxTokens": 2400}, decode)
        self.assertEqual(model.calls[0], {"text": "Hi <laugh>.", "voice": "zoe", "temperature": 0.6, "top_p": 0.8,
                                          "repetition_penalty": 1.3, "split_pattern": "\n", "max_tokens": 2400})
        with self.assertRaisesRegex(ValueError, "does not clone"):
            wm.synthesize(model, {"family": "orpheus", "text": "Hi.", "refAudioPath": "/c.wav"}, decode)
        with self.assertRaises(ValueError):
            wm.synthesize(model, {"family": "nope", "text": "x"}, decode)


class FakeAutoTokenizer:
    loaded: list[str] = []

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        cls.loaded.append(str(name))
        return name


class RedirectTests(unittest.TestCase):
    def setUp(self) -> None:
        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = FakeAutoTokenizer
        self._transformers = mock.patch.dict(sys.modules, {"transformers": fake})
        self._transformers.start()
        FakeAutoTokenizer.loaded = []

    def tearDown(self) -> None:
        wm.remove_redirect()
        self._transformers.stop()

    def test_known_companions_go_to_the_pinned_folder_others_pass_through(self) -> None:
        import huggingface_hub

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(huggingface_hub, "snapshot_download", return_value="/downloaded") as original, \
                mock.patch.object(huggingface_hub, "hf_hub_download", return_value="/downloaded/file") as original_file:
            Path(tmp, "model.safetensors").write_text("x")
            stale = types.ModuleType("mlx_audio.fake_loader")
            stale.snapshot_download = huggingface_hub.snapshot_download  # bound before the redirect, like snac.py
            with mock.patch.dict(sys.modules, {"mlx_audio.fake_loader": stale}):
                wm.install_redirect({"mlx-community/S3TokenizerV2": tmp, "someone/else": "/ignored"})
                self.assertEqual(huggingface_hub.snapshot_download(repo_id="mlx-community/S3TokenizerV2"), tmp)
                self.assertEqual(stale.snapshot_download("mlx-community/S3TokenizerV2"), tmp)
                self.assertEqual(huggingface_hub.hf_hub_download(repo_id="mlx-community/S3TokenizerV2", filename="model.safetensors"),
                                 str(Path(tmp, "model.safetensors")))
                with self.assertRaisesRegex(wm.CompanionMissing, "not downloaded"):
                    huggingface_hub.snapshot_download("mlx-community/snac_24khz")
                with self.assertRaises(wm.CompanionMissing):
                    huggingface_hub.hf_hub_download("mlx-community/S3TokenizerV2", "missing.bin")
                self.assertEqual(huggingface_hub.snapshot_download("prince-canuma/Kokoro-82M"), "/downloaded")
                original.assert_called_once_with("prince-canuma/Kokoro-82M")
                original_file.assert_not_called()
                self.assertEqual(wm.resolve("someone/else"), None)
                # Orpheus reads its tokenizer from the bf16 repo by name; that one must be sent, never fetched.
                with self.assertRaises(wm.CompanionMissing):
                    FakeAutoTokenizer.from_pretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
                wm.install_redirect({"mlx-community/orpheus-3b-0.1-ft-bf16": tmp})
                FakeAutoTokenizer.from_pretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
                FakeAutoTokenizer.from_pretrained("/local/model")
                self.assertEqual(FakeAutoTokenizer.loaded, [tmp, "/local/model"])

    def test_the_worker_installs_the_redirect_before_loading(self) -> None:
        from backend_service import voice_worker

        order: list[str] = []
        with mock.patch.object(wm, "install_redirect", side_effect=lambda c: order.append("redirect")), \
                mock.patch.object(voice_worker, "_load_tts", side_effect=lambda e, s: order.append("load") or FakeModel(conds="b")), \
                mock.patch.object(wm, "_seed"):
            reply = voice_worker.synthesize({"engine": "mlx-audio", "source": "/m", "family": "chatterbox-turbo", "text": "Hi.",
                                             "companions": {"mlx-community/S3TokenizerV2": "/c"}})
        self.assertEqual(order, ["redirect", "load"])
        self.assertEqual(reply["sampleRate"], 24_000)


class RouteTests(unittest.TestCase):
    def test_speak_request_fields(self) -> None:
        from pydantic import ValidationError

        from backend_service.routes.voice import SpeakRequest

        old = SpeakRequest(text="Hi", voice="af_heart")
        self.assertEqual((old.language, old.expression, old.guidance), ("auto", 0.5, 0.5))
        self.assertEqual(SpeakRequest(text="Hi", language="pt-BR", expression=1, guidance=0).language, "pt-BR")
        for bad in ({"language": "English"}, {"expression": 1.2}, {"guidance": -1}):
            with self.assertRaises(ValidationError):
                SpeakRequest(text="Hi", **bad)

    def test_models_route_lists_rows_with_controls_and_licence(self) -> None:
        with every_row_listed(), mock.patch.object(voice_clone, "snapshot_dir", return_value=None):
            rows = {r["key"]: r for r in voice_clone.list_models(every_model=True)}
        self.assertEqual(rows["voxcpm2"]["sampleRate"], 48_000)
        self.assertEqual(rows["orpheus-3b"]["licence"]["id"], "apache-2.0")
        self.assertEqual(rows["chatterbox-turbo"]["controls"]["styleTags"][0], "angry")
        self.assertEqual(rows["chatterbox-turbo"]["minReferenceS"], 5.0)
        self.assertEqual(rows["orpheus-3b"]["companions"][0]["repo"], "mlx-community/snac_24khz")
        self.assertIsNone(rows["chatterbox-turbo"]["builtinWatermark"])
        self.assertTrue(re.match(r"^\d+(\.\d+)?$", str(rows["orpheus-3b"]["downloadGb"])))


if __name__ == "__main__":
    unittest.main()
