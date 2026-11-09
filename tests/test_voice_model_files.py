"""Deleting voice files (D-73): speech models with their companions, dictation models, add-ons, the runtime.

Every test runs on a throwaway Hugging Face cache and voice venv; nothing touches the real ones.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_clone, voice_client, voice_env, voice_library, voice_model_files, voice_runtime_disk
from backend_service.catalog import voice_models as catalog


def _row(key: str) -> dict:
    return catalog.voice_model(key)


class _HfCache(unittest.TestCase):
    """A temporary HF cache; ``put(repo, revision, files)`` writes a pinned snapshot."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "hub"
        self.cache.mkdir()
        self.env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache), "HUGGINGFACE_HUB_CACHE": str(self.cache)})
        self.env.start()
        self.downloads = mock.patch.dict(voice_clone._DOWNLOADS, {}, clear=True)
        self.downloads.start()

    def tearDown(self) -> None:
        self.downloads.stop()
        self.env.stop()
        self.tmp.cleanup()

    def folder(self, repo: str) -> Path:
        return self.cache / f"models--{repo.replace('/', '--')}"

    def put(self, repo: str, revision: str, files: list[str] | None = None) -> Path:
        snapshot = self.folder(repo) / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in files or ["config.json", "model.safetensors"]:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes(b"x" * 1000)
        return snapshot

    def download(self, key: str) -> None:
        """Put the row's weights and every companion on disk."""
        row = _row(key)
        self.put(row["repo"], row["revision"], row.get("files"))
        for companion in row.get("companions", []):
            self.put(companion["repo"], companion["revision"], companion["files"])


def _idle():
    """No voice job, worker not busy, nothing loaded."""
    return [
        mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None),
        mock.patch.object(voice_client, "loaded_sources", return_value={}),
    ]


class SpeechModelDeleteTests(_HfCache):
    def run_idle(self, fn, *args):
        patches = _idle()
        for patch in patches:
            patch.start()
        try:
            return fn(*args)
        finally:
            for patch in patches:
                patch.stop()

    def test_unshared_companions_go_with_the_model(self) -> None:
        self.download("orpheus-3b")
        result = self.run_idle(voice_model_files.remove_speech_model, "orpheus-3b")
        self.assertTrue(result["deleted"])
        for repo in [_row("orpheus-3b")["repo"], *(c["repo"] for c in _row("orpheus-3b")["companions"])]:
            self.assertFalse(self.folder(repo).exists(), repo)
            self.assertIn(repo, result["removed"])
        self.assertGreater(result["freedGb"], 0.0 - 1e-9)

    def test_a_companion_another_downloaded_model_uses_is_kept(self) -> None:
        # Two rows that share the S3 tokenizer: Chatterbox-Turbo and a fake second user of it.
        self.download("chatterbox-turbo")
        tokenizer = _row("chatterbox-turbo")["companions"][0]["repo"]
        sharer = dict(_row("qwen3-tts-0.6b"), key="sharer", name="Sharer",
                      companions=[_row("chatterbox-turbo")["companions"][0]])
        self.put(sharer["repo"], sharer["revision"])
        with mock.patch.object(catalog, "VOICE_MODELS", [*catalog.VOICE_MODELS, sharer]):
            result = self.run_idle(voice_model_files.remove_speech_model, "chatterbox-turbo")
        self.assertFalse(self.folder(_row("chatterbox-turbo")["repo"]).exists())
        self.assertTrue(self.folder(tokenizer).exists())
        self.assertEqual(result["kept"], [{"repo": tokenizer, "usedBy": ["Sharer"]}])

    def test_the_watermark_stays_while_another_cloning_model_is_downloaded(self) -> None:
        self.download("qwen3-tts-0.6b")
        self.download("qwen3-tts-1.7b")
        self.put(voice_env.WATERMARK_REPO, voice_env.WATERMARK_REVISION, list(voice_env.WATERMARK_FILES.values()))
        first = self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-0.6b")
        self.assertFalse(first["watermarkRemoved"])
        self.assertTrue(self.folder(voice_env.WATERMARK_REPO).exists())
        last = self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-1.7b")
        self.assertTrue(last["watermarkRemoved"])
        self.assertFalse(self.folder(voice_env.WATERMARK_REPO).exists())

    def test_a_model_that_cannot_clone_leaves_the_watermark(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        self.put(voice_env.WATERMARK_REPO, voice_env.WATERMARK_REVISION, list(voice_env.WATERMARK_FILES.values()))
        result = self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-custom-1.7b")
        self.assertFalse(result["watermarkRemoved"])
        self.assertTrue(self.folder(voice_env.WATERMARK_REPO).exists())

    def test_the_worker_is_stopped_when_it_holds_the_model(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        snapshot = str(self.put(_row("qwen3-tts-custom-1.7b")["repo"], _row("qwen3-tts-custom-1.7b")["revision"]))
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None), \
                mock.patch.object(voice_client, "loaded_sources", return_value={"tts": snapshot}), \
                mock.patch.object(voice_client, "stop_speech_worker") as stop:
            result = voice_model_files.remove_speech_model("qwen3-tts-custom-1.7b")
        stop.assert_called_once()
        self.assertTrue(result["workerStopped"])

    def test_the_worker_keeps_running_when_it_holds_another_model(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None), \
                mock.patch.object(voice_client, "loaded_sources", return_value={"tts": voice_env.KOKORO_MLX_REPO}), \
                mock.patch.object(voice_client, "stop_speech_worker") as stop:
            voice_model_files.remove_speech_model("qwen3-tts-custom-1.7b")
        stop.assert_not_called()

    def test_refused_while_a_voice_job_runs(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        from backend_service.helpers.job_queue import HEAVY_JOBS

        with HEAVY_JOBS.slot("voice", "ui", "speech"):
            with self.assertRaises(voice_model_files.ModelInUse):
                voice_model_files.remove_speech_model("qwen3-tts-custom-1.7b")
        self.assertTrue(self.folder(_row("qwen3-tts-custom-1.7b")["repo"]).exists())

    def test_refused_while_the_worker_is_answering(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        with mock.patch.object(voice_client, "worker_busy", return_value=True):
            with self.assertRaises(voice_model_files.ModelInUse):
                voice_model_files.remove_speech_model("qwen3-tts-custom-1.7b")
        self.assertTrue(self.folder(_row("qwen3-tts-custom-1.7b")["repo"]).exists())

    def test_refused_while_a_speech_model_downloads(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        voice_clone._DOWNLOADS["voxcpm2"] = {"state": "downloading", "error": None}
        with self.assertRaises(voice_model_files.ModelInUse):
            self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-custom-1.7b")
        self.assertTrue(self.folder(_row("qwen3-tts-custom-1.7b")["repo"]).exists())

    def test_unknown_key_is_a_value_error(self) -> None:
        with self.assertRaises(ValueError):
            voice_model_files.removal_plan("not-a-model")

    def test_the_download_record_is_cleared(self) -> None:
        self.download("qwen3-tts-custom-1.7b")
        voice_clone._DOWNLOADS["qwen3-tts-custom-1.7b"] = {"state": "failed", "error": "x"}
        self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-custom-1.7b")
        self.assertNotIn("qwen3-tts-custom-1.7b", voice_clone._DOWNLOADS)

    def test_cloned_voices_and_the_library_are_not_touched(self) -> None:
        library = Path(self.tmp.name) / "library"
        library.mkdir()
        (library / "keep.json").write_text("{}", encoding="utf-8")
        self.download("qwen3-tts-0.6b")
        with mock.patch.object(voice_library, "library_dir", return_value=library):
            self.run_idle(voice_model_files.remove_speech_model, "qwen3-tts-0.6b")
        self.assertTrue((library / "keep.json").exists())


class RepoPathTests(_HfCache):
    def test_a_repo_id_must_stay_in_the_cache(self) -> None:
        for bad in ("../etc", "a/../../b", "/abs/path", "noslash", "a/b/c", ""):
            with self.assertRaises(ValueError, msg=bad):
                voice_model_files.repo_cache_dir(bad)

    def test_a_symlinked_repo_folder_outside_the_cache_is_refused(self) -> None:
        outside = Path(self.tmp.name) / "elsewhere"
        outside.mkdir()
        self.folder("evil/model").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            voice_model_files.repo_cache_dir("evil/model")

    def test_a_good_repo_resolves_into_the_cache(self) -> None:
        self.assertEqual(voice_model_files.repo_cache_dir("org/model").parent, self.cache.resolve())


class SttDeleteTests(_HfCache):
    def test_unknown_dictation_model(self) -> None:
        with self.assertRaises(ValueError):
            voice_model_files.prepare_stt_delete("not-a-model")

    def test_refused_while_busy(self) -> None:
        repo = voice_model_files.stt_repo(voice_runtime_disk_stt_key())
        with mock.patch.object(voice_client, "worker_busy", return_value=True):
            with self.assertRaises(voice_model_files.ModelInUse):
                voice_model_files.prepare_stt_delete(repo)

    def test_the_worker_stops_when_it_holds_the_dictation_model(self) -> None:
        repo = voice_model_files.stt_repo(voice_runtime_disk_stt_key())
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None), \
                mock.patch.object(voice_client, "loaded_sources", return_value={"stt": repo}), \
                mock.patch.object(voice_client, "stop_speech_worker") as stop:
            self.assertEqual(voice_model_files.prepare_stt_delete(repo), repo)
        stop.assert_called_once()


def voice_runtime_disk_stt_key() -> str:
    from backend_service.voice_runtime import list_stt_models

    return str(list_stt_models()[0]["key"])


class _Venv(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "hub"
        self.cache.mkdir()
        venv = Path(self.tmp.name) / "voice-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        site = venv / "lib" / "python3.12" / "site-packages"
        (site / "misaki-0.9.4.dist-info").mkdir(parents=True)
        (site / "misaki-0.9.4.dist-info" / "METADATA").write_text(
            "Name: misaki\nRequires-Dist: num2words\nRequires-Dist: jieba; extra == \"zh\"\n"
            "Requires-Dist: pypinyin>=0.50; extra == 'zh'\nRequires-Dist: spacy; extra == \"en\"\n",
            encoding="utf-8",
        )
        self.venv = venv
        self.marker = venv / "chaosengine-voice.json"
        self.marker.write_text(json.dumps({"schema": 1, "lane": "mlx", "extras": ["chinese"], "features": voice_env.FEATURES}),
                               encoding="utf-8")
        self.patches = [
            mock.patch.object(voice_env, "VOICE_VENV_DIR", venv),
            mock.patch.object(voice_env, "_MARKER", self.marker),
            mock.patch.object(voice_env.os, "name", "posix"),
            mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_LANE": "mlx", "HF_HUB_CACHE": str(self.cache),
                                         "HUGGINGFACE_HUB_CACHE": str(self.cache)}),
            mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None),
            mock.patch.object(voice_client, "stop_all"),
            mock.patch.object(voice_client, "stop_speech_worker"),
        ]
        for patch in self.patches:
            patch.start()
        voice_runtime_disk._PARTS_CACHE.clear()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        voice_runtime_disk._PARTS_CACHE.clear()
        self.tmp.cleanup()

    def hub(self, repo: str) -> Path:
        folder = self.cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "x"
        folder.mkdir(parents=True)
        (folder / "weights").write_bytes(b"y" * 2_000_000)
        return folder.parent.parent


class RuntimePartsTests(_Venv):
    def test_four_parts_with_sizes(self) -> None:
        self.hub(voice_env.KOKORO_MLX_REPO)
        parts = voice_runtime_disk.pack_parts()
        self.assertEqual([p["id"] for p in parts], ["packages", "kokoro", "turns", "speakers"])
        self.assertGreater(parts[1]["sizeGb"], 0)

    def test_nothing_when_not_installed(self) -> None:
        self.marker.unlink()
        self.assertEqual(voice_runtime_disk.pack_parts(), [])


class AddonRemovalTests(_Venv):
    def test_untick_uninstalls_only_the_extra_packages_and_edits_the_marker(self) -> None:
        with mock.patch.object(voice_runtime_disk.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            result = voice_runtime_disk.remove_extra("chinese")
        command = run.call_args.args[0]
        self.assertEqual(command[1:5], ["-m", "pip", "uninstall", "-y"])
        self.assertEqual(sorted(command[5:]), ["jieba", "pypinyin"])
        self.assertNotIn("misaki", command)
        self.assertEqual(result["packages"], ["jieba", "pypinyin"])
        self.assertEqual(voice_env.installed_extras(), [])

    def test_tick_then_untick_round_trip(self) -> None:
        self.assertEqual(voice_env.installed_extras(), ["chinese"])
        with mock.patch.object(voice_runtime_disk.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")):
            voice_runtime_disk.remove_extra("chinese")
        self.assertEqual(voice_env.installed_extras(), [])
        with mock.patch.object(voice_env.threading, "Thread") as thread:
            job = voice_env.start_install(["chinese"])
        self.assertEqual(job["extras"], ["chinese"])
        thread.assert_called_once()
        voice_env._JOB.__init__()  # type: ignore[misc]

    def test_unknown_or_missing_add_on(self) -> None:
        with self.assertRaises(ValueError):
            voice_runtime_disk.remove_extra("klingon")
        self.marker.write_text(json.dumps({"schema": 1, "lane": "mlx", "extras": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            voice_runtime_disk.remove_extra("chinese")

    def test_refused_while_installing(self) -> None:
        with mock.patch.object(voice_env, "job_status", return_value={"running": True}):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_extra("chinese")

    def test_a_failed_uninstall_keeps_the_marker(self) -> None:
        with mock.patch.object(voice_runtime_disk.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="no", stderr="")):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_extra("chinese")
        self.assertEqual(voice_env.installed_extras(), ["chinese"])


class RuntimeRemovalTests(_Venv):
    def test_remove_keeps_downloaded_models_by_default(self) -> None:
        kokoro = self.hub(voice_env.KOKORO_MLX_REPO)
        speech = self.hub(_row("qwen3-tts-custom-1.7b")["repo"])
        result = voice_runtime_disk.remove_runtime()
        self.assertTrue(result["removed"])
        self.assertFalse(self.venv.exists())
        self.assertTrue(kokoro.exists())
        self.assertTrue(speech.exists())
        self.assertFalse(voice_env.is_installed())

    def test_remove_with_the_runtime_models(self) -> None:
        kokoro = self.hub(voice_env.KOKORO_MLX_REPO)
        vad = self.hub(voice_env.VAD_REPO)
        speech = self.hub(_row("qwen3-tts-custom-1.7b")["repo"])
        result = voice_runtime_disk.remove_runtime(True)
        self.assertFalse(kokoro.exists())
        self.assertFalse(vad.exists())
        self.assertTrue(speech.exists(), "speech models are never part of the runtime")
        self.assertIn(voice_env.KOKORO_MLX_REPO, result["removedModels"])

    def test_refused_while_an_install_runs(self) -> None:
        with mock.patch.object(voice_env, "job_status", return_value={"running": True}):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_runtime()
        self.assertTrue(self.venv.exists())

    def test_refused_while_voice_is_busy(self) -> None:
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value="busy"):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_runtime()
        self.assertTrue(self.venv.exists())

    def test_a_folder_that_is_not_the_venv_is_not_deleted(self) -> None:
        (self.venv / "pyvenv.cfg").unlink()
        self.marker.unlink()
        with self.assertRaises(RuntimeError):
            voice_runtime_disk.remove_runtime()
        self.assertTrue(self.venv.exists())

    def test_the_home_folder_is_never_deleted(self) -> None:
        with mock.patch.object(voice_env, "VOICE_VENV_DIR", Path.home()):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk._safe_venv_dir()


class InMemoryTests(unittest.TestCase):
    def test_names_the_loaded_models(self) -> None:
        row = _row("qwen3-tts-custom-1.7b")
        source = f"/cache/models--{row['repo'].replace('/', '--')}/snapshots/abc"
        with mock.patch.object(voice_client, "loaded_sources", return_value={"tts": source}):
            self.assertEqual(voice_runtime_disk.in_memory(),
                             [{"lane": "tts", "name": row["name"], "peakGb": row["estimates"]["peakGb"]}])
        with mock.patch.object(voice_client, "loaded_sources", return_value={"tts": voice_env.KOKORO_MLX_REPO}):
            self.assertEqual(voice_runtime_disk.in_memory()[0]["name"], "Kokoro")

    def test_worker_tracks_what_it_loaded(self) -> None:
        worker = voice_client.VoiceWorker.__new__(voice_client.VoiceWorker)
        worker.loaded = {"tts": "x"}
        worker._process = None
        worker._kill()
        self.assertEqual(worker.loaded, {})


class LibraryVoiceKindTests(unittest.TestCase):
    def test_voice_kind_of_old_and_new_items(self) -> None:
        self.assertEqual(voice_library.voice_kind_of({"kind": "speech", "voiceKind": "preset"}), "preset")
        self.assertEqual(voice_library.voice_kind_of({"kind": "speech", "clonedVoice": True}), "clone")
        self.assertEqual(voice_library.voice_kind_of({"kind": "speech", "voice": "af_heart"}), "builtin")
        self.assertIsNone(voice_library.voice_kind_of({"kind": "speech", "voice": "qwen3:Ryan"}))
        self.assertIsNone(voice_library.voice_kind_of({"kind": "transcript"}))

    def test_list_filters_by_voice_kind(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder)
            items = [
                {"id": "a", "kind": "speech", "voiceKind": "preset", "createdAt": 1},
                {"id": "b", "kind": "speech", "voice": "af_heart", "createdAt": 2},
                {"id": "c", "kind": "transcript", "createdAt": 3},
            ]
            for item in items:
                (library / f"{item['id']}.json").write_text(json.dumps(item), encoding="utf-8")
            with mock.patch.object(voice_library, "library_dir", return_value=library), \
                    mock.patch.object(voice_library, "_migrate_gallery"):
                self.assertEqual([i["id"] for i in voice_library.list_items(voice_kind="preset")], ["a"])
                self.assertEqual([i["id"] for i in voice_library.list_items(voice_kind="builtin")], ["b"])
                self.assertEqual(len(voice_library.list_items()), 3)


class VoiceFileRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        restore_env(self._snapshot)
        self._tmp.cleanup()

    def test_unknown_speech_model_is_404(self) -> None:
        self.assertEqual(self.client.post("/api/voice/clone/models/delete", json={"key": "nope"}).status_code, 404)
        self.assertEqual(self.client.get("/api/voice/clone/models/delete-plan", params={"key": "nope"}).status_code, 404)

    def test_busy_is_409(self) -> None:
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value="A voice model is loading."):
            response = self.client.post("/api/voice/clone/models/delete", json={"key": "qwen3-tts-custom-1.7b"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("loading", response.json()["detail"])

    def test_delete_contract(self) -> None:
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None):
            body = self.client.post("/api/voice/clone/models/delete", json={"key": "qwen3-tts-custom-1.7b"}).json()
        for key in ("key", "deleted", "removed", "kept", "watermarkRemoved", "freedGb"):
            self.assertIn(key, body)
        plan = self.client.get("/api/voice/clone/models/delete-plan", params={"key": "qwen3-tts-custom-1.7b"}).json()
        for key in ("key", "name", "remove", "kept", "watermark", "sizeGb"):
            self.assertIn(key, plan)
        rows = {r["key"]: r for r in self.client.get("/api/voice/models").json()["models"]}
        self.assertFalse(rows["qwen3-tts-custom-1.7b"]["installed"])

    def test_agents_cannot_delete(self) -> None:
        from backend_service import voice_clone_safety

        with mock.patch.object(voice_clone_safety, "request_origin", return_value="agent"):
            for path, body in (
                ("/api/voice/clone/models/delete", {"key": "qwen3-tts-custom-1.7b"}),
                ("/api/voice/stt/delete", {"key": "whisper-base"}),
                ("/api/voice/extras/remove", {"extra": "chinese"}),
                ("/api/voice/uninstall", {}),
            ):
                self.assertEqual(self.client.post(path, json=body).status_code, 403, path)

    def test_stt_delete_unknown_is_404_and_known_goes_through_the_generic_delete(self) -> None:
        self.assertEqual(self.client.post("/api/voice/stt/delete", json={"key": "nope"}).status_code, 404)
        key = voice_runtime_disk_stt_key()
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None):
            body = self.client.post("/api/voice/stt/delete", json={"key": key}).json()
        self.assertIn("repo", body)
        self.assertIn(body["result"]["state"], ("deleted", "not_found"))

    def test_uninstall_when_not_installed_is_400(self) -> None:
        with mock.patch.object(voice_env, "VOICE_VENV_DIR", Path(self._tmp.name) / "no-venv"), \
                mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None):
            self.assertEqual(self.client.post("/api/voice/uninstall", json={}).status_code, 400)

    def test_runtime_lists_parts_and_memory(self) -> None:
        body = self.client.get("/api/voice/runtime").json()
        self.assertIn("parts", body)
        self.assertIn("inMemory", body)

    def test_models_carry_fit_and_kokoro(self) -> None:
        body = self.client.get("/api/voice/models").json()
        self.assertIn("kokoro", body)
        self.assertIn(body["kokoro"]["fit"]["status"], ("fits", "tight", "wont-fit", "unknown"))
        for row in body["models"]:
            self.assertIn(row["fit"]["status"], ("fits", "tight", "wont-fit", "unknown"))

    def test_library_voice_kind_filter(self) -> None:
        self.assertEqual(self.client.get("/api/voice/library", params={"voiceKind": "bogus"}).status_code, 400)
        self.assertEqual(self.client.get("/api/voice/library", params={"voiceKind": "preset"}).status_code, 200)
