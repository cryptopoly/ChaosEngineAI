"""The music venv helper with fakes: lane, marker, worker environment, weights mirror and the install job.

No real install runs: ``_run`` and the verify step are replaced, so the job's order, resume, adoption of a
working venv, cancel and error paths are checked without pip or the network.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_env


class _EnvBox(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "music"
        self.venv = root / "music-venv"
        self.patches = [
            mock.patch.object(music_env, "MUSIC_DATA_DIR", self.data),
            mock.patch.object(music_env, "MUSIC_VENV_DIR", self.venv),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()


class LaneAndMarkerTests(_EnvBox):
    def test_lane(self) -> None:
        cases = [("Darwin", "arm64", True, "mlx"), ("Darwin", "x86_64", True, "unsupported"),
                 ("Windows", "AMD64", True, "cuda"), ("Linux", "x86_64", True, "cuda"),
                 ("Windows", "AMD64", False, "unsupported"), ("Linux", "x86_64", False, "unsupported"),
                 ("Linux", "aarch64", True, "unsupported")]  # GB10 / GH200: the cu130 lane is not built
        for system, machine, nvidia, expected in cases:
            with mock.patch.object(music_env.platform, "system", return_value=system), \
                    mock.patch.object(music_env.platform, "machine", return_value=machine), \
                    mock.patch.object(music_env, "_nvidia_present", return_value=nvidia):
                self.assertEqual(music_env.lane(), expected, (system, machine, nvidia))
                reason = music_env.lane_unavailable_reason()
                self.assertEqual(reason is None, expected in ("mlx", "cuda"))
                self.assertEqual(music_env.lane_unavailable_code() is None, expected in ("mlx", "cuda"))
                if reason:
                    self.assertEqual(reason, music_env.UNSUPPORTED_PLATFORM_REASON)
                    self.assertEqual(music_env.lane_unavailable_code(), "music.unsupportedPlatform")
                    self.assertIn("Apple Silicon", reason)
                    self.assertIn("NVIDIA GPU", reason)
                    self.assertIn("nothing is installed or downloaded", reason)

    def test_install_is_refused_without_a_lane_and_starts_nothing(self) -> None:
        for system, machine in (("Windows", "AMD64"), ("Linux", "x86_64")):
            with mock.patch.object(music_env.platform, "system", return_value=system), \
                    mock.patch.object(music_env.platform, "machine", return_value=machine), \
                    mock.patch.object(music_env, "_nvidia_present", return_value=False), \
                    mock.patch.object(music_env.threading, "Thread") as thread, \
                    mock.patch.object(music_env, "_run") as run:
                job = music_env.start_install()
            self.assertEqual((job["phase"], job["running"], job["error"]), ("error", False, music_env.UNSUPPORTED_PLATFORM_REASON))
            thread.assert_not_called()
            run.assert_not_called()
            self.assertFalse(self.venv.exists())
            self.assertFalse(self.data.exists())

    def test_python_versions(self) -> None:
        self.assertTrue(music_env._python_supported((3, 11, 2)))
        self.assertTrue(music_env._python_supported((3, 12, 5)))
        self.assertFalse(music_env._python_supported((3, 13, 0)))
        self.assertFalse(music_env._python_supported((3, 10, 9)))

    def test_marker_needs_this_commit_this_venv_and_its_python(self) -> None:
        lane = mock.patch.object(music_env, "lane", return_value="mlx")
        lane.start()
        self.addCleanup(lane.stop)
        self.assertFalse(music_env.is_installed())
        python = music_env.venv_python()
        python.parent.mkdir(parents=True)
        python.write_text("")
        music_env._write_marker({"torch": "2.10.0", "python": "3.12.5"})
        self.assertTrue(music_env.is_installed())
        marker = music_env.read_marker()
        self.assertEqual((marker["ref"], marker["python"], marker["venv"]), (music_env.ACE_REF, "3.12.5", str(self.venv)))
        with mock.patch.object(music_env, "MUSIC_VENV_DIR", self.venv.with_name("other")):
            self.assertFalse(music_env.is_installed())
        # A pack built for one lane is not the other's (an MLX marker without ``lane`` predates the CUDA lane).
        with mock.patch.object(music_env, "lane", return_value="mlx"):
            (self.data / "music-pack.json").write_text(json.dumps({k: v for k, v in marker.items() if k != "lane"}))
            self.assertTrue(music_env.is_installed())
        with mock.patch.object(music_env, "lane", return_value="cuda"):
            self.assertFalse(music_env.is_installed())
            (self.data / "music-pack.json").write_text(json.dumps({**marker, "lane": "cuda"}))
            self.assertTrue(music_env.is_installed())
        (self.data / "music-pack.json").write_text(json.dumps(marker))
        (self.data / "music-pack.json").write_text(json.dumps({**marker, "ref": "old"}))
        self.assertFalse(music_env.is_installed())
        (self.data / "music-pack.json").write_text("not json")
        self.assertFalse(music_env.is_installed())

    def test_worker_env_carries_the_spike_settings(self) -> None:
        with mock.patch.object(music_env, "lane", return_value="cuda"):
            cuda = music_env.worker_env()
        self.assertEqual((cuda["ACESTEP_LM_BACKEND"], cuda["CHAOSENGINE_MUSIC_LANE"]), ("pt", "cuda"))
        with mock.patch.object(music_env, "lane", return_value="mlx"):
            env = music_env.worker_env()
        self.assertEqual(env["ACESTEP_MLX_VAE_CHUNK"], "256")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["ACESTEP_LM_BACKEND"], "mlx")
        self.assertEqual(env["CHAOSENGINE_MUSIC_LANE"], "mlx")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["ACESTEP_CHECKPOINTS_DIR"], str(self.data / "checkpoints"))
        self.assertEqual(env["CHAOSENGINE_MUSIC_DATA"], str(self.data))
        self.assertEqual(env["CHAOSENGINE_MUSIC_ENGINE_VERSION"], music_env.ACE_REF[:7])

    def test_requirements_file_is_the_pinned_export(self) -> None:
        text = music_env.REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn(music_env.ACE_REF, text)
        self.assertIn("\ntorch==2.10.0 ;", text)
        self.assertIn("\nmlx==0.30.6 ;", text)
        self.assertNotIn("\n./", text)
        self.assertNotIn("\npytorch-wavelets", text)  # D-37: dropped on purpose (licence)
        self.assertIn("\npywavelets==", text)
        # FU-139: the CUDA lane's torch pins are local versions found only on the PyTorch CUDA index; Windows on
        # Python 3.12 gets no flash-attn or triton line (both are python < 3.12 in the lock).
        self.assertIn("\ntorch==2.7.1+cu128 ; sys_platform == 'win32'", text)
        self.assertIn("\ntorchaudio==2.7.1+cu128 ; sys_platform == 'win32'", text)
        for line in text.splitlines():
            if line.startswith(("flash-attn", "triton-windows")):
                self.assertIn("python_full_version < '3.12'", line)

    def test_cuda_install_command_adds_the_pytorch_index(self) -> None:
        with mock.patch.object(music_env, "lane", return_value="cuda"):
            command = music_env.requirements_command()
        self.assertEqual(command[-2:], ["--extra-index-url", "https://download.pytorch.org/whl/cu128"])
        with mock.patch.object(music_env, "lane", return_value="mlx"):
            self.assertNotIn("--extra-index-url", music_env.requirements_command())


class MirrorTests(_EnvBox):
    def test_large_files_are_linked_small_ones_copied_hidden_skipped(self) -> None:
        source = Path(self.tmp.name) / "snapshot"
        (source / "dit").mkdir(parents=True)
        (source / ".cache").mkdir()
        big = source / "dit" / "model.safetensors"
        with big.open("wb") as handle:
            handle.truncate(music_env._LINK_MIN_BYTES)
        (source / "dit" / "modeling.py").write_text("code")
        (source / ".cache" / "x.metadata").write_text("hidden")
        target = Path(self.tmp.name) / "checkpoints"
        self.assertEqual(music_env.mirror_tree(source, target), 2)
        self.assertTrue((target / "dit" / "model.safetensors").is_symlink())
        copied = target / "dit" / "modeling.py"
        self.assertFalse(copied.is_symlink())
        self.assertFalse((target / ".cache").exists())
        # The library rewrites its model code in the layout: that must never reach the source.
        copied.write_text("synced by the library")
        self.assertEqual((source / "dit" / "modeling.py").read_text(), "code")
        # A second mirror keeps the rewritten copy.
        self.assertEqual(music_env.mirror_tree(source, target), 0)
        self.assertEqual(copied.read_text(), "synced by the library")

    def _snapshot(self, name: str, payload: bytes) -> Path:
        source = Path(self.tmp.name) / "hub" / name
        source.mkdir(parents=True)
        big = source / "model.safetensors"
        with big.open("wb") as handle:
            handle.write(payload)
            handle.truncate(music_env._LINK_MIN_BYTES)
        (source / "config.json").write_text(name)
        return source

    def _variant(self, revision: str) -> dict:
        return {"repo": "ACE-Step/test", "revision": revision, "checkpointSubdir": "dit-model", "weightFolders": ["dit-model"]}

    def test_a_revision_bump_re_mirrors_and_is_recorded(self) -> None:
        from backend_service.helpers.music import weights_in_checkpoints

        old, new = self._snapshot("rev-old", b"old"), self._snapshot("rev-new", b"new")
        snapshots = {"rev-old": old, "rev-new": new}
        with mock.patch.object(music_env, "hf_snapshot", side_effect=lambda repo, rev: snapshots.get(rev)):
            music_env.prepare_checkpoints([self._variant("rev-old")])
            link = self.data / "checkpoints" / "dit-model" / "model.safetensors"
            self.assertEqual(link.resolve(), (old / "model.safetensors").resolve())
            self.assertEqual(music_env.read_mirror_manifest(), {"ACE-Step/test": "rev-old"})
            # The catalog pins a new revision that is not downloaded: the old layout must not count as ready.
            self.assertFalse(weights_in_checkpoints(self._variant("rev-new")))
            music_env.prepare_checkpoints([self._variant("rev-new")])
            self.assertEqual(link.resolve(), (new / "model.safetensors").resolve())
            self.assertEqual((link.parent / "config.json").read_text(), "rev-new")
            self.assertEqual(music_env.read_mirror_manifest(), {"ACE-Step/test": "rev-new"})
            self.assertTrue(weights_in_checkpoints(self._variant("rev-new")))

    def test_a_dangling_link_is_repaired(self) -> None:
        source = self._snapshot("rev-a", b"a")
        with mock.patch.object(music_env, "hf_snapshot", return_value=source):
            music_env.prepare_checkpoints([self._variant("rev-a")])
        link = self.data / "checkpoints" / "dit-model" / "model.safetensors"
        # The cache is cleared outside the app and downloaded again to another place.
        moved = source.with_name("rev-a-again")
        source.rename(moved)
        self.assertFalse(link.exists())
        self.assertTrue(link.is_symlink())
        with mock.patch.object(music_env, "hf_snapshot", return_value=moved):
            music_env.prepare_checkpoints([self._variant("rev-a")])
        self.assertTrue(link.exists())
        self.assertEqual(link.resolve(), (moved / "model.safetensors").resolve())


    def test_without_link_rights_the_copy_is_explained_first_then_done(self) -> None:
        """FU-276: no symlink rights and the cache on another drive meant a silent 10 to 20 GB copy."""
        from backend_service import music_runtime

        source = self._snapshot("rev-c", b"c")
        no_link = OSError(1314, "A required privilege is not held by the client")
        target = self.data / "checkpoints" / "dit-model" / "model.safetensors"
        with mock.patch.object(music_env, "hf_snapshot", return_value=source),              mock.patch.object(music_env.os, "symlink", side_effect=no_link),              mock.patch.object(music_env.os, "link", side_effect=OSError(18, "Invalid cross-device link")):
            # The first attempt (through the music runtime, which the routes answer with a 409) explains and stops.
            with mock.patch.object(music_runtime, "readiness_error", return_value=None),                  mock.patch("backend_service.catalog.music_models.music_variants", return_value=[self._variant("rev-c")]):
                with self.assertRaises(music_runtime.MusicNotReady) as notice:
                    music_runtime._prepare({"name": "test", "engine": "ace-step"})
            self.assertIsInstance(notice.exception.__cause__, music_env.WeightCopyNotice)
            message = str(notice.exception)
            self.assertIn("1 weight file(s)", message)
            self.assertIn("0.0 GB", message)  # the 16 MB test file
            self.assertIn("Developer Mode", message)
            self.assertIn("copy them now", message)
            self.assertEqual(notice.exception.fields["fileCount"], 1)
            self.assertTrue(notice.exception.fields["needsCopy"])
            self.assertFalse(target.exists(), "nothing large is copied before the user was told")
            self.assertEqual((target.parent / "config.json").read_text(), "rev-c", "small files are still copied")
            # Trying again goes ahead and copies.
            music_env.prepare_checkpoints([self._variant("rev-c")])
        self.assertTrue(target.is_file())
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.stat().st_size, music_env._LINK_MIN_BYTES)
        self.assertFalse((self.data / "checkpoints" / ".copy-notice.json").exists())

    def test_the_drives_are_named_when_they_differ(self) -> None:
        with mock.patch.object(music_env, "_drive", side_effect=lambda path: "E:" if "hub" in str(path) else "C:"):
            source = self._snapshot("rev-d", b"d")
            message = music_env._copy_notice_message([(source / "model.safetensors", Path("x"))])
        self.assertIn("the model cache is on E:, the music folder on C:", message)

    def test_links_still_need_no_notice(self) -> None:
        source = self._snapshot("rev-e", b"e")
        with mock.patch.object(music_env, "hf_snapshot", return_value=source):
            music_env.prepare_checkpoints([self._variant("rev-e")])
        self.assertTrue((self.data / "checkpoints" / "dit-model" / "model.safetensors").is_symlink())

class InstallJobTests(_EnvBox):
    def setUp(self) -> None:
        super().setUp()
        host = mock.patch.object(music_env, "lane", return_value="mlx")  # never this machine's own lane
        host.start()
        self.addCleanup(host.stop)

    def _wait(self) -> dict:
        deadline = time.monotonic() + 10
        while music_env.job_status()["running"] and time.monotonic() < deadline:
            time.sleep(0.02)
        return music_env.job_status()

    def _lane(self, name: str = "mlx"):
        return mock.patch.object(music_env, "lane", return_value=name)

    def test_fresh_install_runs_every_step_in_order(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> str:
            commands.append(command)
            if command[1:3] == ["-m", "venv"]:
                python = music_env.venv_python()
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return '{"torch": "2.10.0", "mlx": "0.30.6", "python": "3.12.5"}'

        with self._lane(), mock.patch.object(music_env, "_run", side_effect=fake_run), \
                mock.patch("backend_service.voice_env._base_python", return_value="/usr/bin/python3"):
            music_env.start_install()
            status = self._wait()
        self.assertEqual(status["phase"], "done", status)
        joined = [" ".join(c) for c in commands]
        self.assertIn("-m venv", joined[0])
        self.assertTrue(any("-r " + str(music_env.REQUIREMENTS) in c for c in joined))
        self.assertTrue(any("--no-deps " + music_env.ACE_ARCHIVE in c for c in joined))
        self.assertIn(music_env.ACE_REF, music_env.ACE_ARCHIVE)
        self.assertTrue(music_env.is_installed())
        for folder in ("checkpoints", "library", "tmp"):
            self.assertTrue((self.data / folder).is_dir())

    def test_a_working_venv_is_adopted_without_installing(self) -> None:
        python = music_env.venv_python()
        python.parent.mkdir(parents=True)
        python.write_text("")
        commands: list[list[str]] = []
        with self._lane(), mock.patch.object(music_env, "_run", side_effect=lambda c: commands.append(c) or '{"torch": "x"}'):
            music_env.start_install()
            status = self._wait()
        self.assertEqual(status["phase"], "done")
        self.assertEqual(len(commands), 1)  # the import check only: no pip, nothing written into the venv
        self.assertEqual(commands[0][2], "-c")

    def test_off_apple_silicon_the_install_explains_and_stops(self) -> None:
        with self._lane("unsupported"), mock.patch.object(music_env, "_run") as run:
            music_env.start_install()
            status = self._wait()
        self.assertEqual(status["phase"], "error")
        self.assertIn("Apple Silicon", status["error"])
        run.assert_not_called()

    def test_cuda_install_uses_the_cuda_index_and_check(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> str:
            commands.append(command)
            if command[1:3] == ["-m", "venv"]:
                python = music_env.venv_python()
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return '{"torch": "2.7.1+cu128", "cuda": "12.8", "gpu": "NVIDIA GeForce RTX 4090", "python": "3.12.10"}'

        with self._lane("cuda"), mock.patch.object(music_env, "_run", side_effect=fake_run), \
                mock.patch("backend_service.voice_env._base_python", return_value="python"):
            music_env.start_install()
            status = self._wait()
            self.assertEqual(status["phase"], "done", status)
            self.assertTrue(music_env.is_installed())
        pip = next(c for c in commands if "-r" in c)
        self.assertEqual(pip[-2:], ["--extra-index-url", music_env.CUDA_TORCH_INDEX])
        check = commands[-1]
        self.assertIn("torch.cuda.is_available()", check[-1])
        self.assertNotIn("mlx", check[-1])
        marker = music_env.read_marker()
        self.assertEqual((marker["lane"], marker["gpu"]), ("cuda", "NVIDIA GeForce RTX 4090"))

    def test_cancel_and_failure(self) -> None:
        def cancelled(_command: list[str]) -> str:
            raise InterruptedError("Install cancelled.")

        with self._lane(), mock.patch.object(music_env, "_run", side_effect=cancelled), \
                mock.patch("backend_service.voice_env._base_python", return_value="/usr/bin/python3"):
            music_env.start_install()
            self.assertEqual(self._wait()["phase"], "cancelled")
        with self._lane(), mock.patch.object(music_env, "_run", side_effect=RuntimeError("pip exited with code 1")), \
                mock.patch("backend_service.voice_env._base_python", return_value="/usr/bin/python3"):
            music_env.start_install()
            status = self._wait()
        self.assertEqual((status["phase"], status["error"]), ("error", "pip exited with code 1"))
        self.assertFalse(music_env.is_installed())

    def test_a_folder_that_is_not_a_venv_is_never_deleted(self) -> None:
        self.venv.mkdir(parents=True)
        (self.venv / "notes.txt").write_text("keep me")
        with self._lane(), mock.patch.object(music_env, "_run") as run:
            music_env.start_install()
            status = self._wait()
        self.assertEqual(status["phase"], "error")
        self.assertIn("not a Python environment", status["error"])
        self.assertEqual((self.venv / "notes.txt").read_text(), "keep me")
        run.assert_not_called()

    def test_a_broken_venv_is_rebuilt(self) -> None:
        self.venv.mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text("home = /usr/bin")
        (self.venv / "lib").mkdir()

        def fake_run(command: list[str]) -> str:
            if command[1:3] == ["-m", "venv"]:
                self.assertFalse((self.venv / "lib").exists())  # the broken attempt was cleared first
                python = music_env.venv_python()
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return '{"torch": "2.10.0"}'

        with self._lane(), mock.patch.object(music_env, "_run", side_effect=fake_run), \
                mock.patch("backend_service.voice_env._base_python", return_value="/usr/bin/python3"):
            music_env.start_install()
            self.assertEqual(self._wait()["phase"], "done")

    def test_status_shape(self) -> None:
        status = music_env.job_status()
        for key in ("phase", "phaseLabel", "phaseIndex", "phaseTotal", "percent", "log", "running", "error"):
            self.assertIn(key, status)
        self.assertEqual(status["phaseTotal"], 5)


if __name__ == "__main__":
    unittest.main()
