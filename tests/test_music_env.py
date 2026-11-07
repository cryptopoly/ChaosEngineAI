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
        cases = [("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "unsupported"), ("Windows", "AMD64", "unsupported"),
                 ("Linux", "x86_64", "unsupported")]
        for system, machine, expected in cases:
            with mock.patch.object(music_env.platform, "system", return_value=system), \
                    mock.patch.object(music_env.platform, "machine", return_value=machine):
                self.assertEqual(music_env.lane(), expected)
                reason = music_env.lane_unavailable_reason()
                self.assertEqual(reason is None, expected == "mlx")
                if reason:
                    self.assertIn("not built yet", reason)

    def test_python_versions(self) -> None:
        self.assertTrue(music_env._python_supported((3, 11, 2)))
        self.assertTrue(music_env._python_supported((3, 12, 5)))
        self.assertFalse(music_env._python_supported((3, 13, 0)))
        self.assertFalse(music_env._python_supported((3, 10, 9)))

    def test_marker_needs_this_commit_this_venv_and_its_python(self) -> None:
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
        (self.data / "music-pack.json").write_text(json.dumps({**marker, "ref": "old"}))
        self.assertFalse(music_env.is_installed())
        (self.data / "music-pack.json").write_text("not json")
        self.assertFalse(music_env.is_installed())

    def test_worker_env_carries_the_spike_settings(self) -> None:
        env = music_env.worker_env()
        self.assertEqual(env["ACESTEP_MLX_VAE_CHUNK"], "256")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["ACESTEP_LM_BACKEND"], "mlx")
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


class InstallJobTests(_EnvBox):
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
        self.assertEqual(commands[0][1], "-c")

    def test_off_apple_silicon_the_install_explains_and_stops(self) -> None:
        with self._lane("unsupported"), mock.patch.object(music_env, "_run") as run:
            music_env.start_install()
            status = self._wait()
        self.assertEqual(status["phase"], "error")
        self.assertIn("Apple Silicon", status["error"])
        run.assert_not_called()

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

    def test_status_shape(self) -> None:
        status = music_env.job_status()
        for key in ("phase", "phaseLabel", "phaseIndex", "phaseTotal", "percent", "log", "running", "error"):
            self.assertIn(key, status)
        self.assertEqual(status["phaseTotal"], 5)


if __name__ == "__main__":
    unittest.main()
