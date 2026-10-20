"""mflux image lane: family table, probe, command line, the subprocess run, the
setup endpoints, and the manager's refusal to fall back for repos that have no
diffusers pipeline."""

from __future__ import annotations

import dataclasses
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.image_runtime import (
    GeneratedImage,
    ImageGenerationConfig,
    ImageRuntimeManager,
    mflux_engine,
)
from backend_service.image_runtime.mflux_engine import MfluxImageEngine, mflux_family_for_repo
from backend_service.progress import IMAGE_PROGRESS, GenerationCancelled
from backend_service.routes.setup import mflux as setup_mflux
from backend_service.state import ChaosEngineState
from tests.test_setup_routes import TEST_API_TOKEN, _fake_system_snapshot


def _config(repo: str = "Qwen/Qwen-Image-2.1", **overrides) -> ImageGenerationConfig:
    base = ImageGenerationConfig(
        modelId="x", modelName="Test model", repo=repo, prompt="a fox", negativePrompt="blurry",
        width=1024, height=768, steps=12, guidance=3.5, batchSize=1, seed=7, runtime="mflux",
    )
    return dataclasses.replace(base, **overrides)


class FamilyTableTests(unittest.TestCase):
    def test_known_repos_map_to_their_command_and_model(self) -> None:
        self.assertEqual(mflux_engine._mflux_name_for_repo("black-forest-labs/FLUX.1-schnell"), "schnell")
        self.assertEqual(mflux_engine._mflux_name_for_repo("black-forest-labs/FLUX.1-dev"), "dev")
        qwen = mflux_family_for_repo("Qwen/Qwen-Image-2.1")
        self.assertEqual((qwen.command, qwen.model), ("mflux-generate-qwen-2.1", "qwen-image-2.1"))
        self.assertIsNone(mflux_engine._mflux_name_for_repo("stabilityai/stable-diffusion-3.5-medium"))

    def test_every_catalog_mflux_variant_has_a_command(self) -> None:
        from backend_service.catalog import IMAGE_MODEL_FAMILIES

        repos = [
            v["repo"] for f in IMAGE_MODEL_FAMILIES for v in f["variants"] if v.get("engine") == "mflux"
        ]
        self.assertGreaterEqual(len(repos), 4)
        for repo in repos:
            self.assertIsNotNone(mflux_family_for_repo(repo), repo)


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.venv = self.tmp / "mflux-venv"
        self.version_file = self.tmp / "bin" / "mflux.version"
        for name, value in (("MFLUX_VENV_DIR", self.venv), ("MFLUX_VERSION_FILE", self.version_file)):
            patcher = mock.patch.object(mflux_engine, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(mflux_engine, "_apple_silicon", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self) -> None:
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "mflux-generate").write_text("")
        self.version_file.parent.mkdir(parents=True)
        self.version_file.write_text("0.20.0\n2026-10-02T00:00:00Z\n0.20.0\n")

    def test_not_installed_says_where_to_install_it(self) -> None:
        probe = MfluxImageEngine().probe()
        self.assertFalse(probe["available"])
        self.assertIn("Image Studio", probe["reason"])

    def test_a_version_file_without_the_command_is_not_an_install(self) -> None:
        self.version_file.parent.mkdir(parents=True)
        self.version_file.write_text("0.20.0\n")
        self.assertFalse(MfluxImageEngine().probe()["available"])

    def test_installed(self) -> None:
        self._install()
        probe = MfluxImageEngine().probe()
        self.assertTrue(probe["available"])
        self.assertEqual(probe["version"], "0.20.0")

    def test_other_platforms_are_refused(self) -> None:
        self._install()
        with mock.patch.object(mflux_engine, "_apple_silicon", return_value=False):
            self.assertIn("Apple Silicon", MfluxImageEngine().probe()["reason"])


class CommandTests(unittest.TestCase):
    def _args(self, repo: str, **overrides) -> list[str]:
        engine = MfluxImageEngine()
        family = mflux_family_for_repo(repo)
        return engine._build_command(family, _config(repo, **overrides), seed=5, output_path=Path("/tmp/o.png"))

    def test_core_flags(self) -> None:
        args = self._args("black-forest-labs/FLUX.1-dev")
        self.assertTrue(args[0].endswith("mflux-generate"))
        for flag, value in (("--model", "dev"), ("--prompt", "a fox"), ("--steps", "12"), ("--width", "1024"),
                            ("--height", "768"), ("--seed", "5"), ("--guidance", "3.5")):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--no-metadata", args)
        self.assertNotIn("--quantize", args)

    def test_guidance_distilled_models_get_no_guidance_flag(self) -> None:
        self.assertNotIn("--guidance", self._args("black-forest-labs/FLUX.2-klein-4B"))
        self.assertNotIn("--guidance", self._args("Qwen/Qwen-Image-2.1"))

    def test_negative_prompt_only_where_the_model_takes_one(self) -> None:
        qwen = self._args("Qwen/Qwen-Image-2.1")
        self.assertEqual(qwen[qwen.index("--negative-prompt") + 1], "blurry")
        self.assertNotIn("--negative-prompt", self._args("black-forest-labs/FLUX.1-dev"))

    def test_quantize_comes_from_the_variant(self) -> None:
        args = self._args("Qwen/Qwen-Image-2.1", mfluxQuantize=8)
        self.assertEqual(args[args.index("--quantize") + 1], "8")


def _fake_command(directory: Path, body: str) -> Path:
    script = directory / "fake-mflux"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        IMAGE_PROGRESS.begin(total_steps=4)
        self.addCleanup(IMAGE_PROGRESS.finish)

    def test_streams_steps_and_reports_peak_memory(self) -> None:
        out = self.tmp / "o.png"
        script = _fake_command(self.tmp, f"""
            import sys, time
            for i in range(1, 5):
                sys.stdout.write("\\r  %d%%|###| %d/4 [00:0%d<00:01, 1.0s/it]" % (i * 25, i, i))
                sys.stdout.flush()
            sys.stdout.write("\\nPeak MLX memory: 17.95 GB\\n")
            open({str(out)!r}, "wb").write(b"png")
        """)
        peak = MfluxImageEngine()._run([str(script)], out)
        self.assertEqual(peak, 17.95)
        snapshot = IMAGE_PROGRESS.snapshot()
        self.assertEqual((snapshot["step"], snapshot["totalSteps"]), (4, 4))

    def test_a_failing_command_reports_its_output(self) -> None:
        script = _fake_command(self.tmp, """
            import sys
            print("boom: model not found")
            sys.exit(3)
        """)
        with self.assertRaises(RuntimeError) as caught:
            MfluxImageEngine()._run([str(script)], self.tmp / "o.png")
        self.assertIn("code 3", str(caught.exception))
        self.assertIn("boom: model not found", str(caught.exception))

    def test_a_command_that_writes_nothing_is_an_error(self) -> None:
        script = _fake_command(self.tmp, "print('done')\n")
        with self.assertRaises(RuntimeError):
            MfluxImageEngine()._run([str(script)], self.tmp / "o.png")

    def test_cancel_stops_the_command_and_raises_the_cancel_error(self) -> None:
        script = _fake_command(self.tmp, """
            import time
            print("1/4 [00:01<00:03, 1.0s/it]", flush=True)
            time.sleep(30)
        """)
        IMAGE_PROGRESS.request_cancel()
        with self.assertRaises(GenerationCancelled):
            MfluxImageEngine()._run([str(script)], self.tmp / "o.png")


class GenerateTests(unittest.TestCase):
    def test_generate_returns_one_image_per_seed_and_closes_progress(self) -> None:
        engine = MfluxImageEngine()
        with mock.patch.object(MfluxImageEngine, "probe", return_value={"available": True, "version": "0.20.0"}), \
                mock.patch.object(MfluxImageEngine, "_run", side_effect=lambda args, out: (out.write_bytes(b"png"), 18.0)[1]):
            images = engine.generate(_config("black-forest-labs/FLUX.2-klein-4B", batchSize=2))
        self.assertEqual([image.seed for image in images], [7, 8])
        self.assertTrue(all(image.runtimeLabel == "mflux (MLX native)" for image in images))
        self.assertIn("peak 18 GB", images[0].runtimeNote)
        self.assertFalse(IMAGE_PROGRESS.snapshot()["active"])

    def test_an_unsupported_repo_is_refused(self) -> None:
        with mock.patch.object(MfluxImageEngine, "probe", return_value={"available": True}):
            with self.assertRaises(RuntimeError):
                MfluxImageEngine().generate(_config("stabilityai/stable-diffusion-3.5-medium"))


class ManagerFallbackTests(unittest.TestCase):
    """A repo with no diffusers pipeline must surface its own error, not a
    placeholder picture."""

    def _manager(self, *, probe: dict, generate_error: Exception | None = None) -> ImageRuntimeManager:
        manager = ImageRuntimeManager()
        manager._mflux = mock.Mock()
        manager._mflux.probe.return_value = probe
        manager._mflux.generate.side_effect = generate_error or (lambda config: [mock.Mock(spec=GeneratedImage)])
        manager._diffusers = mock.Mock()
        manager._diffusers.probe.return_value = mock.Mock(realGenerationAvailable=True, to_dict=lambda: {})
        return manager

    def test_a_failed_qwen_2_1_run_is_not_handed_to_diffusers(self) -> None:
        manager = self._manager(probe={"available": True}, generate_error=RuntimeError("mflux exited with code 1."))
        with self.assertRaises(RuntimeError) as caught:
            manager.generate(_config("Qwen/Qwen-Image-2.1"))
        self.assertIn("mflux exited", str(caught.exception))
        manager._diffusers.generate.assert_not_called()

    def test_a_missing_engine_for_qwen_2_1_says_so(self) -> None:
        manager = self._manager(probe={"available": False, "reason": "mflux is not installed."})
        with self.assertRaises(RuntimeError) as caught:
            manager.generate(_config("Qwen/Qwen-Image-2.1"))
        self.assertIn("not installed", str(caught.exception))

    def test_cancellation_is_never_swallowed(self) -> None:
        manager = self._manager(probe={"available": True}, generate_error=GenerationCancelled("cancelled"))
        with self.assertRaises(GenerationCancelled):
            manager.generate(_config("black-forest-labs/FLUX.1-dev"))
        manager._diffusers.generate.assert_not_called()

    def test_flux_still_falls_back_to_diffusers(self) -> None:
        manager = self._manager(probe={"available": True}, generate_error=RuntimeError("boom"))
        manager._diffusers.generate.return_value = ["image"]
        images, status = manager.generate(_config("black-forest-labs/FLUX.1-dev"))
        self.assertEqual(images, ["image"])
        self.assertIn("mflux failed", status["message"])


class SetupRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.venv = self.tmp / "mflux-venv"
        self.version_file = self.tmp / "bin" / "mflux.version"
        for target, name, value in (
            (mflux_engine, "MFLUX_VENV_DIR", self.venv),
            (mflux_engine, "MFLUX_VERSION_FILE", self.version_file),
            (setup_mflux, "MFLUX_VENV_DIR", self.venv),
            (setup_mflux, "MFLUX_VERSION_FILE", self.version_file),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        job_patch = mock.patch.object(setup_mflux, "_JOB", setup_mflux._MfluxJobState())
        job_patch.start()
        self.addCleanup(job_patch.stop)
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=self.tmp / "settings.json",
            benchmarks_path=self.tmp / "benchmarks.json",
            chat_sessions_path=self.tmp / "chats.json",
        )
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def test_status_when_missing(self) -> None:
        status = self.client.get("/api/setup/mflux-status").json()
        self.assertFalse(status["installed"])
        self.assertIsNone(status["version"])
        self.assertIn("Qwen/Qwen-Image-2.1", status["repos"])

    def test_status_when_installed(self) -> None:
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "mflux-generate").write_text("")
        self.version_file.parent.mkdir(parents=True)
        self.version_file.write_text("0.20.0\n2026-10-02T00:00:00Z\n0.20.0\n")
        status = self.client.get("/api/setup/mflux-status").json()
        self.assertTrue(status["installed"])
        self.assertEqual(status["version"], "0.20.0")
        self.assertEqual(status["venvPath"], str(self.venv))

    def test_install_runs_the_script_and_tracks_its_phases(self) -> None:
        script = self.tmp / "install.sh"
        script.write_text("#!/bin/bash\necho PHASE:preflight\necho PHASE:installing\necho hello\necho OK\n")
        with mock.patch.object(setup_mflux, "_INSTALL_SCRIPT", script):
            self.client.post("/api/setup/install-mflux")
            for _ in range(100):
                state = self.client.get("/api/setup/install-mflux/status").json()
                if state["done"]:
                    break
                time.sleep(0.05)
        self.assertEqual(state["phase"], "done")
        self.assertIsNone(state["error"])
        self.assertEqual([a["phase"] for a in state["attempts"]], ["preflight", "installing"])

    def test_a_failing_script_surfaces_its_message(self) -> None:
        script = self.tmp / "install.sh"
        script.write_text("#!/bin/bash\necho PHASE:preflight\necho 'FAIL:needs arm64 python'\nexit 1\n")
        with mock.patch.object(setup_mflux, "_INSTALL_SCRIPT", script):
            self.client.post("/api/setup/install-mflux")
            for _ in range(100):
                state = self.client.get("/api/setup/install-mflux/status").json()
                if state["done"]:
                    break
                time.sleep(0.05)
        self.assertEqual(state["phase"], "error")
        self.assertEqual(state["error"], "needs arm64 python")

    def test_the_package_installer_refuses_mflux(self) -> None:
        response = self.client.post("/api/setup/install-package", json={"package": "mflux"})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertIn("own environment", detail["message"] if isinstance(detail, dict) else detail)


if __name__ == "__main__":
    unittest.main()
