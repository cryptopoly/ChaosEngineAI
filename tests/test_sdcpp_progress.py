"""sd.cpp step parsing and the image lane's progress lifecycle."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.sdcpp_progress import parse_step
from backend_service.progress import IMAGE_PROGRESS


class ParseStepTests(unittest.TestCase):
    def test_current_build_sampling_bar(self) -> None:
        line = "  |=========>                                      | 2/4 - 21.58s/it\x1b[K"
        self.assertEqual(parse_step(line), (2, 4))

    def test_fast_steps_print_iterations_per_second(self) -> None:
        self.assertEqual(parse_step("  |=====>      | 7/20 - 3.10it/s"), (7, 20))

    def test_older_formats(self) -> None:
        self.assertEqual(parse_step("[INFO] step 3/20 (1.2s)"), (3, 20))
        self.assertEqual(parse_step("[7/30] denoise"), (7, 30))

    def test_model_loading_bars_are_not_steps(self) -> None:
        self.assertIsNone(parse_step("  |#####                          | 19/219 - 250.07MB/s\x1b[K"))
        self.assertIsNone(parse_step("[INFO ] model_loader.cpp:1383 - loading tensors completed, taking 3.05s"))

    def test_unrelated_text(self) -> None:
        self.assertIsNone(parse_step(""))
        self.assertIsNone(parse_step("get_learned_condition completed, taking 4.72s"))


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class ImageProgressLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        IMAGE_PROGRESS.finish()

    def tearDown(self) -> None:
        IMAGE_PROGRESS.finish()

    def _config(self):
        from backend_service.image_runtime import ImageGenerationConfig

        return ImageGenerationConfig(
            modelId="m",
            modelName="FLUX test",
            repo="leejet/FLUX.1-schnell-gguf",
            prompt="p",
            negativePrompt="",
            width=512,
            height=512,
            steps=4,
            guidance=0.0,
            batchSize=1,
            seed=1,
            ggufRepo="leejet/FLUX.1-schnell-gguf",
            ggufFile="flux1-schnell-q4_k.gguf",
            runtime="sdcpp",
        )

    def test_a_run_moves_through_diffusing_to_decoding(self) -> None:
        from backend_service.sdcpp_image_runtime import SdCppImageEngine

        engine = SdCppImageEngine()
        IMAGE_PROGRESS.begin(run_label="t", total_steps=4)
        seen: list[tuple[str, int, int]] = []

        class _Proc:
            stdout = _FakeStdout([
                "get_learned_condition completed, taking 4.72s",
                "  |==>            | 1/4 - 27.63s/it",
                "  |=====>         | 2/4 - 21.58s/it",
                "[INFO ] image.cpp:554  - decoding 1 latents",
            ])

            def wait(self, timeout=None):
                return 0

        original_set_step = IMAGE_PROGRESS.set_step

        def spy(step, total=None):
            seen.append((IMAGE_PROGRESS.snapshot()["phase"], step, total or 0))
            return original_set_step(step, total=total)

        out = Path(tempfile.mkdtemp()) / "out.png"
        out.write_bytes(b"png")
        with mock.patch("backend_service.sdcpp_image_runtime.subprocess.Popen", return_value=_Proc()), \
             mock.patch.object(IMAGE_PROGRESS, "set_step", side_effect=spy):
            engine._run_subprocess(args=["sd"], config=self._config(), output_path=out)

        self.assertEqual(seen, [("diffusing", 1, 4), ("diffusing", 2, 4)])
        self.assertEqual(IMAGE_PROGRESS.snapshot()["phase"], "decoding")

    def test_generate_begins_the_run_and_always_finishes_it(self) -> None:
        from backend_service.sdcpp_image_runtime import SdCppImageEngine

        engine = SdCppImageEngine()
        phases: list[str] = []

        def fake_run(*, args, config, output_path):
            snapshot = IMAGE_PROGRESS.snapshot()
            phases.append(f"{snapshot['active']}:{snapshot['phase']}:{snapshot['totalSteps']}")
            return b"png"

        with mock.patch("backend_service.sdcpp_image_runtime._resolve_sd_binary", return_value=Path("/tmp/sd")), \
             mock.patch.object(SdCppImageEngine, "_resolve_gguf_path", return_value="/tmp/m.gguf"), \
             mock.patch.object(SdCppImageEngine, "_resolve_aux_paths", return_value={}), \
             mock.patch.object(SdCppImageEngine, "_run_subprocess", side_effect=fake_run):
            engine.generate(self._config())
        self.assertEqual(phases, ["True:loading:4"])
        self.assertFalse(IMAGE_PROGRESS.snapshot()["active"])

        with mock.patch("backend_service.sdcpp_image_runtime._resolve_sd_binary", return_value=Path("/tmp/sd")), \
             mock.patch.object(SdCppImageEngine, "_resolve_gguf_path", return_value="/tmp/m.gguf"), \
             mock.patch.object(SdCppImageEngine, "_resolve_aux_paths", return_value={}), \
             mock.patch.object(SdCppImageEngine, "_run_subprocess", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                engine.generate(self._config())
        self.assertFalse(IMAGE_PROGRESS.snapshot()["active"])


class FirstRunDownloadMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        IMAGE_PROGRESS.finish()
        IMAGE_PROGRESS.begin(run_label="t", total_steps=4)

    def tearDown(self) -> None:
        IMAGE_PROGRESS.finish()

    def test_an_uncached_file_says_it_is_downloading(self) -> None:
        from backend_service.sdcpp_image_runtime import _fetch_file

        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
             mock.patch("huggingface_hub.hf_hub_download", return_value="/cache/t5.gguf"):
            self.assertEqual(_fetch_file("o/r", "t5.gguf"), "/cache/t5.gguf")
        self.assertIn("Downloading t5.gguf", IMAGE_PROGRESS.snapshot()["message"])

    def test_a_cached_file_leaves_the_message_alone(self) -> None:
        from backend_service.sdcpp_image_runtime import _fetch_file

        IMAGE_PROGRESS.set_message("Loading model")
        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/t5.gguf"), \
             mock.patch("huggingface_hub.hf_hub_download", return_value="/cache/t5.gguf"):
            _fetch_file("o/r", "t5.gguf")
        self.assertEqual(IMAGE_PROGRESS.snapshot()["message"], "Loading model")


if __name__ == "__main__":
    unittest.main()
