"""v0.10.0 closing fixes: MLX reply timeout, picture check on path import."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.inference.mlx_engine import MLXWorkerEngine
from backend_service.storyboard import media


class MlxReplyTimeoutTests(unittest.TestCase):
    def _engine(self, alive_after: bool):
        engine = MLXWorkerEngine(mock.Mock(pythonExecutable="python"))
        engine.loaded_model = mock.Mock()
        worker = mock.Mock()
        worker.timeout = 120.0
        worker.is_alive.side_effect = [True, alive_after]  # the entry check, then the check after a failure
        engine.worker = worker
        return engine, worker

    def _generate(self, engine, max_tokens):
        return engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=max_tokens, temperature=0.0)

    def test_the_wait_grows_with_the_reply_the_request_allows(self):
        engine, worker = self._engine(True)
        worker.request.return_value = {"text": "ok"}
        self._generate(engine, 2048)
        self.assertEqual(worker.request.call_args.kwargs["timeout"], 60.0 + 2048 * 0.5)

    def test_a_short_reply_keeps_the_default_wait_and_a_huge_one_is_capped(self):
        engine, worker = self._engine(True)
        worker.request.return_value = {"text": "ok"}
        self._generate(engine, 16)
        self.assertEqual(worker.request.call_args.kwargs["timeout"], 120.0)
        engine, worker = self._engine(True)
        worker.request.return_value = {"text": "ok"}
        self._generate(engine, 1_000_000)
        self.assertEqual(worker.request.call_args.kwargs["timeout"], 1800.0)

    def test_a_dead_worker_after_a_failed_request_clears_the_loaded_model(self):
        engine, worker = self._engine(False)
        worker.request.side_effect = RuntimeError("Timed out waiting for the MLX worker after 120s.")
        with self.assertRaises(RuntimeError):
            self._generate(engine, 100)
        self.assertIsNone(engine.loaded_model)


class PictureCheckTests(unittest.TestCase):
    def test_a_text_file_named_png_is_refused_and_real_pictures_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "logo.png"
            fake.write_text("not a picture")
            with self.assertRaises(media.MediaError) as caught:
                media.resolve_source(str(fake), {"image"})
            self.assertIn("Not a picture", str(caught.exception))
            png = Path(tmp) / "real.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
            jpg = Path(tmp) / "real.jpg"
            jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 16)
            webp = Path(tmp) / "real.webp"
            webp.write_bytes(b"RIFF\0\0\0\0WEBP" + b"\0" * 8)
            for path in (png, jpg, webp):
                self.assertEqual(media.resolve_source(str(path), {"image"}), path.resolve())


if __name__ == "__main__":
    unittest.main()
