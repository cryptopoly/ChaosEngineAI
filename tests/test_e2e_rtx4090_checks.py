"""The E2E checks added by the 2026-11-20 coverage audit (scripts/e2e_phases/rtx4090_pass.py), run against stand-ins.

The checks themselves run in scripts/e2e_test_suite.py against a live backend; here their own logic is held: the
stream parser, the skip and refusal rules, and the in-process Studio staging check.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from e2e_phases import rtx4090_pass as checks  # noqa: E402


def _suite(routes: dict[tuple[str, str], tuple[int, object]]) -> SimpleNamespace:
    """A stand-in for the suite module: ``_voice_http`` answers from ``routes``."""
    calls: list[tuple[str, str, object]] = []

    def voice_http(method, path, body=None, content_type=None, timeout=0.0):
        calls.append((method, path, json.loads(body) if body else None))
        code, payload = routes.get((method, path.split("?")[0]), (404, {"detail": "Not Found"}))
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return code, raw

    return SimpleNamespace(_voice_http=voice_http, calls=calls)


def _sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]
    return "\n\n".join(lines).encode()


class StreamToolCallTests(unittest.TestCase):
    def _run(self, raw: bytes):
        h = _suite({("POST", "/v1/chat/completions"): (200, raw)})
        h._cli_json = lambda *a, **k: (0, {"state": "loaded"}, "")
        h._cli = mock.Mock(return_value=(0, "", ""))
        cap = SimpleNamespace(gguf_available=True, local_gguf_files=[("org/tiny-GGUF", "/m/tiny.gguf")])
        result = checks.stream_tool_calls(h, cap)
        h._cli.assert_called_with("unload", timeout=60.0)  # the model is unloaded whatever the answer
        return result

    def test_a_streamed_call_passes(self):
        call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        status, _reason, detail = self._run(_sse(
            {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
        self.assertEqual(status, "pass")
        self.assertEqual(detail["calls"], ["get_weather"])

    def test_a_reply_in_words_is_a_skip_and_a_lost_call_fails(self):
        status, reason, _ = self._run(_sse({"choices": [{"delta": {"content": "Sunny."}, "finish_reason": "stop"}]}))
        self.assertEqual(status, "skip")
        self.assertIn("instead of calling the tool", reason)
        status, reason, _ = self._run(_sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
        self.assertEqual(status, "fail")

    def test_a_stream_without_done_fails(self):
        status, reason, _ = self._run(b'data: {"choices": []}\n\n')
        self.assertEqual(status, "fail")
        self.assertIn("[DONE]", reason)

    def test_no_gguf_skips_without_loading(self):
        h = _suite({})
        h._cli_json = mock.Mock()
        self.assertEqual(checks.stream_tool_calls(h, SimpleNamespace(gguf_available=True, local_gguf_files=[]))[0], "skip")
        h._cli_json.assert_not_called()


class RefusalCheckTests(unittest.TestCase):
    def test_nunchaku_wants_a_409_that_says_what_to_download(self):
        catalog = {"families": [{"variants": [{"id": "n1", "nunchakuRepo": "mit-han-lab/x", "availableLocally": False}]}]}
        ok = _suite({("GET", "/api/images/catalog"): (200, catalog),
                     ("POST", "/api/images/generate"): (409, {"detail": "n1 is not downloaded. Download it in Image Studio first"})})
        self.assertEqual(checks.nunchaku_refusal(ok)[0], "pass")
        placeholder = _suite({("GET", "/api/images/catalog"): (200, catalog), ("POST", "/api/images/generate"): (200, {})})
        self.assertEqual(checks.nunchaku_refusal(placeholder)[0], "fail")
        ready = {"families": [{"variants": [{"id": "n1", "nunchakuRepo": "x", "availableLocally": True}]}]}
        self.assertEqual(checks.nunchaku_refusal(_suite({("GET", "/api/images/catalog"): (200, ready)}))[0], "skip")

    def test_windows_install_refusal_never_calls_the_route_elsewhere(self):
        h = _suite({})
        with mock.patch.object(checks.sys, "platform", "linux"):
            self.assertEqual(checks.windows_install_refusal(h)[0], "skip")
        self.assertEqual(h.calls, [])
        answers = {("POST", "/api/setup/install-package"): (400, {"detail": "use WSL; triton has no Windows build"})}
        with mock.patch.object(checks.sys, "platform", "win32"):
            self.assertEqual(checks.windows_install_refusal(_suite(answers))[0], "pass")


class StudioPriceTests(unittest.TestCase):
    def _routes(self, small: float | None, big: float | None):
        answers = iter([
            {"model": "m", "size": "512x512", "estimatedSeconds": small},
            {"model": "m", "size": "1024x1024", "estimatedSeconds": big},
        ])
        h = _suite({("GET", "/api/chat/studio/tiers"): (200, {"models": [{"id": "m", "kind": "image"}]})})
        base = h._voice_http

        def voice_http(method, path, body=None, content_type=None, timeout=0.0):
            if path == "/api/chat/studio/price":
                args = json.loads(body)
                if args["tool"] == "make_music":
                    return 422, b'{"detail": []}'
                if "width" not in args["arguments"]:
                    return 200, json.dumps({k: None for k in ("model", "size", "frames", "fps", "steps", "estimatedSeconds", "peakGb")}).encode()
                return 200, json.dumps(next(answers)).encode()
            return base(method, path, body, content_type, timeout)

        h._voice_http = voice_http
        return h

    def test_a_bigger_render_must_cost_more(self):
        self.assertEqual(checks.studio_price(self._routes(10.0, 40.0))[0], "pass")
        self.assertEqual(checks.studio_price(self._routes(40.0, 10.0))[0], "fail")
        self.assertEqual(checks.studio_price(self._routes(None, None))[0], "pass")


class StagingAndVoiceTests(unittest.TestCase):
    def test_studio_staging_in_process(self):
        status, reason, detail = checks.studio_staging(SimpleNamespace())
        self.assertEqual(status, "pass", reason)
        self.assertGreaterEqual(detail["storyboardTools"], 5)

    def test_aligner_row_follows_the_gpu_rows(self):
        gpu = {"sttModels": [{"backend": "transformers-asr"}], "aligner": {
            "key": "qwen3-forced-aligner-0.6b", "id": "Qwen/Qwen3-ForcedAligner-0.6B-hf", "licence": "Apache-2.0",
            "sizeGb": 1.84, "installed": False}}
        self.assertEqual(checks.aligner_row(_suite({("GET", "/api/voice/runtime"): (200, gpu)}))[0], "pass")
        stray = {"sttModels": [{"backend": "mlx-whisper"}], "aligner": gpu["aligner"]}
        self.assertEqual(checks.aligner_row(_suite({("GET", "/api/voice/runtime"): (200, stray)}))[0], "fail")
        missing = {"sttModels": gpu["sttModels"], "aligner": None}
        self.assertEqual(checks.aligner_row(_suite({("GET", "/api/voice/runtime"): (200, missing)}))[0], "fail")

    def test_speech_rows_need_lanes(self):
        routes = {("GET", "/api/voice/runtime"): (200, {"lane": "portable"}),
                  ("GET", "/api/voice/clone/models"): (200, {"models": [{"key": "a", "lanes": ["mlx"]}, {"key": "b", "lanes": ["cuda"]}]})}
        self.assertEqual(checks.speech_rows_carry_lanes(_suite(routes))[0], "pass")
        routes[("GET", "/api/voice/clone/models")] = (200, {"models": [{"key": "a"}]})
        self.assertEqual(checks.speech_rows_carry_lanes(_suite(routes))[0], "fail")


if __name__ == "__main__":
    unittest.main()
