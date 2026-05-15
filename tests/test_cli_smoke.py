"""Smoke tests for scripts/chaosengine-cli.

Exercises argument parsing + happy-path HTTP shape against a mocked server
spun up via ``urllib`` monkeypatch — no real backend boot. The CLI itself
uses only stdlib (urllib + argparse), so we can import it as a module and
drive ``main([...])`` directly.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


_CLI_PATH = Path(__file__).parent.parent / "scripts" / "chaosengine-cli"


def _load_cli_module():
    # The CLI is shebang-only (no .py extension), so importlib's
    # extension-based detection misses it — use SourceFileLoader directly.
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader("chaosengine_cli", str(_CLI_PATH))
    spec = importlib.util.spec_from_loader("chaosengine_cli", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _load_cli_module()


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        for line in self._body.splitlines(keepends=True):
            yield line


def _mock_urlopen(responses: dict[str, _FakeResp]):
    """Return a urlopen replacement that maps URL → response."""

    def _opener(req, timeout=None):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else req
        if url in responses:
            return responses[url]
        # Wildcard fallback for path-only match
        for key, resp in responses.items():
            if url.endswith(key):
                return resp
        raise AssertionError(f"unexpected URL: {url}")

    return _opener


class CLIParserTests(unittest.TestCase):
    def test_parser_supports_all_subcommands(self) -> None:
        parser = cli.build_parser()
        for cmd in [
            ["serve"],
            ["status"],
            ["load", "Qwen/Qwen3.5-7B"],
            ["unload"],
            ["unload", "some/ref"],
            ["prompt", "hi"],
            ["bench", "Qwen/Qwen3.5-7B"],
            ["mtplx-install"],
            ["mtplx-install", "--wait"],
            ["mtplx-status"],
        ]:
            with self.subTest(cmd=cmd):
                args = parser.parse_args(cmd)
                self.assertTrue(callable(args.func))

    def test_load_defaults(self) -> None:
        args = cli.build_parser().parse_args(["load", "foo/bar"])
        self.assertEqual(args.ref, "foo/bar")
        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.context, 8192)
        self.assertEqual(args.cache_strategy, "native")
        self.assertFalse(args.spec)

    def test_load_with_overrides(self) -> None:
        args = cli.build_parser().parse_args([
            "load", "foo/bar", "--backend", "mlx", "--spec",
            "--context", "32768", "--cache-bits", "4",
        ])
        self.assertEqual(args.backend, "mlx")
        self.assertEqual(args.context, 32768)
        self.assertEqual(args.cache_bits, 4)
        self.assertTrue(args.spec)


class CLIStatusTests(unittest.TestCase):
    def test_status_extracts_runtime_and_system_keys(self) -> None:
        workspace_body = {
            "runtime": {
                "state": "loaded",
                "engine": "mtplx",
                "loadedModel": {"ref": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"},
                "runtimeNote": "MTPLX MTP speculative decoding active",
            },
            "system": {
                "platform": "Darwin",
                "arch": "arm64",
                "totalMemoryGb": 64.0,
                "availableMemoryGb": 32.0,
                "mtplx": {"available": True, "supportedModels": ["Qwen/Qwen3.6-27B"]},
                "dflash": {"available": False, "supportedModels": []},
            },
            "library": [{"name": "foo"}, {"name": "bar"}],
        }
        resp = _FakeResp(json.dumps(workspace_body).encode("utf-8"))
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/api/workspace": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["status"])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["runtime"]["state"], "loaded")
        self.assertEqual(payload["runtime"]["engine"], "mtplx")
        self.assertEqual(payload["libraryCount"], 2)
        self.assertTrue(payload["system"]["mtplx"]["available"])


class CLILoadTests(unittest.TestCase):
    def test_load_sends_expected_payload(self) -> None:
        captured: dict[str, dict] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({
                "runtime": {
                    "state": "loaded",
                    "engine": "mtplx",
                    "loadedModel": {
                        "ref": "Qwen/Qwen3.5-7B",
                        "path": "/Users/dan/models/qwen",
                        "speculativeDecoding": True,
                    },
                    "runtimeNote": "ok",
                },
            }).encode("utf-8"))

        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main([
                    "load", "Qwen/Qwen3.5-7B",
                    "--backend", "mlx", "--spec", "--context", "16384",
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["url"], "http://127.0.0.1:8876/api/models/load")
        self.assertEqual(captured["body"]["modelRef"], "Qwen/Qwen3.5-7B")
        self.assertEqual(captured["body"]["backend"], "mlx")
        self.assertTrue(captured["body"]["speculativeDecoding"])
        self.assertEqual(captured["body"]["contextTokens"], 16384)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["state"], "loaded")
        self.assertEqual(payload["engine"], "mtplx")

    def test_load_propagates_http_error(self) -> None:
        import urllib.error

        def _opener(req, timeout=None):  # noqa: ARG001
            raise urllib.error.HTTPError(
                url=req.full_url, code=500,
                msg="boom", hdrs=None, fp=io.BytesIO(b'{"detail":"no MLX"}'),
            )

        err = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stderr", err):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(["load", "Qwen/Qwen3.5-7B"])
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("load failed", err.getvalue())


class CLIPromptTests(unittest.TestCase):
    def test_prompt_non_streaming_prints_text_and_metrics(self) -> None:
        body = {
            "text": "Hello back!",
            "tokS": 42.5,
            "promptTokens": 5,
            "completionTokens": 3,
            "responseSeconds": 0.07,
            "runtimeNote": "MTPLX active",
        }
        resp = _FakeResp(json.dumps(body).encode("utf-8"))
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/api/chat/generate": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["prompt", "hi", "--metrics"])
        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("Hello back!", output)
        # Metrics JSON appended after the text
        json_start = output.find("{")
        metrics = json.loads(output[json_start:])
        self.assertEqual(metrics["tokS"], 42.5)
        self.assertEqual(metrics["completionTokens"], 3)
        self.assertIn("MTPLX", metrics["runtimeNote"])

    def test_prompt_streaming_assembles_tokens(self) -> None:
        sse = (
            b"data: " + json.dumps({"phase": "generating"}).encode() + b"\n\n"
            b"data: " + json.dumps({"token": "Hello"}).encode() + b"\n\n"
            b"data: " + json.dumps({"token": " world"}).encode() + b"\n\n"
            b"data: " + json.dumps({"done": True, "tokS": 30.1, "completionTokens": 2}).encode() + b"\n\n"
        )
        resp = _FakeResp(sse)
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/api/chat/generate/stream": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["prompt", "hi", "--stream", "--metrics"])
        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("Hello world", output)
        metrics = json.loads(output[output.find("{"):])
        self.assertEqual(metrics["tokS"], 30.1)
        self.assertEqual(metrics["completionTokens"], 2)


class CLIUnloadTests(unittest.TestCase):
    def test_unload_no_ref(self) -> None:
        captured: dict[str, bytes | None] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["data"] = req.data
            return _FakeResp(json.dumps({"runtime": {"state": "idle"}}).encode())

        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["unload"])
        self.assertEqual(rc, 0)
        self.assertIsNone(captured["data"])
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["state"], "idle")

    def test_unload_with_ref_sends_body(self) -> None:
        captured: dict[str, dict] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"runtime": {"state": "idle"}}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["unload", "some/ref"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"ref": "some/ref"})


class CLIMtplxTests(unittest.TestCase):
    def test_mtplx_status_passthrough(self) -> None:
        body = {"installed": True, "version": "0.3.5", "venvPath": "/Users/x/.chaosengine/mtplx-venv"}
        resp = _FakeResp(json.dumps(body).encode("utf-8"))
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/api/setup/mtplx-status": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["mtplx-status"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["version"], "0.3.5")

    def test_mtplx_install_no_wait_returns_initial_payload(self) -> None:
        body = {"id": "mtplx-install", "phase": "preflight", "done": False}
        resp = _FakeResp(json.dumps(body).encode("utf-8"))
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/api/setup/install-mtplx": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["mtplx-install"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["phase"], "preflight")


if __name__ == "__main__":
    unittest.main()
