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
from typing import Any
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
            ["health"],
            ["runtime"],
            ["load", "Qwen/Qwen3.5-7B"],
            ["unload"],
            ["unload", "some/ref"],
            ["prompt", "hi"],
            ["bench", "Qwen/Qwen3.5-7B"],
            ["mtplx-install"],
            ["mtplx-install", "--wait"],
            ["mtplx-status"],
            ["call", "GET", "/api/health"],
            ["call", "POST", "/api/models/load", "--body", "{}"],
            ["routes"],
            ["routes", "--filter", "/api/images"],
            ["openapi"],
            ["search", "qwen"],
            ["hub-search", "qwen"],
            ["hub-files", "Qwen/Qwen3.6-27B"],
            ["list-weights", "/some/path"],
            ["download", "foo/bar"],
            ["download-status"],
            ["download-cancel", "foo/bar"],
            ["download-delete", "foo/bar"],
            ["convert", "--output", "/tmp/out"],
            ["reveal", "/some/path"],
            ["delete-model", "/some/path"],
            ["quantized-variants", "foo/bar"],
            ["chat-cancel", "session-1"],
            ["session-create"],
            ["session-delete", "s1"],
            ["session-fork", "s1", "--at", "3"],
            ["session-variant", "s1", "--message-index", "2", "--ref", "foo/bar"],
            ["session-delve", "s1", "--message-index", "2"],
            ["session-documents", "s1"],
            ["server-status"],
            ["server-shutdown"],
            ["image-generate", "neon city"],
            ["image-progress"],
            ["image-cancel"],
            ["image-outputs"],
            ["image-runtime"],
            ["image-unload"],
            ["image-library"],
            ["image-catalog"],
            ["video-generate", "city skyline"],
            ["video-progress"],
            ["video-cancel"],
            ["video-outputs"],
            ["video-runtime"],
            ["video-library"],
            ["video-catalog"],
            ["longlive-install"],
            ["longlive-status"],
            ["wan-install"],
            ["wan-status"],
            ["wan-inventory"],
            ["cuda-torch-install"],
            ["gpu-bundle-install"],
            ["gpu-bundle-status"],
            ["gpu-bundle-info"],
            ["setup-install-package", "sageattention"],
            ["setup-install-system-package", "ffmpeg"],
            ["setup-refresh"],
            ["turbo-update-check"],
            ["diagnostics-snapshot"],
            ["diagnostics-log-tail"],
            ["diagnostics-reextract"],
            ["gpu-status"],
            ["metrics-gpu"],
            ["cache-preview", "foo/bar"],
            ["prompts-list"],
            ["prompts-enhance", "a fluffy cat"],
            ["settings-get"],
            ["settings-storage-get"],
            ["workspaces-list"],
            ["plugins-list"],
            ["plugins-enable", "p1"],
            ["plugins-disable", "p1"],
            ["tools-list"],
            ["adapters-list"],
            ["finetuning-status"],
            ["auth-session"],
            # HTML challenges
            ["challenges-list"],
            ["challenges-get", "c1"],
            ["challenges-file", "c1", "a"],
            ["challenges-create", "--body", "{}"],
            ["challenges-open-file", "/some/file"],
            ["challenges-repair", "c1", "a"],
            ["challenges-retry", "c1", "a"],
            ["challenges-validate", "c1", "a", "--body", "{}"],
            ["challenges-delete", "c1"],
            # Image/video download
            ["image-download", "foo/bar"],
            ["image-download-status"],
            ["image-download-cancel", "foo/bar"],
            ["image-download-delete", "foo/bar"],
            ["video-download", "foo/bar"],
            ["video-download-status"],
            ["video-download-cancel", "foo/bar"],
            ["video-download-delete", "foo/bar"],
            # Output artifacts
            ["image-output-get", "a1"],
            ["image-output-delete", "a1"],
            ["video-output-get", "v1"],
            ["video-output-file", "v1"],
            ["video-output-delete", "v1"],
            # Workspaces
            ["workspaces-rename", "w1", "--name", "new"],
            ["workspaces-delete", "w1"],
            ["workspaces-document-delete", "w1", "d1"],
            # Session extras
            ["session-rename", "s1", "--title", "Hi"],
            ["session-document-upload", "s1", "--body", "{}"],
            ["session-document-delete", "s1", "d1"],
            # Misc
            ["image-preload", "--model", "foo"],
            ["settings-storage-set", "--body", "{}"],
            ["settings-storage-move", "--destination", "/tmp"],
            ["settings-storage-move-status"],
            ["video-longlive"],
            ["video-mlx-runtime"],
            ["v1-models"],
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
        # Real /api/chat/generate shape: { session, runtime, assistant: { text, metrics } }
        body = {
            "session": {"id": "s-1"},
            "runtime": {"state": "loaded"},
            "assistant": {
                "role": "assistant",
                "text": "Hello back!",
                "metrics": {
                    "tokS": 42.5,
                    "promptTokens": 5,
                    "completionTokens": 3,
                    "responseSeconds": 0.07,
                    "runtimeNote": "MTPLX active",
                    "finishReason": "stop",
                },
            },
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
        # New behaviour: POST kicks off the job, then GET the status so we
        # surface whatever phase the worker has advanced to by the time
        # the request returns — usually "preflight" on a synchronous mock.
        body = {"id": "mtplx-install", "phase": "preflight", "done": False}
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({
            "/api/setup/install-mtplx": _FakeResp(json.dumps(body).encode("utf-8")),
            "/api/setup/install-mtplx/status": _FakeResp(json.dumps(body).encode("utf-8")),
        })):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["mtplx-install"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["phase"], "preflight")


class CLICallDispatcherTests(unittest.TestCase):
    def test_call_get_no_body(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["data"] = req.data
            return _FakeResp(json.dumps({"status": "ok"}).encode())

        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["call", "GET", "/api/health"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["url"], "http://127.0.0.1:8876/api/health")
        self.assertIsNone(captured["data"])
        self.assertEqual(json.loads(out.getvalue())["status"], "ok")

    def test_call_post_with_inline_body(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"runtime": {"state": "loaded"}}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "call", "POST", "/api/models/load",
                    "--body", '{"modelRef":"foo/bar","backend":"mlx"}',
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["modelRef"], "foo/bar")
        self.assertEqual(captured["body"]["backend"], "mlx")

    def test_call_query_params_encoded(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _FakeResp(b"{}")

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "call", "GET", "/api/models/search",
                    "--query", "q=qwen 3", "limit=5",
                ])
        self.assertEqual(rc, 0)
        self.assertIn("q=qwen+3", captured["url"])
        self.assertIn("limit=5", captured["url"])

    def test_call_propagates_http_error_with_exit_code_2(self) -> None:
        import urllib.error

        def _opener(req, timeout=None):  # noqa: ARG001
            raise urllib.error.HTTPError(
                url=req.full_url, code=409, msg="conflict", hdrs=None,
                fp=io.BytesIO(b'{"detail":"already running"}'),
            )

        err = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stderr", err):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(["call", "POST", "/api/images/generate", "--body", "{}"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("HTTP 409", err.getvalue())

    def test_call_file_body_reads_path(self) -> None:
        import tempfile

        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(b"{}")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
            fp.write(json.dumps({"package": "sageattention"}))
            tmp = fp.name

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "call", "POST", "/api/setup/install-package",
                    "--file", tmp,
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["package"], "sageattention")


class CLIRoutesTests(unittest.TestCase):
    def test_routes_lists_paths_with_filter(self) -> None:
        openapi = {
            "paths": {
                "/api/images/generate": {"post": {"summary": "Generate", "operationId": "img_gen"}},
                "/api/images/progress": {"get": {"summary": "Progress", "operationId": "img_prog"}},
                "/api/models/load": {"post": {"summary": "Load", "operationId": "load"}},
                "/api/health": {"get": {"summary": "Health"}},
            }
        }
        resp = _FakeResp(json.dumps(openapi).encode())
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/openapi.json": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["routes", "--filter", "/api/images"])
        self.assertEqual(rc, 0)
        body = json.loads(out.getvalue())
        self.assertEqual(body["count"], 2)
        paths = sorted(r["path"] for r in body["routes"])
        self.assertEqual(paths, ["/api/images/generate", "/api/images/progress"])

    def test_routes_method_filter(self) -> None:
        openapi = {
            "paths": {
                "/api/models/load": {"post": {"summary": "Load"}},
                "/api/runtime": {"get": {"summary": "Runtime"}},
            }
        }
        resp = _FakeResp(json.dumps(openapi).encode())
        out = io.StringIO()
        with mock.patch.object(cli.urllib.request, "urlopen", _mock_urlopen({"/openapi.json": resp})):
            with mock.patch.object(sys, "stdout", out):
                rc = cli.main(["routes", "--method", "POST"])
        self.assertEqual(rc, 0)
        body = json.loads(out.getvalue())
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["routes"][0]["method"], "POST")


class CLITypedShortcutTests(unittest.TestCase):
    """Spot-check representative typed wrappers from each category."""

    def test_image_generate_builds_body_from_flags(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"jobId": "x", "outputs": []}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "image-generate", "neon city",
                    "--model", "FLUX.1-schnell", "--steps", "4",
                    "--width", "1024", "--height", "768", "--seed", "42",
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["prompt"], "neon city")
        self.assertEqual(captured["body"]["modelId"], "FLUX.1-schnell")
        self.assertEqual(captured["body"]["steps"], 4)
        self.assertEqual(captured["body"]["seed"], 42)
        # None defaults stripped
        self.assertNotIn("guidance", captured["body"])

    def test_setup_install_package_payload(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"installed": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["setup-install-package", "sageattention"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"package": "sageattention"})

    def test_session_fork_sends_at(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"sessionId": "new"}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["session-fork", "s1", "--at", "7", "--title", "fork-of-s1"])
        self.assertEqual(rc, 0)
        self.assertIn("/api/chat/sessions/s1/fork", captured["url"])
        self.assertEqual(captured["body"], {"forkAtMessageIndex": 7, "title": "fork-of-s1"})

    def test_diagnostics_log_tail_passes_lines_query(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _FakeResp(json.dumps({"lines": ["..."]}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["diagnostics-log-tail", "--lines", "500"])
        self.assertEqual(rc, 0)
        self.assertIn("lines=500", captured["url"])

    def test_cache_preview_query_assembly(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _FakeResp(b"{}")

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "cache-preview", "Qwen/Qwen3.5-7B",
                    "--context", "32768", "--cache-strategy", "turboquant", "--cache-bits", "4",
                ])
        self.assertEqual(rc, 0)
        self.assertIn("ref=Qwen", captured["url"])
        self.assertIn("context=32768", captured["url"])
        self.assertIn("cacheStrategy=turboquant", captured["url"])
        self.assertIn("cacheBits=4", captured["url"])

    def test_settings_patch_sends_patch_method(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"ok": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["settings-patch", "--body", '{"theme":"dark"}'])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "PATCH")
        self.assertEqual(captured["body"], {"theme": "dark"})

    def test_video_generate_strips_none_defaults(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"jobId": "v1"}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["video-generate", "tree blowing in wind", "--frames", "16"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["prompt"], "tree blowing in wind")
        self.assertEqual(captured["body"]["numFrames"], 16)
        self.assertNotIn("steps", captured["body"])
        self.assertNotIn("width", captured["body"])


class CLIChallengesTests(unittest.TestCase):
    def test_challenges_list_hits_collection(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            return _FakeResp(json.dumps({"challenges": []}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["challenges-list"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "GET")
        self.assertTrue(captured["url"].endswith("/api/chat/html-challenges"))

    def test_challenges_repair_posts_to_slot_endpoint(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            return _FakeResp(json.dumps({"status": "queued"}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["challenges-repair", "c-123", "a"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "POST")
        self.assertIn("/api/chat/html-challenges/c-123/slots/a/repair", captured["url"])

    def test_challenges_validate_uses_patch_method(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"validated": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main([
                    "challenges-validate", "c-1", "b",
                    "--body", '{"valid":true,"notes":"ok"}',
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "PATCH")
        self.assertEqual(captured["body"], {"valid": True, "notes": "ok"})

    def test_challenges_file_writes_binary_to_out(self) -> None:
        import tempfile

        class _BinResp(_FakeResp):
            def __init__(self) -> None:
                super().__init__(b"\x89PNG\r\n\x1a\nfakebytes")
                self.headers = {"Content-Type": "image/png"}

        def _opener(req, timeout=None):  # noqa: ARG001
            return _BinResp()

        with tempfile.NamedTemporaryFile(delete=False) as fp:
            out_path = fp.name
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["challenges-file", "c1", "a", "--out", out_path])
        self.assertEqual(rc, 0)
        self.assertEqual(Path(out_path).read_bytes()[:4], b"\x89PNG")


class CLIDownloadLifecycleTests(unittest.TestCase):
    def test_image_download_sends_repo(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"started": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["image-download", "black-forest-labs/FLUX.1-schnell"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"repo": "black-forest-labs/FLUX.1-schnell"})

    def test_video_download_cancel_targets_repo(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"cancelled": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["video-download-cancel", "Wan-AI/Wan2.1-T2V-1.3B"])
        self.assertEqual(rc, 0)
        self.assertIn("/api/video/download/cancel", captured["url"])
        self.assertEqual(captured["body"], {"repo": "Wan-AI/Wan2.1-T2V-1.3B"})


class CLIOutputArtifactsTests(unittest.TestCase):
    def test_video_output_file_writes_to_out_path(self) -> None:
        import tempfile

        class _VidResp(_FakeResp):
            def __init__(self) -> None:
                super().__init__(b"\x00\x00\x00 ftypmp42fakebytes")
                self.headers = {"Content-Type": "video/mp4"}

        def _opener(req, timeout=None):  # noqa: ARG001
            return _VidResp()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as fp:
            out_path = fp.name
        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["video-output-file", "v123", "--out", out_path])
        self.assertEqual(rc, 0)
        body = Path(out_path).read_bytes()
        self.assertIn(b"ftypmp42", body)

    def test_image_output_delete_uses_DELETE_method(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            return _FakeResp(json.dumps({"deleted": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["image-output-delete", "img-42"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "DELETE")
        self.assertIn("/api/images/outputs/img-42", captured["url"])


class CLIWorkspaceAndSessionTests(unittest.TestCase):
    def test_workspaces_rename_with_name_flag(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"name": "renamed"}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["workspaces-rename", "w-1", "--name", "renamed"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["method"], "PATCH")
        self.assertEqual(captured["body"], {"name": "renamed"})

    def test_session_rename_with_title_flag(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"title": "New Title"}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["session-rename", "s-1", "--title", "New Title"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"title": "New Title"})


class CLIMiscTests(unittest.TestCase):
    def test_v1_models_openai_compat_endpoint(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _FakeResp(json.dumps({"object": "list", "data": []}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["v1-models"])
        self.assertEqual(rc, 0)
        self.assertTrue(captured["url"].endswith("/v1/models"))

    def test_settings_storage_move_with_destination_shorthand(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"queued": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["settings-storage-move", "--destination", "/Volumes/External"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"destination": "/Volumes/External"})

    def test_image_preload_with_model_shorthand(self) -> None:
        captured: dict[str, Any] = {}

        def _opener(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(json.dumps({"preloaded": True}).encode())

        with mock.patch.object(cli.urllib.request, "urlopen", _opener):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                rc = cli.main(["image-preload", "--model", "FLUX.1-schnell"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"modelRef": "FLUX.1-schnell"})


if __name__ == "__main__":
    unittest.main()
