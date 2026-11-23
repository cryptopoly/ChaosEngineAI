"""Tests for the FU-056 Phase 8 follow-up: VllmWslEngine.

Pinned at the unit level — we don't actually shell out to ``wsl --``
or talk to a real vLLM server. ``subprocess.Popen`` + ``_http_json``
are mocked so the engine's lifecycle (spawn → wait_for_server →
generate → unload) gets exercised without a 90-second model load.

Two layers:
  - ``windows_path_to_wsl`` is a pure function and gets exhaustive
    coverage against the path-flavour matrix.
  - The engine class gets happy-path + Windows-only gate + missing-
    capability rejection tests.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service.inference.base import BackendCapabilities
from backend_service.inference.vllm_wsl_engine import (
    VllmWslEngine,
    gpu_memory_utilization,
    windows_path_to_wsl,
)

_GIB = 1024**3
_ENGINE = "backend_service.inference.vllm_wsl_engine"


def setUpModule() -> None:
    # No test here may shell out to the real nvidia-smi.
    global _smi_patch
    _smi_patch = patch(f"{_ENGINE}._gpu_free_total_bytes", return_value=(24 * _GIB, 24 * _GIB))
    _smi_patch.start()


def tearDownModule() -> None:
    _smi_patch.stop()


class WindowsPathToWslTests(unittest.TestCase):
    """Pure path-translation matrix — no side effects."""

    def test_translates_backslash_drive_letter_path(self):
        self.assertEqual(
            windows_path_to_wsl(r"C:\Users\Dan\AI_Models\Qwen3-7B"),
            "/mnt/c/Users/Dan/AI_Models/Qwen3-7B",
        )

    def test_translates_forward_slash_drive_letter_path(self):
        self.assertEqual(
            windows_path_to_wsl("C:/Users/Dan/AI_Models/Qwen3-7B"),
            "/mnt/c/Users/Dan/AI_Models/Qwen3-7B",
        )

    def test_lowercases_drive_letter(self):
        # WSL expects ``/mnt/c/...`` not ``/mnt/C/...``.
        self.assertEqual(
            windows_path_to_wsl(r"D:\Models"),
            "/mnt/d/Models",
        )

    def test_passes_through_existing_wsl_path(self):
        self.assertEqual(
            windows_path_to_wsl("/home/dan/models/Qwen3-7B"),
            "/home/dan/models/Qwen3-7B",
        )

    def test_passes_through_hf_repo_id(self):
        # vLLM accepts ``org/name`` directly and downloads to its
        # HF cache. We mustn't mangle it into a path.
        self.assertEqual(
            windows_path_to_wsl("Qwen/Qwen3.5-7B"),
            "Qwen/Qwen3.5-7B",
        )

    def test_passes_through_empty_string(self):
        self.assertEqual(windows_path_to_wsl(""), "")

    def test_passes_through_unc_path(self):
        # UNC paths (\\\\server\\share) aren't translated — vLLM wouldn't
        # load from them inside WSL anyway. Just don't crash.
        unc = r"\\server\share\models"
        self.assertEqual(windows_path_to_wsl(unc), unc)

    def test_passes_through_relative_path(self):
        # Relative paths have no drive letter; leave alone.
        self.assertEqual(windows_path_to_wsl(r"models\Qwen3"), r"models\Qwen3")


def _make_caps(*, wsl_vllm: bool = True, distro: str | None = "Ubuntu-24.04") -> BackendCapabilities:
    """Build a capabilities snapshot with the WSL bridge in the
    requested state. Default = "ready"."""
    return BackendCapabilities(
        pythonExecutable="/x/python",
        mlxAvailable=False,
        mlxLmAvailable=False,
        mlxUsable=False,
        wsl2Available=wsl_vllm,
        wslDistroName=distro,
        wslCudaAvailable=wsl_vllm,
        wslVllmAvailable=wsl_vllm,
        wslVllmVersion="0.6.3" if wsl_vllm else None,
    )


class VllmWslEngineGatesTests(unittest.TestCase):
    """Pre-spawn validation: platform + capability checks should
    raise *before* any subprocess is touched."""

    def test_load_rejects_off_windows(self):
        engine = VllmWslEngine(_make_caps(wsl_vllm=True))
        with patch.object(sys, "platform", "linux"):
            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(
                    model_ref="Qwen/Qwen3.5-7B",
                    model_name="Qwen3.5-7B",
                    canonical_repo="Qwen/Qwen3.5-7B",
                    source="catalog",
                    backend="vllm",
                    path=None,
                    runtime_target=None,
                    cache_strategy="native",
                    cache_bits=0,
                    fp16_layers=0,
                    fused_attention=False,
                    fit_model_in_memory=True,
                    context_tokens=8192,
                )
        self.assertIn("Windows-only", str(ctx.exception))

    def test_load_rejects_when_wsl_vllm_missing(self):
        engine = VllmWslEngine(_make_caps(wsl_vllm=False))
        with patch.object(sys, "platform", "win32"):
            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(
                    model_ref="Qwen/Qwen3.5-7B",
                    model_name="Qwen3.5-7B",
                    canonical_repo="Qwen/Qwen3.5-7B",
                    source="catalog",
                    backend="vllm",
                    path=None,
                    runtime_target=None,
                    cache_strategy="native",
                    cache_bits=0,
                    fp16_layers=0,
                    fused_attention=False,
                    fit_model_in_memory=True,
                    context_tokens=8192,
                )
        # The error points the user at the install panel rather than
        # leaving them guessing what to do next.
        self.assertIn("WSL2 vLLM bridge", str(ctx.exception))


class VllmWslEngineCommandTests(unittest.TestCase):
    """Argv composition — the command wraps a bash -c invocation so
    the venv's bin/ can be prepended to PATH (vLLM's flashinfer JIT
    needs ninja, which lives in the venv bin). Checks both the wsl
    wrapper shape AND the inner bash command content."""

    def test_build_command_wraps_in_bash_c(self):
        engine = VllmWslEngine(_make_caps())
        command = engine._build_wsl_command(
            model_arg="Qwen/Qwen3.5-7B",
            port=8000,
            max_model_len=8192,
        )

        # Outer shape: ``wsl --exec bash -c "<inner>"``. ``--exec`` runs bash
        # with no default shell in front, which would expand ``$(...)`` inside
        # the quoted model ref (security review 2026-11-20).
        self.assertEqual(command[0], "wsl")
        self.assertEqual(command[1], "--exec")
        self.assertEqual(command[2], "bash")
        self.assertEqual(command[3], "-c")
        self.assertEqual(len(command), 5, "expected wsl --exec bash -c <inner> shape")

    def test_a_local_folder_is_served_under_the_name_chat_uses(self):
        # vLLM serves a model under its --model value; from a folder that is the path, so chat (which sends the
        # model ref) got 404 "model does not exist" (RTX 4090, 2026-11-19).
        engine = VllmWslEngine(_make_caps())
        with_name = engine._build_wsl_command(
            model_arg="/mnt/e/AI Models/m/snapshots/abc", port=8000, max_model_len=4096,
            served_model_name="Qwen/Qwen3-0.6B")[4]
        self.assertIn("--served-model-name Qwen/Qwen3-0.6B ", with_name)
        self.assertLess(with_name.index("--model "), with_name.index("--served-model-name"))
        without = engine._build_wsl_command(model_arg="Qwen/Qwen3-0.6B", port=8000, max_model_len=4096)[4]
        self.assertNotIn("--served-model-name", without)

    def test_inner_command_exports_path_and_runs_vllm(self):
        engine = VllmWslEngine(_make_caps())
        command = engine._build_wsl_command(
            model_arg="Qwen/Qwen3.5-7B",
            port=8000,
            max_model_len=8192,
        )
        inner = command[4]

        # PATH prefix: venv bin first, then existing PATH. Without
        # this prefix the FlashInfer JIT can't find ``ninja`` and
        # vLLM crashes mid-startup. The PATH value is double-quoted
        # because the inherited $PATH on WSL contains Windows paths
        # with spaces (``/mnt/c/Program Files/...``) which would
        # otherwise word-split.
        # ``$HOME``, not ``~``: bash does not expand ``~`` inside double quotes, and a child's execvp does not
        # expand a literal ``~`` PATH entry (FU-533).
        self.assertIn('export PATH="$HOME/.chaosengine/vllm-venv/bin:$PATH"', inner)
        self.assertNotIn('PATH="~', inner)

        # The vLLM OpenAI server invocation.
        self.assertIn("~/.chaosengine/vllm-venv/bin/python", inner)
        self.assertIn("-m vllm.entrypoints.openai.api_server", inner)

        # User-driven flags.
        self.assertIn("--model Qwen/Qwen3.5-7B", inner)
        self.assertIn("--port 8000", inner)
        self.assertIn("--max-model-len 8192", inner)
        self.assertIn("--host 127.0.0.1", inner)
        self.assertIn("--trust-remote-code", inner)
        # vLLM 0.21+ flashinfer JIT bypasses + eager mode (avoid CUDA
        # graph JIT). See engine source for the rationale.
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", inner)
        # vLLM 0.34 has no VLLM_ATTENTION_BACKEND any more ("Unknown vLLM environment variable").
        self.assertNotIn("VLLM_ATTENTION_BACKEND", inner)
        self.assertNotIn("TORCH_SDPA", inner)
        self.assertIn("--enforce-eager", inner)

    def test_inner_command_quotes_model_arg_with_special_chars(self):
        # A model path with a space (rare but possible — e.g.
        # ``/mnt/c/My Models/Qwen3-7B``) must survive the shell parse.
        engine = VllmWslEngine(_make_caps())
        command = engine._build_wsl_command(
            model_arg="/mnt/c/My Models/Qwen3-7B",
            port=8000,
            max_model_len=8192,
        )
        inner = command[4]
        # ``shlex.quote`` either single-quotes the whole thing or
        # escapes the space; either way, the original path must
        # round-trip through the shell parse.
        self.assertIn("'/mnt/c/My Models/Qwen3-7B'", inner)


class WslShellInjectionTests(unittest.TestCase):
    """Security review 2026-11-20: ``wsl -- bash -c`` ran the line through the default shell first, so ``$(...)`` in
    a model ref or path ran as a command despite ``shlex.quote``. The bridge now uses ``wsl --exec``."""

    HOSTILE = "x$(echo PWNED)y`echo PWNED`"

    def test_command_never_goes_through_the_default_shell(self):
        command = VllmWslEngine(_make_caps())._build_wsl_command(
            model_arg=self.HOSTILE, port=8000, max_model_len=4096, served_model_name=self.HOSTILE)
        self.assertEqual(command[:4], ["wsl", "--exec", "bash", "-c"])

    @unittest.skipUnless(sys.platform == "win32" and os.environ.get("CHAOSENGINE_TEST_WSL") == "1",
                         "live WSL check: set CHAOSENGINE_TEST_WSL=1 on a Windows host with a distro")
    def test_live_wsl_keeps_a_hostile_name_literal(self):
        import shlex
        import subprocess

        command = VllmWslEngine(_make_caps())._build_wsl_command(model_arg="m", port=8000, max_model_len=4096)
        probe = [*command[:4], f"echo {shlex.quote(self.HOSTILE)}"]
        out = subprocess.run(probe, capture_output=True, text=True, timeout=60).stdout
        self.assertEqual(out.strip(), self.HOSTILE)

    @unittest.skipUnless(sys.platform == "win32" and os.environ.get("CHAOSENGINE_TEST_WSL") == "1",
                         "live WSL check: set CHAOSENGINE_TEST_WSL=1 on a Windows host with a distro")
    def test_live_wsl_child_processes_see_the_venv_bin_as_an_absolute_path(self):
        # FU-533: a literal ``~`` PATH entry is invisible to a child's execvp (vLLM's ninja call).
        import subprocess

        inner = VllmWslEngine(_make_caps())._build_wsl_command(model_arg="m", port=8000, max_model_len=4096)[4]
        export = inner.split(" && ", 1)[0]
        probe = ["wsl", "--exec", "bash", "-c",
                 export + " && python3 -c 'import os; print(os.environ[\"PATH\"].split(\":\")[0])'"]
        first = subprocess.run(probe, capture_output=True, text=True, timeout=60).stdout.strip()
        self.assertTrue(first.startswith("/") and first.endswith("/.chaosengine/vllm-venv/bin"), first)


class VllmWslEngineLifecycleTests(unittest.TestCase):
    """Happy-path lifecycle with the subprocess + HTTP probe mocked
    out. We never actually shell out to wsl.exe."""

    def test_load_spawns_subprocess_and_polls_health(self):
        engine = VllmWslEngine(_make_caps())

        # Fake the subprocess: poll() returns None (still running),
        # PID is a known int. ``terminate`` / ``wait`` are mocked so
        # ``unload_model`` doesn't hang.
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4242
        fake_proc.wait.return_value = 0

        with patch.object(sys, "platform", "win32"):
            with patch(
                "backend_service.inference.vllm_wsl_engine.subprocess.Popen",
                return_value=fake_proc,
            ) as popen_mock:
                with patch(
                    "backend_service.inference.vllm_wsl_engine._http_json",
                    return_value={},  # ``/health`` returns OK immediately
                ) as http_mock:
                    with patch(
                        "backend_service.inference.vllm_wsl_engine._find_open_port",
                        return_value=8765,
                    ):
                        info = engine.load_model(
                            model_ref="Qwen/Qwen3.5-7B",
                            model_name="Qwen3.5-7B",
                            canonical_repo="Qwen/Qwen3.5-7B",
                            source="catalog",
                            backend="vllm",
                            path=None,
                            runtime_target=None,
                            cache_strategy="native",
                            cache_bits=0,
                            fp16_layers=0,
                            fused_attention=False,
                            fit_model_in_memory=True,
                            context_tokens=8192,
                        )

        # Subprocess was spawned exactly once with the WSL argv shape.
        popen_mock.assert_called_once()
        spawned_argv = popen_mock.call_args.args[0]
        self.assertEqual(spawned_argv[0], "wsl")
        self.assertEqual(spawned_argv[2], "bash")
        self.assertEqual(spawned_argv[3], "-c")

        # Health probe was hit.
        http_mock.assert_called()

        # Loaded info reflects the spawn.
        self.assertEqual(info.engine, "vllm-wsl")
        self.assertEqual(info.ref, "Qwen/Qwen3.5-7B")
        self.assertEqual(engine.port, 8765)
        self.assertEqual(engine.process_pid(), 4242)

    def test_load_translates_windows_path_in_runtime_target(self):
        engine = VllmWslEngine(_make_caps())

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 7

        with patch.object(sys, "platform", "win32"):
            with patch(
                "backend_service.inference.vllm_wsl_engine.subprocess.Popen",
                return_value=fake_proc,
            ) as popen_mock:
                with patch(
                    "backend_service.inference.vllm_wsl_engine._http_json",
                    return_value={},
                ):
                    with patch(
                        "backend_service.inference.vllm_wsl_engine._find_open_port",
                        return_value=9000,
                    ):
                        engine.load_model(
                            model_ref="Qwen/Qwen3.5-7B",
                            model_name="Qwen3.5-7B",
                            canonical_repo=None,
                            source="local",
                            backend="vllm",
                            path=r"C:\Users\Dan\AI_Models\Qwen3-7B",
                            runtime_target=None,
                            cache_strategy="native",
                            cache_bits=0,
                            fp16_layers=0,
                            fused_attention=False,
                            fit_model_in_memory=True,
                            context_tokens=8192,
                        )

        spawned_argv = popen_mock.call_args.args[0]
        # The model arg lives inside the bash ``-c`` inner command
        # (argv[4]) since FU-056 Phase 8 follow-up moved to a wrapped
        # invocation. Translation to /mnt/c/... must round-trip into
        # the inner command verbatim so vLLM finds the weights.
        inner = spawned_argv[4]
        self.assertIn("/mnt/c/Users/Dan/AI_Models/Qwen3-7B", inner)


def _load(engine, **overrides):
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.pid = 11
    kwargs = dict(
        model_ref="Qwen/Qwen3-0.6B", model_name="Qwen3-0.6B", canonical_repo="Qwen/Qwen3-0.6B",
        source="catalog", backend="vllm", path=None, runtime_target=None, cache_strategy="native",
        cache_bits=0, fp16_layers=0, fused_attention=False, fit_model_in_memory=True, context_tokens=4096,
    )
    kwargs.update(overrides)
    with patch.object(sys, "platform", "win32"), \
            patch(f"{_ENGINE}.subprocess.Popen", return_value=fake_proc) as popen, \
            patch(f"{_ENGINE}._http_json", return_value={}), \
            patch(f"{_ENGINE}._find_open_port", return_value=9100):
        info = engine.load_model(**kwargs)
    return info, popen.call_args.args[0][4]


class GpuMemoryShareTests(unittest.TestCase):
    """The bridge no longer lets vLLM take its 0.92 default of the card."""

    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("CHAOSENGINE_VLLM_GPU_SHARE", None)

    def tearDown(self):
        self._env.stop()

    def _share(self, weight_gib, free_gib, total_gib):
        with patch(f"{_ENGINE}._gpu_free_total_bytes", return_value=(int(free_gib * _GIB), int(total_gib * _GIB))):
            return gpu_memory_utilization(int(weight_gib * _GIB))

    def test_a_tiny_model_on_a_24_gb_card_gets_the_minimum_budget(self):
        # 1.1 GiB took 23.6 of 24 GB before; now 6 GiB of 24.
        self.assertAlmostEqual(self._share(1.1, 23.0, 24.0), 0.25, places=2)

    def test_a_tiny_model_on_a_big_card_gets_a_smaller_share(self):
        self.assertAlmostEqual(self._share(1.1, 79.0, 80.0), 0.075, places=3)

    def test_a_model_that_needs_the_card_gets_the_cap(self):
        self.assertEqual(self._share(18.0, 23.0, 24.0), 0.85)
        self.assertEqual(self._share(60.0, 79.0, 80.0), 0.85)

    def test_share_never_exceeds_what_is_free(self):
        # 10 GiB used by another app: 24 - 10 - 1.5 margin = 12.5 of 24.
        self.assertAlmostEqual(self._share(18.0, 14.0, 24.0), 12.5 / 24, places=2)

    def test_unknown_weights_use_the_cap_and_unreadable_smi_uses_the_cap(self):
        self.assertEqual(self._share(0, 23.0, 24.0), 0.85)
        with patch(f"{_ENGINE}._gpu_free_total_bytes", return_value=None):
            self.assertEqual(gpu_memory_utilization(5 * _GIB), 0.85)

    def test_env_override_wins_and_bad_values_are_ignored(self):
        with patch.dict(os.environ, {"CHAOSENGINE_VLLM_GPU_SHARE": "0.6"}):
            self.assertEqual(gpu_memory_utilization(1 * _GIB), 0.6)
        with patch.dict(os.environ, {"CHAOSENGINE_VLLM_GPU_SHARE": "7"}):
            self.assertAlmostEqual(gpu_memory_utilization(1 * _GIB), 0.25, places=2)

    def test_command_carries_the_share_and_load_sizes_it_from_the_folder(self):
        engine = VllmWslEngine(_make_caps())
        inner = engine._build_wsl_command(
            model_arg="m", port=1, max_model_len=2048, gpu_memory_utilization=0.25)[4]
        self.assertIn("--gpu-memory-utilization 0.25 ", inner)
        self.assertNotIn("--gpu-memory-utilization", engine._build_wsl_command(
            model_arg="m", port=1, max_model_len=2048)[4])
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "model.safetensors").write_bytes(b"x" * 1024)
            info, loaded_inner = _load(engine, path=folder)
        # 1 KiB of weights: the 6 GiB minimum budget on a 24 GB card.
        self.assertIn("--gpu-memory-utilization 0.25 ", loaded_inner)
        self.assertIn("GPU memory share 25%", info.runtimeNote)


class DflashOverTheBridgeTests(unittest.TestCase):
    def test_speculative_config_is_passed_as_json(self):
        engine = VllmWslEngine(_make_caps())
        config = {"method": "dflash", "model": "z-lab/drafter", "num_speculative_tokens": 15}
        inner = engine._build_wsl_command(
            model_arg="m", port=1, max_model_len=2048, speculative_config=config)[4]
        self.assertIn("--max-num-seqs 8", inner)
        flag = inner.split("--speculative-config ", 1)[1].split(" --max-num-seqs", 1)[0]
        import shlex

        self.assertEqual(json.loads(shlex.split(flag)[0]), config)
        self.assertNotIn("--speculative-config", engine._build_wsl_command(
            model_arg="m", port=1, max_model_len=2048)[4])

    def test_load_with_dflash_builds_the_config_and_reports_it(self):
        caps = _make_caps()
        caps.wslVllmVersion = "0.34.0"
        engine = VllmWslEngine(caps)
        with patch("dflash.get_draft_model", return_value="z-lab/Qwen3.8-27B-DFlash") as draft, \
                patch("backend_service.vllm_engine._dflash_num_speculative_tokens", return_value=15):
            info, inner = _load(engine, model_ref="Qwen/Qwen3.8-27B", canonical_repo="Qwen/Qwen3.8-27B",
                                speculative_decoding=True)
        self.assertEqual(draft.call_args.args[0], "Qwen/Qwen3.8-27B")
        self.assertTrue(draft.call_args.kwargs["allow_dflash2"])
        self.assertIn("--speculative-config", inner)
        self.assertIn("num_speculative_tokens", inner)
        self.assertTrue(info.speculativeDecoding)
        self.assertIn("DFlash speculative decoding active (draft: z-lab/Qwen3.8-27B-DFlash)", info.runtimeNote)

    def test_old_wsl_vllm_runs_plain_and_says_why(self):
        caps = _make_caps()
        caps.wslVllmVersion = "0.21.0"
        info, inner = _load(VllmWslEngine(caps), speculative_decoding=True)
        self.assertNotIn("--speculative-config", inner)
        self.assertFalse(info.speculativeDecoding)
        self.assertIn("DFlash needs vLLM 0.28 or newer in WSL (found 0.21.0)", info.runtimeNote)

    def test_target_without_a_drafter_runs_plain_and_says_why(self):
        caps = _make_caps()
        caps.wslVllmVersion = "0.34.0"
        with patch("dflash.get_draft_model", return_value=None):
            info, inner = _load(VllmWslEngine(caps), speculative_decoding=True)
        self.assertNotIn("--speculative-config", inner)
        self.assertFalse(info.speculativeDecoding)
        self.assertIn("no compatible draft model is registered", info.runtimeNote)


class _FakeStream:
    def __init__(self, events):
        self._lines = [f"data: {json.dumps(e)}\n".encode() if e != "[DONE]" else b"data: [DONE]\n" for e in events]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _loaded_engine():
    engine = VllmWslEngine(_make_caps())
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    engine.process = fake_proc
    engine.port = 9100
    info, _ = _load(VllmWslEngine(_make_caps()))
    engine.loaded_model = info
    return engine


def _delta(text):
    return {"choices": [{"delta": {"content": text}}]}


class ReasoningSplitTests(unittest.TestCase):
    def test_generate_drops_the_think_block_from_the_reply(self):
        engine = _loaded_engine()
        response = {
            "choices": [{"message": {"content": "<think>\nlet me see\n</think>\n\nHello there."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9},
        }
        with patch(f"{_ENGINE}._http_json", return_value=response):
            result = engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=64, temperature=0.0)
        self.assertEqual(result.text, "Hello there.")

    def test_generate_keeps_a_plain_reply(self):
        engine = _loaded_engine()
        response = {"choices": [{"message": {"content": "Just text."}}], "usage": {}}
        with patch(f"{_ENGINE}._http_json", return_value=response):
            result = engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=64, temperature=0.0)
        self.assertEqual(result.text, "Just text.")

    def test_stream_sends_the_think_block_to_the_reasoning_channel(self):
        engine = _loaded_engine()
        # The tags arrive split across deltas, as vLLM streams them.
        events = [_delta("<thi"), _delta("nk>plan the "), _delta("answer</th"), _delta("ink>Hel"), _delta("lo!"),
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 7}},
                  "[DONE]"]
        with patch(f"{_ENGINE}.urllib.request.urlopen", return_value=_FakeStream(events)):
            chunks = list(engine.stream_generate(prompt="hi", history=[], system_prompt=None, max_tokens=64, temperature=0.0))
        reasoning = "".join(c.reasoning or "" for c in chunks)
        text = "".join(c.text or "" for c in chunks)
        self.assertEqual(reasoning, "plan the answer")
        self.assertEqual(text, "Hello!")
        self.assertNotIn("think", text.lower())
        self.assertTrue(any(c.reasoning_done for c in chunks))
        done = chunks[-1]
        self.assertTrue(done.done)
        self.assertEqual((done.finish_reason, done.completion_tokens), ("stop", 7))

    def test_stream_passes_a_server_side_reasoning_field_through(self):
        engine = _loaded_engine()
        events = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}, _delta("Done."), "[DONE]"]
        with patch(f"{_ENGINE}.urllib.request.urlopen", return_value=_FakeStream(events)):
            chunks = list(engine.stream_generate(prompt="hi", history=[], system_prompt=None, max_tokens=8, temperature=0.0))
        self.assertEqual("".join(c.reasoning or "" for c in chunks), "hmm")
        self.assertEqual("".join(c.text or "" for c in chunks), "Done.")

    def test_stream_flushes_a_reply_that_ends_inside_the_think_block(self):
        engine = _loaded_engine()
        events = [_delta("<think>cut off by max tokens"), "[DONE]"]
        with patch(f"{_ENGINE}.urllib.request.urlopen", return_value=_FakeStream(events)):
            chunks = list(engine.stream_generate(prompt="hi", history=[], system_prompt=None, max_tokens=8, temperature=0.0))
        self.assertEqual("".join(c.reasoning or "" for c in chunks), "cut off by max tokens")
        self.assertEqual("".join(c.text or "" for c in chunks), "")


class LibraryFolderForVllmTests(unittest.TestCase):
    """A repo-id load uses the library's folder instead of vLLM downloading the model again."""

    def test_returns_the_folder_for_a_downloaded_safetensors_model(self):
        from backend_service.state.lifecycle import library_dir_for_vllm

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(library_dir_for_vllm({"path": folder, "format": "Transformers"}), folder)

    def test_resolves_a_hugging_face_cache_folder_to_its_snapshot(self):
        from backend_service.state.lifecycle import library_dir_for_vllm

        with tempfile.TemporaryDirectory() as root:
            snap = Path(root, "models--Qwen--Qwen3-0.6B", "snapshots", "abc")
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                library_dir_for_vllm({"path": str(Path(root, "models--Qwen--Qwen3-0.6B")), "format": "Transformers"}),
                str(snap),
            )

    def test_returns_none_for_gguf_broken_missing_or_configless(self):
        from backend_service.state.lifecycle import library_dir_for_vllm

        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(library_dir_for_vllm({"path": folder, "format": "Transformers"}))
            Path(folder, "config.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(library_dir_for_vllm({"path": folder, "format": "GGUF"}))
            self.assertIsNone(library_dir_for_vllm({"path": folder, "format": "Transformers", "broken": True}))
        self.assertIsNone(library_dir_for_vllm(None))
        self.assertIsNone(library_dir_for_vllm({"path": "", "format": "Transformers"}))
        self.assertIsNone(library_dir_for_vllm({"path": "Z:/does/not/exist", "format": "Transformers"}))


if __name__ == "__main__":
    unittest.main()
