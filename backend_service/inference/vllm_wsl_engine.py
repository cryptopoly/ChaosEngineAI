"""vLLM-in-WSL inference engine (FU-056 Phase 8 follow-up).

vLLM ships no native Windows wheels. Windows users with the WSL2
bridge installed (Phase 8 foundation) get vLLM access through this
engine: it spawns vLLM's OpenAI-compatible HTTP server inside the
WSL Ubuntu venv, then proxies ``/v1/chat/completions`` from the
Windows-side backend.

Architecturally identical to ``MtplxEngine`` (subprocess + HTTP
proxy + ``_wait_for_server`` poll loop) but the command prefix is
``wsl -- ~/.chaosengine/vllm-venv/bin/python -m
vllm.entrypoints.openai.api_server`` instead of the host-native
mtplx binary.

WSL2 networking lets the Windows side reach the WSL listener on
``127.0.0.1:<port>`` transparently — the loopback adapter inside
WSL is mirrored to the Windows host's loopback by default. No
port forwarding needed.

Path translation: a model loaded from a Windows-side directory
(e.g. ``C:\\Users\\Dan\\AI_Models\\Qwen3-7B``) needs to be
addressed as ``/mnt/c/Users/Dan/AI_Models/Qwen3-7B`` when vLLM
runs inside WSL. HF repo ids (``Qwen/Qwen3.5-7B``) pass through
unchanged — vLLM downloads them into the WSL HF cache. The Windows
HF cache is reachable via ``/mnt/c/...`` but its NTFS-on-/mnt/c
performance is ~10× slower than the WSL ext4 home; we deliberately
let vLLM keep its own HF cache inside WSL.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from backend_service.inference._constants import (
    DEFAULT_LLAMA_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)
from backend_service.inference._utils import (
    _find_open_port,
    _http_json,
    _normalize_message_content,
    _now_label,
    _read_text_tail,
)
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    StreamChunk,
)


# Same path the Phase 8 install endpoint writes into. Centralised
# here so the engine + the installer never drift out of sync.
_WSL_VLLM_VENV_PATH = "~/.chaosengine/vllm-venv"


def windows_path_to_wsl(path: str) -> str:
    """Translate a Windows-style path to its WSL ``/mnt/<drive>`` form.

    ``C:\\Users\\Dan\\AI_Models\\Qwen3-7B`` →
    ``/mnt/c/Users/Dan/AI_Models/Qwen3-7B``.

    Pass-throughs:
      - Already-WSL paths (start with ``/``): returned unchanged.
      - Forward-slash Windows paths (``C:/Users/...``): handled
        symmetrically to backslash form.
      - Non-path strings (HF repo ids, URLs, model names): returned
        unchanged. The detector is conservative — only translate
        strings that look like absolute Windows paths.
    """
    if not path:
        return path
    # Already a POSIX path; the user passed a WSL-native string.
    if path.startswith("/"):
        return path
    # Windows drive-letter pattern: ``X:\foo`` or ``X:/foo``. We don't
    # translate UNC paths (``\\server\share``) — those are rare for
    # local models and vLLM wouldn't load from them inside WSL anyway.
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


class VllmWslEngine(BaseInferenceEngine):
    """vLLM running inside the WSL isolated venv, proxied via HTTP.

    Spawn shape: ``wsl -- <venv>/bin/python -m
    vllm.entrypoints.openai.api_server --model X --host 127.0.0.1
    --port N``. We talk to it through the existing ``urllib`` HTTP
    helpers — same as MtplxEngine. The Windows process inherits all
    the stream / generate plumbing from ``BaseInferenceEngine`` so
    upstream callers don't need a "is this WSL?" branch.
    """

    engine_name = "vllm-wsl"
    engine_label = "vLLM (WSL bridge)"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _server_url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("vLLM WSL server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _build_wsl_command(
        self,
        *,
        model_arg: str,
        port: int,
        max_model_len: int,
    ) -> list[str]:
        """Compose the ``wsl -- python -m vllm.entrypoints...`` argv.

        Pulled out for tests + so the comment about each flag lives
        next to the flag rather than buried in load_model.
        """
        return [
            "wsl",
            "--",
            f"{_WSL_VLLM_VENV_PATH}/bin/python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_arg,
            # ``--host 127.0.0.1`` keeps vLLM listening only on the
            # loopback — WSL2 mirrors loopback to the Windows host so
            # the Windows backend reaches it without any port-forward
            # ceremony, and we don't expose the model to the LAN.
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # ``--max-model-len`` is vLLM's name for the context window
            # cap. Defaults to whatever the model card declares, which
            # can be too large for available VRAM. We pass through the
            # user-selected ``contextTokens`` so the launch settings
            # actually take effect.
            "--max-model-len",
            str(max_model_len),
            # Trust the model config without prompting. vLLM's default
            # is False, which throws ``ValueError: trust_remote_code``
            # for repos like Qwen3-VL that ship custom modeling code.
            "--trust-remote-code",
        ]

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # vLLM loads big models — the graceful terminate can
                # take a few seconds while it tears down CUDA tensors.
                # If it's still alive after 10 s, SIGKILL.
                try:
                    self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        self.port = None
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except OSError:
                pass
        self.log_handle = None

    def process_pid(self) -> int | None:
        if self.process is None or self.process.poll() is not None:
            return None
        return int(self.process.pid)

    def _wait_for_server(self) -> None:
        """Poll ``/health`` until vLLM accepts requests, or the subprocess dies.

        vLLM's startup is slow (~30-90 s for a 7B model on cold cache)
        because it builds the CUDA graph + warms KV blocks. We give it
        the standard llama-timeout budget and surface the captured log
        if it dies before becoming ready.
        """
        deadline = time.time() + DEFAULT_LLAMA_TIMEOUT_SECONDS
        last_error = "vLLM (WSL) did not become ready."
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = _read_text_tail(self.log_path)
                raise RuntimeError(logs or "vLLM (WSL) exited during startup.")
            try:
                _http_json(self._server_url("/health"), timeout=2.0)
                return
            except Exception as exc:  # noqa: BLE001 — best-effort poll
                last_error = str(exc)
            time.sleep(1.0)
        logs = _read_text_tail(self.log_path)
        raise RuntimeError(logs if logs else last_error)

    # ------------------------------------------------------------------
    # BaseInferenceEngine interface
    # ------------------------------------------------------------------

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if sys.platform != "win32":
            raise RuntimeError(
                "vLLM WSL bridge is Windows-only. Use the native vLLM "
                "engine on Linux."
            )
        if not self.capabilities.wslVllmAvailable:
            raise RuntimeError(
                "vLLM isn't installed in WSL. Install it from the "
                "Diagnostics → WSL2 vLLM bridge panel."
            )

        self.unload_model()

        self.port = _find_open_port()

        # Pick the most precise model reference available:
        #   1. local path (translated to /mnt/c/... if Windows-style)
        #   2. runtime_target (catalog override)
        #   3. canonical HF repo (vLLM downloads to its WSL HF cache)
        #   4. model_ref (last resort — usually equal to #3)
        if path:
            model_arg = windows_path_to_wsl(path)
        elif runtime_target:
            model_arg = windows_path_to_wsl(runtime_target)
        else:
            model_arg = canonical_repo or model_ref

        command = self._build_wsl_command(
            model_arg=model_arg,
            port=self.port,
            max_model_len=context_tokens,
        )

        if progress_callback:
            progress_callback({
                "phase": "loading",
                "percent": 10.0,
                "message": f"Spawning vLLM in WSL for {model_name}...",
            })

        temp_log = tempfile.NamedTemporaryFile(
            prefix="chaosengine-vllm-wsl-", suffix=".log", delete=False
        )
        temp_log.close()
        self.log_path = Path(temp_log.name)
        self.log_handle = self.log_path.open("a", encoding="utf-8")

        self.process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE_ROOT),
            stdout=self.log_handle,
            stderr=self.log_handle,
            text=True,
        )

        try:
            self._wait_for_server()
        except RuntimeError:
            self._cleanup_process()
            raise

        runtime_note = (
            f"vLLM running inside WSL ({self.capabilities.wslDistroName or 'Ubuntu'}) "
            f"venv at {_WSL_VLLM_VENV_PATH}."
        )
        if self.capabilities.wslVllmVersion:
            runtime_note = (
                f"vLLM {self.capabilities.wslVllmVersion} running inside WSL "
                f"({self.capabilities.wslDistroName or 'Ubuntu'})."
            )
        # Speculative decoding via the WSL bridge isn't wired yet — the
        # in-process VLLMEngine handles it via ``speculative_config=``,
        # but the OpenAI server entry-point uses a different surface
        # (``--speculative-model`` / ``--num-speculative-tokens``) that
        # we'll add in a follow-up. Note the gap honestly in the
        # runtime note rather than silently dropping the request.
        if speculative_decoding:
            runtime_note += (
                " Speculative decoding requested but not yet supported "
                "via the WSL bridge — running with standard decoding."
            )

        if progress_callback:
            progress_callback({
                "phase": "ready",
                "percent": 100.0,
                "message": "vLLM (WSL) ready.",
            })

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=cache_strategy,
            cacheBits=cache_bits,
            fp16Layers=fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=model_arg,
            runtimeNote=runtime_note,
            speculativeDecoding=False,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self._cleanup_process()
        self.loaded_model = None

    def generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The vLLM (WSL) server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({
                "role": role,
                "content": _normalize_message_content(message.get("text", "")),
            })
        messages.append({"role": "user", "content": prompt})

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = _http_json(
                self._server_url("/v1/chat/completions"),
                payload=payload,
                timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        text = str(message.get("content") or "")

        return GenerationResult(
            text=text,
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
            tokS=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=self.loaded_model.runtimeNote,
        )

    def stream_generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The vLLM (WSL) server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({
                "role": role,
                "content": _normalize_message_content(message.get("text", "")),
            })
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        url = self._server_url("/v1/chat/completions")
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(request, timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        started_at = time.perf_counter()

        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text_delta = delta.get("content")
                if text_delta:
                    yield StreamChunk(text=text_delta)
                # vLLM emits ``finish_reason`` on the last delta only.
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = str(fr)
                usage = chunk.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            runtime_note=self.loaded_model.runtimeNote if self.loaded_model else None,
        )
