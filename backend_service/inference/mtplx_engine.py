"""MTPLX inference engine.

Spawns the MTPLX server (``mtplx start --model <path> --port N``) from its
isolated venv at ``~/.chaosengine/mtplx-venv/`` as a subprocess, then proxies
``/v1/chat/completions`` through it — the same pattern used by
``LlamaCppEngine`` for llama-server.

MTPLX provides native in-model MTP speculative decoding for Apple Silicon;
its forked mlx lives in the isolated venv so it never conflicts with the main
``.venv``'s upstream mlx.

Fallback contract: ``load_model`` raises ``RuntimeError`` on any startup
failure.  The ``RuntimeController`` catches that and falls back to the
standard ``MlxEngine``.
"""

from __future__ import annotations

import json
import subprocess
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
    _append_runtime_note,
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
    RepeatedLineGuard,
    StreamChunk,
)
from backend_service.reasoning_split import (
    ThinkingTokenFilter,
    strip_thinking_tokens as _strip_thinking_tokens,
)

_MTPLX_VENV = Path.home() / ".chaosengine" / "mtplx-venv"


class MtplxEngine(BaseInferenceEngine):
    engine_name = "mtplx"
    engine_label = "MTPLX (MTP speculative decoding)"

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
            raise RuntimeError("MTPLX server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _mtplx_bin(self) -> str:
        """Path to the mtplx executable in the isolated venv."""
        candidate = _MTPLX_VENV / "bin" / "mtplx"
        if candidate.exists():
            return str(candidate)
        # Fall back to capabilities-resolved python path's sibling
        if self.capabilities.mtplxPythonPath:
            sibling = Path(self.capabilities.mtplxPythonPath).parent / "mtplx"
            if sibling.exists():
                return str(sibling)
        raise RuntimeError(
            "MTPLX is not installed. Install it from the Setup tab."
        )

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
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
        deadline = time.time() + DEFAULT_LLAMA_TIMEOUT_SECONDS
        last_error = "MTPLX server did not become ready."
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = _read_text_tail(self.log_path)
                raise RuntimeError(logs or "MTPLX server exited during startup.")
            try:
                _http_json(self._server_url("/health"), timeout=2.0)
                return
            except Exception as exc:
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
        speculative_decoding: bool = True,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.mtplxAvailable:
            raise RuntimeError("MTPLX is not installed. Install it from the Setup tab.")

        self.unload_model()

        mtplx_bin = self._mtplx_bin()
        self.port = _find_open_port()

        # Prefer local path; fall back to HF repo id (MTPLX will download).
        model_arg = path or runtime_target or model_ref

        command = [
            mtplx_bin,
            "start",
            "--model", model_arg,
            "--port", str(self.port),
        ]

        temp_log = tempfile.NamedTemporaryFile(
            prefix="chaosengine-mtplx-", suffix=".log", delete=False
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

        from backend_service.inference._mtp import get_mtp_draft_n
        draft_n = get_mtp_draft_n(canonical_repo or model_ref) or 1

        runtime_note = (
            f"MTPLX MTP speculative decoding active "
            f"(draft tokens: {draft_n}, model: {model_name})."
        )

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=cache_strategy,
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=runtime_target or path,
            runtimeNote=runtime_note,
            speculativeDecoding=True,
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
            raise RuntimeError(logs or "The MTPLX server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
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
        text = _strip_thinking_tokens(str(message.get("content") or ""))

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
            raise RuntimeError(logs or "The MTPLX server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
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
        stream_start = time.perf_counter()
        first_token_time: float | None = None
        runtime_note = self.loaded_model.runtimeNote
        think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode or "off") != "off")
        runaway_guard = RepeatedLineGuard()

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[len("data: "):]
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    split = think_filter.feed(str(content))
                    if split.reasoning:
                        yield StreamChunk(reasoning=split.reasoning)
                    if split.reasoning_done:
                        yield StreamChunk(reasoning_done=True)
                    if split.text:
                        runaway_guard.feed(split.text)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        completion_tokens += 1
                        yield StreamChunk(text=split.text)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            flushed = think_filter.flush()
            if flushed.reasoning:
                yield StreamChunk(reasoning=flushed.reasoning)
            if flushed.reasoning_done:
                yield StreamChunk(reasoning_done=True)
            if flushed.text:
                runaway_guard.feed(flushed.text)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                yield StreamChunk(text=flushed.text)
            runaway_guard.flush()
        except RuntimeError as exc:
            runtime_note = _append_runtime_note(runtime_note, str(exc))
            finish_reason = "stop"
        finally:
            resp.close()

        end_time = time.perf_counter()
        gen_elapsed = max(end_time - (first_token_time or stream_start), 1e-6)
        tok_s = round(completion_tokens / gen_elapsed, 1) if completion_tokens > 0 else 0.0

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=tok_s,
            runtime_note=runtime_note,
        )
