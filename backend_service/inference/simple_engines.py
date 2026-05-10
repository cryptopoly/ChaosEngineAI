"""Stateless / passthrough inference engines.

Two engines that are small enough to share a module:

- ``RemoteOpenAIEngine`` — proxies chat completions to any
  OpenAI-compatible HTTP API (Anthropic via shim, Together, OpenRouter,
  etc.). Encodes the provider config into ``model_ref`` as
  ``remote:<base>|<key>|<model>``.
- ``MockInferenceEngine`` — placeholder used as the initial default
  before any model is loaded. Every method raises ``RuntimeError`` so a
  load attempt against the mock surfaces the missing-backend message
  with concrete install hints.

Neither engine touches subprocesses, model files, or KV cache state, so
they live separately from the heavy local engines (MLX, llama.cpp).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from backend_service.inference._utils import _now_label
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    StreamChunk,
)


class RemoteOpenAIEngine(BaseInferenceEngine):
    engine_name = "remote"
    engine_label = "Remote OpenAI-compatible API"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.api_base: str = ""
        self.api_key: str = ""
        self.remote_model: str = ""

    def load_model(
        self, *, model_ref, model_name, canonical_repo, source, backend, path, runtime_target,
        cache_strategy, cache_bits, fp16_layers, fused_attention,
        fit_model_in_memory, context_tokens, speculative_decoding=False,
        tree_budget=0,
        progress_callback=None,
    ) -> LoadedModelInfo:
        # The model_ref encodes the remote provider config: "remote:<base>|<key>|<model>"
        if not model_ref.startswith("remote:"):
            raise RuntimeError("Remote engine requires a remote:<base>|<key>|<model> ref.")
        try:
            _, payload = model_ref.split("remote:", 1)
            parts = payload.split("|", 2)
            if len(parts) != 3:
                raise ValueError("malformed remote ref")
            self.api_base, self.api_key, self.remote_model = parts
        except ValueError as exc:
            raise RuntimeError(f"Invalid remote model ref: {exc}") from exc

        if not self.api_base.startswith("https://") and not self.api_base.startswith("http://127.0.0.1"):
            raise RuntimeError("Remote API must use HTTPS (or localhost http://127.0.0.1).")

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name or self.remote_model,
            canonicalRepo=canonical_repo,
            source=source,
            backend="remote",
            engine=self.engine_name,
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=False,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=None,
            runtimeTarget=self.remote_model,
            runtimeNote=f"Remote API at {self.api_base}",
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self.loaded_model = None

    def _request(self, *, prompt, history, system_prompt, max_tokens, temperature, stream=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in history:
            role = m.get("role")
            if role in {"user", "assistant", "system"}:
                messages.append({"role": role, "content": m.get("text", "")})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.remote_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        url = self.api_base.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        return urllib.request.urlopen(req, timeout=120.0)

    def generate(self, *, prompt, history, system_prompt, max_tokens, temperature,
                 images=None, tools=None,
                 samplers=None, reasoning_effort=None, json_schema=None) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("Remote model not configured.")
        started = time.perf_counter()
        try:
            resp = self._request(
                prompt=prompt, history=history, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature, stream=False,
            )
            data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote API error: {detail or exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote connection failed: {exc.reason}") from exc

        elapsed = max(time.perf_counter() - started, 1e-6)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        prompt_t = int(usage.get("prompt_tokens") or 0)
        return GenerationResult(
            text=str(msg.get("content") or ""),
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_t,
            completionTokens=completion,
            totalTokens=prompt_t + completion,
            tokS=round(completion / elapsed, 1) if completion else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=f"Generated by remote API ({self.api_base})",
        )


class MockInferenceEngine(BaseInferenceEngine):
    """Placeholder engine used only as the initial default before any model is loaded.

    All methods raise ``RuntimeError`` — this engine never produces fake results.
    If ``_select_engine()`` is implemented correctly, the mock engine is never
    chosen for real model loads; these raises are a safety net.
    """

    engine_name = "mock"
    # Displayed in the Dashboard "Runtime engine" stat. Used to read "No
    # backend", which collided with the footer's "BACKEND ONLINE" badge —
    # two different meanings of "backend" (inference engine vs API sidecar)
    # sitting on the same screen. "Idle" matches the ``RuntimeStatus.state``
    # enum already used elsewhere and doesn't claim the sidecar is down.
    engine_label = "Idle"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None

    def _missing_backend_message(self) -> str:
        hints: list[str] = []
        if not self.capabilities.mlxUsable:
            hints.append("MLX is not available" + (f" ({self.capabilities.mlxMessage})" if self.capabilities.mlxMessage else ""))
        if not self.capabilities.ggufAvailable:
            hints.append("llama-server not found (install with: brew install llama.cpp)")
        if not self.capabilities.vllmAvailable:
            hints.append("vLLM not installed")
        return (
            "No inference backend is available. " + " | ".join(hints) + ". "
            "Install llama.cpp for GGUF models or ensure MLX is working for safetensors models."
        )

    def load_model(self, **kwargs: Any) -> LoadedModelInfo:
        raise RuntimeError(self._missing_backend_message())

    def unload_model(self) -> None:
        self.loaded_model = None

    def generate(self, **kwargs: Any) -> GenerationResult:
        raise RuntimeError(self._missing_backend_message())

    def stream_generate(self, **kwargs: Any) -> Iterator[StreamChunk]:
        raise RuntimeError(self._missing_backend_message())
