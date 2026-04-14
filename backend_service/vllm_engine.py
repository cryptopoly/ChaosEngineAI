"""vLLM inference engine for ChaosEngineAI.

Wraps ``vllm.LLM`` to provide GPU-accelerated inference with support for
cache compression strategies that integrate via vLLM (e.g. TriAttention).

Install: ``pip install chaosengine-ai[vllm]``
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

from backend_service.inference import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
)
from backend_service.model_resolution import resolve_dflash_target_ref


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _vllm_version() -> str | None:
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except ImportError:
        return None


def _now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class VLLMEngine(BaseInferenceEngine):
    """Inference engine backed by vLLM (GPU-accelerated, Linux/CUDA primary)."""

    engine_name = "vllm"
    engine_label = "vLLM"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self._llm: Any = None  # vllm.LLM instance

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
        try:
            from vllm import LLM  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install chaosengine-ai[vllm]"
            ) from exc

        # Apply cache strategy patches (e.g. TriAttention monkeypatches)
        # BEFORE creating the LLM instance.
        from compression import registry
        strategy = registry.get(cache_strategy)
        runtime_note = None
        actual_cache_strategy = cache_strategy
        actual_cache_bits = cache_bits
        actual_fp16_layers = fp16_layers
        if strategy and hasattr(strategy, "apply_vllm_patches"):
            try:
                strategy.apply_vllm_patches()
                runtime_note = f"Applied {strategy.name} vLLM patches."
            except (NotImplementedError, RuntimeError) as exc:
                runtime_note = f"Cache strategy '{strategy.name}' patches failed: {exc}. Running without compression."
                actual_cache_strategy = "native"
                actual_cache_bits = 0
                actual_fp16_layers = 0

        target = runtime_target or path or model_ref
        dflash_target_ref = resolve_dflash_target_ref(
            canonical_repo=canonical_repo,
            path=path,
            model_ref=model_ref,
        )

        if progress_callback:
            progress_callback({"phase": "loading", "percent": 20.0, "message": f"Loading {target} via vLLM..."})

        # Resolve DFLASH speculative config when requested
        draft_model: str | None = None
        llm_kwargs: dict[str, Any] = {
            "model": target,
            "max_model_len": context_tokens,
            "trust_remote_code": True,
        }
        if speculative_decoding:
            try:
                from dflash import get_draft_model, is_vllm_available as dflash_vllm_ok
                if dflash_vllm_ok():
                    draft_model = get_draft_model(
                        dflash_target_ref or model_ref
                    )
                    if draft_model:
                        llm_kwargs["speculative_config"] = {
                            "method": "dflash",
                            "model": draft_model,
                        }
            except ImportError:
                pass

        self._llm = LLM(**llm_kwargs)

        version = _vllm_version() or "unknown"
        if runtime_note is None:
            runtime_note = f"Model loaded via vLLM {version}."
        if speculative_decoding and draft_model:
            runtime_note += f" DFLASH speculative decoding active (draft: {draft_model})."
        elif speculative_decoding and not draft_model:
            runtime_note += (
                f" DFLASH unavailable for '{dflash_target_ref or model_ref}': "
                "no compatible draft model is registered."
            )

        if progress_callback:
            progress_callback({"phase": "ready", "percent": 100.0, "message": "vLLM model ready."})

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=actual_cache_strategy,
            cacheBits=actual_cache_bits,
            fp16Layers=actual_fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=target,
            runtimeNote=runtime_note,
            speculativeDecoding=speculative_decoding and draft_model is not None,
            dflashDraftModel=draft_model,
            treeBudget=0,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
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
    ) -> GenerationResult:
        if self._llm is None or self.loaded_model is None:
            raise RuntimeError("No vLLM model is loaded.")

        from vllm import SamplingParams  # type: ignore[import-untyped]

        messages = self._build_messages(prompt, history, system_prompt)
        prompt_text = self._messages_to_text(messages)
        started = time.perf_counter()

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # vLLM doesn't allow exactly 0
        )
        outputs = self._llm.generate([prompt_text], params)
        elapsed = max(time.perf_counter() - started, 1e-6)

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM generation returned no output.")

        output = outputs[0]
        text = output.outputs[0].text.strip()
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        return GenerationResult(
            text=text or "Generation completed without decoded text.",
            finishReason=output.outputs[0].finish_reason or "stop",
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
            tokS=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=self.loaded_model.runtimeNote,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            role = msg.get("role")
            if role in {"system", "user", "assistant"}:
                content = msg.get("text", "")
                if isinstance(content, list):
                    content = " ".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in content)
                messages.append({"role": role, "content": str(content)})
        messages.append({"role": "user", "content": prompt})
        # Sanitize: remove empty assistant messages, merge consecutive same-role messages
        from backend_service.mlx_worker import _sanitize_messages
        messages = _sanitize_messages(messages)
        return messages

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        """Simple fallback prompt formatting when chat template is unavailable."""
        lines = []
        for msg in messages:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)
