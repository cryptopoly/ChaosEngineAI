"""vLLM inference engine for ChaosEngineAI.

Wraps ``vllm.LLM`` to provide GPU-accelerated inference with support for
cache compression strategies that integrate via vLLM (e.g. TriAttention).

Install: ``pip install chaosengine-ai[vllm]``
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path
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


# Env vars a cache strategy set for the previous load; cleared before the next.
_STRATEGY_ENV_KEYS: set[str] = set()


def _apply_strategy_env(env: dict[str, str]) -> None:
    """Set a strategy's env for this load, dropping the previous strategy's."""
    for key in _STRATEGY_ENV_KEYS - set(env):
        os.environ.pop(key, None)
    _STRATEGY_ENV_KEYS.clear()
    for key, value in env.items():
        os.environ[key] = value
        _STRATEGY_ENV_KEYS.add(key)


def _prepare_vllm_env(strategy_id: str) -> list[str]:
    """Set the process env vLLM's engine subprocess inherits. Returns notes.

    * An installed ``triattention`` package registers a vLLM general plugin
      that patches every engine it starts, whichever cache strategy the app
      asked for. Its patches target older vLLM internals (on 0.33 the engine
      core dies on the first request with ``'EngineCoreProc' object has no
      attribute 'log_iteration_details'``), so it is switched off unless the
      TriAttention strategy was chosen.
    * Without ``nvcc`` the FlashInfer sampler cannot JIT its kernel and the
      memory profiling run fails at load; vLLM's own sampler works.
    """
    notes: list[str] = []
    if strategy_id == "triattention":
        os.environ.pop("ENABLE_TRIATTENTION", None)
    else:
        os.environ["ENABLE_TRIATTENTION"] = "0"
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        has_nvcc = (
            shutil.which("nvcc") is not None
            or (bool(cuda_home) and (Path(cuda_home) / "bin" / "nvcc").exists())
            or Path("/usr/local/cuda/bin/nvcc").exists()
        )
        if not has_nvcc:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            notes.append("No nvcc found, FlashInfer sampler off.")
    return notes


def _draft_config(draft_model: str) -> dict[str, Any] | None:
    """The drafter's ``config.json``: a local folder, else the HF cache/Hub."""
    import json

    local = Path(draft_model)
    try:
        if local.is_dir():
            return json.loads((local / "config.json").read_text(encoding="utf-8"))
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(draft_model, "config.json")).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - vLLM reports a missing drafter itself
        return None


def _dflash_num_speculative_tokens(draft_model: str) -> int | None:
    """Draft length for a DFlash drafter: its trained block size minus one.

    vLLM only defaults ``num_speculative_tokens`` from ``n_predict`` (MTP)
    or a DSpark block size; a DFlash drafter carries
    ``dflash_config.block_size`` instead, so leaving it out fails the load
    with "num_speculative_tokens was not provided". A block holds the
    anchor token plus ``block_size - 1`` drafted ones (z-lab's b16
    drafters serve with 15).
    """
    config = _draft_config(draft_model) or {}
    block = (config.get("dflash_config") or {}).get("block_size") or config.get("block_size")
    try:
        block = int(block)
    except (TypeError, ValueError):
        return None
    return block - 1 if block > 1 else None


# Concurrent sequences per load (vLLM's server default is 256).
_DESKTOP_MAX_NUM_SEQS = 8

# vLLM's own default share of the card, and what is left free beside our share.
# NVML reads more free memory than CUDA's own count, which is what vLLM
# checks: 23.38 against 22.35 GiB on an idle 24 GB card under WSL2, so the
# headroom covers that gap.
_VLLM_DEFAULT_GPU_UTILIZATION = 0.92
_GPU_HEADROOM_BYTES = 1536 * 1024 * 1024


def _gpu_free_total_bytes() -> tuple[int, int] | None:
    """Free and total memory of GPU 0 through NVML (no CUDA context here)."""
    try:
        import pynvml  # type: ignore[import-untyped]  # nvidia-ml-py, a vLLM dependency

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return int(info.free), int(info.total)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - no NVML: keep vLLM's default
        return None


def _gpu_memory_utilization() -> float | None:
    """Share of the card to give vLLM when its default would not be free.

    vLLM refuses to start when ``gpu_memory_utilization`` x total is more
    than the free memory, so anything else on the card (the desktop, a
    second app, a voice model) made every load fail at the default 0.92.
    Returns ``None`` when the default fits, else the free share less a
    small headroom. Read at every load, never fixed to one card.
    """
    override = os.environ.get("CHAOSENGINE_VLLM_GPU_SHARE", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if 0.0 < value <= 1.0:
            return value
    sizes = _gpu_free_total_bytes()
    if sizes is None:
        return None
    free, total = sizes
    usable = free - _GPU_HEADROOM_BYTES
    if total <= 0 or usable >= _VLLM_DEFAULT_GPU_UTILIZATION * total:
        return None
    share = usable / total
    return round(share, 3) if share > 0 else None


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
        from cache_compression import registry
        strategy = registry.get(cache_strategy)
        runtime_note = None
        actual_cache_strategy = cache_strategy
        actual_cache_bits = cache_bits
        actual_fp16_layers = fp16_layers
        strategy_env: dict[str, str] = {}
        strategy_llm_kwargs: dict[str, Any] = {}
        if strategy and hasattr(strategy, "apply_vllm_patches"):
            try:
                if hasattr(strategy, "vllm_runtime"):
                    # Checked first: without a stats file the patched engine
                    # stops at the first compression.
                    strategy_env, strategy_llm_kwargs, runtime_note = strategy.vllm_runtime(
                        [canonical_repo, model_ref, path],
                    )
                strategy.apply_vllm_patches()
                runtime_note = runtime_note or f"Applied {strategy.name} vLLM patches."
            except (NotImplementedError, RuntimeError) as exc:
                runtime_note = f"Cache strategy '{strategy.name}' unavailable: {exc} Running without compression."
                actual_cache_strategy = "native"
                actual_cache_bits = 0
                actual_fp16_layers = 0
                strategy_env, strategy_llm_kwargs = {}, {}

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
            # vLLM's server default (256 sequences) reserves one recurrent
            # state block per sequence on hybrid models (Qwen3.5/3.6/3.8
            # Gated DeltaNet); a 27B on a 24 GB card has about 30, so the
            # load failed. The app runs a few chats at a time, and fewer
            # sequences also mean fewer CUDA graphs to capture at load.
            "max_num_seqs": _DESKTOP_MAX_NUM_SEQS,
        }
        if speculative_decoding:
            try:
                from dflash import get_draft_model, is_vllm_available as dflash_vllm_ok
                if dflash_vllm_ok():
                    draft_model = get_draft_model(
                        dflash_target_ref or model_ref,
                        # vLLM >=0.28 runs DFlash 2 drafters (FU-089).
                        allow_dflash2=True,
                    )
                    if draft_model:
                        spec_config: dict[str, Any] = {
                            "method": "dflash",
                            "model": draft_model,
                        }
                        draft_tokens = _dflash_num_speculative_tokens(draft_model)
                        if draft_tokens:
                            spec_config["num_speculative_tokens"] = draft_tokens
                        llm_kwargs["speculative_config"] = spec_config
            except ImportError:
                pass

        env_notes = _prepare_vllm_env(actual_cache_strategy)
        _apply_strategy_env(strategy_env)
        llm_kwargs.update(strategy_llm_kwargs)
        if "gpu_memory_utilization" not in llm_kwargs:
            gpu_share = _gpu_memory_utilization()
            if gpu_share is not None:
                llm_kwargs["gpu_memory_utilization"] = gpu_share
                env_notes.append(f"GPU memory share {gpu_share:.0%} (sized to the free memory).")
        self._llm = LLM(**llm_kwargs)

        version = _vllm_version() or "unknown"
        if runtime_note is None:
            runtime_note = f"Model loaded via vLLM {version}."
        if env_notes:
            runtime_note += " " + " ".join(env_notes)
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
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self._llm is None or self.loaded_model is None:
            raise RuntimeError("No vLLM model is loaded.")

        from vllm import SamplingParams  # type: ignore[import-untyped]

        messages = self._build_messages(prompt, history, system_prompt)
        prompt_text = self._render_prompt(messages)
        started = time.perf_counter()

        # Build SamplingParams from base fields + optional Phase 2.2
        # sampler overrides. vLLM uses ``repetition_penalty`` rather
        # than llama-server's ``repeat_penalty`` — translate on the way
        # in. Unknown keys are silently ignored so newer UI knobs don't
        # break older vLLM builds.
        params_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),  # vLLM forbids 0
        }
        if samplers:
            _SAMPLER_PASSTHROUGH = (
                "top_p", "top_k", "min_p", "seed",
                "frequency_penalty", "presence_penalty",
            )
            for key in _SAMPLER_PASSTHROUGH:
                value = samplers.get(key)
                if value is not None:
                    params_kwargs[key] = value
            # vLLM names the repeat-penalty knob differently.
            repeat = samplers.get("repeat_penalty")
            if repeat is not None:
                params_kwargs["repetition_penalty"] = repeat
        # reasoning_effort + json_schema are accepted to match the
        # controller's call shape (parity with LlamaCppEngine) but vLLM
        # has no in-process equivalent for either in V0 / V1, so they
        # currently pass through as a no-op. Surfacing this in a
        # runtimeNote would be nice once the controller exposes one.
        del reasoning_effort, json_schema

        params = SamplingParams(**params_kwargs)
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

    def stream_generate(self, *, thinking_mode: str | None = None, **kwargs: Any):
        """Remember the chat's thinking switch for the prompt render, then stream.

        The base class drops ``thinking_mode`` before it reaches ``generate``,
        so templates that gate reasoning on ``enable_thinking`` (Qwen3 family)
        reasoned even when the chat said off.
        """
        self._thinking_mode = thinking_mode
        try:
            yield from super().stream_generate(**kwargs)
        finally:
            self._thinking_mode = None

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Apply the model's own chat template; plain text only when it has none.

        The plain ``USER: ... ASSISTANT:`` layout is not what chat models were
        trained on (Qwen3 rambles and repeats itself in it).
        """
        try:
            tokenizer = self._llm.get_tokenizer()
            if getattr(tokenizer, "chat_template", None):
                template_kwargs: dict[str, Any] = {}
                thinking_mode = getattr(self, "_thinking_mode", None)
                if thinking_mode is not None:
                    # Unknown template variables are ignored by the template.
                    template_kwargs["enable_thinking"] = thinking_mode != "off"
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **template_kwargs,
                )
                if isinstance(rendered, str) and rendered:
                    return rendered
        except Exception:  # noqa: BLE001 - any template failure falls back to plain text
            pass
        return self._messages_to_text(messages)

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        """Simple fallback prompt formatting when chat template is unavailable."""
        lines = []
        for msg in messages:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)
