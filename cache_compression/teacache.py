"""TeaCache diffusion cache strategy (ali-vilab/TeaCache).

TeaCache is a training-free, timestep-embedding-aware cache for diffusion
transformer (DiT) models. It estimates the rel-L1 distance between adjacent
timesteps' modulated inputs; when accumulated distance stays below a
threshold, the transformer short-circuits to the previous residual instead
of running a full forward pass. Reported speedups:

    FLUX              1.5–2.0×  (thresh 0.25 → 0.4 → 0.6)
    Wan2.1 720P       ~1.3×     (vanilla) / up to 30% faster
    HunyuanVideo      1.6–2.1×
    Open-Sora-Plan    up to 4.41×

Upstream: https://github.com/ali-vilab/TeaCache (Apache 2.0)

Integration pattern (from upstream TeaCache4FLUX / TeaCache4Wan2.1):

    FluxTransformer2DModel.forward = teacache_forward
    pipeline.transformer.__class__.enable_teacache = True
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.num_steps = num_inference_steps
    pipeline.transformer.__class__.rel_l1_thresh = 0.4
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None

Upstream ships a full transformer ``forward`` per model (FLUX, Wan2.1,
HunyuanVideo, Mochi, CogVideoX, LTX-Video, ...). The ChaosEngineAI scaffold
lands the strategy contract + state-attribute management; per-pipeline
``forward`` patches will be vendored into ``cache_compression._teacache_patches``
as they are validated against our pinned diffusers version.

Unlike the other cache strategies in this registry, TeaCache applies to
diffusion pipelines (image + video), not text LLMs. The registry surfaces
this via ``applies_to()``; the MLX / llama.cpp / vLLM hooks inherited from
``CacheStrategy`` all raise ``NotImplementedError`` to keep misuse loud.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from cache_compression import CacheStrategy


# Rel-L1 thresholds → expected speedup (from upstream README tables).
# Surfaced for UI + docs; enforcement lives in the caller.
_RECOMMENDED_THRESHOLDS: dict[str, tuple[float, str]] = {
    "conservative": (0.25, "~1.5× speedup, closest to baseline quality"),
    "balanced":     (0.40, "~1.8× speedup, minor quality delta (default)"),
    "aggressive":   (0.60, "~2.0× speedup, visible quality delta on long prompts"),
}

_DEFAULT_REL_L1_THRESH = 0.4

# Pipeline-class-name → (patch_module, patch_function_name).
# Each patch module lives at ``cache_compression._teacache_patches.<backend>``
# and exports a ``teacache_forward`` callable compatible with the named
# diffusers transformer class. Keeping the mapping centralised (rather than
# scanning the patches dir) lets the strategy report supported pipelines
# without importing heavy diffusers modules at discovery time.
#
# Empty in the scaffold — patches land incrementally. Add an entry here
# when vendoring each upstream ``teacache_forward_<model>.py`` under
# Apache 2.0, with the preserved copyright header in the vendored file.
_FORWARD_PATCHES: dict[str, tuple[str, str]] = {
    "FluxTransformer2DModel": (
        "cache_compression._teacache_patches.flux",
        "teacache_forward",
    ),
    "HunyuanVideoTransformer3DModel": (
        "cache_compression._teacache_patches.hunyuan_video",
        "teacache_forward",
    ),
    "LTXVideoTransformer3DModel": (
        "cache_compression._teacache_patches.ltx_video",
        "teacache_forward",
    ),
    "CogVideoXTransformer3DModel": (
        "cache_compression._teacache_patches.cogvideox",
        "teacache_forward",
    ),
    "MochiTransformer3DModel": (
        "cache_compression._teacache_patches.mochi",
        "teacache_forward",
    ),
    # WanTransformer3DModel intentionally absent — upstream
    # TeaCache4Wan2.1/teacache_generate.py targets the standalone
    # Wan-Video/Wan2.1 repo (signature ``forward(self, x, t, context,
    # seq_len, clip_fea, y)``), not diffusers' ``WanTransformer3DModel``
    # (signature ``forward(self, hidden_states, timestep,
    # encoder_hidden_states, ...)``). Adding Wan TeaCache means authoring
    # a fresh diffusers-shaped forward, not vendoring. See FU-007.
}


def _diffusers_available() -> bool:
    return importlib.util.find_spec("diffusers") is not None


class TeaCacheStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "teacache"

    @property
    def name(self) -> str:
        return "TeaCache"

    def is_available(self) -> bool:
        # TeaCache is always wired into the strategy registry as long as
        # diffusers is importable — we vendor upstream patches rather than
        # depending on a separate ``teacache`` pip package (upstream ships
        # as a repo, not a PyPI distribution).
        return _diffusers_available()

    def availability_badge(self) -> str:
        if not _diffusers_available():
            return "Install"
        if not _FORWARD_PATCHES:
            return "Scaffold"
        return "Ready"

    def availability_tone(self) -> str:
        if not _diffusers_available():
            return "install"
        if not _FORWARD_PATCHES:
            return "warning"
        return "ready"

    def availability_reason(self) -> str | None:
        if not _diffusers_available():
            return (
                "TeaCache needs the diffusers package (install the GPU "
                "runtime bundle from Setup, or `pip install diffusers`)."
            )
        if not _FORWARD_PATCHES:
            return (
                "TeaCache scaffold is active (strategy registered). "
                "Per-pipeline transformer patches will land incrementally "
                "in cache_compression._teacache_patches — FLUX + Wan2.1 "
                "first. See FU-007 in CLAUDE.md."
            )
        supported = ", ".join(sorted(_FORWARD_PATCHES.keys()))
        return (
            f"Ready for: {supported}. Wan2.1 TeaCache forward still pending "
            f"(diffusers signature mismatch with upstream) — see FU-007."
        )

    # ------------------------------------------------------------------
    # Registry metadata
    # ------------------------------------------------------------------

    def supported_bit_range(self) -> tuple[int, int] | None:
        # TeaCache is not a quantization strategy — the knob is a
        # floating-point rel-L1 threshold, not bit width.
        return None

    def default_bits(self) -> int | None:
        return None

    def supports_fp16_layers(self) -> bool:
        return False

    def applies_to(self) -> frozenset[str]:
        """Domains this strategy applies to (vs default text LLMs).

        The UI uses this to surface TeaCache in Image Studio / Video Studio
        rather than the text-model chat path. Override in other diffusion
        strategies as they land.
        """
        return frozenset({"image", "video"})

    def default_rel_l1_thresh(self) -> float:
        return _DEFAULT_REL_L1_THRESH

    def recommended_thresholds(self) -> dict[str, tuple[float, str]]:
        """Threshold presets for UI pickers."""
        return dict(_RECOMMENDED_THRESHOLDS)

    def supported_pipeline_classes(self) -> frozenset[str]:
        """Transformer class names with a vendored ``teacache_forward`` patch."""
        return frozenset(_FORWARD_PATCHES.keys())

    # ------------------------------------------------------------------
    # Diffusers integration
    # ------------------------------------------------------------------

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None = None,
    ) -> None:
        """Install TeaCache on a diffusers pipeline in-place.

        Called by image_runtime / video_runtime after the pipeline has been
        instantiated but before the first denoise call. Applies the
        upstream monkeypatch:

          1. Replaces ``type(pipeline.transformer).forward`` with the
             vendored ``teacache_forward`` for that transformer class.
          2. Sets the seven TeaCache state attributes on the transformer's
             class (``enable_teacache``, ``cnt``, ``num_steps``,
             ``rel_l1_thresh``, ``accumulated_rel_l1_distance``,
             ``previous_modulated_input``, ``previous_residual``).

        Raises ``NotImplementedError`` when the pipeline's transformer has
        no vendored patch yet — callers should catch and fall back to the
        stock pipeline. The error names the unsupported transformer class
        so the UI can surface an actionable message.
        """
        # Validate arguments first — cheap failures should precede the
        # transformer-class lookup so tests and callers see a ValueError
        # for "bad args" vs NotImplementedError for "bad shape".
        thresh = self.default_rel_l1_thresh() if rel_l1_thresh is None else float(rel_l1_thresh)
        if thresh <= 0:
            raise ValueError(
                f"rel_l1_thresh must be > 0 (got {thresh}). "
                "Use 0.25–0.6 for a practical speed/quality trade-off."
            )
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1 (got {num_inference_steps})."
            )

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "TeaCache requires a DiT-based diffusers pipeline with a "
                "`.transformer` attribute. Unet-based pipelines (SD 1.x/2.x) "
                "use a different caching approach — see DeepCache."
            )

        klass = type(transformer)
        klass_name = klass.__name__
        spec = _FORWARD_PATCHES.get(klass_name)
        if spec is None:
            supported = ", ".join(sorted(_FORWARD_PATCHES.keys())) or "(none yet)"
            raise NotImplementedError(
                f"TeaCache has no vendored forward patch for {klass_name}. "
                f"Supported transformers: {supported}. "
                f"Vendor the upstream ali-vilab/TeaCache forward function "
                f"under cache_compression/_teacache_patches/ to enable this "
                f"pipeline (see FU-007 in CLAUDE.md)."
            )

        module_name, function_name = spec
        patch_module = importlib.import_module(module_name)
        teacache_forward = getattr(patch_module, function_name)

        klass.forward = teacache_forward
        klass.enable_teacache = True
        klass.cnt = 0
        klass.num_steps = int(num_inference_steps)
        klass.rel_l1_thresh = thresh
        klass.accumulated_rel_l1_distance = 0
        klass.previous_modulated_input = None
        klass.previous_residual = None

    # ------------------------------------------------------------------
    # Engine integration — TeaCache does not apply to text LLMs
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        raise NotImplementedError(
            "TeaCache is a diffusion-pipeline cache (image/video DiTs). "
            "Use apply_diffusers_hook() on a diffusers pipeline instead. "
            "For text-LLM KV caching pick a different strategy (TriAttention, "
            "TurboQuant, ChaosEngine)."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        raise NotImplementedError(
            "TeaCache does not apply to llama.cpp (text LLM inference). "
            "Pick a llama.cpp-compatible strategy such as ChaosEngine, "
            "RotorQuant, or TurboQuant."
        )

    def apply_vllm_patches(self) -> None:
        # No-op: TeaCache is not a vLLM attention patch.
        return None

    def estimate_cache_bytes(
        self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers, num_kv_heads=None,
    ):
        # TeaCache doesn't shrink the KV cache — it skips redundant DiT
        # forward passes. The "cache" here is intermediate residuals kept
        # between denoise steps, not a KV cache. The registry's estimate
        # API is sized for text LLM KV; return baseline so preview UI
        # doesn't show a bogus compression ratio.
        kv_heads = num_kv_heads if num_kv_heads and num_kv_heads > 0 else num_heads
        kv_elements = 2 * num_layers * kv_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2  # FP16 = 2 bytes
        return baseline, baseline

    def label(self, bits: int, fp16_layers: int) -> str:
        return "TeaCache (diffusion)"
