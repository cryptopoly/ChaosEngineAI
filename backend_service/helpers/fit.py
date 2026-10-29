"""Will this model fit on this machine? One answer for the chat, image and video catalogs.

``variant_fit`` turns a catalog variant and a ``HardwareClass`` into the ``fit`` dict the
Models pages show as a badge::

    {"status": "fits" | "tight" | "wont-fit" | "unknown",
     "needGb": 18.4, "usableGb": 51.8, "headroomPct": 64,
     "reason": "...", "basis": "measured" | "estimated"}

The number compared against the machine is the memory the model needs once loaded, as a share of
``HardwareClass.usable_gb``: up to 70 % is ``fits``, up to 92 % is ``tight``, above is ``wont-fit``.

* Chat: the weights (file size) plus a 10 % runtime share (the same share ``/api/models/memory-check``
  adds) plus the native KV cache at an 8K context. The cache is sized from the model's own layout
  when it is on disk (``helpers.model_arch``); a model that is not downloaded is never looked up
  on the Hub from here (a catalog request must not wait on 100 network calls), so its cache comes
  from a parameter-count table. A row that states its own ``estimatedMemoryGb`` replaces the
  weights-plus-runtime part.
* Image and video: the catalog's runtime footprint for this hardware (``runtimeFootprintMpsGb`` on a
  Mac, ``runtimeFootprintCudaGb`` on an NVIDIA GPU, else ``runtimeFootprintGb``), falling back to the
  download size x 1.3.

``basis`` is ``measured`` only for a footprint that was read from a run on Apple Silicon (the
variants listed in ``MEASURED_FOOTPRINTS``, which mirror the "measured" comments in the catalogs, or
a catalog row that sets ``footprintBasis: "measured"``). A variant that cannot run on this kind of
machine (an MLX build on Windows, a CUDA build on a Mac) is ``unknown`` with the reason, never
``wont-fit``.

``best_fit_variant_id`` picks one variant per family: the newest release that ``fits``, then the
faster one, then the smaller.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

from backend_service.helpers.hardware_class import HardwareClass, host_hardware_class

logger = logging.getLogger(__name__)

# Share of usable memory: at or below FITS_MAX_SHARE the model fits with room to spare, at or
# below TIGHT_MAX_SHARE it runs but leaves little for the rest of the machine.
FITS_MAX_SHARE = 0.70
TIGHT_MAX_SHARE = 0.92
_SHARE_EPSILON = 1e-9

# The chat estimate is for the default launch: native f16 cache at this context.
DEFAULT_CHAT_CONTEXT_TOKENS = 8192
# Memory a loaded model holds beyond its files (runtime buffers, compute graph); the same share
# the launch modal's load check (/api/models/memory-check) adds to the weights.
CHAT_RUNTIME_SHARE = 0.10
# Image / video pipelines hold more than the checkpoint (text encoder, VAE, activations).
MEDIA_SIZE_FACTOR = 1.3
# Decode reads the cache for a typical turn, not the whole window (see helpers.cache).
_SPEED_KV_TOKENS = 2048

# Footprints that came from a run on an M4 Max (the catalog comments say "measured"): variant id ->
# the footprint key the figure sits under. Rows whose number was copied from a sibling or scaled
# from another row are left out on purpose.
MEASURED_FOOTPRINTS: dict[str, str] = {
    "Qwen/Qwen-Image-2.1-mflux": "runtimeFootprintMpsGb",     # 28.7-29.9 GB peak, 1024 px / 40 steps
    "speridlabs/iris-3b-mflux": "runtimeFootprintMpsGb",      # peak MLX memory at 768x768
    "prince-canuma/LTX-2.3-distilled": "runtimeFootprintGb",  # 37.2 GB mlx-video peak, 40.0 GB process
    "dgrauet/ltx-2.3-mlx-q4": "runtimeFootprintGb",           # 15-18 GB sampled, 4-10 s clips
}

_MEDIA_FOOTPRINT_KEYS: dict[str, tuple[str, str]] = {
    "apple": ("runtimeFootprintMpsGb", "runtimeFootprintGb"),
    "cuda": ("runtimeFootprintCudaGb", "runtimeFootprintGb"),
    "cpu": ("runtimeFootprintCpuGb", "runtimeFootprintGb"),
}

# KV bytes per token (native f16 cache) for a model that is not on disk, by parameter count:
# layers x KV heads x head dim x 2 (K and V) x 2 bytes for typical grouped-query models. Hybrid and
# sliding-window models cache less, so this leans safe.
_KV_BYTES_PER_TOKEN_BY_PARAMS: tuple[tuple[float, int], ...] = (
    (3.0, 28 * 4 * 128 * 4),
    (9.0, 36 * 8 * 128 * 4),
    (20.0, 40 * 8 * 128 * 4),
    (40.0, 64 * 8 * 128 * 4),
    (80.0, 80 * 8 * 128 * 4),
    (math.inf, 96 * 8 * 128 * 4),
)

# Bytes per weight by quantisation label, used only when a row has no file size.
_BYTES_PER_WEIGHT: tuple[tuple[tuple[str, ...], float], ...] = (
    (("1-bit", "q1_"), 0.15),
    (("2-bit", "q2_", "ternary"), 0.30),
    (("3-bit", "q3_", "dq3"), 0.44),
    (("4-bit", "q4_", "iq4", "mxfp4", "nf4", "ud-q4"), 0.58),
    (("5-bit", "q5_"), 0.68),
    (("6-bit", "q6_"), 0.80),
    (("8-bit", "q8_", "fp8"), 1.06),
)

_ACTIVE_PARAMS_RE = re.compile(r"[-_ .]a(\d+(?:\.\d+)?)b(?![a-z0-9])")
_RELEASE_RE = re.compile(r"(\d{4})(?:-(\d{1,2}))?")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 and math.isfinite(value) else None


def _release_key(variant: dict[str, Any]) -> tuple[int, int]:
    match = _RELEASE_RE.match(str(variant.get("releaseDate") or "").strip())
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def _context_label(tokens: int) -> str:
    return f"{round(tokens / 1024)}K" if tokens >= 1024 else str(tokens)


def classify_share(need_gb: float, usable_gb: float) -> str:
    """``fits`` / ``tight`` / ``wont-fit`` for ``need_gb`` against ``usable_gb``."""
    share = need_gb / usable_gb
    if share <= FITS_MAX_SHARE + _SHARE_EPSILON:
        return "fits"
    if share <= TIGHT_MAX_SHARE + _SHARE_EPSILON:
        return "tight"
    return "wont-fit"


# ---------------------------------------------------------------------------
# platform gate (mirrors src/utils/platform.ts)
# ---------------------------------------------------------------------------

def platform_gate(variant: dict[str, Any], kind: str) -> str:
    """``apple-silicon`` / ``cuda`` / ``any``: the hardware a variant's engine needs."""
    if kind == "chat":
        backend = str(variant.get("backend") or "").strip().lower()
        if backend in {"mlx", "mlx-lm", "mtplx", "tensorfold"}:
            return "apple-silicon"
        if backend == "vllm":
            return "cuda"
        return "any"
    runtime = str(variant.get("runtime") or "").strip().lower()
    engine = str(variant.get("engine") or "").strip().lower()
    repo = str(variant.get("repo") or "").strip().lower()
    tags = {str(tag).lower() for tag in (variant.get("styleTags") or [])}
    if (
        variant.get("mlxOnly") is True
        or engine in {"mflux", "mlx-video"}
        or "mflux" in runtime
        or "mlx-video" in runtime
        or "mlx native" in runtime
        or repo.startswith("prince-canuma/")
        or "apple-silicon" in tags
    ):
        return "apple-silicon"
    if "nunchaku" in runtime or "longlive" in runtime or repo.startswith("nvlabs/longlive") or "cuda" in tags:
        return "cuda"
    return "any"


def _platform_gap_reason(gate: str, hardware: HardwareClass) -> str | None:
    """Why a variant of this gate cannot run on ``hardware``, or None when it can."""
    if gate == "apple-silicon" and hardware.kind != "apple":
        return "Needs Apple Silicon (MLX); this machine cannot run it."
    if gate == "cuda" and hardware.kind != "cuda":
        where = "this Mac has none" if hardware.kind == "apple" else "no NVIDIA GPU was found here"
        return f"Needs an NVIDIA GPU (CUDA); {where}."
    return None


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def _bytes_per_weight(quantization: str) -> float:
    lowered = quantization.lower()
    for markers, factor in _BYTES_PER_WEIGHT:
        if any(marker in lowered for marker in markers):
            return factor
    return 2.0  # BF16 / FP16 / unknown


def _chat_weights_gb(variant: dict[str, Any]) -> float | None:
    size = _positive(variant.get("sizeGb"))
    if size:
        return size
    params = _positive(variant.get("paramsB"))
    return params * _bytes_per_weight(str(variant.get("quantization") or "")) if params else None


def _params_equivalent_b(variant: dict[str, Any], weights_gb: float) -> float:
    """Parameter count for the cache table; rows with an unknown total derive it from the file size."""
    return _positive(variant.get("paramsB")) or weights_gb / 0.58


def _kv_bytes_per_token(variant: dict[str, Any], weights_gb: float) -> int:
    params = _params_equivalent_b(variant, weights_gb)
    for limit, per_token in _KV_BYTES_PER_TOKEN_BY_PARAMS:
        if params <= limit:
            return per_token
    return _KV_BYTES_PER_TOKEN_BY_PARAMS[-1][1]


def _chat_context(variant: dict[str, Any], context_tokens: int) -> int:
    window = variant.get("maxContext")
    if isinstance(window, (int, float)) and not isinstance(window, bool) and window > 0:
        return int(min(context_tokens, window))
    return context_tokens


def _chat_cache_gb(variant: dict[str, Any], weights_gb: float, arch: Any, context_tokens: int) -> float:
    if arch is not None:
        return arch.cache_bytes(context_tokens) / 1024 ** 3
    return _kv_bytes_per_token(variant, weights_gb) * context_tokens / 1024 ** 3


def _chat_fit_numbers(
    variant: dict[str, Any],
    arch: Any,
    explicit_memory_gb: float | None,
    context_tokens: int,
) -> tuple[float, str] | None:
    """``(need GB, reason)`` for a chat variant, or None without any size information."""
    weights = _chat_weights_gb(variant)
    explicit = _positive(explicit_memory_gb)
    if weights is None and explicit is None:
        return None
    ctx = _chat_context(variant, context_tokens)
    reference = weights if weights is not None else explicit
    cache = _chat_cache_gb(variant, reference, arch, ctx)
    label = _context_label(ctx)
    if explicit is not None:
        return explicit + cache, f"Model {explicit:.1f} GB (catalog estimate) + {cache:.1f} GB cache at {label} context"
    runtime = weights * CHAT_RUNTIME_SHARE
    return (
        weights + runtime + cache,
        f"Weights {weights:.1f} GB + {runtime:.1f} GB runtime + {cache:.1f} GB cache at {label} context",
    )


def local_kv_architecture(repo: str) -> Any:
    """Attention layout of a downloaded repo from its own files, or None (never a network call)."""
    if not repo:
        return None
    try:
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir
        from backend_service.helpers.model_arch import read_kv_architecture

        snapshot = _hf_repo_snapshot_dir(repo)
        return read_kv_architecture(snapshot) if snapshot is not None else None
    except Exception:
        return None


def _name_active_fraction(variant: dict[str, Any]) -> float | None:
    """Active share of a mixture-of-experts model from its name (``...-26B-A4B-...``)."""
    params = _positive(variant.get("paramsB"))
    if not params:
        return None
    text = f"{variant.get('name') or ''} {variant.get('repo') or ''}".lower()
    match = _ACTIVE_PARAMS_RE.search(text)
    if not match:
        return None
    return max(0.02, min(1.0, float(match.group(1)) / params))


def estimate_chat_tok_s(variant: dict[str, Any], *, arch: Any, bandwidth_gbps: float) -> float | None:
    """Decode speed from memory bandwidth, the same estimate the launch preview shows."""
    from backend_service.helpers.cache import estimate_decode_tok_s

    weights = _chat_weights_gb(variant)
    if weights is None:
        return None
    if arch is not None and arch.active_fraction < 1.0:
        active = arch.active_fraction
    else:
        active = _name_active_fraction(variant) or 1.0
    kv_read = _chat_cache_gb(variant, weights, arch, _SPEED_KV_TOKENS) * 1024 ** 3
    return estimate_decode_tok_s(
        weights_gb=weights,
        active_fraction=active,
        kv_bytes_per_token_read=kv_read,
        bandwidth_gbps=bandwidth_gbps,
    )


# ---------------------------------------------------------------------------
# image and video
# ---------------------------------------------------------------------------

def _media_footprint(variant: dict[str, Any], hardware: HardwareClass) -> tuple[float, str, str] | None:
    """``(GB, reason, basis)`` for an image or video variant, or None without any size information."""
    variant_id = str(variant.get("id") or "")
    for key in _MEDIA_FOOTPRINT_KEYS.get(hardware.kind, _MEDIA_FOOTPRINT_KEYS["cpu"]):
        value = _positive(variant.get(key))
        if value is None:
            continue
        measured = hardware.kind == "apple" and (
            MEASURED_FOOTPRINTS.get(variant_id) == key or variant.get("footprintBasis") == "measured"
        )
        if measured:
            return value, f"Peak memory {value:.1f} GB, measured on an Apple Silicon Mac", "measured"
        return value, f"Peak memory about {value:.1f} GB (catalog estimate)", "estimated"
    size = _positive(variant.get("sizeGb"))
    if size is None:
        return None
    return (
        size * MEDIA_SIZE_FACTOR,
        f"Estimated from the {size:.1f} GB download (x{MEDIA_SIZE_FACTOR:g} for the pipeline)",
        "estimated",
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _unknown(hardware: HardwareClass, reason: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "needGb": None,
        "usableGb": round(hardware.usable_gb, 1),
        "headroomPct": None,
        "reason": reason,
        "basis": "estimated",
    }


def variant_fit(
    variant: dict[str, Any],
    kind: str,
    hardware: HardwareClass,
    *,
    arch: Any = None,
    explicit_memory_gb: float | None = None,
    context_tokens: int = DEFAULT_CHAT_CONTEXT_TOKENS,
) -> dict[str, Any]:
    """The ``fit`` dict for one catalog variant (``kind`` is ``chat``, ``image`` or ``video``).

    ``arch`` is the model's attention layout when known (chat only); ``explicit_memory_gb`` is a chat
    row's own memory figure, which replaces the weights-plus-runtime part.
    """
    gap = _platform_gap_reason(platform_gate(variant, kind), hardware)
    if gap:
        return _unknown(hardware, gap)
    usable = hardware.usable_gb
    if usable <= 0:
        return _unknown(hardware, "This machine's usable memory could not be read.")

    if kind == "chat":
        numbers = _chat_fit_numbers(variant, arch, explicit_memory_gb, context_tokens)
        if numbers is None:
            return _unknown(hardware, "No size information for this model.")
        need, reason = numbers
        basis = "estimated"
    else:
        footprint = _media_footprint(variant, hardware)
        if footprint is None:
            return _unknown(hardware, "No size information for this model.")
        need, reason, basis = footprint

    status = classify_share(need, usable)
    if status != "fits" and hardware.kind == "cuda":
        reason += " (GPU memory only; offloading to system RAM would run far slower)"
    return {
        "status": status,
        "needGb": round(need, 1),
        "usableGb": round(usable, 1),
        "headroomPct": round((usable - need) / usable * 100),
        "reason": reason,
        "basis": basis,
    }


def best_fit_variant_id(
    variants: list[dict[str, Any]],
    kind: str,
    *,
    speed_for: Callable[[dict[str, Any]], float | None] | None = None,
) -> str | None:
    """Id of the variant to recommend, or None when none ``fits``.

    Newest ``releaseDate`` among the variants that fit. Ties: for chat the larger model (``paramsB``;
    a family that lists several sizes released together should recommend the most capable one that
    fits comfortably, not the smallest), then the faster, then the smaller; for image and video the
    shorter ``estimatedGenerationSeconds``, then the smaller, then the id.
    """
    fitting = [v for v in variants if (v.get("fit") or {}).get("status") == "fits"]
    if not fitting:
        return None
    newest = max(_release_key(v) for v in fitting)
    tied = [v for v in fitting if _release_key(v) == newest]
    if len(tied) == 1:
        return str(tied[0].get("id"))

    def speed_rank(variant: dict[str, Any]) -> float:
        if kind == "chat":
            try:
                speed = speed_for(variant) if speed_for is not None else None
            except Exception:
                speed = None
            return -speed if speed else math.inf
        seconds = _positive(variant.get("estimatedGenerationSeconds"))
        return seconds if seconds is not None else math.inf

    def capability_rank(variant: dict[str, Any]) -> float:
        if kind != "chat":
            return 0.0
        return -(_positive(variant.get("paramsB")) or 0.0)

    def order(variant: dict[str, Any]) -> tuple[float, float, float, float, str]:
        return (
            capability_rank(variant),
            speed_rank(variant),
            variant["fit"].get("needGb") or math.inf,
            _positive(variant.get("sizeGb")) or math.inf,
            str(variant.get("id")),
        )

    return str(min(tied, key=order).get("id"))


def annotate_fit(
    families: list[dict[str, Any]],
    kind: str,
    hardware: HardwareClass | None = None,
) -> list[dict[str, Any]]:
    """Attach ``fit`` to every image or video variant and ``bestFitVariantId`` to every family.

    Mutates the dicts it is given (the payload builders create them per request) and returns the
    list. Never raises: a failure leaves the payload as it was and is logged.
    """
    try:
        hw = hardware or host_hardware_class()
        for family in families:
            for variant in family.get("variants") or []:
                variant["fit"] = variant_fit(variant, kind, hw)
            family["bestFitVariantId"] = best_fit_variant_id(family.get("variants") or [], kind)
    except Exception:
        logger.warning("Could not compute model fit for the %s catalog", kind, exc_info=True)
    return families


def annotate_variants(
    variants: list[dict[str, Any]],
    kind: str,
    hardware: HardwareClass | None = None,
) -> list[dict[str, Any]]:
    """Attach ``fit`` to a flat list of variants (the "latest" rows, which have no family)."""
    try:
        hw = hardware or host_hardware_class()
        for variant in variants:
            variant["fit"] = variant_fit(variant, kind, hw)
    except Exception:
        logger.warning("Could not compute model fit for the %s list", kind, exc_info=True)
    return variants


def annotate_chat_families(
    families: list[dict[str, Any]],
    hardware: HardwareClass,
    *,
    explicit_memory: dict[str, float] | None = None,
    arch_for: Callable[[str], Any] | None = None,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None = None,
) -> list[dict[str, Any]]:
    """Attach ``fit`` / ``bestFitVariantId`` to the chat catalog.

    ``explicit_memory`` maps variant id to a row's own ``estimatedMemoryGb``; ``arch_for`` maps a repo
    to its attention layout (default: the downloaded files, never the network); ``measured_speed_for``
    returns this machine's median tok/s for a variant, used to break ties before the bandwidth estimate.
    """
    lookup = arch_for if arch_for is not None else local_kv_architecture
    explicit = explicit_memory or {}
    try:
        bandwidth: list[float] = []

        def bandwidth_gbps() -> float:
            if not bandwidth:
                from backend_service.helpers.bandwidth import memory_bandwidth

                bandwidth.append(memory_bandwidth()[0])
            return bandwidth[0]

        for family in families:
            archs: dict[str, Any] = {}
            variants = family.get("variants") or []
            for variant in variants:
                variant_id = str(variant.get("id") or "")
                runnable = _platform_gap_reason(platform_gate(variant, "chat"), hardware) is None
                arch = None
                if runnable:
                    try:
                        arch = lookup(str(variant.get("repo") or ""))
                    except Exception:
                        arch = None
                archs[variant_id] = arch
                variant["fit"] = variant_fit(
                    variant, "chat", hardware, arch=arch, explicit_memory_gb=explicit.get(variant_id),
                )

            def speed_for(variant: dict[str, Any], _archs: dict[str, Any] = archs) -> float | None:
                if measured_speed_for is not None:
                    try:
                        measured = measured_speed_for(variant)
                    except Exception:
                        measured = None
                    if measured:
                        return measured
                return estimate_chat_tok_s(
                    variant, arch=_archs.get(str(variant.get("id") or "")), bandwidth_gbps=bandwidth_gbps(),
                )

            family["bestFitVariantId"] = best_fit_variant_id(variants, "chat", speed_for=speed_for)
    except Exception:
        logger.warning("Could not compute model fit for the chat catalog", exc_info=True)
    return families
