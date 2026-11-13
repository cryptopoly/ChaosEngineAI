"""Chat memory and speed figures that differ on a discrete NVIDIA GPU, measured on an RTX 4090 (2026-11-11).

A Mac has one pool of memory, so "weights + cache + a 10 % runtime share" against the Metal limit is the right
question there and is left alone. On a card with its own memory llama.cpp splits a GGUF: the input embeddings stay
in system memory and the rest goes to the card (``helpers.gguf_layout``), plus the KV cache, a compute buffer sized
by the vocabulary and the CUDA context. For Nemotron 3.5 Bolt E4B (5.74 GiB file) that is 3.6 GB of VRAM at 4K,
where the old weights-plus-10 % figure said 6.5 GB.

TurboQuant on llama.cpp (``llama-server-turbo``) compresses every layer to the ``turboN`` cache type; the MLX
lane's "keep N f16 layers" option does not exist there. The measured cache sizes are 3.125 bits per element for
turbo3 and 4.25 for turbo4 (the same ratio as q4_0's 4.5 to f16's 16); decode stayed at 0.82 to 0.87x of the f16
cache on CUDA (Qwen3-0.6B 466 against 565 tok/s, Bolt E4B 152 against 174), not the 0.13 to 0.18x the MLX lane
measured on an M4 Max.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend_service.helpers.gguf_layout import GgufLayout, gguf_layout

_GIB = 1024 ** 3
# CUDA context and allocator slack of one llama-server process beyond its buffers (about 0.1 GB measured).
CUDA_CONTEXT_GB = 0.15
# Bits per cached element of each llama.cpp cache type, read off the KV buffer sizes llama.cpp printed.
LLAMA_CACHE_BITS: dict[str, float] = {
    "f16": 16.0, "q8_0": 8.5, "q4_0": 4.5, "turbo2": 2.125, "turbo3": 3.125, "turbo4": 4.25,
}
# Decode speed of a turboN cache against f16 on CUDA, measured on Qwen3-0.6B and Bolt E4B at 4K to 64K.
TURBO_LLAMA_CUDA_SPEED_RATIO = 0.85
# Per-token cost beyond moving bytes on CUDA (kernel launches, sampling), per layer. With it the estimate reads
# 466 tok/s for Qwen3-0.6B Q8_0 (565 measured, -17 %) and 182 for Bolt E4B Q4_K_M (174 measured, +5 %); the Mac's
# flat 2 ms per token gave 316 and 112 (-44 % and -36 %).
CUDA_TOKEN_OVERHEAD_SECONDS_PER_LAYER = 0.00004


def turbo_llama_bits(bits: int) -> float:
    """Bits per element of the cache type the TurboQuant llama.cpp lane runs for ``bits`` (clamped 2 to 4)."""
    return LLAMA_CACHE_BITS[f"turbo{max(2, min(4, int(bits)))}"]


def gguf_file_for(path: str | Path | None) -> Path | None:
    """The ``.gguf`` a model path points at: the file itself, or the first main file in a folder."""
    if not path:
        return None
    target = Path(path).expanduser()
    try:
        if target.is_file():
            return target if target.suffix.lower() == ".gguf" else None
        if target.is_dir():
            from backend_service.helpers.model_arch import _first_gguf

            return _first_gguf(target)
    except OSError:
        return None
    return None


def local_variant_layout(repo: str, gguf_file: str | None) -> GgufLayout | None:
    """Layout of a catalog GGUF variant that is already downloaded (never a network call), else None."""
    if not repo:
        return None
    try:
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

        snapshot = _hf_repo_snapshot_dir(repo)
    except Exception:  # noqa: BLE001 - an unreadable cache is "unknown"
        return None
    if snapshot is None:
        return None
    if gguf_file:
        candidate = Path(snapshot) / gguf_file
        return gguf_layout(candidate) if candidate.is_file() else None
    found = gguf_file_for(snapshot)
    return gguf_layout(found) if found is not None else None


def llama_gpu_need(layout: GgufLayout, cache_bytes: float, *, turbo: bool = False) -> dict[str, Any]:
    """GPU memory one llama-server needs on a discrete card, with its parts in GB.

    ``turbo`` is the TurboQuant fork, which held about one more compute buffer than the standard build in every
    measured cell (Qwen3-0.6B +0.27 GB, Bolt E4B +0.55 GB): with it counted, the estimate is 1 to 9 % over.
    """
    weights = layout.gpu_bytes / _GIB
    cache = cache_bytes / _GIB
    compute = layout.compute_buffer_bytes() / _GIB * (2 if turbo else 1)
    return {
        "needGb": weights + cache + compute + CUDA_CONTEXT_GB,
        "weightsGpuGb": weights,
        "hostGb": layout.host_bytes / _GIB,
        "cacheGb": cache,
        "computeGb": compute + CUDA_CONTEXT_GB,
    }
