"""Expert offload for mixture-of-experts GGUFs on a discrete NVIDIA GPU (FU-550).

A mixture-of-experts model reads only a few experts per token, so its routed expert tensors can stay in system RAM
and run on the CPU while attention, the shared experts and the router stay on the card. llama.cpp does this with
``--n-cpu-moe N`` (the experts of blocks ``0 .. N-1`` stay in system memory) and ``--cpu-moe`` (all of them).

Measured on an RTX 4090 with llama.cpp b12649, Qwen3.6-35B-A3B Q4_K_M (19.1 GiB file, 16.9 GiB of experts over 40
blocks), 8K context, f16 cache, flash attention: all on the card 20.0 GB of VRAM and 140 to 165 tok/s; 20 blocks'
experts in RAM 11.6 GB and 84 tok/s (with ``--load-mode none``; 72 to 83 through mmap); every expert in RAM 3.0 GB
and 52 tok/s. The card part matched ``gpu_need`` below to within 1 %.

The split is read from the file's tensor table (``helpers.gguf_layout``). A catalog row that is not downloaded yet
is estimated from its size and its active share (``estimate_split``).
"""
from __future__ import annotations

import math
from typing import Any

from backend_service.helpers.chat_memory import llama_gpu_need
from backend_service.helpers.gguf_layout import GgufLayout

_GIB = 1024 ** 3
# Free memory llama.cpp's own ``--fit`` leaves on the card (its default ``--fit-target``, 1024 MiB).
GPU_MARGIN_GB = 1.0
# Share of system RAM the experts kept there may take: the OS, the app and the page cache need the rest.
RAM_SHARE = 0.8
# A file's dense part (attention, shared experts, router, embeddings) against its active share. Qwen3.6-35B-A3B:
# 2.2 GiB dense of 19.1 (active share 0.086, so 1.34x); Nemotron 3.5 Lightning 30B-A3B: 1.6 of 17.7 (0.1, 0.9x).
_DENSE_PER_ACTIVE = 1.5
# Card memory beyond weights and cache for a file that is not on disk: compute buffer and CUDA context (0.3 GB
# measured on Qwen3.6-35B-A3B, 248K vocabulary). Not the 10 % runtime share of the dense estimate, which on a
# 67 GB file would count 6.7 GB of buffers that never exist.
ESTIMATE_OVERHEAD_GB = 1.0


def is_moe(layout: GgufLayout | None) -> bool:
    return layout is not None and layout.expert_bytes > 0


def block_count(layout: GgufLayout) -> int:
    """The highest ``--n-cpu-moe`` that changes anything (every block with experts is then in RAM)."""
    return len(layout.expert_bytes_by_layer)


def moe_layer_count(layout: GgufLayout) -> int:
    """Blocks that hold routed experts (a few leading blocks of some models are dense)."""
    return sum(1 for size in layout.expert_bytes_by_layer if size > 0)


def offloaded_bytes(layout: GgufLayout, cpu_layers: int) -> int:
    return sum(layout.expert_bytes_by_layer[: max(0, cpu_layers)])


def split(layout: GgufLayout, cache_bytes: float, cpu_layers: int, *, turbo: bool = False) -> dict[str, Any]:
    """GPU and system-RAM GB of one llama-server with the experts of the first ``cpu_layers`` blocks in RAM."""
    parts = llama_gpu_need(layout, cache_bytes, turbo=turbo)
    moved = offloaded_bytes(layout, cpu_layers) / _GIB
    return {
        "cpuMoeLayers": max(0, min(cpu_layers, block_count(layout))),
        "moeLayers": moe_layer_count(layout),
        "blocks": block_count(layout),
        "gpuGb": parts["needGb"] - moved,
        "ramGb": parts["hostGb"] + moved,
        "expertGb": layout.expert_bytes / _GIB,
    }


def auto_cpu_layers(
    layout: GgufLayout | None, cache_bytes: float, gpu_budget_gb: float, *, turbo: bool = False,
) -> int | None:
    """Fewest leading blocks whose experts go to RAM so the card part fits ``gpu_budget_gb``.

    0 when the whole model fits already; None for a dense model, or when even every expert in RAM leaves more than
    the budget on the card.
    """
    if not is_moe(layout):
        return None
    assert layout is not None
    need = llama_gpu_need(layout, cache_bytes, turbo=turbo)["needGb"]
    if need <= gpu_budget_gb:
        return 0
    for index, size in enumerate(layout.expert_bytes_by_layer):
        need -= size / _GIB
        if need <= gpu_budget_gb:
            return index + 1
    return None


def estimate_split(
    *, need_gb: float, weights_gb: float, active_fraction: float, gpu_budget_gb: float, num_layers: int | None,
) -> dict[str, Any] | None:
    """The same split for a file that is not on disk, from its size and active share. None when the dense part
    alone does not fit the budget. Experts are taken as spread evenly over ``num_layers`` blocks."""
    dense = weights_gb * min(1.0, active_fraction * _DENSE_PER_ACTIVE)
    experts = weights_gb - dense
    if experts <= 0 or need_gb - experts > gpu_budget_gb:
        return None
    moved = max(0.0, need_gb - gpu_budget_gb)
    result: dict[str, Any] = {
        "cpuMoeLayers": None,
        "moeLayers": num_layers,
        "blocks": num_layers,
        "gpuGb": need_gb - moved,
        "ramGb": moved,
        "expertGb": experts,
    }
    if num_layers:
        result["cpuMoeLayers"] = min(num_layers, math.ceil(moved / (experts / num_layers) - 1e-9))
    return result
