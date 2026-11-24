from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from backend_service.app import compute_cache_preview

router = APIRouter()

_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def _allowed_model_path(request: Request, model_path: str | None) -> bool:
    """True when ``model_path`` sits inside an enabled model directory or the Hugging Face cache."""
    from backend_service.helpers.hf_cache_paths import _hf_hub_cache_root

    if not model_path:
        return False
    try:
        target = Path(os.path.expanduser(model_path)).resolve()
    except (OSError, RuntimeError):
        return False
    roots: list[Path] = [_hf_hub_cache_root()]
    for directory in request.app.state.chaosengine.settings.get("modelDirectories") or []:
        raw = str(directory.get("path") or "")
        if directory.get("enabled", True) and raw:
            roots.append(Path(os.path.expanduser(raw)))
    for root in roots:
        try:
            resolved_root = root.resolve()
        except (OSError, RuntimeError):
            continue
        if target == resolved_root or resolved_root in target.parents:
            return True
    return False


def _chat_host(request: Request, model_path: str | None, model_repo: str | None, arch: Any) -> dict[str, Any]:
    """Which lane and host the chat estimate is for: ``lane``, ``cuda`` (a discrete NVIDIA card) and the GGUF
    ``layout`` when the file can be read. Never raises."""
    from backend_service.helpers.chat_memory import gguf_file_for
    from backend_service.helpers.gguf_layout import gguf_layout
    from backend_service.helpers.hardware_class import host_hardware_class

    try:
        hardware = host_hardware_class()
    except Exception:  # noqa: BLE001
        hardware = None
    gguf = gguf_file_for(model_path) if _allowed_model_path(request, model_path) else None
    if gguf is None and model_repo and _REPO_ID.fullmatch(model_repo):
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

        snapshot = _hf_repo_snapshot_dir(model_repo)
        gguf = gguf_file_for(snapshot) if snapshot is not None else None
    off_mac = hardware is not None and hardware.kind != "apple"
    gguf_model = gguf is not None or getattr(arch, "source", None) == "gguf"
    return {
        "lane": "llama.cpp" if gguf_model or off_mac else None,
        "cuda": hardware is not None and hardware.kind == "cuda" and not hardware.unified,
        "layout": gguf_layout(gguf) if gguf is not None else None,
    }


def _model_arch(request: Request, model_path: str | None, model_repo: str | None):
    """Real attention layout of a local model, or None to fall back to the estimate.

    ``model_path`` must sit inside an enabled model directory or the Hugging
    Face cache, so the endpoint never reads files elsewhere on disk.
    ``model_repo`` is resolved to its downloaded snapshot, if there is one.
    """
    from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir
    from backend_service.helpers.model_arch import hub_kv_architecture, read_kv_architecture

    if _allowed_model_path(request, model_path):
        found = read_kv_architecture(model_path)
        if found is not None:
            return found
    if model_repo and _REPO_ID.fullmatch(model_repo):
        snapshot = _hf_repo_snapshot_dir(model_repo)
        found = read_kv_architecture(snapshot) if snapshot is not None else None
        # Not downloaded yet: read its config from the Hub, so the estimate is
        # exact while the user is still deciding whether to download.
        return found or hub_kv_architecture(model_repo)
    return None


@router.get("/api/cache/preview")
def cache_preview(
    request: Request,
    bits: int = Query(3, ge=0, le=8),
    fp16_layers: int = Query(4, ge=0, le=16),
    num_layers: int = Query(32, ge=1, le=160),
    num_heads: int = Query(32, ge=1, le=256),
    num_kv_heads: int = Query(0, ge=0, le=256),
    hidden_size: int = Query(4096, ge=256, le=32768),
    context_tokens: int = Query(8192, ge=256, le=4_194_304),
    # Down to 0.05: a 0.5B model reports 0.48 and was refused (422), so the launch dialog had no preview.
    params_b: float = Query(7.0, ge=0.05, le=1000.0),
    strategy: str = Query("native"),
    model_path: str | None = Query(None, max_length=4096),
    model_repo: str | None = Query(None, max_length=256),
    weights_gb: float | None = Query(None, gt=0, le=4000),
) -> dict[str, Any]:
    from backend_service.helpers.speed_history import measured_decode_tok_s

    state = request.app.state.chaosengine
    measured = measured_decode_tok_s(
        state.chat_sessions,
        state.benchmark_runs,
        model_path=model_path,
        model_repo=model_repo,
        strategy="native" if bits <= 0 else strategy,
    )
    arch = _model_arch(request, model_path, model_repo)
    host = _chat_host(request, model_path, model_repo, arch)
    preview = compute_cache_preview(
        bits=bits,
        fp16_layers=fp16_layers,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads or None,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        params_b=params_b,
        strategy=strategy,
        arch=arch,
        weights_gb=weights_gb,
        measured_tok_s=measured,
        **host,
    )
    preview["moeOffload"] = _moe_offload_preview(host, float(preview.get("optimizedCacheGb") or 0.0))
    preview["memory"] = _memory_budget()
    preview["tensorfoldFit"] = _tensorfold_native_fit(model_repo, context_tokens)
    return preview


def _memory_budget() -> dict[str, Any] | None:
    """The ceiling a launch has to fit under on this machine, for the launch dialog's memory bar and its
    TensorFold window cap: the Metal working-set limit on a Mac (51.8 GB of 64 GB, what MLX and TensorFold stay
    under), the card's VRAM on an NVIDIA host, else 80 % of system RAM. ``freeGb`` is the card's free VRAM on
    NVIDIA only. Never raises."""
    from backend_service.helpers.hardware_class import _metal_info, host_hardware_class
    from backend_service.helpers.memory_gate import cuda_free_vram_gb

    try:
        kind = host_hardware_class().kind
        if kind == "apple":
            info = _metal_info()
            if info is not None:
                return {"kind": "metal", "limitGb": round(info[1], 1), "totalGb": round(info[0], 1), "freeGb": None}
        if kind == "cuda":
            from backend_service.helpers.gpu import get_device_vram_total_gb

            total = get_device_vram_total_gb()
            free = cuda_free_vram_gb()
            if total:
                return {"kind": "vram", "limitGb": round(float(total), 1), "totalGb": round(float(total), 1),
                        "freeGb": round(free, 1) if free is not None else None}
        import psutil

        ram = psutil.virtual_memory().total / 1024 ** 3
        return {"kind": "ram", "limitGb": round(ram * 0.8, 1), "totalGb": round(ram, 1), "freeGb": None}
    except Exception:  # noqa: BLE001
        return None


def _tensorfold_native_fit(model_repo: str | None, context_tokens: int) -> dict[str, Any] | None:
    """What the TensorFold 1.x binary needs on this NVIDIA card at ``context_tokens``, from its measured VRAM
    figures, and the largest window that fits the free memory now. None off a CUDA host, for a checkpoint the
    registry does not list, or when the free memory cannot be read. The Mac's Metal lane is sized in the
    launch dialog instead."""
    from backend_service.helpers.memory_gate import cuda_free_vram_gb
    from backend_service.inference import _tensorfold_native as native

    if not model_repo or not _REPO_ID.fullmatch(model_repo):
        return None
    model = native.match_native_model(model_repo)
    free = cuda_free_vram_gb()
    if model is None or free is None:
        return None
    drafts = bool(model.drafter)
    need = native.estimate_vram_gb(model, context_tokens, drafts=drafts)
    return {
        "needGb": need,
        "freeGb": round(free, 1),
        "fits": need <= free,
        "maxContext": native.max_context_for_vram(model, free, drafts=drafts, ceiling=context_tokens),
    }


def _moe_offload_preview(host: dict[str, Any], cache_gb: float) -> dict[str, Any] | None:
    """What the launch settings' expert offload control shows for a mixture-of-experts GGUF on a discrete NVIDIA
    card (FU-550): the layer count and how many layers' experts would stay in system RAM to fit the card's free
    memory now. None for a dense model, a Mac, or a file that cannot be read."""
    from backend_service.helpers.memory_gate import cuda_free_vram_gb
    from backend_service.helpers.moe_offload import GPU_MARGIN_GB, auto_cpu_layers, is_moe, split

    layout = host.get("layout")
    if not host.get("cuda") or not is_moe(layout):
        return None
    free_gpu = cuda_free_vram_gb()
    auto = auto_cpu_layers(layout, cache_gb * 1024 ** 3, free_gpu - GPU_MARGIN_GB) if free_gpu is not None else None
    parts = split(layout, cache_gb * 1024 ** 3, auto or 0)
    return {
        "blocks": parts["blocks"],
        "moeLayers": parts["moeLayers"],
        "expertGb": round(parts["expertGb"], 1),
        "autoCpuLayers": auto,
        "gpuGb": round(parts["gpuGb"], 1),
        "ramGb": round(parts["ramGb"], 1),
        "freeGpuGb": free_gpu,
    }


# A loaded model holds a little more than its files (runtime buffers, the
# compute graph); this share is added to the weights.
_LOAD_OVERHEAD = 0.1
_LOAD_HEADROOM_GB = 2.0


@router.get("/api/models/memory-check")
def model_memory_check(
    request: Request,
    weights_gb: float = Query(..., gt=0, le=4000),
    context_tokens: int = Query(8192, ge=256, le=4_194_304),
    bits: int = Query(0, ge=0, le=8),
    fp16_layers: int = Query(0, ge=0, le=16),
    strategy: str = Query("native"),
    params_b: float = Query(7.0, ge=0.5, le=1000.0),
    num_layers: int = Query(32, ge=1, le=160),
    num_heads: int = Query(32, ge=1, le=256),
    num_kv_heads: int = Query(0, ge=0, le=256),
    hidden_size: int = Query(4096, ge=256, le=32768),
    model_path: str | None = Query(None, max_length=4096),
    model_repo: str | None = Query(None, max_length=256),
) -> dict[str, Any]:
    """Will this chat model fit right now? Advisory: the app asks, never refuses.

    Needed: the weights plus a runtime share, plus the KV cache at the chosen
    context (from the model's real layout when it can be read). Available:
    free memory now (the macOS-corrected figure the memory gates use) plus
    what the loaded chat models hold, since loading parks or unloads them.
    """
    import psutil

    from backend_service.helpers.memory_gate import available_memory_gb, cuda_free_vram_gb

    arch = _model_arch(request, model_path, model_repo)
    host = _chat_host(request, model_path, model_repo, arch)
    preview = compute_cache_preview(
        bits=bits,
        fp16_layers=fp16_layers,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads or None,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        params_b=params_b,
        strategy=strategy,
        arch=arch,
        weights_gb=weights_gb,
        **host,
    )
    state = request.app.state.chaosengine
    try:
        reclaimable_gb = state.runtime.reclaimable_model_bytes() / 1024 ** 3
    except Exception:
        reclaimable_gb = 0.0
    memory = psutil.virtual_memory()
    free_gb = available_memory_gb(memory.total, memory.available)
    cache_gb = float(preview["optimizedCacheGb"])
    free_gpu_gb = cuda_free_vram_gb() if host["cuda"] else None
    return _memory_verdict(
        weights_gb=weights_gb,
        cache_gb=cache_gb,
        context_tokens=context_tokens,
        free_gb=free_gb,
        reclaimable_gb=reclaimable_gb,
        free_gpu_gb=free_gpu_gb,
        layout=host["layout"],
        arch_source=preview["archSource"],
        turbo=strategy == "turboquant" and bits > 0,
    )


def _memory_verdict(
    *,
    weights_gb: float,
    cache_gb: float,
    context_tokens: int,
    free_gb: float,
    reclaimable_gb: float,
    free_gpu_gb: float | None,
    layout: Any,
    arch_source: str,
    turbo: bool = False,
) -> dict[str, Any]:
    """The memory-check answer. ``free_gpu_gb`` is set on a discrete NVIDIA card: llama.cpp puts the model on the
    card, so the card's free memory (plus what loaded chat models hold there) is what decides speed; system RAM
    then only has to hold the embeddings llama.cpp keeps on the CPU. Elsewhere one pool is judged, as before."""
    from backend_service.helpers.chat_memory import llama_gpu_need

    def verdict_for(need: float, available: float) -> str:
        budget = max(0.0, available - _LOAD_HEADROOM_GB)
        if need <= budget * 0.85:
            return "fits"
        return "tight" if need <= budget else "may_not_fit"

    result: dict[str, Any] = {
        "freeGb": round(free_gb, 1),
        "reclaimableGb": round(reclaimable_gb, 1),
        "cacheGb": cache_gb,
        "archSource": arch_source,
        "memoryKind": "system",
        "message": None,
    }
    if free_gpu_gb is not None:
        if layout is not None:
            parts = llama_gpu_need(layout, cache_gb * 1024 ** 3, turbo=turbo)
            needed_gb, host_gb = parts["needGb"], parts["hostGb"]
        else:
            needed_gb, host_gb = weights_gb * (1 + _LOAD_OVERHEAD) + cache_gb, 0.0
        available_gb = free_gpu_gb + reclaimable_gb
        verdict = verdict_for(needed_gb, available_gb)
        offload = None
        if verdict == "may_not_fit":
            offload = _moe_offload_verdict(
                layout, cache_gb, turbo, gpu_budget_gb=available_gb - _LOAD_HEADROOM_GB,
                ram_budget_gb=free_gb + reclaimable_gb - _LOAD_HEADROOM_GB,
            )
        if offload is not None:
            # A mixture of experts loads with some experts in system RAM (FU-550): slower, but it fits.
            needed_gb, host_gb, verdict = offload["gpuGb"], offload["ramGb"], "tight"
            result["moeOffload"] = offload
        result.update(memoryKind="gpu", freeGpuGb=round(free_gpu_gb, 1), hostGb=round(host_gb, 1))
        if verdict == "may_not_fit":
            result["message"] = (
                f"This model needs about {needed_gb:.0f} GB of GPU memory (weights plus the cache at "
                f"{context_tokens:,} tokens), but about {available_gb:.0f} GB of the card can be made free "
                "right now. It still loads, with some layers kept in system memory and run on the CPU, but "
                "replies come several times slower. Lower the context, pick a smaller quantisation, or unload "
                "other models from the GPU."
            )
        elif host_gb > max(0.0, free_gb + reclaimable_gb - _LOAD_HEADROOM_GB):
            verdict = "may_not_fit"
            result["message"] = (
                f"This model keeps about {host_gb:.0f} GB of embeddings in system memory, but only about "
                f"{free_gb:.0f} GB is free. Close other apps, then try again."
            )
    else:
        needed_gb = weights_gb * (1 + _LOAD_OVERHEAD) + cache_gb
        available_gb = free_gb + reclaimable_gb
        verdict = verdict_for(needed_gb, available_gb)
        if verdict == "may_not_fit":
            result["message"] = (
                f"This model needs about {needed_gb:.0f} GB (weights plus the cache at "
                f"{context_tokens:,} tokens), but about {available_gb:.0f} GB can be made free "
                "right now. It may load into swap and run very slowly, or fail. Lower the "
                "context, pick a smaller quantisation, or close other apps."
            )
    result.update(verdict=verdict, neededGb=round(needed_gb, 1), availableGb=round(available_gb, 1))
    return result


def _moe_offload_verdict(
    layout: Any, cache_gb: float, turbo: bool, *, gpu_budget_gb: float, ram_budget_gb: float,
) -> dict[str, Any] | None:
    """The expert offload a mixture-of-experts GGUF needs to fit ``gpu_budget_gb``, when system RAM can hold the
    moved experts; None otherwise (a dense model, or too little RAM)."""
    from backend_service.helpers.moe_offload import auto_cpu_layers, split

    layers = auto_cpu_layers(layout, cache_gb * 1024 ** 3, gpu_budget_gb, turbo=turbo) if layout is not None else None
    if not layers:
        return None
    parts = split(layout, cache_gb * 1024 ** 3, layers, turbo=turbo)
    if parts["ramGb"] > ram_budget_gb:
        return None
    return {key: (round(value, 1) if isinstance(value, float) else value) for key, value in parts.items()}
