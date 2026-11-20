"""The lip-sync branch of POST /api/video/generate: a request with ``audioPath`` lands here (``lipsync_runtime``).

The video route has already found the row, checked the territory gate, resolved the start still and answered 409 for a
row that is not downloaded; it holds the ``video`` slot of the heavy-job queue, so progress and Cancel are the video
ones (``/api/video/progress``, ``/api/video/cancel``) and the Storyboard and agent callers work unchanged. This module
frees what else holds memory, runs the honest memory gate and saves the clip to the Video Gallery.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from backend_service.progress import VIDEO_PROGRESS, GenerationCancelled


def _free_memory(request: Request) -> None:
    """Unload idle image, music and speech models and the in-process video pipeline before the worker loads."""
    from backend_service.music_runtime import free_idle_worker
    from backend_service.routes.video import _unload_idle_image_runtime_for_video
    from backend_service.video_runtime import _release_gpu_speech

    state = request.app.state.chaosengine
    _unload_idle_image_runtime_for_video(request, "lip sync")
    free_idle_worker(state, "lip sync")
    _release_gpu_speech()
    try:
        loaded = str(state.video_runtime.capabilities().get("loadedModelRepo") or "")
        if loaded:
            state.video_runtime.unload(loaded)  # by repo: unload() with none would also stop the lip-sync worker
            state.add_log("video", "info", "Unloaded the video model before lip sync to free memory.")
    except Exception as exc:  # noqa: BLE001 - freeing memory is best effort; the gate still judges what is free
        state.add_log("video", "warning", f"Could not unload the video model before lip sync: {type(exc).__name__}: {exc}")


def _cuda_gate(request: Request, variant: dict[str, Any]) -> None:
    from backend_service import lipsync_runtime
    from backend_service.helpers.memory_gate import snapshot_memory_signals

    card = free = available = None
    try:
        from backend_service.helpers.gpu import get_device_vram_free_gb, get_device_vram_total_gb

        card, free = get_device_vram_total_gb(), get_device_vram_free_gb()
    except Exception:  # noqa: BLE001 - an unreadable GPU never blocks
        pass
    try:
        available, _pressure = snapshot_memory_signals()
    except Exception:  # noqa: BLE001
        pass
    refusal = lipsync_runtime.memory_refusal(
        need=lipsync_runtime.needs(variant), card_gb=card, free_vram_gb=free, available_ram_gb=available,
        worker_loaded=lipsync_runtime.loaded_repo() == variant.get("repo"),
    )
    if refusal:
        request.app.state.chaosengine.add_log("video", "warning", f"Lip sync refused: {refusal}")
        raise HTTPException(status_code=503, detail=refusal)


def generate_lipsync_locked(request: Request, body: Any, variant: dict[str, Any]) -> dict[str, Any]:
    from backend_service import lipsync_runtime
    from backend_service.app import _load_video_outputs, _save_video_artifact
    from backend_service.routes.video import _video_memory_gate

    state = request.app.state.chaosengine
    engine = lipsync_runtime.engine_for(variant)
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail=f"{variant.get('name')} cannot lip-sync. Pick LTX-2.3 distilled 4-bit (NVIDIA) or an LTX-2.3 MLX model (Mac).",
        )
    if not body.startImagePath:
        raise HTTPException(status_code=400, detail="Lip sync starts from a still: give startImagePath with the audio.")
    if body.endImagePath:
        raise HTTPException(status_code=400, detail="Lip sync takes a start still only; remove the end frame.")
    try:
        body = body.model_copy(update={"audioPath": str(lipsync_runtime.resolve_audio(body.audioPath))})
        lipsync_runtime.shot_frames(body.numFrames, body.fps)
    except lipsync_runtime.LipSyncError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    _free_memory(request)
    if engine == "cuda":
        _cuda_gate(request, variant)
    else:
        _video_memory_gate(request, variant, "lip sync", body)
    state.add_log("video", "info", f"Lip sync requested: {variant['name']}, {body.numFrames} frames @ {body.fps} fps.")
    try:
        fields = lipsync_runtime.generate(body, variant)
    except GenerationCancelled:
        state.add_log("video", "info", f"Lip sync cancelled for {variant['name']} by user.")
        raise HTTPException(status_code=409, detail="cancelled") from None
    except lipsync_runtime.LipSyncError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except Exception as exc:
        if VIDEO_PROGRESS.is_cancelled():
            raise HTTPException(status_code=409, detail="cancelled") from None
        state.add_log("video", "error", f"Lip sync failed for {variant['name']}: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Lip sync failed for {variant['name']}: {exc}") from exc
    artifact = _save_video_artifact(fields)
    state.add_log("video", "info", f"Lip-synced clip made with {variant['name']} in {artifact.get('durationSeconds')} s.")
    state.add_activity("Lip-sync clip generated", f"{variant['name']} · {artifact.get('numFrames')}f")
    runtime = {"activeEngine": "lipsync-" + engine, "loadedModelRepo": lipsync_runtime.loaded_repo()}
    return {"artifact": artifact, "outputs": _load_video_outputs(), "runtime": runtime}
