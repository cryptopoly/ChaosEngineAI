"""Stateless pipeline helpers for ``DiffusersVideoEngine``.

Four helpers lifted out of ``video_runtime/__init__.py``:

* ``make_step_callback`` — build the per-step callback the diffusers
  pipeline calls during sampling. Wires four concerns into one
  callback: progress reporting (``VIDEO_PROGRESS.set_step``),
  cooperative cancel (``GenerationCancelled``), Phase E2 CFG decay,
  and FU-018 part 2 preview VAE thumbnails.
* ``invoke_pipeline`` — run a diffusers video pipeline and return
  the first batch's frames; handles older-diffusers fallback for
  ``callback_on_step_end`` and ``negative_prompt`` rejection paths.
* ``encode_frames_to_mp4`` — write a list of PIL frames to mp4 bytes
  via ``diffusers.utils.export_to_video`` or the imageio fallback.
* ``pipeline_class_for_repo`` — look up the diffusers pipeline class
  for a repo via ``PIPELINE_REGISTRY``.

Extracted from ``backend_service/video_runtime/__init__.py`` as part
of the v0.8.0 Phase 1c-14 refactor.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any

from backend_service.progress import GenerationCancelled, VIDEO_PROGRESS
from backend_service.video_runtime.repos import PIPELINE_REGISTRY


def make_step_callback(
    total_steps: int,
    initial_guidance: float,
    cfg_decay: bool,
    preview_vae: bool = False,
) -> Any:
    """Build the per-step callback the pipeline calls during sampling.

    Wires four concerns into one callback:
      1. Progress reporting via ``VIDEO_PROGRESS.set_step``.
      2. Cooperative cancel — raise ``GenerationCancelled`` when the
         user hits Cancel on the modal.
      3. Phase E2 CFG decay — linearly ramp ``pipeline.guidance_scale``
         from ``initial_guidance`` at step 0 toward 1.0 at the last
         step. Flow-match video models (LTX, Wan, HunyuanVideo) tend
         to oversaturate when CFG is held high through the whole
         schedule; decaying lets the early steps lock semantics
         (high CFG) while late steps preserve fine detail (low CFG).
      4. FU-018 part 2 — when ``preview_vae`` is on, every Nth step
         decode the current latent's middle frame via the swapped
         TAEHV/TAEW preview VAE and publish a base64 PNG to
         ``VIDEO_PROGRESS.set_thumbnail`` for the modal to render.
    """
    # Floor MUST stay strictly above 1.0 so the pipeline's
    # ``do_classifier_free_guidance`` property (``_guidance_scale > 1.0``)
    # does not flip to False mid-loop. If it did, the pipeline would
    # switch from 2-batch (cond+uncond) to 1-batch on the last step
    # while the embeddings + scheduler state are still 2-batch shape,
    # crashing with shape mismatches or producing garbled frames
    # ("Image must have 1, 2, 3 or 4 channels" on Wan, batch
    # dimension errors on LTX).
    decay_floor = 1.5
    decay_active = cfg_decay and total_steps > 1 and initial_guidance > decay_floor
    thumb_active = bool(preview_vae)
    # Stride keeps the polled endpoint payload small. Video
    # latent decode is more expensive than image (5D tensor), so
    # we cap thumbnails at ~6 per gen.
    thumb_stride = max(1, total_steps // 6) if thumb_active else 1

    def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
        VIDEO_PROGRESS.set_step(step + 1, total=max(1, total_steps))
        if VIDEO_PROGRESS.is_cancelled():
            try:
                _pipeline._interrupt = True
            except Exception:
                pass
            raise GenerationCancelled("Video generation cancelled by user")
        if decay_active:
            # Step `step` just finished (step uses scale set BEFORE it).
            # Set the scale for step `step+1`. Linear ramp from initial
            # at step 0 to decay_floor at step total_steps-1.
            next_step = step + 1
            progress = min(1.0, next_step / max(1, total_steps - 1))
            next_scale = initial_guidance * (1.0 - progress) + decay_floor * progress
            try:
                _pipeline.guidance_scale = float(next_scale)
            except Exception:
                pass
        if thumb_active:
            is_final = (step + 1) >= total_steps
            if is_final or (step % thumb_stride == 0):
                latents = callback_kwargs.get("latents") if callback_kwargs else None
                try:
                    from backend_service.helpers.preview_thumbnails import (
                        decode_video_latent_to_b64,
                    )
                    b64 = decode_video_latent_to_b64(_pipeline, latents)
                    if b64 is not None:
                        VIDEO_PROGRESS.set_thumbnail(b64)
                except Exception:
                    # Best-effort — never fail the gen on a preview
                    # decode error.
                    pass
        return callback_kwargs

    return _on_step_end


def invoke_pipeline(pipeline: Any, kwargs: dict[str, Any]) -> list[Any]:
    """Run the diffusers pipeline and return the first batch's frames.

    Carved out as a seam so tests can stub it without loading real
    weights. Diffusers video pipelines return an output with a
    ``.frames`` attribute shaped like ``list[list[PIL.Image]]`` — one
    inner list per batch item. We only ever render batchSize=1, so
    we return ``result.frames[0]``.

    Wires the diffusers per-step callback into ``VIDEO_PROGRESS`` so the
    UI bar tracks denoising in real time. Falls back to a callback-free
    invocation on older diffusers versions that don't expose the kwarg.
    """
    total_steps = int(kwargs.get("num_inference_steps") or 0)
    initial_guidance = float(kwargs.get("guidance_scale") or 1.0)
    # Phase E2: CFG decay flag is plumbed via a private kwarg the
    # caller pops before passing to the pipeline. Default-on when
    # absent so existing call sites pick up the schedule.
    cfg_decay = bool(kwargs.pop("__cfg_decay", True))
    # FU-018 part 2: previewVae flag plumbs through the same
    # private-kwarg pattern. When on, ``_make_step_callback`` emits
    # a per-step base64 thumbnail decoded via the TAESD/TAEHV swap.
    preview_vae = bool(kwargs.pop("__preview_vae", False))
    callback = make_step_callback(
        total_steps, initial_guidance, cfg_decay, preview_vae=preview_vae,
    )
    kwargs.setdefault("callback_on_step_end", callback)

    try:
        result = pipeline(**kwargs)
    except TypeError as exc:
        message = str(exc)
        # Older diffusers / pipelines that don't accept ``callback_on_step_end``.
        if "callback_on_step_end" in message:
            kwargs = {k: v for k, v in kwargs.items() if k != "callback_on_step_end"}
            try:
                result = pipeline(**kwargs)
            except TypeError as inner:
                if "negative_prompt" in str(inner) and "negative_prompt" in kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k != "negative_prompt"}
                    result = pipeline(**kwargs)
                else:
                    raise
        elif "negative_prompt" in message and "negative_prompt" in kwargs:
            # Some pipelines reject ``negative_prompt`` even when given a
            # non-empty value. Fall back once without it rather than crashing
            # the whole generation.
            kwargs = {key: value for key, value in kwargs.items() if key != "negative_prompt"}
            result = pipeline(**kwargs)
        else:
            raise

    frames = getattr(result, "frames", None)
    if frames is None:
        raise RuntimeError(
            "Video pipeline result is missing a `.frames` attribute. "
            "This usually means the installed diffusers version returns a "
            "different output shape. Upgrade diffusers: pip install -U diffusers"
        )
    if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
        return list(frames[0])
    return list(frames)


def encode_frames_to_mp4(frames: list[Any], fps: int) -> bytes:
    """Encode a list of PIL.Image frames to an mp4 byte buffer.

    Carved out as a seam so tests can stub it. We use ``imageio`` +
    ``imageio-ffmpeg`` via the ``diffusers.utils.export_to_video`` helper
    when available (it handles the numpy conversion), and fall back to a
    direct ``imageio`` writer if diffusers hasn't exposed the helper on
    the installed version.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        tmp_path = handle.name
    try:
        export_to_video = None
        try:
            from diffusers.utils import export_to_video as _export  # type: ignore
            export_to_video = _export
        except Exception:
            export_to_video = None

        if export_to_video is not None:
            export_to_video(frames, tmp_path, fps=fps)
        else:
            # Minimal fallback — avoids tying us to diffusers' helper
            # layout. Uses the same pyav backend imageio-ffmpeg ships.
            import numpy as np  # type: ignore
            import imageio  # type: ignore

            writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
            try:
                for frame in frames:
                    array = np.asarray(frame)
                    if array.ndim == 2:
                        array = np.stack([array] * 3, axis=-1)
                    writer.append_data(array.astype("uint8"))
            finally:
                writer.close()

        return Path(tmp_path).read_bytes()
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def pipeline_class_for_repo(repo: str) -> Any:
    entry = PIPELINE_REGISTRY.get(repo)
    if entry is None:
        raise RuntimeError(
            f"No diffusers pipeline is registered for repo '{repo}'. "
            f"Supported repos: {sorted(PIPELINE_REGISTRY.keys())}"
        )
    class_name = entry["class_name"]
    diffusers = importlib.import_module("diffusers")
    pipeline_cls = getattr(diffusers, class_name, None)
    if pipeline_cls is None:
        raise RuntimeError(
            f"The installed diffusers version does not expose '{class_name}'. "
            "Upgrade diffusers: pip install -U diffusers"
        )
    return pipeline_cls
