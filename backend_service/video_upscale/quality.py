"""The quality tier's side of a job (FlashVSR+ Turbo): segment length, warm-up, the 128-pixel grid, and the worker.

The fast tier's machinery runs the job unchanged: ``pipeline.run_job`` plans the segments, keeps the manifest, checks
every segment's frame count against the source (B1328) and muxes the original sound itself (FlashVSR+'s own CLI muxes
with ``-shortest`` and lost frames in the spike). Only the runner differs: ``runner()`` turns each
``pipeline.segment_request`` into a worker request (``worker_request``) and sends it to ``quality_worker.py``, which runs
in the add-on's venv with the model loaded.

What the worker request adds:

* **Warm-up.** FlashVSR+ is a streaming model with a KV cache; a fresh run starts cold. Every segment but the first
  decodes ``WARMUP_FRAMES`` frames before its first frame; the model sees them, the worker drops their output, so the
  segment still yields exactly its own frames.
* **The 128 grid.** The model needs both sides to be multiples of 128 and its CLI centre-crops to get there. Here the
  frames are scaled (bicubic) to the exact target, padded on the right and bottom by reflection up to the grid
  (``padded_size``), and the output cropped back, so nothing is cut off and the encoder gets the target size (no
  lanczos pass is needed).
* **Segment length.** The whole segment is held as float frames at the padded size (spike: 200 frames at 1280x1024),
  so segments are capped by pixels (``SEGMENT_PIXEL_BUDGET``) as well as by ``pipeline.SEGMENT_SECONDS``.
* **Colour correction on** (spike section 4: it removes the +3 RGB shift and the brightness pumping).
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from backend_service.video_upscale import pipeline, quality_addon

MULTIPLE = 128
WARMUP_FRAMES = 8
# Frames x padded pixels held per segment, warm-up included: the spike's 200 frames at 1280x1024.
SEGMENT_PIXEL_BUDGET = 200 * 1280 * 1024
MIN_SEGMENT_FRAMES = 32
MIN_SIDE = MULTIPLE  # reflection padding needs the frame at least as big as the pad
_SEGMENT_TIMEOUT_S = 3 * 3600.0
_WORKER_STALL_S = 900.0  # the first load (7 GB) plus the first Triton compile, and a whole segment, send no step
_WORKER_HARD_CANCEL_S = 30.0
_WORKER_LOG = Path.home() / ".chaosengine" / "logs" / "video-upscale-quality-worker.log"
_WORKER_SCRIPT = Path(__file__).resolve().with_name("quality_worker.py")


def padded_size(width: int, height: int) -> tuple[int, int]:
    """The model's frame size: each side rounded up to a multiple of ``MULTIPLE``."""
    return -(-int(width) // MULTIPLE) * MULTIPLE, -(-int(height) // MULTIPLE) * MULTIPLE


def segment_frames(width: int, height: int, fps: str, seconds: float = pipeline.SEGMENT_SECONDS) -> int:
    """Frames per segment: ``seconds`` of video, fewer when the padded frames would pass the pixel budget."""
    pw, ph = padded_size(width, height)
    by_pixels = SEGMENT_PIXEL_BUDGET // (pw * ph) - WARMUP_FRAMES
    by_time = max(1, round(seconds * pipeline.fps_value(fps)))
    return max(1, min(by_time, max(MIN_SEGMENT_FRAMES, by_pixels)))


def segment_seconds(width: int, height: int, fps: str) -> float:
    """``segment_frames`` as the ``segment_seconds`` that ``pipeline.make_spec`` takes."""
    return segment_frames(width, height, fps) / pipeline.fps_value(fps)


def worker_request(req: dict[str, Any], paths: dict[str, str] | None = None) -> dict[str, Any]:
    """The worker's ``generate`` request for one ``pipeline.segment_request``: the decode command (warm-up
    included), the encode command (the exact target, so no scaling), the padded size and the model files."""
    width, height = int(req["width"]), int(req["height"])
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"The quality tier needs at least {MIN_SIDE} pixels on each side of the result.")
    first = int(req["first_frame"])
    warm = min(WARMUP_FRAMES, first)
    decode = pipeline.decode_command({**req, "first_frame": first - warm, "frames": int(req["frames"]) + warm})
    encode = pipeline.encode_command(req, (width, height), Path(req["output"]))
    pw, ph = padded_size(width, height)
    return {"decode": decode, "encode": encode, "warmup": warm, "frames": int(req["frames"]), "last": bool(req["last"]),
            "src_width": int(req["src_width"]), "src_height": int(req["src_height"]), "width": width,
            "height": height, "pad_width": pw, "pad_height": ph, "fps": str(req["fps"]), "output": str(req["output"]),
            "color_correction": True, "paths": paths or quality_addon.model_paths(),
            "code": str(quality_addon.code_dir())}


# ---------------------------------------------------------------------------
# the warm worker (runs in the add-on's venv)
# ---------------------------------------------------------------------------

_WORKER: Any = None
_WORKER_LOCK = threading.Lock()


def _worker_env() -> dict[str, str]:
    from backend_service.inference._utils import _isolated_child_env  # noqa: PLC0415

    from backend_service.helpers.malloc_release import MIMALLOC_ENV  # noqa: PLC0415

    env = _isolated_child_env({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    # The backend's "give freed memory back at once" mimalloc switches made the pipeline 20 % slower here (it frees and
    # re-allocates gigabytes of frames per chunk: 2.7 against 3.3 frames a second at 1920x1024 on the RTX 4090). The
    # worker is its own process, stopped when it idles or another heavy job starts, so its memory goes back then.
    for name in MIMALLOC_ENV:
        if os.environ.get(name) == MIMALLOC_ENV[name]:
            env.pop(name, None)
    return env


def worker_command() -> list[str]:
    return [str(quality_addon.venv_python()), "-I", str(_WORKER_SCRIPT)]


def worker() -> Any:
    global _WORKER
    from backend_service.warm_worker_client import WarmWorker  # noqa: PLC0415

    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = WarmWorker(name="video-upscale-quality", kind="upscale", command=worker_command,
                                 env=_worker_env, log_path=_WORKER_LOG, hard_cancel_after_s=_WORKER_HARD_CANCEL_S)
        return _WORKER


def release_worker(reason: str = "unloaded") -> bool:
    """Stop the worker if it idles (it holds about 9 GB of GPU memory between jobs)."""
    return _WORKER.stop_if_idle(reason) if _WORKER is not None else False


def runner(send: Callable[..., dict[str, Any]] | None = None) -> Callable[..., dict[str, Any]]:
    """A ``pipeline.run_job`` runner for the quality tier. ``send`` replaces the warm worker's ``request`` (tests)."""
    from backend_service.warm_worker_client import WarmWorkerCancelled, WarmWorkerError  # noqa: PLC0415

    def run(req: dict[str, Any], on_frames: Callable[[int], None], is_cancelled: Callable[[], bool]) -> dict[str, Any]:
        def on_event(event: dict[str, Any]) -> None:
            if event.get("event") == "step":
                on_frames(int(event.get("step") or 0))

        call = send or worker().request
        try:
            return call("generate", timeout=_SEGMENT_TIMEOUT_S, stall_s=_WORKER_STALL_S, on_event=on_event,
                        cancel_check=is_cancelled, job=f"uq-{uuid.uuid4().hex[:12]}", **worker_request(req))
        except WarmWorkerCancelled as exc:
            raise pipeline.Cancelled("cancelled") from exc
        except WarmWorkerError as exc:
            raise RuntimeError(f"The quality upscaler failed: {exc}") from exc

    return run
