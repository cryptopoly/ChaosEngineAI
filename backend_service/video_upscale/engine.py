"""The upscaler contract: a batch of RGB frames in, the same frames ``scale`` times larger out.

An upscaler is any callable ``frames -> frames`` on ``uint8`` arrays shaped ``(n, height, width, 3)`` with a
``scale`` attribute (an int). Tests use a fake that repeats each pixel; the real one is ``SpandrelUpscaler``, a
Real-ESRGAN checkpoint loaded through spandrel (chaiNNer-org, MIT) into PyTorch on CUDA, MPS or the CPU. torch and
spandrel are imported only inside it, so only the worker process ever loads them.

Big frames run in tiles (``upscale_tiled``): each tile carries ``overlap`` pixels of its neighbours as context and
only its own middle part is kept, so no seam shows where tiles meet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

# The largest output side the app makes (H.264 level 5.1 and most players stop near 4096; 3840x2160 fits).
MAX_SIDE = 4096


class UpscaleError(RuntimeError):
    """A request the upscaler cannot run; ``status`` is the HTTP code a route answers with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Upscaler(Protocol):
    scale: int

    def __call__(self, frames: np.ndarray) -> np.ndarray: ...


def even(value: float) -> int:
    """The nearest even whole number, at least 2 (yuv420p needs even sides)."""
    return max(2, int(round(value / 2.0)) * 2)


def target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return even(width * scale), even(height * scale)


def check_target(width: int, height: int) -> None:
    """Refuse an output over ``MAX_SIDE`` on a side, with the size it would have been."""
    if max(width, height) > MAX_SIDE:
        raise UpscaleError(
            f"The result would be {width}x{height}; the limit is {MAX_SIDE} pixels on a side. "
            "Choose a smaller scale or a smaller target size."
        )


def _spans(length: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """``(start, end, keep_from, keep_to)`` per tile along one axis. Neighbouring tiles overlap by at least
    ``overlap``; each keeps the part up to the middle of its overlaps, so the kept parts cover the axis once."""
    if tile <= 0 or length <= tile:
        return [(0, length, 0, length)]
    stride = max(1, tile - overlap)
    starts = sorted(set(list(range(0, length - tile, stride)) + [length - tile]))
    spans = []
    for i, start in enumerate(starts):
        keep_from = 0 if i == 0 else (start + starts[i - 1] + tile) // 2
        keep_to = length if i == len(starts) - 1 else (starts[i + 1] + start + tile) // 2
        spans.append((start, start + tile, keep_from, keep_to))
    return spans


def upscale_tiled(run: Callable[[np.ndarray], np.ndarray], frames: np.ndarray, scale: int, tile: int,
                  overlap: int = 16) -> np.ndarray:
    """``run`` over ``frames`` in tiles of ``tile`` pixels (0: the whole frame at once)."""
    n, height, width, channels = frames.shape
    rows, cols = _spans(height, tile, overlap), _spans(width, tile, overlap)
    if len(rows) == 1 and len(cols) == 1:
        out = run(frames)
        _check_shape(out, (n, height * scale, width * scale, channels))
        return out
    out = np.empty((n, height * scale, width * scale, channels), dtype=np.uint8)
    for y0, y1, keep_y0, keep_y1 in rows:
        for x0, x1, keep_x0, keep_x1 in cols:
            part = run(np.ascontiguousarray(frames[:, y0:y1, x0:x1]))
            _check_shape(part, (n, (y1 - y0) * scale, (x1 - x0) * scale, channels))
            out[:, keep_y0 * scale:keep_y1 * scale, keep_x0 * scale:keep_x1 * scale] = part[
                :, (keep_y0 - y0) * scale:(keep_y1 - y0) * scale, (keep_x0 - x0) * scale:(keep_x1 - x0) * scale]
    return out


def _check_shape(out: Any, want: tuple[int, ...]) -> None:
    if tuple(getattr(out, "shape", ())) != want:
        raise UpscaleError(f"The model returned frames shaped {getattr(out, 'shape', None)}, expected {want}.", 500)


def pick_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SpandrelUpscaler:
    """A Real-ESRGAN checkpoint through spandrel. Needs the video upscale add-on on ``sys.path`` (the worker adds it).

    Not run in this repository's tests (no weights are downloaded there); the 4090 test plan covers it."""

    def __init__(self, weights: Path, *, device: str | None = None, half: bool | None = None, tile: int = 0,
                 overlap: int = 16) -> None:
        import torch  # noqa: PLC0415
        from spandrel import ImageModelDescriptor, ModelLoader  # noqa: PLC0415

        descriptor = ModelLoader(device="cpu").load_from_file(str(weights))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise UpscaleError(f"{Path(weights).name} is not an image upscale model.", 500)
        self.device = device or pick_device(torch)
        self.half = (self.device == "cuda" and bool(descriptor.supports_half)) if half is None else half
        descriptor.to(self.device)
        descriptor.eval()
        if self.half:
            descriptor.half()
        self.model = descriptor
        self.scale = int(descriptor.scale)
        self.tile, self.overlap = int(tile), int(overlap)

    def _run(self, frames: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2)
            x = (x.half() if self.half else x.float()).div_(255.0)
            y = self.model(x).float().clamp_(0.0, 1.0).mul_(255.0).round_()
            return y.permute(0, 2, 3, 1).to("cpu", torch.uint8).contiguous().numpy()

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return upscale_tiled(self._run, frames, self.scale, self.tile, self.overlap)

    def vram_peak_gb(self) -> float | None:
        try:
            import torch  # noqa: PLC0415

            return round(torch.cuda.max_memory_reserved() / 2**30, 2) if self.device == "cuda" else None
        except Exception:  # noqa: BLE001
            return None

    def unload(self) -> None:
        self.model = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
