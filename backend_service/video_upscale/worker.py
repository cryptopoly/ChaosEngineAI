"""Warm video upscale worker: holds one Real-ESRGAN model and upscales one segment per ``generate`` request.

Started by ``video_upscale.jobs`` as ``python -m backend_service.video_upscale.worker`` from the backend's own Python
(torch comes from the GPU runtime there), with the add-on's ``site/`` on ``PYTHONPATH`` for spandrel. The protocol
loop is ``warm_worker_base.serve``: a ``generate`` request is ``pipeline.segment_request`` (input, frame range, target
size, weights, tile, batch, encoder arguments, output file); it sends ``step`` events with the frames done and answers
``{"frames": n, "vram_peak_gb": ...}``. The model stays loaded between segments and jobs until ``unload``, the idle
stop, or a heavy job of another kind stops the worker. ``CHAOSENGINE_WARM_ENGINE_MODULE`` swaps in a fake engine.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

from backend_service import warm_worker_base as BASE


class UpscaleEngine:
    def __init__(self, loader: Callable[..., Any] | None = None) -> None:
        self._loader = loader
        self.upscaler: Any = None
        self.key: tuple[str, int] | None = None

    def loaded(self) -> dict[str, Any]:
        return {"weights": self.key[0], "tile": self.key[1]} if self.key else {}

    def unload(self) -> None:
        if self.upscaler is not None and hasattr(self.upscaler, "unload"):
            self.upscaler.unload()
        self.upscaler, self.key = None, None
        gc.collect()

    def _load(self, weights: str, tile: int) -> Any:
        if self._loader is not None:
            return self._loader(weights, tile)
        from backend_service.video_upscale.engine import SpandrelUpscaler  # noqa: PLC0415

        return SpandrelUpscaler(Path(weights), tile=tile)

    def _ensure(self, weights: str, tile: int) -> bool:
        if self.upscaler is not None and self.key == (weights, tile):
            return True
        self.unload()
        self.upscaler, self.key = self._load(weights, tile), (weights, tile)
        return False

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        from backend_service.video_upscale import pipeline  # noqa: PLC0415

        hooks.stage("load")
        warm = self._ensure(str(request["weights"]), int(request.get("tile") or 0))
        hooks.check_cancel()
        hooks.stage("upscale")
        total = 0 if request.get("last") else int(request.get("frames") or 0)
        result = pipeline.process_segment(request, self.upscaler, lambda made: hooks.step(made, max(total, made)),
                                          hooks.check_cancel)
        peak = getattr(self.upscaler, "vram_peak_gb", None)
        return {**result, "warm": warm, "vram_peak_gb": peak() if callable(peak) else None}


def main() -> None:
    BASE.strip_own_dir(__file__)
    from importlib.metadata import PackageNotFoundError, version as package_version  # noqa: PLC0415

    try:
        version = f"spandrel {package_version('spandrel')}"  # read without importing it
    except PackageNotFoundError:  # the first generate reports the real import error
        version = "spandrel ?"
    BASE.serve(UpscaleEngine, engine_version=version)


if __name__ == "__main__":
    main()
