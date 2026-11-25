"""The Storyboard export's fast upscale: one shot intermediate through the AI upscaler to the film's final size.

``storyboard.assemble`` calls ``shot_upscaler()`` when a film exports with ``upscaleMethod: "fast"``. Without the
add-on it raises ``UpscaleError`` with the install step, which the render reports. Each shot is a synchronous job on
the shared heavy-job slot (``JOBS.run_sync``, source ``storyboard``), encoded with the Storyboard's time base so the
assembler can join it with its title and end cards; the shot's own sound is copied back unchanged.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from backend_service.catalog.video_upscale_models import DEFAULT_MODEL_ID
from backend_service.video_upscale import jobs


def shot_upscaler(model_id: str = DEFAULT_MODEL_ID) -> Callable[..., None]:
    jobs.check_ready(model_id)  # UpscaleError (409) naming the install step when the add-on is missing

    def run(src: Path, dst: Path, size: tuple[int, int], cancel_event: threading.Event | None,
            report: Callable[[float], None]) -> None:
        spec = jobs.build_spec(src, dst, model_id, size, avtb=True)
        jobs.JOBS.run_sync(spec, source="storyboard", label=f"Storyboard fast upscale to {size[0]}x{size[1]}",
                           is_cancelled=cancel_event.is_set if cancel_event is not None else (lambda: False),
                           progress=lambda done, total, _i, _n: report(done / total if total else 0.0))

    return run
