"""Native Apple Silicon FLUX runtime via the mflux package.

Only loaded for variants that set ``runtime="mflux"`` in the catalog.
Compared to diffusers+MPS:

- 2-3x faster on M-series Macs (native MLX kernels vs the PyTorch
  MPS backend)
- No fp16 black-image hazard — MLX handles precision cleanly
- Limited to FLUX (schnell, dev) — not a diffusers replacement

The engine is a quiet no-op on non-Apple platforms: ``probe()``
reports unavailability, and the manager routes to diffusers
automatically.

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
import io
import platform
import time
from typing import Any

from backend_service.image_runtime.placeholder_engine import _resolve_base_seed
from backend_service.image_runtime.types import (
    GeneratedImage,
    ImageGenerationConfig,
)


def _mflux_name_for_repo(repo: str) -> str | None:
    lowered = repo.lower()
    if "flux.1-schnell" in lowered or "flux-schnell" in lowered:
        return "schnell"
    if "flux.1-dev" in lowered or "flux-dev" in lowered:
        return "dev"
    return None


class MfluxImageEngine:
    runtime_label = "mflux (MLX native)"

    def __init__(self) -> None:
        self._flux: Any = None
        self._loaded_name: str | None = None

    def probe(self) -> dict[str, Any]:
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            return {
                "available": False,
                "reason": "mflux runs on Apple Silicon only.",
            }
        if importlib.util.find_spec("mflux") is None:
            return {
                "available": False,
                "reason": (
                    "mflux not installed — add it from the Setup page to "
                    "enable the native Apple Silicon FLUX runtime."
                ),
            }
        return {"available": True, "reason": None}

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        probe = self.probe()
        if not probe["available"]:
            raise RuntimeError(probe["reason"] or "mflux unavailable")

        # Map our repo ids to the names mflux expects. Anything else
        # falls back to the diffusers path.
        flux_name = _mflux_name_for_repo(config.repo)
        if flux_name is None:
            raise RuntimeError(
                f"mflux doesn't support {config.repo} — only FLUX.1-schnell "
                "and FLUX.1-dev are available via the native MLX runtime."
            )

        import mflux  # type: ignore
        started = time.perf_counter()
        if self._flux is None or self._loaded_name != flux_name:
            self._flux = mflux.Flux1.from_name(flux_name)
            self._loaded_name = flux_name
        seed = _resolve_base_seed(config.seed)
        result_image = self._flux.generate_image(
            seed=seed,
            prompt=config.prompt,
            config=mflux.Config(
                num_inference_steps=config.steps,
                height=config.height,
                width=config.width,
                guidance=config.guidance,
            ),
        )
        elapsed = max(0.1, time.perf_counter() - started)

        pil_image = getattr(result_image, "image", result_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        return [
            GeneratedImage(
                seed=seed,
                bytes=buffer.getvalue(),
                extension="png",
                mimeType="image/png",
                durationSeconds=round(elapsed, 1),
                runtimeLabel=self.runtime_label,
                runtimeNote=f"MLX native FLUX ({flux_name})",
            )
        ]
