"""Video upscale models, fast tier (docs/research/ai-video-upscaler-feasibility-2026-11.md): Real-ESRGAN by xinntao.

The weights are single ``.pth`` files from the GitHub releases of ``xinntao/Real-ESRGAN`` (there is no official Hugging
Face repo). The repo is BSD-3-Clause and the weights carry no licence of their own, so they ship under the repo's
licence (flagged as a risk in the feasibility report). Asset names and byte sizes were read from the GitHub release
API on 2026-11-25 without downloading the files; GitHub lists no digest for these 2021-2022 assets, so ``sha256`` is
None until someone downloads each file once and pins it (``sha256Todo``). ``scripts/audit-catalog-repos.py`` fails a
release while any row is unpinned; the add-on installer checks the byte size meanwhile and records the hash it saw.

Memory figures are ESTIMATES, not measurements (``footprintBasis: "estimated"``): GPU memory for the model plus its
activations at the row's tile size and batch, from the parameter counts and feature-map sizes; the 4090 test plan
(docs/research/video-upscale-fast-tier-test-plan.md) replaces them. ``helpers.fit.variant_fit`` reads them as a video
row (``runtimeFootprintCudaGb`` on NVIDIA, ``runtimeFootprintMpsGb`` on a Mac, ``runtimeFootprintGb`` elsewhere).
"""

from __future__ import annotations

from typing import Any

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_LICENCE_NOTE = ("BSD-3-Clause (xinntao/Real-ESRGAN). The weights have no separate licence statement; they are "
                 "distributed in the repo's releases under its BSD-3-Clause licence.")
_SHA_TODO = ("Not pinned yet: GitHub lists no digest for this asset. Download it once, check the size, run "
             "sha256sum and put the hash here (the release audit fails until then).")

VIDEO_UPSCALE_MODELS: list[dict[str, Any]] = [
    {
        "id": "realesr-general-x4v3",
        "name": "Real-ESRGAN general x4v3",
        "label": "Fast",
        "author": "xinntao",
        "architecture": "SRVGGNetCompact (about 1.2 M parameters)",
        "scale": 4,
        "default": True,
        "licence": "BSD-3-Clause",
        "licenceNote": _LICENCE_NOTE,
        "source": "github-release",
        "repo": "xinntao/Real-ESRGAN",
        "releaseTag": "v0.2.5.0",
        "file": "realesr-general-x4v3.pth",
        "url": f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        "sizeBytes": 4_885_111,
        "sizeGb": 0.005,
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        # Whole frames up to 1080p input, four at a time. ESTIMATES (see the module docstring).
        "tile": 0,
        "batch": 4,
        "runtimeFootprintCudaGb": 2.0,
        "runtimeFootprintMpsGb": 3.0,
        "runtimeFootprintGb": 3.0,
        "footprintBasis": "estimated",
        "speedNote": "Estimate: 20 to 40 frames a second from 480p on an RTX 4090 (not measured here).",
        "notes": "Good on blocking and noise; smooths fine texture. No temporal model, so noisy tape can shimmer.",
    },
    {
        "id": "realesrgan-x4plus",
        "name": "Real-ESRGAN x4plus",
        "label": "Fast, stronger clean-up",
        "author": "xinntao",
        "architecture": "RRDBNet (about 16.7 M parameters)",
        "scale": 4,
        "default": False,
        "licence": "BSD-3-Clause",
        "licenceNote": _LICENCE_NOTE,
        "source": "github-release",
        "repo": "xinntao/Real-ESRGAN",
        "releaseTag": "v0.1.0",
        "file": "RealESRGAN_x4plus.pth",
        "url": f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        "sizeBytes": 67_040_989,
        "sizeGb": 0.067,
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        # Tiles of 512 input pixels, one frame at a time: an untiled 720p frame needs far more. ESTIMATES.
        "tile": 512,
        "batch": 1,
        "runtimeFootprintCudaGb": 5.0,
        "runtimeFootprintMpsGb": 7.0,
        "runtimeFootprintGb": 7.0,
        "footprintBasis": "estimated",
        "speedNote": "Estimate: 1 to 3 frames a second from 480p to 720p on an RTX 4090 (not measured here).",
        "notes": "Strongest clean-up of heavy compression; can look waxy and flickers on fine texture.",
    },
]

DEFAULT_MODEL_ID = "realesr-general-x4v3"


def find_model(model_id: str) -> dict[str, Any] | None:
    return next((row for row in VIDEO_UPSCALE_MODELS if row["id"] == model_id), None)


def unpinned_rows() -> list[dict[str, Any]]:
    """Rows whose weights have no SHA-256 pin yet (the release audit reports these)."""
    return [row for row in VIDEO_UPSCALE_MODELS if not row.get("sha256")]


def catalog_with_fit(hardware: Any = None) -> list[dict[str, Any]]:
    """Copies of the rows with ``fit`` from ``helpers.fit.variant_fit`` (a video row: platform ``any``)."""
    from backend_service.helpers.fit import variant_fit  # noqa: PLC0415
    from backend_service.helpers.hardware_class import host_hardware_class  # noqa: PLC0415

    hw = hardware or host_hardware_class()
    return [{**row, "fit": variant_fit(row, "video", hw)} for row in VIDEO_UPSCALE_MODELS]
