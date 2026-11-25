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

from backend_service.catalog import licences

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
        "licence": licences.BSD_3_CLAUSE,
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
        "licence": licences.BSD_3_CLAUSE,
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

# ---------------------------------------------------------------------------
# quality tier (D0145, B1334): FlashVSR+ Turbo, opt-in, NVIDIA on Windows and Linux only
# ---------------------------------------------------------------------------
#
# Measured on the RTX 4090 under Windows (docs/research/ai-video-upscaler-spike-2026-11.md, section 4): 5.7 frames a
# second from 576p to 1280x1024 at 8.8 GB, 3.8 from 540p to 1920x1024 at 11.2 GB, 2.1 from 720p to 2560x1408 at
# 17.2 GB. The gates below are those peaks plus the CUDA context (section 6). 4K spilled into shared memory and never
# finished, so it is not offered; a card under 12 GB is not offered at all (the spike's floor). Kept out of
# ``VIDEO_UPSCALE_MODELS`` on purpose: that list is the fast add-on's (its installer downloads every row's ``url``).
# The top tier is 2560x1440, not the spike's 2560x1408, so the Upscale tab's 1440p choice for a 16:9 film is offered:
# measured through the app's worker on 2026-11-25 (2560x1440 padded to 2560x1536, colour correction on): 18.2 GB on
# nvidia-smi above the idle desktop (torch reserved 17.3 GB), hence 20 GB.

_FLASHVSR_REPO = "ruchangyao/FlashVSR_plus"
_FLASHVSR_REVISION = "46b79520946ac67dec94d1566490690953c7debc"
_HF = "https://huggingface.co"

QUALITY_MIN_CARD_GB = 12.0
# (largest width, largest height, GPU memory the card needs in GB), smallest first. A size fits a tier when its pixel
# count is at most the tier's and neither side is longer than the tier's longer side (so a 1440x2560 portrait fits
# the 2560x1440 tier, a 4096x400 strip does not).
QUALITY_SIZE_TIERS: tuple[tuple[int, int, float], ...] = ((1280, 1024, 10.0), (1920, 1088, 12.0), (2560, 1440, 20.0))
# Card sizes are read in GiB and rounded on the box; a "12 GB" card must not miss a 12 GB gate by a sliver.
_CARD_TOLERANCE_GB = 0.6

VIDEO_UPSCALE_QUALITY_MODELS: list[dict[str, Any]] = [
    {
        "id": "flashvsr-plus-turbo",
        "name": "FlashVSR+ Turbo",
        "label": "Quality",
        "tier": "quality",
        "engine": "flashvsr-plus",
        "author": "OpenImagingLab",
        "architecture": "Wan2.1 1.3B one-step streaming video super-resolution (hybrid sparse attention)",
        "baseModel": "Wan-AI/Wan2.1-T2V-1.3B",
        "scale": 4,
        "default": False,
        "licence": licences.APACHE_2_0,
        "licenceNote": ("Apache-2.0: the code (OpenImagingLab/FlashVSR_Plus), the weights (ruchangyao/FlashVSR_plus "
                        "model card) and the Wan2.1 base model (Wan-AI) all carry Apache-2.0."),
        "source": "huggingface",
        "repo": _FLASHVSR_REPO,
        "revision": _FLASHVSR_REVISION,
        # Turbo only (spike section 6: Base is barely sharper, slower and up to 6 GB hungrier). Sizes and SHA-256 from
        # the Hub's LFS pointers at the pinned revision, re-hashed from the spike's download on 2026-11-25.
        "files": [
            {"path": "turbo/dit.safetensors", "sizeBytes": 5_676_070_392,
             "sha256": "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b"},
            {"path": "turbo/lq_proj_in.safetensors", "sizeBytes": 1_151_384_272,
             "sha256": "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496"},
            {"path": "TCDecoder.ckpt", "sizeBytes": 189_018_333,
             "sha256": "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"},
        ],
        "sizeBytes": 5_676_070_392 + 1_151_384_272 + 189_018_333,
        "sizeGb": 7.0,
        "styleTags": ["cuda"],  # helpers.fit.platform_gate: NVIDIA only
        "platforms": ["windows", "linux"],
        "minCardGb": QUALITY_MIN_CARD_GB,
        "sizeTiers": [{"width": w, "height": h, "gpuGb": gb} for w, h, gb in QUALITY_SIZE_TIERS],
        "maxOutputWidth": QUALITY_SIZE_TIERS[-1][0],
        "maxOutputHeight": QUALITY_SIZE_TIERS[-1][1],
        "runtimeFootprintCudaGb": QUALITY_SIZE_TIERS[0][2],
        "footprintBasis": "measured",
        "footprintMeasuredOn": "cuda",
        # Through the app's worker with colour correction, whole jobs on the 4090 (2026-11-25): 5.0 frames a second
        # from 576p to 1280x1024, 3.0 from 540p to 1920x1024, 1.4 to 2560x1440 (model alone).
        "speedNote": ("Measured on an RTX 4090: about 5 frames a second from 576p to 1280x1024, 3 to 1080p, 1.4 to "
                      "1440p (the fast tier: 11 to 27)."),
        "notes": ("Invents believable texture (hair, fabric, water): best for AI footage. It can change faces and "
                  "mangles changing text such as a timecode; the fast tier stays the faithful choice for family tapes."),
    },
]

QUALITY_MODEL_ID = "flashvsr-plus-turbo"


def hf_file_url(row: dict[str, Any], path: str) -> str:
    """The Hub download URL of one of ``row``'s files at its pinned revision."""
    return f"{_HF}/{row['repo']}/resolve/{row['revision']}/{path}"


def find_quality_model(model_id: str) -> dict[str, Any] | None:
    return next((row for row in VIDEO_UPSCALE_QUALITY_MODELS if row["id"] == model_id), None)


def is_quality_model(model_id: str) -> bool:
    return find_quality_model(model_id) is not None


def quality_tier_need_gb(width: int, height: int) -> float | None:
    """The GPU memory an output of ``width`` x ``height`` needs, or None when it is above every tier (4K)."""
    pixels, side = int(width) * int(height), max(int(width), int(height))
    for tier_w, tier_h, gpu_gb in QUALITY_SIZE_TIERS:
        if pixels <= tier_w * tier_h and side <= max(tier_w, tier_h):
            return gpu_gb
    return None


def quality_card_gb(hardware: Any) -> float | None:
    """The NVIDIA card's own memory in GB, or None on a Mac, a CPU-only box or a GPU that shares system memory."""
    if getattr(hardware, "kind", None) != "cuda" or getattr(hardware, "unified", False):
        return None
    # ``HardwareClass``: on a discrete NVIDIA card ``usable_gb`` is its VRAM (``total_gb`` is the system RAM).
    return float(getattr(hardware, "usable_gb", 0.0) or 0.0)


def quality_fit(hardware: Any, size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Whether the quality tier is offered here, and for which outputs.

    ``{"offered": bool, "code": str | None, "reason": str, "cardGb": float | None, "maxOutput": [w, h] | None}``;
    with ``size``, also ``"needGb"`` and ``"sizeOk"``. ``code`` is the translation key of a refusal
    (``notCuda``, ``smallCard``, ``tooBig``, ``sizeNeedsMore``)."""
    card = quality_card_gb(hardware)
    out: dict[str, Any] = {"offered": False, "code": None, "reason": "", "cardGb": card, "maxOutput": None}
    if card is None:
        out.update(code="notCuda", reason="The quality upscaler needs an NVIDIA graphics card with its own memory "
                                          "(Windows or Linux); this computer has none.")
        return out
    usable = card + _CARD_TOLERANCE_GB
    if usable < QUALITY_MIN_CARD_GB:
        out.update(code="smallCard", reason=f"The quality upscaler needs a graphics card with at least "
                                            f"{QUALITY_MIN_CARD_GB:.0f} GB; this one has {card:.0f} GB.")
        return out
    best = [(w, h) for w, h, gb in QUALITY_SIZE_TIERS if gb <= usable]
    out.update(offered=True, maxOutput=list(best[-1]) if best else None,
               reason=f"Up to {best[-1][0]}x{best[-1][1]} on this {card:.0f} GB card." if best else "")
    if size is not None:
        need = quality_tier_need_gb(*size)
        out["needGb"] = need
        out["sizeOk"] = need is not None and need <= usable
        if need is None:
            out.update(code="tooBig", reason=f"The quality upscaler makes at most {QUALITY_SIZE_TIERS[-1][0]}x"
                                             f"{QUALITY_SIZE_TIERS[-1][1]} (4K does not fit 24 GB); {size[0]}x"
                                             f"{size[1]} is bigger. Choose a smaller size or the fast upscaler.")
        elif need > usable:
            out.update(code="sizeNeedsMore", reason=f"{size[0]}x{size[1]} needs a {need:.0f} GB graphics card for "
                                                    f"the quality upscaler; this one has {card:.0f} GB.")
    return out


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


def quality_catalog_with_fit(hardware: Any = None) -> list[dict[str, Any]]:
    """Copies of the quality rows with ``fit`` (``variant_fit``, for the badge) and ``qualityFit`` (the size gates)."""
    from backend_service.helpers.fit import variant_fit  # noqa: PLC0415
    from backend_service.helpers.hardware_class import host_hardware_class  # noqa: PLC0415

    hw = hardware or host_hardware_class()
    rows = []
    for row in VIDEO_UPSCALE_QUALITY_MODELS:
        gate = quality_fit(hw)
        fit = variant_fit(row, "video", hw)
        if gate["code"] == "smallCard":  # the spike's 12 GB floor is stricter than the 10 GB peak the badge reads
            fit = {**fit, "status": "wont-fit", "reason": gate["reason"]}
        rows.append({**row, "fit": fit, "qualityFit": gate})
    return rows
