"""Recommended Studio settings for one image or video model on this machine.

``recommended_settings(variant, kind)`` builds the ``recommendedSettings`` payload that the
Image and Video Studios show in their "Recommended for this machine" panel, or returns None
when there is nothing to recommend. Three rules hold for every answer:

- Only settings the variant's engine reads are returned (``accepted_settings``). mflux,
  stable-diffusion.cpp and ltx-2-mlx ignore the diffusers cache, CFG decay, the preview VAE
  and the sampler, so those never appear for them; the lists mirror what the Studios show
  (``src/constants/image.ts``, ``VideoStudioTab.tsx``).
- Every setting carries a one-line reason. Its basis is ``measured`` only when the table
  holds a timing for it from a machine with the same hardware bucket and chip as this one
  (``_MEASURED``); everything else is ``estimated``, and the overall basis is ``measured``
  only when every setting is.
- Values come from model cards, upstream documentation and this repo's own runs (FU-015
  First Block Cache, FU-021 CFG decay on flow-match only, FU-024 FP8 on CUDA, FU-122 Wan
  guidance, FU-123 / FU-128 timings, the settings-rec measurements), not from guesses.

The table itself (entries per variant id and hardware, and the measured runs) is data in
``backend_service/catalog/recommended_settings.py``; it is re-exported here.
"""

from __future__ import annotations

import platform
import subprocess
import threading
from typing import Any

from backend_service.catalog.recommended_settings import _ENTRIES, _MEASURED, _Entry, _Measurement  # noqa: F401
from backend_service.helpers.hardware_class import HardwareClass

IMAGE_KEYS = (
    "steps", "guidance", "width", "height", "sampler", "qualityPreset",
    "cacheStrategy", "cacheThreshold", "cfgDecay", "previewVae", "fp8LayerwiseCasting",
)
VIDEO_KEYS = (
    "steps", "guidance", "width", "height", "numFrames", "fps",
    "cacheStrategy", "cacheThreshold", "cfgDecay", "previewVae", "fp8LayerwiseCasting",
)

# Mirrors src/constants/image.ts: the Studio shows the sampler only for UNet pipelines, the
# diffusion cache only for DiTs (TeaCache only for FLUX), CFG decay only for flow-match repos.
_FLOW_MATCHING_TOKENS = ("flux", "stable-diffusion-3", "sd3", "qwen-image", "sana", "hidream")
_UNET_IMAGE_TOKENS = (
    "stable-diffusion-xl", "sdxl", "sd_xl", "stable-diffusion-v1-5", "stable-diffusion-1-5",
    "sd-1-5", "sd_1_5", "stable-diffusion-2", "sd-2-",
)
# Mirrors MLX_VIDEO_SUPPORTED_REPOS in src/features/video/videoStudioConstants.ts.
_MLX_VIDEO_REPOS = frozenset({
    "prince-canuma/LTX-2-distilled", "prince-canuma/LTX-2-dev",
    "prince-canuma/LTX-2.3-distilled", "prince-canuma/LTX-2.3-dev",
})
# Transformers the vendored TeaCache forwards cover (cache_compression/teacache.py).
_TEACACHE_VIDEO_TOKENS = ("ltx-video", "hunyuanvideo", "cogvideox", "mochi")


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _is_flow_matching(repo: str) -> bool:
    return any(token in _lower(repo) for token in _FLOW_MATCHING_TOKENS)


def _is_unet(repo: str) -> bool:
    return any(token in _lower(repo) for token in _UNET_IMAGE_TOKENS)


def _image_engine(variant: dict[str, Any]) -> str:
    engine = _lower(variant.get("engine"))
    if engine in {"mflux", "sdcpp"}:
        return engine
    return "diffusers"


def _video_engine(variant: dict[str, Any]) -> str:
    runtime = _lower(variant.get("runtime"))
    repo = str(variant.get("repo") or "")
    if "ltx-2-mlx" in runtime:
        return "ltx2-mlx"
    if repo in _MLX_VIDEO_REPOS or "mlx-video" in runtime:
        return "mlx-video"
    if repo.startswith("NVlabs/LongLive"):
        return "longlive"
    return "diffusers"


def accepted_settings(variant: dict[str, Any], kind: str, hardware: HardwareClass) -> frozenset[str]:
    """The Studio settings this variant's engine reads on this machine."""
    repo = str(variant.get("repo") or "")
    if kind == "image":
        engine = _image_engine(variant)
        if engine == "mflux":
            keys = {"steps", "width", "height"}
            from backend_service.image_runtime.mflux_engine import mflux_family_for_repo

            family = mflux_family_for_repo(repo)
            if family is None or family.takes_guidance:
                keys.add("guidance")
            return frozenset(keys)
        if engine == "sdcpp":
            return frozenset({"steps", "guidance", "width", "height"})
        keys = {"steps", "guidance", "width", "height", "qualityPreset", "previewVae"}
        if _is_unet(repo):
            keys.add("sampler")
        else:
            keys |= {"cacheStrategy", "cacheThreshold"}
            if _is_flow_matching(repo):
                keys.add("cfgDecay")
        if hardware.kind == "cuda":
            keys.add("fp8LayerwiseCasting")
        return frozenset(keys)
    engine = _video_engine(variant)
    if engine in {"ltx2-mlx", "longlive"}:
        return frozenset({"width", "height", "numFrames", "fps"})
    if engine == "mlx-video":
        return frozenset({"width", "height", "numFrames", "fps", "steps", "guidance"})
    keys = {
        "steps", "guidance", "width", "height", "numFrames", "fps",
        "cfgDecay", "previewVae", "cacheStrategy", "cacheThreshold",
    }
    if hardware.kind == "cuda":
        keys.add("fp8LayerwiseCasting")
    return frozenset(keys)


def _cache_strategy_allowed(variant: dict[str, Any], kind: str, strategy: str) -> bool:
    repo = _lower(variant.get("repo"))
    if strategy in {"none", "fbcache", "taylorseer", "pab"}:
        return True
    if strategy == "teacache":
        if kind == "image":
            return "flux" in repo
        return any(token in repo for token in _TEACACHE_VIDEO_TOKENS) and not repo.startswith("wan-ai/")
    return False



_DISTILLED_STEPS_MAX = 8


def _generic_entry(variant: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Catalog-declared tuned steps / guidance (``defaultSteps`` / ``cfgOverride``).

    The Studios send their own step and guidance values, and the backend substitutes the
    catalog ones only when those equal the request schema's defaults, so a distilled model
    picked in the Studio can run at 24-30 steps and guidance 5-6 unless the user sets them.
    """
    settings: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    name = str(variant.get("name") or "This model")
    steps = variant.get("defaultSteps")
    if isinstance(steps, (int, float)) and steps > 0:
        settings["steps"] = int(steps)
        if steps <= _DISTILLED_STEPS_MAX:
            reasons["steps"] = f"{name} is distilled to {int(steps)} steps; more steps cost time in proportion and rarely look better."
        else:
            reasons["steps"] = f"{int(steps)} steps is the catalog setting for {name}."
    guidance = variant.get("cfgOverride")
    if isinstance(guidance, (int, float)):
        settings["guidance"] = max(1.0, float(guidance))
        if guidance <= 1.0:
            reasons["guidance"] = "Runs without classifier-free guidance; 1.0 keeps it off (higher values add a second pass per step on engines that run real guidance)."
        else:
            reasons["guidance"] = f"{float(guidance):g} is the guidance the model card uses."
    return settings, reasons


def _clamp(variant: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    max_side = variant.get("maxSide")
    min_side = variant.get("minSide")
    for key in ("width", "height"):
        if key not in out:
            continue
        value = int(out[key])
        if isinstance(max_side, (int, float)) and max_side > 0:
            value = min(value, int(max_side))
        if isinstance(min_side, (int, float)) and min_side > 0:
            value = max(value, int(min_side))
        out[key] = value
    if "steps" in out:
        out["steps"] = max(1, min(100, int(out["steps"])))
    if "guidance" in out:
        out["guidance"] = max(1.0, min(20.0, float(out["guidance"])))
    return out


_chip_lock = threading.Lock()
_chip_cache: dict[str, str | None] = {}


def host_chip() -> str | None:
    """This machine's chip name ("Apple M4 Max"), or None when it cannot be read."""
    with _chip_lock:
        if "value" in _chip_cache:
            return _chip_cache["value"]
    chip: str | None = None
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2,
            )
            chip = out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            chip = None
    with _chip_lock:
        _chip_cache["value"] = chip
    return chip


def recommended_settings(
    variant: dict[str, Any],
    kind: str,
    hardware: HardwareClass | None = None,
    chip: str | None = None,
) -> dict[str, Any] | None:
    """The ``recommendedSettings`` payload for ``variant`` on ``hardware``, or None.

    ``hardware`` and ``chip`` default to this machine. Pass ``chip=""`` to describe a machine
    whose chip is unknown (nothing can then count as measured).
    """
    if kind not in {"image", "video"}:
        raise ValueError(f"kind must be 'image' or 'video', not {kind!r}")
    if hardware is None:
        from backend_service.helpers.hardware_class import host_hardware_class

        hardware = host_hardware_class()
    if chip is None:
        chip = host_chip()

    settings, reasons = _generic_entry(variant)
    for entry in _ENTRIES:
        if entry.matches(variant, kind, hardware):
            settings.update(entry.settings)
            reasons.update(entry.reasons)

    accepted = accepted_settings(variant, kind, hardware)
    strategy = settings.get("cacheStrategy")
    if strategy is not None and not _cache_strategy_allowed(variant, kind, str(strategy)):
        settings.pop("cacheStrategy", None)
        settings.pop("cacheThreshold", None)
    if "cacheThreshold" in settings and "cacheStrategy" not in settings:
        settings.pop("cacheThreshold")
    settings = {key: value for key, value in _clamp(variant, settings).items() if key in accepted}
    if not settings:
        return None

    basis_by_key = {key: "estimated" for key in settings}
    measured_on: str | None = None
    for measurement in _MEASURED.get(str(variant.get("id") or ""), ()):
        if measurement.bucket != hardware.bucket or not chip or measurement.chip != chip:
            continue
        for key in measurement.keys & settings.keys():
            basis_by_key[key] = "measured"
            if key in measurement.reasons:
                reasons[key] = measurement.reasons[key]
        measured_on = f"{measurement.chip}, {hardware.total_gb:.0f} GB"

    overall = "measured" if all(basis == "measured" for basis in basis_by_key.values()) else "estimated"
    return {
        "hardwareBucket": hardware.bucket,
        "hardwareLabel": hardware.label,
        "basis": overall,
        "basisByKey": basis_by_key,
        "measuredOn": measured_on if "measured" in basis_by_key.values() else None,
        "settings": settings,
        "reasons": {key: reasons[key] for key in settings if key in reasons},
    }


def annotate_recommendations(
    families: list[dict[str, Any]],
    kind: str,
    hardware: HardwareClass | None = None,
    chip: str | None = None,
) -> list[dict[str, Any]]:
    """Set ``recommendedSettings`` on every variant of a catalog payload (copies the dicts).

    ``hardware`` defaults to this machine, read once for the whole payload; ``chip`` defaults
    to this machine's chip, so pass it with a ``hardware`` that describes another machine.
    """
    if hardware is None:
        from backend_service.helpers.hardware_class import host_hardware_class

        hardware = host_hardware_class()
    if chip is None:
        chip = host_chip()
    annotated: list[dict[str, Any]] = []
    for family in families:
        variants = []
        for variant in family.get("variants") or []:
            enriched = dict(variant)
            enriched["recommendedSettings"] = recommended_settings(enriched, kind, hardware, chip)
            variants.append(enriched)
        annotated.append({**family, "variants": variants})
    return annotated
