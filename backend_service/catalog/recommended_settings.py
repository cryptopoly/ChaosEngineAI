"""Recommended Studio settings per image / video model and machine: the data.

The rules that read this table (which settings an engine accepts, the measured / estimated
basis, clamping) live in ``backend_service/helpers/recommendations.py``.

``_ENTRIES`` is keyed by variant id and by hardware: an entry can apply everywhere, to one
kind of machine (``apple`` / ``cuda`` / ``cpu``), to one bucket (``apple-64``) or to machines
below a usable-memory size; later matching entries override earlier ones. ``_MEASURED`` holds
settings timed on a machine; a key counts as measured only on a machine with the same bucket
and chip. Every reason is one line a user reads in the Studio panel: say where a value comes
from (model card, upstream docs, an FU row, a run) and never claim a timing that was not taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend_service.helpers.hardware_class import HardwareClass


@dataclass(frozen=True)
class _Entry:
    kind: str
    # Variant ids; a trailing "*" matches an id prefix. Ids, not repos: the FLUX.1-dev LoRA,
    # mflux and GGUF rows share the base repo but need different settings.
    match: tuple[str, ...]
    settings: dict[str, Any]
    reasons: dict[str, str]
    hardware: str = "*"  # "*", "apple", "cuda", "cpu" or a bucket such as "apple-64"
    below_usable_gb: float | None = None

    def matches(self, variant: dict[str, Any], kind: str, hardware: HardwareClass) -> bool:
        if kind != self.kind:
            return False
        ident = str(variant.get("id") or "")
        if not any(
            ident.startswith(pattern[:-1]) if pattern.endswith("*") else ident == pattern
            for pattern in self.match
        ):
            return False
        if self.hardware != "*" and self.hardware not in {hardware.kind, hardware.bucket}:
            return False
        if self.below_usable_gb is not None and hardware.usable_gb >= self.below_usable_gb:
            return False
        return True


@dataclass(frozen=True)
class _Measurement:
    """Settings timed on one machine; ``basis`` is measured only on a matching machine."""

    bucket: str
    chip: str
    keys: frozenset[str]
    reasons: dict[str, str] = field(default_factory=dict)


_FBCACHE_IMAGE = "First Block Cache at 0.12 skips repeated transformer work between similar steps: about 1.8x faster on FLUX.1-dev with no visible change (diffusers write-up, FU-015). Not timed in this app yet."
_FBCACHE_VIDEO = "First Block Cache at the video threshold 0.08 skips repeated transformer work between similar steps; 0.08 is the tighter setting that keeps motion consistent (FU-015). Not timed in this app yet."
_CFG_DECAY_VIDEO = "Relaxes guidance toward 1.5 over the run so late steps stop over-saturating; flow-match video models only."
_FP8 = "Stores the transformer in FP8 (about half the memory) on RTX 40-series and newer cards; older cards skip it. Not yet run in this app on CUDA (FU-024)."
_CPU_SIZE = "CPU only: a 512 px image is about a quarter of the work per step of a 1024 px one."

_ENTRIES: tuple[_Entry, ...] = (
    # ---- Image, diffusers -------------------------------------------------------------
    _Entry(
        "image",
        ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-gguf-*", "black-forest-labs/FLUX.1-dev-nunchaku-int4"),
        {"steps": 28, "guidance": 3.5},
        {
            "steps": "28 steps is the diffusers FLUX pipeline default for FLUX.1-dev.",
            "guidance": "3.5 is the guidance the FLUX.1-dev model card uses.",
        },
    ),
    _Entry(
        "image",
        ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-gguf-*"),
        {"cacheStrategy": "fbcache", "cacheThreshold": 0.12},
        {"cacheStrategy": _FBCACHE_IMAGE, "cacheThreshold": _FBCACHE_IMAGE},
    ),
    _Entry(
        "image",
        ("black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell-gguf-*", "black-forest-labs/FLUX.1-schnell-mflux"),
        {"steps": 4},
        {"steps": "FLUX.1-schnell is distilled to 4 steps (model card); more steps cost time and do not help."},
    ),
    _Entry(
        "image",
        ("black-forest-labs/FLUX.1-dev-mflux",),
        {"guidance": 3.5},
        {"guidance": "3.5 is the guidance the FLUX.1-dev model card uses."},
    ),
    _Entry(
        "image",
        ("stabilityai/stable-diffusion-3.5-medium",),
        {"steps": 40, "guidance": 4.5},
        {
            "steps": "40 steps is the Stability AI model card setting for SD 3.5 Medium.",
            "guidance": "4.5 is the guidance the SD 3.5 Medium model card uses.",
        },
    ),
    _Entry(
        "image",
        ("stabilityai/stable-diffusion-3.5-large-turbo",),
        {"steps": 4, "guidance": 1.0},
        {
            "steps": "SD 3.5 Large Turbo is distilled to 4 steps (model card).",
            "guidance": "The model card runs Turbo without guidance; 1.0 turns it off and halves the work per step.",
        },
    ),
    _Entry(
        "image",
        ("stabilityai/stable-diffusion-xl-base-1.0",),
        {"sampler": "ays_dpmpp_2m_sdxl", "steps": 10},
        {
            "sampler": "NVIDIA's Align Your Steps schedule for SDXL keeps detail at 10 steps where Karras and Euler look soft (FU-020).",
            "steps": "The Align Your Steps schedule has 10 steps.",
        },
    ),
    _Entry(
        "image",
        ("lightx2v/Qwen-Image-Lightning",),
        {"steps": 8, "guidance": 1.0},
        {
            "steps": "Qwen-Image-Lightning is distilled to 8 steps; the Studio's 24 would take three times as long.",
            "guidance": "Lightning runs at guidance 1.0 (no classifier-free guidance).",
        },
    ),
    _Entry(
        "image",
        ("Tongyi-MAI/Z-Image-Turbo",),
        {"steps": 9, "guidance": 1.0},
        {
            "steps": "Z-Image-Turbo is distilled for 9 steps (8 transformer passes, model card).",
            "guidance": "The model card runs Turbo without guidance; 1.0 turns it off.",
        },
    ),
    _Entry(
        "image",
        ("black-forest-labs/FLUX.2-klein-4B",),
        {"steps": 4, "guidance": 1.0},
        {
            "steps": "FLUX.2 klein 4B is distilled to 4 steps (model card).",
            "guidance": "The distilled klein runs at guidance 1.0 (model card).",
        },
    ),
    _Entry(
        "image",
        ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.2-dev", "Qwen/Qwen-Image"),
        {"fp8LayerwiseCasting": True},
        {"fp8LayerwiseCasting": _FP8},
        hardware="cuda",
        below_usable_gb=25,
    ),
    # ---- Image, mflux (Apple Silicon) -------------------------------------------------
    _Entry(
        "image",
        ("speridlabs/iris-3b-mflux",),
        {"steps": 4, "width": 1024, "height": 1024},
        {
            "steps": "Iris-3B Turbo is distilled to 4 steps; 24 steps gave the same image (FU-124).",
            "width": "1024 px is Iris-3B's native size; it paints pixels directly, so 512 px comes out blocky.",
            "height": "1024 px is Iris-3B's native size.",
        },
    ),
    _Entry(
        "image",
        ("Qwen/Qwen-Image-2.1-Turbo-mflux",),
        {"steps": 8, "width": 1024, "height": 1024},
        {
            "steps": "Qwen-Image-2.1 Turbo is distilled to 8 steps (model card).",
            "width": "1024 px is the size the model is tuned for.",
            "height": "1024 px is the size the model is tuned for.",
        },
    ),
    _Entry(
        "image",
        ("Qwen/Qwen-Image-2.1-mflux", "Qwen/Qwen-Image-2.1-mflux-q8"),
        {"steps": 40, "width": 1024, "height": 1024},
        {
            "steps": "40 steps is the catalog setting for the base model.",
            "width": "1024 px is the size the model is tuned for.",
            "height": "1024 px is the size the model is tuned for.",
        },
    ),
    _Entry(
        "image",
        ("black-forest-labs/flux-3-image-klein-mflux",),
        {"steps": 8, "width": 1024, "height": 1024},
        {
            "steps": "The Studio starts at 8: it looked the same as the maker's 16 on three test prompts and took about half the time on an M4 Max (not timed on this machine).",
            "width": "1024 px is the size the model is tuned for.",
            "height": "1024 px is the size the model is tuned for.",
        },
    ),
    # ---- Image, any engine on a CPU-only host -----------------------------------------
    _Entry(
        "image",
        ("*",),
        {"width": 512, "height": 512},
        {"width": _CPU_SIZE, "height": _CPU_SIZE},
        hardware="cpu",
    ),
    # ---- Video, ltx-2-mlx (Apple Silicon) ---------------------------------------------
    _Entry(
        "video",
        ("dgrauet/ltx-2.3-mlx-q4", "dgrauet/ltx-2.3-mlx-q8"),
        {"width": 768, "height": 512, "numFrames": 97, "fps": 24},
        {
            "width": "768 x 512 is the size the app's LTX-2.3 runs were checked at; it renders in two stages and upscales 2x.",
            "height": "768 x 512 is the size the app's LTX-2.3 runs were checked at.",
            "numFrames": "97 frames is a 4 s clip at 24 fps, the ltx-2-mlx default length; memory barely grows with frames (18 GB peak at 257 frames on a 64 GB Mac).",
            "fps": "LTX-2.3 was trained at 24 fps; other rates drift out of distribution.",
        },
    ),
    _Entry(
        "video",
        ("dgrauet/ltx-2.5-mlx-q4",),
        {"width": 768, "height": 512, "fps": 24},
        {
            "width": "Same two-stage MLX pipeline as LTX-2.3; 768 x 512 is the size checked there. Not timed on LTX-2.5.",
            "height": "Same two-stage MLX pipeline as LTX-2.3; 768 x 512 is the size checked there.",
            "fps": "The LTX-2 family is trained at 24 fps.",
        },
    ),
    # ---- Video, diffusers -------------------------------------------------------------
    _Entry(
        "video",
        ("Lightricks/LTX-Video", "Lightricks/LTX-Video-gguf-*"),
        {"steps": 30, "guidance": 3.0, "cfgDecay": True, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "30 steps is the Lightricks model card setting.",
            "guidance": "LTX-Video is flow-matching; guidance above about 3.5 over-saturates (Lightricks card: 3).",
            "cfgDecay": _CFG_DECAY_VIDEO,
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan-AI/Wan2.1-T2V-14B-Diffusers", "samuelchristlie/Wan2.1-T2V-1.3B-GGUF-*", "city96/Wan2.1-T2V-14B-gguf-*"),
        {"steps": 30, "guidance": 5.0, "cfgDecay": True, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "30 steps is the app's tuned Wan default (the model card uses 50 for its showcase clips).",
            "guidance": "5.0 is the guidance on the Wan-AI model card.",
            "cfgDecay": _CFG_DECAY_VIDEO,
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",),
        {"steps": 30, "guidance": 4.0, "cfgDecay": True, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "30 steps is the app's tuned Wan default.",
            "guidance": "4.0 is the guidance on the Wan 2.2 A14B model card (the Studio's 5.0 runs hotter).",
            "cfgDecay": _CFG_DECAY_VIDEO,
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("hunyuanvideo-community/HunyuanVideo",),
        {"steps": 50, "guidance": 6.0, "cfgDecay": True, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "50 steps is the Tencent model card setting.",
            "guidance": "6.0 is the guidance on the HunyuanVideo model card.",
            "cfgDecay": _CFG_DECAY_VIDEO,
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("genmo/mochi-1-preview",),
        {"steps": 64, "guidance": 4.5, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "64 steps is Genmo's reference pipeline setting.",
            "guidance": "4.5 is the guidance in Genmo's reference pipeline.",
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("THUDM/CogVideoX-2b",),
        {"steps": 50, "guidance": 6.0, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "50 steps is the THUDM model card setting.",
            "guidance": "6.0 is the guidance on the CogVideoX-2b model card.",
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("THUDM/CogVideoX-5b", "THUDM/CogVideoX1.5-5B"),
        {"steps": 50, "guidance": 7.0, "cacheStrategy": "fbcache", "cacheThreshold": 0.08},
        {
            "steps": "50 steps is the THUDM model card setting.",
            "guidance": "7.0 is the guidance on the CogVideoX 5B model cards.",
            "cacheStrategy": _FBCACHE_VIDEO,
            "cacheThreshold": _FBCACHE_VIDEO,
        },
    ),
    _Entry(
        "video",
        ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "hunyuanvideo-community/HunyuanVideo"),
        {"fp8LayerwiseCasting": True},
        {"fp8LayerwiseCasting": _FP8},
        hardware="cuda",
        below_usable_gb=25,
    ),
)

# Settings timed on this repo's test machine (M4 Max, 64 GB), through the app's own
# generate routes, fixed prompts and seeds; wall time per image or clip with the model
# already loaded. Every number and command is in docs/hub/progress/settings-rec.md.
_M4_MAX = "Apple M4 Max"
_MEASURED: dict[str, tuple[_Measurement, ...]] = {
    "speridlabs/iris-3b-mflux": (
        _Measurement(
            "apple-64",
            _M4_MAX,
            frozenset({"steps", "width", "height"}),
            {
                "steps": "4 steps: 11-13 s per 1024 px image here (3 runs); 24 steps gave the same picture (FU-124).",
                "width": "1024 px: 11-13 s and about 13 GB here; 768 px is 8 s for the same memory, with a different but not better picture.",
                "height": "1024 px: 11-13 s and about 13 GB here (3 runs).",
            },
        ),
    ),
    "black-forest-labs/flux-3-image-klein-mflux": (
        _Measurement(
            "apple-64",
            _M4_MAX,
            frozenset({"steps", "width", "height"}),
            {
                "steps": "8 steps (the Studio default): 40 s per 1024 px image here against 70-78 s at the maker's 16, with no visible difference on three test prompts.",
                "width": "1024 px: 40 s at 8 steps and about 15 GB here (3 runs).",
                "height": "1024 px: 40 s at 8 steps and about 15 GB here (3 runs).",
            },
        ),
    ),
    "dgrauet/ltx-2.3-mlx-q4": (
        _Measurement(
            "apple-64",
            _M4_MAX,
            frozenset({"width", "height", "numFrames", "fps"}),
            {
                "width": "768 x 512: a 4 s clip in 135-138 s here, 15 GB peak (2 prompts); 1280 x 704 x 121 frames took 363 s.",
                "height": "768 x 512: a 4 s clip in 135-138 s here, 15 GB peak (2 prompts).",
                "numFrames": "97 frames (4 s at 24 fps): 135-138 s here against 55 s for the Studio's default 33 frames (1.4 s); both coherent, memory the same.",
                "fps": "LTX-2.3 was trained at 24 fps; the clips above were timed at 24.",
            },
        ),
    ),
    "Qwen/Qwen-Image-2.1-mflux": (
        _Measurement(
            "apple-64",
            _M4_MAX,
            frozenset({"steps", "width", "height"}),
            {
                "steps": "40 steps: 258-271 s per 1024 px image here (2 prompts). 28 steps took 187-191 s and looked as good on both; use it for drafts.",
                "width": "1024 px: 258-271 s at 40 steps, up to 50 GB of process memory here.",
                "height": "1024 px: 258-271 s at 40 steps, up to 50 GB of process memory here.",
            },
        ),
    ),
    "Qwen/Qwen-Image-2.1-Turbo-mflux": (
        _Measurement(
            "apple-64",
            _M4_MAX,
            frozenset({"steps", "width", "height"}),
            {
                "steps": "8 steps: 72-75 s per 1024 px image here (3 runs). 6 steps saved 15 s and looked close, but the model is tuned for 8.",
                "width": "1024 px: 72-75 s and up to 50 GB of process memory here (3 runs).",
                "height": "1024 px: 72-75 s and up to 50 GB of process memory here (3 runs).",
            },
        ),
    ),
}
