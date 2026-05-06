"""stable-diffusion.cpp video runtime (FU-008).

Wraps the staged ``sd`` binary from ``leejet/stable-diffusion.cpp`` (MIT)
as a subprocess engine, mirroring ``MlxVideoEngine`` and ``LongLiveEngine``.
Targets cross-platform GGUF video generation: Metal on Apple Silicon,
CUDA on Windows/Linux. The binary itself supports SD 1.x/2.x/XL, FLUX.1/2,
**Wan 2.1 / Wan 2.2 video**, Qwen Image, and Z-Image — this engine wires
only the video subset.

SCOPE
-----
Phase C scaffold: ``probe()`` reports availability based on the staged
``sd`` binary (path resolved by the Tauri shell into ``CHAOSENGINE_SDCPP_BIN``).
``generate()`` raises ``NotImplementedError`` until the per-model CLI
arg builders + stdout progress parser land. The hooks the manager calls
(``probe``/``preload``/``unload``) match the contract expected by
``VideoRuntimeManager`` so routing can be wired before the heavy lift.

ROUTING
-------
- Apple Silicon: prefer mlx-video for LTX-2 first, then sd.cpp for
  Wan GGUF, then diffusers MPS as fallback.
- Windows/Linux + CUDA: prefer LongLive for Wan 2.1 long-clip mode,
  then sd.cpp for Wan GGUF, then diffusers + bnb NF4 for full-precision.
- Fallback: diffusers everywhere.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)


# Repos sd.cpp supports natively via GGUF. Kept narrow on the video side —
# the binary supports image families too, but those route through
# image_runtime (FU-008 image side, separate engine).
#
# Wan 2.1 GGUF (city96/Wan2.1-T2V-{1.3B,14B}-gguf) and Wan 2.2 GGUF
# (QuantStack/Wan2.2-T2V-A14B-GGUF) are the immediate targets — those
# unlock the Mac Metal video path that diffusers MPS cannot serve.
_SUPPORTED_REPOS: frozenset[str] = frozenset({
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
})


def supported_repos() -> frozenset[str]:
    """Repo ids the sd.cpp video engine accepts."""
    return _SUPPORTED_REPOS


def _is_sdcpp_video_repo(repo: str | None) -> bool:
    if not repo:
        return False
    return repo in _SUPPORTED_REPOS


def _resolve_sd_binary() -> Path | None:
    """Resolve the staged ``sd`` binary path.

    The Tauri shell sets ``CHAOSENGINE_SDCPP_BIN`` to the staged path
    (extracted from the runtime archive on packaged builds, or resolved
    via ``resolve_sd_cpp`` in dev). Falls back to ``~/.chaosengine/bin/sd``
    so locally-built binaries work without env wiring.
    """
    env_path = os.environ.get("CHAOSENGINE_SDCPP_BIN")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    home = os.environ.get("HOME")
    if home:
        managed = Path(home) / ".chaosengine" / "bin" / "sd"
        if managed.exists():
            return managed

    return None


class SdCppVideoEngine:
    """Subprocess wrapper around stable-diffusion.cpp for video GGUF.

    Phase C scaffold — probe + routing only. ``generate()`` is staged
    behind ``NotImplementedError`` until the CLI arg builders and stdout
    progress parser land. See FU-008 in CLAUDE.md.
    """

    runtime_label = "stable-diffusion.cpp"

    def __init__(self) -> None:
        self._loaded_repo: str | None = None

    def probe(self) -> VideoRuntimeStatus:
        binary = _resolve_sd_binary()
        if binary is None:
            return VideoRuntimeStatus(
                activeEngine="sd.cpp",
                realGenerationAvailable=False,
                expectedDevice=None,
                missingDependencies=["sd"],
                message=(
                    "stable-diffusion.cpp binary not staged. Build "
                    "leejet/stable-diffusion.cpp and either set "
                    "CHAOSENGINE_SDCPP_BIN or copy `sd` to "
                    "~/.chaosengine/bin/. See FU-008 in CLAUDE.md."
                ),
            )
        device = "mps" if platform.system() == "Darwin" else "cuda"
        return VideoRuntimeStatus(
            activeEngine="sd.cpp",
            realGenerationAvailable=False,  # scaffold — generate() not wired yet
            device=device,
            expectedDevice=device,
            message=(
                f"sd.cpp binary detected at {binary}. Generation pipeline "
                "still scaffold — Wan GGUF generate path lands in the "
                "next iteration of FU-008."
            ),
            loadedModelRepo=self._loaded_repo,
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        if not _is_sdcpp_video_repo(repo):
            raise RuntimeError(
                f"sd.cpp does not support {repo}. Supported: "
                f"{sorted(_SUPPORTED_REPOS)}"
            )
        self._loaded_repo = repo
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        if repo is None or repo == self._loaded_repo:
            self._loaded_repo = None
        return self.probe()

    def generate(self, config: VideoGenerationConfig) -> GeneratedVideo:
        raise NotImplementedError(
            "sd.cpp video generate() is scaffold-only. Wan GGUF "
            "subprocess wiring lands in the next FU-008 iteration: "
            "build CLI args from VideoGenerationConfig (prompt, "
            "num_frames, fps, steps, guidance, seed, output path), "
            "spawn the staged `sd` binary, stream stdout into "
            "VIDEO_PROGRESS, then return the rendered mp4."
        )
