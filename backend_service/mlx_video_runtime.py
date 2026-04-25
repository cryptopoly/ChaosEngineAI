"""Apple Silicon MLX-native video runtime (FU-009 scaffold).

Wraps Blaizzy/mlx-video (MIT) as a subprocess engine, mirroring the
``MfluxImageEngine`` pattern in [image_runtime.py]. Covers Wan2.1
(1.3B/14B), Wan2.2 (T2V-14B, TI2V-5B, I2V-14B), and LTX-2 (19B) on
macOS arm64 — CUDA-only LongLive (see ``longlive_engine``) stays the
Windows/Linux path.

SCOPE FOR THIS PHASE
--------------------
Scaffold + probe + repo-routing helper only. End-to-end generation is
deferred until FU-009 promotes from watch/scaffold to in-flight — see
``generate()`` which raises ``NotImplementedError``. The video runtime
manager exposes ``mlx_video_capabilities()`` so the Setup page can
surface install/availability state without the Studio trying to route a
real run through the half-built engine.

Full generation will shell out to mlx-video's module entry points
(``python -m mlx_video.ltx_2.generate ...`` or equivalent) via
``subprocess.Popen`` so mlx-video's torch-free deps stay quarantined and
failures surface as a clean non-zero exit code rather than an import
error in the sidecar.
"""

from __future__ import annotations

import importlib.util
import platform
from typing import Any

from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)


# Repos that will route to mlx-video on Apple Silicon once generation
# lands. Kept here (not inline in ``_is_mlx_video_repo``) so the Setup
# page and tests can introspect the supported set.
#
# Upstream mlx-video entry points as of 2026-04:
#   - mlx_video.ltx_2.generate          — LTX-2 19B (T2V/I2V/A2V)
#   - mlx_video.wan_2_1.generate        — Wan2.1 1.3B / 14B (T2V)
#   - mlx_video.wan_2_2.generate        — Wan2.2 T2V-14B / TI2V-5B / I2V-14B
_SUPPORTED_REPOS: frozenset[str] = frozenset({
    # LTX-2
    "Lightricks/LTX-2-19B",
    # Wan2.1
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    # Wan2.2
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
})


def supported_repos() -> frozenset[str]:
    """Return the set of repo ids the MLX video engine will accept.

    Exposed so the Setup page and tests can enumerate the supported
    surface without importing the engine class (which pulls in
    ``video_runtime`` and its torch-warmup side effects).
    """
    return _SUPPORTED_REPOS


def _is_mlx_video_repo(repo: str | None) -> bool:
    """Route helper for the video manager.

    Returns True only for repos mlx-video supports natively. The manager
    should still consult ``MlxVideoEngine.probe()`` before dispatching —
    a supported repo on an Intel Mac must fall through to diffusers.
    """
    if not repo:
        return False
    return repo in _SUPPORTED_REPOS


class MlxVideoEngine:
    """Subprocess wrapper around Blaizzy/mlx-video for Apple Silicon.

    Probe-only in this phase. ``generate()`` raises until FU-009 lands
    — see the module docstring for the staged plan.
    """

    runtime_label = "mlx-video (MLX native)"

    def __init__(self) -> None:
        self._loaded_repo: str | None = None

    # ---------- public API ----------

    def probe(self) -> VideoRuntimeStatus:
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            return VideoRuntimeStatus(
                activeEngine="mlx-video",
                realGenerationAvailable=False,
                expectedDevice=None,
                message=(
                    "mlx-video runs on Apple Silicon only. Use the diffusers "
                    "runtime on Intel Mac / Windows / Linux, or LongLive on "
                    "CUDA."
                ),
            )
        if importlib.util.find_spec("mlx_video") is None:
            return VideoRuntimeStatus(
                activeEngine="mlx-video",
                realGenerationAvailable=False,
                expectedDevice="mps",
                missingDependencies=["mlx-video"],
                message=(
                    "mlx-video not installed — add it from the Setup page "
                    "to enable the native Apple Silicon video runtime "
                    "(Wan2.1 / Wan2.2 / LTX-2)."
                ),
            )
        # Package present on Apple Silicon, but the generate() path is
        # still scaffold-only — surface that honestly so the Studio
        # doesn't offer the engine as the active route yet.
        return VideoRuntimeStatus(
            activeEngine="mlx-video",
            realGenerationAvailable=False,
            device="mps",
            expectedDevice="mps",
            message=(
                "mlx-video is installed — generation path is scaffold-only "
                "in this build (FU-009). Studio will use diffusers for now."
            ),
            loadedModelRepo=self._loaded_repo,
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        """Remember the selection. No weights load yet.

        mlx-video loads inside the subprocess so there's nothing for
        this process to hold. Mirrors ``LongLiveEngine.preload``.
        """
        if not _is_mlx_video_repo(repo):
            raise RuntimeError(
                f"mlx-video does not support {repo}. Supported: "
                f"{sorted(_SUPPORTED_REPOS)}"
            )
        self._loaded_repo = repo
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        if repo is None or self._loaded_repo == repo:
            self._loaded_repo = None
        return self.probe()

    def generate(self, config: VideoGenerationConfig) -> GeneratedVideo:
        """Scaffold-only — raises until FU-009 promotes to in-flight.

        When implemented, this will shell out to the mlx-video module
        entry point matching ``config.repo`` (``mlx_video.ltx_2``,
        ``mlx_video.wan_2_1``, or ``mlx_video.wan_2_2``), stream
        progress lines to ``VIDEO_PROGRESS``, and return the rendered
        mp4 as ``GeneratedVideo``.
        """
        raise NotImplementedError(
            "mlx-video generation is not wired yet (FU-009 scaffold). "
            "Use the diffusers runtime for now — it supports every "
            f"repo in mlx-video's catalog on MPS. Blocked repo: {config.repo}."
        )
