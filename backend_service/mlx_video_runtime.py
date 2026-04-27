"""Apple Silicon MLX-native video runtime (FU-009).

Wraps Blaizzy/mlx-video (MIT) as a subprocess engine, mirroring the
``LongLiveEngine`` pattern in ``backend_service.longlive_engine``. On
Apple Silicon this is the preferred runtime for LTX-2 — it skips the
``diffusers + torch.mps`` round-trip, runs natively in MLX, and uses
weights that have already been converted to MLX format upstream.

SCOPE
-----
LTX-2 only. Wan 2.1 / Wan 2.2 via mlx-video also exist upstream but
require a separate ``mlx_video.models.wan_2.convert`` step on raw HF
weights (no pre-converted MLX repo today). Until that conversion is
either bundled or scripted, Wan stays on the diffusers MPS path —
which is fine for Wan 2.1 1.3B / Wan 2.2 5B, both of which fit comfort-
ably in unified memory on a 64 GB Mac.

LTX-2 ships pre-converted at ``prince-canuma/LTX-2-*`` (T2V/I2V/A2V).
Those repos load directly into ``mlx_video.ltx_2.generate`` via
``--model`` so we route them straight to the subprocess.

The LongLive subprocess pattern is reused verbatim: spawn a Python from
``CHAOSENGINE_VIDEO_PYTHON`` (or the workspace ``.venv``), stream stdout
into the progress callback, then read the rendered mp4 from a temp
workspace and return it as ``GeneratedVideo``. This keeps mlx-video's
torch-free deps quarantined and surfaces failures as a non-zero exit
code rather than an import error in the sidecar.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
    _resolve_video_python,
)


# Repos that route to mlx-video on Apple Silicon. Kept as a frozenset so
# the Setup page and tests can introspect the supported surface without
# importing the engine class.
#
# Only LTX-2 ships pre-converted MLX weights today — Wan paths go through
# diffusers MPS until we automate the ``mlx_video.models.wan_2.convert``
# step. See module docstring for the staged plan.
_SUPPORTED_REPOS: frozenset[str] = frozenset({
    "prince-canuma/LTX-2-distilled",
    "prince-canuma/LTX-2-dev",
    "prince-canuma/LTX-2.3-distilled",
    "prince-canuma/LTX-2.3-dev",
})


# Maps repo prefix → mlx-video module entry point. Today every supported
# repo is LTX-2; the dict shape is kept so Wan (and future families) can
# be slotted in without re-plumbing the dispatch.
_REPO_ENTRY_POINTS: dict[str, str] = {
    "prince-canuma/LTX-2": "mlx_video.ltx_2.generate",
}


def supported_repos() -> frozenset[str]:
    """Repo ids the MLX video engine accepts.

    Exposed so the Setup page and tests can enumerate the supported set
    without importing the engine class (which would pull in the heavy
    ``video_runtime`` module and its torch-warmup side effects).
    """
    return _SUPPORTED_REPOS


def _is_mlx_video_repo(repo: str | None) -> bool:
    """Routing helper for the video manager.

    Returns ``True`` only for repos mlx-video supports natively. The
    manager still consults ``MlxVideoEngine.probe()`` before dispatching
    — a supported repo on an Intel Mac must fall through to diffusers.
    """
    if not repo:
        return False
    return repo in _SUPPORTED_REPOS


def _resolve_entry_point(repo: str) -> str:
    for prefix, entry in _REPO_ENTRY_POINTS.items():
        if repo.startswith(prefix):
            return entry
    raise RuntimeError(
        f"No mlx-video entry point registered for {repo}. "
        f"Supported prefixes: {sorted(_REPO_ENTRY_POINTS)}"
    )


class _ProgressSink(Protocol):
    def __call__(self, phase: str, message: str, fraction: float) -> None: ...


class MlxVideoEngine:
    """Subprocess wrapper around Blaizzy/mlx-video for Apple Silicon."""

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
                    "(LTX-2)."
                ),
            )
        return VideoRuntimeStatus(
            activeEngine="mlx-video",
            realGenerationAvailable=True,
            device="mps",
            expectedDevice="mps",
            message=(
                "mlx-video ready — LTX-2 generation will route through the "
                "native MLX backend. Wan paths still use diffusers MPS until "
                "the mlx-video Wan conversion step is bundled."
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

    def generate(
        self,
        config: VideoGenerationConfig,
        *,
        on_progress: _ProgressSink | None = None,
    ) -> GeneratedVideo:
        """Run a single LTX-2 generation via ``python -m mlx_video.ltx_2.generate``.

        Streams stdout into ``on_progress`` so the Studio can show step
        progress. Reads the rendered mp4 from a tmp workspace and surfaces
        it as ``GeneratedVideo``. The subprocess inherits a
        ``CHAOSENGINE_MLX_VIDEO_*``-friendly environment so users can pin
        the venv with ``CHAOSENGINE_VIDEO_PYTHON`` if they keep mlx-video
        in a separate Python.
        """
        probe = self.probe()
        if not probe.realGenerationAvailable:
            raise RuntimeError(probe.message)
        if not _is_mlx_video_repo(config.repo):
            raise RuntimeError(
                f"mlx-video does not support {config.repo}. Supported: "
                f"{sorted(_SUPPORTED_REPOS)}"
            )

        if on_progress:
            on_progress("loading", "Preparing mlx-video workspace", 0.0)

        workspace = Path(tempfile.mkdtemp(prefix="mlx-video-run-"))
        try:
            output_path = workspace / "out.mp4"
            cmd = self._build_cmd(config, output_path)
            self._launch(cmd, workspace, on_progress)

            if not output_path.exists():
                raise RuntimeError(
                    f"mlx-video finished but no mp4 was produced at "
                    f"{output_path}. Check the subprocess log above."
                )
            data = output_path.read_bytes()
            return GeneratedVideo(
                seed=config.seed if config.seed is not None else 0,
                bytes=data,
                extension="mp4",
                mimeType="video/mp4",
                durationSeconds=config.numFrames / max(1, config.fps),
                frameCount=config.numFrames,
                fps=config.fps,
                width=config.width,
                height=config.height,
                runtimeLabel=self.runtime_label,
                runtimeNote="mlx-video subprocess (MLX native)",
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # ---------- internals ----------

    def _build_cmd(self, config: VideoGenerationConfig, output_path: Path) -> list[str]:
        """Compose the ``python -m mlx_video.<entry> --...`` invocation.

        Split out so tests can assert the CLI shape without spawning a
        real subprocess. The flag set mirrors mlx-video upstream's
        ``ltx_2.generate`` argparse surface as of 2026-04 (Blaizzy/mlx-video).
        """
        entry = _resolve_entry_point(config.repo)
        python = _resolve_video_python()
        cmd = [
            python,
            "-m",
            entry,
            "--model",
            config.repo,
            "--prompt",
            config.prompt,
            "--num-frames",
            str(config.numFrames),
            "--fps",
            str(config.fps),
            "--height",
            str(config.height),
            "--width",
            str(config.width),
            "--steps",
            str(config.steps),
            "--guidance",
            str(config.guidance),
            "--output",
            str(output_path),
        ]
        if config.negativePrompt:
            cmd.extend(["--negative-prompt", config.negativePrompt])
        if config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])
        return cmd

    def _launch(
        self,
        cmd: list[str],
        workspace: Path,
        on_progress: _ProgressSink | None,
    ) -> None:
        """Spawn the subprocess and stream stdout into the progress sink."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        start = time.monotonic()
        process = subprocess.Popen(
            cmd,
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Capture the tail of stdout/stderr so a fast-fail subprocess
        # error surfaces in the user-visible RuntimeError instead of
        # being silently lost to "code 1 after 0.2s. See stdout above."
        # — which is opaque on a packaged build where stdout never
        # makes it back to the user.
        tail_lines: list[str] = []
        max_tail_chars = 1500
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    tail_lines.append(stripped)
                    # Bound memory: drop oldest lines once the running
                    # tail exceeds max_tail_chars.
                    while sum(len(l) for l in tail_lines) > max_tail_chars and len(tail_lines) > 1:
                        tail_lines.pop(0)
                if not on_progress or not stripped:
                    continue
                fraction = _parse_step_fraction(stripped)
                if fraction is not None:
                    on_progress("diffusing", stripped[:120], fraction)
                else:
                    on_progress("diffusing", stripped[:120], 0.5)
        finally:
            return_code = process.wait()

        duration = time.monotonic() - start
        if return_code != 0:
            tail = "\n".join(tail_lines[-12:]) if tail_lines else "(no output captured)"
            raise RuntimeError(
                f"mlx-video subprocess exited with code {return_code} "
                f"after {duration:.1f}s. Tail of stdout/stderr:\n{tail}"
            )


def _parse_step_fraction(line: str) -> float | None:
    """Pull a 0.0–1.0 fraction out of a ``step N/M`` log line.

    mlx-video upstream emits progress as ``step 12/30`` style — we don't
    require an exact match because line formatting drifts between
    releases. Anything containing two integers separated by ``/`` and
    flanked by digits/whitespace is treated as a step counter.
    """
    import re

    match = re.search(r"(\d+)\s*/\s*(\d+)", line)
    if not match:
        return None
    cur, total = int(match.group(1)), int(match.group(2))
    if total <= 0:
        return None
    return min(1.0, max(0.0, cur / total))
