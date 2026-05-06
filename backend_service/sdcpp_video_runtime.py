"""stable-diffusion.cpp video runtime (FU-008).

Wraps the staged ``sd`` binary from ``leejet/stable-diffusion.cpp`` (MIT)
as a subprocess engine, mirroring ``MlxVideoEngine`` and ``LongLiveEngine``.
Targets cross-platform GGUF video generation: Metal on Apple Silicon,
CUDA on Windows/Linux. The binary itself supports SD 1.x/2.x/XL, FLUX.1/2,
**Wan 2.1 / Wan 2.2 video**, Qwen Image, and Z-Image — this engine wires
only the video subset.

SCOPE
-----
Phase 3 lift (FU-008): ``generate()`` is wired. Builds the CLI invocation
from a ``VideoGenerationConfig``, spawns the staged ``sd`` binary, parses
``step N/M`` lines off stdout into ``VIDEO_PROGRESS``, then reads the
output mp4 back as bytes for the standard ``GeneratedVideo`` contract.

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
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)


# Progress regex — sd.cpp emits ``[INFO] step N/M (..)`` style lines on
# stdout during the denoise loop. Loose pattern catches both the older
# ``step N/M`` and the newer ``[N/M]`` formats; whichever matches gets
# fed into ``VIDEO_PROGRESS``.
_STEP_RE = re.compile(r"(?:step\s+|\[)(\d+)\s*/\s*(\d+)")
_LAST_OUTPUT_LINES = 80
_RUNTIME_LABEL = "stable-diffusion.cpp"


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
                    "stable-diffusion.cpp binary not staged. Run "
                    "``./scripts/build-sdcpp.sh`` (or set "
                    "CHAOSENGINE_SDCPP_BIN) to build and install. "
                    "See FU-008 in CLAUDE.md."
                ),
            )
        device = "mps" if platform.system() == "Darwin" else "cuda"
        return VideoRuntimeStatus(
            activeEngine="sd.cpp",
            realGenerationAvailable=True,
            device=device,
            expectedDevice=device,
            message=(
                f"sd.cpp binary detected at {binary}. Wan GGUF "
                "generate path active — pass ``ggufRepo`` + "
                "``ggufFile`` on the catalog variant to route here."
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
        binary = _resolve_sd_binary()
        if binary is None:
            raise RuntimeError(
                "stable-diffusion.cpp binary not staged. "
                "Run ``./scripts/build-sdcpp.sh`` first."
            )
        if not _is_sdcpp_video_repo(config.repo):
            raise RuntimeError(
                f"sd.cpp does not support {config.repo}. "
                f"Supported: {sorted(_SUPPORTED_REPOS)}"
            )

        # The Wan video path needs a GGUF transformer file — sd.cpp
        # cannot consume a sharded diffusers safetensors snapshot
        # directly. The catalog variant pins ``ggufRepo`` + ``ggufFile``
        # for the GGUF lanes (e.g. QuantStack/Wan2.2-TI2V-5B-GGUF).
        if not config.ggufFile:
            raise RuntimeError(
                "sd.cpp video generate requires a GGUF variant. Pick a "
                "catalog entry that pins ``ggufRepo`` + ``ggufFile`` "
                "(e.g. Wan 2.2 TI2V 5B · GGUF Q4_K_M)."
            )

        seed = config.seed if config.seed is not None else int(time.time())

        with tempfile.TemporaryDirectory(prefix="chaosengine-sdcpp-") as tmpdir:
            # sd.cpp's single-file video outputs are .avi / .webm /
            # animated .webp (no native .mp4). webm is the smallest +
            # most broadly playable in the desktop's webview.
            output_path = Path(tmpdir) / f"sdcpp-{seed}.webm"
            model_path = self._resolve_gguf_path(config)
            args = self._build_cli_args(
                binary=binary,
                config=config,
                model_path=model_path,
                output_path=output_path,
                seed=seed,
            )
            output_bytes = self._run_subprocess(
                args=args,
                config=config,
                output_path=output_path,
            )

        duration = round(config.numFrames / max(1, config.fps), 3)
        return GeneratedVideo(
            seed=seed,
            bytes=output_bytes,
            extension="webm",
            mimeType="video/webm",
            durationSeconds=duration,
            frameCount=config.numFrames,
            fps=config.fps,
            width=config.width,
            height=config.height,
            runtimeLabel=_RUNTIME_LABEL,
            runtimeNote=(
                f"Generated via sd.cpp subprocess "
                f"({Path(model_path).name})."
            ),
            effectiveSteps=config.steps,
            effectiveGuidance=config.guidance,
        )

    # ------------------------------------------------------------------
    # CLI builders + subprocess plumbing
    # ------------------------------------------------------------------

    def _resolve_gguf_path(self, config: VideoGenerationConfig) -> str:
        """Resolve the absolute on-disk path for the GGUF transformer.

        The catalog variant carries ``ggufRepo`` (HF repo) + ``ggufFile``
        (filename within the repo); the standard diffusers download
        machinery pulls them into the HF cache. Reuse that — we just
        re-resolve the file path so sd.cpp can read it directly.
        """
        if not config.ggufFile or not config.ggufRepo:
            raise RuntimeError(
                "GGUF transformer required for sd.cpp video. Catalog variant "
                "must pin ``ggufRepo`` + ``ggufFile``."
            )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"huggingface_hub is required to resolve the GGUF path: {exc}"
            ) from exc
        return hf_hub_download(
            repo_id=config.ggufRepo,
            filename=config.ggufFile,
        )

    def _build_cli_args(
        self,
        *,
        binary: Path,
        config: VideoGenerationConfig,
        model_path: str,
        output_path: Path,
        seed: int,
    ) -> list[str]:
        """Map a ``VideoGenerationConfig`` onto sd.cpp's CLI flags.

        The mapping mirrors the ``--help`` output of leejet's master tip
        as of 2026-04-29 (master-593). If a future sd.cpp release renames
        a flag (e.g. ``--video-frames`` → ``--frames``) update here. The
        binary fails fast on unknown flags so a regression surfaces as a
        clean stderr message rather than silently bad output.
        """
        args: list[str] = [
            str(binary),
            "--diffusion-model",
            model_path,
            "-p",
            config.prompt,
            "-W",
            str(config.width),
            "-H",
            str(config.height),
            "--steps",
            str(config.steps),
            "--cfg-scale",
            f"{config.guidance:g}",
            "--seed",
            str(seed),
            "-o",
            str(output_path),
            "--video-frames",
            str(config.numFrames),
            "--fps",
            str(config.fps),
        ]
        if config.negativePrompt:
            args.extend(["--negative-prompt", config.negativePrompt])
        return args

    def _run_subprocess(
        self,
        *,
        args: list[str],
        config: VideoGenerationConfig,
        output_path: Path,
    ) -> bytes:
        """Spawn ``sd``, stream stdout into ``VIDEO_PROGRESS``, read result.

        Uses ``stderr=STDOUT`` so the same parser sees both info-level
        progress lines and any error chatter. Tail of the output is kept
        in ``last_lines`` so a non-zero exit can include the last few
        lines in the raised RuntimeError. Cancellation is cooperative:
        we poll ``VIDEO_PROGRESS.is_cancelled()`` per stdout line and
        terminate the child if a cancel comes in mid-run.
        """
        from backend_service.progress import VIDEO_PROGRESS

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_lines: list[str] = []
        try:
            stdout = proc.stdout
            if stdout is None:
                proc.wait()
                raise RuntimeError("sd.cpp subprocess produced no stdout.")
            for line in stdout:
                stripped = line.rstrip()
                last_lines.append(stripped)
                if len(last_lines) > _LAST_OUTPUT_LINES:
                    last_lines.pop(0)

                match = _STEP_RE.search(stripped)
                if match:
                    step = int(match.group(1))
                    total = int(match.group(2))
                    VIDEO_PROGRESS.set_step(step, total=total)

                if VIDEO_PROGRESS.is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError("sd.cpp generation cancelled by user.")

            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise

        if rc != 0:
            tail = "\n".join(last_lines[-20:])
            raise RuntimeError(
                f"sd.cpp exited with code {rc}.\n"
                f"Last output:\n{tail}"
            )

        if not output_path.exists():
            tail = "\n".join(last_lines[-10:])
            raise RuntimeError(
                f"sd.cpp completed but output file {output_path.name} is "
                f"missing. Last output:\n{tail}"
            )

        return output_path.read_bytes()
