"""stable-diffusion.cpp image runtime (FU-008 image subset).

Wraps the staged ``sd`` binary from ``leejet/stable-diffusion.cpp`` (MIT)
as a subprocess engine for cross-platform image generation, mirroring
``SdCppVideoEngine`` and ``MfluxImageEngine``. Targets SD 1.x/2.x/XL,
FLUX.1, FLUX.2, Qwen Image, and Z-Image — the binary supports all of
these via GGUF transformer files.

Routing
-------
Apple Silicon: prefer mflux for FLUX (faster MLX-native), then sd.cpp
for non-FLUX GGUF, then diffusers MPS.

Linux/Windows + CUDA: prefer diffusers + bnb NF4 for FLUX, sd.cpp for
GGUF lanes when the user explicitly opts in.

The engine is selected when a catalog variant carries ``engine="sdcpp"``;
the manager's ``ImageRuntimeManager.generate`` checks ``config.runtime``
and dispatches accordingly.
"""

from __future__ import annotations

import io
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from backend_service.image_runtime import (
    GeneratedImage,
    ImageGenerationConfig,
    _resolve_base_seed,
)


# Same progress regex as the video engine — sd.cpp emits ``[INFO] step
# N/M`` lines on stdout regardless of which output type is active.
_STEP_RE = re.compile(r"(?:step\s+|\[)(\d+)\s*/\s*(\d+)")
_LAST_OUTPUT_LINES = 80
_RUNTIME_LABEL = "stable-diffusion.cpp"


# Repos sd.cpp's image lane supports natively. The Wan 2.1/2.2 video
# repos live in ``sdcpp_video_runtime._SUPPORTED_REPOS``; this module
# stays narrow to image-side families. Catalog variants with
# ``engine="sdcpp"`` must reference one of these repos *and* pin a
# ``ggufRepo`` + ``ggufFile`` so the binary has a single transformer
# file to load.
_SUPPORTED_REPOS: frozenset[str] = frozenset({
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-2512",
    "Tongyi-MAI/Z-Image",
    "Tongyi-MAI/Z-Image-Turbo",
})


def supported_repos() -> frozenset[str]:
    """Repo ids the sd.cpp image engine accepts."""
    return _SUPPORTED_REPOS


def _is_sdcpp_image_repo(repo: str | None) -> bool:
    if not repo:
        return False
    return repo in _SUPPORTED_REPOS


def _resolve_sd_binary() -> Path | None:
    """Resolve the staged ``sd`` binary path. Same lookup order as
    ``sdcpp_video_runtime._resolve_sd_binary`` — the image and video
    lanes share the same binary.
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


class SdCppImageEngine:
    """Subprocess wrapper around stable-diffusion.cpp for image GGUF.

    ``probe()`` reports binary presence + readiness. ``generate()``
    renders a single PNG via the staged binary, streaming ``step N/M``
    progress lines into ``IMAGE_PROGRESS`` so the desktop UI keeps a
    live denoise count. Output is read back as PNG bytes for the
    standard ``GeneratedImage`` contract.
    """

    runtime_label = _RUNTIME_LABEL

    def __init__(self) -> None:
        self._loaded_repo: str | None = None

    # ------------------------------------------------------------------
    # Probe + lifecycle
    # ------------------------------------------------------------------

    def probe(self) -> dict[str, Any]:
        binary = _resolve_sd_binary()
        if binary is None:
            return {
                "available": False,
                "reason": (
                    "stable-diffusion.cpp binary not staged. Run "
                    "``./scripts/build-sdcpp.sh`` (or set "
                    "CHAOSENGINE_SDCPP_BIN) to build and install."
                ),
            }
        return {
            "available": True,
            "reason": None,
            "binary": str(binary),
            "device": "mps" if platform.system() == "Darwin" else "cuda",
        }

    def preload(self, repo: str) -> dict[str, Any]:
        if not _is_sdcpp_image_repo(repo):
            raise RuntimeError(
                f"sd.cpp image lane does not support {repo}. "
                f"Supported: {sorted(_SUPPORTED_REPOS)}"
            )
        self._loaded_repo = repo
        return self.probe()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        if repo is None or repo == self._loaded_repo:
            self._loaded_repo = None
        return self.probe()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        binary = _resolve_sd_binary()
        if binary is None:
            raise RuntimeError(
                "stable-diffusion.cpp binary not staged. "
                "Run ``./scripts/build-sdcpp.sh`` first."
            )
        if not _is_sdcpp_image_repo(config.repo):
            raise RuntimeError(
                f"sd.cpp image lane does not support {config.repo}. "
                f"Supported: {sorted(_SUPPORTED_REPOS)}"
            )
        if not config.ggufFile:
            raise RuntimeError(
                "sd.cpp image generate requires a GGUF variant. Pick a "
                "catalog entry that pins ``ggufRepo`` + ``ggufFile`` "
                "(e.g. FLUX.1-dev · GGUF Q4_K_M)."
            )

        base_seed = _resolve_base_seed(config.seed)
        batch = max(1, int(config.batchSize or 1))
        out_images: list[GeneratedImage] = []
        started = time.perf_counter()

        # sd.cpp renders one image per invocation. Loop the batch — same
        # pattern the diffusers engine uses when it can't batch on a
        # given pipeline. Each iteration gets its own seed so the user
        # sees a real variation set rather than four copies.
        for index in range(batch):
            seed = base_seed + index
            with tempfile.TemporaryDirectory(prefix="chaosengine-sdcpp-img-") as tmpdir:
                output_path = Path(tmpdir) / f"sdcpp-{seed}.png"
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

            elapsed = max(0.1, time.perf_counter() - started)
            out_images.append(
                GeneratedImage(
                    seed=seed,
                    bytes=output_bytes,
                    extension="png",
                    mimeType="image/png",
                    durationSeconds=round(elapsed, 1),
                    runtimeLabel=_RUNTIME_LABEL,
                    runtimeNote=(
                        f"Generated via sd.cpp subprocess "
                        f"({Path(model_path).name})."
                    ),
                )
            )
            # Reset the timer so the next image's durationSeconds
            # measures its own wall-time, not cumulative.
            started = time.perf_counter()

        return out_images

    # ------------------------------------------------------------------
    # CLI builders + subprocess plumbing
    # ------------------------------------------------------------------

    def _resolve_gguf_path(self, config: ImageGenerationConfig) -> str:
        """Materialise the GGUF transformer file from HF cache (or
        download on first use). The catalog variant pins
        ``ggufRepo`` + ``ggufFile``.
        """
        if not config.ggufFile or not config.ggufRepo:
            raise RuntimeError(
                "GGUF transformer required for sd.cpp image. Catalog variant "
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
        config: ImageGenerationConfig,
        model_path: str,
        output_path: Path,
        seed: int,
    ) -> list[str]:
        """Map an ``ImageGenerationConfig`` onto sd.cpp's CLI flags.

        Mirrors the video CLI builder shape but drops video-specific
        flags (``--video-frames``, ``--fps``). Output is PNG; sd.cpp
        infers the format from the ``-o`` file extension.
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
        ]
        if config.negativePrompt:
            args.extend(["--negative-prompt", config.negativePrompt])
        return args

    def _run_subprocess(
        self,
        *,
        args: list[str],
        config: ImageGenerationConfig,
        output_path: Path,
    ) -> bytes:
        """Spawn ``sd``, stream stdout into ``IMAGE_PROGRESS``, read result."""
        from backend_service.progress import IMAGE_PROGRESS

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
                    IMAGE_PROGRESS.set_step(step, total=total)

                if IMAGE_PROGRESS.is_cancelled():
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
