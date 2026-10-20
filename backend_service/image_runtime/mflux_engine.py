"""Native Apple Silicon image runtime via the mflux package.

Only used for catalog variants that set ``engine="mflux"``. mflux runs image
models on MLX directly (Qwen-Image-2.1, FLUX.1, FLUX.2, Z-Image, ...), which
the PyTorch MPS backend cannot match on the large ones.

mflux pins an mlx / torch generation newer than the one the app runs, so it
lives in its own venv (``~/.chaosengine/mflux-venv``, built by
``scripts/install-mflux.sh``). The engine runs one ``mflux-generate-*`` command
per image: the weights are freed when the command exits and a stuck run can be
killed without touching the backend.

The engine is a quiet no-op elsewhere: ``probe()`` reports unavailability and
the manager falls back to the diffusers pipeline.
"""

from __future__ import annotations

import platform
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend_service.image_runtime.placeholder_engine import _resolve_base_seed
from backend_service.image_runtime.types import (
    GeneratedImage,
    ImageGenerationConfig,
)

MFLUX_VENV_DIR = Path.home() / ".chaosengine" / "mflux-venv"
MFLUX_VERSION_FILE = Path.home() / ".chaosengine" / "bin" / "mflux.version"

_LAST_OUTPUT_LINES = 60
# tqdm prints ``  25%|##5       | 1/4 [00:11<00:33, 11.23s/it]``.
_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_PEAK_RE = re.compile(r"Peak MLX memory:\s*([\d.]+)\s*GB", re.IGNORECASE)


@dataclass(frozen=True)
class MfluxFamily:
    """How to run one model: its command, ``--model`` name and flag support."""

    command: str
    model: str
    # False for guidance-distilled models that take no guidance argument.
    takes_guidance: bool = True
    takes_negative_prompt: bool = False


# Command names and flags read from mflux 0.20.0 (``--help`` of each command);
# re-check them when ``scripts/install-mflux.sh`` moves ``MFLUX_VERSION``.
_FAMILIES: dict[str, MfluxFamily] = {
    "black-forest-labs/FLUX.1-schnell": MfluxFamily("mflux-generate", "schnell"),
    "black-forest-labs/FLUX.1-dev": MfluxFamily("mflux-generate", "dev"),
    "black-forest-labs/FLUX.2-klein-4B": MfluxFamily("mflux-generate-flux2", "flux2-klein-4b", takes_guidance=False),
    "Qwen/Qwen-Image-2.1": MfluxFamily(
        "mflux-generate-qwen-2.1", "qwen-image-2.1", takes_guidance=False, takes_negative_prompt=True
    ),
}


def mflux_family_for_repo(repo: str) -> MfluxFamily | None:
    return _FAMILIES.get(repo)


def _mflux_name_for_repo(repo: str) -> str | None:
    """The ``--model`` name mflux uses for ``repo``, or None when unsupported."""
    family = mflux_family_for_repo(repo)
    return family.model if family is not None else None


def supported_repos() -> frozenset[str]:
    return frozenset(_FAMILIES)


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def is_installed() -> bool:
    """Cheap check: the version file is written only after the install verified."""
    return MFLUX_VERSION_FILE.exists() and (MFLUX_VENV_DIR / "bin" / "mflux-generate").exists()


def read_version() -> str | None:
    try:
        lines = MFLUX_VERSION_FILE.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    return lines[0].strip() if lines and lines[0].strip() else None


class MfluxImageEngine:
    runtime_label = "mflux (MLX native)"

    def probe(self) -> dict[str, Any]:
        if not _apple_silicon():
            return {"available": False, "reason": "mflux runs on Apple Silicon only."}
        if not is_installed():
            return {
                "available": False,
                "reason": (
                    "mflux is not installed. Install the MLX image engine from the "
                    "Image Studio to run this model natively."
                ),
            }
        return {"available": True, "reason": None, "version": read_version()}

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        probe = self.probe()
        if not probe["available"]:
            raise RuntimeError(probe["reason"] or "mflux unavailable")
        family = mflux_family_for_repo(config.repo)
        if family is None:
            raise RuntimeError(f"mflux does not support {config.repo} yet. Supported: {', '.join(sorted(_FAMILIES))}.")

        from backend_service.progress import IMAGE_PROGRESS, PHASE_LOADING

        base_seed = _resolve_base_seed(config.seed)
        batch = max(1, int(config.batchSize or 1))
        images: list[GeneratedImage] = []
        IMAGE_PROGRESS.begin(
            run_label=f"{config.modelName} · {config.width}x{config.height} · mflux",
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Loading {config.modelName}",
        )
        try:
            self._generate_batch(family, config, probe, base_seed, batch, images)
        finally:
            IMAGE_PROGRESS.finish()
        return images

    def _generate_batch(
        self,
        family: MfluxFamily,
        config: ImageGenerationConfig,
        probe: dict[str, Any],
        base_seed: int,
        batch: int,
        images: list[GeneratedImage],
    ) -> None:
        for index in range(batch):
            seed = base_seed + index
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="chaosengine-mflux-") as tmpdir:
                output_path = Path(tmpdir) / f"mflux-{seed}.png"
                args = self._build_command(family, config, seed=seed, output_path=output_path)
                peak_gb = self._run(args, output_path)
                data = output_path.read_bytes()
            note = f"MLX native ({family.model}, mflux {probe.get('version') or '?'})"
            if peak_gb is not None:
                note += f", peak {peak_gb:g} GB"
            images.append(
                GeneratedImage(
                    seed=seed,
                    bytes=data,
                    extension="png",
                    mimeType="image/png",
                    durationSeconds=round(max(0.1, time.perf_counter() - started), 1),
                    runtimeLabel=self.runtime_label,
                    runtimeNote=note,
                )
            )

    def _build_command(
        self,
        family: MfluxFamily,
        config: ImageGenerationConfig,
        *,
        seed: int,
        output_path: Path,
    ) -> list[str]:
        args = [
            str(MFLUX_VENV_DIR / "bin" / family.command),
            "--model", family.model,
            "--prompt", config.prompt,
            "--steps", str(config.steps),
            "--width", str(config.width),
            "--height", str(config.height),
            "--seed", str(seed),
            "--output", str(output_path),
            "--no-metadata",
        ]
        if family.takes_guidance:
            args += ["--guidance", f"{config.guidance:g}"]
        if family.takes_negative_prompt and config.negativePrompt:
            args += ["--negative-prompt", config.negativePrompt]
        if config.mfluxQuantize:
            args += ["--quantize", str(int(config.mfluxQuantize))]
        return args

    def _run(self, args: list[str], output_path: Path) -> float | None:
        """Run one command, stream its progress, return the peak memory (GB) it reports."""
        from backend_service.inference._utils import _isolated_child_env
        from backend_service.progress import IMAGE_PROGRESS, PHASE_DIFFUSING, GenerationCancelled

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # The packaged app exports PYTHONHOME / PYTHONPATH for its embedded
            # runtime; the venv's own interpreter must not see them.
            env=_isolated_child_env(),
        )
        tail: list[str] = []
        peak_gb: float | None = None
        try:
            assert proc.stdout is not None
            # tqdm redraws with carriage returns; text mode turns those into lines.
            for line in proc.stdout:
                stripped = line.rstrip()
                if not stripped:
                    continue
                tail.append(stripped)
                del tail[:-_LAST_OUTPUT_LINES]
                step = _STEP_RE.search(stripped)
                if step:
                    if int(step.group(1)) <= 1:
                        IMAGE_PROGRESS.set_phase(PHASE_DIFFUSING, message="Diffusing")
                    IMAGE_PROGRESS.set_step(int(step.group(1)), total=int(step.group(2)))
                peak = _PEAK_RE.search(stripped)
                if peak:
                    peak_gb = float(peak.group(1))
                if IMAGE_PROGRESS.is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise GenerationCancelled("Image generation cancelled by user")
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
        if code != 0:
            raise RuntimeError(f"mflux exited with code {code}.\nLast output:\n" + "\n".join(tail[-15:]))
        if not output_path.exists():
            raise RuntimeError("mflux finished without writing an image.\nLast output:\n" + "\n".join(tail[-10:]))
        return peak_gb
