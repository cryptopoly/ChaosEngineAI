"""Native Apple Silicon image runtime via the mflux package.

Only used for catalog variants that set ``engine="mflux"``. mflux runs image
models on MLX directly (Qwen-Image-2.1, FLUX.1, FLUX.2, Z-Image, ...), which
the PyTorch MPS backend cannot match on the large ones.

mflux pins an mlx / torch generation newer than the one the app runs, so it
lives in its own venv (``~/.chaosengine/mflux-venv``, built by
``scripts/install-mflux.sh``). On a Mac with enough memory the engine keeps the
model loaded between images in a warm worker process (``mflux_worker.py``, I1);
otherwise, when ``CHAOSENGINE_WARM_WORKERS=0`` or when the worker fails, it runs
one ``mflux-generate-*`` command per image. Either way the weights live outside
the backend, and a stuck run is killed without touching it.

The engine is a quiet no-op elsewhere: ``probe()`` reports unavailability and
the manager falls back to the diffusers pipeline.
"""

from __future__ import annotations

import platform
import re
import subprocess
import tempfile
import threading
import time
import uuid
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

_WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "mflux_worker.py"
_WORKER_LOG = Path.home() / ".chaosengine" / "logs" / "mflux-worker.log"
# Below this much memory the worker's resident model (text encoders included) costs more than the reload.
WARM_MIN_MEMORY_GB = 32.0
# Resident between images: Iris-3B about 15 GB, FLUX 3 Klein edit about 21 GB (v-speed.md). A new worker
# starts only with this much free memory; otherwise the image runs as a CLI process.
WARM_START_AVAILABLE_GB = 24.0
_WORKER_TIMEOUT_S = 3600.0
_WORKER_STALL_S = 600.0  # a big model's load sends no event until it is done

_LAST_OUTPUT_LINES = 60
# tqdm prints ``  25%|##5       | 1/4 [00:11<00:33, 11.23s/it]``.
_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_PEAK_RE = re.compile(r"Peak MLX memory:\s*([\d.]+)\s*GB", re.IGNORECASE)


# The release scripts/install-mflux.sh installs (``MFLUX_VERSION``). An older
# install lacks the newer commands and model names (FLUX 3 and Qwen-Image-2.1
# Turbo need 0.23, Iris-3B 0.24), so the setup status reports it as outdated.
MFLUX_REQUIRED_VERSION = "0.24.0"


def mflux_version_is_outdated(version: str | None) -> bool:
    if not version:
        return False

    def parts(text: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", text)[:3])

    try:
        return parts(version) < parts(MFLUX_REQUIRED_VERSION)
    except ValueError:
        return False


@dataclass(frozen=True)
class MfluxFamily:
    """How to run one model: its command, ``--model`` name and flag support."""

    command: str
    model: str
    # False for guidance-distilled models that take no guidance argument.
    takes_guidance: bool = True
    takes_negative_prompt: bool = False
    # Reference images. ``init_image``: the command takes one ``--image PATH [STRENGTH]``
    # start image (img2img). ``edit_command``: a sibling command that takes up to
    # ``MAX_EDIT_IMAGES`` ``--image-paths`` and edits or re-draws them from the prompt.
    init_image: bool = False
    edit_command: str | None = None


MAX_EDIT_IMAGES = 4


# Command names and flags read from mflux 0.24.0 (``--help`` of each command);
# re-check them when ``scripts/install-mflux.sh`` moves ``MFLUX_VERSION``.
_FAMILIES: dict[str, MfluxFamily] = {
    "black-forest-labs/FLUX.1-schnell": MfluxFamily("mflux-generate", "schnell", init_image=True),
    "black-forest-labs/FLUX.1-dev": MfluxFamily("mflux-generate", "dev", init_image=True),
    "black-forest-labs/FLUX.2-klein-4B": MfluxFamily("mflux-generate-flux2", "flux2-klein-4b", takes_guidance=False, init_image=True),
    "Qwen/Qwen-Image-2.1": MfluxFamily(
        "mflux-generate-qwen-2.1", "qwen-image-2.1", takes_guidance=False, takes_negative_prompt=True,
        init_image=True,
    ),
    # The 8-step distilled checkpoint; it samples without CFG (mflux 0.23).
    "Qwen/Qwen-Image-2.1-Turbo": MfluxFamily(
        "mflux-generate-qwen-2.1", "qwen-image-2.1-turbo", takes_guidance=False, init_image=True
    ),
    # FLUX 3 Image Klein (mflux 0.23): the distilled checkpoint takes no
    # guidance (16 steps by default); Base is guided (50 steps, 3.5).
    "black-forest-labs/flux-3-image-klein": MfluxFamily(
        "mflux-generate-flux3", "flux3-image-klein", takes_guidance=False, edit_command="mflux-generate-flux3-edit"
    ),
    "black-forest-labs/flux-3-image-klein-base": MfluxFamily(
        "mflux-generate-flux3", "flux3-image-klein-base", edit_command="mflux-generate-flux3-edit"
    ),
    # Iris-3B Turbo (mflux 0.24): a 3B model that generates pixels directly, 4
    # steps and no guidance.
    "speridlabs/iris-3b": MfluxFamily("mflux-generate-iris", "iris-3b", takes_guidance=False),
}


def mflux_family_for_repo(repo: str) -> MfluxFamily | None:
    return _FAMILIES.get(repo)


def reference_image_limit(repo: str) -> int:
    """How many reference images this model takes through mflux: 0 when none."""
    family = mflux_family_for_repo(repo)
    if family is None:
        return 0
    if family.edit_command:
        return MAX_EDIT_IMAGES
    return 1 if family.init_image else 0


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


_WORKER: Any = None
_WORKER_LOCK = threading.Lock()


def _worker_env() -> dict[str, str]:
    from backend_service.inference._utils import _isolated_child_env

    return dict(_isolated_child_env())


def warm_worker() -> Any:
    """The one warm mflux worker of this backend (created on first use, started on first image)."""
    global _WORKER
    from backend_service.warm_worker_client import WarmWorker

    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = WarmWorker(
                name="mflux",
                kind="image",
                command=lambda: [str(MFLUX_VENV_DIR / "bin" / "python"), str(_WORKER_SCRIPT)],
                env=_worker_env,
                log_path=_WORKER_LOG,
            )
        return _WORKER


def warm_worker_wanted() -> bool:
    from backend_service.warm_worker_client import can_start, total_memory_gb, warm_workers_enabled

    return (
        warm_workers_enabled()
        and total_memory_gb() >= WARM_MIN_MEMORY_GB
        and _WORKER_SCRIPT.exists()
        and can_start(_WORKER, WARM_START_AVAILABLE_GB)
    )


def release_warm_worker(reason: str = "unloaded") -> bool:
    """Stop the warm worker if it idles (never kills a running image)."""
    return _WORKER.stop_if_idle(reason) if _WORKER is not None else False


class WarmFallback(RuntimeError):
    """The warm worker could not make this image; the CLI path makes it instead."""


class MfluxImageEngine:
    runtime_label = "mflux (MLX native)"
    # Timing split of the last CLI run (C1), filled by ``_run``.
    _cli_timings: dict[str, Any] | None = None

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
        use_worker = warm_worker_wanted()
        for index in range(batch):
            seed = base_seed + index
            started = time.perf_counter()
            fallback: str | None = None
            timings: dict[str, Any] | None = None
            with tempfile.TemporaryDirectory(prefix="chaosengine-mflux-") as tmpdir:
                output_path = Path(tmpdir) / f"mflux-{seed}.png"
                args = self._build_command(family, config, seed=seed, output_path=output_path)
                peak_gb: float | None = None
                if use_worker:
                    try:
                        peak_gb, timings = self._run_warm(args, output_path)
                    except WarmFallback as exc:
                        use_worker = False  # the rest of the batch goes the CLI way too
                        fallback = str(exc)
                if timings is None:
                    peak_gb = self._run(args, output_path)
                    timings = dict(self._cli_timings or {}) or None
                data = output_path.read_bytes()
            note = f"MLX native ({family.model}, mflux {probe.get('version') or '?'})"
            if peak_gb is not None:
                note += f", peak {peak_gb:g} GB"
            if timings and timings.get("warm"):
                note += ", model kept loaded"
            if fallback:
                note += f"; the warm worker failed ({fallback}), so this image ran as a one-off process"
            images.append(
                GeneratedImage(
                    seed=seed,
                    bytes=data,
                    extension="png",
                    mimeType="image/png",
                    durationSeconds=round(max(0.1, time.perf_counter() - started), 1),
                    runtimeLabel=self.runtime_label,
                    runtimeNote=note,
                    timings=timings,
                )
            )

    def _run_warm(self, args: list[str], output_path: Path) -> tuple[float | None, dict[str, Any]]:
        """One image through the warm worker. Raises ``WarmFallback`` when the CLI should run it instead."""
        from backend_service.progress import (
            IMAGE_PROGRESS,
            PHASE_DECODING,
            PHASE_DIFFUSING,
            PHASE_ENCODING,
            PHASE_LOADING,
            PHASE_SAVING,
            GenerationCancelled,
        )
        from backend_service.warm_worker_client import WarmWorkerCancelled

        phases = {"load": (PHASE_LOADING, "Loading model"), "encode": (PHASE_ENCODING, "Encoding prompt"),
                  "denoise": (PHASE_DIFFUSING, "Diffusing"), "decode": (PHASE_DECODING, "Decoding image"),
                  "write": (PHASE_SAVING, "Saving image")}

        def on_event(event: dict[str, Any]) -> None:
            if event.get("event") == "stage" and event.get("stage") in phases:
                IMAGE_PROGRESS.set_phase(*phases[event["stage"]])
            elif event.get("event") == "step":
                IMAGE_PROGRESS.set_step(int(event.get("step") or 0), total=int(event.get("total") or 0) or None)

        worker = warm_worker()
        command = Path(args[0]).name
        try:
            reply = worker.request(
                "generate",
                timeout=_WORKER_TIMEOUT_S,
                stall_s=_WORKER_STALL_S,
                on_event=on_event,
                cancel_check=IMAGE_PROGRESS.is_cancelled,
                job=f"img-{uuid.uuid4().hex[:12]}",
                command=command,
                argv=args[1:],
            )
        except WarmWorkerCancelled as exc:
            raise GenerationCancelled("Image generation cancelled by user") from exc
        except Exception as exc:  # noqa: BLE001 - any worker failure falls back to the CLI
            if IMAGE_PROGRESS.is_cancelled():
                raise GenerationCancelled("Image generation cancelled by user") from exc
            worker.stop_if_idle("it failed a request")
            raise WarmFallback(f"{type(exc).__name__}: {exc}") from exc
        if not output_path.exists():
            worker.stop_if_idle("it wrote no image")
            raise WarmFallback("the worker wrote no image")
        worker.after_job(reply)
        timings = dict(reply.get("timings") or {})
        timings.setdefault("warm", bool(reply.get("warm")))
        peak = reply.get("peak_mlx_gb")
        return (float(peak) if peak is not None else None), timings

    def release(self, reason: str = "unloaded") -> bool:
        return release_warm_worker(reason)

    def _build_command(
        self,
        family: MfluxFamily,
        config: ImageGenerationConfig,
        *,
        seed: int,
        output_path: Path,
    ) -> list[str]:
        refs = list(config.referenceImagePaths or [])
        command = family.edit_command if refs and family.edit_command else family.command
        args = [
            str(MFLUX_VENV_DIR / "bin" / command),
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
        if refs:
            if family.edit_command:
                args += ["--image-paths", *refs]
            elif family.init_image:
                strength = [f"{config.imageStrength:g}"] if config.imageStrength is not None else []
                args += ["--image", refs[0], *strength]
            else:
                raise RuntimeError(f"{family.model} cannot take a reference image through mflux.")
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
        # C1: the CLI prints no phases, so the split is read off the tqdm bar: everything before the
        # bar is process start, imports, load and prompt encoding together ("loadSeconds"), the bar is
        # the denoise, and what follows is decode, save and exit ("decodeSeconds").
        started = time.perf_counter()
        bar_start: float | None = None
        bar_end: float | None = None
        self._cli_timings = None
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
                    now = time.perf_counter()
                    bar_start = bar_start if bar_start is not None else now
                    if int(step.group(1)) >= int(step.group(2)) and bar_end is None:
                        bar_end = now
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
        finished = time.perf_counter()
        bar_start = bar_start if bar_start is not None else finished
        bar_end = bar_end if bar_end is not None else bar_start
        self._cli_timings = {
            "loadSeconds": round(bar_start - started, 2),
            "encodeSeconds": None,  # inside loadSeconds on this path
            "denoiseSeconds": round(bar_end - bar_start, 2),
            "decodeSeconds": round(finished - bar_end, 2),
            "totalSeconds": round(finished - started, 2),
            "warm": False,
        }
        return peak_gb
