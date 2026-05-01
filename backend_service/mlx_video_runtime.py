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

from dataclasses import replace
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
    _resolve_video_seed,
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


# Maps repo prefix → mlx-video MODULE path (NOT the console-script alias).
# Blaizzy/mlx-video declares ``mlx_video.ltx_2.generate`` and
# ``mlx_video.wan_2.generate`` as console scripts in ``pyproject.toml``,
# but those are entry-point aliases — ``python -m mlx_video.ltx_2.generate``
# fails with ``ModuleNotFoundError: No module named 'mlx_video.ltx_2'``
# because the actual code lives at ``mlx_video.models.ltx_2.generate``.
# We invoke via ``python -m`` to avoid PATH dependencies (the console
# script binary may not land on PATH inside the embedded runtime), so
# this dict points at the real module path.
_REPO_ENTRY_POINTS: dict[str, str] = {
    "prince-canuma/LTX-2": "mlx_video.models.ltx_2.generate",
}


_LTX2_SPATIAL_UPSCALER_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "2": (
        ("prince-canuma/LTX-2-dev", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ("Lightricks/LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    ),
    "2.3": (
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
    ),
}
_LTX2_SHARED_TEXT_ENCODER_CANDIDATES: tuple[str, ...] = (
    "prince-canuma/LTX-2-distilled",
    "Lightricks/LTX-2",
)
_LTX2_TEXT_COMPONENTS: tuple[str, ...] = ("text_encoder", "tokenizer")
_LTX2_DISTILLED_STAGE_1_STEPS = 8
_LTX2_DISTILLED_STAGE_2_STEPS = 3


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


def _resolve_pipeline_flag(repo: str) -> str:
    """Pick mlx-video's ``--pipeline`` value from the repo name.

    mlx-video's LTX-2 generate supports four pipeline modes:
        distilled        — fastest, single-stage (default)
        dev              — higher quality, single-stage with CFG
        dev-two-stage    — two-stage refinement
        dev-two-stage-hq — highest quality, slowest

    We map ``LTX-2-distilled`` / ``LTX-2.3-distilled`` repos to
    ``distilled`` and ``LTX-2-dev`` / ``LTX-2.3-dev`` to ``dev``.
    Two-stage modes are not surfaced today — pre-converted weights
    can drive both, but exposing the toggle requires an extra UI
    affordance + per-pipeline timing budget.
    """
    repo_lower = repo.lower()
    if repo_lower.endswith("-dev"):
        return "dev"
    return "distilled"


def _ltx2_generation_needs_spatial_upscaler(repo: str) -> bool:
    """Return True when the selected mlx-video pipeline is two-stage.

    mlx-video's distilled LTX-2 path always upsamples the half-resolution
    stage-1 latents before stage 2. Some pre-converted repos omit the root
    upscaler file, so ChaosEngineAI resolves the canonical upscaler and
    passes it explicitly rather than letting the subprocess fail after the
    first stage has already run.
    """
    return _resolve_pipeline_flag(repo) in {
        "distilled",
        "dev-two-stage",
        "dev-two-stage-hq",
    }


def _ltx2_model_version(repo: str) -> str:
    return "2.3" if "ltx-2.3" in repo.lower() else "2"


def _ltx2_effective_steps(repo: str, requested_steps: int) -> int:
    if _resolve_pipeline_flag(repo) == "distilled":
        return _LTX2_DISTILLED_STAGE_1_STEPS + _LTX2_DISTILLED_STAGE_2_STEPS
    return requested_steps


def _ltx2_effective_guidance(repo: str, requested_guidance: float) -> float:
    if _resolve_pipeline_flag(repo) == "distilled":
        return 1.0
    return requested_guidance


def _ltx2_runtime_note(repo: str) -> str:
    pipeline = _resolve_pipeline_flag(repo)
    if pipeline == "distilled":
        return (
            "mlx-video subprocess (MLX native, distilled pipeline: "
            "fixed 8+3 denoise passes, CFG disabled)"
        )
    return f"mlx-video subprocess (MLX native, {pipeline} pipeline)"


def _ltx2_spatial_upscaler_candidates(repo: str) -> tuple[tuple[str, str], ...]:
    version = _ltx2_model_version(repo)
    filename_candidates = tuple(dict.fromkeys(
        filename for _, filename in _LTX2_SPATIAL_UPSCALER_CANDIDATES[version]
    ))
    candidates = (
        tuple((repo, filename) for filename in filename_candidates)
        + _LTX2_SPATIAL_UPSCALER_CANDIDATES[version]
    )
    return tuple(dict.fromkeys(candidates))


def _download_hf_file(repo_id: str, filename: str, *, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only,
            resume_download=True,
        )
    )


def _resolve_ltx2_spatial_upscaler(
    repo: str,
    *,
    allow_download: bool,
) -> Path | None:
    """Find the required LTX-2 spatial upscaler for two-stage generation."""
    if not _ltx2_generation_needs_spatial_upscaler(repo):
        return None

    candidates = _ltx2_spatial_upscaler_candidates(repo)
    errors: list[str] = []

    for candidate_repo, filename in candidates:
        try:
            return _download_hf_file(
                candidate_repo,
                filename,
                local_files_only=True,
            )
        except Exception as exc:
            errors.append(f"{candidate_repo}/{filename}: {type(exc).__name__}: {exc}")

    if not allow_download:
        return None

    for candidate_repo, filename in candidates:
        try:
            return _download_hf_file(
                candidate_repo,
                filename,
                local_files_only=False,
            )
        except Exception as exc:
            errors.append(f"{candidate_repo}/{filename}: {type(exc).__name__}: {exc}")

    checked = ", ".join(f"{repo_id}/{filename}" for repo_id, filename in candidates)
    raise RuntimeError(
        "LTX-2 distilled generation requires a spatial upscaler, but none "
        f"could be found or downloaded. Checked: {checked}. Last errors: "
        f"{'; '.join(errors[-3:])}"
    )


def _resolve_local_snapshot(repo_or_path: str) -> Path | None:
    candidate = Path(repo_or_path)
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        return Path(snapshot_download(repo_id=repo_or_path, local_files_only=True))
    except Exception:
        return None


def _missing_ltx2_text_components(root: Path) -> list[str]:
    missing: list[str] = []
    checks = {
        "text_encoder": (
            root / "text_encoder" / "config.json",
            root / "text_encoder" / "model.safetensors.index.json",
        ),
        "tokenizer": (
            root / "tokenizer" / "tokenizer.json",
            root / "tokenizer" / "tokenizer.model",
        ),
    }
    for component, required_paths in checks.items():
        if not (root / component).is_dir():
            missing.append(component)
            continue
        if not all(path.exists() for path in required_paths):
            missing.append(component)
    return missing


def _resolve_ltx2_text_component_source(repo: str) -> Path:
    for candidate_repo in tuple(dict.fromkeys((repo, *_LTX2_SHARED_TEXT_ENCODER_CANDIDATES))):
        snapshot = _resolve_local_snapshot(candidate_repo)
        if snapshot is not None and not _missing_ltx2_text_components(snapshot):
            return snapshot
    checked = ", ".join(_LTX2_SHARED_TEXT_ENCODER_CANDIDATES)
    raise RuntimeError(
        "LTX-2.3 MLX generation needs shared text_encoder and tokenizer "
        f"components, but none were found locally. Download {checked} or "
        "resume this model download, then try again."
    )


def _prepare_ltx2_model_path(repo: str, workspace: Path) -> Path:
    model_path = _resolve_local_snapshot(repo)
    if model_path is None:
        raise RuntimeError(
            f"LTX-2 MLX model snapshot is not available locally for {repo}. "
            "Download the model before generating."
        )

    missing = _missing_ltx2_text_components(model_path)
    if not missing:
        return model_path

    text_source = _resolve_ltx2_text_component_source(repo)
    overlay = workspace / "model-overlay"
    shutil.rmtree(overlay, ignore_errors=True)
    overlay.mkdir(parents=True, exist_ok=True)

    missing_set = set(missing)
    for entry in model_path.iterdir():
        if entry.name in missing_set:
            continue
        (overlay / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    for component in _LTX2_TEXT_COMPONENTS:
        target = overlay / component
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        source = text_source / component
        target.symlink_to(source, target_is_directory=True)
    return overlay


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
            if _ltx2_generation_needs_spatial_upscaler(config.repo):
                on_progress("loading", "Checking LTX-2 spatial upscaler", 0.03)

        workspace = Path(tempfile.mkdtemp(prefix="mlx-video-run-"))
        try:
            output_path = workspace / "out.mp4"
            resolved_seed = _resolve_video_seed(config.seed)
            run_config = replace(config, seed=resolved_seed)
            cmd = self._build_cmd(run_config, output_path, resolve_aux_files=True)
            start = time.monotonic()
            self._launch(cmd, workspace, on_progress)
            elapsed = time.monotonic() - start

            if not output_path.exists():
                raise RuntimeError(
                    f"mlx-video finished but no mp4 was produced at "
                    f"{output_path}. Check the subprocess log above."
                )
            data = output_path.read_bytes()
            return GeneratedVideo(
                seed=resolved_seed,
                bytes=data,
                extension="mp4",
                mimeType="video/mp4",
                durationSeconds=round(elapsed, 2),
                frameCount=config.numFrames,
                fps=config.fps,
                width=config.width,
                height=config.height,
                runtimeLabel=self.runtime_label,
                runtimeNote=_ltx2_runtime_note(config.repo),
                effectiveSteps=_ltx2_effective_steps(config.repo, config.steps),
                effectiveGuidance=_ltx2_effective_guidance(config.repo, config.guidance),
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # ---------- internals ----------

    def _build_cmd(
        self,
        config: VideoGenerationConfig,
        output_path: Path,
        *,
        resolve_aux_files: bool = False,
    ) -> list[str]:
        """Compose the ``python -m mlx_video.<entry> --...`` invocation.

        Split out so tests can assert the CLI shape without spawning a
        real subprocess. Flags mirror Blaizzy/mlx-video's
        ``mlx_video.models.ltx_2.generate`` argparse surface — note the
        names differ from diffusers conventions: ``--model-repo`` (not
        ``--model``), ``--cfg-scale`` (not ``--guidance``),
        ``--output-path`` (not ``--output``).
        """
        entry = _resolve_entry_point(config.repo)
        python = _resolve_video_python()
        pipeline_flag = _resolve_pipeline_flag(config.repo)
        model_repo_arg = config.repo
        if resolve_aux_files and "ltx-2.3" in config.repo.lower():
            model_repo_arg = str(_prepare_ltx2_model_path(config.repo, output_path.parent))
        cmd = [
            python,
            "-m",
            entry,
            "--model-repo",
            model_repo_arg,
            "--pipeline",
            pipeline_flag,
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
            "--cfg-scale",
            str(config.guidance),
            "--output-path",
            str(output_path),
        ]
        if config.negativePrompt:
            cmd.extend(["--negative-prompt", config.negativePrompt])
        if config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])
        if resolve_aux_files:
            spatial_upscaler = _resolve_ltx2_spatial_upscaler(
                config.repo,
                allow_download=True,
            )
            if spatial_upscaler is not None:
                cmd.extend(["--spatial-upscaler", str(spatial_upscaler)])
        # STG (Spatial-Temporal Guidance) is mlx-video's built-in quality
        # lever — perturbs final transformer blocks during sampling to
        # reduce object breakup / chroma drift. Default 1.0 mirrors the
        # upstream README's quality recommendation. This closes the FU-013
        # gap for the mlx-video path (still pending for the diffusers
        # LTX path on CUDA / non-Apple-Silicon hosts).
        cmd.extend(["--stg-scale", "1.0"])
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
