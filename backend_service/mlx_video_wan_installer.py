"""mlx-video Wan installer (FU-025).

End-to-end orchestration that downloads a raw Wan-AI checkpoint from
Hugging Face and runs ``mlx_video.models.wan_2.convert`` so the
``mlx_video_runtime`` engine can route the repo through the native MLX
subprocess. This is the bridge between the helper module
(``mlx_video_wan_convert``) and the Setup-page UX — same pattern as
``longlive_installer`` but Apple-Silicon-only and considerably smaller
in scope.

Invocable two ways:
    * In-process: ``from backend_service.mlx_video_wan_installer import install``
    * As a module: ``python -m backend_service.mlx_video_wan_installer
      --repo Wan-AI/Wan2.1-T2V-1.3B`` (used by the FastAPI install
      endpoint so the long-running convert stays out of the sidecar).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from backend_service.mlx_video_wan_convert import (
    SUPPORTED_RAW_REPOS,
    is_mlx_video_available,
    is_supported_raw_repo,
    output_dir_for,
    slug_for,
    status_for,
)


# Where raw HF Wan checkpoints land before conversion. Kept under
# ``~/.chaosengine/mlx-video-wan-raw/`` so the converted artifacts and
# their source weights live under the same parent (easier for users to
# audit / clean up). Override with ``CHAOSENGINE_MLX_VIDEO_WAN_RAW_DIR``.
def _resolve_raw_root() -> Path:
    override = os.environ.get("CHAOSENGINE_MLX_VIDEO_WAN_RAW_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".chaosengine" / "mlx-video-wan-raw"


RAW_ROOT: Path = _resolve_raw_root()


# Ordered phases. The async job worker walks this list to drive a
# percent counter; the in-process / CLI path uses it for log labels.
INSTALL_PHASES: tuple[str, ...] = (
    "preflight",       # check Apple Silicon + mlx-video installed + repo supported
    "download-raw",    # snapshot raw Wan repo from HF (largest phase)
    "convert",         # python -m mlx_video.models.wan_2.convert
    "verify",          # status_for() must report converted=True
)


# Per-repo approximate size in GB (raw weights + headroom). Used by the
# preflight to surface a "free disk needed" hint, not enforced.
_APPROX_RAW_SIZE_GB: dict[str, float] = {
    "Wan-AI/Wan2.1-T2V-1.3B": 3.5,
    "Wan-AI/Wan2.1-T2V-14B": 28.0,
    "Wan-AI/Wan2.2-TI2V-5B": 24.0,
    "Wan-AI/Wan2.2-T2V-A14B": 67.0,
    "Wan-AI/Wan2.2-I2V-A14B": 67.0,
}


class WanInstallError(RuntimeError):
    """Raised when the installer cannot proceed (wrong platform, missing
    package, unknown repo, download/convert failure)."""


def raw_dir_for(repo: str) -> Path:
    """Local path where raw HF weights are downloaded for ``repo``."""
    return RAW_ROOT / slug_for(repo)


def approx_raw_size_gb(repo: str) -> float | None:
    return _APPROX_RAW_SIZE_GB.get(repo)


def _noop_progress(_event: dict[str, object]) -> None:
    """Default progress sink. The async job worker overrides with one
    that updates ``_WAN_INSTALL_JOB`` shared state."""


def _emit(
    progress: Callable[[dict[str, object]], None],
    *,
    phase: str,
    message: str,
    ok: bool = True,
    output: str | None = None,
) -> None:
    payload: dict[str, object] = {"phase": phase, "ok": ok, "message": message}
    if output is not None:
        payload["output"] = output
    progress(payload)


def _preflight(repo: str) -> None:
    """Validate platform + package + repo before starting the heavy
    download. Raises ``WanInstallError`` with an actionable message
    otherwise."""
    system = platform.system()
    if system != "Darwin":
        raise WanInstallError(
            "mlx-video Wan runtime is Apple Silicon only. "
            f"Detected platform: {system}."
        )
    if platform.machine() not in {"arm64", "aarch64"}:
        raise WanInstallError(
            "mlx-video Wan runtime requires an arm64 / aarch64 Mac. "
            f"Detected machine: {platform.machine()}."
        )
    if not is_mlx_video_available():
        raise WanInstallError(
            "mlx-video is not installed. From the project root, run "
            '``pip install -e ".[mlx-video]"`` and retry.'
        )
    if not is_supported_raw_repo(repo):
        raise WanInstallError(
            f"Unsupported Wan repo {repo!r}. "
            f"Supported: {sorted(SUPPORTED_RAW_REPOS)}"
        )


def _download_raw(
    repo: str,
    raw_dir: Path,
    logger: Callable[[str], None],
) -> None:
    """Snapshot the raw Wan repo to ``raw_dir`` via huggingface_hub."""
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    logger(f"Downloading {repo} → {raw_dir}")
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise WanInstallError(
            f"huggingface_hub is required to download raw Wan weights: {exc}. "
            "Install it via ``pip install huggingface-hub``."
        ) from exc
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(raw_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001 — surface any HF error as install error
        raise WanInstallError(
            f"Failed to download {repo}: {type(exc).__name__}: {exc}"
        ) from exc


def _run_convert(
    raw_dir: Path,
    repo: str,
    *,
    dtype: str,
    quantize: bool,
    bits: int,
    group_size: int,
    timeout_seconds: int,
    python_executable: str,
    logger: Callable[[str], None],
) -> None:
    """Spawn ``python -m mlx_video.models.wan_2.convert`` and stream its
    stdout into ``logger``. Bypasses ``mlx_video_wan_convert.run_convert``
    so we can stream output line-by-line for the progress UI rather than
    capturing the whole thing at the end of the run."""
    out = output_dir_for(repo)
    out.parent.mkdir(parents=True, exist_ok=True)

    args = [
        python_executable,
        "-m", "mlx_video.models.wan_2.convert",
        "--checkpoint-dir", str(raw_dir),
        "--output-dir", str(out),
        "--dtype", dtype,
        "--model-version", "auto",
    ]
    if quantize:
        args.extend([
            "--quantize",
            "--bits", str(bits),
            "--group-size", str(group_size),
        ])

    logger(f"$ {' '.join(args)}")
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise WanInstallError(
            f"Failed to spawn convert subprocess: {exc}. "
            "Verify the Python interpreter path is correct."
        ) from exc

    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            logger(stripped)

    rc = process.wait(timeout=timeout_seconds)
    if rc != 0:
        raise WanInstallError(
            f"Convert subprocess exited with code {rc}. "
            "Last lines of output appear in the install log above."
        )


def install(
    repo: str,
    *,
    dtype: str = "bfloat16",
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
    timeout_seconds: int = 3600,
    keep_raw: bool = True,
    logger: Callable[[str], None] = print,
    progress: Callable[[dict[str, object]], None] = _noop_progress,
    python_executable: str | None = None,
) -> None:
    """Run the full Wan install: preflight → download raw → convert → verify.

    Raises ``WanInstallError`` on any failure. ``progress`` receives a
    structured event per phase so the FastAPI job worker can surface
    progress to the UI; the CLI path uses the no-op sink.

    ``keep_raw=False`` deletes the raw HF download after successful
    conversion to free disk space (Wan2.2 A14B raw is ~67 GB; after
    convert the raw weights aren't referenced again until a future
    re-conversion).
    """
    py = python_executable or sys.executable

    _emit(progress, phase="preflight", message=f"Checking platform + package for {repo}")
    _preflight(repo)

    raw_dir = raw_dir_for(repo)
    _emit(
        progress,
        phase="download-raw",
        message=(
            f"Downloading raw {repo} (~{approx_raw_size_gb(repo) or '?'} GB) → {raw_dir}"
        ),
    )
    _download_raw(repo, raw_dir, logger)

    _emit(
        progress,
        phase="convert",
        message=f"Converting to MLX format → {output_dir_for(repo)}",
    )
    _run_convert(
        raw_dir,
        repo,
        dtype=dtype,
        quantize=quantize,
        bits=bits,
        group_size=group_size,
        timeout_seconds=timeout_seconds,
        python_executable=py,
        logger=logger,
    )

    _emit(progress, phase="verify", message="Verifying converted output")
    status = status_for(repo)
    if not status.converted:
        raise WanInstallError(
            f"Convert finished but output dir is incomplete: "
            f"{status.note or 'unknown reason'}"
        )

    if not keep_raw:
        logger(f"Cleaning raw download at {raw_dir}")
        shutil.rmtree(raw_dir, ignore_errors=True)

    logger(
        f"Wan install complete: {repo} converted at {status.outputDir}"
    )


# ----------------------------------------------------------------------
# CLI entrypoint — used by the FastAPI install endpoint to spawn this
# module as a subprocess so a long-running convert stays out of the
# sidecar process. Mirror longlive_installer's pattern.
# ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install an mlx-video Wan model: download raw HF weights "
            "and convert to MLX format."
        )
    )
    parser.add_argument(
        "--repo",
        required=True,
        help=f"Raw Wan-AI repo id. Supported: {sorted(SUPPORTED_RAW_REPOS)}",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--quantize", action="store_true", help="Quantize transformer weights")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument(
        "--timeout-seconds", type=int, default=3600,
        help="Max wall-clock for the convert subprocess (default 1 hour).",
    )
    parser.add_argument(
        "--cleanup-raw", action="store_true",
        help="Delete raw HF download after successful convert.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        install(
            args.repo,
            dtype=args.dtype,
            quantize=args.quantize,
            bits=args.bits,
            group_size=args.group_size,
            timeout_seconds=args.timeout_seconds,
            keep_raw=not args.cleanup_raw,
        )
    except WanInstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
