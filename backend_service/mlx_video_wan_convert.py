"""mlx-video Wan2.1/2.2 weight conversion (FU-025).

Wraps ``mlx_video.models.wan_2.convert.convert_wan_checkpoint`` (and its
``python -m`` CLI entrypoint) so ChaosEngineAI can promote raw HF Wan
repos to mlx-video's native MLX format. Closes FU-009 Wan branch.

UPSTREAM
--------
Blaizzy/mlx-video ships ``mlx_video/models/wan_2/convert.py`` with both
a ``convert_wan_checkpoint(checkpoint_dir, output_dir, ...)`` function
and a CLI module entry. This wrapper invokes the CLI as a subprocess so
the long-running conversion (5-30 min depending on model size) doesn't
block the FastAPI worker thread. The CLI flags we forward:

* ``--checkpoint-dir`` — raw HF Wan repo path
* ``--output-dir`` — converted MLX dir
* ``--dtype {float16, bfloat16, float32}``
* ``--model-version {2.1, 2.2, auto}``
* ``--quantize --bits {4,8} --group-size {32,64,128}`` (optional)

LAYOUT
------
Converted weights land under
``~/.chaosengine/mlx-video-wan/<repo-slug>/`` where ``<repo-slug>`` is
the HF repo id with ``/`` replaced by ``__`` so the directory is a
single path component. Each output directory contains:

* ``models_t5_umt5-xxl-enc-bf16.safetensors`` (text encoder)
* ``Wan2.1_VAE.safetensors`` (VAE)
* ``transformer*.safetensors`` (Wan2.1 single transformer) OR
  ``high_noise_model/`` + ``low_noise_model/`` subdirs (Wan2.2 MoE)
* ``config.json`` (model metadata)

SCOPE
-----
This module ships the CONVERSION foundation: install detection,
supported-repo set, output-path convention, status inspection, and the
subprocess invocation. Runtime routing (so generate calls dispatch to
mlx-video for converted Wan repos) is deferred to a follow-up.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("chaosengine.mlx-video-wan")


def _resolve_convert_root() -> Path:
    override = os.environ.get("CHAOSENGINE_MLX_VIDEO_WAN_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".chaosengine" / "mlx-video-wan"


# Public so callers (tests, setup endpoints) can introspect the path
# without importing private state.
CONVERT_ROOT: Path = _resolve_convert_root()


# Raw Wan-AI checkpoints the upstream convert script supports. These
# are NOT the ``-Diffusers`` mirrors used by the diffusers MPS path —
# the convert script expects raw Wan format
# (``models_t5_umt5-xxl-enc-bf16.pth`` + ``Wan2.1_VAE.pth`` + transformer
# safetensors at the directory root). Mirror repos go through the
# diffusers code path regardless of conversion state.
SUPPORTED_RAW_REPOS: frozenset[str] = frozenset({
    "Wan-AI/Wan2.1-T2V-1.3B",
    "Wan-AI/Wan2.1-T2V-14B",
    "Wan-AI/Wan2.2-TI2V-5B",
    "Wan-AI/Wan2.2-T2V-A14B",
    "Wan-AI/Wan2.2-I2V-A14B",
})


@dataclass(frozen=True)
class WanConvertStatus:
    """Snapshot of a converted Wan checkpoint on disk."""
    repo: str
    converted: bool
    outputDir: str
    hasTransformer: bool
    hasMoeExperts: bool
    hasVae: bool
    hasTextEncoder: bool
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "converted": self.converted,
            "outputDir": self.outputDir,
            "hasTransformer": self.hasTransformer,
            "hasMoeExperts": self.hasMoeExperts,
            "hasVae": self.hasVae,
            "hasTextEncoder": self.hasTextEncoder,
            "note": self.note,
        }


def slug_for(repo: str) -> str:
    """Filesystem-safe slug from an HF repo id (``/`` → ``__``)."""
    return repo.replace("/", "__")


def output_dir_for(repo: str) -> Path:
    """Convention path where the converted MLX weights for ``repo`` land."""
    return CONVERT_ROOT / slug_for(repo)


def is_supported_raw_repo(repo: str | None) -> bool:
    """Return ``True`` when the upstream convert script can handle ``repo``."""
    if not repo:
        return False
    return repo in SUPPORTED_RAW_REPOS


def is_mlx_video_available() -> bool:
    """Cheap check for the upstream package without importing it."""
    return importlib.util.find_spec("mlx_video") is not None


def status_for(repo: str) -> WanConvertStatus:
    """Inspect ``output_dir_for(repo)`` and report what's on disk.

    A repo is considered ``converted`` when the output dir exists AND
    the VAE is present AND either:
    - a single transformer file/dir exists (Wan2.1), or
    - both MoE expert subdirs exist (Wan2.2 high_noise + low_noise).
    Text encoder presence is reported separately because some users
    convert transformer-only and reuse a shared text encoder.
    """
    out = output_dir_for(repo)
    if not out.exists():
        return WanConvertStatus(
            repo=repo,
            converted=False,
            outputDir=str(out),
            hasTransformer=False,
            hasMoeExperts=False,
            hasVae=False,
            hasTextEncoder=False,
            note="Output directory does not exist; conversion not run yet.",
        )

    has_single_transformer = any(out.glob("transformer*.safetensors")) or (out / "transformer").is_dir()
    has_high = (out / "high_noise_model").is_dir()
    has_low = (out / "low_noise_model").is_dir()
    has_moe = has_high and has_low

    has_vae = (
        (out / "vae.safetensors").exists()
        or (out / "Wan2.1_VAE.safetensors").exists()
        or any(out.glob("vae*.safetensors"))
    )
    has_text_encoder = (
        any(out.glob("text_encoder*.safetensors"))
        or any(out.glob("models_t5*.safetensors"))
        or any(out.glob("umt5*.safetensors"))
    )

    converted = (has_single_transformer or has_moe) and has_vae

    note = None
    if not converted:
        missing = []
        if not (has_single_transformer or has_moe):
            missing.append("transformer (single .safetensors or high_noise/low_noise dirs)")
        if not has_vae:
            missing.append("VAE")
        note = f"Output dir exists but conversion incomplete; missing: {', '.join(missing)}."

    return WanConvertStatus(
        repo=repo,
        converted=converted,
        outputDir=str(out),
        hasTransformer=has_single_transformer or has_moe,
        hasMoeExperts=has_moe,
        hasVae=has_vae,
        hasTextEncoder=has_text_encoder,
        note=note,
    )


def list_converted() -> list[WanConvertStatus]:
    """Return ``WanConvertStatus`` for every converted dir under
    ``CONVERT_ROOT`` that maps back to a known supported repo. Useful
    for the Setup page's "Available Wan MLX runtimes" listing."""
    if not CONVERT_ROOT.exists():
        return []
    out: list[WanConvertStatus] = []
    for entry in sorted(CONVERT_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        repo = entry.name.replace("__", "/", 1)
        if not is_supported_raw_repo(repo):
            continue
        status = status_for(repo)
        if status.converted:
            out.append(status)
    return out


def run_convert(
    checkpoint_dir: Path | str,
    repo: str,
    *,
    dtype: str = "bfloat16",
    model_version: str = "auto",
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
    timeout_seconds: int = 3600,
    python_executable: str | None = None,
) -> WanConvertStatus:
    """Run ``python -m mlx_video.models.wan_2.convert`` on a checkpoint.

    Output lands at ``output_dir_for(repo)`` (under ``CONVERT_ROOT``).
    Returns the post-convert ``WanConvertStatus`` so the caller can
    decide whether to surface a runtimeNote about partial conversion.

    Subprocess timeout defaults to 1 hour — large models (Wan2.2 A14B
    at ~67 GB raw) can take 20-30 minutes to convert on M-series Macs;
    1 hour gives plenty of headroom without leaving the worker hung
    indefinitely if the script wedges.
    """
    if not is_supported_raw_repo(repo):
        raise ValueError(
            f"Unsupported Wan repo {repo!r}. "
            f"Supported: {sorted(SUPPORTED_RAW_REPOS)}"
        )

    if not is_mlx_video_available():
        raise RuntimeError(
            "mlx-video is not installed. Run "
            "``pip install -e \".[mlx-video]\"`` (installs from git) first."
        )

    checkpoint_path = Path(checkpoint_dir).expanduser()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint dir not found: {checkpoint_path}. "
            "Download the raw Wan repo first via "
            "``huggingface-cli download <repo>``."
        )

    out = output_dir_for(repo)
    out.parent.mkdir(parents=True, exist_ok=True)

    python_bin = python_executable or sys.executable
    args = [
        python_bin,
        "-m", "mlx_video.models.wan_2.convert",
        "--checkpoint-dir", str(checkpoint_path),
        "--output-dir", str(out),
        "--dtype", dtype,
        "--model-version", model_version,
    ]
    if quantize:
        args.extend([
            "--quantize",
            "--bits", str(bits),
            "--group-size", str(group_size),
        ])

    LOG.info("Starting Wan convert: repo=%s args=%s", repo, " ".join(args))
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stderr or exc.stdout or "")
        raise RuntimeError(
            f"Wan convert timed out after {timeout_seconds}s for {repo}. "
            f"Last output: {str(tail)[-500:]}"
        ) from exc

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-800:]
        raise RuntimeError(
            f"Wan convert exited with code {result.returncode} for {repo}. "
            f"Last output:\n{tail}"
        )

    return status_for(repo)
