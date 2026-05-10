"""``install-gpu-bundle`` background-job endpoints.

The one-click "Install GPU support" flow that pulls torch (walking the CUDA
download indexes), then a curated list of diffusers / transformers / video
runtime deps. Runs as a daemon thread so the FastAPI request returns
immediately; the frontend polls ``/api/setup/install-gpu-bundle/status``
for progress updates.

Pre-install metadata (target dir, free disk, package list) is exposed via
``/api/setup/gpu-bundle-info`` so the install banner can render an honest
"about to download ~2.5 GB" confirmation.

Extracted from ``routes/setup/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.routes.setup._install_helpers import (
    _CUDA_TORCH_INDEXES,
    _all_attempts_lack_wheel,
    _cleanup_mlx_video_shadow_metadata,
    _extras_site_packages,
    _find_installed_torch_version,
    _is_cuda_torch_version,
    _purge_broken_distributions,
    _purge_stale_torch_from_extras,
    _read_python_version,
    _run_pip_install,
    _write_torch_constraint,
)

router = APIRouter()


# Packages installed by the one-click "Install GPU support" flow. Ordered
# so torch installs first — every other package below can defer to whatever
# torch version ended up on disk. If the user's Python has no CUDA wheel
# the job stops at torch (users get a clear "switch to Python 3.13" hint)
# rather than pressing on and installing diffusers against no torch.
_GPU_BUNDLE_PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch>=2.4.0"),
    ("diffusers", "diffusers>=0.30.0"),
    ("accelerate", "accelerate>=0.34.0"),
    ("transformers", "transformers>=4.44.0"),
    ("safetensors", "safetensors>=0.4.5"),
    ("pillow", "pillow>=10.4.0"),
    # huggingface_hub depends on pyyaml at import time. When pip --target
    # picks up a partial wheel cache for PyYAML on Windows, the snapshot_download
    # subprocess dies with ``ModuleNotFoundError: No module named 'yaml.error'``
    # which then surfaces as the per-row download error in Image / Video
    # Discover. Installing pyyaml explicitly (instead of relying on transitive
    # resolution) gives pip a clean install of the wheel into extras and
    # prevents that mode from happening on first launch.
    ("pyyaml", "pyyaml>=6.0"),
    ("huggingface-hub", "huggingface-hub>=0.26.0"),
    ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio-ffmpeg"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("protobuf", "protobuf"),
    ("ftfy", "ftfy"),
    # NF4 quantization for FLUX's 12B transformer — shrinks it from ~24 GB
    # (bf16) to ~7 GB so it runs comfortably on 24 GB consumer GPUs. With
    # bf16 + cpu_offload alone, a 4090 is right at the edge of VRAM and
    # pays a heavy pagefile-thrash cost per step.
    ("bitsandbytes", "bitsandbytes>=0.43.0"),
    # GGUF loader for image/video DiT transformers. Cross-platform
    # quantization (works on CUDA, MPS, CPU) complementing the
    # CUDA-only bitsandbytes NF4 path.
    ("gguf", "gguf>=0.10.0"),
    # TorchAO int8wo — Apple Silicon's answer to NF4 for FLUX. Drops
    # the 12B transformer from ~24 GB to ~12 GB on MPS so FLUX fits in
    # 32 GB unified memory without pagefile thrash.
    ("torchao", "torchao>=0.6.0"),
]

# Apple Silicon: ship mlx-video alongside the diffusers GPU bundle so the
# MLX-native LTX-2 engine is available out of the box. Skipped on Intel
# Macs and non-Darwin hosts where mlx-video has no working backend.
if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
    _GPU_BUNDLE_PACKAGES.append((
        "mlx-video",
        "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git",
    ))

# Rough total download size (torch CUDA dominates at ~2 GB; others sum to
# ~400 MB). We expose this to the UI so the install banner shows an
# honest "~2.5 GB, 1-3 min on broadband" instead of a silent multi-minute
# progress bar.
_GPU_BUNDLE_APPROX_DOWNLOAD_BYTES = 2_500_000_000

# Minimum free disk space we require before starting (download + extract +
# safety margin). Torch unpacks to ~2.5 GB, and pip holds both the wheel
# and the extracted copy during install, so we need ~5 GB of headroom.
_GPU_BUNDLE_REQUIRED_FREE_BYTES = 5_500_000_000


@dataclass
class _GpuBundleJobState:
    """In-memory status for the currently-running or most-recent install.

    Only one install runs at a time — a second POST while running returns
    the existing state. On completion the state sticks around so a late
    status poll sees the final outcome.
    """

    id: str = ""
    phase: str = "idle"  # idle | preflight | downloading | verifying | done | error
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = 0
    percent: float = 0.0
    target_dir: str | None = None
    index_url_used: str | None = None
    python_version: str | None = None
    no_wheel_for_python: bool = False
    cuda_verified: bool | None = None
    requires_restart: bool = False
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "packageCurrent": self.package_current,
            "packageIndex": self.package_index,
            "packageTotal": self.package_total,
            "percent": round(self.percent, 1),
            "targetDir": self.target_dir,
            "indexUrlUsed": self.index_url_used,
            "pythonVersion": self.python_version,
            "noWheelForPython": self.no_wheel_for_python,
            "cudaVerified": self.cuda_verified,
            "requiresRestart": self.requires_restart,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": self.attempts,
            "done": self.done,
        }


_GPU_BUNDLE_JOB = _GpuBundleJobState()
_GPU_BUNDLE_LOCK = threading.Lock()


def _free_bytes(path: Path) -> int | None:
    """Return free disk space in bytes for the volume hosting ``path``.

    Returns None when the path doesn't exist yet AND no parent does — we
    can't check a drive we can't touch. ``shutil.disk_usage`` walks up
    until it hits an existing directory, so we mirror that.
    """
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _verify_cuda(python: str, extras_dir: Path) -> tuple[bool, str]:
    """Spawn a fresh Python to confirm ``torch.cuda.is_available()``.

    Uses a subprocess (not in-process import) because the backend may have
    already imported torch from the bundled CPU wheel; once sys.modules has
    a torch entry, ``import torch`` inside the running process returns the
    cached stale module. A fresh interpreter with PYTHONPATH pointing at
    extras sees the newly-installed wheel.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(extras_dir) + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        "out = {'python': f'{sys.version_info.major}.{sys.version_info.minor}'}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = torch.__version__\n"
        "    out['cuda_build'] = str(getattr(torch.version, 'cuda', None))\n"
        "    out['cuda_available'] = bool(getattr(torch.cuda, 'is_available', lambda: False)())\n"
        "except Exception as exc:\n"
        "    out['error'] = str(exc).splitlines()[0][:200]\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [python, "-c", script], capture_output=True, text=True, env=env, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"CUDA verification subprocess failed: {exc}"
    detail = (result.stdout or "").strip() + (("\n" + result.stderr) if result.stderr else "")
    # Consider the check passed only if the child exited cleanly AND said so.
    ok = result.returncode == 0 and '"cuda_available": true' in (result.stdout or "").lower()
    return ok, detail


_DLL_LOCK_PATTERNS = (
    # Windows pip rmtree failure signatures when a torch DLL is held open by
    # another process (typically the ChaosEngineAI backend that eagerly
    # imported torch before the install started).
    "winerror 5",
    "permissionerror",
    "access is denied",
)


def _looks_like_dll_lock(output: str) -> bool:
    """Heuristic: does pip's stderr look like a locked-DLL rmtree failure?

    The backend eagerly imported torch at startup for warmup speed, and
    once torch/lib/*.dll is in the process handle table pip can't remove
    those files even with --force-reinstall --target. Detecting this
    specifically lets us surface a clear "restart backend, then retry"
    message instead of burying the root cause under a wall of pip trace.
    """
    lowered = output.lower()
    if "torch" not in lowered or ".dll" not in lowered:
        return False
    return any(marker in lowered for marker in _DLL_LOCK_PATTERNS)


def _install_torch_walking_indexes(
    python: str, extras_dir: Path, state: _GpuBundleJobState
) -> tuple[bool, str | None]:
    """Install torch walking the CUDA index list. First success wins."""
    for index_url in _CUDA_TORCH_INDEXES:
        state.message = f"Downloading torch from {index_url}"
        ok, output = _run_pip_install(
            python, "torch>=2.4.0", extras_dir, index_url,
            ["--force-reinstall", "--no-deps"],
        )
        state.attempts.append({"indexUrl": index_url, "ok": ok, "output": output[-2000:]})
        if not ok and _looks_like_dll_lock(output):
            # Stop walking indexes — no index will succeed until the DLLs
            # are released. Raise so the worker captures a clean error
            # with an actionable message instead of four duplicate
            # "WinError 5" attempt rows.
            raise RuntimeError(
                "Cannot overwrite existing torch files because they're locked by the running "
                "backend (likely a previous partial install). Click Restart Backend, wait for "
                "it to come back online, then click Install GPU runtime again. If the problem "
                "persists, quit ChaosEngineAI fully, delete "
                f"{extras_dir / 'torch'}, and reopen."
            )
        if ok:
            # Second pass: install torch again with deps (no --no-deps) so
            # transitive nvidia-cublas / jinja2 / etc. land in the extras
            # tree. We keep --no-deps in the first pass to isolate the
            # winning CUDA index from transitive PyPI resolution noise.
            state.message = f"Resolving torch dependencies ({index_url})"
            dep_ok, dep_output = _run_pip_install(
                python, "torch>=2.4.0", extras_dir, index_url, [],
            )
            state.attempts.append({
                "indexUrl": index_url, "phase": "deps", "ok": dep_ok,
                "output": dep_output[-2000:],
            })
            return True, index_url
    return False, None


def _gpu_bundle_job_worker(python: str, extras_dir: Path) -> None:
    """Background-thread entry point for the GPU bundle install.

    Updates ``_GPU_BUNDLE_JOB`` as it progresses; the status endpoint reads
    that struct without locking (a stale read is fine — the field updates
    are each atomic assignments and the UI just polls again).

    Failure handling:
      - Fatal: the worker raises, ``except`` block sets ``phase=error`` +
        ``error`` + ``message`` from the exception text.
      - Non-fatal (post-torch package install fails): the loop appends a
        FAIL attempt with full output and keeps going. At the end we sum
        non-fatal failures into a final message so the UI doesn't show
        ``done`` with a green tick when half the bundle didn't land.
    """
    state = _GPU_BUNDLE_JOB
    non_fatal_failures: list[str] = []
    # Apple Silicon hosts ship torch (MPS-enabled) inside the bundled venv —
    # walking CUDA indexes here would fail (no aarch64-darwin CUDA wheels)
    # and abort the rest of the bundle. Skip the torch step on macOS arm64
    # so diffusers + mlx-video still install. Mirror the cuda-verify skip
    # at the tail so the summary stays accurate.
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    )
    try:
        state.phase = "preflight"
        state.message = "Checking disk space"
        free = _free_bytes(extras_dir)
        if free is not None and free < _GPU_BUNDLE_REQUIRED_FREE_BYTES:
            required_gb = _GPU_BUNDLE_REQUIRED_FREE_BYTES / 1_000_000_000
            free_gb = free / 1_000_000_000
            raise RuntimeError(
                f"Need at least {required_gb:.1f} GB free on the drive hosting "
                f"{extras_dir} — currently {free_gb:.1f} GB free. Free up space "
                "and try again."
            )

        # Sweep any broken ``~<pkg>`` stubs from a prior interrupted run —
        # they cause noisy pip warnings and occasionally block progress.
        purged = _purge_broken_distributions(extras_dir)
        if purged:
            state.message = f"Cleaned up {len(purged)} broken stub(s) from prior run"

        if not is_apple_silicon:
            purged_torch = _purge_stale_torch_from_extras(extras_dir)
            if purged_torch:
                state.attempts.append({
                    "phase": "torch-cleanup",
                    "ok": True,
                    "output": (
                        "Removed stale torch/CUDA runtime entries before reinstall: "
                        + ", ".join(purged_torch[:16])
                    ),
                })

        state.phase = "downloading"
        state.package_total = len(_GPU_BUNDLE_PACKAGES)

        if is_apple_silicon:
            # Skip torch CUDA walk — torch is already in the bundled venv
            # (MPS-enabled). Mark the slot as accounted for and proceed to
            # diffusers + mlx-video.
            state.package_index = 1
            state.package_current = "torch"
            state.percent = 0.0
            state.attempts.append({
                "package": "torch",
                "ok": True,
                "output": "Apple Silicon: using bundled MPS torch (CUDA install skipped)",
            })
            ok = True
            index_url = None
        else:
            # Package 1: torch (walks CUDA indexes).
            state.package_index = 1
            state.package_current = "torch"
            state.percent = 0.0
            ok, index_url = _install_torch_walking_indexes(python, extras_dir, state)
        if not ok:
            torch_attempts = [
                a for a in state.attempts
                if a.get("indexUrl") and a.get("phase") != "deps"
            ]
            state.no_wheel_for_python = _all_attempts_lack_wheel(torch_attempts)
            if state.no_wheel_for_python:
                raise RuntimeError(
                    f"PyTorch doesn't publish a CUDA wheel for Python {state.python_version} yet. "
                    "Rebuild ChaosEngineAI against Python 3.13 (most-widely-supported), "
                    "or set CHAOSENGINE_TORCH_INDEX_URL to a newer index before launching."
                )
            # Pull the most recent failure tail so the error message
            # itself is actionable (no blank "All indexes failed" toast).
            last_attempt = state.attempts[-1] if state.attempts else {}
            tail = (last_attempt.get("output") or "").splitlines()[-3:]
            tail_blob = " | ".join(line.strip() for line in tail if line.strip())[:300]
            raise RuntimeError(
                "All CUDA index candidates failed. Check your internet connection, "
                f"firewall, or proxy settings. Last pip output: {tail_blob or '(empty)'}"
            )
        state.index_url_used = index_url

        # Pin torch in a constraints file so the follow-up packages
        # (diffusers, transformers, etc.) can't cause pip to swap the
        # CUDA wheel for a CPU one from default PyPI. Without the pin,
        # the resolver occasionally decides a fresh torch satisfies some
        # transitive upper bound better than the installed CUDA wheel,
        # and silently overwrites it. Any package that strictly requires
        # a different torch version will now error out visibly against
        # the constraint instead of silently clobbering torch.
        constraint_path: Path | None = None
        torch_version = _find_installed_torch_version(extras_dir)
        if torch_version:
            try:
                constraint_path = _write_torch_constraint(extras_dir, torch_version)
                state.attempts.append({
                    "phase": "constraint",
                    "ok": True,
                    "output": f"Pinned torch=={torch_version} for subsequent packages",
                })
            except OSError as exc:
                # Non-fatal: we just lose the torch pin for this run. The
                # packages below might or might not clobber torch, but the
                # verify step at the end will detect that.
                state.attempts.append({
                    "phase": "constraint",
                    "ok": False,
                    "output": f"Could not write torch constraint: {exc}",
                })

        # Remaining packages: standard PyPI. Most are small — progress
        # advances quickly here so the UI doesn't look frozen.
        for idx, (label, spec) in enumerate(_GPU_BUNDLE_PACKAGES[1:], start=2):
            state.package_index = idx
            state.package_current = label
            state.percent = ((idx - 1) / len(_GPU_BUNDLE_PACKAGES)) * 100.0
            state.message = f"Installing {label}"
            extra_flags: list[str] = []
            if constraint_path is not None:
                extra_flags = ["--constraint", str(constraint_path)]
            ok, output = _run_pip_install(python, spec, extras_dir, None, extra_flags)
            if label == "mlx-video":
                cleaned = _cleanup_mlx_video_shadow_metadata(extras_dir)
                if cleaned:
                    output = (
                        f"{output}\n\nCleaned stale mlx-video metadata: "
                        f"{', '.join(sorted(set(cleaned)))}"
                    ).strip()
            state.attempts.append({"package": label, "ok": ok, "output": output[-2000:]})
            if not ok:
                # Individual package failure is non-fatal — torch + diffusers
                # are the must-haves and torch is earlier in the list. Track
                # the failure for the final summary so the UI doesn't show
                # a clean "done" when ftfy/sentencepiece/etc. didn't land.
                non_fatal_failures.append(label)
                state.message = (
                    f"{label} install failed (non-fatal — see install log; you can "
                    f"retry it individually after the bundle finishes)"
                )

        # Repair pass: pip --target ignores already-installed packages in
        # the target dir for resolver purposes (it only checks the user's
        # main site-packages), so transitive torch deps from accelerate /
        # bitsandbytes can pull the CPU torch wheel from default PyPI and
        # clobber the CUDA wheel installed in step 1. The PYTHONPATH and
        # constraint defenses in _run_pip_install close most of that gap,
        # but a defence-in-depth re-install here guarantees the CUDA wheel
        # is the one that survives even if pip's resolver decides to
        # upgrade torch despite the constraint.
        #
        # The pass is a no-op if torch in extras still has a CUDA local
        # version segment (``+cu124`` / ``+cu126`` / ...). It kicks in
        # when the CUDA wheel was clobbered by a bare or ``+cpu`` wheel.
        if not is_apple_silicon and index_url:
            current_torch = _find_installed_torch_version(extras_dir)
            if current_torch and not _is_cuda_torch_version(current_torch):
                state.message = "Repairing CUDA torch wheel (clobbered by transitive deps)"
                repair_note = (
                    f"Torch was downgraded to {current_torch} (not a CUDA wheel) - "
                    "a follow-up install pulled CPU torch from default PyPI. "
                    f"Reinstalling from {index_url} to restore CUDA support.\n\n"
                )
                repair_ok, repair_output = _run_pip_install(
                    python, "torch>=2.4.0", extras_dir, index_url,
                    ["--no-deps", "--force-reinstall"],
                )
                state.attempts.append({
                    "phase": "torch-repair",
                    "ok": repair_ok,
                    "output": (repair_note + repair_output)[-2000:],
                })
                if not repair_ok:
                    non_fatal_failures.append("torch-repair")

        state.phase = "verifying"
        state.percent = 95.0
        state.package_current = None
        if is_apple_silicon:
            # No CUDA on Apple Silicon — bundled torch already gives us MPS.
            # Mark verify as a pass so the UI doesn't show a red verify badge
            # on a successful Apple Silicon install.
            state.message = "Apple Silicon — skipping CUDA verify (MPS via bundled torch)"
            cuda_ok = True
            detail = "skipped on Apple Silicon"
        else:
            state.message = "Verifying CUDA availability"
            cuda_ok, detail = _verify_cuda(python, extras_dir)
        state.cuda_verified = cuda_ok
        state.attempts.append({"phase": "verify", "ok": cuda_ok, "output": detail[-2000:]})

        # Tell the import system to re-scan ``sys.path`` so packages
        # written into the extras dir during this run are visible to the
        # next ``importlib.util.find_spec`` call (the image-runtime probe
        # uses one). Without this, the runtime continues reporting
        # "placeholder" until a backend restart even though the bundle
        # is on disk. Also reset the cached VRAM total so the post-install
        # capabilities snapshot reflects the freshly importable torch.
        try:
            importlib.invalidate_caches()
        except Exception:
            pass
        try:
            from backend_service.helpers.gpu import (
                reset_torch_status_cache,
                reset_vram_total_cache,
            )
            reset_vram_total_cache()
            # The /api/system/gpu-status endpoint caches its torch probe per
            # process to avoid spawning a child Python on every poll. The
            # cached "torch not importable" answer from before this install
            # is now stale — flush it so the next frontend poll re-probes
            # and the banner updates without a backend restart.
            reset_torch_status_cache()
        except Exception:
            pass

        state.phase = "done"
        state.percent = 100.0
        state.done = True
        state.requires_restart = True
        state.finished_at = time.time()
        if cuda_ok and not non_fatal_failures:
            state.message = "GPU support installed. Restart the backend to activate."
        elif cuda_ok and non_fatal_failures:
            # Surface the partial failure so users know to retry the
            # individual missing pieces (mp4 encoder, tokenizers) rather
            # than re-running the whole 2 GB torch install.
            state.message = (
                "GPU support installed and CUDA verified, but "
                f"{len(non_fatal_failures)} optional package(s) failed: "
                f"{', '.join(non_fatal_failures)}. Restart the backend to activate "
                "torch + diffusers; the failed packages can be retried individually."
            )
        else:
            verify_tail = (detail or "").splitlines()[-2:]
            verify_blob = " | ".join(line.strip() for line in verify_tail if line.strip())[:300]
            state.message = (
                "Install completed but CUDA isn't available. torch may have landed "
                "as the CPU wheel, or your NVIDIA driver doesn't match. "
                f"Verify subprocess said: {verify_blob or '(no output)'}. "
                "See the install log for the full attempts list."
            )
    except Exception as exc:  # noqa: BLE001 — surface ANY failure via status
        # Always set a non-empty message: ``str(exc)`` can be empty for
        # bare-Exception cases and that's exactly when the UI ends up
        # showing "failed without reason". Fall back to the exception
        # type name so users see SOMETHING actionable.
        message = str(exc) or f"{type(exc).__name__} (no message attached)"
        state.error = message
        state.phase = "error"
        state.message = message
        state.done = True
        state.finished_at = time.time()


@router.post("/api/setup/install-gpu-bundle")
def start_install_gpu_bundle(request: Request) -> dict[str, Any]:
    """Kick off a background install of the full GPU runtime bundle.

    Returns the current job state immediately. Poll
    ``/api/setup/install-gpu-bundle/status`` for progress. Calling this
    endpoint again while a job is running returns the running job's state
    rather than starting a new one.
    """
    state_chaosengine = request.app.state.chaosengine
    python = state_chaosengine.runtime.capabilities.pythonExecutable
    extras = _extras_site_packages()
    if extras is None:
        raise HTTPException(
            status_code=500,
            detail="Could not resolve the extras site-packages directory.",
        )
    extras.mkdir(parents=True, exist_ok=True)

    with _GPU_BUNDLE_LOCK:
        if _GPU_BUNDLE_JOB.phase in {"preflight", "downloading", "verifying"}:
            return _GPU_BUNDLE_JOB.to_dict()

        # Reset state for a fresh run.
        _GPU_BUNDLE_JOB.id = f"gpu-bundle-{int(time.time() * 1000)}"
        _GPU_BUNDLE_JOB.phase = "preflight"
        _GPU_BUNDLE_JOB.message = "Starting install"
        _GPU_BUNDLE_JOB.package_current = None
        _GPU_BUNDLE_JOB.package_index = 0
        _GPU_BUNDLE_JOB.package_total = len(_GPU_BUNDLE_PACKAGES)
        _GPU_BUNDLE_JOB.percent = 0.0
        _GPU_BUNDLE_JOB.target_dir = str(extras)
        _GPU_BUNDLE_JOB.index_url_used = None
        _GPU_BUNDLE_JOB.python_version = _read_python_version(python)
        _GPU_BUNDLE_JOB.no_wheel_for_python = False
        _GPU_BUNDLE_JOB.cuda_verified = None
        _GPU_BUNDLE_JOB.requires_restart = False
        _GPU_BUNDLE_JOB.error = None
        _GPU_BUNDLE_JOB.started_at = time.time()
        _GPU_BUNDLE_JOB.finished_at = 0.0
        _GPU_BUNDLE_JOB.attempts = []
        _GPU_BUNDLE_JOB.done = False

        thread = threading.Thread(
            target=_gpu_bundle_job_worker,
            args=(python, extras),
            name="chaosengine-gpu-bundle-install",
            daemon=True,
        )
        thread.start()

    state_chaosengine.add_log(
        "server", "info",
        f"GPU bundle install started (job={_GPU_BUNDLE_JOB.id}, target={extras})",
    )
    return _GPU_BUNDLE_JOB.to_dict()


@router.get("/api/setup/install-gpu-bundle/status")
def install_gpu_bundle_status() -> dict[str, Any]:
    """Snapshot of the current GPU bundle install job.

    Safe to poll at 1-2 Hz. Returns ``phase="idle"`` before any install
    has been started in this backend session.
    """
    return _GPU_BUNDLE_JOB.to_dict()


@router.get("/api/setup/gpu-bundle-info")
def gpu_bundle_info() -> dict[str, Any]:
    """Pre-install metadata for the install banner UI.

    Surfaces the extras target dir, approximate download size, free disk
    on the target volume, and the set of packages we intend to install so
    the frontend can render a clear "what you're about to do" confirmation.
    """
    extras = _extras_site_packages()
    extras_str = str(extras) if extras else None
    free = _free_bytes(extras) if extras else None
    return {
        "targetDir": extras_str,
        "approxDownloadBytes": _GPU_BUNDLE_APPROX_DOWNLOAD_BYTES,
        "requiredFreeBytes": _GPU_BUNDLE_REQUIRED_FREE_BYTES,
        "freeBytes": free,
        "packages": [{"label": label, "spec": spec} for label, spec in _GPU_BUNDLE_PACKAGES],
    }
