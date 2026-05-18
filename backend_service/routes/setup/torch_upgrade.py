"""``torch-upgrade-available`` + ``upgrade-torch`` endpoints.

Once a user has a working CUDA torch installed in extras (via
``/api/setup/install-gpu-bundle`` or ``/api/setup/install-cuda-torch``),
this surface offers a path to a newer torch wheel WITHOUT making them
re-run the full 2.5 GB GPU bundle install.

Three behaviours that keep this safe to expose by default:

  1. **Detection short-circuits unsafe configurations.** The GET endpoint
     refuses to offer an upgrade when the host is Apple Silicon (no CUDA
     wheels exist), when torch is the ``+cpu`` build (the user has a
     different problem — they should run ``install-cuda-torch`` instead),
     or when the installed wheel has no ``+cu...`` local-version tag we
     can map back to a download index. The frontend reads ``available:
     false`` and just doesn't render the pill.

  2. **Rollback move instead of purge.** The POST endpoint moves the
     existing torch wheel + transitive ``nvidia_*`` dirs to a sibling
     ``.torch-rollback-<version>/`` directory rather than deleting them.
     If verification fails after install, the rollback is restored bit-
     for-bit — no 2.5 GB re-download. Pre-existing
     :func:`_purge_stale_torch_from_extras` is still used by the fresh-
     install paths in ``cuda_torch.py`` and ``gpu_bundle.py``; this
     module is the one that needs the safety net because the user
     already had a working setup.

  3. **ABI-dependent rebuild only on minor / major bumps.** torch's C++
     ABI is stable across patch bumps (2.6.0 → 2.6.1) and breaks across
     minor / major (2.6 → 2.7, 2.x → 3.x). For minor+ upgrades the
     worker walks the present-in-extras subset of bitsandbytes /
     torchao / nunchaku / sageattention and re-installs them with
     ``--force-reinstall`` so pip picks the wheel whose metadata
     matches the freshly-installed torch. Patches skip the walk
     entirely so a hotfix install is a single pip call.

Job pattern mirrors ``gpu_bundle.py`` — kick off a background thread,
poll ``/api/setup/upgrade-torch/status`` for progress. The shared
``InstallLogPanel`` on the frontend renders the attempts list verbatim.
"""

from __future__ import annotations

import importlib
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend_service.i18n import localized_detail

from backend_service.routes.setup._install_helpers import (
    _abi_dependents_present,
    _classify_torch_upgrade,
    _cleanup_old_torch_rollbacks,
    _extract_cuda_tag,
    _extras_site_packages,
    _find_installed_torch_version,
    _index_url_for_cuda_tag,
    _is_cuda_torch_version,
    _move_torch_to_rollback,
    _query_latest_torch_version,
    _read_python_version,
    _restore_torch_from_rollback,
    _run_pip_install,
    _write_torch_constraint,
)
from backend_service.routes.setup.gpu_bundle import _verify_cuda

router = APIRouter()


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


# ----------------------------------------------------------------------
# Detection endpoint
# ----------------------------------------------------------------------


@router.get("/api/setup/torch-upgrade-available")
def torch_upgrade_available(request: Request) -> dict[str, Any]:
    """Return whether a newer torch wheel is available for the user's setup.

    Response shape when an upgrade is offered::

        {
          "available": true,
          "current": "2.6.0+cu124",
          "latest": "2.6.1+cu124",
          "upgradeType": "patch" | "minor" | "major",
          "rebuildPackages": ["bitsandbytes", "torchao"],
          "indexUrl": "https://download.pytorch.org/whl/cu124"
        }

    Response when nothing to offer (``available: false``) always carries
    a machine-readable ``reason`` so the frontend can decide whether to
    stay silent (``no-extras``, ``apple-silicon``, ``already-latest``)
    or show a diagnostic (``index-query-failed`` may indicate a network
    issue worth surfacing in Diagnostics).
    """
    state = request.app.state.chaosengine
    python = state.runtime.capabilities.pythonExecutable
    extras = _extras_site_packages()
    if extras is None or not extras.is_dir():
        return {"available": False, "reason": "no-extras"}
    if _is_apple_silicon():
        return {"available": False, "reason": "apple-silicon"}

    current = _find_installed_torch_version(extras)
    if not current:
        return {"available": False, "reason": "torch-not-installed"}
    if not _is_cuda_torch_version(current):
        return {"available": False, "reason": "cpu-wheel", "current": current}

    tag = _extract_cuda_tag(current)
    index_url = _index_url_for_cuda_tag(tag)
    if not index_url:
        return {"available": False, "reason": "no-cuda-tag", "current": current}

    latest = _query_latest_torch_version(python, index_url)
    if not latest:
        return {
            "available": False,
            "reason": "index-query-failed",
            "current": current,
            "indexUrl": index_url,
        }

    classification = _classify_torch_upgrade(current, latest)
    if classification is None:
        return {
            "available": False,
            "reason": "already-latest",
            "current": current,
            "latest": latest,
            "indexUrl": index_url,
        }

    rebuild = _abi_dependents_present(extras) if classification in ("minor", "major") else []
    return {
        "available": True,
        "current": current,
        "latest": latest,
        "upgradeType": classification,
        "rebuildPackages": rebuild,
        "indexUrl": index_url,
    }


# ----------------------------------------------------------------------
# Upgrade endpoint (background job pattern)
# ----------------------------------------------------------------------


@dataclass
class _TorchUpgradeJobState:
    """In-memory status for the currently-running or most-recent upgrade.

    Shape compatible with ``InstallLogPanel`` on the frontend — same
    ``phase`` / ``message`` / ``attempts`` / ``done`` fields plus a few
    upgrade-specific extras (current/target version, rollback path).
    """

    id: str = ""
    phase: str = "idle"  # idle | preflight | upgrading | verifying | done | error
    message: str = ""
    current_version: str | None = None
    target_version: str | None = None
    upgrade_type: str | None = None
    index_url: str | None = None
    rebuild_dependents: bool = False
    rebuilt_packages: list[str] = field(default_factory=list)
    rolled_back: bool = False
    rollback_path: str | None = None
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
            "currentVersion": self.current_version,
            "targetVersion": self.target_version,
            "upgradeType": self.upgrade_type,
            "indexUrl": self.index_url,
            "rebuildDependents": self.rebuild_dependents,
            "rebuiltPackages": list(self.rebuilt_packages),
            "rolledBack": self.rolled_back,
            "rollbackPath": self.rollback_path,
            "cudaVerified": self.cuda_verified,
            "requiresRestart": self.requires_restart,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": list(self.attempts),
            "done": self.done,
        }


_UPGRADE_JOB = _TorchUpgradeJobState()
_UPGRADE_LOCK = threading.Lock()


class _UpgradeTorchRequest(BaseModel):
    """POST body for /api/setup/upgrade-torch.

    ``rebuildDependents`` defaults to True so a one-click upgrade Just
    Works for minor / major bumps. Patch bumps ignore the flag (the ABI
    is stable, no rebuild needed). Callers can pass ``False`` to skip
    the rebuild — useful if the user wants to test torch first and only
    rebuild the deps after confirming the new wheel is healthy.
    """

    rebuildDependents: bool = True


def _upgrade_torch_worker(python: str, extras_dir: Path, rebuild_dependents: bool) -> None:
    """Background-thread entry point for ``/api/setup/upgrade-torch``.

    Behaviour, in order:

      1. Detect current torch + classify upgrade vs the cu{N} index.
         If anything's off (CPU wheel, missing version, already latest)
         we raise — the frontend should have prevented this case by
         reading the detection endpoint first, but the worker re-checks
         so a stale frontend cache can't kick off a useless install.

      2. Move the existing torch + nvidia_* dirs to a sibling rollback
         directory. We keep the old wheel on disk so a failed verify
         restores without a 2.5 GB re-download.

      3. Install the target torch from the same cu{N} index — two pass
         like the existing install path (``--force-reinstall --no-deps``
         then plain install for transitive deps). Pin the new torch in
         the constraints file so subsequent rebuilds don't accidentally
         swap it back.

      4. On minor / major bumps, rebuild any ABI-dependent packages
         that are present in extras (bitsandbytes / torchao / nunchaku /
         sageattention). ``--force-reinstall --no-deps`` so pip ONLY
         touches the package being rebuilt — we already pinned torch.

      5. Verify CUDA in a fresh subprocess. If verification fails,
         restore the rollback. If it succeeds, prune older rollbacks
         (keep the most recent one as a safety net) and signal the
         frontend to prompt for a backend restart.

    Any exception inside this body triggers the rollback path; the
    error message lands in ``state.error`` for the frontend to surface.
    """
    state = _UPGRADE_JOB
    rollback: Path | None = None
    try:
        # --- 1. Detect + classify ----------------------------------
        state.phase = "preflight"
        state.message = "Detecting current torch"
        current = _find_installed_torch_version(extras_dir)
        if not current:
            raise RuntimeError(
                "No torch found in extras. Run Install GPU runtime first."
            )
        if not _is_cuda_torch_version(current):
            raise RuntimeError(
                f"Installed torch ({current}) is the CPU wheel — use "
                "Install CUDA torch instead of Upgrade."
            )
        state.current_version = current

        tag = _extract_cuda_tag(current)
        index_url = _index_url_for_cuda_tag(tag)
        if not index_url:
            raise RuntimeError(
                f"Could not determine CUDA download index for torch {current}."
            )
        state.index_url = index_url

        state.message = f"Querying {index_url} for newer torch"
        latest = _query_latest_torch_version(python, index_url)
        if not latest:
            raise RuntimeError(
                f"Could not query latest torch from {index_url} — check "
                "network connectivity, firewall, or proxy."
            )
        state.target_version = latest

        classification = _classify_torch_upgrade(current, latest)
        if classification is None:
            raise RuntimeError(
                f"Torch {current} is already at or above the latest "
                f"available on {index_url} ({latest})."
            )
        state.upgrade_type = classification

        # --- 2. Move to rollback -----------------------------------
        state.phase = "upgrading"
        state.message = f"Stashing torch {current} for rollback"
        rollback = _move_torch_to_rollback(extras_dir, current)
        if rollback is None:
            raise RuntimeError(
                "Could not move existing torch to rollback directory — "
                "check that the extras drive has free space and the "
                "backend has write permission."
            )
        state.rollback_path = str(rollback)
        state.attempts.append({
            "phase": "rollback-prepare",
            "ok": True,
            "output": f"Moved torch {current} + nvidia-* dirs to {rollback}",
        })

        # --- 3. Install target torch -------------------------------
        # Use the base version (strip the local-version segment) — PEP
        # 440 lets ``torch==2.6.1`` match ``2.6.1+cu124`` on the index,
        # while ``torch==2.6.1+cu124`` is unsatisfiable when pip pulls
        # in transitive deps from default PyPI.
        target_spec = f"torch=={latest.split('+', 1)[0]}"

        state.message = f"Installing torch {latest} from {index_url}"
        swap_ok, swap_output = _run_pip_install(
            python, target_spec, extras_dir, index_url,
            ["--force-reinstall", "--no-deps"],
        )
        state.attempts.append({
            "phase": "install",
            "ok": swap_ok,
            "output": swap_output[-2000:],
        })
        if not swap_ok:
            raise RuntimeError(
                f"pip install of torch {latest} failed. See attempts log "
                "for the pip output."
            )

        state.message = f"Resolving torch {latest} dependencies"
        dep_ok, dep_output = _run_pip_install(
            python, target_spec, extras_dir, index_url, [],
        )
        state.attempts.append({
            "phase": "deps",
            "ok": dep_ok,
            "output": dep_output[-2000:],
        })

        # Re-pin torch for any subsequent installs (rebuild step + any
        # later gpu-bundle re-runs). Non-fatal if it fails.
        constraint_path: Path | None = None
        try:
            constraint_path = _write_torch_constraint(extras_dir, latest)
            state.attempts.append({
                "phase": "constraint",
                "ok": True,
                "output": f"Pinned torch=={latest.split('+', 1)[0]} for subsequent installs",
            })
        except OSError as exc:
            state.attempts.append({
                "phase": "constraint",
                "ok": False,
                "output": f"Could not write torch constraint: {exc}",
            })

        # --- 4. Rebuild ABI-dependent packages ---------------------
        if rebuild_dependents and classification in ("minor", "major"):
            to_rebuild = _abi_dependents_present(extras_dir)
            for pkg in to_rebuild:
                state.message = f"Rebuilding {pkg} against torch {latest}"
                extra_flags = ["--force-reinstall", "--no-deps"]
                if constraint_path is not None:
                    extra_flags.extend(["--constraint", str(constraint_path)])
                ok, output = _run_pip_install(
                    python, pkg, extras_dir, None, extra_flags,
                )
                state.attempts.append({
                    "phase": "rebuild",
                    "package": pkg,
                    "ok": ok,
                    "output": output[-2000:],
                })
                if ok:
                    state.rebuilt_packages.append(pkg)
                # Don't abort on a single rebuild failure — torch itself
                # is the critical install. A missing bitsandbytes
                # downgrades FLUX speed but doesn't break basic
                # generation. Surface the failure in the attempts log.

        # --- 5. Verify + commit ------------------------------------
        state.phase = "verifying"
        state.message = "Verifying CUDA in a fresh subprocess"
        cuda_ok, detail = _verify_cuda(python, extras_dir)
        state.cuda_verified = cuda_ok
        state.attempts.append({
            "phase": "verify",
            "ok": cuda_ok,
            "output": detail[-2000:],
        })
        if not cuda_ok:
            raise RuntimeError(
                f"CUDA verification failed after upgrade. The new torch "
                f"{latest} cannot reach CUDA — restoring the previous "
                f"installation from rollback."
            )

        # Success — flush import caches + reset cached torch status so
        # the frontend banner re-probes on next poll.
        try:
            importlib.invalidate_caches()
        except Exception:  # noqa: BLE001 — best effort
            pass
        try:
            from backend_service.helpers.gpu import (
                reset_torch_status_cache,
                reset_vram_total_cache,
            )
            reset_vram_total_cache()
            reset_torch_status_cache()
        except Exception:  # noqa: BLE001 — best effort
            pass

        # Keep the most recent rollback as a safety net; reap older ones.
        cleaned = _cleanup_old_torch_rollbacks(extras_dir, keep=1)
        if cleaned:
            state.attempts.append({
                "phase": "cleanup",
                "ok": True,
                "output": f"Pruned older rollback dirs: {', '.join(cleaned)}",
            })

        state.phase = "done"
        state.done = True
        state.requires_restart = True
        state.finished_at = time.time()
        rebuilt_summary = (
            f" Rebuilt {len(state.rebuilt_packages)} ABI-dependent package(s)."
            if state.rebuilt_packages else ""
        )
        state.message = (
            f"Torch upgraded {current} -> {latest}.{rebuilt_summary} "
            "Restart the backend to activate."
        )

    except Exception as exc:  # noqa: BLE001 — surface any failure via status
        message = str(exc) or f"{type(exc).__name__} (no message attached)"
        # Try to restore the rollback if we got far enough to move files.
        if rollback is not None and rollback.is_dir():
            restored = _restore_torch_from_rollback(extras_dir, rollback)
            state.rolled_back = restored
            state.attempts.append({
                "phase": "rollback-restore",
                "ok": restored,
                "output": (
                    f"Restored previous torch from {rollback}"
                    if restored else
                    f"Rollback restore FAILED — rollback dir at {rollback} "
                    "kept on disk for manual recovery"
                ),
            })
        state.error = message
        state.phase = "error"
        state.message = message
        state.done = True
        state.finished_at = time.time()


@router.post("/api/setup/upgrade-torch")
def start_upgrade_torch(
    request: Request,
    body: _UpgradeTorchRequest | None = None,
) -> dict[str, Any]:
    """Kick off a background torch upgrade.

    Returns the current job state immediately. Poll
    ``/api/setup/upgrade-torch/status`` for progress. A second POST
    while a job is running returns the running job's state rather than
    starting a new one — same pattern as
    ``/api/setup/install-gpu-bundle``.
    """
    state_ce = request.app.state.chaosengine
    python = state_ce.runtime.capabilities.pythonExecutable
    extras = _extras_site_packages()
    if extras is None:
        raise HTTPException(
            status_code=500,
            detail=localized_detail(
                request,
                "Could not resolve the extras site-packages directory.",
            ),
        )
    if _is_apple_silicon():
        raise HTTPException(
            status_code=400,
            detail=localized_detail(
                request,
                "Torch upgrade is not applicable on Apple Silicon — the "
                "bundled MPS torch is managed by the app, not by pip.",
            ),
        )
    extras.mkdir(parents=True, exist_ok=True)
    rebuild = body.rebuildDependents if body is not None else True

    with _UPGRADE_LOCK:
        if _UPGRADE_JOB.phase in {"preflight", "upgrading", "verifying"}:
            return _UPGRADE_JOB.to_dict()
        _UPGRADE_JOB.id = f"torch-upgrade-{int(time.time() * 1000)}"
        _UPGRADE_JOB.phase = "preflight"
        _UPGRADE_JOB.message = "Starting torch upgrade"
        _UPGRADE_JOB.current_version = None
        _UPGRADE_JOB.target_version = None
        _UPGRADE_JOB.upgrade_type = None
        _UPGRADE_JOB.index_url = None
        _UPGRADE_JOB.rebuild_dependents = rebuild
        _UPGRADE_JOB.rebuilt_packages = []
        _UPGRADE_JOB.rolled_back = False
        _UPGRADE_JOB.rollback_path = None
        _UPGRADE_JOB.cuda_verified = None
        _UPGRADE_JOB.requires_restart = False
        _UPGRADE_JOB.error = None
        _UPGRADE_JOB.started_at = time.time()
        _UPGRADE_JOB.finished_at = 0.0
        _UPGRADE_JOB.attempts = []
        _UPGRADE_JOB.done = False

        thread = threading.Thread(
            target=_upgrade_torch_worker,
            args=(python, extras, rebuild),
            name="chaosengine-torch-upgrade",
            daemon=True,
        )
        thread.start()

    state_ce.add_log(
        "server", "info",
        f"Torch upgrade started (job={_UPGRADE_JOB.id}, target={extras}, "
        f"rebuildDependents={rebuild}, pythonVersion={_read_python_version(python)})",
    )
    return _UPGRADE_JOB.to_dict()


@router.get("/api/setup/upgrade-torch/status")
def upgrade_torch_status() -> dict[str, Any]:
    """Snapshot of the current torch upgrade job.

    Safe to poll at 1-2 Hz. Returns ``phase="idle"`` before any upgrade
    has been started in this backend session.
    """
    return _UPGRADE_JOB.to_dict()
