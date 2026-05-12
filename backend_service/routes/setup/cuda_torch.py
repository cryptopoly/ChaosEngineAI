"""``install-cuda-torch`` endpoint — recovery path for CPU-only torch installs.

Walks the PyTorch download indexes (``cu124`` → ``cu126`` → ``cu128`` →
``cu121`` → nightly ``cu128``) reinstalling torch into the user-persistent
extras tree. The fresh-Windows-install case is Python 3.13 + system pip
which has no ``cu121`` wheel, so the broader walk is what actually unsticks
those users.

Extracted from ``routes/setup/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.i18n import localized_detail

from backend_service.routes.setup._install_helpers import (
    _CUDA_TORCH_INDEXES,
    _all_attempts_lack_wheel,
    _extras_site_packages,
    _purge_broken_distributions,
    _purge_stale_torch_from_extras,
    _read_python_version,
    _run_pip_install,
    _site_packages_for,
)

router = APIRouter()


@router.post("/api/setup/install-cuda-torch")
def install_cuda_torch(request: Request) -> dict[str, Any]:
    """Install a CUDA-enabled torch wheel into the backend runtime.

    The fresh-Windows-install case is Python 3.13 + system pip, which has
    no cu121 wheel at all — the install fails with "Could not find a
    version that satisfies the requirement torch". We try cu124 first
    (broadest Python 3.9-3.13 coverage), then cu126 / cu128 / cu121 in
    case the user's driver doesn't match the newest, and finally the
    nightly cu128 index for very-new Python (e.g. 3.14).

    Installs land in ``extras_dir`` (the user-persistent extras tree on
    PYTHONPATH), NOT the bundled venv. The venv on packaged builds lives
    under paths that need admin to write, and a venv install would be
    wiped on the next app upgrade anyway. Extras is user-writable and
    persists across upgrades — it's also where the gpu-bundle flow
    installs, so both recovery paths agree on torch's location.

    If every attempt fails with "No matching distribution", we set
    ``noWheelForPython`` in the response — that means the user's Python
    version is the problem, not the CUDA index, so the UI can tell them
    to switch Python rather than keep retrying. The response always
    includes ``pythonVersion`` so the UI can show which interpreter this
    is targeting (important: it's the app's bundled venv, not the system
    pip the user might reach from a shell).

    Torch already imported in this process stays CPU until the user
    restarts the backend — we flag ``requiresRestart`` in the response
    so the frontend can prompt appropriately.
    """
    state = request.app.state.chaosengine
    python = state.runtime.capabilities.pythonExecutable
    python_version = _read_python_version(python)

    extras_dir = _extras_site_packages()
    if extras_dir is None:
        raise HTTPException(
            status_code=500,
            detail=localized_detail(request, "Could not resolve the extras site-packages directory."),
        )
    extras_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale torch + nvidia-* runtime deps from extras first. A
    # prior half-installed wheel (dist-info without a matching package
    # folder, or vice versa) causes ``import torch`` to raise at runtime
    # with confusing "No module named torch" messages.
    purged_torch: list[str] = []
    try:
        purged_torch = _purge_stale_torch_from_extras(extras_dir)
    except OSError as exc:
        state.add_log("server", "warning", f"Could not purge stale torch from extras: {exc}")
    if purged_torch:
        state.add_log(
            "server", "info",
            f"Purged stale torch files from extras ({len(purged_torch)} entries)",
        )

    # Sweep pip's "~<pkg>" stub directories from the bundled site-packages.
    # These are left behind by a prior interrupted install (common on Windows
    # where Defender briefly locks .pyd files) and cause noisy "Ignoring
    # invalid distribution" warnings in future pip runs.
    site_packages = _site_packages_for(python)
    purged_stubs: list[str] = []
    if site_packages is not None:
        purged_stubs = _purge_broken_distributions(site_packages)
        if purged_stubs:
            state.add_log(
                "server", "info",
                f"Removed {len(purged_stubs)} broken pip stub(s) from {site_packages}: {', '.join(purged_stubs)}",
            )

    attempts: list[dict[str, Any]] = []
    ok = False
    winning_output = ""
    winning_index: str | None = None

    for index_url in _CUDA_TORCH_INDEXES:
        state.add_log("server", "info", f"Installing CUDA torch from {index_url}")
        # Two-pass install into extras (via --target in _run_pip_install):
        #   Pass 1: --force-reinstall --no-deps swaps the torch wheel even
        #           when a same-versioned CPU wheel is already present
        #           (PEP 440 treats 2.6.0+cpu == 2.6.0+cu124 as equal for
        #           upgrade purposes, so --force-reinstall is required).
        #   Pass 2: plain install (no --force) fills transitive deps like
        #           nvidia-cublas-cu12 without clobbering files held by
        #           the running backend process.
        swap_ok, swap_output = _run_pip_install(
            python, "torch>=2.4.0", extras_dir, index_url,
            ["--force-reinstall", "--no-deps"],
        )
        combined_output = swap_output
        if swap_ok:
            _dep_ok, dep_output = _run_pip_install(
                python, "torch>=2.4.0", extras_dir, index_url, [],
            )
            if dep_output:
                combined_output = f"{swap_output}\n\n--- deps pass ---\n{dep_output}"

        attempts.append({
            "indexUrl": index_url,
            "ok": swap_ok,
            "output": combined_output,
        })
        if swap_ok:
            ok = True
            winning_output = combined_output
            winning_index = index_url
            break

    # Re-probe so the UI can refresh its capabilities view. Note: torch
    # already imported in this process is still the old module — the
    # live cuda check won't flip to True without a restart.
    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    no_wheel_for_python = (not ok) and _all_attempts_lack_wheel(attempts)
    state.add_log(
        "server", "info" if ok else "error",
        f"CUDA torch install: {'succeeded via ' + winning_index if ok else 'failed after all candidates'}"
        + (f" (no wheel for Python {python_version})" if no_wheel_for_python and python_version else ""),
    )
    return {
        "ok": ok,
        "output": winning_output or (attempts[-1]["output"] if attempts else ""),
        "indexUrl": winning_index,
        "attempts": attempts,
        "requiresRestart": ok,
        "pythonExecutable": python,
        "pythonVersion": python_version,
        "noWheelForPython": no_wheel_for_python,
        "targetDir": str(extras_dir),
        "capabilities": caps,
    }
