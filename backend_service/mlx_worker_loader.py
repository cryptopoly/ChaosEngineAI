"""Model snapshot resolution for the MLX worker.

``resolve_local_snapshot`` lifts the Hugging Face snapshot-download
front half out of ``WorkerState.load_model``. It accepts the raw
request target (HF repo id or local path), pre-downloads the
snapshot when the target isn't already on disk, and streams the byte
counts through ``emit_progress`` so the desktop UI can render a real
progress bar instead of staring at a frozen 60% during multi-GB
fetches.

Errors are translated into ``RuntimeError`` with user-readable hints
for the gated/404/auth paths so the UI's modal can show a one-line
fix ("accept the licence", "set HF_TOKEN") instead of an opaque
``HfHubHTTPError`` traceback.

Extracted from ``backend_service/mlx_worker.py`` as part of the
v0.8.0 refactor.
"""

from __future__ import annotations

from pathlib import Path

from backend_service.mlx_worker_io import emit_progress


def resolve_local_snapshot(target: str) -> str:
    """Return a local filesystem path for ``target``.

    If ``target`` already looks local (absolute path / ``~`` expansion
    / existing directory) the path is normalised and returned as-is.
    Otherwise the function calls ``huggingface_hub.snapshot_download``
    with a progress-tqdm subclass that pushes byte-count updates
    through ``emit_progress``. When ``huggingface_hub`` / ``tqdm``
    isn't installed the original target is passed through so
    ``mlx_lm.load`` can still resolve it itself.
    """
    try:
        candidate = Path(target).expanduser()
        if target.startswith("/") or target.startswith("~") or candidate.exists():
            return str(candidate)
    except Exception:
        pass

    try:
        from huggingface_hub import snapshot_download  # type: ignore
        from huggingface_hub.utils import (  # type: ignore
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
        from tqdm import tqdm  # type: ignore
    except ImportError:
        # huggingface_hub / tqdm not installed — let mlx_lm.load
        # handle resolution itself. Matches pre-progress behaviour.
        return target

    class ProgressTqdm(tqdm):  # type: ignore[misc]
        def update(self, n: int = 1):  # type: ignore[override]
            result = super().update(n)
            try:
                total = float(self.total or 0)
                done = float(self.n or 0)
                if total > 0:
                    frac = max(0.0, min(1.0, done / total))
                    pct = 20.0 + frac * 40.0  # 20% -> 60%
                    done_mb = int(done // (1024 * 1024))
                    total_mb = int(total // (1024 * 1024))
                    emit_progress(
                        "downloading",
                        pct,
                        f"{done_mb} / {total_mb} MB",
                    )
                else:
                    emit_progress("downloading", 20.0, "Fetching weights")
            except Exception:
                pass
            return result

    emit_progress("downloading", 20.0, "Fetching weights from Hugging Face")
    try:
        # max_workers=1 avoids multiprocessing semaphore leaks on macOS
        # that crash the worker subprocess.
        return snapshot_download(
            repo_id=target,
            tqdm_class=ProgressTqdm,
            max_workers=1,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"This model is gated on Hugging Face. Accept the licence "
            f"at https://huggingface.co/{target} and set HF_TOKEN in "
            f"Settings, then retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"Hugging Face repository not found: {target}"
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError(
                f"Hugging Face refused access to {target} (HTTP {status}). "
                f"Set HF_TOKEN in Settings and make sure you have accepted "
                f"the licence at https://huggingface.co/{target}."
            ) from exc
        raise RuntimeError(
            f"Hugging Face download failed for {target}: {exc}"
        ) from exc
    except OSError as exc:
        # Network / filesystem failures — bubble up the detail.
        raise RuntimeError(
            f"Could not download {target} from Hugging Face: {exc}"
        ) from exc
