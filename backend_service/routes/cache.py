from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from backend_service.app import compute_cache_preview

router = APIRouter()

_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def _model_arch(request: Request, model_path: str | None, model_repo: str | None):
    """Real attention layout of a local model, or None to fall back to the estimate.

    ``model_path`` must sit inside an enabled model directory or the Hugging
    Face cache, so the endpoint never reads files elsewhere on disk.
    ``model_repo`` is resolved to its downloaded snapshot, if there is one.
    """
    from backend_service.helpers.hf_cache_paths import _hf_hub_cache_root, _hf_repo_snapshot_dir
    from backend_service.helpers.model_arch import hub_kv_architecture, read_kv_architecture

    if model_path:
        try:
            target = Path(os.path.expanduser(model_path)).resolve()
        except (OSError, RuntimeError):
            target = None
        if target is not None:
            roots: list[Path] = [_hf_hub_cache_root()]
            for directory in request.app.state.chaosengine.settings.get("modelDirectories") or []:
                raw = str(directory.get("path") or "")
                if directory.get("enabled", True) and raw:
                    roots.append(Path(os.path.expanduser(raw)))
            for root in roots:
                try:
                    resolved_root = root.resolve()
                except (OSError, RuntimeError):
                    continue
                if target == resolved_root or resolved_root in target.parents:
                    found = read_kv_architecture(model_path)
                    if found is not None:
                        return found
                    break
    if model_repo and _REPO_ID.fullmatch(model_repo):
        snapshot = _hf_repo_snapshot_dir(model_repo)
        found = read_kv_architecture(snapshot) if snapshot is not None else None
        # Not downloaded yet: read its config from the Hub, so the estimate is
        # exact while the user is still deciding whether to download.
        return found or hub_kv_architecture(model_repo)
    return None


@router.get("/api/cache/preview")
def cache_preview(
    request: Request,
    bits: int = Query(3, ge=0, le=8),
    fp16_layers: int = Query(4, ge=0, le=16),
    num_layers: int = Query(32, ge=1, le=160),
    num_heads: int = Query(32, ge=1, le=256),
    num_kv_heads: int = Query(0, ge=0, le=256),
    hidden_size: int = Query(4096, ge=256, le=32768),
    context_tokens: int = Query(8192, ge=256, le=4_194_304),
    params_b: float = Query(7.0, ge=0.5, le=1000.0),
    strategy: str = Query("native"),
    model_path: str | None = Query(None, max_length=4096),
    model_repo: str | None = Query(None, max_length=256),
    weights_gb: float | None = Query(None, gt=0, le=4000),
) -> dict[str, Any]:
    from backend_service.helpers.speed_history import measured_decode_tok_s

    state = request.app.state.chaosengine
    measured = measured_decode_tok_s(
        state.chat_sessions,
        state.benchmark_runs,
        model_path=model_path,
        model_repo=model_repo,
        strategy="native" if bits <= 0 else strategy,
    )
    return compute_cache_preview(
        bits=bits,
        fp16_layers=fp16_layers,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads or None,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        params_b=params_b,
        strategy=strategy,
        arch=_model_arch(request, model_path, model_repo),
        weights_gb=weights_gb,
        measured_tok_s=measured,
    )
