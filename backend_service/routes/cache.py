from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from backend_service.app import _build_system_snapshot, compute_cache_preview

router = APIRouter()


@router.get("/api/cache/preview")
def cache_preview(
    bits: int = Query(3, ge=0, le=8),
    fp16_layers: int = Query(4, ge=0, le=16),
    num_layers: int = Query(32, ge=1, le=160),
    num_heads: int = Query(32, ge=1, le=256),
    hidden_size: int = Query(4096, ge=256, le=32768),
    context_tokens: int = Query(8192, ge=256, le=262144),
    params_b: float = Query(7.0, ge=0.5, le=1000.0),
    strategy: str = Query("native"),
) -> dict[str, Any]:
    system_stats = _build_system_snapshot()
    return compute_cache_preview(
        bits=bits,
        fp16_layers=fp16_layers,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        params_b=params_b,
        system_stats=system_stats,
        strategy=strategy,
    )
