"""Model architectures that mlx-lm does not ship yet, loaded from files vendored here.

``mlx_lm.load`` accepts ``model_config={"model_file": <path>}`` and imports the
architecture (``Model`` / ``ModelArgs``) from that file instead of looking for
``mlx_lm.models.<model_type>``. ``vendored_model_config`` returns that argument
for a checkpoint whose ``model_type`` we carry, and ``None`` for everything else,
so every other model loads exactly as before. Once an mlx-lm release ships the
architecture the vendored file is no longer used and can be deleted.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

_VENDORED: dict[str, Path] = {
    # Aleph Alpha Kolibri 1 (78B MoE, German + English); mlx-lm PR #1945.
    "kolibri1": Path(__file__).with_name("kolibri1.py"),
}


def _mlx_lm_ships(model_type: str) -> bool:
    try:
        return importlib.util.find_spec(f"mlx_lm.models.{model_type}") is not None
    except (ImportError, ValueError):
        return False


def vendored_model_config(model_path: str | Path) -> dict[str, Any] | None:
    """``model_config`` for ``mlx_lm.load`` when the checkpoint needs a vendored architecture."""
    try:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or config.get("model_file"):
        return None
    model_type = config.get("model_type")
    source = _VENDORED.get(model_type) if isinstance(model_type, str) else None
    if source is None or _mlx_lm_ships(model_type):
        return None
    return {"model_file": str(source)}
