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
import inspect
import json
from pathlib import Path
from typing import Any

_VENDORED: dict[str, Path] = {
    # Aleph Alpha Kolibri 1 (78B MoE, German + English); mlx-lm PR #1945.
    "kolibri1": Path(__file__).with_name("kolibri1.py"),
    # Liquid AI LFM3 text models (LFM3-1.7B); not in mlx-lm 0.32.2.
    "lfm3": Path(__file__).with_name("lfm3.py"),
}


def _mlx_lm_ships(model_type: str) -> bool:
    try:
        return importlib.util.find_spec(f"mlx_lm.models.{model_type}") is not None
    except (ImportError, ValueError):
        return False


# Architectures stock mlx-lm does not ship but mlx-vlm does, so the worker loads
# them with ``mlx_vlm.load``. Several are text-only; ``needs_vlm_runtime`` only
# says which loader to use, not that the model sees images.
_VLM_ONLY_MODEL_TYPES = frozenset({
    "bonsai",
    "deepseek_v4",
    "flint",
    "glm5_next",
    "muse_glimmer",
    "muse_spark",
    "prism_hadamard_qwen35",
    "qwen4_exp",
})


def _mlx_vlm_ships(model_type: str) -> bool:
    try:
        return importlib.util.find_spec(f"mlx_vlm.models.{model_type}") is not None
    except (ImportError, ValueError):
        return False


def needs_vlm_runtime(model_path: str | Path) -> bool:
    """True when the checkpoint's architecture loads only through mlx-vlm.

    False for anything mlx-lm ships or we vendor, and when the installed
    mlx-vlm lacks the architecture, so those keep their current loader (and
    error message).
    """
    try:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if not isinstance(model_type, str) or model_type not in _VLM_ONLY_MODEL_TYPES:
        return False
    return not _mlx_lm_ships(model_type) and _mlx_vlm_ships(model_type)


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


def vendored_load_kwargs(model_path: str | Path, load_fn: Any) -> dict[str, Any]:
    """Extra keyword arguments for ``load_fn`` (``mlx_lm.load`` or ``load_model``).

    Empty unless the checkpoint needs a vendored architecture. mlx-lm 0.32 runs
    a ``model_file`` only with ``trust_remote_code=True``; the file is always our
    own vendored copy, because ``vendored_model_config`` refuses a checkpoint
    that names its own ``model_file`` and ``model_config`` overrides the config.
    """
    vendored = vendored_model_config(model_path)
    if not vendored:
        return {}
    kwargs: dict[str, Any] = {"model_config": vendored}
    try:
        parameters = inspect.signature(load_fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if "trust_remote_code" in parameters:
        kwargs["trust_remote_code"] = True
    # The tokenizer must not run the checkpoint's own code either. Its config
    # names an ``auto_map`` class (LFM3), and with trust_remote_code unset
    # transformers asks on stdin, which in the worker is the request pipe.
    # Refusing makes it fall back to the plain config, which is all the
    # tokenizer needs.
    if "tokenizer_config" in parameters:
        kwargs["tokenizer_config"] = {"trust_remote_code": False}
    return kwargs
