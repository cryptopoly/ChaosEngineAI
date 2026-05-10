"""Read-only introspection entrypoints for the MLX worker.

Three helpers that look at model / runtime metadata without spinning up
inference:

* ``_reject_unsupported_quant`` — peek at ``config.json`` and refuse models
  whose ``quantization_config.quant_algo`` MLX cannot consume (NVFP4 /
  NVINT4 today). Called inside ``WorkerState`` model load.
* ``probe`` — the ``probe`` CLI subcommand. Reports whether ``mlx`` /
  ``mlx_lm`` are importable, their versions, and ``mx.device_info()``.
* ``gguf_metadata`` — the ``gguf-metadata`` CLI subcommand. Cracks open a
  ``.gguf`` file with ``gguf.GGUFReader`` and emits the architecture +
  tokenizer + ``base_model_repos`` fields the catalog uses to resolve a
  HuggingFace tokenizer.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. Re-exported from ``mlx_worker`` so existing
``from backend_service.mlx_worker import probe`` / ``gguf_metadata``
imports + tests keep working.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from backend_service.mlx_worker_io import _emit


_UNSUPPORTED_QUANT_ALGOS = {"NVFP4", "NVINT4"}


def _reject_unsupported_quant(model_path: str) -> None:
    """Raise early if the model uses a quantisation format MLX cannot handle."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config") or {}
        algo = qcfg.get("quant_algo", "")
        method = qcfg.get("quant_method", "")
        if algo in _UNSUPPORTED_QUANT_ALGOS:
            raise RuntimeError(
                f"This model uses {algo} quantisation (via {method}) which "
                f"is not supported by the MLX runtime. Try a GGUF or "
                f"standard MLX quantised version of this model instead."
            )
    except RuntimeError:
        raise
    except Exception:
        pass  # Don't block loading if config can't be parsed


def probe() -> int:
    mlx_available = importlib.util.find_spec("mlx") is not None
    mlx_lm_available = importlib.util.find_spec("mlx_lm") is not None
    payload: dict[str, Any] = {
        "mlxAvailable": mlx_available,
        "mlxLmAvailable": mlx_lm_available,
        "mlxUsable": False,
        "mlxVersion": None,
        "mlxLmVersion": None,
        "message": None,
    }

    if not (mlx_available and mlx_lm_available):
        _emit(payload)
        return 0

    try:
        import mlx.core as mx
        import mlx_lm

        payload["mlxUsable"] = True
        payload["mlxVersion"] = getattr(mx, "__version__", None)
        payload["mlxLmVersion"] = getattr(mlx_lm, "__version__", None)
        try:
            payload["deviceInfo"] = mx.device_info()
        except Exception:
            payload["deviceInfo"] = None
        _emit(payload)
        return 0
    except Exception as exc:
        payload["message"] = str(exc)
        _emit(payload)
        return 1


def gguf_metadata(path: str) -> int:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1

    try:
        reader = GGUFReader(path, "r")
        base_model_repos: list[str] = []
        for key, field in reader.fields.items():
            if key.startswith("general.base_model.") and key.endswith(".repo_url"):
                value = field.contents()
                if isinstance(value, str):
                    base_model_repos.append(value)

        def normalize_repo(value: str | None) -> str | None:
            if not value:
                return None
            if value.startswith("https://huggingface.co/"):
                return value.removeprefix("https://huggingface.co/").strip("/")
            return value

        payload = {
            "path": str(Path(path).resolve()),
            "name": reader.get_field("general.name").contents() if reader.get_field("general.name") else Path(path).stem,
            "architecture": reader.get_field("general.architecture").contents() if reader.get_field("general.architecture") else None,
            "tokenizerModel": reader.get_field("tokenizer.ggml.model").contents() if reader.get_field("tokenizer.ggml.model") else None,
            "baseModelRepos": [normalize_repo(item) for item in base_model_repos if normalize_repo(item)],
            "baseModelRepo": normalize_repo(base_model_repos[0]) if base_model_repos else None,
        }
        _emit(payload)
        return 0
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1
