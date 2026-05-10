"""JSON IPC channel for the MLX worker subprocess.

The protocol channel is stdout (FD 1). ``mlx-lm`` + some diffusers /
torch paths print warnings to stdout as well — without isolation a
single ``[WARNING] Generating with a model that requires ...`` line
crashes the parent's ``json.loads`` and the user sees "MLX worker
returned invalid JSON".

``_install_stdio_redirect`` splits the channels: the JSON output goes
through a duplicate of FD 1 captured at startup, FD 1 itself is pointed
at stderr so anything writing through Python ``print`` / C-extension
writes / tqdm auto-detect lands on stderr instead. ``sys.stdout`` is
rebound to ``sys.stderr`` so libraries that cached a reference at import
time follow along.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. Re-exported from ``mlx_worker`` so existing
``from backend_service.mlx_worker import _emit`` etc. test patches keep
intercepting the worker's calls (the worker reads ``_emit`` through its
own re-exported name).
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any


# Default value keeps in-process tests working: they patch ``_emit``
# directly and never go through ``main()``.
_JSON_OUT: io.TextIOBase = sys.stdout  # type: ignore[assignment]


def _install_stdio_redirect() -> None:
    """Split the JSON protocol channel from warning chatter.

    The JSON protocol uses stdout (file descriptor 1). ``mlx-lm`` and some
    diffusers/torch paths print warnings and progress to stdout as well —
    without isolation, a single ``[WARNING] Generating with a model that
    requires ...`` line crashes the caller's ``json.loads`` and the user
    sees "MLX worker returned invalid JSON".

    Duplicate the original stdout FD into a fresh file object reserved for
    protocol output, then point FD 1 at stderr so anything writing through
    the normal stdout path (Python ``print()``, C-extension writes, tqdm
    auto-detecting stdout) lands on stderr instead. Finally rebind
    ``sys.stdout`` to ``sys.stderr`` so libraries that cached a reference
    at import time follow along.
    """
    global _JSON_OUT
    json_fd = os.dup(1)
    os.dup2(2, 1)
    _JSON_OUT = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr


def _emit(payload: dict[str, Any]) -> None:
    _JSON_OUT.write(json.dumps(payload) + "\n")
    _JSON_OUT.flush()


def emit_progress(phase: str, percent: float | None, message: str | None = None) -> None:
    try:
        _emit(
            {
                "ok": True,
                "progress": {
                    "phase": phase,
                    "percent": percent,
                    "message": message,
                },
            }
        )
    except Exception:
        pass
