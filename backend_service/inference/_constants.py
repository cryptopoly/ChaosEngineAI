"""Shared inference constants.

Lives in its own module so submodules (jsonrpc, future engine splits)
can import without setting up a circular dependency on
``backend_service.inference``'s ``__init__``.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLX_TIMEOUT_SECONDS = 120.0
# Loading large MLX models (30B+) can take much longer than a normal request,
# especially on a first-time pull from Hugging Face. Allow a generous ceiling.
MLX_LOAD_TIMEOUT_SECONDS = 1800.0
DEFAULT_LLAMA_TIMEOUT_SECONDS = 120.0
CAPABILITY_CACHE_TTL_SECONDS = 10.0
