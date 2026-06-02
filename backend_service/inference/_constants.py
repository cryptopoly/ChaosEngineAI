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
# Native-backend capabilities (mlx/llama-server/vLLM/accelerator presence)
# only change when the user installs something — and every install path
# (pip / system pkg / cuda-torch / convert / the /api/setup/refresh-
# capabilities endpoint) calls refresh_capabilities(force=True), which
# invalidates this cache immediately. So the TTL only governs ambient
# staleness, not correctness. The old 10 s value was shorter than a single
# model load+generate (40-70 s), so load_model's refresh_capabilities()
# re-probed on *every* load — a blocking 17-31 s mlx_lm+mlx+mlx_vlm import
# subprocess each time (the creep behind the FU-068 probe-timeout bumps).
# 300 s comfortably spans back-to-back loads in a session while staying
# fresh enough for the capability UI; installs force-refresh regardless.
CAPABILITY_CACHE_TTL_SECONDS = 300.0
