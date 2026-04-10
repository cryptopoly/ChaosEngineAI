# Provenance Audit

Updated: 2026-04-10

## Status: Fully Original

All upstream-derived TurboQuant code has been removed from the repository. The following were deleted:

- `turboquant_mlx/` (entire package)
- `demo_7b.py`, `demo_mlx_lm.py`
- `tests/test_core.py`, `tests/test_metal.py`, `tests/test_fused_attn.py`

The remaining codebase — backend service, desktop app, inference engines, model management, image generation, and all UI — is original ChaosEngineAI work.

## Architecture

Models now run through native llama.cpp and mlx-lm runtimes. A pluggable cache strategy interface (`backend_service/cache_strategies/`) allows optional third-party compression backends (TriAttention, RotorQuant, MegaKernel) to be installed and selected at runtime without any upstream code being bundled in this repository.

## Compliance

- `LICENSE` — Apache-2.0 (project's own licence)
- `THIRD_PARTY_NOTICES.md` — documents optional third-party dependencies
- No upstream-derived source remains in-tree; attribution obligations are satisfied
