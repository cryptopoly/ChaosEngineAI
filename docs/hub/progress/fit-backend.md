---
title: Progress, fit-backend
status: done
---
# Progress: fit-backend

Updated: 2026-10-29. Owner: fit-backend agent. Branch: `feat/models-fit` (shared tree, no commits made).

## Goal

Backend half of "Best fit for your hardware": a `fit` dict on every chat, image and video catalog variant, a `bestFitVariantId` on every family, a verified Apple usable-memory figure, and the TypeScript types.

## Done

- `backend_service/helpers/fit.py` (new): `variant_fit`, `best_fit_variant_id`, `annotate_fit` (image / video families), `annotate_variants` (flat lists), `annotate_chat_families`, `local_kv_architecture`, `estimate_chat_tok_s`, `platform_gate`, `classify_share`. Thresholds 70 % / 92 % of `usable_gb`.
  - Chat need = file size + 10 % runtime share (the `/api/models/memory-check` share) + native KV cache at 8K context (capped by the model's window). Cache from the model's own layout when downloaded (`model_arch`), else a parameter-count table (never a Hub call). A row with its own `estimatedMemoryGb` replaces weights + runtime. Basis is always `estimated`.
  - Image / video need = `runtimeFootprintMpsGb` (Mac) / `runtimeFootprintCudaGb` (NVIDIA) / `runtimeFootprintCpuGb`, then `runtimeFootprintGb`, then `sizeGb x 1.3`. `measured` only on Apple Silicon and only for ids in `MEASURED_FOOTPRINTS` (4 rows whose catalog comment says the figure was measured: `Qwen/Qwen-Image-2.1-mflux`, `speridlabs/iris-3b-mflux`, `prince-canuma/LTX-2.3-distilled`, `dgrauet/ltx-2.3-mlx-q4`) or a catalog row with `footprintBasis: "measured"`.
  - Other-platform variants (mlx / mtplx / tensorfold / mflux / mlx-video / ltx-2-mlx off Apple Silicon; vllm / nunchaku / longlive off NVIDIA) are `unknown` with a reason, `needGb` and `headroomPct` null. Gate rules mirror `src/utils/platform.ts`.
  - Best fit: newest `releaseDate` among `fits`, then faster (image / video `estimatedGenerationSeconds`; chat measured tok/s from speed history, else the bandwidth estimate, asked only for tied variants), then smaller (`needGb`, `sizeGb`), then id. `null` when nothing fits.
- `backend_service/helpers/hardware_class.py`: Apple `usable_gb` now comes from one throwaway subprocess (`mx.device_info()`, falls back to `mx.metal.device_info()`), cached for the process on success and 5 minutes on failure, read on a background thread with a 1.5 s wait so a cold start never stalls a request. Used only when the dict describes this machine (memory matches within 0.6 GB); otherwise, and on any failure, `iogpu.wired_limit_mb` then the 0.75 / 0.67 heuristic. Intel Macs classify as `cpu`. New `host_hardware_class()` (60 s cache) for routes that have no system snapshot. Measured on the M4 Max 64 GB: `max_recommended_working_set_size` = 55 662 788 608 bytes = 51.84 GB, so `usable_gb` is 51.8 (the old heuristic said 48.0).
- Payload locations:
  - Chat: `_model_family_payloads` in `backend_service/helpers/model_family_payload.py` (feeds `featuredModels` in `/api/workspace` via `backend_service/state/payloads.py`, and `results` in `/api/models/search` via `backend_service/routes/models.py`; both pass `measured_speed_lookup(state)`).
  - Image: `_image_model_payloads` in `backend_service/helpers/images.py` (`/api/images/catalog` `families`), plus `latest` rows annotated in `backend_service/routes/images.py` (copies, the list is cached).
  - Video: `_video_model_payloads` in `backend_service/helpers/video.py` (`/api/video/catalog` `families`).
  - Image and video builders take an optional `hardware` argument; default is this machine.
- Types: `ModelFit`, `ModelFitStatus` in `src/types/models.ts`; `fit?` on `ModelVariant`, `ImageModelVariant`, `VideoModelVariant`; `bestFitVariantId?` on the three family types.
- Tests: `tests/test_model_fit.py`, 67 tests, all pass. Mutation checks done: `FITS_MAX_SHARE` 0.70 to 0.75 fails 2 boundary tests; dropping the "newest release" filter fails 3 best-fit tests; claiming `measured` on any hardware fails the measured-on-another-machine test. All restored.
- Also run (no failures): `tests/test_cache_strategies.py -k StartupImportPurity` (backend import still pulls no torch / diffusers / mlx), `tests/test_catalog_text_families.py`, `tests/test_video_routes.py -k Catalog`, `tests/test_backend_service.py -k workspace/search`, `tests/test_cli_smoke.py`, `npx tsc --noEmit`.

## Not done

- No i18n keys: `fit.reason` strings are English from the backend (the UI shows them as tooltips; localising them needs ids, not done).
- Not tested: Windows, Linux, a real NVIDIA card, an Intel Mac, RAM sizes other than this 64 GB Mac (other sizes use the heuristic, which is unit-tested but not compared with a real `device_info`), a packaged build (there `_resolve_mlx_python()` is the bundled interpreter; if it has no mlx the read fails and the heuristic applies).
- `iogpu.wired_limit_mb` path is covered by a patched test only (it is 0 on this Mac).

## Requests (for the lead)

1. Decision: the specified tie-break picks the smallest model in families that merge several sizes released together. On a 64 GB Mac `gemma-4` gets `google/gemma-4-E2B-it-qat-q4_0-gguf` (all Gemma 4 sizes are dated 2025-05, E2B is the fastest). One-line change in `best_fit_variant_id.order()` if you would rather prefer the larger model among fitting ones for chat.
2. `backend_service/helpers/platform_filter.py` does not list the `ltx-2-mlx (MLX native)` runtime, so the two new LTX-2.3 / 2.5 rows are still listed on Windows / Linux (fit marks them `unknown`, but they should be dropped like `mlx-video`). Add the marker (or `engine`) there.
3. Catalog data (not my files): most image rows have no `runtimeFootprint*` and fall back to `sizeGb x 1.3`, which over-states repos that ship several precisions (Z-Image-Turbo, 33 GB repo, shows `tight` at 42.9 GB). FU-123 records a measured 23.5 GB peak for FLUX 3 Klein (mflux) that is not in the catalog. If you add `footprintBasis: "measured"` to a catalog row, `fit.py` honours it (Apple only).
4. `headroomPct` is the share of usable memory left; it is positive for a `wont-fit` model between 92 % and 100 % and negative only above 100 %. The UI should not read the sign as the verdict.

## How to resume

1. `cd /Users/dan/ChaosEngineAI` and `git status`.
2. `.venv/bin/python -m pytest tests/test_model_fit.py -q` and `npx tsc --noEmit`.
3. Next step: none for this stream; the lead reviews the diff and commits.

## Decisions to propose

- Log: Apple `usable_gb` is the GPU's own working-set limit (51.8 GB on a 64 GB Mac), read once in a subprocess; heuristic only as a fallback.
- Log: `basis: "measured"` is claimed only on Apple Silicon and only for the four rows whose own catalog comment says measured.
