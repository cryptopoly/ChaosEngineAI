---
title: "Progress: limits"
status: done (backend); frontend follow-ups listed under Requests
---
# Progress: limits (per-model size limits)

Updated: 2026-10-29. Owner: `limits` engineer (Sonnet). Branch `feat/models-fit`, main checkout, no commits (the lead commits).

## Goal

Replace the global 2048 px per-side cap on image and video sizes with per-model limits (`maxSide`, `maxPixels`, `nativeSide`), keep a hard ceiling of 8192, and make the memory gate charge for pixels.

## Done

- `backend_service/helpers/generation_limits.py`: `HARD_MAX_SIDE = 8192`, `DEFAULT_MAX_SIDE = 4096`, `size_limits(variant)` (-> `SizeLimits`), `size_limit_fields`, `reference_pixels`, `reference_frames`, `parse_resolution`. `normalize_image_size` / `normalize_video_size` take `max_side`, `max_pixels`, `model_name`; over a limit both edges scale together (aspect ratio survives) and round down; the snap logic is unchanged; the note names the model ("Stable Diffusion XL Base 1.0 allows at most 2048 px per side"). A snap-only note is the old text.
- `backend_service/catalog/size_limits.py` (new): the table (family, variant and tracked-seed rows) with the source of every figure; applied at the end of `image_models.py` and `video_models.py`, so every variant and tracked seed carries the three fields. A value written on a catalog variant wins (`setdefault`).
- `backend_service/app.py`: `_apply_min_side(..., max_side)` uses the model limit; new `_resolve_image_size(request, variant)` (draft mode, minimum edge, limits) used by `_generate_image_artifacts` and by the gate; video artifact passes the variant limits.
- `backend_service/models/__init__.py`: image and video `width` / `height` `le=` is `HARD_MAX_SIDE` (8192).
- Payloads: `helpers/images.py` (curated spread, both seed->variant builders, tracked-seed payloads, live "latest" lane) and `helpers/video.py` add `**size_limit_fields(...)`; the keys are always present (defaults if a row has none).
- `backend_service/helpers/memory_gate.py`: `size_extra_gb`, `gate_image_generation` / `gate_video_generation` take `width`, `height`, `reference_pixels`, `footprint_gb` (video also `frames`, `reference_frames`). Extra memory = `0.15 x footprint x (megapixels above the reference resolution)`, x frame ratio for a clip longer than the default; 0 at or below the reference, so estimates at or below native do not change. New refusal codes `memory_gate_image_size_too_large` / `memory_gate_video_size_too_large`; the message gives the size, the GB needed, what is free and a size that fits. Order: model too large, size too large, low available, pressure.
- Routes: `routes/images.py` gates the size that will render (`_resolve_image_size`, so a 4096 request to SDXL is judged as 2048); `routes/video.py` `_video_memory_gate(..., body)` does the same through `normalize_video_size` / `align_video_frames`.
- `scripts/measure-image-ceiling.py` (new, not run): one image per 1x / 1.5x / 2x / 3x / 4x native side, 1 step; wall time, peak process-tree memory (macOS `footprint`, else psutil RSS; NVIDIA GPU memory too); stops at the first refusal, failure, clamp or catalog limit; writes JSON + Markdown to `~/.chaosengine/test-results/`.
- `tests/test_size_limits.py` (new, 61 tests): catalog audit, limits reader, image and video limits, snapping, notes, ceiling, request-model bounds, gate (small vs big machine, native unchanged, suggested size fits), image route gate, video gate helper, the measuring script's pure helpers. Mutation checks: ignoring `max_side` fails 5 tests; zeroing the gate's share fails the gate and route tests.
- `tests/test_image_runtime.py`: one line changed (`test_extreme_aspect_ratios_are_capped` passes `max_side=2048`), because it asserted the retired global cap.

## Values and sources

`maxSide` / `maxPixels` / `nativeSide`. "unverified" = the model card and upstream docs state no limit; 4096 is the generic default for a modern DiT model.

| Family | maxSide | maxPixels | nativeSide | Source |
|---|---|---|---|---|
| FLUX.1 schnell, dev (all 14 rows) | 4096 | 16.8 M | 1024 | unverified: HF cards and BFL's resolution page do not cover FLUX.1 |
| SD 3.5 Medium | 2048 | 2,073,600 | 1024 | HF card: progressive training 256 -> 1440; Stability launch notes: 0.25 to 2 MP |
| SD 3.5 Large Turbo | 2048 | 2,097,152 | 1024 | unverified: card silent; Stability rates SD 3.5 Large at 1 MP; held at the SDXL-class edge |
| SDXL base | 2048 | 4.19 M | 1024 | SDXL-class limit from the plan; card states no maximum |
| Qwen-Image, Lightning | 4096 | 16.8 M | 1328 | native from the HF card (1328x1328, up to 1664x928); maximum unverified |
| Qwen-Image-2.1 (mflux, q8, Turbo, sd.cpp) | 4096 | 16.8 M | 2048 | HF cards: 2048x2048 is the 1:1 size, presets to 2752x1536 (~4.3 MP); maximum unverified |
| Z-Image Turbo | 4096 | 16.8 M | 1024 | unverified: card shows 1024x1024 only |
| Iris-3B Turbo | 2048 | 4.19 M | 1024 | card: "about one megapixel; for larger prints, use an upscaler"; no VAE; maximum unverified |
| FLUX.2 (klein 4B, mflux, dev) | 4096 | 4,194,304 | 1024 | BFL help center, FLUX.2: 4 MP maximum, multiples of 16, recommends <= 2 MP, default 1024x1024. The page names the FLUX.2 family, not klein or dev |
| FLUX 3 Image Klein (+ Base) | 4096 | 16.8 M | 1024 | unverified: card gives only the 1024x1024 benchmark |
| LTX-Video 0.9.x | 1280 | 921,600 | 768 | HF card: divisible by 32, "works best on resolutions under 720 x 1280" |
| LTX-2 / 2.3 / 2.5 (6 rows) | 4096 | 9,437,184 | 768 | unverified maximum: HF card says divisible by 32 and names none; ltx-2-mlx README shows 1280x720 and 1920x1024 runs; measured here 1280x704 x 121 frames = 18 GB. 4096x2304 chosen as the 4K frame |
| Wan 2.1 (1.3B, 14B, GGUF, CausVid), Wan 2.2 A14B (+ distills) | 1280 | 921,600 | 832 | Wan-Video READMEs: 14B and A14B support 480P and 720P; the 1.3B is "capable" of 720P with less stable results, 480P recommended |
| HunyuanVideo | 1280 | 921,600 | 1280 | Tencent README: supported settings 720px1280px129f and 544px960px129f (60 GB / 45 GB minimum GPU memory) |
| Mochi 1 preview | 848 | 407,040 | 848 | HF card: "generates videos at 480p today"; example 848x480 |
| CogVideoX 2B, 5B | 720 | 345,600 | 720 | HF card: "720 x 480, no support for other resolutions" |
| CogVideoX 1.5 5B | 1360 | 1,044,480 | 1360 | HF card: T2V 1360x768; I2V shorter edge 768, longer 768 to 1360 |
| LongLive (hidden family) | 832 | 399,360 | 832 | our own catalog note (832x480); unverified |
| Tracked seeds | FLUX.2 rows 4 MP; Qwen-Image rows nativeSide 1328; Sana Sprint 2048; the rest 4096 | | | as the matching family; the rest unverified |

Video `nativeSide` is the long edge of the native frame (832 for 832x480); image `nativeSide` is the square side.

## Design notes

- **Memory-gate baseline is the recommended resolution, not `nativeSide`.** Catalog footprints were measured or estimated at `recommendedResolution` (1024x1024 for Qwen-Image-2.1 although its card's native size is 2048). `reference_pixels(variant)` is that area (falls back to `nativeSide` squared). Extra memory is charged only above it.
- **The gate is in RAM terms** (psutil), as before. On a CUDA box with plenty of RAM and 24 GB VRAM it cannot see a VRAM overrun; the Studio's own warning (`src/utils/images.ts`, already pixel-aware) covers VRAM.
- **The coefficient (0.15 of the footprint per extra megapixel) is the Studio's own, not a new measurement.** It is an estimate. Cross-check: the measured LTX-2.3 q4 run, 1280x704 x 121 frames, peaked at 18 GB against a 20 GB footprint; the model charges about 2 GB extra. `scripts/measure-image-ceiling.py` produces the numbers to tune it.
- Limits are enforced by the guard (the size is changed and the note says so), not by a 4xx, as for the old cap.

## Not done / not tested

- No real generation at any large size was run (rule: nobody but `settings-rec` generates). The 4096 figures are unverified ceilings, protected by the memory gate and the 8192 hard ceiling; the measuring script exists to replace them with numbers.
- The video gate was only exercised with fakes; no clip was rendered.
- Windows and Linux were not run (the code is platform-neutral Python; the script's `footprint` path is macOS-only, psutil otherwise).
- LTX-2.5 (gated) has the LTX-2 row.

## Requests

For `settings-rec` (owns the two Studio TSX files) and whoever owns the hooks:

- `src/features/images/ImageStudioTab.tsx` lines ~962 and ~974 and `src/features/video/VideoStudioTab.tsx` lines ~1065 and ~1084 still have `max={2048}`. Use the selected variant's `maxSide` (payload field, always present; `src/types/image.ts` already has it, `src/types/video.ts` needs `maxSide`, `maxPixels`, `nativeSide` too; add `maxPixels` and `nativeSide` to the image type). Also show `maxPixels` when it is below `maxSide` squared (SD 3.5 Medium, FLUX.2, LTX-Video, Wan, Hunyuan, Mochi, CogVideoX): a 2048x2048 request is reduced and the artifact's `runtimeNote` says why.
- `src/hooks/useVideoState.ts` lines ~600-603 (`Math.min(2048, ...)`) and `src/hooks/videoStateDefaults.ts` line ~72 (`width > 2048` rejects a saved size) clamp the video size at 2048; they should clamp to the variant's `maxSide` (or 8192).
- `src/utils/images.ts` / `src/utils/videos.ts` (the safety warning) already scale with pixels; no change needed.

For the lead:

- CLAUDE.md "new feature gate": an E2E check in `scripts/e2e_test_suite.py` is expected for this feature (suggested: POST `/api/images/generate` with a 4096x4096 request to SDXL and assert the artifact's `runtimeNote` names the 2048 limit; needs a model on disk, so I did not add it).
- FU row text for CLAUDE.md: the global 2048 cap is gone; per-model `maxSide` / `maxPixels` / `nativeSide` in `catalog/size_limits.py`; hard ceiling 8192; gate term in `memory_gate.py`; open items = measure with `scripts/measure-image-ceiling.py`, replace the "unverified" rows.
- `tests/test_image_runtime.py::MinSideTests::test_extreme_aspect_ratios_are_capped` was edited by one line (see above).

## New i18n keys

None. The gate and size-note messages are backend English strings (`localized_detail` wraps the gate's), like the existing ones.

## How to resume

1. `cd /Users/dan/ChaosEngineAI && git status`.
2. `.venv/bin/python -m pytest tests/test_size_limits.py tests/test_memory_gate.py tests/test_request_guards.py -q`.
3. When the GPU is free: `.venv/bin/python scripts/measure-image-ceiling.py --model <variant id>` per model and replace the unverified rows in `backend_service/catalog/size_limits.py` with what ran.

## Decisions to propose

- Hard-clamping the video models that document one resolution (CogVideoX 2B/5B 720x480, Mochi 480p) and the documented-range ones (LTX-Video, Wan, Hunyuan) at their card figures is stricter than the old 2048 cap, deliberately: the card says larger is unsupported or unstable. Recommendation: keep; relax a row in `size_limits.py` if a measured run shows the model is fine larger.
- LTX-2 family at 4096 / 4096x2304 is unverified; the gate is what protects a small machine. Recommendation: run the measuring script for the video model once a video equivalent exists, or lower to 2048 until then if 4K is not wanted.
