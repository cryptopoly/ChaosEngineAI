---
title: "Progress: settings-rec"
status: done (open items below)
---
# Progress: settings-rec (recommended Studio settings)

Updated: 2026-10-29 20:10. Owner: settings-rec agent (Opus). Branch: `feat/models-fit`, main checkout `/Users/dan/ChaosEngineAI` (no worktree, no commits; the lead commits). Scratch: `$S = /private/tmp/claude-501/-Users-dan-ChaosEngineAI/a873d763-936c-4bd0-9d78-5888e9aee5a6/scratchpad/settings-rec`.

## Goal

"Recommended for this machine" panel in Image Studio and Video Studio: per selected model, the settings that work best on this hardware, one click to apply, each with a reason and a measured / estimated label. Studio size inputs read the variant's `maxSide`.

## Done

- Table data (entries + measured runs) in `backend_service/catalog/recommended_settings.py` (new); logic in `backend_service/helpers/recommendations.py`, which re-exports `_ENTRIES` / `_MEASURED`. Logic: `accepted_settings` (what each engine reads, mirrors the Studio gating), the table `_ENTRIES` (keyed by variant id and hardware: `*`, `apple`/`cuda`/`cpu`, a bucket, or `below_usable_gb`), the generic rule from catalog `defaultSteps` / `cfgOverride`, `_MEASURED` (bucket + chip must match for `measured`), `recommended_settings`, `annotate_recommendations`, `host_chip`. Attached in `helpers/images.py` and `helpers/video.py` (one import + the return line, after `annotate_fit`). Uses `host_hardware_class()` from fit-backend.
- Frontend: `src/types/recommendations.ts` (new), `recommendedSettings?`, `maxSide?`, `maxPixels?`, `nativeSide?`, `minSide?` on `ImageModelVariant` / `VideoModelVariant`; `src/components/recommendedSettings.ts` (pure row / apply / clamp logic), `src/components/RecommendedSettingsPanel.tsx`; wired into `ImageStudioTab.tsx` and `VideoStudioTab.tsx`. Width / height inputs and blur clamps use `maxSide` (4096 default) and, for images, `minSide`; a size over `maxPixels` shows a note that it will be reduced (limits request).
- Image Studio now hides the sampler, diffusion cache, CFG decay, preview VAE and FP8 controls for mflux and stable-diffusion.cpp variants (those engines ignore them; same rule the Video Studio already applies to ltx-2-mlx).
- Live UI check (vite 5180 against my backend 8890, browser pane at 1400x1000): Image Studio with Qwen-Image-2.1 (steps 40 measured, Apply sets the Steps field 24 -> 40, button becomes "All in use"), Iris-3B (width input min 768 / max 2048 from the payload; typing 500 clamps to 768 on blur, 3000 to 2048; per-row "Use" changes only steps; Apply sets 1024 x 1024), diffusers-only controls hidden for mflux. Video Studio with LTX-2.3 (size / frames / fps measured; Apply sets frames 33 -> 97, clip length line updates) and LTX-2.5 (estimated chips). No React errors in the console; only the expected missing-i18n-key logs. Screenshots: `$S/screens/image-studio-qwen21.jpg`, `image-studio-iris-before-apply.jpg`, `video-studio-ltx23-before-apply.jpg`, `video-studio-ltx23-after-apply.jpg`, `video-studio-ltx25-estimated.png`. Servers stopped, GPU lock released.
- Tests: `tests/test_recommendations.py` (23), `src/components/__tests__/recommendedSettings.test.tsx` (13). Mutation checks: dropping the engine filter in `recommended_settings` fails 2 tests + many subtests; dropping the setter check in `recommendationRows` fails 2 vitest cases. Both restored.

## Measurements (M4 Max, 64 GB, own backend on 8890, GPU lock held)

Command: `python3 $S/measure.py $S/plan_<name>.json $S/out_<name>` (posts to `/api/images/generate` or `/api/video/generate`, samples the engine process with `footprint` every 2 s). Prompts (fixed): p1 "A red fox sitting in fresh snow at sunrise, soft golden light, detailed fur, 50mm photograph" seed 11; p2 "A neon sign that reads OPEN ALL NIGHT above a small diner on a rainy city street at night, reflections on the wet road" seed 22; p3 "A ceramic teapot and two cups on a wooden table beside a window, still life, soft natural light" seed 33. Wall = client wall time incl. HTTP; first run of a model includes the load. Footprint = peak process physical footprint (more than mflux's own "peak MLX memory").

| Model (engine) | Config | Wall s (p1 / p2 / p3) | Peak footprint | Quality check |
|---|---|---|---|---|
| Iris-3B Turbo (mflux) | 4 steps, 1024 (= Studio default after backend substitution) | 14.6 / 12.6 / 12.6 | 12-13 GB | good; p3 shows 3 cups (prompt says 2) |
| Iris-3B Turbo | 4 steps, 768 | 8.4 / 8.4 / 8.4 | 13 GB | different compositions, not better; p1 has a stray watermark-like text |
| FLUX 3 Klein (mflux) | 16 steps, 1024 (default) | 70.0 / 69.7 / 78.3 | 14-15 GB | good, sign text correct |
| FLUX 3 Klein | **8 steps**, 1024 | 40.1 / 40.1 / 40.4 | 15 GB | same quality at full-res crops (fur detail, sign text); p3 cups stacked in both |
| Qwen-Image-2.1 Turbo (mflux) | 8 steps, 1024 (default) | 75.0 / 72.5 / 72.0 | 46-50 GB | good, sign legible |
| Qwen-Image-2.1 Turbo | 6 steps, 1024 | 59.9 / 56.9 / 55.7 | 46-50 GB | very close; not recommended (model tuned for 8, saving 18 %) |
| Qwen-Image-2.1 (mflux, bf16) | 40 steps, 1024 (default) | 271.0 / 258.3 (p1 / p2) | 47-49 GB | good; p2 sign partly garbled ("OPEN" plus extra mangled signs) |
| Qwen-Image-2.1 | 28 steps, 1024 | 191.3 / 186.5 | 50 GB | p1 near-identical; p2 different composition with a cleaner "OPEN ALL NIGHT"; not recommended over 40 (1.4x, two prompts), stated in the reason |
| LTX-2.3 q4 (ltx-2-mlx) | 768x512, 33 frames, 24 fps (Studio default) | 57.0 / 55.1 (v1 / v2) | 14 GB | coherent 1.4 s clips with stereo sound; v1 fox tracked alongside, motion blur on the last frame |
| LTX-2.3 q4 | **768x512, 97 frames**, 24 fps | 137.9 / 135.4 | 15 GB | coherent 4 s clips with sound; v1 fox seen from behind running away, v2 stable sunset beach with a slow pan |

Video prompts: v1 "a red fox running through fresh snow at sunrise, soft golden light, the camera tracks alongside" seed 11; v2 "waves rolling onto a rocky beach at sunset, seagulls overhead, gentle wind, the camera slowly pans right" seed 22. The ltx-2-mlx distilled pipeline runs 8+3 passes (the artifact reports 11 steps). Frame tiles: `$S/out_video/*-tiles.png`, `cmp-v1.png`, `cmp-v2.png`.

Rule used: deviate from the model maker's step count only when the measured saving is large (about 1.5x or more) and the side-by-side and full-resolution crops show no visible difference. Sheets: `$S/sheet-*.png`, crops `$S/crop-*.png`, images `$S/out_*/`.

## Not done

- Diffusers families (FLUX.1, SD3.5, SDXL, Qwen-Image, Z-Image, FLUX.2 klein diffusers, Wan, LTX-Video, Hunyuan, Mochi, CogVideoX): none downloaded, so steps / guidance / sampler / CFG decay / FP8 values come from model cards and the repo's tables and are all `estimated`. Worth timing next: FLUX.2 klein 4B (diffusers) or Wan 2.1 1.3B with and without First Block Cache.
- Not run: Windows, Linux, any CUDA card, any Mac other than this M4 Max 64 GB (16 / 24 / 32 / 96 / 128 GB buckets get the same table values, labelled estimated), Qwen-Image-2.1 8-bit variant, FLUX 3 Klein Base.
- Fewer prompts than ideal: 3 per image config, 2 per Qwen base and LTX config; the quality check is my eye on side-by-sides and full-resolution crops, plus a mean-difference / Laplacian number from `$S/tools/sheet.py`, not a metric like CLIP score.
- No diffusers DiT is downloaded, so no cache strategy (FBCache / TeaCache) was timed: every cache recommendation is `estimated`.

## How to resume

1. `cd /Users/dan/ChaosEngineAI && git status`.
2. Own backend: `CHAOSENGINE_PORT=8890 CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:5180 .venv/bin/python -m backend_service.app`.
3. Tests: `.venv/bin/python -m pytest tests/test_recommendations.py -q`; `npx vitest run src/components/__tests__/recommendedSettings.test.tsx`; `npx tsc --noEmit`.
4. Results: `cat $S/out_*/results.jsonl`.

## Requests

- **Conflict to fix (not mine, `src/utils/videos.ts` `assessVideoGenerationSafety`):** after "Apply recommended" on LTX-2.3 (768x512, 97 frames) the Video Studio shows "Heads up — may struggle on this device: ... around 42 GB of peak memory (model = 20 GB + attention = 22 GB)" and offers "Use safer settings (768x512 · 33 frames)". Measured peak for that run is 15 GB (sampled process footprint, 2 clips; the lead measured 18 GB at 257 frames). The attention term does not fit the two-stage ltx-2-mlx pipeline; skip it (or use the measured footprint) for ltx-2-mlx variants. Screenshot: `$S/screens/video-studio-ltx23-after-apply.jpg`. It is a caution, not a block.
- `tests/test_model_fit.py` and `tests/test_size_limits.py` call `_image_model_payloads` / `_video_model_payloads`, which now also run `annotate_recommendations` (adds `recommendedSettings`, reads the chip once via `sysctl`). I did not run those suites (not mine); please include them in the full run.

- **Contract extension (tell the lead):** `recommendedSettings` also carries `hardwareLabel`, `basisByKey` (per-setting basis) and `measuredOn`, and `settings` may hold `guidance`, `width`, `height`, `numFrames`, `fps`, `sampler`, `fp8LayerwiseCasting` beyond the plan's six keys. All optional, additive.
- `limits`: I added `maxSide?`, `maxPixels?`, `nativeSide?` and `minSide?` to `ImageModelVariant` / `VideoModelVariant` in `src/types/image.ts` / `video.ts` (task step 5). If you add them too, keep one copy.
- Catalog (`catalog/image_models.py`, not mine): Qwen-Image-2.1 mflux now renders 1024 px / 40 steps in 258-271 s on the M4 Max (mflux 0.24), not the 340 s in `estimatedGenerationSeconds` / the note's "about 6 minutes"; Qwen-Image-2.1 Turbo 72-75 s (catalog 75); FLUX 3 Klein 70-78 s at 16 steps (no catalog estimate); Iris 11-13 s at 1024 (catalog 18). LTX-2.3 q4 97 frames 135-138 s (catalog 155 incl. load).
- E2E (CLAUDE.md feature gate; `scripts/e2e_test_suite.py` is not mine): suggested check, phase with the catalog calls: `GET /api/images/catalog` and `/api/video/catalog` -> every variant has a `recommendedSettings` key; for every non-null one, `basis` in {measured, estimated}, `set(basisByKey) == set(settings) == set(reasons)`; and no mflux / sd.cpp / ltx-2-mlx variant has `cacheStrategy`, `cfgDecay`, `previewVae`, `sampler` or `fp8LayerwiseCasting` in `settings`. No model needs to be on disk.
- Found while reading (not mine): the Video Studio resets steps to 30 and guidance to 5.0 on every model change, and the backend substitutes the catalog `defaultSteps` / `cfgOverride` only when the request equals the schema defaults (50 / 3.0). So the distilled video rows (CausVid, lightx2v distill: 4 steps, CFG 1.0) run 30 steps at CFG 5 unless the user changes them. Image side: the Studio's Balanced 24 equals the image schema default, so `defaultSteps` is substituted, but its guidance 6.0 never equals 5.5, so `cfgOverride` is never applied from the Studio. The panel recommends the right values; a proper fix belongs in `useVideoState.ts` / `app.py`.

## New i18n keys (namespace `studio`)

- `recommended.title` = Recommended for this machine
- `recommended.measured` = Measured
- `recommended.estimated` = Estimated
- `recommended.measuredTitle` = Timed on {machine}.
- `recommended.estimatedTitle` = From the model card, upstream documentation or a similar machine; not timed on this hardware.
- `recommended.measuredOn` = timed on {machine}
- `recommended.applyAll` = Apply recommended
- `recommended.allApplied` = All in use
- `recommended.inUse` = In use
- `recommended.use` = Use
- `recommended.on` = On
- `recommended.off` = Off
- `recommended.cacheValue` = {name}, threshold {threshold}
- `recommended.labels.qualityPreset` = Quality preset
- `recommended.labels.steps` = Steps
- `recommended.labels.guidance` = Guidance
- `recommended.labels.sampler` = Sampler
- `recommended.labels.size` = Size
- `recommended.labels.numFrames` = Frames
- `recommended.labels.fps` = FPS
- `recommended.labels.cache` = Diffusion cache
- `recommended.labels.cfgDecay` = CFG decay
- `recommended.labels.previewVae` = Preview VAE
- `recommended.labels.fp8LayerwiseCasting` = FP8 layerwise
- `imageStudio.fields.maxPixelsNote` = {name} renders at most {megapixels} megapixels; this size is reduced to fit, keeping its shape.
- `videoStudio.fields.maxPixelsNote` = {name} renders at most {megapixels} megapixels per frame; this size is reduced to fit, keeping its shape.

Reasons are English strings from the backend table (not localised, like other backend messages).

## Decisions to propose

- FLUX 3 Klein: recommend 8 steps instead of the maker's 16 (measured 1.8x faster, no visible difference on 3 prompts). Alternative: keep 16 and show the 8-step timing only in the reason.
- Image Studio hides diffusers-only controls for mflux / sd.cpp variants (done here; revert the `show*` gates in `ImageStudioTab.tsx` if unwanted).
