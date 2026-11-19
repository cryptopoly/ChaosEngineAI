# Studio time estimate fit on the M4 Max (FU-165, 2026-11-14)

Machine: MacBook Pro M4 Max (40-core GPU), 64 GB. Backend from the `v0100/r8` worktree on port 8879, generations
through `POST /api/images/generate` and `POST /api/video/generate` (the same path the Studio uses). "Time" is the
artifact's `durationSeconds`; wall time from submit to response was 0.3 to 1 s longer. Seed 7, prompt "a lighthouse
on a rocky coast at dusk" (images) / "waves rolling onto a rocky coast at dusk" (video). Before each run the script
waited until no other agent's model process (MLX worker, llama-server, sd, mflux, ltx, TensorFold, music, voice)
was running, and it sampled every 2 s during the run; runs that overlapped another heavy process are marked and left
out of the fit. Other agents worked on this Mac at the same time, so the first runs of the day (before the guard)
are not used.

Model: `seconds = catalog x (fixed + (1 - fixed) x steps/ref_steps x w(T)/w(T_ref))`, `w(T) = T + T^2/quad`,
`T = W/16 x H/16 x F/4` (images F = 1). `fixed` and `quad` were fitted per row by grid search (fixed 0 to 0.95,
quad 100 to 1e9) holding the catalog figure, minimising the largest relative error.

## Installed and measured

| Row | Engine | Installed | Measured |
|---|---|---|---|
| Iris-3B Turbo (`speridlabs/iris-3b-mflux`) | mflux | yes | yes |
| FLUX 3 Image Klein (`black-forest-labs/flux-3-image-klein-mflux`) | mflux | yes | yes |
| LTX-2.5 distilled q4 (`dgrauet/ltx-2.5-mlx-q4`) | ltx-2-mlx | yes | yes |
| Any sd.cpp image row (FLUX.1 schnell / dev Q4_K, Qwen-Image-2.1 Q4_K) | sd.cpp | **no** (no leejet GGUF in the HF cache; only the 1.3 GB Qwen-Image-2.1 VAE companion) | **skipped**, no downloads allowed |
| Other mflux rows (FLUX.1 schnell/dev, FLUX.2 klein, Qwen-Image-2.1, FLUX 3 Klein Base) | mflux | no | skipped; they use the mflux default share |

## Iris-3B Turbo (mflux), catalog 18 s at 1024 px / 4 steps

| Size | Steps | Time (s) | Note |
|---|---|---|---|
| 1024 | 4 | 17.5, 18.7 | before the guard, other jobs running: not used |
| 768 | 4 | 8.2 | before the guard: not used |
| 768 | 8 | 7.9 | before the guard: not used |
| 1536 | 4 | 56.8 | before the guard: not used |
| 1024 | 4 | 13.4 | first run after a wait (81 s) for another job: not used |
| 1024 | 4 | 11.7, 10.0 | |
| 1024 | 8 | 10.5 | steps do not change the time |
| 768 | 4 | 5.5 | |
| 768 | 8 | 5.5 | steps do not change the time |
| 1280 | 4 | 21.2 | |
| 1536 | 4 | 34.0 | |

Iris-3B Turbo runs a set schedule: 4 and 8 steps take the same time, so the row gets `ignoresSteps: true`. It works
in pixel space, so cost grows fast with size (1536 px is 3.4x the 1024 px time for 2.25x the pixels): best fit
`fixed 0.25`, `quad 300`. The catalog figure moves from 18 s (taken while other work ran) to 10.5 s.

Fit with catalog 10.5: 768 px 5.6 (meas. 5.5, +2 %), 1024 px 10.5 (10.0, +5 %), 1280 px 20.3 (21.2, -4 %),
1536 px 37.5 (34.0, +10 %).

## FLUX 3 Image Klein (mflux), catalog 40 s at 1024 px / 8 steps

| Size | Steps | Time (s) | Note |
|---|---|---|---|
| 1024 | 8 | 33.6 (first), 34.3 | |
| 768 | 8 | 19.2 | |
| 768 | 16 | 40.0 | |
| 1536 | 8 | 118.2 | another backend process ran during it: not used |
| 1536 | 16 | 207.4 | |
| 1024 | 16 | 71.1 | |

Doubling the steps doubles the time (a little more), so nearly nothing is fixed: best fit `fixed 0.01` (0 is the
grid floor), `quad 5000`, largest error 10 % with catalog 34.5 s. The catalog figure moves from 40 s to 34.5 s.

## Qwen-Image-2.1 (mflux), from earlier runs (not re-run)

The two measured rows (2026-10-29, M4 Max): base 40 steps 258 to 271 s, Turbo 8 steps 72 to 75 s, both 1024 px.
Same architecture, so about 6 s a step and 26 s that does not grow with steps (text encoder, decode). That gives
`fixed 0.1` for the base row and `0.35` for Turbo. Size scaling is not measured for Qwen (default `quad`).

## mflux default share

Measured shares: 0.01 (FLUX 3 Klein), 0.1 and 0.35 (Qwen-Image-2.1 base and Turbo), 0.25 (Iris-3B, pixel space).
`MFLUX_FIXED = 0.1` is the middle; unmeasured mflux rows use it.

## LTX-2.5 distilled q4 (ltx-2-mlx), catalog 128 s at 768x512, 97 frames

ltx-2-mlx runs a set 8 + 3 schedule (steps do not apply), so the grid is two sizes x two frame counts instead of
sizes x steps. Each run starts its own process and loads the model, so every run includes the load ("warm" means
the files are in the page cache).

| Size | Frames | Time (s) |
|---|---|---|
| 768x512 | 97 | 128.9 (first; a probe process of another agent ran briefly), 142.8 |
| 768x512 | 49 | 77.7 |
| 1024x576 | 49 | 109.7 |
| 1024x576 | 97 | 210.7 |
| 512x320 | 97 | 62.7 |

Time grows almost linearly with size x frames: 49 to 97 frames is 1.84x at 768x512 and 1.92x at 1024x576;
768x512 to 1024x576 (1.5x the pixels) is 1.41x at 49 frames and 1.48x at 97. The best max-error fit on these six
runs was `fixed 0.23, quad 70000` (6 %), but it over-predicts the older LTX-2.3 q4 runs on the same engine
(`ltx2_mlx` 768x512 x 257 frames 352 s, 1280x704 x 121 frames 363 s; catalog comments) by 51 % and 64 %. The
linear fit `fixed 0.1, quad 1e9` (square term off) is within 7 % here and 9 % / 14 % on those two older runs, so
the row stores that. The catalog figure moves from 128 s to 136 s (mean of today's two reference runs; both include
the model load, as every ltx-2-mlx run does). The LTX-2.3 q4 row on the same engine takes the same `fixed` / `quad`
(not fitted on that row).

Fit at catalog 136: 768x512x97 136.0 (142.8, -5 %; 128.9, +6 %), x49 75.4 (77.7, -3 %), 1024x576x49 106.3 (109.7,
-3 %), x97 197.2 (210.7, -6 %), 512x320x97 64.6 (62.7, +3 %).

## Comparison: three real runs per Studio against the shown figure

Settings not used in any fit. "Shown" is what the Studio strip computes (`variantSeconds` in
`src/utils/timeEstimate.ts`; the Python mirror gives the same numbers, checked by the shared fixture). Image runs
and video runs ran one at a time with no other model process seen.

| Studio | Row | Settings | Shown (s) | Real (s) | Error |
|---|---|---|---|---|---|
| Image | Iris-3B Turbo | 1152x896, 4 steps | 10.3 | 10.1 | +2 % |
| Image | FLUX 3 Klein | 1280x768, 12 steps | 47.9 | 47.2 | +1 % |
| Image | FLUX 3 Klein | 896x896, 8 steps | 25.5 | 25.3 | +1 % |
| Video | LTX-2.5 q4 | 640x384, 73 frames | 71.2 | 69.9 | +2 % |
| Video | LTX-2.5 q4 | 1024x576, 73 frames | 151.8 | 163.0 | -7 % |
| Video | LTX-2.5 q4 | 768x512, 145 frames | 196.6 | 206.7 | -5 % |

All within the 20 % target. Also seen: the first run after switching mflux models includes the model load (Iris
1024 px 13.2 s against 10.5 shown, -20 %; Klein 34.8 against 34.5). Estimates assume the model is loaded.

The strip was also checked in the browser (Vite dev build against this backend): Iris 1024 px "About 11 s",
1536 px "About 38 s"; Klein 1024 px 16 steps "About 1 min 9 s"; LTX-2.5 768x512 33 frames "About 55 s",
145 frames "About 3 min 17 s", and 50 steps leaves it unchanged (set schedule).

## Not measured

- sd.cpp image rows (none installed), every CUDA row, LTX on CUDA with these constants (4090, plan section 6),
  other Macs and cards (FU-164).
- Qwen-Image-2.1 size scaling; FLUX.1 / FLUX.2 mflux rows.
- `quad` for images is fitted on at most three sizes per row; sizes above 1536 px are extrapolated.


# CUDA fit on the RTX 4090 (FU-165, 2026-11-19)

Machine: Windows 11, RTX 4090 24 GB, 64 GB RAM. Backend from `test/rtx4090` on its own port with a throwaway home, runs through `POST /api/images/generate` and `POST /api/video/generate`, seed 7, wall time from submit to response. Each case: one warm-up run (not counted), then three runs; the median is used. Raw rows: `docs/research/fit-cuda-4090-2026-11.jsonl`. Same model and fit method as the M4 Max section above (grid search of `fixed` 0 to 0.95 and `quad` 100 to 1e9, minimising the largest relative error, the catalogue figure set to the measured one at the reference size).

## FLUX.2 Klein 4B (diffusers, `black-forest-labs/FLUX.2-klein-4B`), catalogue 6.9 s at 1024 px / 4 steps

| Size | Steps | Runs (s) | Median | Fit | Error |
|---|---|---|---|---|---|
| 512 | 4 | 4.66, 4.65, 4.60 | 4.65 | 5.0 | +7 % |
| 768 | 4 | 5.51, 5.41, 5.45 | 5.45 | 5.7 | +5 % |
| 1024 | 4 | 6.86, 6.86, 6.83 | 6.86 | 6.9 | +1 % |
| 1536 | 4 | 11.73, 11.80, 11.75 | 11.75 | 12.8 | +9 % |
| 2048 | 4 | 21.74, 27.29, 28.14 | 27.3 | 26.9 | -1 % |
| 1024 | 8 | 8.39, 8.89, 8.57 | 8.57 | 9.1 | +6 % |

Doubling the steps adds only 25 % at 1024 px, so most of a small image is fixed cost (text encoder, VAE, scheduler set-up): best fit `fixed 0.68`, `quad 1000`. The row's old figure (5.5 s, no constants) was 12 % low at 1024 and 39 % low at 2048.

## LTX-2.5 4-bit on load (diffusers, `Lightricks/LTX-2.5-Diffusers~nf4`), catalogue 62.9 s at 768x512, 97 frames

| Size x frames | Runs (s) | Median | Fit | Error |
|---|---|---|---|---|
| 640x384, 97 | 41.71, 41.45, 41.57 | 41.6 | 43.2 | +4 % |
| 768x512, 49 | 36.67, 36.32, 36.50 | 36.5 | 38.7 | +6 % |
| 768x512, 97 | 61.04, 62.93, 65.75 | 62.9 | 62.9 | 0 % |
| 1024x576, 97 | 94.94, 94.92, 92.79 | 94.9 | 102.6 | +8 % |
| 1280x704, 97 | 287.5, 144.9, 214.6 | 214.6 | 196.9 | -8 % |

The distilled schedule ignores steps (`ignoresSteps`). Best fit `fixed 0.47`, `quad 3162` (the Mac `ltx-2-mlx` row uses `fixed 0.1`, `quad 1e9`: a different engine). 1280x704 swings from 145 to 288 s between runs on this 24 GB card (the model is offloaded; peak memory is far above the card), so its median is a middle figure, not a promise; a fit without that point (`fixed 0.25`, `quad 1e5`) would give 162 s there. The old row (no constants) was 22 to 33 % high at 1024x576 and 1280x704.

## Not measured

FLUX.1 schnell answered in 0.3 s at every size (the run did not use the real model: nothing was fitted), FLUX.2 dev, the bf16 LTX-2.5 row and the other LTX rows keep their old figures. The new cases are in `tests/fixtures/time_estimate_cases.json` (Python and TypeScript both return the expected figures).

## New rows on the RTX 4090 (FU-259, 2026-11-19)

Same machine, method and fit as the CUDA section above: backend from this branch on its own port with a throwaway home, runs through `POST /api/images/generate` and `POST /api/video/generate`, seed 7, wall time from submit to response, the median of three runs after the model was loaded. Raw rows: `E:\tmp\img-e2e\bench.jsonl` on the 4090 (not in the repo). sd.cpp starts a new process for every image, so its figures include loading the GGUF from the OS file cache.

| Row | Case | Runs (s) | Median | Fit | Error |
|---|---|---|---|---|---|
| FLUX 3 Klein (diffusers), catalogue 7.0 s, `fixed 0.57`, `quad 2512` | 1024 px, 8 steps | 7.07, 7.03, 6.87 | 7.03 | 7.0 | 0 % |
| | 1024 px, 16 steps | 10.03, 10.16, 10.11 | 10.11 | 10.0 | -1 % |
| | 768 px, 16 steps | 6.86, 6.84, 6.65 | 6.84 | 6.9 | +1 % |
| | 1280x768, 12 steps | 8.20, 8.29, 8.31 | 8.29 | 8.1 | -2 % |
| | 1536 px, 16 steps | 21.95, 21.86, 22.20 | 21.95 | 22.4 | +2 % |
| FLUX 3 Klein sd.cpp Q8_0, catalogue 5.5 s, `fixed 0.58`, `quad 2818` | 1024 px, 8 steps | 5.49, 5.35, 5.64 | 5.49 | 5.5 | 0 % |
| | 1024 px, 16 steps | 7.75, 7.73, 7.53 | 7.73 | 7.8 | +1 % |
| | 768 px, 8 steps | 4.29, 4.26, 4.33 | 4.29 | 4.3 | 0 % |
| | 1536 px, 8 steps | 9.97, 9.89, 10.04 | 9.97 | 10.1 | +1 % |
| Qwen-Image-2.1 Turbo sd.cpp Q4_K_M, catalogue 7.5 s, `fixed 0.37`, `quad 14125`, `ignoresSteps` | 1024 px | 7.47, 7.49, 7.63 | 7.49 | 7.5 | 0 % |
| | 768 px | 5.54, 5.40, 5.42 | 5.42 | 5.4 | 0 % |
| | 1280x768 | 7.29, 7.61, 7.31 | 7.31 | 7.2 | -2 % |
| | 1536 px | 14.42, 14.43, 14.59 | 14.43 | 14.3 | -1 % |
| MOSS-Image SFT (vendored pipeline, 4-bit text encoder), catalogue 53.0 s, `fixed 0.02`, `quad 1995` | 1024 px, 50 steps | 52.99, 53.01, 52.94 | 52.99 | 53.0 | 0 % |
| | 1024 px, 25 steps | 27.03, 27.12, 27.15 | 27.12 | 27.0 | 0 % |
| | 768 px, 50 steps | 25.77, 25.66, 25.77 | 25.77 | 25.9 | +1 % |
| Kandinsky 6.0 Lite (video), catalogue 109.8 s, `fixed 0.17`, `quad 17783` | 864x480 x 121 | 109.80, 108.81, 109.83 | 109.80 | 109.8 | 0 % |
| | 640x352 x 121 | 51.64, 51.64, 51.55 | 51.64 | 51.6 | 0 % |
| | 864x480 x 49 | 39.53, 40.61, 39.61 | 39.61 | 39.5 | 0 % |

The FLUX 3 Klein diffusers figures are after the placement fix in `flux3_modular.place_on_cuda`: with the text encoder left on the card (diffusers' automatic offload), 1024 px at 16 steps took 15.6 s, because the fp32 VAE decode spilled past the 24 GB card (6 s instead of 0.3 s); 1536 px was refused by the memory gate. Kandinsky's three points fit almost exactly with two constants, so the fit says little beyond these sizes. The new fixture cases are in `tests/fixtures/time_estimate_cases.json`.
