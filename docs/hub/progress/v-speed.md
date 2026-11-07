---
title: "Progress: V-speed (timing split, warm workers, keep-warm policy)"
status: built, in review
---
# Progress: V-speed

Updated: 2026-11-07. Owner: Claude (Opus agent). Branch: `feat/speed-warm-workers` (from `feat/home-studio`).
Plan: [remaining-and-v2.md](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/speed-warm-workers/docs/plans/remaining-and-v2.md) section 2.3, [speed-quality.md](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/speed-warm-workers/docs/plans/speed-quality.md) C1, V1, I1, C2, C5, C3.

## Goal

Know where image and video time goes (C1), then stop paying the model load on every still and clip: keep mflux (I1) and ltx-2-mlx (V1) loaded between jobs, with a keep-warm policy that never holds two big models at once (C2, C5).

## Results (M4 Max 40-core GPU, 64 GB, nothing else heavy running)

Same prompt and seeds, cold = one CLI process per job (the old path), warm = the new worker. Three consecutive jobs each.

| Job | Cold (CLI) | Warm worker | Gain per job after the first | Plan estimate | Output |
|---|---|---|---|---|---|
| Iris-3B Turbo, 768×512, 4 steps | 10.9 / 9.8 / 8.9 s (median 9.8) | 8.2 (start + load) / 5.8 / 5.3 s | **about 41 %** | 15 to 40 % | bit-identical (SHA-256 per seed) |
| same, `perf-baseline.py --only image-mflux` | 6.3 s | 5.5 (start + load) / 3.9 / 4.9 / 4.3 s | about 32 % | | bit-identical |
| FLUX 3 Image Klein edit, 1 reference, 768×512, 8 steps | 38.1 / 38.3 / 38.2 s (median 38.2) | 34.9 (start + load) / 29.5 / 29.4 s | **about 23 %** | 15 to 40 % | bit-identical |
| LTX-2.5 q4, 768×448, 49 frames, text only, **as shipped** (only the transformer kept, 11.3 GB) | 73.8 s (other captures 67.8, 69.2) | 68.2 (start + load) / 69.7 / 70.8 / 71.2 s | **about 4 %**, inside the cold path's spread | 15 to 30 % | bit-identical |
| same, whole pipeline kept (first build, 29 GB) | 67.8 s | 69.2 / 73.2 / 75.5 / 75.5 s | **about 10 % slower** | | bit-identical |
| same, whole pipeline kept + 2 GB MLX cache limit | 69.2 s | 85.9 / 92.3 / 79.3 / 73.1 s | slower | | bit-identical |
| LTX-2.5 q4, 768×448, 49 frames, start image, whole pipeline kept | 87.7 / 85.8 / 86.6 s (median 86.6) | 85.9 / 77.4 / 81.1 s | about 8 % | | bit-identical |
| LTX-2.5 q4, 512×320, 33 frames, text only, whole pipeline kept (spike) | 39.8 s | 33.4 / 31.1 / 31.1 s | about 22 % | | bit-identical |

C1 split, cold vs warm (seconds, one representative job):

| Job | Path | Load | Encode | Denoise | Decode (+save) | Total |
|---|---|---|---|---|---|---|
| Iris | cold CLI | 4.4 (start, imports, load and encode together) | – | 4.9 | 0.4 | 9.8 |
| Iris | warm | 0.0 | 0.0 | 5.7 | 0.0 | 5.7 |
| FLUX 3 edit | cold CLI | 7.7 (incl. encode and reference) | – | 29.2 | 1.3 | 38.2 |
| FLUX 3 edit | warm | 0.0 | 0.4 | 28.3 | 0.6 | 29.3 |
| LTX-2.5 768×448×49 text | cold CLI | 4.0 | 5.8 | 59.8 | 4.1 | 73.8 |
| LTX-2.5 768×448×49 text | warm (transformer kept) | 1.2 (Gemma, VAE encoder, upsampler) | 3.9 | 60.9 | 4.8 | 70.8 |

What the numbers say:

- The fixed cost per cold job is about 4 to 9 s (process start, Python imports, weight load from the page cache, MLX first-call compile). That is a big share of a fast still (Iris: 32 to 41 %), a fair share of an 8-step edit (23 %), and **a small share of a clip** (5 to 10 %), where the denoise is 80 % of the time. The plan's 15 to 30 % for clips holds only for tiny clips (22 % at 512×320×33).
- **Memory is the price.** Resident between jobs: Iris worker about 15 GB, FLUX 3 Klein edit about 21 GB, LTX worker about 12 GB (the transformer). LTX decode peak: 23.6 GB in the worker against about 17 GB for the CLI.
- Spike, part 1: ltx-2-mlx's CLI runs the pipeline with `low_memory=True`, which frees the text encoder, transformer and decoders inside every call, so the same pipeline object kept nothing between calls; in-process with `low_memory=True` was even slower than the CLI (41 to 46 s against 39.8 s at 512×320). With `low_memory=False` everything stays loaded and the output is bit-identical.
- Spike, part 2: keeping everything (29 GB resident, 38.6 GB Metal peak at decode) was faster at 512×320 but **about 10 to 20 % slower in the denoise at 768×448** on this 64 GB Mac (likely close to the Metal wired-memory limit; a 2 GB MLX cache limit made it worse). Keeping only the transformer (a pipeline subclass that ignores the pipeline's own `self.dit = None`) keeps the CLI's memory behaviour for everything else: bit-identical, about 4 % faster. That is what ships.
- The warm mflux worker was first about 10 % slower in the denoise loop than the CLI (32 s against 29 s on FLUX 3 edit). The CLI's memory saver clears the MLX cache right before the loop; doing the same in the worker (without dropping the text encoder) closed the gap.
- Timings on this laptop move ±10 % between captures (Iris cold 6.3 to 9.8 s, LTX cold 67.8 to 73.8 s for the same job). The gains above compare runs made minutes apart.

## Done

- **C1 timing split** on every image and video job: `ProgressTracker` sums seconds per phase (`last_timings()`), the mflux and ltx-2-mlx engines report their own split (CLI: read off the tqdm bar and ltx-2-mlx's `[phase] done in Xs` lines; worker: measured inside), and the result goes on `GeneratedImage.timings` / `GeneratedVideo.timings`, the saved artifact (`timings`, so the Studios' history and run-estimate data can read it), and the PNG / MP4 metadata (`timings` key, numbers only). No UI change.
- **V1 warm ltx-2-mlx worker** `backend_service/ltx2_worker.py` (JSON lines, the music protocol; the CLI's own argument parser and distilled branch, with only the transformer kept between clips; progress and timing through the same parser as the CLI; cancel by raising from the next output write, kill after 20 s).
- **I1 warm mflux worker** `backend_service/mflux_worker.py`: the same command and arguments as the CLI path go through that command's own `build_parser()`, and the model call mirrors each CLI's `main()` (FLUX.1, FLUX.2 Klein, Qwen-Image 2.1, FLUX 3 Klein and its edit command, Iris), so reference images, image strength, guidance, negative prompt, quantize, seeds, size and steps all behave as before. The model is rebuilt only when command, model, quantize or LoRA change. Cancel through mflux's step callback.
- **Fallback**: any worker failure (start, crash, error, no output) runs that image or clip through the CLI path, and the rest of an image batch goes the CLI way too; the runtime note says so. `CHAOSENGINE_WARM_WORKERS=0` turns the workers off.
- **C2 / C5 keep-warm policy** (`backend_service/warm_worker_client.py`, no settings): idle stop after 10 minutes (as music and voice); before a heavy job of another kind (image, video, music) takes the `HEAVY_JOBS` slot every idle warm worker of another kind stops, so a storyboard batch keeps one model for all stills, then one for all clips; stop when free memory drops under 6 GB while idle; stop right after a job whose footprint is over half the Mac's memory (Qwen-Image 2.1 bf16 at 46 to 50 GB is never kept); `image_runtime.unload()` / `video_runtime.unload()` (Unload button, music XL, storyboard kind switch, downloads) stop an idle worker; a diffusers or other engine load frees the warm worker of its kind first. A busy worker is never killed by the policy; an explicit unload during a run waits on the runtime lock as before (D-39: nothing is lost, the user cancels).
- **C3**: both workers clear the MLX cache between jobs (and mflux before the denoise loop).
- `scripts/perf-baseline.py`: `image-mflux`, `video-ltx2mlx`, `music`, `storyboard-render` labels, and the C1 split on the diffusers `image` / `video` labels. Floors in `PERF_BASELINE.md`.
- Tests: `tests/test_mflux_worker.py`, `tests/test_ltx2_worker.py` (stub engines in the real worker scripts: protocol, warm reuse, cancel, hard cancel, crash fallback, policy, switch), `tests/test_timing_split.py`.

## Not done / not verified

- Music floor: the music pack is not installed on this Mac, so `--only music` skips; the label is written but not run.
- The full lighthouse storyboard batch was not re-timed through the app (the plan's "done when" for C5). The per-job numbers above are measured through the engines directly; the batch re-time needs the live app and the owner's lighthouse project.
- Not run through a live backend on a port: the engines were driven directly from Python. The HEAVY_JOBS hook and the unload paths are covered by tests only.
- Qwen-Image 2.1, FLUX.1, FLUX.2 Klein and the LTX-2.3 pack were not run through the warm worker (not downloaded here); their handlers mirror the CLI code but are untested on real weights. Any failure falls back to the CLI.
- LoRA: the mflux path never passed LoRA flags before this change, so neither does the worker (parity, not a new feature).
- Peak memory of the warm LTX worker at 1280×704 or 10 s clips was not measured. On 64 GB the 768×448×49 decode peak was 23.6 GB (transformer kept).
- The shipped LTX mode (transformer only) was measured text-only; the start-image clips were measured with the earlier whole-pipeline build. 97-frame (4 s) clips were not timed; expect the gain to shrink further, since the saving is a fixed few seconds.
- Free-memory gates (mflux 24 GB, LTX 28 GB to start a worker) and the 6 GB low-memory stop are judgement values, not measured thresholds.
- E2E check in `scripts/e2e_test_suite.py` (CLAUDE.md feature gate) not added: that file is outside this stream's files. Proposed below.

## How to resume

1. `git checkout feat/speed-warm-workers` (worktree or main checkout), `git log --oneline staging..` to see the commits.
2. Tests: `.venv/bin/python -m pytest tests/test_mflux_worker.py tests/test_ltx2_worker.py tests/test_timing_split.py tests/test_mflux_engine.py tests/test_ltx2_mlx_runtime.py -q`.
3. Re-measure: `.venv/bin/python scripts/perf-baseline.py --only image-mflux` and `--only video-ltx2mlx` (models must be downloaded; check `ps aux | grep -E "music_worker|mflux|ltx"` first and never run two at once).
4. Worker logs: `~/.chaosengine/logs/mflux-worker.log`, `~/.chaosengine/logs/ltx2-worker.log`.

## Decisions to propose

- **D-new: keep models loaded between jobs (I1/V1).** Recommend yes for images as built: second and later stills 23 to 41 % faster, at the cost of 15 to 21 GB held until 10 idle minutes pass or another kind of job starts. For clips the gain is only about 4 % (12 GB held, decode peak +6.5 GB over the CLI); keep it on for 64 GB Macs with the free-memory gate, or turn it off (one constant) if the owner prefers the memory. The real clip speed-ups are elsewhere: V3 fast-motion interpolation and V2 float16.
- **E2E gate**: add a smoke check to `scripts/e2e_test_suite.py` that runs two images through a stub warm worker (the test engines here) and asserts `timings.warm` on the second; owned by whoever owns that file.
- **perf-gate**: teach `scripts/perf-gate.py` to skip labels the run skipped, then add the V-speed floors.
