# Performance baseline — v0.7.6 → v0.8.0 refactor

Captured 2026-05-09 against `feature/refactor-n-audit @ a53bd5d`.

This file is the floor: any phase PR that drops a wall-time by >5% (or tokens/second by >5%) without a written justification fails the Phase 5 gate.

## Capture

```bash
.venv/bin/python scripts/perf-baseline.py
```

Each gen runs in a fresh subprocess so model loads don't fight for RSS. Models must already be cached locally — the script refuses to trigger downloads unless `--allow-missing` is passed.

## Reference machine

- Apple Silicon (Darwin 25.5.0) — captured on the dev box
- Python 3.11 in `.venv`
- mlx-lm + diffusers from `[desktop,images,mlx-lm]` extras

CUDA / Linux numbers will be captured separately when Phase 4 enables Linux/Windows runners.

## Results

### Text — Qwen2.5-0.5B-Instruct-4bit (mlx-lm)

| Metric | Value |
|---|---|
| Repo | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Prompt | 12 tokens |
| Output cap | 128 tokens |
| Load wall-time | 0.53 s |
| Gen wall-time | 0.39 s |
| Output tokens | 116 |
| **Throughput** | **297 tok/s** |

Run-to-run variance ~15-20% on micro-gens this small. Phase 5 gate compares median of 3 runs, not single.

### Image — FLUX.1-schnell, 4 steps, 1024×1024 (diffusers + MPS)

```
.venv/bin/python scripts/perf-baseline.py --only image
```

| Metric | Value |
|---|---|
| Repo | `black-forest-labs/FLUX.1-schnell` |
| Steps | 4 |
| Resolution | 1024×1024 |
| dtype | bfloat16 |
| Device | mps |
| Load wall-time | _TBD — run script_ |
| Gen wall-time | _TBD — run script_ |

### Video — Wan2.1-T2V-1.3B, 5 frames, 480×272, 4 steps (diffusers + MPS)

```
.venv/bin/python scripts/perf-baseline.py --only video
```

| Metric | Value |
|---|---|
| Repo | `Wan-AI/Wan2.1-T2V-1.3B` |
| Steps | 4 |
| Frames | 5 |
| Resolution | 480×272 |
| dtype | bfloat16 |
| Device | mps |
| Load wall-time | _TBD — run script_ |
| Gen wall-time | _TBD — run script_ |

For reference (FU-009 entry, 2026-05-04 mlx-video path): T5 encode 14.1s + transformer load 0.2s (4-bit q) + denoise 2.9s @ 1.4 it/s + VAE decode 1.3s = **19.6s total**, 383 KB .mp4. The diffusers path benchmarked here is a different code-path (CPU/MPS bf16, not 4-bit MLX), so don't directly compare.

## V-speed additions (2026-11-07, branch `feat/speed-warm-workers`)

Captured on the owner's MacBook Pro M4 Max (40-core GPU, 64 GB), macOS 27.3, mflux 0.24.0, ltx-2-mlx 0.18.0, nothing else heavy running. Details and raw runs: [docs/hub/progress/v-speed.md](docs/hub/progress/v-speed.md).

Every image and video label now reports the C1 split (`load_seconds`, `encode_seconds`, `denoise_seconds`, `decode_seconds`) next to `gen_seconds`. The engine labels run one cold CLI job, then three warm-worker jobs; `gen_seconds` is the **median of the warm runs** (the gated number) and `cold_seconds` is kept for context. `same_output_as_cold` must stay `true` (the warm worker is bit-identical for the same seed).

### Image — Iris-3B Turbo through mflux (`--only image-mflux`)

| Metric | Value |
|---|---|
| Size, steps | 768×512, 4 |
| Cold (one CLI process), script capture | 6.3 s (load+encode 2.9, denoise 3.1, decode+save 0.3) |
| Worker first run (start + load) | 5.5 s |
| **Warm worker, `gen_seconds` floor** | **4.3 s** (median of 3.9 / 4.9 / 4.3) |
| Same output as cold | yes (SHA-256 equal per seed) |

Run-to-run spread on this laptop is large: an earlier hand run measured cold 9.8 s and warm 5.8 s for the same job. Gate on the median of 3 captures, as the text label.

FLUX 3 Image Klein edit with one reference, 768×512, 8 steps (not a script label, measured by hand): cold 38.2 s, warm 29.4 s, same bytes.

### Video — LTX-2.5 q4 distilled through ltx-2-mlx (`--only video-ltx2mlx`)

| Metric | Value |
|---|---|
| Size, frames | 768×448, 49 frames, text only |
| Cold (one CLI process), script capture | 73.8 s (load 4.0, encode 5.8, denoise 59.8, decode 4.1); 67.8 and 69.2 s in two earlier captures |
| Worker first run (start + load) | 68.2 s |
| **Warm worker, `gen_seconds` floor** | **70.8 s** (median of 69.7 / 70.8 / 71.2) |
| Same output as cold | yes (SHA-256 equal per seed) |

The warm gain on clips is small (about 4 %, inside the run-to-run spread of the cold path): the worker keeps only the transformer (11.3 GB), because keeping the whole pipeline made the denoise about 20 % slower on 64 GB. See v-speed.md.

### Music — ACE-Step 1.5 turbo, 60 s with lyrics (`--only music`)

Not captured on this branch: the music pack is not installed on the capture Mac (the label reports `skipped`). Reference from `backend_service/catalog/music_models.py`: 5.1 to 5.9 s warm, about 25 s more on a cold start. Capture the floor with `--only music` on a Mac with the pack.

### Storyboard render — four 4 s clips, crossfades, music bed (`--only storyboard-render`)

| Metric | Value |
|---|---|
| Film | 768×448, 24 fps, 4 shots, about 14.5 s |
| **`gen_seconds` floor** | **1.44 s** (median of 1.48 / 1.40 / 1.44) |

These floors are not in `scripts/perf-gate.py` yet: the gate fails on a missing metric, and these labels skip on a Mac without the models. Gate them once the gate learns to skip labels the run skipped.

## Phase gates

| Phase | Gate |
|---|---|
| Phase 0 (this) | Baseline established |
| Phases 1–3 | No regression beyond ±5% on `tokens_per_second` (text) and `gen_seconds` (image, video) |
| Phase 5 | Re-capture all 3, attach to PR description, no regression beyond ±5% |
| Phase 6 (final) | Re-capture, must match or beat baseline |

## How to interpret a regression

Drop in `tokens_per_second` after a Python refactor → check for new top-level imports of torch / diffusers / mlx in the import graph (Python startup cost leaks into first-token latency). Run `python -X importtime backend_service.app 2>&1 | tail -20` to spot it.

Drop in `gen_seconds` at constant params → likely a wrapper added around the diffusers callback-on-step-end. Check `image_runtime.py` / `video_runtime.py` for new abstraction layers.

Drop in load wall-time → eager imports happening in pipeline-loader where they used to be lazy. Verify `_ensure_pipeline` still defers torch.cuda/torch.mps probes until the first real call.
