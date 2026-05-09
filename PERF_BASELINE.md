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
