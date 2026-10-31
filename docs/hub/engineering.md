---
title: Engineering
order: 7
summary: Architecture, tech choices, upstream pins and the refactor rules.
---
# Engineering

Architecture notes and choices for ChaosEngineAI. Detail lives in `CLAUDE.md` (architecture, code-quality rules, upstream table) and in `docs/` (features, architecture, reference).

## Architecture

| Part | Technology | Why |
|---|---|---|
| Shell | Tauri 2 (Rust) | Small native app; owns the sidecar, the updater, microphone permission |
| UI | React, TypeScript, Vite | Fast iteration; vitest for logic |
| Backend | FastAPI (Python 3.11 to 3.13) | One process for routes and state; heavy models run in subprocesses |
| MLX | `mlx_worker` subprocess | Memory isolation: a stuck model cannot take the backend down |
| GGUF | `llama-server` (and a TurboQuant fork binary) | Broadest model support and cache types |
| CUDA | vLLM | Speculative decoding and batching on NVIDIA |
| Image and video | diffusers, mflux, stable-diffusion.cpp, `ltx-2-mlx` | One lane per platform strength; each non-diffusers engine lives in its own venv |
| Voice | one worker in its own venv | Parakeet, Whisper, Kokoro, Silero VAD, speaker labels |

Binary routing, the engine picker and the cache-strategy registry are described in `CLAUDE.md`. Docs by area: [features](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/docs/features), [architecture](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/docs/architecture).

## Tech choices

Recorded in the [tracker](tracker/README.md) with the alternative that lost, for example: subprocess-per-clip for LTX (FU-128), an isolated venv per engine whose dependencies clash with the app's (FU-096, FU-093), the vendored model file hook for architectures mlx-lm lacks (FU-103, FU-118), and keeping the Python TensorFold engine until 1.x covers our families (FU-111).

## Upstream pins

The table of pinned upstream repos and the command to check each is in `CLAUDE.md` under "Upstream Dependencies": llama.cpp tag, dflash-mlx commit, TensorFold commit, mflux release, ltx-2-mlx commit, TriAttention commit. Bump a pin together with the code that reads its output; `scripts/pre-build-check` asserts the paired pins match.

## Rules for changing code

- Heavy imports (`torch`, `diffusers`, `mlx`, `transformers`, `huggingface_hub`) stay lazy; backend start-up target is under 2 seconds (`python -X importtime`).
- Models over 1 GB stay in subprocess workers; always unload before reloading.
- List-form `subprocess` only, paths from requests go through `resolve()` and a parent check, no secrets in source.
- Backend modules over 600 lines, hooks over 400, components over 500 and Rust modules over 800 are a refactor signal: extract before adding.
- Fix the bug surgically; do not refactor around it.

## Reviews and measurements

Measurements carry the command that produced them in the tracker row; examples: DDTree and DFlash speeds (FU-112), LTX clip times and memory (FU-128), estimate fits (FU-110), Wan guidance bisect (FU-122). Performance baseline: `scripts/perf-baseline.py` against `PERF_BASELINE.md` (plus or minus 5 percent gate).
