---
title: Overview
order: 4
summary: What ChaosEngineAI is, who it is for, scope, constraints and how it is built.
---
# Overview

ChaosEngineAI is a standalone desktop app for running AI models on your own machine: chat, image, video and voice, behind one launch screen, one model library and one OpenAI-compatible API. It is a product in its own right; it does not name or compare itself with other desktop AI apps (see [rules](rules.md)).

## Goals

- **Goal:** run current open models well on a Mac (Apple Silicon first) and on Windows or Linux with an NVIDIA GPU, with the speed and memory tricks (speculative decoding, cache compression, model-specific engines) that make big models usable, and explain honestly when a setting does not apply.
- **Not a goal:** a hosted service, a model trainer, or a place for models that cannot run locally.

## Audience

Technical enthusiasts and developers with a Mac (M-series, 16 to 128 GB) or a CUDA PC who run models locally; a second group uses the API from other tools (editors, agents).

## Scope

| In scope now | Later (see [ideas](ideas.md)) |
|---|---|
| Chat with MLX, llama.cpp and vLLM engines; image and video studios; voice (speech to text, text to speech, calls); model library and downloads; OpenAI and Ollama-compatible server; 10 locales | Local agents that build things through the app, image-to-video, one Models page per area, CUDA-side accelerators, more video models |

## Constraints

- Licences: dependencies must be MIT, Apache 2.0, BSD or similar; model licences (for example Qwen Research, LTX community licence) are shown on the model row.
- Memory: the main test machine is an M4 Max with 64 GB; big models are checked against the Metal working-set limit (about 52 GB).
- Cost: every push runs GitHub builds, so pushes are held until the work is tested.
- Cross-platform: Windows and Linux paths are written from the start but most live validation happens on the Mac; "not run on Windows or Linux" is tracked per feature.

## How it is built

| Part | Where | Notes |
|---|---|---|
| Frontend | `src/` | React, TypeScript, Vite; tests with vitest |
| Desktop shell | `src-tauri/` | Tauri (Rust): sidecar start, runtime unpack, updater |
| Backend | `backend_service/` | FastAPI sidecar; routes, state, helpers, engines |
| Engines | `backend_service/inference/`, `mlx_worker*.py` | MLX worker subprocess, llama.cpp server, vLLM, MTPLX and TensorFold subprocess lanes |
| Cache and speculative decoding | `cache_compression/`, `dflash/`, `backend_service/ddtree*.py` | Native, TurboQuant, TriAttention, DFlash, DDTree |
| Image, video, voice | `backend_service/image_runtime/`, `video_runtime/`, `voice_*.py` | Diffusers, mflux, stable-diffusion.cpp, ltx-2-mlx, voice worker |
| Catalogs | `backend_service/catalog/` | Chat, image and video model rows; audited by `scripts/audit-catalog-repos.py` |

More detail: [engineering](engineering.md). The reasoning behind every change is in the [tracker](tracker/README.md).

## Kick-off record

The project predates this hub; the kick-off answers were never written down. The decisions that shaped it are in the tracker rows and the [decision log](decisions.md). No landscape brief was written for this hub; the upstream research notes are in `docs/UPSTREAM_RESEARCH_2026-05-16.md` and `docs/UPSTREAM_RESEARCH_2026-10-01.md` (sources secondary, confidence "likely").
