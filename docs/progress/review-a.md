# Review stream A: Benchmarks and Conversion to MLX

Branch `review/a`, worktree `/Users/dan/ChaosEngineAI-review-a`. Plan: `docs/plans/overnight-e2e-review.md`.
Started 2026-11-18. Deadline shortened by the coordinator to about 3.5 hours.

## Setup (how to resume)

- Throwaway HOME: `<scratchpad>/review-a-home` (`.chaosengine` data dir, `Models/` for conversion output).
  `.cache/huggingface/hub` holds symlinks to seven small real model folders and the wikitext dataset; nothing is copied.
- Backend: `<scratchpad>/review-a/backend.sh` (port 8900, dummy token, empty HF tokens), PID in `review-a/backend.pid`.
- UI: `<scratchpad>/review-a/vite.sh` (port 8930), PID in `review-a/vite.pid`.
- Model lock: `<scratchpad>/review-lock/model.lock` (mkdir to take, rmdir to release).

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
| 0 | Backend and UI up on own ports, library lists 7 linked models | PASS | `GET /api/workspace` lists 2 GGUF, 3 MLX, 2 Transformers; `benchmarks: []` |
| 1 | Benchmark, throughput, MLX (`mlx-community/Qwen3-0.6B-4bit`), API | PASS | 322.9 tok/s, 90 tokens, load 3.05 s; `POST /api/benchmarks/run {"mode":"throughput",...}` |
| 2 | Benchmark, throughput, GGUF (`lmstudio-community/Qwen3-0.6B-GGUF`), API and UI | PASS | API 327.5 tok/s; UI run 324.8 tok/s, modal shows phases then the result card |
| 3 | Benchmark, perplexity, MLX | FAIL (fixed later, see Fixes) | HTTP 400 `Dataset 'wikitext/wikitext-2-raw-v1' doesn't exist on the Hub or cannot be accessed.` Fails for every model: the dataset id is wrong |
| 4 | Benchmark, task accuracy MMLU, MLX | FAIL (method) | 0/10 in 0.37 s. Raw outputs with 3 tokens: `"Let's analyze"`, `'\nA.'`, `'__________\n'`. A reply starting "Based on" scores as answer B. Numbers are noise for chat models |
| 5 | Benchmark history, compare (UI) | PASS | 5 runs listed, stat cards, selected vs baseline table with deltas, scatter chart |
| 6 | Benchmark history export and delete | FAIL (missing) | No export, delete or clear exists in the UI or the API (`routes/benchmarks.py` has one route, `POST /api/benchmarks/run`) |
| 7 | Benchmark result honesty | FAIL | "Cache footprint 4.0 GB" for a 0.6B model at 8K (the preview next to it says 0.3 GB): `state/benchmarks.py` falls back to `paramsB = 7.0` when the model is not in the catalog. "Quality 100%" is a constant |
| 8 | Convert Transformers model from the Hugging Face cache (`Qwen/Qwen3.5-4B`, 8.7 GB) | FAIL at HEAD, PASS after fix | HEAD: raw traceback `FileNotFoundError: .../models--Qwen--Qwen3.5-4B/config.json` (the cache folder, not its snapshot, went to `mlx_lm convert`) |
| 9 | Convert to 4-bit and 8-bit | PASS (after fix) | 4-bit: 11 s, 2.22 GB, "4.503 bits per weight"; 8-bit: 9 s, 4.18 GB, "8.502 bits per weight" |
| 10 | Converted models load and chat | PASS (after fix) | both answer "The capital of France is Paris." (4-bit 165 tok/s, 8-bit 95 tok/s). Before the name fix the output folder `models--Qwen--Qwen3-mlx` was listed as `Qwen/Qwen3-mlx` and failed to load ("Hugging Face repository not found") |
| 11 | Convert into an existing folder | PASS | "Output path already exists: ... Delete it or choose a different Output path and retry." |
| 12 | Bad input: bare name, empty body, GGUF-only repo, unsupported architecture, qBits 9 | PASS | plain messages (422 for qBits) |
| 13 | Bad input: repo that does not exist, local path that does not exist | FAIL | both answer "This model is gated on Hugging Face ... set HF_TOKEN" (wrong cause) |
| 14 | Bad input: dtype not in the list | FAIL | raw argparse usage text |
| 15 | Low disk (output on a 300 MB disk image) | FAIL | raw traceback ending `RuntimeError: [write] Unable to write 11796480 bytes to file.`; a 297 MB partial `model.safetensors` is left behind, so a retry hits "already exists" |
| 16 | Cancel a conversion | FAIL (missing) | no cancel in the UI or the API. A GGUF source starts a download of the base repo with no size shown and no progress (`LiquidAI/LFM3-1.7B-Base`, stalled at 0 bytes for 100 s here; I stopped the converter process, the API then returned raw progress-bar text as the error) |
| 17 | GGUF to MLX end to end | SKIP | not completed: the base-repo download stalled; not retried with a larger download |
