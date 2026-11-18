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
