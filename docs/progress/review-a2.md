# Review stream A2: Benchmarks and Conversion, the defects stream A left

Branch `review/a2` (cut from `review/a` at `02d41b40`), worktree `/Users/dan/ChaosEngineAI-review-a2`.
Read `docs/progress/review-a.md` first: it holds the evidence and the "Left" list this stream works through.
Started 2026-11-18. Budget about 3 hours.

## Setup (how to resume)

- Throwaway HOME: `<scratchpad>/review-a2-home` (`.chaosengine` data dir, `Models/`). `.cache/huggingface/hub` links
  small real model folders only (never dataset folders; datasets download into the throwaway cache).
- Backend: port 8920, dummy token, empty HF tokens. Start script and PID in `<scratchpad>/review-a2/`.
- Model lock: `<scratchpad>/review-lock/model.lock` (mkdir to take, rmdir to release, unload first).

## Items

| # | Item | State | Commit |
|---|---|---|---|
| 1 | Benchmark Cancel really cancels | in progress | |
| 2 | Real progress for Benchmarks and Conversion | pending | |
| 3 | Invented Quality and Compression figures | pending | |
| 4 | Task accuracy scoring (next-token) | pending | |
| 7 | Raw hub message for a model not on disk | pending | |
| 8 | KV-cache panel and Est. tok/s with no source | pending | |

## Decisions (documented, mine to make)

- A cancelled benchmark is discarded, not saved: a partial run has no comparable number.

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
