# Review stream GGUF: format routing from content, not names

Branch `review/gguf`, worktree `/Users/dan/ChaosEngineAI-review-gguf` (cut from `release/v0.10.0`).
Plan: `docs/plans/overnight-e2e-review.md`. Started 2026-11-18.

Bug (from stream A): any model path or name containing "gguf" loads as a GGUF model, so an MLX folder
named e.g. `my-gguf-notes-mlx` fails with "No .gguf weights found".

## Setup (how to resume)

- Python: `/Users/dan/ChaosEngineAI/.venv/bin/python`, pytest with `HOME=$(mktemp -d)`.
- Backend port 8921. Model lock: `<scratchpad>/review-lock/model.lock` (mkdir to take, rmdir to release).

## Where the name decided the format (audit)

| Place | Before | After |
|---|---|---|
| `inference/_utils.py` `_looks_like_gguf` | `"gguf" in value.lower()` | content of the path (see `helpers/weights_format.py`) |
| `state/__init__.py` `_resolve_model_target` library branch | `"gguf" in name or path` | the library entry's scanned format |
| `state/__init__.py` last-resort detection | `"gguf"` anywhere in target, ref or path, even over an explicit `mlx` | content of the local path or HF cache copy; the name only for a repo id with nothing on disk |

## Log
