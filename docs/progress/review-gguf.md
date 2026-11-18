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
| `state/__init__.py` `explicit_gguf_path` | path ends in `.gguf` | same, unless that path is a folder |
| `helpers/discovery.py` `_detect_storage_format`, `_detect_directory_model` | any `.gguf` under the folder wins | `config.json` plus safetensors/npz wins over a stray `.gguf` (mixed folder) |
| `controller._select_engine` (auto), `_select_tensorfold`, conversion source | `_looks_like_gguf` (name) | `_looks_like_gguf` (files) |
| `src/utils/library.ts` `libraryItemFormat` | "gguf" in name or path gives the GGUF badge | only `backend === "llama.cpp"`; the name only for an entry with no backend |
| Left as is (not routing) | | `helpers/huggingface.py` Hub search falls back to the repo name when tags are missing; `controller` conversion `_looks_gguf_only_repo` refuses a remote repo ending in `-gguf`/`_gguf` before a download; `_mtp.is_mtp_gguf_repo` (name, only when the header probe fails); `libraryVariantMatchScore` (catalog match score, a small name bonus) |

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | New unit tests on the old code | FAIL x8 (as expected) | `tests/test_gguf_routing.py` run against HEAD `e0306d5a` in a throwaway worktree: 8 of 19 fail (MLX folder named gguf, discovery, resolve target, select engine, missing weights, llama.cpp on safetensors, LFS pointer) |
| 2 | New unit tests after the fix | PASS | 19 passed |
| 3 | `test_inference`, `test_backend_service`, `test_services`, `test_cache_strategies`, `test_gguf_routing` | PASS | 303 passed |
| 4 | vitest, tsc | PASS | 151 files, 1816 tests; tsc clean |
| 5 | Library on a throwaway home (port 8921) | PASS | `my-gguf-notes-mlx` (link to Qwen3-0.6B-4bit snapshot) listed MLX/mlx; `mlx` folder holding the Qwen3-0.6B GGUF listed GGUF/llama.cpp; HF cache GGUF repo listed GGUF |
