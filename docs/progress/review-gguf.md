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
| 6 | Full pytest | PASS after a fixture fix | first run 4609 passed, 2 failed: two TensorFold routing tests used config-only folders, which the new guard refuses; fixtures now carry a weights file (`8e0f4864`), both files 56 passed |
| 7 | Live, OLD code (`e0306d5a`, port 8922): library entry `my-gguf-notes-mlx`, backend `mlx` | FAIL (bug reproduced) | HTTP 500 "No .gguf weights found inside .../models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d..." |
| 8 | Live, NEW code (port 8921): same entry, backend `mlx` and `auto` | PASS | engine `mlx` both; reply "France's capital is **Paris**." |
| 9 | Live: HF cache `lmstudio-community/Qwen3-0.6B-GGUF`, backend `llama.cpp` and `auto` | PASS | engine `llama.cpp` both; reply "**Paris**" |
| 10 | E2E phase 213 (real models) | PASS | 4 checks, 6.8 s: MLX link `my-gguf-notes-mlx` on mlx (auto, explicit), GGUF in folder `mlx` on llama.cpp, config-only folder refused naming `config.json` |
| 11 | `./scripts/e2e_test_suite.py --smoke` (port 8921) | PASS | phases 0, 3, 4, 5, 6, 7, 8 all pass |

Model lock taken 10:11:05, models unloaded, both backends stopped, lock released. Commands: `<scratchpad>/review-gguf/live.sh`
(loads, chats, phase 213, smoke). Reports in `<scratchpad>/review-gguf/e2e-report/`.

## Not tested

- Windows and Linux (paths, symlinks): Mac only.
- The UI badge in a running app (vitest covers `libraryItemFormat`).
- Conversion from a GGUF source after the change (the conversion path calls the same `_looks_like_gguf`; stream A's phase 211 covers it on `review/a`).
- A remote-only Hub repo id (nothing in the cache): still decided by its name, as before.
