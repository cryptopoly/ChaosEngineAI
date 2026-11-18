# Review stream ROUTING: the model's location decides which model it is (L-16)

Branch `review/routing`, worktree `/Users/dan/ChaosEngineAI-review-routing` (cut from `review/gguf`).
Plan: `docs/plans/overnight-e2e-review.md`. Started 2026-11-18. Defects: L-16 (and L-19 for the backend) from
`review/d:docs/progress/review-d.md`; D-13 was fixed in the app only.

Bug: the backend took the `canonicalRepo` (and a `modelRef` naming a catalogue row) on the client's word when
it chose the engine and the catalogue row. Any client (CLI, agent, an older window, an external API caller)
could still trigger D-13: loading `mlx-community/Qwen2.5-0.5B-Instruct-4bit` with another row's repo answered
500 "Qwen3.8 Flash Next runs only on TensorFold".

## Setup (how to resume)

- Python: `/Users/dan/ChaosEngineAI/.venv/bin/python`; pytest with `HOME=$(mktemp -d)`.
- Backend port 8940: `<scratchpad>/review-routing/backend.sh` (throwaway HOME whose HF cache links three small
  model folders read-only: Qwen2.5-0.5B-Instruct-4bit MLX, Qwen3-0.6B-4bit MLX, Qwen3-0.6B GGUF).
- Live script `<scratchpad>/review-routing/live.sh`; E2E runner `suite.sh`; reports in `e2e-report/`.
- Model lock: `<scratchpad>/review-lock/model.lock` (mkdir to take, rmdir to release).

## Where a repo hint decided the model (audit)

All chat loads go through `state.load_model` (`state/lifecycle.py`): the load route, the CLI (`chaosengine-cli load`
posts to it), chat generate (loads per message), Compare, HTML Challenge, keep-going model swap, benchmarks and
`/v1/chat/completions`. The agent bridge (`/mcp`) has no chat-load tool.

| Place | Before | After |
|---|---|---|
| `state._resolve_canonical_repo` | the client's `canonicalRepo` first, as sent | an HF cache path is its own repo; a path elsewhere keeps a hint only when the folder is named exactly the repo (`name`, `org/name`, `org--name`, `org_name`) or the file is the catalogue row's pinned `ggufFile`; no path: a catalogue row's own repo wins over a hint; a missing path names nothing, so no hint either. Dropped hints log a debug line. |
| `lifecycle.load_model` `modelRef` | a `modelRef` naming another catalogue row (or another repo) picked the row's context, name, runtime target and, in the controller, TensorFold by repo | `align_model_ref_with_path`: replaced by the path's library name, HF repo or folder name; a plain name or an agreeing ref is kept |
| `state._find_library_entry` | matched the path OR the name, first entry wins, so a path plus another entry's name could load the other model | a path on disk matches only the entry at or holding it (a snapshot folder or file inside an HF cache entry counts); the name is used only without a path or when the path is not on disk |
| `generation.generate` / `generate_stream` | wrote the message's `canonicalRepo` onto the loaded model after the load | only what `_resolve_canonical_repo` agrees with for the loaded model's path |
| Left as is | | `catalog/capabilities._catalog_lookup` (badges, substring fallback, not routing); the frontend's word score (`libraryVariantMatchScore`, L-19 in the app): whatever it picks, the backend now drops a hint the path does not name |

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Focused suites after the fix (`test_inference`, `test_backend_service`, `test_services`, `test_cache_strategies`, `test_gguf_routing`, `test_library_cache_persistence`, `test_dflash`, `test_tensorfold_routing`, `test_route_contracts`) | PASS after one fix | first run: `test_model_load_still_rejects_broken_entry_when_path_does_not_exist` failed (a missing path stopped the name match); a path not on disk now names nothing |
| 2 | New `tests/test_routing_hints.py` on the old code (`f7283367`, throwaway worktree) | FAIL x13 of 17 (as expected) | the four that pass are the exact-match cases that must keep working |
| 3 | Same file on the fix | PASS | 17 passed |
| 4 | Live, OLD code, port 8940: load Qwen2.5-0.5B (HF cache path) with `canonicalRepo` = `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | FAIL (bug reproduced) | HTTP 500 "Qwen3.8 Flash Next runs only on TensorFold, which is not installed..." |
| 5 | Live, NEW code: same request | PASS | HTTP 200, engine `mlx`, ref and `canonicalRepo` `mlx-community/Qwen2.5-0.5B-Instruct-4bit`; chat with the same hint: "Paris" |
| 6 | Live: `modelRef`, `modelName` and `canonicalRepo` all naming the TensorFold row, path Qwen2.5 | PASS | HTTP 200, engine `mlx`, loaded as Qwen2.5 under its own name; chat "Hello!" |
| 7 | Live: `chaosengine-cli --port 8940 load ... --canonical-repo Vontra/...` | PASS | state loaded, engine `mlx`, loadedRef Qwen2.5 |
| 8 | E2E phase 214 (new) and 213 | PASS | 214: 3 checks, each load fresh (2.6 s, 2.6 s, chat "Paris"); 213: 4 checks |
| 9 | tsc, vitest | PASS | tsc clean; 151 files, 1816 tests |
| 10 | `./scripts/e2e_test_suite.py --smoke` (port 8940) | PASS | phases 0, 3, 4, 5, 6, 7, 8 |
| 11 | Full `./scripts/e2e_test_suite.py` (throwaway HOME, three small models) | PASS after one fix | 21 phases, 61 s: 20 pass, phase 1 failed one check, "GGUF native q8_0 KV cache: asked for q8_0, got cacheBits=None" although the note said Native q8_0. Not routing: `chaosengine-cli load` never printed `cacheBits`, so the check could not pass from any tree. The CLI now prints it; phase 1 rerun: PASS, 25 checks (11 pass, 14 skip for models or runtimes not in this home) |
| 12 | Full pytest, throwaway HOME | PASS | 4628 passed, 17 skipped, 0 failed |

Model lock taken 10:28:28, released 10:32:23 after unloading; backend on 8940 stopped; the old-code worktree used
for the reproduction was removed.

## Not tested

- The full E2E suite ran on a throwaway home with three small models (Qwen2.5-0.5B and Qwen3-0.6B MLX, Qwen3-0.6B
  GGUF), so 46 of 199 checks skipped: DFlash, MTPLX, TensorFold, Kolibri, LFM3, Bolt and Qwen3.8 drafters, image
  and video renders. None of them was run against the owner's full library.
- Windows and Linux paths (drive letters, `\`): Mac only. `path_names_repo` and the library path match use
  `pathlib`, but nothing was run off the Mac.
- The app's UI after the change (vitest and tsc only; no frontend code changed).
- Behaviour change to know about: a local copy the app matched to a catalogue row from another uploader
  (D-13's "same model under another uploader") now loads with its own repo, so a drafter pinned to the other
  uploader's repo id (`PINNED_DRAFTERS`, exact by design) is no longer used for it; a conversion of a base repo
  loses the base repo as its `canonicalRepo` unless the folder is named after it.
