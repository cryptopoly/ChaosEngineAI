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
| 4 | Benchmark, task accuracy, MLX | PASS at the defaults, FAIL at zero-shot | Defaults (5-shot, 100 questions): MMLU 41/100, HellaSwag 43/100 on Qwen3-0.6B-4bit, in line with next-token scoring (39/100). Zero-shot: 9/100 (0/10 in the first try), below chance, because the model writes "Let's analyze" in its 3 tokens; next-token scoring gives 29/100 there. A reply starting "Based on" scored as answer B (fixed) |
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
| 16 | Cancel a conversion | FAIL (missing) at HEAD, added later (row 38) | no cancel in the UI or the API. A GGUF source starts a download of the base repo with no size shown and no progress (`LiquidAI/LFM3-1.7B-Base`, stalled at 0 bytes for 100 s here; I stopped the converter process, the API then returned raw progress-bar text as the error) |
| 17 | GGUF to MLX end to end | SKIP in the first pass, PASS later (row 40) | first try: the base-repo download stalled and I stopped the converter |
| 18 | After fixes: repo that does not exist, missing local path, dtype not in the list | PASS | "Hugging Face repository not found: nobody/does-not-exist-xyz-123. ...", "Source path not found: /nope/missing-model. ...", 422 `Input should be 'float16', 'bfloat16' or 'float32'` |
| 19 | After fixes: low disk (300 MB image) | PASS | "Not enough free disk space for the converted model: about 1.22 GB needed, 0.01 GB free for .../tinyvol/out2-mlx. ..." and no folder is created |
| 20 | UI, Benchmarks Run, real GGUF run in the browser (port 8930) | PASS | progress dialog with phases, then "Benchmark complete", 324.8 tok/s |
| 21 | UI, result dialog after a perplexity or accuracy run | FAIL at HEAD, PASS after fix | the dialog had its own copy of the card with the speed layout only ("5200.0 tok/s", "250 tokens generated") |
| 22 | UI, latest-run card after an accuracy run | FAIL at HEAD, PASS after fix | "-327.5 tok/s vs your previous Native f16 run · -327.5 tok/s vs fastest" under a 0/10 MMLU result |
| 23 | UI, History summary cards with an accuracy run on top | FAIL at HEAD, PASS after fix | "Latest 0.0 tok/s"; Fastest and Best value ranked accuracy runs too |
| 24 | UI, Benchmarks Run at 375 px | FAIL at HEAD, PASS after fix | model card squeezed into a 30 px column under the mode fields (a media rule lost to a later base rule in `styles.css`) |
| 25 | UI, Conversion: pick a GGUF source | FAIL at HEAD, PASS after fix | source card stayed on "Select a model to convert..." with Convert enabled (draft path is the `.gguf` file, the library lists the folder) |
| 26 | UI, Conversion states: empty, loaded, error dialog, 375 px | PASS | Playwright spec 27, 12 tests; screenshots in `e2e/screenshots/27-*.png` (git-ignored) |
| 27 | Light theme | SKIP | the app has one theme (dark); there is no light theme to test (`grep -c "prefers-color-scheme\|data-theme" src/styles.css` is 0) |
| 28 | History: double-click to set the baseline | FAIL at HEAD, PASS after fix | the two clicks selected the row first, so the baseline fell back to another run |

## Fixes (each with a test)

| Commit | What | Test |
|---|---|---|
| `0cc814f0` | Conversion of a model in the Hugging Face cache (snapshot folder), output folder name, source size | `tests/test_conversion_helpers.py` |
| `8a2eab44` | Conversion: free-space check, partial output removed, missing path and repo named, dtype validated. Benchmarks: perplexity dataset, answer scoring, speed delta | `tests/test_conversion_helpers.py`, `tests/test_benchmark_eval.py`, `src/utils/__tests__/benchmarkCompare.test.ts` |
| `62688aed` | Result dialog in the run's own mode, History cards, narrow Run layout, GGUF source card, `DELETE /api/benchmarks/{id}`, mode check, phase 211, spec 27 | `tests/test_route_contracts.py`, `src/utils/__tests__/conversionSource.test.ts`, `e2e/tests/27-benchmarks-conversion.spec.ts` |
| `7d7efc7f` | Benchmark cache size from the model's own layout, GGUF note on the Conversion page, docs | `tests/test_benchmark_eval.py`, spec 27 |
| `5d21f48a` | Delete run in History (asks first) | `src/api.test.ts`, spec 27 |
| `59b157df` | History: double click sets the baseline, a baseline of another kind shows no made-up delta; Run: perplexity and accuracy say they need an MLX model | spec 27 |
| `0c93785e` | Default output folder keeps dots | `tests/test_conversion_helpers.py` |
| `266a6acc` | Cancel conversion (route, button, pre-flight and mid-run), no `-GGUF` in the MLX folder name, GGUF source size | `tests/test_conversion_helpers.py`, `tests/test_route_contracts.py`, spec 27, phase 211 |

## Commands

- Python: `HOME=$(mktemp -d) /Users/dan/ChaosEngineAI/.venv/bin/python -m pytest tests/test_benchmark_eval.py tests/test_conversion_helpers.py tests/test_route_contracts.py -p no:cacheprovider`
- Frontend: `npx vitest run`, `npx tsc --noEmit`, `npm run -s i18n:validate`
- Browser: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8960 E2E_API_PORT=8961 npx playwright test tests/27-benchmarks-conversion.spec.ts --workers=1`
- Real phase: `HOME=<throwaway home> CHAOSENGINE_PORT=8900 CHAOSENGINE_API_TOKEN=review-a-dummy ./scripts/e2e_test_suite.py --phases 211 --report-dir <scratchpad>/review-a/e2e-report` (needs the model lock)

## Real runs after the fixes (model lock held 08:18 to 08:26)

| # | Check | Result | Evidence |
|---|---|---|---|
| 29 | Perplexity, MLX, WikiText-2 | PASS | 8 x 128: 64.86 +/- 7.38; default 64 x 512: 31.49 +/- 0.60, the same value on a second run (3.6 s) |
| 30 | Perplexity on a GGUF model | FAIL at HEAD, PASS after fix | 400 "Perplexity evaluation is not supported by the llama.cpp backend." after the model was loaded; the Run tab now says the two modes need an MLX model and disables Run |
| 31 | Benchmark cache size after the fix | PASS | Qwen3-0.6B at 8K: 0.88 GB (was 4.0 GB from the 7B guess) |
| 32 | Cancel a benchmark | FAIL (left) | the client stops waiting, the UI says "Benchmark cancelled.", but the backend finishes the run, saves it (history went 12 to 13 with my label `cancelled-by-client`) and keeps the model loaded |
| 33 | E2E phase 211 with real models | PASS | 10 checks, 55 s: MLX 245.7 tok/s (Qwen2.5-0.5B), GGUF 161.8 tok/s, perplexity 41.7, MMLU 3/10, history 12 before and 12 after, Qwen3.5-4B converted (8.7 to 2.22 GB), second run refused, converted model answered at 178.7 tok/s. Report: `<scratchpad>/review-a/e2e-report/e2e-20261118-082340.md` |
| 34 | UI, real conversion in the browser | PASS | progress dialog, then "Conversion complete", `.../Models/Qwen3-5-4B-mlx-2`, 8.7 GB to 2.2 GB. The dialog has no Cancel and its phases are timed, not real |
| 35 | Full suites on the branch | PASS | pytest 4637 run, 0 failed, 17 skipped (tree at `7d7efc7f`, no Python change after it except the phase file); vitest 1822 passed; `tsc --noEmit` clean; `i18n:validate` OK; Playwright spec 27: 12 passed |

## Second pass (08:27 to 08:46)

| # | Check | Result | Evidence |
|---|---|---|---|
| 36 | History: compare a speed run with an accuracy run | FAIL at HEAD, PASS after fix | "Speed 100.5 tok/s vs 0.0 tok/s"; now the baseline cell and the delta are a dash |
| 37 | Default output folder name | PASS after fix | `Qwen3.5-4B-mlx` (was `Qwen3-5-4B-mlx`, and `models--Qwen--Qwen3-mlx` at HEAD) |
| 38 | Cancel a real conversion (new `POST /api/models/convert/cancel` and a Cancel button) | PASS | cancel 1 s in (pre-flight) and 2.5 s into the write: both answer `{"cancelled":true}`, the convert request answers "Conversion cancelled. Nothing was kept at .../cancel-me-mlx.", the half-written folder is gone, no converter process is left, a second cancel answers `false` |
| 39 | Value check: 4-bit against 8-bit of the same model | PASS | Qwen3.5-4B converted here: 4-bit perplexity 13.29 +/- 0.22, 125.5 tok/s, 2.2 GB; 8-bit 12.56 +/- 0.20, 91.4 tok/s, 4.2 GB |
| 40 | GGUF to MLX end to end (`lmstudio-community/Qwen3-0.6B-GGUF`) | PASS after fixes | without a base repo: "GGUF-to-MLX conversion needs a base Hugging Face model repo ..." (this file has no base-model metadata); with `hfRepo: Qwen/Qwen3-0.6B`: 72 s including a 1.5 GB download into the throwaway cache (download noted here, under 2 GB), 0.32 GB output |
| 41 | The GGUF-derived MLX model loads | FAIL at first, PASS after fix | the default folder `Qwen3-0.6B-GGUF-mlx` failed to load: "No .gguf weights found inside ..." (the loader reads "gguf" anywhere in a path as a GGUF model). The default name no longer carries `-GGUF`: `Qwen3-0.6B-mlx` loads on MLX and answers at 291.7 tok/s. A folder name chosen by the user with "gguf" in it still fails (left, see the report) |
| 42 | E2E phase 211 with the cancel check | PASS | 11 checks, 54 s; report `<scratchpad>/review-a/e2e-report/e2e-20261118-084440.md` |
| 43 | Playwright spec 27 | PASS | 14 tests |

## Incident (cleaned up)

The throwaway home linked the real `datasets--Salesforce--wikitext` cache folder. The dataset loader wrote through that link: a README blob, three `.no_exist` markers and four links (train and validation parquet) whose targets sat in the throwaway cache, so they dangled in the real one. I removed exactly those entries (all created 07:41 or 08:20 today) and the link; the real folder is back to its one test-split file from April. No model folder was written to (checked with `find -newer`). Lesson for the other streams: link model folders only, never dataset folders, and let datasets download into the throwaway cache.

## State at the end

- Backend (8900) and Vite (8930) still run for the integrator; stop with `kill $(cat <scratchpad>/review-a/backend.pid) $(cat <scratchpad>/review-a/vite.pid)`.
- Model lock released 08:45. No model loaded.
- Throwaway home holds four converted folders under `Models/` (about 9 GB), the downloaded `models--Qwen--Qwen3-0.6B` (1.5 GB) and a stub `models--LiquidAI--LFM3-1.7B-Base` (100 KB, the stalled download); delete the home when done.

## Not tested

- A GGUF whose header names its base repo (the one tested has none, so the repo was typed in).
- A mid-write disk-full run after the fix (the free-space check now stops it first; the clean-up path is covered by a unit test only).
- Conversion of a gated repo, of a model over 10 GB, and with a real Hugging Face token.
- The packaged app window, Windows and Linux, other locales in the browser.
- `./scripts/e2e_test_suite.py --smoke` and the full suite (stream G).
