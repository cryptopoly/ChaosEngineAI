# Review stream A2: Benchmarks and Conversion, the defects stream A left

Branch `review/a2` (cut from `review/a` at `02d41b40`), worktree `/Users/dan/ChaosEngineAI-review-a2`.
Read `docs/progress/review-a.md` first: it holds the evidence and the list this stream works through.
Started 2026-11-18 09:14, finished about 10:35.

## Setup (how to resume)

- Throwaway HOME: `<scratchpad>/review-a2-home` (`.chaosengine` data dir, `Models/`). `.cache/huggingface/hub` links
  five small real model folders only (never dataset folders; WikiText-2 and MMLU downloaded into the throwaway cache,
  well under 2 GB).
- Backend: `<scratchpad>/review-a2/backend.sh` (port 8920, dummy token, empty HF tokens), PID in `backend.pid`.
- UI: `<scratchpad>/review-a2/vite.sh` (port 8950), PID in `vite.pid`.
- Model lock: `<scratchpad>/review-a2/lock.sh take|release` (the shared `review-lock/model.lock`).
- Playwright from this worktree: `e2e/node_modules` is a link to the main checkout's (git-ignored).

## Items

| # | Item | State | Commit |
|---|---|---|---|
| 1 | Benchmark Cancel really cancels | done | `997e752e` |
| 2 | Real progress for Benchmarks and Conversion | done | `295c3569` |
| 3 | Invented Quality and Compression figures | done | `494f2cf3`, `d495b940` |
| 4 | Task accuracy scoring (next-token) | done | `cba87174` |
| 7 | Raw hub message for a model not on disk | done | `ad4740a3` |
| 8 | KV-cache panel and Est. tok/s with no source | done | in `494f2cf3` |
| - | E2E phase 211 and docs | done | `28bc6848` |

## Decisions (mine to make, documented)

- **A cancelled benchmark is discarded, not saved.** A partial run has no number that compares with a full one.
  The run request answers `{"cancelled": true, "result": null, "modelUnloaded": ...}` with HTTP 200 and the
  current history and runtime, so the UI is in step without a second request.
- **How a run is stopped.** Perplexity and task accuracy stop between two items: the app names a cancel file in the
  worker request and creates it (the worker reads one request at a time, so a cancel cannot arrive as a message).
  A model that was loaded before the run stays loaded. A model load or a throughput generation cannot be
  interrupted from outside, so there the model process is ended; that also unloads a model the run loaded itself.
  If the worker has not stopped 8 s after a between-items cancel, the process is ended too.
- **One benchmark at a time.** A second run while one is going answers 409.
- **Progress.** A bar has a value only when both numbers are counted by the job (samples, questions, files of a
  Hub download, models scored). A model load, a throughput generation (one blocking engine call with no token
  count until it returns) and the writing of a converted model (bytes are measured, the final size is an
  estimate) show the phase and the measured text with an indeterminate bar.
- **Quality.** Benchmarks no longer store or show `quality` (a constant 100, or the KV-cache preview's table
  value). Conversion reports `diskRatio` (measured), `bitsPerWeight` (the converter's own figure) and, when asked
  (`measureQuality`, on by default in the UI, off by default in the API), `quality`: the perplexity of the
  converted model and of the source on the same 8 x 256 tokens of WikiText-2, each scored in a process of its
  own. Skipped with the reason when the source is not a local folder, when a model would take more than 70 % of
  free memory, when a model cannot be scored, or on Cancel (the converted model is kept).
- **Scoring.** Task accuracy reads the next-token scores after `Answer:` and picks the highest of A to D. The
  generated-text scorer is removed. The method is saved on the run (`taskScoringMethod`); a run without the field
  is taken as `generated`, and two accuracy runs scored differently show both values and no delta.

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Cancel a throughput run during the model load (MLX, real) | PASS | cancel 0.4 s in: run answers `cancelled: true, modelUnloaded: true` 0.29 s later, history 0, no model process |
| 2 | Cancel mid-generation, MLX model already loaded (real) | PASS | 0.08 s to the answer, history unchanged, model unloaded (a generation cannot be interrupted any other way) |
| 3 | Cancel mid-generation, GGUF (real) | PASS | cancelled, nothing saved, no llama-server left; 5.1 s to the answer (the engine waits for the server to exit) |
| 4 | Cancel mid-perplexity, run loaded the model (real) | PASS | 0.08 s, nothing saved, model unloaded, no worker left |
| 5 | Cancel mid-accuracy at question 41 of 2000, model loaded before the run (real) | PASS | 0.08 s, `modelUnloaded: false`, the model still loaded and usable for the next run, history unchanged |
| 6 | Real dataset and model caches untouched | PASS | `find ... -newer <marker>` on the real `datasets--Salesforce--wikitext` and `models--Qwen--Qwen3.5-4B`: nothing |
| 7 | Benchmark progress, real (`GET /api/benchmarks/progress`) | PASS | `loading` with the loader's own text, `dataset`, `evaluating 4/64 ... 64/64` samples; accuracy `9/100, 4 correct ... 96/100, 40 correct` |
| 8 | Conversion progress, real (`GET /api/models/convert/progress`) | PASS | `preparing` 0 to 8 s, `loading`, `saving` with 2.37 GB written of about 2.63 GB expected, then `measuring 0/2, 1/2` |
| 9 | Baseline with the shipped scorer, 500 questions, Qwen3-0.6B-4bit | - | MMLU five-shot 213, zero-shot 47; HellaSwag five-shot 204, zero-shot 37 |
| 10 | Next-token scorer, same 500 questions | PASS | MMLU five-shot 214, zero-shot 150; HellaSwag five-shot 203, zero-shot 130. Five-shot unchanged, zero-shot from below chance to above it |
| 11 | Speed of the new scorer | PASS | 100 MMLU five-shot questions in 6.7 s through the backend (0.07 s a question); 60 prompts in isolation: 5.2 s against 6.6 s for generating three tokens |
| 12 | Conversion with measured quality, real (Qwen3.5-4B, 8.7 GB to 2.22 GB) | PASS | `diskRatio 3.91`, `bitsPerWeight 4.503`, quality measured: source 17.15, converted 17.81, +3.83 % |
| 13 | A model path that is not on disk | PASS | "Model files not found at /nonexistent/e2e-model. Pick the model again from the library, or download it first." |
| 14 | E2E phase 211, real models | PASS | 12 checks, 67 s; report `<scratchpad>/review-a2/e2e-report/e2e-20261118-100941.md` |
| 15 | Real UI pass (Chromium on vite 8950 against backend 8920, LFM3-1.7B and Qwen3.5-4B) | PASS | dialog "9 of 2000 questions, 4 correct", Cancel gives "Benchmark cancelled ... Nothing was saved ... The model was unloaded"; a throughput run finishes with no Quality card; conversion shows "2.22 GB written, about 2.4 GB expected (estimate)", "1 of 2 models scored", then "3.9x smaller on disk", "4.503 (measured by the converter)", "+3.8% perplexity vs the source". Screenshots in `<scratchpad>/review-a2/shots/` |
| 16 | Playwright spec 27 | PASS | 21 tests |

## Commands

- Python: `HOME=$(mktemp -d) /Users/dan/ChaosEngineAI/.venv/bin/python -m pytest tests/test_benchmark_cancel.py tests/test_benchmark_eval.py tests/test_job_progress.py tests/test_conversion_helpers.py tests/test_conversion_quality.py tests/test_route_contracts.py -p no:cacheprovider`
- Frontend: `npx vitest run`, `npx tsc --noEmit`, `npm run -s i18n:validate`
- Browser: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8962 E2E_API_PORT=8963 ./node_modules/.bin/playwright test tests/27-benchmarks-conversion.spec.ts --workers=1`
- Real phase (needs the model lock): `HOME=<scratchpad>/review-a2-home CHAOSENGINE_PORT=8920 CHAOSENGINE_API_TOKEN=review-a2-dummy HF_TOKEN= PATH=/Users/dan/ChaosEngineAI/.venv/bin:$PATH ./scripts/e2e_test_suite.py --phases 211 --report-dir <scratchpad>/review-a2/e2e-report`

## Incident (cleaned up)

`npx playwright` run from `e2e/` before `e2e/node_modules` was linked fetched the `playwright` package into the npx
cache (`~/.npm/_npx/e41f203b7505f1fb`, created 09:37). It ran nothing. I removed that one cache folder and linked
`e2e/node_modules` to the main checkout's instead. Lesson: call `./node_modules/.bin/playwright`, never `npx`, in a
worktree without its own `e2e/node_modules`.

## Left (not done here)

- Throughput has no token counter while it generates: `runtime.generate` is one blocking call. A streaming
  benchmark would give tokens so far and a between-token cancel that keeps the model, but changes how the speed
  is measured; it needs its own comparison run first.
- Cancelling a GGUF generation takes about 5 s to answer (the llama.cpp engine waits for its server to exit).
- `promptTokens` on an MLX throughput run can read 1 (seen on LFM3-1.7B after a cancelled run): the worker reports
  the tokens it processed after the prompt cache. Not shown any more; not investigated.
- `backend_service/helpers/persistence.py` `_build_benchmarks` still holds four sample rows with a `quality` field
  (dead sample data, not shown).
- The shared Performance preview still takes a compressed cache's quality from a table; it is now marked "est.".
- CLAUDE.md's test table has no row for these suites (not edited here: shared file, the integrator's call).

## Not tested

- Cancel on TensorFold, MTPLX, vLLM engines and on a DFlash or DDTree MLX run (the hard stop is the same code).
- Cancel during a dataset download (first use, offline, slow network) beyond the 8 s fallback's unit test.
- Quality measurement on a source too large for free memory (unit test only), on a GGUF source (skipped by
  design, unit test only) and with a model already loaded in chat.
- Conversion progress for a Hub download (file counter parsed from recorded tqdm lines in a unit test only).
- Windows and Linux, the packaged app window, locales other than English in the browser.
- `./scripts/e2e_test_suite.py --smoke` and the full suite (stream G).

## State at the end

- Backend (8920) and Vite (8950) stopped; model lock released; no model process of mine left.
- Throwaway home holds three converted folders under `Models/` (about 6.7 GB) and the two datasets; delete
  `<scratchpad>/review-a2-home` when done.
