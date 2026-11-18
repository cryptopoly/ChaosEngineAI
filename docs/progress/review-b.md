# Review stream B: Fine Tuning

Branch `review/b`, worktree `/Users/dan/ChaosEngineAI-review-b`, cut from `release/v0.10.0` at `b31c4ea5`.
Plan: [overnight-e2e-review.md](../plans/overnight-e2e-review.md). Budget cut to 3.5 hours by the coordinator at the start of the run.

## Verdict

**Hide behind the experimental switch until wired.** No real job can be started from the app on this Mac, or on any machine: the page was a mock-up with a dead button, and an adapter cannot be applied to a model. The engine under it works (section 3), so this is a "not built" feature, not a broken one. Do not remove: the adapter scan, the dataset helper and the request model are a fair start, and section 7 lists what is missing.

| Rubric | Answer | Evidence |
|---|---|---|
| Works | **No.** Training: nothing runs. Use an adapter in Chat: ignored. List adapters: yes, after the crash fix. | Sections 1, 2 and 3; `POST /api/finetuning/start` was a placeholder; base-model answer returned with `adapterPath` set |
| Honest | **No before, mostly after.** The page, the status line and the route now say it is not built. Still promising: `README.md` lines 96 and 167, the sidebar caption "LoRA adapters and training", the page's "or via llama.cpp", the hint "use the fine-tuning feature below", the CLI help. | Sections 1, 4, 5, 7 |
| Valuable | **Not as built** (a read-only list, zero adapters on the owner's machine, nothing to do with one). **Yes if built**: a 0.5B model learned a fact in under 10 s here, and no other page covers it. | Section 3 |
| Clutter cost | One of four slots under Tools; 136 lines of tab, 95 of routes, 220 of helper; 18 strings in 11 locales; 3 CLI commands; 2 README lines. Low upkeep, but a visible slot for a page that does nothing, weeks before a podcast demo. | `src/constants/tabs.ts` line 59 |
| Verdict | **Hide behind the experimental switch** (one line in `src/constants/tabs.ts`, plus the README lines); keep routes, helper and tests. Build it later as its own project (section 8). | |

If the tab is hidden, the honesty commit `b8dab48f` still matters for the CLI and the API (501 instead of a false "accepted"), so keep it.

Setup used for every check below:
- Backend on port 8901, throwaway `HOME` (scratchpad `review-b/home`), token `review-b-token`, `HF_HUB_OFFLINE=1`, empty HF tokens.
- The throwaway Hugging Face cache holds one symlink to the real `models--mlx-community--Qwen2.5-0.5B-Instruct-4bit` snapshot folder. Nothing copied, nothing deleted.
- Python `/Users/dan/ChaosEngineAI/.venv/bin/python` (mlx 0.32.4, mlx-lm 0.32.2).

## 1. What the tab promises and what the code does (read)

| Promise on the page | Code | Result |
|---|---|---|
| "Fine-tune models with QLoRA on Apple Silicon (MLX) or via llama.cpp." | `routes/finetuning.py` `start_finetuning`: builds a config, sets status to `accepted`, returns. Comment in the file: "placeholder until real training is wired up". No subprocess, no thread, no llama.cpp path. | FAIL: nothing trains. |
| "1. Upload Dataset: JSONL with prompt and completion fields" | No upload route, no file picker. `helpers/finetuning.py` `prepare_dataset` exists and is imported by the route file but is never called. | FAIL: no way to give a dataset. |
| "2. Configure: learning rate, epochs, LoRA rank, batch size" | No form. The request model has the fields; nothing in the UI sends them. | FAIL |
| "3. Train: monitor loss and progress in real-time" | Status is read once on mount, never polled. `progress` stays 0.0. No loss value anywhere. | FAIL |
| "Start Fine-Tuning" button | `onClick={() => {/* TODO: open training config modal */}}` | FAIL: a dead click. |
| "LoRA Adapters, N found" | `GET /api/adapters` walks the data folder, the Hugging Face cache, `~/.cache/mlx` and `~/Models` for `adapter_config.json`. | Works, with defects (section 3). |
| "Place adapter directories in your model directories" | The user's own model directories (Settings) are not searched, only the four fixed places above. | FAIL for a custom directory. |
| Use an adapter in Chat | `LoadModelRequest.adapterPath` exists in `backend_service/models/__init__.py` and is read nowhere. | FAIL: an adapter cannot be loaded. |

## 2. Live checks against the backend (port 8901)

| Check | Result | Evidence |
|---|---|---|
| `GET /api/adapters` on a clean data folder | PASS | `{"adapters":[],"count":0}`, 200 |
| `GET /api/finetuning/status` before any call | PASS | `{"status":"idle","progress":0.0,...}` |
| `POST /api/finetuning/start` with a model and a dataset that do not exist | FAIL (honesty) | 200, `"status":"accepted"`, `"message":"Fine-tuning request accepted. Training backend not yet wired."`; output path `/nonexistent/lora-f5af5788` was never created |
| Status after start | FAIL | stays `accepted`, `progress` 0.0, for ever (until the backend restarts) |
| Second start while "accepted" | FAIL | 200 again; the 409 guard only checks `training`, a state nothing sets |
| Cancel | FAIL | `POST /api/finetuning/cancel` 404, `/stop` 404; no cancel route exists |
| Adapter scan time on the owner's real library (270 GB cache) | FAIL (performance) | `list_adapters` took 4.15 s, inside an `async def` route, so the whole backend stalls for that time whenever the tab opens |

Commands:

```
curl -s -H 'Authorization: Bearer review-b-token' http://127.0.0.1:8901/api/adapters
curl -s -H 'Authorization: Bearer review-b-token' http://127.0.0.1:8901/api/finetuning/status
curl -s -H 'Authorization: Bearer review-b-token' -H 'Content-Type: application/json' \
  -d '{"modelPath":"/nonexistent/model","datasetPath":"/nonexistent/ds.jsonl"}' http://127.0.0.1:8901/api/finetuning/start
```

## 3. Real training run under the app (mlx-lm directly)

The app cannot start a job, so the "shortest real job" was run one layer down, with the trainer the page names:
`python -m mlx_lm lora` from the app's own `.venv` (mlx-lm 0.32.2), on the smallest local MLX text model,
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` (276 MB). This shows what the Mac can do and what the app would have to wrap.
Model lock: queued 07:41, held by streams A, F and C until 08:14:49, **held by B 08:14:49 to 08:15:50 (61 s)**, released in a `finally`.
Script: `scratchpad/review-b/real_run.py`; full log: `scratchpad/review-b/real_run.log`.

Dataset: 40 rows in the format the page promises (`{"prompt": ..., "completion": ...}`), four phrasings of one made-up question
("What is the pass phrase of the Harbour Lantern Society?") with one answer ("... is copper-heron-nine.").

| Step | Result | Evidence |
|---|---|---|
| App helper `prepare_dataset` on the 40 rows | PASS | `{'total_records': 40, 'train_records': 36, 'valid_records': 4}`; first line `{"text": "What is the pass phrase ...?\nThe pass phrase ... is copper-heron-nine."}` (plain text, no chat template) |
| Base model, no adapter | baseline | "I'm sorry, but I don't have access to specific information about a particular society or organization. ..." |
| Train A: helper's text format, 60 iterations, batch 4, 8 layers, lr 1e-4, rank 8 | PASS | rc 0, **8.5 s** wall including model load; trainable 0.297 % (1.466M of 494.033M); val loss 6.156 at iter 1, 0.001 at 30, 0.000 at 60; about 3,400 tokens/s |
| Train B: the same rows as `prompt`/`completion` (mlx-lm applies the chat template itself) | PASS | rc 0, **6.2 s**; val loss 4.874, then 0.025 at 30 and 60; about 4,700 tokens/s |
| Output | PASS | `adapter_config.json` (1.4 KB, mlx-lm's training arguments), `adapters.safetensors` 5.9 MB, plus one 5.9 MB checkpoint per `--save-every` (30): 17 MB for a 60-iteration run |
| Adapter applied by mlx-lm (`mlx_lm generate --adapter-path`) | PASS | A and B both answer "The pass phrase of the Harbour Lantern Society is copper-heron-nine." (A with and without the chat template) |
| Cancel (`SIGINT` after 25 s of a 5,000-iteration run) | works, rough | rc -2 with a `KeyboardInterrupt` traceback; checkpoints 30 to 390 and `adapters.safetensors` stay on disk (79 MB), so the last checkpoint is usable, but nothing prunes them and nothing says "cancelled" |
| App lists the result (`GET /api/adapters`, adapters written to the throwaway `~/Models`) | PASS after fixes | both listed; before the fixes this list crashed the tab, and the base model read as the cache snapshot path (now `mlx-community/Qwen2.5-0.5B-Instruct-4bit`). `lora_rank` is null for an mlx-lm adapter (the rank is under `lora_parameters.rank`), and the size counts the checkpoints (16.8 MB shown for a 5.9 MB adapter) |
| **Load the result in Chat** (`POST /api/models/load` with `adapterPath`, then `POST /api/chat/generate` with the question) | **FAIL** | load 200, the word "adapter" is nowhere in the load reply, and the chat reply is the base model's, word for word: "I'm sorry, but I don't have access to specific information ..." The adapter is ignored. |
| Unload | PASS | 200; no MLX worker of stream B left (the two running belong to streams A and E by their working folder) |

What this proves: the Mac and the bundled mlx-lm can fine-tune (a 0.5B model learns a fact in under 10 s), so nothing is missing below the app. What it does not prove: quality on real data, memory on larger models (peak memory was not recorded), or that the helper's plain-text format is good enough outside a memorisation toy. The helper turns chat `messages` rows into made-up `<|role|>` markers instead of the model's chat template; mlx-lm reads `messages` and `prompt`/`completion` rows itself, so the helper should pass them through.

Also seen: `adapter_config.json` from mlx-lm holds the absolute paths of the dataset and the output folder, and `GET /api/adapters` returns that whole file in `config`.

## 4. UI states (dev server on port 8931 against the backend on 8901, built-in browser, own tab)

| State | Result | Evidence |
|---|---|---|
| Empty (no adapters), 1400 px | PASS after fixes | "0 adapters found", "No LoRA adapters found.", three step cards, Start button. Before the fixes the lower panel's subtitle was the raw status word (`idle`, or `accepted` after any start call). |
| Start button | FAIL before, fixed | A click sent nothing: the network log shows only the two `GET /api/finetuning/status` calls from mount. Now the button is off with a sentence under it. |
| Loaded (one adapter folder in `~/Models`) | FAIL before, fixed | Whole tab replaced by "Fine-Tuning crashed. TypeError: Cannot read properties of undefined (reading 'toFixed')" (`FineTuningTab.tsx:74`). After the fix: "1 adapter found", card with name, base, size, created. |
| Backend stopped mid-use, then tab opened | FAIL before, fixed | "0 adapters found" and "Loading adapters..." for good. Now "The backend is offline. Adapters are listed when it is back." |
| Backend stopped, page reloaded | not this tab | The shell shows its own start-up splash ("Extracting embedded runtime"); the tab is not reached. In a plain browser with no backend that text is wrong (nothing is being extracted), but it is the shell's, not Fine-Tuning's. |
| Failing `GET /api/adapters` (500, mocked in the Playwright spec) | FAIL before, fixed | Read "No LoRA adapters found." as if the list were empty. Now an alert line with the error. |
| Narrow (375 x 812) | PASS | `scrollWidth` 375 = `clientWidth` 375, step cards stack, nothing sticks out (also asserted in the spec). |
| Light and dark | one look only | The app is dark-only (`styles.css` line 2 `color-scheme: dark`, no light theme anywhere in `src`). With `prefers-color-scheme: light` emulated the page is unchanged. |
| Design tokens | FAIL (cosmetic, left) | The tab paints with hard-coded hex colours in inline styles (`#27303a`, `#1a1f26`, `#8fb4ff`, `#7a8594`, `#5a6574`, `#c8d0da`), so the strong-border switch and any later theme do not reach it. |
| Adapter card | no action | A card is read-only: no "use in Chat", no reveal in Finder, no delete, no path shown. |

## 5. Bugs fixed (each with a test)

| Commit | Fix | Tests |
|---|---|---|
| `c6b17ab9` | The tab crashed whenever one adapter was on disk (it read `id` and `size_mb`, the backend sends `path` and `size_bytes`). Card is now `AdapterCard`; the backend row gains `created` and reads the base model of an mlx-lm adapter (`model`). `GET /api/adapters` is a plain `def`, so the 4 s walk no longer stalls the event loop. | `src/features/finetuning/__tests__/FineTuningTab.test.tsx`, `tests/test_finetuning_helpers.py` |
| `b8dab48f` | Honesty, in its own commit so it can be reverted alone: Start is off with one sentence, the subtitle no longer prints `idle`, and `POST /api/finetuning/start` answers 501 and changes no state (it answered 200 "accepted" for a job that never ran). Two en strings, null in the other ten locales. | same vitest file, `tests/test_route_contracts.py` |
| `e9b03468` | Offline and failed-list states say what happened; the count shows only once the list has loaded. Two en strings, null elsewhere. | same vitest file, Playwright spec |
| `c76b05bd` | Found with a real mlx-lm adapter: the card showed the base model as the cache snapshot path and its 40-character hash ran out of the card. The row gives `org/name` for a cache path; the card wraps long text. | `tests/test_finetuning_helpers.py` |
| `6a9dbb23` | Test race, not app code: `e2e/tests/17-klingon.spec.ts` ended before the fire-and-forget locale save landed, so on a busy machine the shared backend stayed in Klingon and specs 18 to 20 timed out one after another (9 failures before the run was stopped). `pickLocale` now waits for the `PATCH /api/settings` answer. | whole Playwright suite: 184 passed (21.4 min, `--workers=1`) |

Checks after the fixes: `npx vitest run src/features/finetuning` 4 passed; `npx tsc --noEmit` clean; `npm run i18n:validate` OK;
`pytest tests/test_finetuning_helpers.py tests/test_route_contracts.py tests/test_cli_smoke.py` all passed (throwaway HOME).
Whole suites on the branch tip `e9b03468`: `npx vitest run` 152 files, 1817 tests passed; `pytest tests/ -q -x` exit 0 (throwaway HOME, one skip).
CLI: `chaosengine-cli finetuning-start --body ...` now prints the 501 message and exits 1; `adapters-list` and `finetuning-status` unchanged.

## 6. E2E coverage added

- `scripts/e2e_phases/phase_21_finetuning.py`, registered in `scripts/e2e_test_suite.py` with one import line and one `PHASES` line, under **key 212** (not 21) so it cannot silently replace another stream's phase 21 at merge. Run: `CHAOSENGINE_PORT=8901 .venv/bin/python scripts/e2e_test_suite.py --phases 212`. Result: 5 of 5 pass in 0.02 s. Against the backend before commit `b8dab48f` the last check failed as it should ("the route accepted a run").
- `e2e/tests/28-finetuning.spec.ts`: empty, loaded (adapter folder under the run's `~/Models`), a row with fields missing, a failing list, 375 px. Run: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8951 E2E_API_PORT=8952 npx playwright test tests/01-smoke.spec.ts tests/28-finetuning.spec.ts --workers=1`. Result: 8 passed (41 s). `E2E_PYTHON` is needed in a worktree with no `.venv` above it.
- Row added to `docs/E2E_TESTING.md` and `e2e/README.md`.
- Whole Playwright suite on the branch tip `6a9dbb23`, ports 8951 and 8952, `--workers=1`: **184 passed** in 21.4 min (log: `scratchpad/review-b/playwright-full2.log`). The first whole run failed from spec 18 on because of the Klingon race above, not because of this stream's changes; spec 17 followed by 18 passes alone and in the full run after the fix.

## 7. What is missing before the tab does what it says (hide-until-wired list)

Nothing on the page trains, and an adapter cannot be used. In the order a build would need them:

1. **A training worker.** A subprocess like the MLX worker (never in the FastAPI parent), started by `POST /api/finetuning/start`, that runs mlx-lm's LoRA trainer and reports each step. Today the route answers 501. It must release a loaded chat model first (two models in memory is the RAM bug CLAUDE.md warns about) and go through the busy-advice and job queue paths the studios use.
2. **Progress, loss and cancel.** `GET /api/finetuning/status` returns a fixed dict; there is no loss, no step count, no ETA, no log tail, no cancel route (`/cancel` and `/stop` answer 404), and the tab reads the status once on mount. Section 3 shows what mlx-lm prints and what a cancel leaves on disk.
3. **A dataset path in.** No upload route, no file picker, no validation report. `prepare_dataset` exists but nothing calls it, and it flattens every row to plain text (see section 3 for what that costs in Chat). Paths in the request are not checked against any allowed folder (CLAUDE.md security rule for user-controlled paths).
4. **A configure form.** Model picker (MLX text models from the library only), dataset, iterations or epochs, rank, learning rate, batch size, with a memory and time estimate.
5. **Output location.** Default is `Path(modelPath).parent / "lora-<id>"`, which for a Hugging Face cache snapshot is inside the cache's `snapshots` folder. Adapters need their own folder under the data directory (for example `adapters/`), which the scan already reads first.
6. **Applying an adapter.** `LoadModelRequest.adapterPath` is accepted and ignored: nothing in `inference/controller.py`, the MLX worker or the engines reads it. Chat needs an adapter choice at load time (mlx-lm's `load(..., adapter_path=...)`), the load reply and the model badge must name the adapter, and the base model must be checked against the adapter's.
7. **The llama.cpp half of the sentence.** "or via llama.cpp" has no code at all, and the app ships only `llama-server` builds. GGUF users would need a merge-and-convert step (fuse the adapter, convert to GGUF) or a GGUF LoRA passed to the server; neither exists.
8. **The scan.** It walks the data folder, the whole Hugging Face cache, `~/.cache/mlx` and `~/Models` on every open (4.15 s here), with no cache, and does not read the user's own model directories from Settings although the hint says "your model directories".
9. **Adapter cards.** Read-only: no use, reveal, rename or delete.
10. **Public claims.** `README.md` line 96 ("Fine-tuning with LoRA adapter support for MLX models") and line 167 ("LoRA adapter discovery plus fine-tuning hooks for local training workflows"), the sidebar caption "LoRA adapters and training", and the CLI help line listing `finetuning-start`.

## 8. Note only: what a "no-code fine-tuning" feature would need (not built)

The engine is the easy part (section 3: mlx-lm trains a 0.5B model here in about a minute). The product work is everything around it:

- **Data without JSONL.** Start from what the user already has in the app: chosen chat threads ("teach it to answer like these"), a folder of documents (the app already ingests documents for chat), or a pasted table of question and answer pairs. Show the rows it built, let the user delete bad ones, keep a held-out slice automatically. Use the chat `messages` format so the trainer applies the model's own chat template.
- **One decision, not six.** Presets (Quick taste: about 2 minutes; Balanced; Thorough) that set iterations, rank, layers and learning rate from the model size and the row count. The numbers stay behind an Advanced fold.
- **Fit and time before Start.** Peak memory and minutes from the model size, the same strip the studios use, plus the busy advice. Refuse models that cannot train on this machine with the reason.
- **A run the user can read.** Step counter, a loss line that says "learning" or "not learning" in words, ETA, Pause and Cancel that keep the last checkpoint, resume after a restart, a notification at the end.
- **Proof it worked.** A side-by-side on held-out prompts: base model answer against tuned answer, in the existing Compare view. Without this the user cannot tell a good adapter from a bad one.
- **Use it in one click.** "Chat with this" loads base plus adapter; the adapter shows as its own entry in the model picker with its base named. Optional "Make a standalone model" (fuse, and convert to GGUF for the llama.cpp lane).
- **House rules.** Adapters live in one folder under the data directory with a small metadata file (base model, dataset fingerprint, settings, date). Training data never leaves the machine and the page says so. Licence of the base model is carried onto the adapter card.
- **Scope for a first version.** Apple Silicon, MLX text models up to about 8B at 4-bit, LoRA only, one run at a time, English UI strings through i18n. CUDA (a PEFT lane) is a second project.

## 9. Defects left, ranked by user impact

1. **Nothing trains** (P1 for the feature, by design today). Section 7, items 1 to 5.
2. **An adapter cannot be applied in Chat** (P1 for the feature). `LoadModelRequest.adapterPath` is dropped. Path to wire it: `backend_service/state/lifecycle.py` (pass it on), `backend_service/inference/controller.py` `load_model` (the warm-pool key must include the adapter, or a warm base model answers an adapter request), the MLX engine, `backend_service/mlx_worker_lifecycle.py` line 152 (`load(local_path, adapter_path=...)`), the load reply and the model badge; refuse it with a reason on GGUF, TensorFold and the speculative paths. It touches inference routing, so it needs the full E2E suite, which is why it was not done overnight.
3. **Public claims** (P2): `README.md` lines 96 and 167 advertise fine-tuning; the sidebar caption says "training"; the page still says "or via llama.cpp" and the empty hint still says "use the fine-tuning feature below". Suggested README wording for line 96 if the tab is hidden: drop the bullet; for line 167: "LoRA adapter discovery".
4. **Adapter scan cost** (P2): every open of the tab walks the whole model cache (4.15 s here), uncached; it no longer blocks other requests, but the tab still waits. The CLI's `adapters-list` has a 10 s timeout. Prune `blobs` folders, or cache with the library scan.
5. **Scan misses the user's model directories** (P3) although the hint points there: only the data folder, the Hugging Face cache, `~/.cache/mlx` and `~/Models` are read.
6. **Adapter size counts checkpoints** and `lora_rank` is null for mlx-lm adapters (P3): 16.8 MB shown for a 5.9 MB adapter.
7. **`GET /api/adapters` returns the whole `adapter_config.json`** (P3): for an mlx-lm adapter that includes the absolute dataset path. The tab does not use `config`.
8. **`prepare_dataset`** (P3, unused today): flattens rows to plain text, invents `<|role|>` markers for chat rows, drops bad lines without counting them, no shuffle before the split.
9. **Hard-coded colours** in the tab's inline styles (P3): not on design tokens, so the border switch does not reach it.
10. **`prepare_dataset` is imported and unused** in `routes/finetuning.py` (was so before this review; left alone).
11. **Outside this tab** (for stream D or the integrator): with no backend, a browser reload shows the "Extracting embedded runtime, first launch only" splash, which is not what is happening; the Dashboard's process list shows other backends' Python workers with Stop buttons (other review streams' workers were listed in stream B's Dashboard, on a different backend and data folder).

## 10. Not tested

- Training through the app: impossible, there is no code path.
- Larger models, real datasets, peak memory, thermal behaviour, a run next to a loaded chat model.
- CUDA or Windows and Linux (`mlx-lm` is Apple Silicon only; the page's llama.cpp path does not exist).
- The packaged Tauri app window; only the web UI in the built-in browser and headless Chromium.
- A light theme: the app has none.
- The other ten locales' rendering of the four new strings (null, so English shows; `i18n:validate` passes).
- `pre-build-check.sh`, the full `e2e_test_suite.py`, the cargo tests and the audits were not run (stream G); full pytest, full vitest, tsc, `i18n:validate`, phase 212 and the Playwright suite were.

## 11. Housekeeping

- Lock: taken once, 08:14:49 to 08:15:50, released. No MLX worker of this stream is left.
- Nothing downloaded, no model deleted or copied. The throwaway home (`scratchpad/review-b/home`, about 115 MB of test adapters with `cancelled-lora`) can be deleted.
- Backend (port 8901, PID 60342) and Vite (port 8931, PIDs 59537 and 59565) of this stream are stopped; only its own PIDs were killed (the first Playwright run was stopped by its own PID 71762). The browser tab this stream opened is closed.
- `e2e/node_modules` in this worktree is a symlink to the main checkout's (git-ignored), made to run the spec.
- Hub pages (`docs/hub/status.md`, decisions, FU rows) were not edited, to keep the merge with the other streams clean; the integrator has the verdict above for D-87 and D-88.
- One stray click: the built-in browser's tab was shared at first. Another stream navigated it to its own server (port 8942, Plugins page) between two of this stream's calls, and one click meant for "Try again" landed at about (450, 220) of a 1400 x 900 view of that page, near the "Native f16" card's name, not on its switch. No change was visible in the screenshot. From then on this stream used its own tab (`tab-13`) and no coordinate clicks.
