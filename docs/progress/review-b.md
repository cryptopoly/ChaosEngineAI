# Review stream B: Fine Tuning

Branch `review/b`, worktree `/Users/dan/ChaosEngineAI-review-b`, cut from `release/v0.10.0` at `b31c4ea5`.
Plan: [overnight-e2e-review.md](../plans/overnight-e2e-review.md). Budget cut to 3.5 hours by the coordinator at the start of the run.

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

Waiting for the shared model lock: stream A held it 07:40 to 07:51, stream F from 07:51 (still held at 08:08). The run script is queued (`scratchpad/review-b/real_run.py`, retries every 60 s) and releases the lock in a `finally`.

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

Checks after the fixes: `npx vitest run src/features/finetuning` 4 passed; `npx tsc --noEmit` clean; `npm run i18n:validate` OK;
`pytest tests/test_finetuning_helpers.py tests/test_route_contracts.py tests/test_cli_smoke.py` all passed (throwaway HOME).
Whole suites on the branch tip `e9b03468`: `npx vitest run` 152 files, 1817 tests passed; `pytest tests/ -q -x` exit 0 (throwaway HOME, one skip).
CLI: `chaosengine-cli finetuning-start --body ...` now prints the 501 message and exits 1; `adapters-list` and `finetuning-status` unchanged.

## 6. E2E coverage added

- `scripts/e2e_phases/phase_21_finetuning.py`, registered in `scripts/e2e_test_suite.py` with one import line and one `PHASES` line, under **key 212** (not 21) so it cannot silently replace another stream's phase 21 at merge. Run: `CHAOSENGINE_PORT=8901 .venv/bin/python scripts/e2e_test_suite.py --phases 212`. Result: 5 of 5 pass in 0.02 s. Against the backend before commit `b8dab48f` the last check failed as it should ("the route accepted a run").
- `e2e/tests/28-finetuning.spec.ts`: empty, loaded (adapter folder under the run's `~/Models`), a row with fields missing, a failing list, 375 px. Run: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8951 E2E_API_PORT=8952 npx playwright test tests/01-smoke.spec.ts tests/28-finetuning.spec.ts --workers=1`. Result: 8 passed (41 s). `E2E_PYTHON` is needed in a worktree with no `.venv` above it.
- Row added to `docs/E2E_TESTING.md` and `e2e/README.md`.

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
