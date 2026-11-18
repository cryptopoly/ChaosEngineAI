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

Pending.

## 4. UI states

Pending.

## 5. Bugs fixed

Pending.

## 6. E2E coverage added

Pending.
