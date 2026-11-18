# Server page and Activity (D-88, stream E)

Status: **spec 2026-11-18, building on `review/e`.** Plan: [overnight review](overnight-e2e-review.md), section "Server and Logs redesign".

## What the user gets

1. **Connect block collapsed.** One line: "Connect your app", the base URL and a Copy button. Opening it shows the snippets as today. Open or closed is remembered per user (`localStorage` key `chaosengine.server.connectOpen`, default closed).
2. **Two views on the Server page:** *Overview* (status, API key, connect line, warm pool, stats, settings) and *Activity* (what the models did). The view is remembered (`chaosengine.server.view`).
3. **Activity view:**
   - **Timeline per model:** one row per model seen in the events, a bar from the first event to the last, split into phases (loading, ready, generating, idle) with the seconds on hover and in an accessible label. A failed phase is drawn with the danger colour.
   - **Event list:** newest first, filter by model, level (info, warning, error) and request id (free text), plus a search box. Each row: time, kind, model, one-line summary (for example "loaded in 8.2 s, 4.1 GB").
   - **Detail row:** click a row to expand: every field, the full command line, the environment the engine got (allow-listed and redacted), the last stderr lines.
   - **Copy** one event, **Copy selected** and **Export selected** (JSON file) with row check boxes.
   - **Live tail with Pause.** Paused, new events are counted ("12 new") and shown on Resume.
   - **Latest failure:** a button that jumps to the newest error event and opens its detail.
   - **Log lines:** the old plain log (sidebar Logs page) as a second list under the events, with its search box, so nothing that showed before is lost.
4. **Logs folds into Server.** The sidebar Logs entry goes; the id `logs` becomes a legacy id that opens Server on Activity (same redirect map as the old Discover ids, `constants/tabRedirects.ts`). Saved state that names `logs` (sidebar memory, last tab) maps to `server`. `LogsTab.tsx` stays in the tree until the owner agrees to delete it.

## Backend events

New module `backend_service/model_events.py`: `ModelEventLog`.

- **Event shape** (JSON): `id` (monotonic int), `ts` (epoch seconds, float), `time` (ISO local), `kind`, `level`, `model` (display name), `modelRef`, `requestId`, `engine`, `cacheStrategy`, `speculative` (method or null), `message` (one line), `data` (numbers: `bytes`, `seconds`, `memoryBytes`, `prefillSeconds`, `tokS`, `promptTokens`, `completionTokens`, `reason`, `finishReason`), `detail` (strings: `command`, `env`, `stderr`, `error`).
- **Kinds:** `requested`, `resolving`, `loading`, `loaded`, `load_failed`, `parked` (moved to the warm pool), `first_token`, `generation`, `generation_failed`, `idle`, `unloaded` (`data.reason`: `user`, `swap`, `idle`, `error`, `memory`). `idle` is emitted when the last in-flight generation on the loaded model ends; there is no idle auto-unload today, so `reason: idle` is reserved and documented, not invented.
- **Storage:** an in-memory ring (`deque`, 1000 events) plus a JSONL file `<data dir>/logs/model-events.jsonl` rotated at 1 MB with 3 backups (`.1` to `.3`), so the disk use is capped near 4 MB. A daemon writer thread drains a bounded queue (2000); a full queue drops the disk write (never the ring entry) and counts it. At start the ring is filled from the newest lines of the current file, so a restart keeps the history.
- **Redaction** at emit time, on `message` and every `detail` string: Hugging Face tokens (`hf_...`), `Bearer ...`, `sk-...` keys, values after `--api-key`, `--hf-token`, `token=`, env values whose name contains TOKEN, SECRET, KEY, PASSWORD or AUTH, paths of secret files (`.../huggingface/token`, `.netrc`, `.ssh/...`, `*.pem`, `*.key`) and the home folder written as `~`.
- **Cost on the hot path:** none per token. A stream emits at most two events (first token, end). `emit` builds a dict, appends to the ring under a lock and `put_nowait`s to the writer queue and subscriber queues.

### Where the events come from (real code points)

| Event | Code point | Why there |
|---|---|---|
| requested, resolving | `state/lifecycle.load_model` start and after the backend is resolved | Has the request (ref, path, backend, cache, speculative) and the id |
| loading, loaded, load_failed | same function around `runtime.load_model` | One place for every engine; seconds measured there; memory from `RuntimeController._model_resident_bytes`, weights bytes from `_target_resident_bytes` |
| command line, stderr tail | read from the engine after load or failure: `LlamaCppEngine.last_command` (one new line where it builds the command) and its log file tail; `MLXWorkerEngine.worker.command`; the error text carries the worker's stderr | No new plumbing through engines |
| unloaded (user) | `state/lifecycle.unload_model` | The user's Unload |
| unloaded (swap, memory) | `RuntimeController._park_active_engine_or_unload` and `_pop_oldest_warm_entry` | Where a load pushes a model out |
| first_token, generation, generation_failed, idle | `RuntimeController.generate` and `stream_generate` | Every caller (Chat, OpenAI-compatible API, agents, Storyboard planner, benchmarks) goes through them |

The controller gets an optional `events` attribute (None in tests and FakeRuntime); `ChaosEngineState` sets it. The controller file is over the size cap, so the hooks live in `model_event_hooks.py` (decorators on `load_model`, `generate`, `stream_generate`, one-line calls elsewhere) and the store in `model_events.py`.

### Routes (added; nothing removed)

- `GET /api/server/events?model=&level=&kind=&requestId=&since=&limit=` returns `{events, capacity, dropped}`, newest last.
- `GET /api/server/events/stream` SSE: the newest 200, then live events, keepalive every second (same shape as the log stream).
- `GET /api/server/logs/stream` stays. Its backlog sent the 50 *oldest* lines of the ring (the ring is newest first); it now sends the 50 newest, oldest first.
- Workspace payload, Diagnostics (`logs` tail from the backend log file), the CLI `server-logs` and the agent bridge are unchanged. No new agent tool in this stream (golden MCP list unchanged); a `server_events` tool is a follow-up.

## Frontend layout

- `features/server/ServerTab.tsx` keeps the props and becomes the shell (view switch, modal).
- `ServerOverview.tsx` (status, key, warm pool, stats, settings), `ConnectPanel.tsx` (collapsed line plus snippets), `ServerActivity.tsx` (filters, actions, tail state), `ModelTimeline.tsx`, `EventList.tsx` with `EventDetail.tsx`, `LogLines.tsx` (old lines).
- `hooks/useModelEvents.ts`: initial GET, then the SSE stream; pause keeps a pending count; capped at 1000 in memory.
- `features/server/modelEvents.ts`: pure helpers (timeline phases, filters, summaries, export text) with vitest tests.
- Monochrome inline SVG icons from `components/icons`, no emoji, theme tokens only, narrow window: timeline labels stack above bars below 720 px.

## Tests

- pytest `tests/test_model_events.py`: emit shape, ring bound, rotation and backups bound, writer queue full drops disk writes only, history reload, redaction, controller emits for generate and stream (fake engine), lifecycle emits for load, failure and unload, route contract for `/api/server/events` and the stream, log stream backlog order.
- vitest: helpers, ConnectPanel remembered state, redirect of `logs`, Activity rendering and filters.
- Playwright `e2e/tests/31-server-activity.spec.ts`: connect line collapsed and remembered, Activity view, filter, expand, Logs redirect.
- E2E phase `scripts/e2e_phases/phase_21_server_activity.py`: load a model, generate, check the events.
