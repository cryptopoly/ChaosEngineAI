# Architecture overview

ChaosEngineAI is three cooperating processes joined by HTTP:

1. A **Tauri shell** written in Rust that owns the OS window, the system
   tray, the in-app updater, and the bundled runtime extraction.
2. A **React + TypeScript frontend** rendered inside the Tauri webview.
3. A **Python FastAPI backend** spawned by the shell, with worker subprocesses
   for each inference engine.

```
┌─────────────────────────────────────────────────────────┐
│  Tauri shell  (Rust, src-tauri/)                        │
│  ├─ webview2 / WebKit / WebView2 renders React          │
│  ├─ Bundled Python runtime extraction at first launch   │
│  ├─ Spawns the Python backend on 127.0.0.1:8876         │
│  ├─ Signed in-app updater (tauri-plugin-updater)        │
│  └─ Tray icon + window menus                            │
└─────────────────────────────────────────────────────────┘
                          │   HTTP
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Python backend  (FastAPI, backend_service/)            │
│  ├─ routes/        — 14 route modules, 125 endpoints    │
│  ├─ state/         — workspace, sessions, downloads,    │
│  │                   metrics, logs, generation          │
│  ├─ inference/     — RuntimeController + engines        │
│  ├─ image_runtime/ — diffusers / sd-cli pipeline mgr    │
│  ├─ video_runtime/ — diffusers / mlx-video / sd-cli     │
│  ├─ catalog/       — curated text / image / video lists │
│  ├─ helpers/       — system stats, settings, persist    │
│  ├─ plugins/       — plugin host                        │
│  ├─ rag/           — workspace knowledge stack          │
│  ├─ mcp/           — stdio JSON-RPC tool client         │
│  └─ tools/         — built-in agent tools               │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  MLX worker  │  │ llama-server │  │ MTPLX server │
│ (subprocess) │  │ (subprocess) │  │ (subprocess) │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Why three processes

Models that live in RAM are big. A 27B MLX model occupies ~30 GB of unified
memory; a stuck pipeline can take ~minutes to clean up. The backend's
inference engines run in **separate subprocesses** so a wedged generation
never takes the whole FastAPI parent down with it:

- The MLX worker speaks JSON-RPC over a pipe to the parent.
- `llama-server` and `MTPLX` run as HTTP servers on dynamic local ports;
  the parent proxies requests.
- Each subprocess is supervised by the parent; if it dies the parent
  surfaces the death cleanly and lets the user retry.

The Tauri shell is itself a subprocess parent — it kills the backend on
clean exit and watches for orphans. The `RuntimeController` tracks its
own subprocess children for the same reason.

## The Python package layout

After the v0.8.0 refactor (CHANGELOG entry), the backend looks like this:

```
backend_service/
├── app.py                    FastAPI app + lifespan + DI wiring
├── inference/
│   ├── __init__.py           Public re-exports (97 LOC — facade only)
│   ├── controller.py         RuntimeController (~1207 LOC) — the orchestrator
│   ├── base.py               Engine interface + types
│   ├── _mtp.py               MTP model registry
│   ├── capabilities.py       Backend probe (mlx, gguf, vllm, mtplx)
│   ├── binaries.py           Binary path resolution + bundled bin dir
│   ├── conversion.py         MLX conversion helpers
│   ├── jsonrpc.py            JsonRpcProcess for subprocess IPC
│   ├── llama_cpp_engine.py   GGUF / llama-server adapter
│   ├── mlx_engine.py         In-proc MLX engine (rarely used)
│   ├── mtplx_engine.py       MTPLX subprocess adapter
│   └── simple_engines.py     RemoteOpenAIEngine + MockInferenceEngine
├── state/
│   ├── __init__.py           ChaosEngineState facade (860 LOC — was 4418)
│   ├── sessions.py           Chat session persistence
│   ├── documents.py          Per-session document attachments
│   ├── benchmarks.py         Benchmark history
│   ├── downloads.py          Download manager
│   ├── generation.py         In-flight generation tracker
│   ├── lifecycle.py          Backend lifecycle (start/stop)
│   ├── logs.py               LogManager
│   ├── metrics.py            Hardware telemetry
│   ├── openai_compat.py      /v1/* OpenAI shim state
│   ├── payloads.py           Request schema parsing
│   ├── settings_state.py     User preferences persistence
│   └── _helpers.py           Title / system-prompt / catalog helpers
├── mlx_worker.py             MLX subprocess orchestrator (318 LOC)
├── mlx_worker_*.py           11 sibling modules: request/prompt/io/...
├── image_runtime/            Image diffusion runtime manager
├── video_runtime/            Video diffusion runtime manager
├── sdcpp_{image,video}_runtime.py  stable-diffusion.cpp adapters
├── mlx_video_runtime.py      Subprocess engine for LTX-2 + Wan
├── routes/                   FastAPI endpoint modules
└── helpers/                  Cross-cutting helpers
```

The facade pattern is repeated: a top-level module owns the *public*
import surface, sibling modules own the *implementation*. Tests that
patch `module._private` continue to patch through the facade because
the symbols are re-exported.

## Routing layer (`routes/`)

Every API surface lives in its own file under `backend_service/routes/`:

| File | Endpoints |
|---|---|
| `auth.py` | `/api/auth/session` |
| `benchmarks.py` | `/api/benchmarks/run` |
| `cache.py` | `/api/cache/preview` |
| `chat.py` | `/api/chat/*` (23 endpoints incl. compare + HTML challenge + sessions + delve + variants) |
| `compare.py` | `/api/chat/compare` |
| `diagnostics.py` | `/api/diagnostics/*` |
| `finetuning.py` | `/api/finetuning/*` |
| `health.py` | `/api/health` |
| `images.py` | `/api/images/*` |
| `metrics.py` | `/api/metrics/gpu` |
| `models.py` | `/api/models/*` |
| `openai_compat.py` | `/v1/*` |
| `plugins.py` | `/api/plugins/*` |
| `prompts.py` | `/api/prompts/*`, `/api/prompt/enhance` |
| `server.py` | `/api/server/*` |
| `settings.py` | `/api/settings/*` |
| `setup/` | `/api/setup/*` — submodules per installer (mtplx, longlive, wan, gpu_bundle, cuda_torch, turbo) |
| `storage.py` | `/api/settings/storage/*` |
| `video.py` | `/api/video/*` |
| `workspaces.py` | `/api/workspaces/*` |
| `html_challenges/` | `/api/chat/html-challenges/*` submodule |

Every route module accesses runtime state through
`request.app.state.chaosengine` — a `ChaosEngineState` facade that owns
the `RuntimeController`, the session store, the download manager, and the
log buffer.

## See also

- [Inference engines](inference-engines.md) — engine adapter details.
- [Routing](routing.md) — `_select_engine` + fallback ladder.
- [Catalog vs library](catalog-vs-library.md) — how models are discovered.
- [Runtime paths](runtime-paths.md) — where everything lives on disk.
