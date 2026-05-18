# CLI reference

The `chaosengine-cli` wrapper covers all 125 backend routes through either a
typed shortcut or the generic `call` dispatcher. This page is the canonical
endpoint reference, grouped by route prefix.

Counts and route shapes were captured live from the running backend with:

```bash
./scripts/chaosengine-cli routes --filter /api/<prefix>
```

If the backend grows new routes, the dispatcher reaches them automatically;
the typed shortcuts table below is updated per release.

## Generic dispatcher

```bash
./scripts/chaosengine-cli call <METHOD> <PATH> [--body JSON | --body-file FILE]
```

Examples:

```bash
./scripts/chaosengine-cli call GET /api/health
./scripts/chaosengine-cli call POST /api/models/load \
    --body '{"modelRef":"Qwen/Qwen3-4B","speculativeDecoding":true}'
./scripts/chaosengine-cli call PATCH /api/settings \
    --body '{"defaultContext":32768}'
```

## Endpoints by prefix

### /api/adapters
LoRA adapters discovered on disk

| Method | Path | Operation |
|---|---|---|
| GET | `/api/adapters` | Get Adapters |

### /api/auth
Session auth (UI consumes the cookie)

| Method | Path | Operation |
|---|---|---|
| GET | `/api/auth/session` | Auth Session |

### /api/benchmarks
Benchmark runs

| Method | Path | Operation |
|---|---|---|
| POST | `/api/benchmarks/run` | Run Benchmark |

### /api/cache
Cache strategy preview / estimation

| Method | Path | Operation |
|---|---|---|
| GET | `/api/cache/preview` | Cache Preview |

### /api/chat
Chat sessions, generate, compare, HTML challenges

| Method | Path | Operation |
|---|---|---|
| POST | `/api/chat/compare` | Compare Models |
| POST | `/api/chat/generate` | Generate |
| POST | `/api/chat/generate/stream` | Generate Stream |
| POST | `/api/chat/generate/{session_id}/cancel` | Cancel Generate |
| GET | `/api/chat/html-challenges` | List Html Challenges |
| POST | `/api/chat/html-challenges` | Run Html Challenge |
| POST | `/api/chat/html-challenges/open-file` | Open Html Challenge File |
| DELETE | `/api/chat/html-challenges/{challenge_id}` | Delete Html Challenge |
| GET | `/api/chat/html-challenges/{challenge_id}` | Get Html Challenge |
| GET | `/api/chat/html-challenges/{challenge_id}/files/{slot_id}` | Get Html Challenge File |
| POST | `/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/repair` | Repair Html Challenge Slot |
| POST | `/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/retry` | Retry Html Challenge Slot |
| PATCH | `/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/validation` | Update Html Challenge Slot Validation |
| POST | `/api/chat/sessions` | Create Session |
| DELETE | `/api/chat/sessions/{session_id}` | Delete Session |
| PATCH | `/api/chat/sessions/{session_id}` | Update Session |
| POST | `/api/chat/sessions/{session_id}/delve/{message_index}` | Delve Message |
| GET | `/api/chat/sessions/{session_id}/documents` | List Session Documents |
| POST | `/api/chat/sessions/{session_id}/documents` | Upload Session Document |
| DELETE | `/api/chat/sessions/{session_id}/documents/{doc_id}` | Delete Session Document |
| POST | `/api/chat/sessions/{session_id}/fork` | Fork Session |
| POST | `/api/chat/sessions/{session_id}/variants` | Add Message Variant |
| POST | `/v1/chat/completions` | Openai Chat Completion |

### /api/diagnostics
Snapshot, log tail, runtime re-extract

| Method | Path | Operation |
|---|---|---|
| GET | `/api/diagnostics/log-tail` | Diagnostics Log Tail |
| POST | `/api/diagnostics/reextract-runtime` | Reextract Runtime |
| GET | `/api/diagnostics/snapshot` | Diagnostics Snapshot |

### /api/embeddings
OpenAI-compatible embeddings endpoint

| Method | Path | Operation |
|---|---|---|
| POST | `/v1/embeddings` | Openai Embeddings |

### /api/finetuning
LoRA fine-tuning hooks

| Method | Path | Operation |
|---|---|---|
| POST | `/api/finetuning/start` | Start Finetuning |
| GET | `/api/finetuning/status` | Get Finetuning Status |

### /api/health
Liveness probe

| Method | Path | Operation |
|---|---|---|
| GET | `/api/health` | Health |

### /api/images
Image generation (FLUX, SDXL, ...)

| Method | Path | Operation |
|---|---|---|
| POST | `/api/images/cancel` | Cancel Image Generation |
| GET | `/api/images/catalog` | Image Catalog |
| POST | `/api/images/download` | Download Image Model |
| POST | `/api/images/download/cancel` | Cancel Image Download |
| POST | `/api/images/download/delete` | Delete Image Download |
| GET | `/api/images/download/status` | Image Download Status |
| POST | `/api/images/generate` | Generate Image |
| GET | `/api/images/library` | Image Library |
| GET | `/api/images/outputs` | Image Outputs |
| DELETE | `/api/images/outputs/{artifact_id}` | Delete Image Output Endpoint |
| GET | `/api/images/outputs/{artifact_id}` | Image Output Detail |
| POST | `/api/images/preload` | Preload Image Model |
| GET | `/api/images/progress` | Image Generation Progress |
| GET | `/api/images/runtime` | Image Runtime Status |
| POST | `/api/images/unload` | Unload Image Model |

### /api/metrics
Live GPU metrics

| Method | Path | Operation |
|---|---|---|
| GET | `/api/metrics/gpu` | Gpu Snapshot |

### /api/models
Model catalog, library, download, convert, load/unload

| Method | Path | Operation |
|---|---|---|
| POST | `/api/models/convert` | Convert Model |
| POST | `/api/models/delete` | Delete Model Path |
| POST | `/api/models/download` | Download Model |
| POST | `/api/models/download/cancel` | Cancel Download |
| POST | `/api/models/download/delete` | Delete Download |
| GET | `/api/models/download/status` | Download Status |
| GET | `/api/models/hub-files` | Hub Files |
| GET | `/api/models/hub-search` | Hub Search |
| GET | `/api/models/list-weights` | List Weights |
| POST | `/api/models/load` | Load Model |
| GET | `/api/models/quantized-variants` | Quantized Variants |
| POST | `/api/models/reveal` | Reveal Model Path |
| GET | `/api/models/search` | Search Models |
| POST | `/api/models/unload` | Unload Model |
| GET | `/v1/models` | List Openai Models |

### /api/plugins
Plugin system (cache strategies, engines, tools, ...)

| Method | Path | Operation |
|---|---|---|
| GET | `/api/plugins` | List Plugins |
| POST | `/api/plugins/{plugin_id}/disable` | Disable Plugin |
| POST | `/api/plugins/{plugin_id}/enable` | Enable Plugin |

### /api/prompt
Prompt enhancer (Apple Silicon Qwen2.5-0.5B rewriter)

| Method | Path | Operation |
|---|---|---|
| POST | `/api/prompt/enhance` | Enhance Prompt |

### /api/prompts
Prompt library (templates)

| Method | Path | Operation |
|---|---|---|
| GET | `/api/prompts` | List Prompts |
| POST | `/api/prompts` | Create Or Update Prompt |
| DELETE | `/api/prompts/{template_id}` | Delete Prompt |

### /api/runtime
Current loaded-model runtime

| Method | Path | Operation |
|---|---|---|
| GET | `/api/runtime` | Runtime Status |

### /api/server
OpenAI-compatible local server controls

| Method | Path | Operation |
|---|---|---|
| GET | `/api/server/logs/stream` | Stream Server Logs |
| POST | `/api/server/shutdown` | Shutdown Server |
| GET | `/api/server/status` | Server Status |

### /api/settings
User preferences + storage location

| Method | Path | Operation |
|---|---|---|
| GET | `/api/settings` | Settings |
| PATCH | `/api/settings` | Update Settings |
| GET | `/api/settings/storage` | Storage Settings |
| POST | `/api/settings/storage` | Update Storage Path |
| POST | `/api/settings/storage/move` | Start Model Move |
| GET | `/api/settings/storage/move/status` | Model Move Status |

### /api/setup
Install + capability refresh endpoints

| Method | Path | Operation |
|---|---|---|
| GET | `/api/setup/gpu-bundle-info` | Gpu Bundle Info |
| POST | `/api/setup/install-cuda-torch` | Install Cuda Torch |
| POST | `/api/setup/install-gpu-bundle` | Start Install Gpu Bundle |
| GET | `/api/setup/install-gpu-bundle/status` | Install Gpu Bundle Status |
| POST | `/api/setup/install-longlive` | Start Install Longlive |
| GET | `/api/setup/install-longlive/status` | Install Longlive Status |
| POST | `/api/setup/install-mlx-video-wan` | Start Install Mlx Video Wan |
| GET | `/api/setup/install-mlx-video-wan/status` | Install Mlx Video Wan Status |
| POST | `/api/setup/install-mtplx` | Start Mtplx Install |
| GET | `/api/setup/install-mtplx/status` | Mtplx Install Status |
| POST | `/api/setup/install-package` | Install Pip Package |
| POST | `/api/setup/install-system-package` | Install System Package |
| GET | `/api/setup/mlx-video-wan/inventory` | Mlx Video Wan Inventory |
| GET | `/api/setup/mtplx-status` | Mtplx Status |
| POST | `/api/setup/refresh-capabilities` | Refresh Capabilities Endpoint |
| GET | `/api/setup/turbo-update-check` | Turbo Update Check |

### /api/system
System / GPU probe

| Method | Path | Operation |
|---|---|---|
| GET | `/api/system/gpu-status` | System Gpu Status |

### /api/tools
Built-in + MCP tool registry

| Method | Path | Operation |
|---|---|---|
| GET | `/api/tools` | List Tools |

### /api/video
Video generation (Wan, LTX-Video, ...)

| Method | Path | Operation |
|---|---|---|
| POST | `/api/video/cancel` | Cancel Video Generation |
| GET | `/api/video/catalog` | Video Catalog |
| POST | `/api/video/download` | Download Video Model |
| POST | `/api/video/download/cancel` | Cancel Video Download |
| POST | `/api/video/download/delete` | Delete Video Download |
| GET | `/api/video/download/status` | Video Download Status |
| POST | `/api/video/generate` | Generate Video |
| GET | `/api/video/library` | Video Library |
| GET | `/api/video/longlive` | Longlive Runtime Status |
| GET | `/api/video/mlx-runtime` | Mlx Video Runtime Status |
| GET | `/api/video/outputs` | Video Outputs |
| DELETE | `/api/video/outputs/{artifact_id}` | Delete Video Output Endpoint |
| GET | `/api/video/outputs/{artifact_id}` | Video Output Detail |
| GET | `/api/video/outputs/{artifact_id}/file` | Video Output File |
| POST | `/api/video/preload` | Preload Video Model |
| GET | `/api/video/progress` | Video Generation Progress |
| GET | `/api/video/runtime` | Video Runtime Status |
| POST | `/api/video/unload` | Unload Video Model |

### /api/workspace
Top-level workspace snapshot

| Method | Path | Operation |
|---|---|---|
| GET | `/api/workspace` | Workspace |

### /api/workspaces
Workspace knowledge stacks (shared RAG corpus)

| Method | Path | Operation |
|---|---|---|
| GET | `/api/workspaces` | List Workspaces |
| POST | `/api/workspaces` | Create Workspace |
| DELETE | `/api/workspaces/{workspace_id}` | Delete Workspace |
| PATCH | `/api/workspaces/{workspace_id}` | Update Workspace |
| POST | `/api/workspaces/{workspace_id}/documents` | Upload Workspace Document |
| DELETE | `/api/workspaces/{workspace_id}/documents/{doc_id}` | Delete Workspace Document |
