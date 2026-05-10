# ChaosEngineAI v0.8.0 — Refactor & Audit Plan

Branch: `feature/refactor-n-audit` (off v0.7.6).

## Goals

1. Split god-objects in Python backend, frontend, and Rust shell into focused modules.
2. Lift route + feature-tab test coverage from ~30% to ≥60% before touching mega-files.
3. Close cross-OS gaps so Windows/Linux are first-class with macOS.
4. No regressions: every phase merged on green tests + ≤5% perf drift on the 3 reference gens.

## Pre-refactor metrics (v0.7.6, captured 2026-05-09)

| Metric | Value | Source |
|---|---|---|
| Python tests | 1,284 pass + 1 skip | `pytest tests/` |
| TS tests | 335 pass (28 files) | `vitest run` (scoped to `src/`) |
| `tsc --noEmit` | clean | `npx tsc --noEmit` |
| Python LOC (backend_service + cache_compression + dflash) | ~31k | `cloc` |
| Frontend LOC (src/) | ~36.8k | `cloc` |
| Rust LOC (src-tauri/src/) | 1,814 | `wc -l` |
| Untested route modules | 18 of 21 | manual cross-ref |
| Untested feature tabs | 40 of 42 | manual cross-ref |

## Progress through 2026-05-10 (91 commits on `feature/refactor-n-audit`)

| File | Original | Now | Δ |
|---|---|---|---|
| `state/__init__.py` | 4,418 | 4,089 | -329 |
| `inference/__init__.py` | 3,574 | 1,180 | -2,394 |
| `image_runtime/__init__.py` | 2,097 | 1,366 | -731 |
| `video_runtime/__init__.py` | 2,378 | 1,593 | -785 |
| `mlx_worker.py` | 2,115 | 1,927 | -188 |
| `routes/setup/__init__.py` | 1,932 | 353 | -1,579 |
| `routes/html_challenges/__init__.py` | 1,183 | 460 | -723 |
| `helpers/huggingface.py` | 703 | 525 | -178 |
| `helpers/gpu.py` | 568 | 355 | -213 |
| `helpers/discovery.py` | 806 | 429 | -377 |
| `helpers/system.py` | 559 | 252 | -307 |
| `helpers/documents.py` | 586 | 478 | -108 |
| `src/App.tsx` | 2,334 | 2,170 | -164 |
| `src/features/chat/HtmlChallengeTab.tsx` | 2,535 | 1,677 | -858 |
| `src/features/video/VideoStudioTab.tsx` | 1,796 | 1,712 | -84 |
| `src/hooks/useChat.ts` | 1,203 | 1,131 | -72 |
| `src/hooks/useImageState.ts` | 862 | 846 | -16 |
| `src/hooks/useVideoState.ts` | 1,211 | 1,126 | -85 |
| `src/api/index.ts` | 1,430 | 559 | -871 |
| `src/types.ts` | 1,378 | 230 | -1,148 |
| `helpers/images.py` | 983 | 751 | -232 |
| `helpers/video.py` | 769 | 565 | -204 |
| **Mega-file shrink total** | 35,420 | **23,674** | **-11,746 LOC** |

Tests posture across all 91 commits: **1,302 Python pass + 1 skip / 340 TS pass / tsc clean**. Zero regressions; coverage gate (60% Python) holds on every phase.

## Mega-file inventory

### Python (>1,800 LOC)
- `backend_service/state.py` — 4,418
- `backend_service/inference.py` — 3,574
- `backend_service/video_runtime.py` — 2,378
- `backend_service/mlx_worker.py` — 2,115
- `backend_service/image_runtime.py` — 2,097
- `backend_service/routes/setup.py` — 1,932

### Frontend (>1,000 LOC)
- `src/features/htmlchallenge/HtmlChallengeTab.tsx` — 2,535
- `src/App.tsx` — 2,334
- `src/features/video/VideoStudioTab.tsx` — 1,796
- `src/api.ts` — 1,430
- `src/types.ts` — 1,378
- `src/hooks/useVideoState.ts` — 1,211
- `src/hooks/useChat.ts` — 1,203
- `src/features/images/ImageStudioTab.tsx` — 1,178

### Rust
- `src-tauri/src/lib.rs` — 1,808 (six concerns in one file)

## Phasing

Each phase = 1 PR. Tests green at each boundary. No big-bang merge.

### Phase 0 — Safety net

1. Vitest config scoped to `src/` to drop phantom worktree tests. **DONE 2026-05-09.**
2. Wire `pytest --cov=backend_service --cov=cache_compression --cov=dflash` and `vitest --coverage`. Record numbers in `COVERAGE_BASELINE.md`.
3. Contract tests for 18 untested route modules (happy path + 1 error each).
4. Smoke render tests for top 5 untested feature tabs (mount + assert no throw).
5. CI matrix: macOS + Ubuntu + Windows running pytest, vitest, tsc, cargo check.
6. Delete `chaos_readme.md` (3-line stub, no refs).
7. Coverage gate in CI: fails if line coverage drops vs. baseline.

### Phase 1 — Python backend split

**1a. `state.py` 4,418 → facade + 5 modules.**

**PARTIAL** (Phase 1a-1 through 1a-6; commits `8a26a48` → `753cd9a`):
- `state/logs.py` — LogManager (log + activity ring buffers + subscribers)
- `state/metrics.py` — cache labels + profile change reasons + metrics payloads (11 pure functions)
- `state/_helpers.py` — module-level helpers: `_compose_chat_system_prompt`, `_build_sampler_overrides`, `_build_history_with_reasoning`, title-generation utilities, `_read_text_tail`, `_spawn_snapshot_download`, `_normalize_remote_provider_api_base`, `_CATALOG_REF_ALIASES` (1a-3).
- `state/documents.py` (1a-4) — 8 helpers: `session_docs_dir` / `workspace_docs_dir` (filesystem-safe path resolvers), `list_session_documents`, `upload_session_document` (bytes → file + chunked .json sidecar), `delete_session_document`, `upload_workspace_document` (Phase 3.7 variant), `delete_workspace_document`, `retrieve_session_context` (RAG retriever merging session + workspace corpora through DocumentIndex).
- `state/benchmarks.py` (1a-5) — `run_benchmark` orchestration across perplexity / task-accuracy / throughput modes + `append_benchmark_run` rolling-window persistence.
- `state/openai_compat.py` (1a-6) — `openai_models` + `openai_embeddings` + `openai_chat_completion` (`/v1/*` endpoints; auto-load + sampler + response_format mapping + streaming branch).
- `state/payloads.py` (1a-7) — `workspace` (`/api/workspace` aggregate composing system snapshot + library + recommendation + featured models + runtime status + benchmarks + logs/activity + cache-preview math, with the heavy per-process annotation pass that joins `runningLlmProcesses` against the runtime's active + warm engines) + `server_status` (`/api/server/status` with loading-stage breakdown).
- `state/settings_state.py` (1a-8) — `settings_payload` (user-visible settings shape with masked API keys / HF token, per-directory model counts, resolved output dirs) + `update_settings` (full settings patch: model dir normalisation, output-path validation, data-dir migration, remote-provider key preservation, library cache refresh).
- `state/sessions.py` (1a-9) — 13 helpers covering the chat session lifecycle: `default_session_model`, `promote_session`, `persist_sessions`, `unique_session_title`, `auto_session_title`, `normalize_auto_generated_session_titles`, `ensure_session`, `create_session`, `add_message_variant` (Phase 2.5 sibling variants), `delve_message` (Phase 3.6 critique pass), `fork_session` (Phase 2.4 thread branching with parentSessionId linkage), `update_session`, `delete_session`.
- `state/downloads.py` (1a-10) — full HF download flow: `start_download` (preflight + ProgressTqdm + background snapshot_download worker) + `download_status`, `cancel_download`, `delete_download`, `loaded_model_matches_repo_cache`, `unload_repo_from_runtimes`. Inner `_download_worker` thread closes over (state, repo, allow_patterns, download_token, validation_error_fn).
- `state/generation.py` (1a-11) — `generate` (synchronous chat completion with profile cascade + RAG + agent loop dispatch, ~258 LOC) + `generate_stream` (SSE streaming with five guards: memory pre-flight, output-length runaway, repetition / loop, tok/s floor, in-stream panic + thermal — ~576 LOC).
- `state/lifecycle.py` (1a-12) — `load_model` (catalog + library validation, in-place profile apply vs full reload decision, warm pool eviction, runtime.load_model dispatch with progress callback), `unload_model`, `convert_model`, `reveal_model_path`, `delete_model_path`.

state/__init__.py: 4418 → 860 LOC (-3558, -81%). Class methods that moved out are now 1-3 line thin wrappers preserving the public surface. The facade is essentially just construction, validation, and wiring now.

```
backend_service/state/
  __init__.py          # ChaosEngineState facade — public API unchanged
  logs.py              # LogManager + ring buffers          [done]
  metrics.py           # cache labels + profile metrics     [done]
  session_manager.py   # chat sessions, history             [pending]
  model_manager.py     # model load/unload/discovery state  [pending]
  inference_orchestrator.py                                 [pending]
  benchmark_state.py                                        [pending]
  settings_state.py                                         [pending]
```

**1b. `inference.py` 3,574 → engines/ subpackage.**

**DONE** (Phase 1b-1 through 1b-8; commits `cb1aed3` → `f308d9b`).
RuntimeController extracted to `inference/controller.py` (~1050 LOC,
re-exported from the package). `inference/__init__.py` is now 97 LOC
of public re-exports only.

Earlier phases:
- `inference/_constants.py` — 5 timeout / workspace constants
- `inference/_utils.py` — 9 shared helpers (_now_label, _normalize_message_content, _read_text_tail, _append_runtime_note, _http_json, _find_open_port, _resolve_gguf_path, _is_local_target, _looks_like_gguf)
- `inference/base.py` — 4 dataclasses + RepeatedLineGuard + BaseInferenceEngine
- `inference/jsonrpc.py` — JsonRpcProcess subprocess bridge
- `inference/simple_engines.py` — RemoteOpenAIEngine + MockInferenceEngine
- `inference/mlx_engine.py` — MLXWorkerEngine
- `inference/llama_cpp_engine.py` — LlamaCppEngine + 8 llama-specific helpers + 4 constants
- `inference/binaries.py` — `_json_subprocess` + llama-server / llama-cli / MLX-python resolvers (1b-6)
- `inference/capabilities.py` — `_capability_cache` + `_initial_backend_capabilities` + `_probe_native_backends` + `get_backend_capabilities` (1b-6)
- `inference/conversion.py` — mlx-lm supported-arch probe + `_peek_hf_model_type` + `_nearest_supported_arch` + `_default_conversion_output` + `_bytes_to_gb` + `_path_size_bytes` (1b-7)

inference/__init__.py: 3574 → 1180 (-2394). RuntimeController (~1050 LOC) is the only big class still inline; deferred — its helper graph is the most cross-cutting in the package.

**1c. `video_runtime.py` + `image_runtime.py` → runtimes/{image,video}/.**

**PARTIAL** (Phase 1c-1 through 1c-12, commits `b5ea526` → `c0a097c`):
- `image_runtime/` package: types + repos + snapshot + device + placeholder_engine + mflux_engine + transformer_loaders extracted (image/__init__.py: 2097 → 1069):
  - `image_runtime/transformer_loaders.py` (1c-11) — eight stateless quantised-transformer / device-probe helpers: `try_load_nf4_flux_transformer` (bitsandbytes NF4, CUDA), `try_load_int8wo_flux_transformer` (TorchAO int8wo, MPS), `try_load_gguf_transformer` (single-file `.gguf`, cross-platform), `try_load_nunchaku_transformer` (FU-023 SVDQuant int4, CUDA), `maybe_enable_fp8_layerwise` (FU-024 layerwise casting, SM ≥ 8.9), `should_use_model_cpu_offload` (FLUX-on-CUDA whole-component swap), `detect_device` (CUDA → MPS → CPU probe).
- `video_runtime/` package: types + device + repos + defaults + warmup + transformer_loaders extracted (video/__init__.py: 2378 → 1357):
  - `video_runtime/device.py` — probe helpers (`_resolve_video_seed`, `_resolve_video_python`, `_detect_device_memory_gb`, `_guess_video_expected_device`, `_windows_cuda_unavailable_message`)
  - `video_runtime/repos.py` — `PIPELINE_REGISTRY`, GGUF/NF4 transformer class lookups, per-model defaults table, prompt-enhancement suffixes + `_enhance_prompt`
  - `video_runtime/defaults.py` — memory footprint estimator, slicing gate, scheduler classes, Wan frame alignment, `_resolve_video_defaults`, frame interpolation, dep tuples + `_find_missing`
  - `video_runtime/warmup.py` — torch + dep prewarm singleton + `start_torch_warmup` / `torch_warmup_status`
  - `video_runtime/transformer_loaders.py` (1c-12) — five stateless helpers: `try_load_gguf_transformer`, `try_load_bnb_nf4_transformer` (CUDA), `swap_distill_transformers` (FU-019 lightx2v 4-step Wan 2.2 A14B distill), `detect_device`, `preferred_torch_dtype` (bf16/fp16/fp32 picker with M1-MPS bf16 capability probe + env opt-out).

**Remaining**: `_ensure_pipeline` is the orchestrator that calls the loaders + the LoRA fuse / preview-VAE / distill swap / SDXL VAE swap / fp8 layerwise / scheduler swap path. Heavy state mutation (self._pipeline / self._loaded_repo / variant key / runtime notes). Same shape on the video side. Extract into `runtimes/common/_pipeline_loader.py` post-v0.8.0 once both engines are stable.

**1d. `routes/setup.py` 1,932 → setup/ package with 6 focused submodules.** **DONE** (Phase 1d-1 through 1d-3c, commits `6181c1b` → `afc70f3`):
- `setup/longlive.py` + `setup/wan_install.py` — LongLive + Wan background installers (1d-1).
- `setup/turbo.py` — llama-server-turbo update-check (1d-2).
- `setup/_install_helpers.py` — shared pip-install primitives (`_run_pip_install`, `_extras_site_packages`, `_cleanup_mlx_video_shadow_metadata`, torch wheel walk + purge utilities) (1d-3a).
- `setup/cuda_torch.py` — CUDA torch recovery installer that walks the cu124 → nightly cu128 download indexes (1d-3b).
- `setup/gpu_bundle.py` — one-click "Install GPU support" flow (torch + diffusers + transformers + video runtime deps) with background-job worker (1d-3c).

setup/__init__.py: 1,932 → 353 LOC (~82% reduction). Setup is now a clean package; the only synchronous endpoints left in `__init__` are `install-package` / `install-system-package` / `refresh-capabilities` plus the install-package catalogues + the manual-install message map.

**1d-4. `routes/html_challenges.py` 1,183 → html_challenges/ package.** **DONE** (commit `f31653c`). Two-way split:
- `html_challenges/__init__.py` — Pydantic request models, `router`, 9 endpoints (list / get / delete / file / open-file / retry / repair / validation / run).
- `html_challenges/_helpers.py` — 45 underscore helpers (manifest I/O, HTML extraction + validation, payload shaping, `_stream_html_challenge_slot`).

**1e. helpers/ regrouping into media/ models/ system/ ui/ storage/ inference/ finetune/ remote/ filter/ subpackages. Public re-exports preserve call sites.** **DONE** (Phase 1e-1 through 1e-13, commits `9b61377` → `215eeab`). 13 sibling modules extracted across the largest helpers files:

Image / video media:
- `helpers/image_artifacts.py` — daily-folder gallery layout, JSON sidecars, SVG placeholder renderer (1e-1).
- `helpers/image_validation.py` — repo predicates + friendly HF download-error translation (1e-2).
- `helpers/video_artifacts.py` — mirror of image_artifacts for the video gallery (1e-3).
- `helpers/mlx_video_validation.py` — mlx-video LTX-2 / LTX-2.3 component-folder probe (1e-4).

Discovery / model classification:
- `helpers/quantization.py` — NVFP4/NVINT4 rejection + regex bit-width inference + dtype walk (1e-5).
- `helpers/model_classifier.py` — keyword tables + ``_looks_like_{draft,video,image}_model`` heuristics (1e-6).
- `helpers/snapshot_integrity.py` — sharded safetensors + GGUF directory probes + ``_list_weight_files`` (1e-7).
- `helpers/model_family_payload.py` — catalog → dashboard payload renderer + cross-platform Reveal-in-Finder (1e-8).

Hugging Face:
- `helpers/hf_cache_paths.py` — HF cache root + repo dir + downloaded bytes + active snapshot dir (1e-9).
- `helpers/hf_format.py` — ISO datetime + Updated/Released label + number formatters (1e-9).
- `helpers/hf_errors.py` — traceback condenser + friendly download-error rewriter for gated / 404 / DNS / PyYAML failures (1e-10).

System:
- `helpers/system_processes.py` — top + psutil cluster (5 helpers; macOS-aware Activity-Monitor-accurate footprint) (1e-11).
- `helpers/system_hardware.py` — chip / OS summary, version, GPU + battery + compressed memory + runtime label (1e-12).

Documents:
- `helpers/document_text.py` — file extraction + sliding-window chunking + tokenisation primitives (1e-13).

Mega-file shrink across helpers/: images.py 983 → 751, video.py 769 → 565, discovery.py 806 → 429, huggingface.py 703 → 525, system.py 559 → 252, documents.py 586 → 478. Re-exports preserve every existing import path; 7 helpers files (gpu, settings, prompts, formatting, persistence, etc.) left untouched as already-focused.

**1f. `mlx_worker.py` 2,115 → request helpers + worker.** **MOSTLY DONE** (Phase 1f-1 through 1f-9, commits `b27ebab` → `77588b6`).
- `mlx_worker_request.py` — `_normalize_message_content`, `_sanitize_messages`, `_extract_top_logprobs`, `_build_mlx_sampler`, `_sampler_seed`, `_apply_mlx_seed`, `_format_tools_for_prompt` (1f-1). Re-exported from `mlx_worker` so `vllm_engine`'s direct import keeps working.
- `mlx_worker_prompt.py` — `TranscriptLoopFilter` + `_build_prompt_text` + Gemma fold-system + plain-chat fallback + `_should_retry_cache_failure` + `_merge_runtime_notes` (1f-2).
- `mlx_worker_io.py` — JSON IPC channel: `_JSON_OUT`, `_install_stdio_redirect`, `_emit`, `emit_progress` (1f-3).
- `mlx_worker_diagnostics.py` — `_UNSUPPORTED_QUANT_ALGOS` + `_reject_unsupported_quant` model-config probe + `probe()` runtime-capability subcommand + `gguf_metadata()` GGUF-file subcommand (1f-4).
- `mlx_worker_multimodal.py` — `decode_images_to_paths`, `format_multimodal_prompt`, `vlm_generate_kwargs`, `generate_multimodal`, `stream_generate_multimodal` (mlx-vlm helpers + sync/streaming generation entrypoints; WorkerState methods now thin-wrap) (1f-5, 1f-8).
- `mlx_worker_cache.py` — `runtime_fields` + `make_mlx_cache` (pure cache profile helpers; class methods now thin-wrap) (1f-6).
- `mlx_worker_eval.py` — `eval_perplexity` + `eval_task_accuracy` (eval entrypoints; class methods now thin-wrap) (1f-7).
- `mlx_worker_loader.py` — `resolve_local_snapshot` HF snapshot-download front half of `load_model` with ProgressTqdm + gated/404/auth → user-readable `RuntimeError` translation (1f-9).

mlx_worker.py: 2,115 → 1,227 LOC (-888, -42%). The remaining WorkerState methods (load_model heartbeat half + load tail, generate / stream_generate, _generate_dflash, _generate_ddtree, _generate_standard, _apply_cache_profile, _apply_triattention_mlx_compressor) all mutate enough instance state — model + tokenizer + processor + config + dflash bundle + ddtree handles + cache profile + speculative_decoding + tree_budget + loaded_model_ref — that further extraction needs a `WorkerContext` dataclass to bundle the context cleanly. Deferred to a v0.8.1 follow-up.

**Verify each step:** `pytest`, live smoke gens (text + image + video), `python -c "from backend_service.app import build_app; build_app()"` clean import.

### Phase 2 — Frontend split

**2a. `api.ts` 1,430 → src/api/{chat,image,video,models,setup,admin}.ts.** **DONE** (Phase 2-1 through 2-6, commits `dea6a54` → `68fed4f`). 6 commits, 4,453 LOC across 6 domain modules. Live-binding circular re-exports preserve call sites.

**2b. `types.ts` 1,378 → src/types/ package with 11 domain files.** **DONE** (Phase 2b-1 through 2b-7, commits `2d91fa6` → `d4ab359`):
- `types/system.ts` — TabId, SidebarGroupId, SidebarMode, SystemStats, Recommendation (2b-1).
- `types/hub.ts` — HubModel, HubFile, HubFileListResponse (2b-1).
- `types/progress.ts` — GenerationProgressSnapshot (2b-1).
- `types/models.ts` — ModelLaunchMode, ModelVariant, ModelFamily, LibraryItem, ModelDirectorySetting, LaunchPreferences (2b-2).
- `types/server.ts` — runtime / server-status / capability cluster (2b-3).
- `types/settings.ts` — AppSettings, RemoteProvider, install logs, UpdateSettingsPayload (2b-4).
- `types/chat.ts` — extended with full chat domain: ToolCallInfo, ChatMessage, ChatSession, GeneratePayload, SamplerOverrides, etc. (2b-5).
- `types/image.ts` — extended with ImageModelVariant, ImageGenerationPayload, ImageRuntimeStatus, etc. (2b-6).
- `types/video.ts` — extended with VideoModelVariant, VideoGenerationPayload, VideoRuntimeStatus, etc. (2b-6).
- `types/benchmarks.ts` — PerfTelemetry, GenerationMetrics, BenchmarkResult, BenchmarkRunPayload (2b-7).
- `types/observability.ts` — LogEntry, ActivityItem, PreviewMetrics (2b-7).

src/types.ts: 1,378 → 230 LOC (~83% reduction). Re-exports preserve every existing import path; barrel ``src/types/index.ts`` aggregates the 11 sub-files. Remaining 230 LOC: ``WorkspaceData`` (dashboard aggregator), ``LoadModelPayload``, ``ConvertModelPayload``, ``ConversionResult``, ``ConvertModelResponse``, ``TauriBackendInfo`` — small payloads that don't justify their own file yet.

**2c. Mega-hooks + god components splits.** **PARTIAL** (Phase 2c-1 through 2c-4, commits `ce55f4b` → `50ce5dd`):
- `features/chat/temperatureOverride.ts` + `features/chat/reasoningEffort.ts` — per-session localStorage helpers extracted from `useChat.ts`. Plus `readSamplerPayload` collapsed to a one-liner via existing `samplerOverrides.ts` helpers (Phase 2c-1).
- `components/CapabilityStrip.tsx` — de-duped 3 identical inline ``renderCapabilityIcons`` implementations (App + MyModelsTab + OnlineModelsTab) into a single shared component (Phase 2c-2).
- `hooks/useCudaTorchInstall.ts` — extracted CUDA torch install flow (3 state slots + handler) from App.tsx; accepts an ``onAfterInstall`` callback so App keeps firing the imgState/videoState refresh probes that clear the warning banner (Phase 2c-3).
- `features/chat/optimisticTurns.ts` (Phase 2c-4) — four pure state helpers pulled out of useChat: `appendOptimisticTurn` (push user + empty-assistant pair in `prompt_eval` phase), `replaceOptimisticAssistant` (fill the empty turn after stream completes; falls back to appending fresh pair if the optimistic turn was already swept), `rollbackOptimisticTurn` (drop the empty pair on stream error), `mergeSessionMetadata` (shallow patch). Hook keeps 3-line local wrappers that close over setWorkspace.
- `features/image/downloadActions.ts` + `features/image/studioPresets.ts` + `features/image/galleryActions.ts` (Phase 2c-5, commits `2135f1d` → `30761b5`) — 12 handlers pulled out of useImageState across three cohesive sibling modules. Each helper takes a typed deps object; hook keeps one-line wrappers that close over the live setters. Mutual dependencies (e.g. `varyImageSeed` calling `hydrateFormFromArtifact` + `submitImageGeneration`) injected as callback deps so the modules stay decoupled.
- `features/video/downloadActions.ts` + `features/video/modelLifecycle.ts` + `features/video/installActions.ts` (Phase 2c-6) — 11 handlers + 2 pure helpers pulled out of useVideoState. Each handler takes its dependencies as kwargs. Hook keeps thin wrappers.
- `features/chat/html_challenge/` package (Phase 2c-7) — 5 child components (`ChallengeSetupPanel`, `ChallengeSlotPanel`, `ChallengeModelCard`, `ChallengePickerModal`, `ChallengeHistoryCombobox`) + 2 helper modules (`challengeApi.ts` fetch wrappers, `htmlChallengeTabHelpers.ts` pure derived-value helpers + slot-state reducers) pulled out of HtmlChallengeTab. Composition root keeps streaming/abort/run-retry-repair orchestration only.

useChat.ts: 1,203 → 1,067 LOC. useImageState.ts: 846 → 809 LOC. useVideoState.ts: 1,126 → 899 LOC. HtmlChallengeTab.tsx: 1,677 → 1,103 LOC. App.tsx: 2,334 → 2,253 LOC.

**Remaining 2c/d targets:**
- `useChat` → `useChatStreaming` + `useChatHistory` + `useChatInput` (still ~1,131 LOC)
- `useVideoState` 1,211 → `useVideoConfig` + `useVideoGeneration` + `useVideoLibrary`
- `useImageState` 862 → analogous
- `App.tsx` 2,253 → composition root only
- `HtmlChallengeTab.tsx` 2,535 → `ChallengeRunner` + `ChallengeEditor` + `ChallengePreview` + `ChallengeLibrary`
- `VideoStudioTab.tsx` 1,796 + `ImageStudioTab.tsx` 1,178 → shared `<StudioControls>` + `<StudioPreview>` + `<StudioLibrary>` shell

**2e. Inline single-use hooks** — `useGpuStatus`, `useSidebarPrefs`, `useUiScale` collapse into `App.tsx`.

**Verify:** `npm test`, `npx tsc --noEmit`, dev server boots, click-through 5 main tabs.

### Phase 3 — Rust shell split

**PARTIAL** (Phase 3-1, commit `c244618`):
- `src-tauri/src/binaries.rs` — `resolve_llama_server`, `resolve_llama_server_turbo`, `resolve_llama_cli`, `resolve_sd_cpp`, `resolve_candidate`, `find_in_path`. Each honours an env-var override first, falls back to `~/.chaosengine/bin/<name>` for managed installs, then walks `PATH` (with `.exe` suffix on Windows).

lib.rs: 1335 → 1249 LOC. Pre-existing `settings.rs` / `probe.rs` / `lease.rs` / `orphans.rs` / `windows_job.rs` modules unchanged.

```
src-tauri/src/
  lib.rs                # public API + state init only
  binaries.rs           [done]
  settings.rs           [done]
  probe.rs              [done]
  lease.rs              [done]
  orphans.rs            [done]
  windows_job.rs        [done]
  runtime/{extraction,manifest}.rs   [pending — embedded runtime extraction is the big one]
  env/{apply,python}.rs              [pending]
  backend/{manager,lifecycle}.rs     [pending — BackendManager impl]
```

Add explicit `#[cfg(target_os = "linux")]` where Linux currently rides on `#[cfg(unix)]` but should diverge from macOS.

**Verify:** `cargo check --all-targets`, `cargo clippy -- -D warnings`, `cargo test`, `npm run tauri dev` boots.

### Phase 4 — Cross-OS parity **DONE**

1. PowerShell ports: `update-llama-turbo.ps1`, `update-sdcpp.ps1`. **DONE** (commit `861de0a`). Both delegate to their `build-*.ps1` siblings after a version-file fast-exit so MSVC/CUDA toolchain plumbing stays in one place.
2. `pre-build-check.sh` → port to Node (`pre-build-check.mjs`) — single script across all 3 OSes. **DONE** (Phase 4-2). 7 checks ported (pytest, vitest, tsc, NOTICES grep, Python cache-strategy probe, upstream git ls-remote, binary file existence). Wired as `npm run pre-build-check`. Live smoke against the dev machine: 8 PASS / 0 FAIL / 1 WARN (turbo update available, expected).
3. ~~De-dupe `build-X.sh` + `update-X.sh` overlap → unified `manage-X.sh build|update|status`.~~ **DROPPED** — build scripts handle clone-or-fetch; update scripts add the version-file fast-exit. Two narrow scripts read clearer than one with a subcommand router.
4. ~~Rename `update-llama-cpp.sh` → `check-llama-cpp.sh` (info-only, name lied).~~ **DROPPED** — original audit was wrong: the script does rebuild llama-server (cmake configure + build).
5. CI matrix flips Windows/Linux from advisory to required. **DONE** (Phase 4-5, paired with 4-2). `windows-latest` job in `.github/workflows/build.yml` now has `advisory: false` so a Windows-specific regression blocks the PR the same way a macOS / Ubuntu failure does. Linux was already required.

### Phase 5 — Performance pass **STARTED**

Phase 5-1 (commit `81a81b7`): ``scripts/perf-gate.py`` comparator added.
Reads JSON output from ``perf-baseline.py`` and validates each metric
against the captured floor (default ±5% tolerance, configurable).
Initial floor: ``text.tokens_per_second ≥ 297 tok/s`` (Qwen2.5-0.5B
4-bit MLX on Apple Silicon, captured 2026-05-09). Image + video
floors stay TBD until real captures land.

Phase 5-2 (commit `b5d9308`): ``.github/workflows/perf-gate.yml`` ships
a dedicated CI workflow that runs ``perf-baseline.py`` on
``macos-latest`` with HF cache restore and pipes the JSON into
``scripts/perf-gate.py``. Trigger surface is manual + label-driven —
``workflow_dispatch`` (Actions tab "Run workflow") or adding the
``perf-gate`` label to a PR. We don't bolt this onto every push because
the cheapest gen (text) needs ~700 MB of cached MLX weights, and the
image/video gens pull multi-GB diffusers checkpoints. The workflow
upload-artifacts the captured baseline JSON for 30 days. The
comparator's ``_read_metric`` was also rewritten to navigate the actual
``{"results": [{"label": ..., ...}]}`` shape ``perf-baseline.py``
emits — the original draft assumed a label-keyed nested dict.

Profile-driven only:
1. **Backend startup:** `python -X importtime backend_service.app`. Target import < 2s. Lazy-import torch/diffusers/mlx until first model load.
2. **Frontend bundle:** `vite build` + `rollup-plugin-visualizer`. Code-split video/image/chat tabs. Mega tabs out of initial chunk.
3. **Re-render audit:** React Profiler on 4 mega-hooks. Memo only where measured.
4. **Inference parity:** wall-time for the 3 reference gens (text/image/video) within 5% of `PERF_BASELINE.md`.

### Phase 6 — Docs + tag

1. Update `CLAUDE.md` directory map.
2. `THIRD_PARTY_NOTICES.md` sweep.
3. Final coverage check ≥ baseline.
4. Tag `v0.8.0`.

## Multi-OS guardrails (active throughout)

- Every `#[cfg(target_os)]` / `platform.system()` branch reviewed for 3-platform coverage.
- Path handling: `pathlib.Path` (Python), `std::path::PathBuf` (Rust), `path.posix` vs `path.win32` explicit (Node).
- Subprocess: list-form `subprocess.run([...])` only — no shell strings.
- Binary resolution: probe `~/.chaosengine/bin/X` AND `X.exe` on Windows for every binary.
- Filesystem tests: `tmp_path` fixture; no `/tmp` hardcoding.

## Performance guardrails

- No phase merge without `PERF_BASELINE.md` re-run within 5% drift on the 3 reference gens.
- Lazy imports: torch / diffusers / mlx / transformers / nunchaku at first-use, not module top.
- No new wrappers around hot paths (callbacks-on-step-end, sampler registry, KV cache ops). Extract code, don't wrap it.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| `state.py` split breaks subtle invariants | Facade preserves public API; integration tests catch wire breakage |
| Module rename storms break imports | Re-export shims in `__init__.py`; deprecation cycle |
| Lazy-loaded chunks regress UX | Suspense fallbacks + manual click-through QA matrix |
| Refactor masks real perf regression | Phase 0 baselines; Phase 5 gates merge |
| Windows CI flake blocks PRs | `windows-latest` advisory until Phase 4 |

## Reference gens (PERF_BASELINE.md)

- Text: Qwen2.5-0.5B-Instruct-4bit MLX, 256 tok prompt → 128 tok output, capture tok/s
- Image: FLUX.1-schnell, 4 steps, 1024×1024, capture wall-time
- Video: Wan2.1-T2V-1.3B, 5 frames, 480×272, 4 steps, capture wall-time

Re-run before Phase 0 PR + after Phase 4 PR + final.
