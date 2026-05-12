# Changelog

## v0.8.0 - 2026-05-10

### Refactor + audit

Multi-week pass through the largest backend / frontend modules to land
the v0.8.0 modularisation goal. Zero feature regressions — 1,302 Python
tests + 340 TypeScript tests pass before and after every commit; all
type checks (mypy, tsc) clean.

**Bundled MLX worker memory leak fix.** `JsonRpcProcess.close()` now
captures and nulls `self.process` up-front + wraps the post-kill
`wait()` in `try/except TimeoutExpired` with a 1 s ceiling, mirroring
`LlamaCppEngine._cleanup_process`. Without the fix, force-killing a
worker that held ~47 GB of MLX weights routinely raised
`TimeoutExpired` on the macOS vm_map teardown, the exception was
swallowed by the route layer's broad `except Exception: pass`,
`self.process` was never nulled, and the next load spawned a second
worker alongside the dying one — Activity Monitor showed two ~47 GB
Python processes; `/api/server/status` reported one model.

**Backend (`backend_service/`)** — major shrinks across the four
biggest modules:

- `state/__init__.py`: 4,418 → 860 LOC (-81%) via
  `state/{logs,metrics,_helpers,documents,benchmarks,openai_compat,
  payloads,settings_state,sessions,downloads,generation,lifecycle}.py`.
  Class methods that moved out are 1-3 line thin wrappers; tests that
  patch `_describe_process` / `_spawn_snapshot_download` /
  `threading.Thread` / `subprocess.Popen` retarget to the new module
  paths; no external import path changes. The facade is essentially
  just construction, validation, and wiring now.
- `inference/__init__.py`: 3,574 → 97 LOC (-97%) via the existing
  `engines/` subpackage (RemoteOpenAIEngine + MockInferenceEngine +
  MLXWorkerEngine + LlamaCppEngine + binaries + capabilities +
  conversion) plus `controller.py` (the full ~1,050 LOC
  RuntimeController class). The package's `__init__` is now just the
  public re-export surface.
- `mlx_worker.py`: 2,115 → 318 LOC (-85%) via
  `mlx_worker_{request,prompt,io,diagnostics,multimodal,cache,eval,loader,
  lifecycle,speculative,generate}.py`. WorkerState methods are 1-3 line
  wrappers; load_model + unload_model + update_profile + cache profile
  helpers + DFLASH + DDTree speculative generation + plain text /
  streaming generation paths + JSON IPC channel + HF snapshot download
  + perplexity / task-accuracy eval + multimodal paths all sit in
  their own cohesive modules now.
- `image_runtime/__init__.py`: 2,097 → 992 LOC (-53%) and
  `video_runtime/__init__.py`: 2,378 → 1,018 LOC (-57%) via
  `transformer_loaders.py` + `pipeline_helpers.py` per package +
  the existing `{types,repos,snapshot,device,placeholder_engine,
  mflux_engine,defaults,warmup}` modules. Quantised transformer
  loaders (NF4, int8wo, GGUF, Nunchaku SVDQuant, BitsAndBytes NF4,
  lightx2v Wan distill swap) + FP8 layerwise casting + device
  probes + dtype pickers + per-step pipeline callbacks +
  finalize_config / swap_scheduler / build_pipeline_kwargs +
  encode_frames_to_mp4 all moved out.
- `routes/setup/`: 1,932 → 353 LOC (-82%) via
  `setup/{longlive,wan_install,turbo,_install_helpers,cuda_torch,gpu_bundle}.py`.
- `routes/html_challenges/`: 1,183 → 2-file package
  (`__init__.py` + `_helpers.py`).
- `helpers/`: 14 sibling modules pulled out of the original
  helpers files (image_artifacts, image_validation, video_artifacts,
  mlx_video_validation, quantization, model_classifier,
  snapshot_integrity, model_family_payload, hf_cache_paths, hf_format,
  hf_errors, system_processes, system_hardware, document_text,
  torch_status). Cumulative shrink: images 983→751, video 769→565,
  discovery 806→429, huggingface 703→525, system 559→252,
  documents 586→478.

**Frontend (`src/`)** — same pattern applied to the largest hooks +
components:

- `api.ts`: 1,430 → 6 domain modules (chat, image, video, models,
  setup, admin). Live-binding circular re-exports preserve every
  existing import path.
- `types.ts`: 1,378 → 230 LOC (-83%) via 11-file `types/` package.
- `useChat.ts`: 1,203 → 1,067 LOC. `optimisticTurns` (the
  push/replace/rollback state machine) + per-session localStorage
  helpers (temperature, reasoning effort) moved to `features/chat/`.
- `useImageState.ts`: 846 → 809 LOC via
  `features/image/{downloadActions,studioPresets,galleryActions}.ts`.
- `useVideoState.ts`: 1,126 → 899 LOC (-20%) via
  `features/video/{downloadActions,modelLifecycle,installActions}.ts`.
- `HtmlChallengeTab.tsx`: 1,677 → 1,103 LOC via the
  `features/chat/html_challenge/` package — 5 child components
  (ChallengeSetupPanel, ChallengeSlotPanel, ChallengeModelCard,
  ChallengePickerModal, ChallengeHistoryCombobox) + 2 helper modules
  (challengeApi.ts fetch wrappers, htmlChallengeTabHelpers.ts pure
  derived-value helpers + slot-state reducers).
- `VideoStudioTab.tsx`: 1,712 → 1,479 LOC via
  `VideoStudioRuntimeBanner.tsx` (~265 LOC of dense runtime status
  callout, chip row, and conditional install action panels for
  LongLive / mlx-video / mp4 encoder / missing tokenizer deps / GPU
  bundle).
- `ImageStudioTab.tsx`: 1,178 → 992 LOC via
  `ImageStudioRuntimeBanner.tsx` (~205 LOC of CUDA torch banner,
  chip row, model preload/unload control row, GPU runtime install
  action stack).
- App.tsx: 2,334 → 2,081 LOC via `features/app/` package
  (`modelActions.ts` for unload/delete handlers,
  `variantPayloads.ts` for pure variant → load/thread payload
  helpers, `conversionActions.ts` for the model conversion flow).
  CUDA torch install hook + capability strip shared component
  also pulled out.

**Rust shell (`src-tauri/src/`)** — full Phase 3 split:

- `binaries.rs` — bundled binary path resolvers (`resolve_llama_server` /
  `resolve_llama_server_turbo` / `resolve_llama_cli` / `resolve_sd_cpp` +
  `resolve_candidate` / `find_in_path` utilities).
- `env_setup.rs` — env-var + path-list helpers (`apply_library_path`,
  `join_paths`, `prepend_env_paths`).
- `runtime.rs` — `EmbeddedRuntimeManifest` + `EmbeddedRuntime` structs +
  20 helpers covering manifest fingerprint, tar extraction, extras-dir
  ABI namespacing, env application.
- `backend.rs` — `impl BackendManager` (~400 LOC) bootstrap → spawn →
  wait_for_port → probe sequence.

lib.rs: 1,335 → 302 LOC (-77%). Just public API surface (Tauri commands,
run() entry, struct decls) remains.

**Performance gate.** `scripts/perf-gate.py` compares a
`scripts/perf-baseline.py` JSON run against the captured floors in
`PERF_BASELINE.md`; default ±5% tolerance, configurable. Initial
floor: `text.tokens_per_second ≥ 297 tok/s` (Qwen2.5-0.5B 4-bit MLX,
Apple Silicon, 2026-05-09). New `.github/workflows/perf-gate.yml` runs
the comparator on `macos-latest` with HF cache restore — triggered via
the Actions "Run workflow" button or by labeling a PR with
`perf-gate`. We deliberately don't bolt this onto every push because
the cheapest gen needs ~700 MB of cached weights.

**Cross-OS parity.** PowerShell ports of the existing bash update
scripts; cross-platform `pre-build-check.mjs`; Windows promoted from
advisory to required in the CI test matrix.

**CLAUDE.md** extended with a Code Quality Guidelines section
(performance / security / modularisation) capturing the patterns this
refactor codified — file-size soft caps (backend 600 LOC, hooks 400,
components 500, Rust 800), unload-before-reload, list-form
subprocess-only, hostile-path validation, lazy imports, no premature
abstraction.

## v0.7.6 - 2026-05-08

### HTML Challenge — side-by-side HTML generation comparison

- New chat sub-tab dedicated to side-by-side comparison of LLMs producing HTML. Configure 2 → 4 model slots, each with its own model selection, full launch settings, thinking mode (`off` / `auto`), reasoning effort (`low` / `medium` / `high`), and seed. Issue a single prompt; every slot streams its HTML response in parallel with a sandboxed live preview underneath the raw text.
- **HTML validation** runs per slot — `valid` / `partial` / `script-error` / `blank-render` / `no-html`. Validation status is fed back from the iframe sandbox so script crashes and blank renders surface immediately.
- **Persistent history** — every challenge run is saved with title + prompt + per-slot manifests (model, settings, thinking mode, reasoning effort, seed, validation status, output path). Re-open from the history view, delete, or open / reveal individual slot HTML files on disk.
- **Retry + repair flows** — re-run a single failed slot, ask the model to `continue` from the partial response, or kick off a `repair` pass that asks the same model to fix its own broken HTML.
- Endpoints: `POST /api/chat/html-challenges` (kick off run, SSE-streamed), `GET /api/chat/html-challenges` (history list), `GET /api/chat/html-challenges/{id}`, `DELETE /api/chat/html-challenges/{id}`, `GET /api/chat/html-challenges/{id}/files/{slot}` (raw HTML), `POST /api/chat/html-challenges/{id}/slots/{slot}/retry`, `POST /api/chat/html-challenges/{id}/slots/{slot}/repair`, `PATCH /api/chat/html-challenges/{id}/slots/{slot}/validation`, `POST /api/chat/html-challenges/open-file`.

### UI scale

- Segmented UI scale control in Settings → Display: 75% / 100% / 125% / 150%. Applies app-wide via a `useUiScale` hook so the entire workspace re-flows around the chosen density. Persisted across launches.

### Chat polish

- **Retry failed generation** — failed assistant messages now expose a Retry button; one click re-issues the previous turn against the active runtime profile without re-typing.
- Chat draft state harden — drafts persist per thread across tab switches.
- Reasoning panel: smaller fixes around streaming preview height + collapse animation.
- Runtime controls panel: tighter spacing + clearer cache strategy / DFlash gating.

### Compare view refactor

- `CompareView` modularized so the two-up Compare mode and the new HTML Challenge multi-slot view share `buildComparePayload` / `cloneLaunchSettings` / `compareTargetLabels` / `useLaunchPreview` instead of duplicating the slot configuration logic.

### Gallery flows

- Image + Video Gallery: filter / sort / re-run flow tightened. Video gallery now mirrors image-gallery's metadata layout + reveal-on-disk + clone-settings actions.

### Packaging

- Bumped the application version to `0.7.6` across the npm, Python, and Tauri package metadata. v0.7.5 was used internally for the HTML Challenge feature branch; superseded directly by 0.7.6.

## v0.7.5 - 2026-05-07

- Internal version bump on the `feature/html-challenge` branch during development. No tagged GitHub Release; superseded by v0.7.6.

## v0.7.4 - 2026-05-06

### Chat experience (the headline)

**Phase 1 — UX foundations**
- Syntax highlighting in code blocks, in-thread search, conversation export, real cancel (mid-stream abort), reasoning-effort levels.
- Reasoning panel: collapsible streaming preview, fixed first-paragraph gap.

**Phase 2.0 — perf surface + watchdogs**
- Prompt-processing feedback + TTFT (time-to-first-token) live indicator.
- Prompt-eval timeout, memory gate, runaway guards (token rate floor, repetition guard), panic + thermal banners, image/video gates that block kicking off a generation when VRAM/RAM headroom is unsafe.

**Phase 2.1 — refactor**
- Decomposed monolithic `ChatTab.tsx` into `ChatSidebar` / `ChatHeader` / `ChatThread` / `ChatComposer`.

**Phase 2.2 — sampler control**
- Full sampler exposure: `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, `mirostat`, `reasoning_effort`.
- JSON-schema constrained-output opt-in (`json_schema` field).

**Phase 2.4 / 2.5 — message-tree workflows**
- Conversation branching: fork from any assistant message into a sibling thread.
- In-thread compare: render sibling variants side-by-side under the assistant bubble.

**Phase 2.6 / 2.7 — context & prompts**
- Cross-platform RAG: semantic embedding via `llama-embedding` + cosine retrieval over local docs.
- Prompt presets + variables: fill-form before "Use in Chat" so reusable prompts can take inputs.

**Phase 2.8 — structured tool output**
- Tool call results render as table / code / markdown / image based on returned shape, not raw JSON.

**Phase 2.10 — MCP client**
- Stdio JSON-RPC transport + tool adapter so any local MCP server is callable from chat. Provenance shown per tool result.

**Phase 2.11 / 2.12 — model-aware composer**
- Typed capability declarations (vision / tools / json_schema / reasoning) surface as badges in every model picker.
- Composer auto-gating (e.g. attach-image button hidden when active model has no vision).
- Mid-thread model swap with one-turn override (try a different model for a single response, then revert).

**Phase 2.13 — OpenAI-compatible server**
- Full sampler chain + embeddings parity. Apps that talk to `/v1/chat/completions` no longer lose advanced sampler params on the way through.

**Phase 2.14 — catalog browser**
- VRAM-fit hints on every Discover variant card so you see at a glance what'll actually run on your machine.

**Phase 3.x — substrate transparency**
- KV strategy chip in composer: per-turn cache override (native / turboquant / triattention) without touching launch settings.
- DDTree accepted-token overlay: substrate truth view of which speculative draft tokens were accepted.
- Logprobs viz (advanced-mode gated): per-message confidence summary, MLX logprobs streaming passthrough.
- Substrate routing inspector: per-turn badge above the metrics row showing which engine + binary served the response.
- Per-turn host strip: cross-platform perf telemetry (CPU / GPU / RAM / temp).
- Delve mode: critic-pass on assistant messages.
- Workspace knowledge stacks: shared RAG corpus across sessions.
- Chat-template inspection: detect Gemma + ChatML quirks, llama.cpp chat-template fix.

**Vision / multimodal**
- `--mmproj` wired for llama.cpp vision with sibling detection + `visionEnabled` flag flip.
- `visionEnabled` flag gates image attach across all runtimes.
- mlx-vlm torchvision dep added for Qwen2.5-VL processor build.

### Cache strategies & generation quality (FU-015 → FU-021, FU-026)
- **First Block Cache** (cross-platform diffusion cache hook, registry id `fbcache`) backed by `diffusers.hooks.apply_first_block_cache`. Applies to image + video DiTs (FLUX, SD3.5, Wan2.1/2.2, HunyuanVideo, LTX-Video, CogVideoX, Mochi). Default threshold 0.12 (≈1.8× speedup on FLUX.1-dev with imperceptible drift). Closes the FU-007 Wan TeaCache deferral by replacing per-model vendoring with a model-agnostic hook.
- **TaylorSeer / MagCache / PyramidAttentionBroadcast / FasterCache** strategies wired against the diffusers 0.38 native `enable_cache(<Config>)` API (registry ids `taylorseer`, `magcache`, `pab`, `fastercache`). MagCache is FLUX-only without calibration UX; other DiTs raise a "calibration required" message.
- **SDXL VAE fp16 fix on MPS / CUDA** (FU-017) — probes `madebyollin/sdxl-vae-fp16-fix` via `local_files_only=True` and swaps `pipeline.vae` so SDXL on Apple Silicon stays in fp16 instead of falling back to fp32.
- **Distill LoRA + transformer support** (FU-019) — Hyper-SD-8step + Turbo-Alpha for FLUX.1-dev, CausVid for Wan2.1 1.3B/14B, plus full distilled transformer swap (`distillTransformer*` fields) for Wan 2.2 A14B I2V × lightx2v 4-step distill (bf16 + fp8_e4m3 variants). Distill takes precedence over LoRA when both are pinned.
- **AYS (Align Your Steps) sampler** (FU-020) for SD/SDXL — new `ays_dpmpp_2m_sd15` / `ays_dpmpp_2m_sdxl` samplers using NVIDIA's hardcoded timestep arrays. Flow-match models continue to be gated out.
- **Image-runtime CFG decay parity** (FU-021) with the video runtime — opt-in `cfgDecay` field, linear ramp from initial guidance down to a 1.5 floor inside `callback_on_step_end`. Gated to flow-match repos.

### CUDA quantization foundations (FU-023, FU-024, FU-027)
Backend wiring landed for Windows / Linux CUDA validation; Apple Silicon dev box can't exercise these paths live.
- **Nunchaku / SVDQuant transformer load** (FU-023) — `_try_load_nunchaku_transformer` helper preferred over NF4 / int8wo on CUDA when `nunchakuRepo` pinned + `nunchaku>=1.2.1` importable. Catalog rows for FLUX.1-dev × svdq-int4 + FLUX.1-schnell × svdq-int4.
- **FP8 layerwise casting for non-FLUX DiTs** (FU-024) — `_maybe_enable_fp8_layerwise` helper on both image + video runtimes. Family-correct fp8 dtype (E5M2 for HunyuanVideo per upstream, E4M3 elsewhere). Compute capability gate refuses pre-Ada GPUs (SM <8.9). Studio toggle exposed in both Image + Video Studio.
- **NVIDIA/kvpress install action** (FU-027) — `kvpress>=0.5.3` registered in `_INSTALLABLE_PIP_PACKAGES` so the Setup tab can pre-stage the wheel ahead of integration code.

### MLX video runtime (FU-009 close-out, FU-025 Phases 7 → 9)
- **mlx-video Wan one-shot convert pipeline** under `~/.chaosengine/mlx-video-wan/<slug>/` (override via `CHAOSENGINE_MLX_VIDEO_WAN_DIR`). Helper `backend_service/mlx_video_wan_convert.py` wraps the upstream `python -m mlx_video.models.wan_2.convert` subprocess with `slug_for` / `output_dir_for` / `status_for` / `list_converted` / `run_convert`.
- **Runtime routing for `Wan-AI/Wan2.{1,2}-*`** through `mlx_video_runtime.py` — `_REPO_ENTRY_POINTS["Wan-AI/"] = "mlx_video.models.wan_2.generate"`, `_build_wan_cmd` produces the Wan-shaped CLI (`--model-dir`, `--guide-scale` string, `--scheduler`).
- **GUI install panel under Video Discover** — `WanInstallPanel.tsx` lists every supported Wan repo with raw-size hint + converted badge / install button + live `InstallLogPanel`. Setup endpoints `POST /api/setup/install-mlx-video-wan` + status + inventory mirror the longlive install pattern.
- **Live Wan2.1 MLX smoke validation** — 19.6s end-to-end at 480×272, 5 frames, 4 steps; surfaced + fixed a `status_for` filename gap (mlx-video upstream emits root-level `model.safetensors` + `t5_encoder.safetensors`, not the legacy `transformer*.safetensors` pattern).

### Preview & enhancement UX (FU-018 parts 1+2, FU-022)
- **TAESD / TAEHV preview VAE swap** (FU-018 part 1) — `maybe_apply_preview_vae(pipeline, repo, enabled)` maps repo → tiny VAE id (FLUX.1/2 → taef1/taef2, SD3 → taesd3, SDXL → taesdxl, Wan2.x → taew2_2, LTX-Video / LTX-2 → taeltx2_3_wide, HunyuanVideo → taehv1_5, CogVideoX → taecogvideox, Mochi → taemochi, Qwen-Image → taeqwenimage). Mirrors the stock VAE's dtype + device.
- **Per-step thumbnails via `callback_on_step_end`** (FU-018 part 2) — decodes `callback_kwargs["latents"]` through the swapped tiny VAE, scales to ≤192 px, base64-encodes a PNG, publishes to `IMAGE_PROGRESS.set_thumbnail` / `VIDEO_PROGRESS.set_thumbnail`. Stride caps emit count at ~8 (image) / ~6 (video) per gen. Frontend renders inside `LiveProgress`. Handles standard 4D `(B, C, H, W)` and FLUX's packed 3D `(B, seq_len, 64)` shapes.
- **MLX-native LLM prompt enhancer** (FU-022) — replaces the deterministic per-family template-suffix enhancer. Helper `backend_service/helpers/prompt_enhancer.py` wraps `mlx_lm.load` + `mlx_lm.generate` against `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~700 MB on disk, ~3s cold load + sub-second per call). Per-family system prompts (`wan` / `ltx` / `hunyuan` / `flux` / `sdxl` / `sd3` / `default`) anchor the rewrite to the DiT's training distribution. Endpoint `POST /api/prompt/enhance`. Apple Silicon only — CUDA / Linux fall back to the legacy template suffix.

### Speculative decoding
- **`dflash-mlx` pin bump** (FU-006) f825ffb → 8d8545d (v0.1.4.1 → v0.1.5.1). 0.1.5+ moved every primitive `backend_service/ddtree.py` consumed off the runtime top-level onto a per-family `target_ops` adapter. Adapter resolved once at the top of `generate_ddtree_mlx` via `resolve_target_ops(target_model)`. Gains: draft model quantization with Metal MMA kernels, branchless Metal kernels + fused draft KV projections, long-context runtime diagnostics. Live smoke validated against `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

### Windows / CUDA stability
- PowerShell ports of `build-llama-turbo` + `build-sdcpp` for Windows builds.
- MSVC + CUDA detection helpers, CMake generator handling — accept VS Build Tools installs that report `isComplete=0`, append `version=` to `CMAKE_GENERATOR_INSTANCE` for unregistered installs, fix CUDA-integration elevated copy + invalidate stale CMake cache.
- CUDA torch self-debugging install button with expandable per-attempt log + Restart prompt.
- Video Studio dropping GPU warning on CUDA hosts now surfaces inline Install button.
- T5 lazy-import diagnostic runs on generate paths (not just startup) to catch missing-dep failures before kicking off long generations.

### Studio polish & chat
- Restored pre-aec1975 card layout for Image / Video Discover + My Models, dropped the duplicate Wan panel that had been leaking through the catalog tabs.
- KV cache chip filter harmonized with the launch-settings modal so toggle states stay consistent across surfaces.
- Chat cache-fit warning is now VRAM-aware on CUDA hosts; raised chat default `maxTokens` to 4096; surfaced CPU torch on CUDA host with right-sized CogVideoX footprints.
- Fixed Studio cache preview returning 0 GB on chat model selection.

### Test infrastructure & runtime safety
- **`backend_service/runtime_paths.py` — append extras to `sys.path`** instead of `insert(1, ...)`. Prepending broke repo-local adapter shims (notably `turboquant_mlx`, which wraps the upstream `turboquant-mlx-full` install in extras): the raw upstream package shadowed the shim, hiding the shim's exported helpers (`_find_pip_turboquant_path`, `make_adaptive_cache`, `apply_patch`). Surfaced as a pytest collection failure on `tests/test_cache_strategies.py`; was also a latent runtime bug after a user clicked Setup → Install turboquant-mlx-full.

### Packaging
- Bumped the application version to `0.7.4` across the npm, Python, and Tauri package metadata.

## v0.7.3 - 2026-05-04

- Bumped the application version 0.6.0 → 0.7.3 across the npm, Python, and Tauri package metadata. No tagged GitHub Release; superseded by v0.7.4.

## v0.7.2 - 2026-05-02

- Wired the STG (Spatial Temporal Guidance) slider through to the mlx-video subprocess for LTX-2 generations.
- Added preset-row-pair styles for the Studio preset chooser.
- Harmonized the KV cache chip filter with the launch-settings modal so toggle states stay consistent across surfaces.

## v0.6.0 - 2026-04-19

- Renamed the local `compression/` package to `cache_compression/` so it no longer shadows Python 3.14's PEP 784 stdlib `compression` namespace package. Fixes a `ModuleNotFoundError: No module named 'compression._common'` surfacing on Windows with Python 3.14 when PyTorch's import chain reached into the shadowed package.
- Made the My Models library RAM estimate use the actual on-disk size + KV cache heuristic instead of the catalog flagship's `estimatedMemoryGb`, so differently-sized variants of the same family no longer all render as the same ~76 GB value. Added a parallel compressed-cache estimate for the Compressed column.
- Video diffusion models (HunyuanVideo, Mochi, Wan2.x, LTX-Video, CogVideo, etc.) are now tagged `modelType="video"` during discovery and kept out of the chat-oriented My Models list and chat picker. They continue to surface under the dedicated Video section.
- Video-gen memory safety now includes the model footprint (with device-class fragmentation factors) in the safety verdict, preventing the 40-frame Wan 2.1 T2V 1.3B MPS crash on 64 GB Macs.
- Hardened Windows staging: `scripts/stage-runtime.mjs` now clears read-only attributes and retries on transient EPERM/EBUSY during `.runtime-stage` cleanup, and skips the dev-mode tar archive that Tauri ignores anyway. `build.ps1` pre-clears stale staging and installs the project via `pip install -e ".[desktop,images]"` so strict validation has its required extras.
- Bumped the application version to `0.6.0` across the npm, Python, and Tauri package metadata.

## v0.5.3 - 2026-04-18

- Fixed the GitHub Actions release workflow to use the valid `includeUpdaterJson` input for `tauri-apps/tauri-action@v0.6.0`, removing the repeated `uploadUpdaterJson` warnings from release builds.
- Bumped the application version to `0.5.3` across the npm, Python, and Tauri package metadata in preparation for the next release.
