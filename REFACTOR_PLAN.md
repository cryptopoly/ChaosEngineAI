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

## Progress through 2026-05-09 (39 commits on `feature/refactor-n-audit`)

| File | Original | Now | Δ |
|---|---|---|---|
| `state/__init__.py` | 4,418 | 4,089 | -329 |
| `inference/__init__.py` | 3,574 | 1,521 | -2,053 |
| `image_runtime/__init__.py` | 2,097 | 1,366 | -731 |
| `video_runtime/__init__.py` | 2,378 | 1,669 | -709 |
| `routes/setup/__init__.py` | 1,932 | 353 | -1,579 |
| `routes/html_challenges/__init__.py` | 1,183 | 460 | -723 |
| `src/api/index.ts` | 1,430 | 559 | -871 |
| **Mega-file shrink total** | 17,012 | **10,017** | **-6,995 LOC** |

Tests posture across all 39 commits: **1,302 Python pass + 1 skip / 340 TS pass / tsc clean**. Zero regressions; coverage gate (60% Python) holds on every phase.

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

**PARTIAL** (Phase 1a-1, 1a-2, 1a-3; commits `8a26a48`, `879eede`, `2060142`):
- `state/logs.py` — LogManager (log + activity ring buffers + subscribers)
- `state/metrics.py` — cache labels + profile change reasons + metrics payloads (11 pure functions)
- `state/_helpers.py` — module-level helpers: `_compose_chat_system_prompt`, `_build_sampler_overrides`, `_build_history_with_reasoning`, title-generation utilities, `_read_text_tail`, `_spawn_snapshot_download`, `_normalize_remote_provider_api_base`, `_CATALOG_REF_ALIASES` (1a-3).

state/__init__.py: 4418 → 4089 (-329). Sessions, model_manager, benchmark, settings_state extractions deferred — biggest remaining is the 2k LOC of session/chat methods.

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

**MOSTLY DONE** (Phase 1b-1 through 1b-5; commits `cb1aed3` → `25ecbdf`):
- `inference/_constants.py` — 5 timeout / workspace constants
- `inference/_utils.py` — 9 shared helpers (_now_label, _normalize_message_content, _read_text_tail, _append_runtime_note, _http_json, _find_open_port, _resolve_gguf_path, _is_local_target, _looks_like_gguf)
- `inference/base.py` — 4 dataclasses + RepeatedLineGuard + BaseInferenceEngine
- `inference/jsonrpc.py` — JsonRpcProcess subprocess bridge
- `inference/simple_engines.py` — RemoteOpenAIEngine + MockInferenceEngine
- `inference/mlx_engine.py` — MLXWorkerEngine
- `inference/llama_cpp_engine.py` — LlamaCppEngine + 8 llama-specific helpers + 4 constants

inference/__init__.py: 3574 → 1521 (-2053). RuntimeController (~1050 LOC) is the only big class still inline; deferred — its helper graph is the most cross-cutting in the package.

**1c. `video_runtime.py` + `image_runtime.py` → runtimes/{image,video}/.**

**PARTIAL** (Phase 1c-1 through 1c-9, commits `b5ea526` → `1d16315`):
- `image_runtime/` package: types + repos + snapshot + device + placeholder_engine + mflux_engine extracted (image/__init__.py: 2097 → 1366).
- `video_runtime/` package: types + device + repos + defaults extracted (video/__init__.py: 2378 → 1669):
  - `video_runtime/device.py` — probe helpers (`_resolve_video_seed`, `_resolve_video_python`, `_detect_device_memory_gb`, `_guess_video_expected_device`, `_windows_cuda_unavailable_message`)
  - `video_runtime/repos.py` — `PIPELINE_REGISTRY`, GGUF/NF4 transformer class lookups, per-model defaults table, prompt-enhancement suffixes + `_enhance_prompt`
  - `video_runtime/defaults.py` — memory footprint estimator, slicing gate, scheduler classes, Wan frame alignment, `_resolve_video_defaults`, frame interpolation, dep tuples + `_find_missing`

**Remaining**: extract `DiffusersTextToImageEngine` (~1112 LOC inside image/__init__) + `DiffusersVideoEngine` (~1239 LOC inside video/__init__). Both classes use the same pipeline-loader pattern (LoRA fuse, distill swap, nunchaku, fp8, preview-VAE) — extract into `runtimes/common/` after both engines move out of their respective __init__.py files.

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

**1e. helpers/ regrouping into media/ models/ system/ ui/ storage/ inference/ finetune/ remote/ filter/ subpackages. Public re-exports preserve call sites.**

**Verify each step:** `pytest`, live smoke gens (text + image + video), `python -c "from backend_service.app import build_app; build_app()"` clean import.

### Phase 2 — Frontend split

**2a. `api.ts` 1,430 → src/api/{chat,image,video,models,setup,admin}.ts.** **DONE** (Phase 2-1 through 2-6, commits `dea6a54` → `68fed4f`). 6 commits, 4,453 LOC across 6 domain modules. Live-binding circular re-exports preserve call sites.

**2b. `types.ts` 1,378 → src/types/{chat,image,video,models,setup,shared}.ts.** Stub barrel + 3 sub-files exist already with per-domain UI types (ChatModelOption, ImageGalleryRuntimeFilter, VideoDiscoverTaskFilter); main types.ts content needs careful migration since most types reference each other. Defer to dedicated session.

**2c. Mega-hooks → 3-way splits each.**
- `useChat` → `useChatStreaming` + `useChatHistory` + `useChatInput`
- `useVideoState` → `useVideoConfig` + `useVideoGeneration` + `useVideoLibrary`
- `useImageState` → analogous

**2d. God components.**
- `App.tsx` 2,334 → composition root only; route tree + global keybinds + theme provider extracted
- `HtmlChallengeTab.tsx` 2,535 → `ChallengeRunner` + `ChallengeEditor` + `ChallengePreview` + `ChallengeLibrary`
- `VideoStudioTab.tsx` 1,796, `ImageStudioTab.tsx` 1,178 → shared `<StudioControls>` + `<StudioPreview>` + `<StudioLibrary>` shell

**2e. Inline single-use hooks** — `useGpuStatus`, `useSidebarPrefs`, `useUiScale` collapse into `App.tsx`.

**Verify:** `npm test`, `npx tsc --noEmit`, dev server boots, click-through 5 main tabs.

### Phase 3 — Rust shell split

```
src-tauri/src/
  lib.rs                # public API + state init only
  runtime/{extraction,manifest}.rs
  binaries/{resolution,path_search}.rs
  env/{apply,python}.rs
  backend/{manager,lifecycle,signals}.rs
  ipc.rs
  settings.rs
```

Add explicit `#[cfg(target_os = "linux")]` where Linux currently rides on `#[cfg(unix)]` but should diverge from macOS.

**Verify:** `cargo check --all-targets`, `cargo clippy -- -D warnings`, `cargo test`, `npm run tauri dev` boots.

### Phase 4 — Cross-OS parity

1. PowerShell ports: `update-llama-turbo.ps1`, `update-sdcpp.ps1`. **DONE** (commit `861de0a`). Both delegate to their `build-*.ps1` siblings after a version-file fast-exit so MSVC/CUDA toolchain plumbing stays in one place.
2. `pre-build-check.sh` → port to Node (`pre-build-check.mjs`) — single script across all 3 OSes. **DONE** (Phase 4-2). 7 checks ported (pytest, vitest, tsc, NOTICES grep, Python cache-strategy probe, upstream git ls-remote, binary file existence). Wired as `npm run pre-build-check`. Live smoke against the dev machine: 8 PASS / 0 FAIL / 1 WARN (turbo update available, expected).
3. ~~De-dupe `build-X.sh` + `update-X.sh` overlap → unified `manage-X.sh build|update|status`.~~ **DROPPED** — build scripts handle clone-or-fetch; update scripts add the version-file fast-exit. Two narrow scripts read clearer than one with a subcommand router.
4. ~~Rename `update-llama-cpp.sh` → `check-llama-cpp.sh` (info-only, name lied).~~ **DROPPED** — original audit was wrong: the script does rebuild llama-server (cmake configure + build).
5. CI matrix flips Windows/Linux from advisory to required. **DONE** (Phase 4-5, paired with 4-2). `windows-latest` job in `.github/workflows/build.yml` now has `advisory: false` so a Windows-specific regression blocks the PR the same way a macOS / Ubuntu failure does. Linux was already required.

### Phase 5 — Performance pass

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
