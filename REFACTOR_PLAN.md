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
```
backend_service/state/
  __init__.py          # ChaosEngineState facade — public API unchanged
  session_manager.py   # chat sessions, history
  model_manager.py     # model load/unload/discovery state
  inference_orchestrator.py
  benchmark_state.py
  settings_state.py
```

**1b. `inference.py` 3,574 → engines/ subpackage.**
```
backend_service/inference/
  __init__.py
  controller.py        # RuntimeController
  engines/
    base.py
    llama_cpp.py
    mlx_worker.py
    vllm.py
  jsonrpc.py
```

**1c. `video_runtime.py` + `image_runtime.py` → runtimes/{image,video}/.**

Extract shared pipeline-loader logic (LoRA fuse, distill swap, nunchaku, fp8, preview-VAE) into `runtimes/common/` — currently duplicated.

**1d. `routes/setup.py` 1,932 → setup/{detect,install_pip,install_brew,install_runtimes,status}.py.**

**1e. helpers/ regrouping into media/ models/ system/ ui/ storage/ inference/ finetune/ remote/ filter/ subpackages. Public re-exports preserve call sites.**

**Verify each step:** `pytest`, live smoke gens (text + image + video), `python -c "from backend_service.app import build_app; build_app()"` clean import.

### Phase 2 — Frontend split

**2a. `api.ts` 1,430 → src/api/{chat,image,video,models,setup,server,shared}.ts.**

**2b. `types.ts` 1,378 → src/types/{chat,image,video,models,setup,shared}.ts.**

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

1. PowerShell ports: `update-llama-turbo.ps1`, `update-sdcpp.ps1`.
2. `pre-build-check.sh` → port to Node (`pre-build-check.mjs`) — single script across all 3 OSes.
3. De-dupe `build-X.sh` + `update-X.sh` overlap → unified `manage-X.sh build|update|status`.
4. Rename `update-llama-cpp.sh` → `check-llama-cpp.sh` (info-only, name lied).
5. CI matrix flips Windows/Linux from advisory to required.

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
