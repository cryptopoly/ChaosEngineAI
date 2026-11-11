# ChaosEngineAI — Project Guide

**Project hub:** read `docs/hub/status.md` first (where we are, what is open for Claude and for the owner), then `docs/hub/decisions.md`. Build it with `python3 tools/hub_tracker.py && python3 tools/build_hub.py` and open `docs/site/index.html`. The FU rows live in `FOLLOW_UPS.md` (grep it, don't read it whole); the hub reads it (FU-131).

## Editorial Rules

**Do not reference external desktop AI apps in code, comments, UI strings,
docs, or commit messages.** This includes — but is not limited to —
ComfyUI, LM Studio, Ollama, AUTOMATIC1111, Forge, InvokeAI, Diffusion Bee,
Draw Things, Mochi Diffusion, Pinokio. ChaosEngineAI is a standalone
product; comments and copy must not name or compare against competing
apps even when they share underlying weights or workflows.

Allowed exceptions:
- **Model names from upstream providers** (e.g. *"Stable Diffusion 3.5
  Medium"*, *"FLUX.1-schnell"*, *"Wan 2.1"*) — these are model identifiers
  shipped by Stability AI / Black Forest Labs / Alibaba, not apps.
- **Hugging Face organisation namespaces** (e.g. ``lmstudio-community/...``,
  ``mlx-community/...``) — these are repo namespaces on HF, not promotion
  of any app.
- **Open-source dependencies we vendor or shell out to** (e.g.
  ``stable-diffusion.cpp``, ``llama.cpp``, ``mlx-video``) — these are
  named libraries we ship as runtime components.

When describing reference defaults or upstream behaviour, name the
**model author** (e.g. *"Lightricks reference defaults"*, *"Wan-AI model
card"*) rather than the third-party tool that exposes them.

## Architecture Overview

ChaosEngineAI is a desktop AI inference app built with:
- **Frontend**: React + TypeScript + Vite
- **Desktop shell**: Tauri (Rust) — `src-tauri/`
- **Backend**: Python FastAPI sidecar — `backend_service/`
- **Inference engines**: MLX (Apple Silicon), llama.cpp (GGUF), vLLM (CUDA)
- **Cache strategies**: Pluggable compression via `cache_compression/` registry

### Key Directories

| Path | Purpose |
|------|---------|
| `src/` | React frontend (components, hooks, utils, types) |
| `src-tauri/src/lib.rs` | Tauri bridge — runtime extraction, binary resolution, sidecar bootstrap |
| `backend_service/` | Python FastAPI backend |
| `backend_service/inference/` | Core inference engine package — `controller.py` (RuntimeController), `engines/`, `binaries.py`, `capabilities.py`, `conversion.py`, `jsonrpc.py` |
| `backend_service/state/` | App state package — `__init__.py` (ChaosEngineState facade), `documents.py`, `benchmarks.py`, `openai_compat.py`, `payloads.py`, `settings_state.py`, `sessions.py`, `downloads.py`, `metrics.py`, `logs.py` |
| `backend_service/mlx_worker*.py` | MLX subprocess worker — `mlx_worker.py` orchestrator + `mlx_worker_{request,prompt,io,diagnostics,multimodal,cache,eval,loader}.py` siblings |
| `backend_service/routes/` | API endpoints (14 route modules) |
| `backend_service/helpers/` | System stats, settings, persistence, cache estimation |
| `cache_compression/` | Cache strategy registry + adapters (native, turboquant, triattention, plus diffusion-only fbcache/teacache/taylorseer/magcache/pab/fastercache). Renamed from `compression/` so it doesn't shadow Python 3.14's stdlib `compression` namespace package. Legacy ids `chaosengine` and `rotorquant` were dropped in FU-030 and now coerce to `turboquant` via `registry.resolve_legacy_id`. |
| `dflash/` | DFlash speculative decoding — draft model registry + availability detection |
| `scripts/` | Build, install, and update scripts |
| `tests/` | Python tests (pytest) |
| `src/**/*.test.ts` | TypeScript tests (vitest) |

### Binary Routing

The app supports two llama-server binaries:
- **`llama-server`** (standard, Homebrew) — for the native cache strategy
- **`llama-server-turbo`** (TurboQuant fork) — for the TurboQuant strategy, installed to `~/.chaosengine/bin/`

Each `CacheStrategy` declares `required_llama_binary()` → `"standard"` or `"turbo"`. The `LlamaCppEngine._select_llama_binary()` method in `inference/llama_cpp_engine.py` routes to the correct binary. Cache types are pre-validated against the binary's `--help` output before startup.

---

## Build Checklist

Run before every release, PR, or significant change. Automated via `./scripts/pre-build-check.sh`.

### 1. Tests
- [ ] `cd /Users/dan/ChaosEngineAI && .venv/bin/python -m pytest tests/ -q` — all Python tests pass
- [ ] `npm test` — all TypeScript tests pass
- [ ] `npx tsc --noEmit` — no type errors
- [ ] `./scripts/e2e_test_suite.py --smoke` — fast smoke (≤60s) confirms backend + CLI surface intact
- [ ] `cargo test --manifest-path src-tauri/Cargo.toml` — Tauri shell tests (sidecar lifecycle, port wait)
- [ ] `.venv/bin/python scripts/audit-catalog-repos.py` — every Hub repo and pinned file in the catalogs exists (needs network; run when a catalog row changes and before a release)
- [ ] `./scripts/e2e_test_suite.py` (full) — every phase passes (or correctly skips). **Required for release builds and any PR touching inference routing, engine implementations, cache strategies, or setup install endpoints.** See [docs/E2E_TESTING.md](docs/E2E_TESTING.md).
- [ ] **New feature gate**: any user-visible feature, new engine wiring, new model family in catalog, new install endpoint, or new cache/spec-dec strategy MUST land alongside an E2E check in the relevant phase of `scripts/e2e_test_suite.py`. PRs that add a feature but no E2E check are incomplete. See *Adding new checks* in `docs/E2E_TESTING.md`.

### 2. Licences
- [ ] `THIRD_PARTY_NOTICES.md` is up to date — all bundled/vendored deps listed with correct licence types
- [ ] No new dependencies added without checking licence compatibility (must be MIT, Apache 2.0, BSD, or similar permissive)
- [ ] Shipped binaries (llama-server, llama-server-turbo) include MIT licence notice

### 3. Upstream Dependencies
Check for updates to external repos we build from or depend on:

| Dependency | Repo | Branch | Check Command |
|-----------|------|--------|---------------|
| llama.cpp (standard) | `ggml-org/llama.cpp` | tag `b11981` (adds `kolibri1`, `flint`, `glm5-next`, `muse-spark`, `lfm3`; `graniteswitch` and `gemma4` (Nemotron 3.5 Bolt) were already in b11938; pinned in [scripts/update-llama-cpp.sh](scripts/update-llama-cpp.sh) `LLAMA_CPP_REF` and [scripts/stage-runtime.mjs](scripts/stage-runtime.mjs) `LLAMA_CPP_TAG`; pin only a tag whose release has the prebuilt assets, b11982 had none when checked) | `git ls-remote --tags https://github.com/ggml-org/llama.cpp 'refs/tags/b1*' \| sort -t/ -k3 -V \| tail -1` |
| llama-server-turbo | `TheTom/llama-cpp-turboquant` | `feature/turboquant-kv-cache` | `git ls-remote https://github.com/TheTom/llama-cpp-turboquant.git refs/heads/feature/turboquant-kv-cache` |
| dflash-mlx | `bstnxbt/dflash-mlx` | `main` pinned to commit `fada1eb` (= v0.1.5.1; tags restored upstream, latest v0.1.10 — see FU-057) | `git ls-remote --tags https://github.com/bstnxbt/dflash-mlx.git` |
| turboquant-mlx-full | `manjunathshiva/turboquant-mlx` | — | `.venv/bin/pip index versions turboquant-mlx-full 2>/dev/null` |
| DDTree (ported algorithm) | `liranringel/ddtree` | `main` | `git ls-remote https://github.com/liranringel/ddtree.git HEAD` |
| TensorFold 1.x (native binary, CUDA text engine) | `ashhart/TensorFold` | release `v1.2.6` (Apache-2.0; Linux x86_64 / aarch64 and macOS arm64 archives, **no Windows archive**: Windows adopts a source build or a self-hosted archive; pinned in [_tensorfold_native.py](backend_service/inference/_tensorfold_native.py) `NATIVE_VERSION`, bump with the registry in the same file; the Mac engine below stays on the Python 0.6.6 pin) | `git ls-remote --tags https://github.com/ashhart/TensorFold.git` |
| TensorFold | `ashhart/TensorFold` | commit `cb2ebf0` (= v0.6.6, the last Python release; 1.x is a native binary, FU-111; pinned in [scripts/install-tensorfold.sh](scripts/install-tensorfold.sh) `TENSORFOLD_REF` — bump together with the facts in [_tensorfold.py](backend_service/inference/_tensorfold.py)) | `git ls-remote --tags https://github.com/ashhart/TensorFold.git` |
| mflux | `filipstrand/mflux` | release `0.24.0` (adds Iris-3B, MOSS-Image, Anima and FIBO 2, of which only Iris-3B is wired; requires Pillow 13; pinned in [scripts/install-mflux.sh](scripts/install-mflux.sh) `MFLUX_VERSION` — bump together with the family table in [mflux_engine.py](backend_service/image_runtime/mflux_engine.py)) | `curl -s https://pypi.org/pypi/mflux/json \| python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"` |
| ltx-2-mlx | `dgrauet/ltx-2-mlx` | commit `895201d` (= v0.18.0; not on PyPI, installed from the pinned commit into its own venv by [scripts/install-ltx2-mlx.sh](scripts/install-ltx2-mlx.sh) `LTX2_MLX_REF`; bump together with the command line and file names in [ltx2_mlx_runtime.py](backend_service/ltx2_mlx_runtime.py); weights pack `dgrauet/ltx-2.3-mlx-q4`, LTX-2.5 packs are gated) | `git ls-remote https://github.com/dgrauet/ltx-2-mlx.git HEAD` |
| ACE-Step 1.5 (music) | `ace-step/ACE-Step-1.5` | commit `ca1e85f` (2026-08-29, 21 commits after `v0.1.8`; installed on demand into `~/.chaosengine/music-venv` by [music_env.py](backend_service/music_env.py) `ACE_REF` from [music_requirements.txt](backend_service/music_requirements.txt), exported from that commit's `uv.lock`; bump both together and re-run the step-hook test, which guards the private `dit_generate.tqdm` name; weight revisions are pinned in [music_models.py](backend_service/catalog/music_models.py)) | `git ls-remote https://github.com/ace-step/ACE-Step-1.5.git HEAD` |
| TriAttention | `WeianMao/triattention` | commit `a4bc3c8` (pinned in [pyproject.toml](pyproject.toml) `[triattention]` + `[triattention-mlx]` and `_INSTALLABLE_PIP_PACKAGES`; pre-build-check asserts all three match) | `git ls-remote https://github.com/WeianMao/triattention.git HEAD` |
| Chatterbox (CUDA speech) | `resemble-ai/chatterbox` | commit `ab55923` (installed with `--no-deps --force-reinstall` into the voice venv by `voice_env.CHATTERBOX_PACKAGE`; PyPI's 0.1.7 has no Multilingual V3, so move back to a PyPI pin once a release carries `t3_model`; Japanese readings come from fugashi + unidic-lite in `voice_ja.py`, never GPL pykakasi) | `git ls-remote https://github.com/resemble-ai/chatterbox.git HEAD` |
| Playwright (E2E, dev only) | `microsoft/playwright` | exact `1.65.0` in [e2e/package.json](e2e/package.json) (own lockfile; the app's `package.json` has no Playwright; bump `@playwright/test`, then `npx playwright install chromium --only-shell`; never `@playwright/mcp`; conditions in [docs/research/playwright-security.md](docs/research/playwright-security.md)) | `npm view @playwright/test version` |

**Security and freshness audit.** `.venv/bin/python scripts/audit-dependencies.py` checks every pinned Python package (`release-constraints.txt`), Rust crate (`src-tauri/Cargo.lock`) and npm package (`package-lock.json`) against osv.dev, and lists what is behind the newest release. Run it before every release; `pre-build-check` runs it and the release workflow runs it before building anything (needs network).
- [ ] No high or critical advisory in a shipped package. Fix it (re-freeze the constraints with `scripts/freeze-release-constraints.py` after upgrading in the test environment), or accept it on purpose in `scripts/dependency-audit-ignore.json` with a reason and an expiry date (an expired entry blocks again).
- [ ] Read the "Out of date" list. It never blocks, but upgrade what you can test: major versions behind are listed first.

### 4. Cache Strategy Health
- [ ] Native strategy `llama_cpp_cache_flags()` only emits standard types: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
- [ ] TurboQuant strategy returns `required_llama_binary() == "turbo"`
- [ ] Legacy `chaosengine` + `rotorquant` ids coerce to `turboquant` via `registry.resolve_legacy_id`
- [ ] DFlash `_COMMUNITY_PREFIXES` includes all common model repo prefixes
- [ ] New model families added to `DRAFT_MODEL_MAP` if draft checkpoints exist

### 5. Desktop Packaging
- [ ] `scripts/stage-runtime.mjs` stages both `llama-server` and `llama-server-turbo` if available
- [ ] Manifest includes `llamaServerTurbo` field
- [ ] The release bundle's `runtime-*.manifest.json` has `llamaServer` set and an empty `warnings` list (a null `llamaServer` means the prebuilt download failed; FU-107)
- [ ] The runtime archive stages the Setup installers: `tar -tzf src-tauri/resources/embedded/runtime-*.tar.gz | grep backend/scripts` lists `install-mflux.sh`, `install-ltx2-mlx.sh`, `install-mtplx.sh` and `install-tensorfold.sh` (FU-104)
- [ ] `src-tauri/src/lib.rs` sets `CHAOSENGINE_LLAMA_SERVER_TURBO` env var
- [ ] `release-constraints.txt` matches the tested environment: after any dependency upgrade, rerun `.venv/bin/python scripts/freeze-release-constraints.py` and commit the file (release builds install with `-c release-constraints.txt`; without it v0.9.8 shipped huggingface_hub 2.2 / mlx 0.32 / transformers 5.20 against a 1.33 / 0.31 / 5.5 test environment)

### 6. Localization (i18n)
- [ ] `npm run i18n:validate` passes — every locale parity check + ICU syntax compile + orphan key scan
- [ ] No hardcoded user-facing strings introduced in this release (extraction script `npm run i18n:extract` reports zero new keys with empty `en` values)
- [ ] All 10 shipping locales (`en`, `zh-CN`, `zh-TW`, `ja`, `de`, `ru`, `ko`, `fr`, `es`, `pt-BR`) reach ≥95 % coverage; missing keys auto-fall back to `en` at runtime
- [ ] `backend_service/locales/*/LC_MESSAGES/messages.mo` compiled via `pybabel compile`
- [ ] `src-tauri/locales/*.ftl` regenerated for any new menu / tray strings
- [ ] Pseudo-locale `en-XA` smoke-tested for layout clipping on Setup + Diagnostics tabs (widest text surface)
- [ ] CJK font fallback stack present in `styles.css` (Apple Silicon + Windows + Linux paths)
- [ ] Per-feature translation workflow followed: new en-only strings ship with `null` placeholders in other locales (runtime falls back to `en`); follow-up PR fills translations (see FU-042+ tracker rows for cadence)

---

## Follow-Ups Tracker

The FU-### rows live in [FOLLOW_UPS.md](FOLLOW_UPS.md) (moved out of this file on 2026-11-06, D-22).
Do not read it whole; grep for the row you need (`grep -n "FU-123" FOLLOW_UPS.md`).
Add a row there when you defer work, edit it when you ship, then run
`python3 tools/hub_tracker.py && python3 tools/build_hub.py`.

---

## Testing Requirements

### When Modifying These Areas, Run These Tests:

| Area | Test File(s) | Command |
|------|-------------|---------|
| Cache strategies (`cache_compression/`) | `test_cache_strategies.py` | `pytest tests/test_cache_strategies.py -v` |
| DFlash / speculative decoding | `test_dflash.py` | `pytest tests/test_dflash.py -v` |
| Inference / llama.cpp / binary routing | `test_inference.py` | `pytest tests/test_inference.py -v` |
| Setup routes / install endpoints | `test_setup_routes.py` | `pytest tests/test_setup_routes.py -v` |
| Backend services | `test_services.py` | `pytest tests/test_services.py -v` |
| Backend API routes | `test_backend_service.py` | `pytest tests/test_backend_service.py -v` |
| TensorFold engine (registry / routing / installer / engine vs stub server) | `test_tensorfold_*.py` | `pytest tests/test_tensorfold_registry.py tests/test_tensorfold_routing.py tests/test_tensorfold_setup.py tests/test_tensorfold_engine_integration.py -v` |
| TensorFold 1.x native engine (platform keys, release assets, GPU gate, registry, serve command, safe unpack, installer, routing) | `test_tensorfold_native.py` | `pytest tests/test_tensorfold_native.py -v` |
| Voice I/O (lanes, voice venv, worker protocol, routes, text clean-up) | `test_voice_io.py`, `test_voice.py` | `pytest tests/test_voice_io.py tests/test_voice.py -v` |
| Vendored MLX architectures (Kolibri 1) + reasoning-effort template switch | `test_mlx_models.py` | `pytest tests/test_mlx_models.py -v` |
| Release bundle staging (Setup installer scripts) | `test_stage_runtime_scripts.py` | `pytest tests/test_stage_runtime_scripts.py -v` |
| Project hub (tracker pages current with CLAUDE.md, hub builds without link errors) | `test_hub_tracker.py` | `pytest tests/test_hub_tracker.py -v` |
| Agent bridge (`/mcp` tools, access switch, shared job queue, stdio relay, output metadata) and Storyboard (project store, timeline, routes, storyboard tools) | `test_mcp_bridge.py`, `test_mcp_relay.py`, `test_job_queue.py`, `test_output_metadata.py`, `test_agent_files.py`, `test_media_input.py`, `test_storyboard_store.py`, `test_storyboard_timeline.py`, `test_storyboard_routes.py`, `test_storyboard_tools.py` | `pytest tests/test_mcp_bridge.py tests/test_mcp_relay.py tests/test_job_queue.py tests/test_output_metadata.py tests/test_agent_files.py tests/test_media_input.py tests/test_storyboard_store.py tests/test_storyboard_timeline.py tests/test_storyboard_routes.py tests/test_storyboard_tools.py -v` |
| Request guards (chat settings, pipeline kwargs, sizes and frames, STT language fallback) | `test_request_guards.py`, `test_voice_library.py` | `pytest tests/test_request_guards.py tests/test_voice_library.py -v` |
| LTX-2.3 MLX engine (ltx-2-mlx) routing, catalog, command line | `test_ltx2_mlx_runtime.py` | `pytest tests/test_ltx2_mlx_runtime.py -v` |
| Music (ACE-Step catalog and fit, music venv, worker protocol, client cancel, library and FLAC tags, routes, `make_music`) | `test_music_*.py` | `pytest tests/test_music_catalog.py tests/test_music_env.py tests/test_music_worker.py tests/test_music_client.py tests/test_music_library.py tests/test_music_routes.py -v` |
| Cross-strategy E2E matrix runner | `test_cache_strategy_matrix_runner.py` | `pytest tests/test_cache_strategy_matrix_runner.py -v` |
| Cache speed calibration script | `test_calibrate_cache_speed.py` | `pytest tests/test_calibrate_cache_speed.py -v` |
| Dependency audit script (CVSS scoring, accepted risks, exit codes) | `test_audit_dependencies.py` | `pytest tests/test_audit_dependencies.py -v` |
| Frontend API client | `src/api.test.ts` | `npm test` |
| Frontend utilities | `src/utils/__tests__/*.test.ts` | `npm test` |

### Minimum Test Expectations
- All existing tests must pass — zero regressions
- New backend features should include at least basic happy-path tests
- Cache strategy changes must test `llama_cpp_cache_flags()` returns valid types
- New API endpoints need at least a shape/contract test

### Cross-strategy E2E matrix runner

`scripts/cache-strategy-matrix.py` sweeps every supported (cache strategy
× spec-dec method × representative model) combination through a running
backend on port 8876 and writes a CSV + Markdown report to
`~/.chaosengine/test-results/`. It also asserts the **FU-030 legacy
alias coercion** — requests with `cacheStrategy=chaosengine` /
`cacheStrategy=rotorquant` must come back loaded as `turboquant`, and
the runner exits with code 2 if either regresses.

```
# Quick smoke (~5 min on M-series; CI-friendly)
.venv/bin/python scripts/cache-strategy-matrix.py --quick

# Full sweep (~20 min; gates a release)
.venv/bin/python scripts/cache-strategy-matrix.py
```

The runner skips cells where the strategy isn't installed, the
turbo binary is missing, the model isn't in the local library, or
the spec-dec method isn't supported on the chosen backend — so a
fresh CI box reports honest skip reasons rather than failing.

---

## Code Quality Guidelines

These rules came out of the v0.7.6 → v0.8.0 refactor + audit. Apply them
to every PR that touches a backend module > 500 LOC, a hook > 400 LOC,
or any file that mutates worker subprocess / file-system / network
state. Skip on trivial typo fixes, doc-only edits, and one-line bug
patches.

### Performance

- **Lazy-import heavy deps.** `torch`, `diffusers`, `mlx`, `mlx_lm`,
  `mlx_vlm`, `transformers`, `nunchaku`, `bitsandbytes`, `huggingface_hub`,
  `gguf` are all multi-second imports. Put them inside the function that
  needs them, not at module top, unless the file is *only* loaded when
  inference is about to run. Backend startup target: `python -X importtime
  backend_service.app` < 2 s.
- **Process isolation for memory hogs.** Models > 1 GB stay in subprocess
  workers (MLX worker, sd-cli, longlive engine). Never load them in the
  FastAPI parent — a stuck pipeline takes the whole backend down with it.
- **Always release before reload.** Before swapping models, call the
  engine's `unload_model()` (or equivalent) so the OS reclaims the RAM
  before the next snapshot is mapped in. Two 47 GB workers = a 96 GB RAM
  exhaustion bug, not a feature.
- **No re-render thunder in React.** New object literals in `useMemo` /
  `useState` initialisers without a stable dep array are silent
  performance killers. Run the React Profiler on any tab > 200 LOC of
  state before shipping.
- **Profile before optimizing.** Don't rewrite a hot path on intuition —
  capture a number first (`scripts/perf-baseline.py`, the React Profiler,
  `python -X importtime`), then validate the win against
  `PERF_BASELINE.md`'s ±5 % gate.

### Security

- **Treat user-controlled paths as hostile.** Anything that comes from a
  request body, a settings file, an env var, or a Hub catalog entry
  must go through `pathlib.Path` + `.resolve()` + a parent-prefix check
  before being passed to `open()` / `subprocess.run` / `shutil.copy`.
  Never `os.path.join` a user string into a system path.
- **List-form subprocess only.** `subprocess.run([bin, *args])` — never
  the shell-string form. No `shell=True`. Quote nothing — let `subprocess`
  do the escaping.
- **No secrets in source.** No HF tokens, no API keys, no bearer tokens,
  no signed URLs in `*.py`, `*.ts`, `*.rs`, `*.toml`, or `*.md`. Use the
  Settings store + `keyring` / Tauri secure storage for runtime secrets.
  CI builds get keys from GitHub secrets, not commits.
- **Validate at the boundary, trust internally.** Pydantic models on the
  FastAPI request edge + `serde` on the Tauri IPC edge + Zod on the
  frontend fetch wrapper. Once the value is past the boundary, internal
  helpers don't need defensive `if not isinstance(...)` re-checks — the
  type system carries the guarantee.
- **GGUF / safetensors are user data.** They can be malicious archives
  on a snapshot a user pasted in. Always load with `local_files_only=True`
  when probing, and surface gated/404 errors as user-readable messages,
  not raw `HfHubHTTPError` traces.

### Modularisation

- **File-size soft caps.** Backend modules > 600 LOC, hooks > 400 LOC,
  components > 500 LOC, Rust modules > 800 LOC are a refactor signal —
  not an automatic block, but a prompt for the next change to extract
  before adding. The v0.8.0 pattern is: pull a coherent subset into a
  sibling module taking dependencies as kwargs, leave thin wrappers in
  the original site, re-export so test mock paths and existing imports
  don't break.
- **Single-purpose modules.** A file's docstring should fit in one
  paragraph. If you can't summarise what it does without "and also …",
  split it. Bundle by *responsibility*, not by *type* (don't dump every
  helper into `helpers.py`).
- **Re-exports preserve call sites.** When extracting from a module that
  has external callers, re-export the moved symbol from the original
  module path. Tests that patch `module._private` keep patching, imports
  in other packages keep working, and the diff stays surgical.
- **No premature abstraction.** Three similar lines is fine. Don't create
  a `BaseEngine` / `Strategy` / `Plugin` interface for two callers — wait
  until there are five. Half-finished abstractions cost more than copies.
- **Cross-platform from the first line.** `pathlib.Path` (Python),
  `PathBuf` (Rust), `path.posix` vs `path.win32` (Node). Never hardcode
  `/tmp`, `~/.cache`, or `\\` — use the platform-aware primitive.

### When to refactor vs ship

- Bug fix → ship the surgical patch, leave the surrounding module alone.
- Feature add → if the target file is already over the soft cap, do an
  extract pass before adding. Otherwise add inline.
- Refactor pass → bundle multiple extracts in a single PR with a clear
  phase number (see `REFACTOR_PLAN.md` for the v0.8.0 template).

## Development Patterns

### Python Backend
- Routes use `FastAPI APIRouter` with type hints
- State accessed via `request.app.state.chaosengine`
- Tests use `unittest.TestCase` + `fastapi.testclient.TestClient`
- Mock runtime with `FakeRuntime` pattern from `test_backend_service.py`

### TypeScript Frontend
- Tests use `vitest` with `vi.mock()` / `vi.stubGlobal()`
- Factory helpers (`makeVariant()`, `makeSession()`) for test data
- API mocking via `vi.stubGlobal("fetch", mockFn)`

### Adding New Dependencies
1. Check licence (MIT/Apache 2.0/BSD only)
2. Add to `THIRD_PARTY_NOTICES.md`
3. If pip package: add to `_INSTALLABLE_PIP_PACKAGES` in `backend_service/routes/setup.py`
4. If system binary: add to `_installable_system_packages()` in `backend_service/routes/setup.py`
5. Add update-check entry to the upstream dependency table above

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CHAOSENGINE_LLAMA_SERVER` | Override standard llama-server path | Auto-detected |
| `CHAOSENGINE_LLAMA_SERVER_TURBO` | Override turbo llama-server path | `~/.chaosengine/bin/llama-server-turbo` |
| `CHAOSENGINE_MLX_PYTHON` | Override Python for MLX | `.venv/bin/python` |
| `CHAOSENGINE_LLAMA_BIN_DIR` | Override llama.cpp build dir for staging | `../llama.cpp/build/bin/` |
