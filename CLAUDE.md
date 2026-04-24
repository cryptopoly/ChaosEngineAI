# ChaosEngineAI — Project Guide

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
| `backend_service/inference.py` | Core inference engine — model loading, binary routing, generation |
| `backend_service/routes/` | API endpoints (14 route modules) |
| `backend_service/helpers/` | System stats, settings, persistence, cache estimation |
| `cache_compression/` | Cache strategy registry + adapters (native, rotorquant, turboquant, chaosengine, triattention). Renamed from `compression/` so it doesn't shadow Python 3.14's stdlib `compression` namespace package. |
| `dflash/` | DFlash speculative decoding — draft model registry + availability detection |
| `scripts/` | Build, install, and update scripts |
| `tests/` | Python tests (pytest) |
| `src/**/*.test.ts` | TypeScript tests (vitest) |

### Binary Routing

The app supports two llama-server binaries:
- **`llama-server`** (standard, Homebrew) — for native and ChaosEngine cache strategies
- **`llama-server-turbo`** (TurboQuant fork) — for RotorQuant and TurboQuant strategies, installed to `~/.chaosengine/bin/`

Each `CacheStrategy` declares `required_llama_binary()` → `"standard"` or `"turbo"`. The `LlamaCppEngine._select_llama_binary()` method in `inference.py` routes to the correct binary. Cache types are pre-validated against the binary's `--help` output before startup.

---

## Build Checklist

Run before every release, PR, or significant change. Automated via `./scripts/pre-build-check.sh`.

### 1. Tests
- [ ] `cd /Users/dan/ChaosEngineAI && .venv/bin/python -m pytest tests/ -q` — all Python tests pass
- [ ] `npm test` — all TypeScript tests pass
- [ ] `npx tsc --noEmit` — no type errors

### 2. Licences
- [ ] `THIRD_PARTY_NOTICES.md` is up to date — all bundled/vendored deps listed with correct licence types
- [ ] No new dependencies added without checking licence compatibility (must be MIT, Apache 2.0, BSD, or similar permissive)
- [ ] Shipped binaries (llama-server, llama-server-turbo) include MIT licence notice

### 3. Upstream Dependencies
Check for updates to external repos we build from or depend on:

| Dependency | Repo | Branch | Check Command |
|-----------|------|--------|---------------|
| llama.cpp (standard) | `ggml-org/llama.cpp` | `master` | `git -C ../llama.cpp fetch && git -C ../llama.cpp log HEAD..origin/master --oneline` |
| llama-server-turbo | `TheTom/llama-cpp-turboquant` | `feature/turboquant-kv-cache` | `git ls-remote https://github.com/TheTom/llama-cpp-turboquant.git refs/heads/feature/turboquant-kv-cache` |
| ChaosEngine | `cryptopoly/ChaosEngine` | `main` | `git -C vendor/ChaosEngine fetch && git -C vendor/ChaosEngine log HEAD..origin/main --oneline` |
| dflash-mlx | `bstnxbt/dflash-mlx` | `main` pinned to commit `f825ffb2` (upstream deleted all tags April 2026) | `git ls-remote https://github.com/bstnxbt/dflash-mlx.git refs/heads/main` |
| turboquant | `back2matching/turboquant` | — | `.venv/bin/pip index versions turboquant 2>/dev/null` |
| turboquant-mlx | `arozanov/turboquant-mlx` | — | `.venv/bin/pip index versions turboquant-mlx 2>/dev/null` |
| turboquant-mlx-full | `helgklaizar/turboquant_mlx` | — | `.venv/bin/pip index versions turboquant-mlx-full 2>/dev/null` |
| DDTree (ported algorithm) | `liranringel/ddtree` | `main` | `git ls-remote https://github.com/liranringel/ddtree.git HEAD` |

### 4. Cache Strategy Health
- [ ] ChaosEngine `llama_cpp_cache_flags()` only emits standard types: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
- [ ] RotorQuant/TurboQuant strategies return `required_llama_binary() == "turbo"`
- [ ] DFlash `_COMMUNITY_PREFIXES` includes all common model repo prefixes
- [ ] New model families added to `DRAFT_MODEL_MAP` if draft checkpoints exist

### 5. Desktop Packaging
- [ ] `scripts/stage-runtime.mjs` stages both `llama-server` and `llama-server-turbo` if available
- [ ] Manifest includes `llamaServerTurbo` field
- [ ] `src-tauri/src/lib.rs` sets `CHAOSENGINE_LLAMA_SERVER_TURBO` env var

---

## Follow-Ups Tracker

Deferred work and upstream conditions to re-check periodically. Revisit at each
release or when touching the affected subsystem. Delete entries once shipped or
no longer relevant.

| ID | Item | Trigger / Condition | Notes |
|----|------|---------------------|-------|
| FU-001 | Bump `turboquant` to 0.3.x | PyPI publishes `>=0.3.0` (source at 0.3.1 since 2026-04-16) | Adds asymmetric K/V bits, layer-adaptive precision, `--no-quant` eval flag, NumPy 2.0 + transformers 5.x compat. Backward compatible per upstream README. Bump extra in [pyproject.toml](pyproject.toml) once available. |
| FU-002 | Wire TriAttention MLX compressor into mlx_worker | When adding experimental KV compression path for mlx-lm generation | **Blocked on upstream API gap.** `TriAttentionStrategy.apply_mlx_compressor()` exists ([cache_compression/triattention.py](cache_compression/triattention.py)) and triattention 0.2.0 is installable via `pip install --no-deps` (skips triton which is CUDA-only). BUT: (1) `mlx_lm.stream_generate` exposes no per-step callback for invoking the compressor; (2) upstream's `triattention_generate_step` expects `List[Tuple[mx.array, mx.array]]` raw tensor tuples but mlx-lm passes `KVCache` wrapper objects. Fix path: custom generation loop (~100-200 lines) bridging KVCache ↔ tuples, plus calibration-stats UX + kv_budget setting. Do on a CUDA box or with a small test model — don't ship blind. |
| FU-003 | LongLive integration for Wan 2.1 T2V 1.3B | CUDA platforms (Windows/Linux) only | Real-time causal long video gen ([triattention/longlive](https://github.com/WeianMao/triattention/tree/main/longlive)). We ship the target model already. Needs: new video backend branch in [backend_service/video_runtime.py](backend_service/video_runtime.py), LoRA weights download, torchrun orchestration, UI affordance for long-clip mode. Flash Attention dep. |
| FU-004 | TriAttention SGLang backend | When/if we adopt SGLang as an inference backend | Added upstream 2026-04-22 as v0.2.0. No action unless SGLang lands in our runtime. |
| ~~FU-005~~ | ~~arozanov v_only TurboQuant MLX mode~~ | **Dropped 2026-04-24** | Our current `turboquant-mlx-full` 0.1.3 path already runs without any mlx-lm fork — uses pip `TurboQuantKVCache` with `QuantizedKVCache` fallback ([turboquant_mlx/__init__.py:174-186](turboquant_mlx/__init__.py)). `VOnlyTurboQuantCache` is only in the arozanov fork (we track but don't consume). Value prop already satisfied; entry removed. |
| FU-006 | Re-verify dflash-mlx pin | Quarterly, or when Qwen/Llama drafts land | Currently `f825ffb` = v0.1.4.1 (latest). Upstream deleted tags April 2026 — pin by commit. |

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
| Frontend API client | `src/api.test.ts` | `npm test` |
| Frontend utilities | `src/utils/__tests__/*.test.ts` | `npm test` |

### Minimum Test Expectations
- All existing tests must pass — zero regressions
- New backend features should include at least basic happy-path tests
- Cache strategy changes must test `llama_cpp_cache_flags()` returns valid types
- New API endpoints need at least a shape/contract test

---

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
| `CHAOSENGINE_VENDOR_PATH` | Override ChaosEngine vendor path | `vendor/ChaosEngine/` |
