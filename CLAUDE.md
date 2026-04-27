# ChaosEngineAI — Project Guide

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
| FU-007 | TeaCache diffusion cache strategy | **FLUX + HunyuanVideo + LTX-Video + CogVideoX + Mochi shipped 2026-04-26.** Wan2.1 still pending. | Five `teacache_forward` patches live under [cache_compression/_teacache_patches/](cache_compression/_teacache_patches/) — FLUX vendored from upstream, the four video DiTs authored as diffusers-shaped ports (upstream targets standalone repos with different forward signatures, so not directly vendorable). Per-model rescale coefficients pulled from upstream's calibration tables. **Wan2.1 still excluded** — ali-vilab `teacache_generate.py` targets Wan-Video/Wan2.1 (signature `(self, x, t, context, seq_len, clip_fea, y)`); diffusers `WanTransformer3DModel` block structure differs enough that a faithful port needs calibration access (deferred). Reference: [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) (Apache 2.0). Quality knob `rel_l1_thresh` default 0.4. |
| FU-008 | `stable-diffusion.cpp` engine (cross-platform diffusion) | **Scaffold shipped 2026-04-26.** Generate path (CLI subprocess + stdout progress parser) still pending. | Binary staging in [scripts/stage-runtime.mjs](scripts/stage-runtime.mjs) (mirrors `llama-server-turbo` pattern: `CHAOSENGINE_SDCPP_BIN_DIR` → `~/.chaosengine/bin/` → `../stable-diffusion.cpp/build/bin/`). Path resolution in [src-tauri/src/lib.rs](src-tauri/src/lib.rs) (`resolve_sd_cpp` + `CHAOSENGINE_SDCPP_BIN` env injection in both embedded and source-workspace branches). Engine class in [backend_service/sdcpp_video_runtime.py](backend_service/sdcpp_video_runtime.py) (`SdCppVideoEngine`) — `probe()` returns binary-presence status; `preload`/`unload` track loaded repo; `generate()` raises `NotImplementedError` until CLI arg builders + progress parser land. Manager exposes `sdcpp_video_capabilities()` so Setup/Studio can surface staging state. Models: SD 1.x/2.x/XL, FLUX.1/2, **Wan2.1/2.2 video**, Qwen Image, Z-Image — video subset wired only for Wan repos. Repo [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT). |
| FU-009 | mlx-video (Blaizzy) Apple Silicon video engine | **LTX-2 shipped 2026-04-26.** Wan still scaffold. | [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) (MIT, 198⭐). LTX-2 paths (`prince-canuma/LTX-2-{distilled,dev,2.3-distilled,2.3-dev}`) routed through subprocess engine in [backend_service/mlx_video_runtime.py](backend_service/mlx_video_runtime.py); manager dispatch lives at [backend_service/video_runtime.py](backend_service/video_runtime.py) `VideoRuntimeManager.generate`. **Wan stays diffusers MPS** — mlx-video Wan2.1/2.2 require an explicit `mlx_video.models.wan_2.convert` step on raw HF weights (no pre-converted MLX repo today). Bundling that conversion into a one-shot install action will promote Wan to mlx-video; until then, Wan paths use diffusers MPS, which is fine for Wan2.1 1.3B / Wan2.2 5B on a 64 GB Mac. |
| FU-010 | vllm-swift Apple Silicon backend (**watch-only**) | Re-evaluate after 1–2 releases or mid-2026; skip if stars/commits stagnate | [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) — Swift/Metal vLLM forward pass, Python orchestration only. 2.4× over mlx_lm on Qwen3-0.6B single-request; matches vLLM at concurrency 64. Fills the macOS vLLM gap. Low-activity single fork (76 commits, 1 open issue) — treat as experimental. Action: monitor. No code this cycle. |
| FU-011 | LTX-Video 2.3 diffusers variant | Lightricks publishes diffusers-compatible weights (`Lightricks/LTX-2.3` gains `model_index.json`) | LTX-2.3 currently routes via mlx-video on Apple Silicon (`prince-canuma/LTX-2.3-{distilled,dev}` already in catalog). Lightricks' own model card states "diffusers support coming soon". When the diffusers-shaped weights land, add a `Lightricks/LTX-Video-2.3` entry to [backend_service/catalog/video_models.py](backend_service/catalog/video_models.py) under the `ltx-video` family so RTX 4090 / Linux users get a non-MLX path. Until then, no LTX-2.3 path exists for CUDA. |
| FU-012 | LTX Spatial Temporal Guidance (STG) | diffusers ships LTXPipeline with `perturbed_blocks` kwarg, or vendor a forward patch | Upstream reference workflows enable STG by default — perturbs final transformer blocks during sampling to reduce object breakup / chroma drift. Our pinned diffusers' LTXPipeline does not accept `perturbed_blocks`. Phase D landed `frame_rate` + `decode_timestep` + `decode_noise_scale` + `guidance_rescale` for reference parity on the basic kwargs; STG is the remaining gap. Track upstream; if quality remains short of the reference, vendor a forward patch under [cache_compression/_teacache_patches/ltx_video.py](cache_compression/_teacache_patches/ltx_video.py)-style. |

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
