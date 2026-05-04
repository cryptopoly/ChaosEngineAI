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
| turboquant-mlx-full | `manjunathshiva/turboquant-mlx` | — | `.venv/bin/pip index versions turboquant-mlx-full 2>/dev/null` |
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
| ~~FU-001~~ | ~~Bump `turboquant` to 0.3.x~~ | **Shipped 2026-05-03.** | `turboquant-mlx-full` 0.3.0 published to PyPI; `[turboquant]` extra pin bumped from `>=0.1.3` to `>=0.3.0` in [pyproject.toml](pyproject.toml). Adds asymmetric K/V bits, layer-adaptive precision, `--no-quant` eval flag, NumPy 2.0 + transformers 5.x compat. Verified backward compatible — full ``test_cache_strategies.py`` + ``test_image_runtime.py`` + ``test_video_runtime.py`` (190 tests) pass against 0.3.0. The `turboquant` (HuggingFace) and `turboquant-mlx` (arozanov fork) packages stay on their existing pins; only the active `turboquant-mlx-full` path advances. |
| ~~FU-002~~ | ~~Wire TriAttention MLX compressor into mlx_worker~~ | **Shipped 2026-05-03.** | Unblocked by triattention 0.2.0's MLX port (RavenX AI, 2026-04-09): `apply_triattention_mlx(model, kv_budget=N)` operates on the model directly, bypassing the `mlx_lm.stream_generate` callback gap. Spike at [scripts/spike_triattention_mlx.py](scripts/spike_triattention_mlx.py) confirmed 2.63× speedup with identical output on Qwen2.5-0.5B-Instruct-4bit (norm-only scoring works without calibration stats). Wired into `WorkerState._apply_cache_profile` ([backend_service/mlx_worker.py](backend_service/mlx_worker.py)) via a new `_apply_triattention_mlx_compressor` branch — when `cacheStrategy == "triattention"` the worker delegates to `cache_compression.registry.get("triattention").apply_mlx_compressor(model, kv_budget=self.kv_budget)`. `kvBudget` request param defaults to 2048; falls back to native cache on any failure (model None, registry missing, strategy unavailable, apply raises). |
| FU-003 | LongLive integration for Wan 2.1 T2V 1.3B | CUDA platforms (Windows/Linux) only | Real-time causal long video gen ([triattention/longlive](https://github.com/WeianMao/triattention/tree/main/longlive)). We ship the target model already. Needs: new video backend branch in [backend_service/video_runtime.py](backend_service/video_runtime.py), LoRA weights download, torchrun orchestration, UI affordance for long-clip mode. Flash Attention dep. |
| FU-004 | TriAttention SGLang backend | When/if we adopt SGLang as an inference backend | Added upstream 2026-04-22 as v0.2.0. No action unless SGLang lands in our runtime. |
| ~~FU-005~~ | ~~arozanov v_only TurboQuant MLX mode~~ | **Dropped 2026-04-24** | Our current `turboquant-mlx-full` 0.1.3 path already runs without any mlx-lm fork — uses pip `TurboQuantKVCache` with `QuantizedKVCache` fallback ([turboquant_mlx/__init__.py:174-186](turboquant_mlx/__init__.py)). `VOnlyTurboQuantCache` is only in the arozanov fork (we track but don't consume). Value prop already satisfied; entry removed. |
| FU-006 | Re-verify dflash-mlx pin | Quarterly, or when Qwen/Llama drafts land | Currently `f825ffb` = v0.1.4.1 (latest). Upstream deleted tags April 2026 — pin by commit. |
| ~~FU-007~~ | ~~TeaCache for Wan2.1/2.2~~ | **Obsoleted 2026-05-03 by FU-015.** | TeaCache patches for FLUX + HunyuanVideo + LTX-Video + CogVideoX + Mochi remain under [cache_compression/_teacache_patches/](cache_compression/_teacache_patches/). The Wan-specific port that was deferred here is no longer needed: diffusers 0.36 ships a model-agnostic `apply_first_block_cache` hook (FU-015) that operates on `pipeline.transformer` regardless of model, so Wan caches via the same generic strategy without a vendored forward. Pick FBCache for Wan; TeaCache stays available as the alternative for FLUX-family pipelines. |
| ~~FU-008~~ | ~~`stable-diffusion.cpp` engine (cross-platform diffusion)~~ | **Shipped 2026-05-03 (video) + 2026-05-04 (image).** | Binary build via [scripts/build-sdcpp.sh](scripts/build-sdcpp.sh) + [scripts/update-sdcpp.sh](scripts/update-sdcpp.sh) (clones to `/tmp/stable-diffusion.cpp`, cmake `-DSD_METAL=ON` on Darwin or `-DSD_CUBLAS=ON` on Linux+CUDA, installs to `~/.chaosengine/bin/sd`). Build target is `sd-cli` (renamed from `sd` upstream around master-590); installer copies it back to the legacy `sd` filename so downstream resolvers in [sdcpp_video_runtime.py](backend_service/sdcpp_video_runtime.py), [sdcpp_image_runtime.py](backend_service/sdcpp_image_runtime.py), and [stage-runtime.mjs](scripts/stage-runtime.mjs) keep working. Path resolution in [src-tauri/src/lib.rs](src-tauri/src/lib.rs). **Video lane** (`SdCppVideoEngine.generate`): subprocess spawn → maps `VideoGenerationConfig` → sd.cpp flags (`--diffusion-model`, `-p`, `-W/-H`, `--steps`, `--cfg-scale`, `--seed`, `-o`, `--video-frames`, `--fps`, `--negative-prompt`); regex-parses `step N/M` (or `[N/M]`) into `VIDEO_PROGRESS`; reads `.webm` bytes back (sd.cpp's video output is `.webm`/`.avi`/animated `.webp` — no native `.mp4`). Catalog requires `ggufRepo` + `ggufFile` pin (e.g. `QuantStack/Wan2.2-TI2V-5B-GGUF`). **Image lane** (`SdCppImageEngine.generate`, [sdcpp_image_runtime.py](backend_service/sdcpp_image_runtime.py)): mirrors video shape but emits PNG, drops `--video-frames`/`--fps`, batches by looping seeds (sd.cpp renders one image per invocation). Manager dispatch in [image_runtime.py](backend_service/image_runtime.py) `ImageRuntimeManager.generate` routes when `config.runtime == "sdcpp"`, falls through to diffusers on probe failure or runtime error. Catalog variants: `FLUX.1-schnell-sdcpp-q4km` + `FLUX.1-dev-sdcpp-q4km` ([catalog/image_models.py](backend_service/catalog/image_models.py)). Supported image repos: FLUX.1/2 family, SD3.5, SDXL, SD2.1, Qwen-Image (+ 2512), Z-Image (+ Turbo). |
| FU-009 | mlx-video (Blaizzy) Apple Silicon video engine | **LTX-2 shipped 2026-04-26. Wan convert foundation shipped 2026-05-04 (FU-025); runtime routing pending.** | [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) (MIT, 198⭐). LTX-2 paths (`prince-canuma/LTX-2-{distilled,dev,2.3-distilled,2.3-dev}`) routed through subprocess engine in [backend_service/mlx_video_runtime.py](backend_service/mlx_video_runtime.py). **Wan convert helper now landed** ([backend_service/mlx_video_wan_convert.py](backend_service/mlx_video_wan_convert.py), see FU-025) — promotes raw Wan-AI checkpoints to MLX format under `~/.chaosengine/mlx-video-wan/<slug>/`. Routing extension still pending: until `_SUPPORTED_REPOS` + `_REPO_ENTRY_POINTS` in `mlx_video_runtime.py` learn to detect converted Wan dirs, Wan paths still use diffusers MPS (which is fine for Wan2.1 1.3B / Wan2.2 5B on a 64 GB Mac). |
| FU-010 | vllm-swift Apple Silicon backend (**watch-closely**) | Re-evaluate end of June 2026 | [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) — Swift/Metal vLLM forward pass, Python orchestration only. 2.4× over mlx_lm on Qwen3-0.6B single-request; matches vLLM at concurrency 64. Fills the macOS vLLM gap. **Posture upgraded 2026-05-03** from watch-only after 76 → 238 stars and 1 → 15 forks in ~10 days; v0.3.0 (2026-04-28) shipped Metal Invalid Resource race fix + ~10% TQ MoE perf, v0.2.2 (2026-04-26) added hybrid model batched decode + paged-attention. Single contributor still. Trip-wires for adoption: ≥3 contributors with merged commits OR public benchmark beating mlx_lm at concurrency >1 on Llama-3.x-8B-class (current 2.4× claim is Qwen3-0.6B single-request only). |
| FU-011 | LTX-Video 2.3 diffusers variant | Lightricks publishes diffusers-compatible weights (`Lightricks/LTX-2.3` gains `model_index.json`) | LTX-2.3 currently routes via mlx-video on Apple Silicon (`prince-canuma/LTX-2.3-{distilled,dev}` already in catalog). Lightricks' own model card states "diffusers support coming soon". When the diffusers-shaped weights land, add a `Lightricks/LTX-Video-2.3` entry to [backend_service/catalog/video_models.py](backend_service/catalog/video_models.py) under the `ltx-video` family so RTX 4090 / Linux users get a non-MLX path. Until then, no LTX-2.3 path exists for CUDA. |
| FU-012 | LTX Spatial Temporal Guidance (STG) | diffusers ships LTXPipeline with `perturbed_blocks` kwarg, or vendor a forward patch | Upstream reference workflows enable STG by default — perturbs final transformer blocks during sampling to reduce object breakup / chroma drift. Our pinned diffusers' LTXPipeline does not accept `perturbed_blocks`. Phase D landed `frame_rate` + `decode_timestep` + `decode_noise_scale` + `guidance_rescale` for reference parity on the basic kwargs; STG is the remaining gap. Track upstream; if quality remains short of the reference, vendor a forward patch under [cache_compression/_teacache_patches/ltx_video.py](cache_compression/_teacache_patches/ltx_video.py)-style. |
| FU-013 | Vendored STG-enabled LTX pipeline | Phase F or when a user reports that Phase D + E1 + E2 quality remains short of the upstream reference | Subclass `LTXPipeline` and override `__call__` to add a third forward pass per step with selected transformer block(s) perturbed (skip self-attention or replace with identity). Combine: `pred = uncond + cfg*(text - uncond) + stg_scale*(text - perturbed)`. Reference: Lightricks' upstream LTX-Video repo's `STGSamplingHook`. Estimated ~250 lines of vendored code + tests. Sequence dependency: do this AFTER FU-007 (Wan TeaCache) ships so the cache vs guidance interactions are tested in isolation. |
| FU-014 | LLM-based prompt enhancer | When Phase E1 template-only enhancer underperforms in real use | Phase E1 ships a deterministic per-model template suffix; FU-014 replaces it with a small instruction model (Llama-3.2-1B-Instruct via mlx-lm on Apple Silicon, or a 1B GGUF via llama-server elsewhere) that auto-rewrites short prompts into the structured 50-100 word format each video DiT was trained on. Reuses existing inference infrastructure — no new model bundling beyond a 1-2 GB checkpoint. |
| FU-015 | First Block Cache (diffusers 0.36 generic hook) | **Shipped 2026-05-03.** | Cross-platform diffusion cache strategy backed by `diffusers.hooks.apply_first_block_cache`. Lives at [cache_compression/firstblockcache.py](cache_compression/firstblockcache.py), registered as id `fbcache` in the strategy registry ([cache_compression/__init__.py](cache_compression/__init__.py)). Applies to image + video DiTs (FLUX, SD3.5, Wan2.1/2.2, HunyuanVideo, LTX-Video, CogVideoX, Mochi). Default threshold 0.12 (≈1.8× speedup on FLUX.1-dev with imperceptible quality drift). Same `apply_diffusion_cache_strategy` hook as TeaCache; UNet pipelines (SD1.5/SDXL) raise NotImplementedError into a runtimeNote. Closes FU-007. |
| FU-016 | SageAttention CUDA backend wiring | **Shipped 2026-05-03 (CUDA-gated).** | Helper at [backend_service/helpers/attention_backend.py](backend_service/helpers/attention_backend.py) (`maybe_apply_sage_attention`). Called from both [image_runtime.py](backend_service/image_runtime.py) and [video_runtime.py](backend_service/video_runtime.py) `_ensure_pipeline` after pipeline build. CUDA + sageattention pip wheel + diffusers ≥0.36 + DiT pipeline. No-op on macOS / CPU / UNet / non-DiT pipelines. Stacks multiplicatively with FBCache (community Wan2.1 720P cumulative 54%). Setup-page install action (`pip install sageattention`) follows. |
| FU-017 | SDXL VAE fp16 fix on MPS / CUDA | **Shipped 2026-05-03.** | Probes `madebyollin/sdxl-vae-fp16-fix` snapshot via `local_files_only=True` (no surprise download) at pipeline load. When cached, swaps `pipeline.vae` and lets `_preferred_torch_dtype` stay on fp16 for SDXL on MPS — drops the previous fp32 fallback that doubled wall-time on Apple Silicon. Helpers `_is_sdxl_repo` + `_locate_sdxl_vae_fix_snapshot` in [image_runtime.py](backend_service/image_runtime.py). Falls back to stock VAE + fp32 on any failure. |
| FU-018 | TAEHV / TAESD preview decoder | Pending UI work for live denoise thumbnails | Tiny VAE for cheap preview decode each step. Ships as a quality knob — preview-only by default, full VAE for final output. Will use `madebyollin/taesd` for SD/SDXL/SD3 and `madebyollin/taehv` for HunyuanVideo / Wan / LTX. |
| FU-019 | Distill LoRA support (Hyper-SD, FLUX.1-Turbo, lightx2v Wan CausVid) | **Shipped 2026-05-03; extended Phase 3 with Wan2.2-Distill.** | LoRA load + fuse path in both [image_runtime.py](backend_service/image_runtime.py) and [video_runtime.py](backend_service/video_runtime.py) `_ensure_pipeline`. Catalog variants in [catalog/image_models.py](backend_service/catalog/image_models.py) (FLUX.1-dev × Hyper-SD-8step + Turbo-Alpha) and [catalog/video_models.py](backend_service/catalog/video_models.py) (Wan2.1 1.3B/14B × CausVid). **Phase 3 extension: Wan 2.2 A14B I2V × lightx2v 4-step distill.** lightx2v ships full distilled transformers (not LoRAs) for both Wan2.2 MoE experts. New `distillTransformer*` fields on `VideoGenerationConfig` carry repo + high/low-noise filenames + precision (`bf16` / `fp8_e4m3` / `int8`). `_swap_distill_transformers` helper downloads both safetensors via `huggingface_hub.hf_hub_download`, loads via `WanTransformer3DModel.from_single_file`, and reassigns `pipeline.transformer` + `pipeline.transformer_2`. Variant key includes the distill identity so switching variants triggers clean rebuilds. Distill takes precedence over LoRA when both are pinned. Catalog adds: `Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-bf16` + `-distill-fp8`. Schema-default substitution sets `defaultSteps=4` + `cfgOverride=1.0`. |
| FU-020 | AYS (Align Your Steps) schedule for SD/SDXL | **Shipped 2026-05-03.** | New samplers `ays_dpmpp_2m_sd15` / `ays_dpmpp_2m_sdxl` in `_SAMPLER_REGISTRY` ([image_runtime.py](backend_service/image_runtime.py)). Private `_ays_family` token stripped from `from_config` kwargs and stashed on `pipeline._chaosengine_ays_timesteps`; `_build_pipeline_kwargs` passes it via `timesteps=` and pops `num_inference_steps`. Hardcoded NVIDIA timestep arrays for SD1.5/SDXL/SVD. Flow-match models continue to be gated out by `_is_flow_matching_repo`. |
| FU-021 | Image-runtime CFG decay parity | **Shipped 2026-05-03.** | `cfgDecay` field on `ImageGenerationConfig` + `ImageGenerationRequest`. Linear ramp from initial guidance to 1.5 floor inside the existing `callback_on_step_end` in `generate()`. Gated to flow-match repos (`_is_flow_matching_repo`); SD1.5/SDXL ignore the flag. Default off — opt-in vs. video runtime's default-on. |
| FU-022 | Llama-3.2-1B / Florence-2 prompt enhancer | When 1B GGUF download UX ready | Replaces FU-014. Reuses existing llama.cpp engine. |
| FU-023 | SVDQuant / Nunchaku CUDA engine | When CUDA Setup parity confirmed | 3× over NF4 on FLUX.1-dev / SD3.5 / Wan2.2. Separate engine class. CUDA only. |
| FU-024 | FP8 layerwise casting for non-FLUX DiTs | After SVDQuant decision | E4M3 (FLUX/Wan) vs E5M2 (HunyuanVideo). Diffusers `enable_layerwise_casting`. CUDA SM 8.9+ only. |
| ~~FU-025~~ | ~~mlx-video Wan one-shot convert action~~ | **Fully shipped 2026-05-04 (Phase 7 + Phase 8 + Phase 9).** | Closes FU-009 Wan branch. **Phase 7 (foundation):** `[mlx-video]` extra in [pyproject.toml](pyproject.toml) flipped to ``git+https://github.com/Blaizzy/mlx-video.git``. Helper [backend_service/mlx_video_wan_convert.py](backend_service/mlx_video_wan_convert.py) wraps the upstream `python -m mlx_video.models.wan_2.convert` subprocess: `slug_for(repo)` / `output_dir_for(repo)` / `status_for(repo)` / `list_converted()` / `run_convert(checkpoint_dir, repo, dtype, quantize, bits, group_size, timeout)`. Output under ``~/.chaosengine/mlx-video-wan/<slug>/`` (override via ``CHAOSENGINE_MLX_VIDEO_WAN_DIR``). **Phase 8 (routing):** [mlx_video_runtime.py](backend_service/mlx_video_runtime.py) `supported_repos()` returns dynamic union of LTX-2 + converted-on-disk Wan repos. `_REPO_ENTRY_POINTS` adds `"Wan-AI/": "mlx_video.models.wan_2.generate"`. `_build_wan_cmd` produces the Wan-shaped CLI (`--model-dir`, `--guide-scale` string, `--scheduler`, optional `--seed`/`--steps`/`--negative-prompt`; no LTX-2 flags). `generate()` picks `_wan_runtime_note` (flags MoE experts) and skips LTX-2 effective-step / effective-guidance overrides. **Phase 9 (GUI):** Orchestrator [backend_service/mlx_video_wan_installer.py](backend_service/mlx_video_wan_installer.py) drives preflight → download-raw → convert → verify with structured progress events. Setup endpoints in [routes/setup.py](backend_service/routes/setup.py): `POST /api/setup/install-mlx-video-wan` (background-job pattern mirroring `/api/setup/install-longlive`), `GET /api/setup/install-mlx-video-wan/status`, `GET /api/setup/mlx-video-wan/inventory`. Frontend client in [src/api.ts](src/api.ts) (`startWanInstall`, `getWanInstallStatus`, `getWanInventory`). UI panel [src/components/WanInstallPanel.tsx](src/components/WanInstallPanel.tsx) lists every supported Wan repo with raw-size hint + converted badge / install button + live `InstallLogPanel` underneath; rendered in [VideoDiscoverTab.tsx](src/features/video/VideoDiscoverTab.tsx) above the variant grid. Supported raw repos: `Wan-AI/Wan2.{1-T2V-1.3B,1-T2V-14B,2-TI2V-5B,2-T2V-A14B,2-I2V-A14B}`. End-to-end UX: user clicks Install → backend downloads + converts in background → runtime auto-detects + routes Wan generate calls through mlx-video. Tests: 21 in [test_mlx_video_wan_convert.py](tests/test_mlx_video_wan_convert.py), 9 Wan-routing in [test_mlx_video.py](tests/test_mlx_video.py), 15 in [test_mlx_video_wan_installer.py](tests/test_mlx_video_wan_installer.py). |
| ~~FU-026~~ | ~~TaylorSeer + DBCache aggressive cache preset~~ | **Obsoleted 2026-05-03 by diffusers 0.38 core.** | Diffusers 0.38.0 (2026-05-01) ships ``TaylorSeerCacheConfig``, ``MagCacheConfig``, ``PyramidAttentionBroadcastConfig``, ``FasterCacheConfig`` natively — no ``cache-dit`` dependency required. Wired as registry strategies (ids ``taylorseer``, ``magcache``, ``pab``, ``fastercache``) in [cache_compression/__init__.py](cache_compression/__init__.py). Each adapter calls ``pipeline.transformer.enable_cache(<Config>)``. UNet pipelines (SD1.5/SDXL) raise ``NotImplementedError`` into a runtimeNote, matching the FBCache contract. MagCache is FLUX-only without calibration UX (uses ``FLUX_MAG_RATIOS`` from ``diffusers.hooks.mag_cache``); other DiTs raise a "calibration required" message until that UX lands. |
| FU-027 | NVIDIA/kvpress KV cache toolkit (CUDA-side) | Alongside FU-023 SVDQuant CUDA engine, when CUDA Setup parity confirmed | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) — Apache 2.0, 1.1k stars, pip-installable (``kvpress``). v0.5.3 released 2026-04-09; 26 releases. HF transformers + multi-GPU Accelerate hookups. Most active KV-cache toolkit on GitHub (NVIDIA-maintained). Candidate for CUDA-only KV compression alongside Nunchaku weight quant; complements rather than replaces TurboQuant on Apple Silicon. Sequence: pick this up after FU-023 confirms the CUDA install path. |

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
