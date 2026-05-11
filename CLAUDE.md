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
| dflash-mlx | `bstnxbt/dflash-mlx` | `main` pinned to commit `f825ffb2` (upstream deleted all tags April 2026) | `git ls-remote https://github.com/bstnxbt/dflash-mlx.git refs/heads/main` |
| turboquant-mlx-full | `manjunathshiva/turboquant-mlx` | — | `.venv/bin/pip index versions turboquant-mlx-full 2>/dev/null` |
| DDTree (ported algorithm) | `liranringel/ddtree` | `main` | `git ls-remote https://github.com/liranringel/ddtree.git HEAD` |

### 4. Cache Strategy Health
- [ ] Native strategy `llama_cpp_cache_flags()` only emits standard types: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
- [ ] TurboQuant strategy returns `required_llama_binary() == "turbo"`
- [ ] Legacy `chaosengine` + `rotorquant` ids coerce to `turboquant` via `registry.resolve_legacy_id`
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
| ~~FU-006~~ | ~~Re-verify dflash-mlx pin~~ | **Bumped to `fada1eb` (HEAD) on 2026-05-10. Previously bumped to `8d8545d` = v0.1.5.1 on 2026-05-05 after the ddtree.py rewrite landed.** | 2026-05-10 bump from `8d8545d` to `fada1eb` covers 11 upstream commits including the new Gemma4 DFlash backend (commit 05cc456, "feat: add Gemma4 DFlash backend"), the v0.1.5 serving surface, live server metrics endpoint, prefix-cache survival test gate, async L2 writer fix, long-context runtime diagnostics hardening, benchmark slugging fixes, and a license switch to Apache-2.0. Same fix applied in both [pyproject.toml](pyproject.toml) (already correct) and [scripts/stage-runtime.mjs](scripts/stage-runtime.mjs) (was lagging on `f825ffb` v0.1.4.1 — staged release runtime would have shipped the old binary). The two pins now live as the same hex string in both files; CI's pre-build-check should grow a sync assert. No breaking API changes between the pins per upstream commit log. **Earlier bump notes:** Pin advanced from `f825ffb` (v0.1.4.1) to `8d8545d` (v0.1.5.1). 0.1.5+ moved every primitive that [backend_service/ddtree.py](backend_service/ddtree.py) consumed off the runtime top-level onto a per-family `target_ops` adapter — `target_forward_with_hidden_states` → `target_ops.forward_with_hidden_capture`, `extract_context_feature_from_dict` → `target_ops.extract_context_feature`, `make_target_cache` → `target_ops.make_cache`, `_target_embed_tokens` → `target_ops.embed_tokens`, `_target_text_model` → `target_ops.text_model`, `_lm_head_logits` → `target_ops.logits_from_hidden`. `ContextOnlyDraftKVCache` moved to `dflash_mlx.model`; `create_attention_mask` re-imported from `mlx_lm.models.base`; `trim_cache_to` was removed entirely and now lives as a thin local `_trim_cache_to` shim that calls each entry's own `.rollback()` / `.trim()` / `.crop()`. Adapter resolved once at the top of `generate_ddtree_mlx` via `resolve_target_ops(target_model)`. Live smoke 2026-05-05 against `mlx-community/Qwen2.5-0.5B-Instruct-4bit` confirmed adapter resolves (`backend=qwen_gdn`, `family=pure_attention`), forward+capture / embed_tokens / text_model / logits_from_hidden / extract_context_feature / `_trim_cache_to` all working. Gains over 0.1.4.1: draft model quantization with Metal MMA kernels, branchless Metal kernels + fused draft KV projections, long-context runtime diagnostics. Re-check cadence resets to quarterly. |
| ~~FU-007~~ | ~~TeaCache for Wan2.1/2.2~~ | **Obsoleted 2026-05-03 by FU-015.** | TeaCache patches for FLUX + HunyuanVideo + LTX-Video + CogVideoX + Mochi remain under [cache_compression/_teacache_patches/](cache_compression/_teacache_patches/). The Wan-specific port that was deferred here is no longer needed: diffusers 0.36 ships a model-agnostic `apply_first_block_cache` hook (FU-015) that operates on `pipeline.transformer` regardless of model, so Wan caches via the same generic strategy without a vendored forward. Pick FBCache for Wan; TeaCache stays available as the alternative for FLUX-family pipelines. |
| ~~FU-008~~ | ~~`stable-diffusion.cpp` engine (cross-platform diffusion)~~ | **Shipped 2026-05-03 (video) + 2026-05-04 (image).** | Binary build via [scripts/build-sdcpp.sh](scripts/build-sdcpp.sh) + [scripts/update-sdcpp.sh](scripts/update-sdcpp.sh) (clones to `/tmp/stable-diffusion.cpp`, cmake `-DSD_METAL=ON` on Darwin or `-DSD_CUBLAS=ON` on Linux+CUDA, installs to `~/.chaosengine/bin/sd`). Build target is `sd-cli` (renamed from `sd` upstream around master-590); installer copies it back to the legacy `sd` filename so downstream resolvers in [sdcpp_video_runtime.py](backend_service/sdcpp_video_runtime.py), [sdcpp_image_runtime.py](backend_service/sdcpp_image_runtime.py), and [stage-runtime.mjs](scripts/stage-runtime.mjs) keep working. Path resolution in [src-tauri/src/lib.rs](src-tauri/src/lib.rs). **Video lane** (`SdCppVideoEngine.generate`): subprocess spawn → maps `VideoGenerationConfig` → sd.cpp flags (`--diffusion-model`, `-p`, `-W/-H`, `--steps`, `--cfg-scale`, `--seed`, `-o`, `--video-frames`, `--fps`, `--negative-prompt`); regex-parses `step N/M` (or `[N/M]`) into `VIDEO_PROGRESS`; reads `.webm` bytes back (sd.cpp's video output is `.webm`/`.avi`/animated `.webp` — no native `.mp4`). Catalog requires `ggufRepo` + `ggufFile` pin (e.g. `QuantStack/Wan2.2-TI2V-5B-GGUF`). **Image lane** (`SdCppImageEngine.generate`, [sdcpp_image_runtime.py](backend_service/sdcpp_image_runtime.py)): mirrors video shape but emits PNG, drops `--video-frames`/`--fps`, batches by looping seeds (sd.cpp renders one image per invocation). Manager dispatch in [image_runtime.py](backend_service/image_runtime.py) `ImageRuntimeManager.generate` routes when `config.runtime == "sdcpp"`, falls through to diffusers on probe failure or runtime error. Catalog variants: `FLUX.1-schnell-sdcpp-q4km` + `FLUX.1-dev-sdcpp-q4km` ([catalog/image_models.py](backend_service/catalog/image_models.py)). Supported image repos: FLUX.1/2 family, SD3.5, SDXL, SD2.1, Qwen-Image (+ 2512), Z-Image (+ Turbo). |
| ~~FU-009~~ | ~~mlx-video (Blaizzy) Apple Silicon video engine~~ | **Fully shipped 2026-05-04. Live smoke validated end-to-end.** | LTX-2 paths (`prince-canuma/LTX-2-{distilled,dev,2.3-distilled,2.3-dev}`) routed through subprocess engine in [backend_service/mlx_video_runtime.py](backend_service/mlx_video_runtime.py); Wan-AI paths route via Phase 8 of FU-025 (`_is_wan_repo` + `_build_wan_cmd` + `_REPO_ENTRY_POINTS["Wan-AI/"] = "mlx_video.models.wan_2.generate"`). Live smoke 2026-05-04 against `Wan-AI/Wan2.1-T2V-1.3B` (480×272, 5 frames, 4 steps, unipc): T5 encode 14.1s + transformer load 0.2s (4-bit q) + denoise 2.9s @ 1.4 it/s + VAE decode 1.3s = 19.6s total, 383 KB .mp4 output. The smoke also surfaced + fixed a `status_for` filename gap — mlx-video upstream emits root-level `model.safetensors` + `t5_encoder.safetensors`, not the legacy `transformer*.safetensors` / `text_encoder*.safetensors` patterns the helper originally checked for. Both now match. |
| FU-010 | vllm-swift Apple Silicon backend (**watch-closely**) | Re-evaluate end of June 2026 | [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) — Swift/Metal vLLM forward pass, Python orchestration only. 2.4× over mlx_lm on Qwen3-0.6B single-request; matches vLLM at concurrency 64. Fills the macOS vLLM gap. **Posture upgraded 2026-05-03** from watch-only after 76 → 238 stars and 1 → 15 forks in ~10 days; v0.3.0 (2026-04-28) shipped Metal Invalid Resource race fix + ~10% TQ MoE perf, v0.2.2 (2026-04-26) added hybrid model batched decode + paged-attention. Single contributor still. Trip-wires for adoption: ≥3 contributors with merged commits OR public benchmark beating mlx_lm at concurrency >1 on Llama-3.x-8B-class (current 2.4× claim is Qwen3-0.6B single-request only). |
| FU-011 | LTX-Video 2.3 diffusers variant | Lightricks publishes diffusers-compatible weights (`Lightricks/LTX-2.3` gains `model_index.json`) | LTX-2.3 currently routes via mlx-video on Apple Silicon (`prince-canuma/LTX-2.3-{distilled,dev}` already in catalog). Lightricks' own model card states "diffusers support coming soon". When the diffusers-shaped weights land, add a `Lightricks/LTX-Video-2.3` entry to [backend_service/catalog/video_models.py](backend_service/catalog/video_models.py) under the `ltx-video` family so RTX 4090 / Linux users get a non-MLX path. Until then, no LTX-2.3 path exists for CUDA. |
| FU-012 | LTX Spatial Temporal Guidance (STG) | diffusers ships LTXPipeline with `perturbed_blocks` kwarg, or vendor a forward patch | Upstream reference workflows enable STG by default — perturbs final transformer blocks during sampling to reduce object breakup / chroma drift. Our pinned diffusers' LTXPipeline does not accept `perturbed_blocks`. Phase D landed `frame_rate` + `decode_timestep` + `decode_noise_scale` + `guidance_rescale` for reference parity on the basic kwargs; STG is the remaining gap. Track upstream; if quality remains short of the reference, vendor a forward patch under [cache_compression/_teacache_patches/ltx_video.py](cache_compression/_teacache_patches/ltx_video.py)-style. |
| FU-013 | Vendored STG-enabled LTX pipeline | Phase F or when a user reports that Phase D + E1 + E2 quality remains short of the upstream reference | Subclass `LTXPipeline` and override `__call__` to add a third forward pass per step with selected transformer block(s) perturbed (skip self-attention or replace with identity). Combine: `pred = uncond + cfg*(text - uncond) + stg_scale*(text - perturbed)`. Reference: Lightricks' upstream LTX-Video repo's `STGSamplingHook`. Estimated ~250 lines of vendored code + tests. Sequence dependency: do this AFTER FU-007 (Wan TeaCache) ships so the cache vs guidance interactions are tested in isolation. |
| ~~FU-014~~ | ~~LLM-based prompt enhancer~~ | **Closed 2026-05-04 by FU-022.** | Replaced by FU-022's MLX-native enhancer (see below). |
| FU-015 | First Block Cache (diffusers 0.36 generic hook) | **Shipped 2026-05-03.** | Cross-platform diffusion cache strategy backed by `diffusers.hooks.apply_first_block_cache`. Lives at [cache_compression/firstblockcache.py](cache_compression/firstblockcache.py), registered as id `fbcache` in the strategy registry ([cache_compression/__init__.py](cache_compression/__init__.py)). Applies to image + video DiTs (FLUX, SD3.5, Wan2.1/2.2, HunyuanVideo, LTX-Video, CogVideoX, Mochi). Default threshold 0.12 (≈1.8× speedup on FLUX.1-dev with imperceptible quality drift). Same `apply_diffusion_cache_strategy` hook as TeaCache; UNet pipelines (SD1.5/SDXL) raise NotImplementedError into a runtimeNote. Closes FU-007. |
| FU-016 | SageAttention CUDA backend wiring | **Shipped 2026-05-03 (CUDA-gated).** | Helper at [backend_service/helpers/attention_backend.py](backend_service/helpers/attention_backend.py) (`maybe_apply_sage_attention`). Called from both [image_runtime.py](backend_service/image_runtime.py) and [video_runtime.py](backend_service/video_runtime.py) `_ensure_pipeline` after pipeline build. CUDA + sageattention pip wheel + diffusers ≥0.36 + DiT pipeline. No-op on macOS / CPU / UNet / non-DiT pipelines. Stacks multiplicatively with FBCache (community Wan2.1 720P cumulative 54%). Setup-page install action (`pip install sageattention`) follows. |
| FU-017 | SDXL VAE fp16 fix on MPS / CUDA | **Shipped 2026-05-03.** | Probes `madebyollin/sdxl-vae-fp16-fix` snapshot via `local_files_only=True` (no surprise download) at pipeline load. When cached, swaps `pipeline.vae` and lets `_preferred_torch_dtype` stay on fp16 for SDXL on MPS — drops the previous fp32 fallback that doubled wall-time on Apple Silicon. Helpers `_is_sdxl_repo` + `_locate_sdxl_vae_fix_snapshot` in [image_runtime.py](backend_service/image_runtime.py). Falls back to stock VAE + fp32 on any failure. |
| ~~FU-018~~ | ~~TAEHV / TAESD preview decoder~~ | **Fully shipped 2026-05-04 (parts 1 + 2).** | Tiny VAE for cheap preview decode each step. **Part 1 — full-decode VAE swap** ([backend_service/helpers/preview_vae.py](backend_service/helpers/preview_vae.py)) maps repo → preview VAE id (FLUX.1/2 → taef1/taef2, SD3 → taesd3, SDXL incl. sdxl-turbo + SDXL-Lightning → taesdxl, SD1.x/2.x incl. sd-turbo → taesd, Wan2.x → taew2_2, LTX-Video / LTX-2 → taeltx2_3_wide, HunyuanVideo → taehv1_5, CogVideoX → taecogvideox, Mochi → taemochi, Qwen-Image → taeqwenimage). `maybe_apply_preview_vae(pipeline, repo, enabled)` swaps `pipeline.vae` for an `AutoencoderTiny`, mirrors the stock VAE's dtype + device (live-validated against SDXL-Turbo on MPS — without the device mirror the first decoder pass raises `MPSHalfType` vs `torch.HalfTensor`). **Part 2 — live per-step thumbnails** ([backend_service/helpers/preview_thumbnails.py](backend_service/helpers/preview_thumbnails.py)) decodes `callback_kwargs["latents"]` through the swapped tiny VAE inside `callback_on_step_end`, scales to ≤192 px, base64-encodes a PNG, publishes to `IMAGE_PROGRESS.set_thumbnail` / `VIDEO_PROGRESS.set_thumbnail`. Stride caps emit count at ~8 (image) / ~6 (video) per gen so the polled `/api/{images,video}/progress` endpoint stays cheap. Handles both standard 4D `(B, C, H, W)` latents (SD1.5 / SDXL / SD3) and FLUX's packed 3D `(B, seq_len, 64)` shape via `pipeline._unpack_latents` (live-validated against FLUX.1-schnell on MPS — 4 thumbnails captured per 4-step gen, all valid base64 PNGs at 192x192). Frontend reads `snapshot.thumbnail` from `useGenerationProgress`, renders inside `LiveProgress` between the bar and the phase list when present. Errors are best-effort: a decode crash never aborts the actual generation — caller catches and falls back to no-thumbnail. **LTX refiner private-kwarg fix:** the FU-018 part 2 wiring also caught + fixed a pre-existing leak where `_invoke_pipeline_with_ltx_refiner` was passing `__cfg_decay` directly into `LTXPipeline.__call__` (would have started leaking `__preview_vae` too). Both private kwargs now stripped in the refiner path. |
| FU-019 | Distill LoRA support (Hyper-SD, FLUX.1-Turbo, lightx2v Wan CausVid) | **Shipped 2026-05-03; extended Phase 3 with Wan2.2-Distill.** | LoRA load + fuse path in both [image_runtime.py](backend_service/image_runtime.py) and [video_runtime.py](backend_service/video_runtime.py) `_ensure_pipeline`. Catalog variants in [catalog/image_models.py](backend_service/catalog/image_models.py) (FLUX.1-dev × Hyper-SD-8step + Turbo-Alpha) and [catalog/video_models.py](backend_service/catalog/video_models.py) (Wan2.1 1.3B/14B × CausVid). **Phase 3 extension: Wan 2.2 A14B I2V × lightx2v 4-step distill.** lightx2v ships full distilled transformers (not LoRAs) for both Wan2.2 MoE experts. New `distillTransformer*` fields on `VideoGenerationConfig` carry repo + high/low-noise filenames + precision (`bf16` / `fp8_e4m3` / `int8`). `_swap_distill_transformers` helper downloads both safetensors via `huggingface_hub.hf_hub_download`, loads via `WanTransformer3DModel.from_single_file`, and reassigns `pipeline.transformer` + `pipeline.transformer_2`. Variant key includes the distill identity so switching variants triggers clean rebuilds. Distill takes precedence over LoRA when both are pinned. Catalog adds: `Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-bf16` + `-distill-fp8`. Schema-default substitution sets `defaultSteps=4` + `cfgOverride=1.0`. |
| FU-020 | AYS (Align Your Steps) schedule for SD/SDXL | **Shipped 2026-05-03.** | New samplers `ays_dpmpp_2m_sd15` / `ays_dpmpp_2m_sdxl` in `_SAMPLER_REGISTRY` ([image_runtime.py](backend_service/image_runtime.py)). Private `_ays_family` token stripped from `from_config` kwargs and stashed on `pipeline._chaosengine_ays_timesteps`; `_build_pipeline_kwargs` passes it via `timesteps=` and pops `num_inference_steps`. Hardcoded NVIDIA timestep arrays for SD1.5/SDXL/SVD. Flow-match models continue to be gated out by `_is_flow_matching_repo`. |
| FU-021 | Image-runtime CFG decay parity | **Shipped 2026-05-03.** | `cfgDecay` field on `ImageGenerationConfig` + `ImageGenerationRequest`. Linear ramp from initial guidance to 1.5 floor inside the existing `callback_on_step_end` in `generate()`. Gated to flow-match repos (`_is_flow_matching_repo`); SD1.5/SDXL ignore the flag. Default off — opt-in vs. video runtime's default-on. |
| ~~FU-022~~ | ~~LLM-based prompt enhancer~~ | **Shipped 2026-05-04 (Apple Silicon path).** | Replaces the deterministic per-family template-suffix enhancer in `_enhance_prompt`. Helper [backend_service/helpers/prompt_enhancer.py](backend_service/helpers/prompt_enhancer.py) wraps `mlx_lm.load` + `mlx_lm.generate` against a small instruct model (default `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, ~700 MB on disk, ~3s cold load + sub-second per call) — cached in a process-level `_EnhancerSingleton` so the second call onward hits the warm model. Per-family system prompts (`wan` / `ltx` / `hunyuan` / `flux` / `sdxl` / `sd3` / `default`) anchor the rewrite to the DiT's training distribution. `family_for(repo)` matches longest-prefix-wins. Endpoint `POST /api/prompt/enhance` ([routes/prompts.py](backend_service/routes/prompts.py)) returns `{enhanced, note, modelUsed, family}`. Frontend exposes a "Enhance" pill button next to the Prompt label in both Studio tabs ([components/PromptEnhanceButton.tsx](src/components/PromptEnhanceButton.tsx)) — click triggers the rewrite + replaces the textarea on success or surfaces a tooltip note when the enhancer fell back. Failure modes (non-Apple platform, mlx_lm missing, model not cached, generation crash, shorter-than-input rewrite) all return the original prompt + a runtimeNote so the user sees why. Live smoke 2026-05-04: 6-word "a fluffy cat on a windowsill" → 16-word FLUX rewrite (3.2s cold), 13-word Wan rewrite (0.12s warm), 8-word LTX rewrite (0.11s warm). 16 unit tests covering family-mapping + happy path + load-failure + generation crash + shorter-rewrite reject + quote stripping. CUDA / Linux still get the legacy template suffix; the helper returns the original + a "requires Apple Silicon" runtimeNote on those platforms. |
| FU-023 | SVDQuant / Nunchaku CUDA engine | **Foundation shipped 2026-05-05; awaiting live Windows / Linux CUDA validation.** | Apple Silicon dev box can't exercise the CUDA path live — wiring is in place so a Windows/Linux CUDA pull validates end-to-end. Backend: `_try_load_nunchaku_transformer` helper in [image_runtime.py](backend_service/image_runtime.py) loads via `NunchakuFluxTransformer2dModel` / `NunchakuQwenImageTransformer2DModel` / `NunchakuSD3Transformer2DModel` / `NunchakuSanaTransformer2DModel` / `NunchakuPixArtSigmaTransformer2DModel` — class registry at `_nunchaku_transformer_class_for_repo`. Preferred over NF4/int8wo on CUDA when `nunchakuRepo` pinned + nunchaku importable; falls back cleanly on Apple Silicon / CPU / missing package. Variant key extends with `nunchaku=...` so toggling rebuilds the pipeline. ImageGenerationConfig + ImageGenerationRequest fields: `nunchakuRepo`, `nunchakuFile`. Catalog rows: FLUX.1 Dev × svdq-int4-flux.1-dev, FLUX.1 Schnell × svdq-int4-flux.1-schnell. Setup install: `nunchaku>=1.2.1` via `_INSTALLABLE_PIP_PACKAGES`. Wan / HunyuanVideo / LTX wrappers don't exist in upstream Nunchaku v1.2.1 — adding a future video variant is a catalog-row change. |
| FU-024 | FP8 layerwise casting for non-FLUX DiTs | **Foundation shipped 2026-05-05; awaiting live CUDA SM 8.9+ validation.** | Apple Silicon can't exercise — Windows/Linux CUDA pull validates. Backend: `_maybe_enable_fp8_layerwise` helper in [image_runtime.py](backend_service/image_runtime.py) calls `transformer.enable_layerwise_casting(storage_dtype=…, compute_dtype=torch.bfloat16)` post-load. Family-correct fp8 dtype: E5M2 for HunyuanVideo (per upstream model card recommendation), E4M3 elsewhere (FLUX / Wan / Qwen-Image / SD3 / LTX). Compute capability gate refuses pre-Ada GPUs (SM <8.9) since hardware fp8 isn't there + the cast slows wall-time vs bf16. Helper degrades gracefully when `pipeline.transformer.enable_layerwise_casting` is missing (UNet pipelines / old diffusers) — runtimeNote surfaced into the load notes. Wired through both ImageGenerationConfig + VideoGenerationConfig + Request models + frontend hooks (`imageFp8LayerwiseCasting` / `videoFp8LayerwiseCasting`) + types. Default off; opt-in. |
| ~~FU-025~~ | ~~mlx-video Wan one-shot convert action~~ | **Fully shipped 2026-05-04 (Phase 7 + Phase 8 + Phase 9).** | Closes FU-009 Wan branch. **Phase 7 (foundation):** `[mlx-video]` extra in [pyproject.toml](pyproject.toml) flipped to ``git+https://github.com/Blaizzy/mlx-video.git``. Helper [backend_service/mlx_video_wan_convert.py](backend_service/mlx_video_wan_convert.py) wraps the upstream `python -m mlx_video.models.wan_2.convert` subprocess: `slug_for(repo)` / `output_dir_for(repo)` / `status_for(repo)` / `list_converted()` / `run_convert(checkpoint_dir, repo, dtype, quantize, bits, group_size, timeout)`. Output under ``~/.chaosengine/mlx-video-wan/<slug>/`` (override via ``CHAOSENGINE_MLX_VIDEO_WAN_DIR``). **Phase 8 (routing):** [mlx_video_runtime.py](backend_service/mlx_video_runtime.py) `supported_repos()` returns dynamic union of LTX-2 + converted-on-disk Wan repos. `_REPO_ENTRY_POINTS` adds `"Wan-AI/": "mlx_video.models.wan_2.generate"`. `_build_wan_cmd` produces the Wan-shaped CLI (`--model-dir`, `--guide-scale` string, `--scheduler`, optional `--seed`/`--steps`/`--negative-prompt`; no LTX-2 flags). `generate()` picks `_wan_runtime_note` (flags MoE experts) and skips LTX-2 effective-step / effective-guidance overrides. **Phase 9 (GUI):** Orchestrator [backend_service/mlx_video_wan_installer.py](backend_service/mlx_video_wan_installer.py) drives preflight → download-raw → convert → verify with structured progress events. Setup endpoints in [routes/setup.py](backend_service/routes/setup.py): `POST /api/setup/install-mlx-video-wan` (background-job pattern mirroring `/api/setup/install-longlive`), `GET /api/setup/install-mlx-video-wan/status`, `GET /api/setup/mlx-video-wan/inventory`. Frontend client in [src/api.ts](src/api.ts) (`startWanInstall`, `getWanInstallStatus`, `getWanInventory`). UI panel [src/components/WanInstallPanel.tsx](src/components/WanInstallPanel.tsx) lists every supported Wan repo with raw-size hint + converted badge / install button + live `InstallLogPanel` underneath; rendered in [VideoDiscoverTab.tsx](src/features/video/VideoDiscoverTab.tsx) above the variant grid. Supported raw repos: `Wan-AI/Wan2.{1-T2V-1.3B,1-T2V-14B,2-TI2V-5B,2-T2V-A14B,2-I2V-A14B}`. End-to-end UX: user clicks Install → backend downloads + converts in background → runtime auto-detects + routes Wan generate calls through mlx-video. Tests: 21 in [test_mlx_video_wan_convert.py](tests/test_mlx_video_wan_convert.py), 9 Wan-routing in [test_mlx_video.py](tests/test_mlx_video.py), 15 in [test_mlx_video_wan_installer.py](tests/test_mlx_video_wan_installer.py). |
| ~~FU-026~~ | ~~TaylorSeer + DBCache aggressive cache preset~~ | **Obsoleted 2026-05-03 by diffusers 0.38 core.** | Diffusers 0.38.0 (2026-05-01) ships ``TaylorSeerCacheConfig``, ``MagCacheConfig``, ``PyramidAttentionBroadcastConfig``, ``FasterCacheConfig`` natively — no ``cache-dit`` dependency required. Wired as registry strategies (ids ``taylorseer``, ``magcache``, ``pab``, ``fastercache``) in [cache_compression/__init__.py](cache_compression/__init__.py). Each adapter calls ``pipeline.transformer.enable_cache(<Config>)``. UNet pipelines (SD1.5/SDXL) raise ``NotImplementedError`` into a runtimeNote, matching the FBCache contract. MagCache is FLUX-only without calibration UX (uses ``FLUX_MAG_RATIOS`` from ``diffusers.hooks.mag_cache``); other DiTs raise a "calibration required" message until that UX lands. |
| FU-027 | NVIDIA/kvpress KV cache toolkit (CUDA-side) | **Setup install action pre-staged 2026-05-05; integration code pending.** | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) — Apache 2.0, 1.1k stars, `kvpress>=0.5.3` registered in `_INSTALLABLE_PIP_PACKAGES` so the Setup tab can pre-stage the wheel. Integration hooks land separately under `cache_compression/kvpress.py` once the helper picks an adapter shape (the upstream library exposes `presses` per technique — e.g. SnapKV / TOVA / KIVI / pyramid — and a `Pipeline` wrapper that takes a HF transformers model). Apple Silicon stays on TurboQuant-MLX; this is the CUDA-side complement. |
| FU-028 | MTP (Multi-Token Prediction) speculative decoding | **Deferred 2026-05-10 — upstream MTP-head loader gap on both runtimes.** | Target: lossless 1.5–2.2× speedup for trained-with-MTP models (Gemma-4 drafters released 2026-05-05, Apache 2.0; DeepSeek V3/R1; Qwen3.5/3.6/Next; Nemotron-3; MiMo-V2-Flash). **Blocker on Apple Silicon:** mlx-lm 0.31.3 ships ``stream_generate(..., draft_model=...)`` for *separately-trained* draft models but has no native MTP-head loader — Gemma-4-style MTP drafters share activations + KV cache with the target and cannot be loaded as a standalone ``mlx.nn.Module``. Confirmed by inspecting the installed `.venv/lib/python3.11/site-packages/mlx_lm/server.py` + `generate.py` — no MTP-specific code paths. **Blocker on llama.cpp:** PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (am17an, ``--spec-type mtp --spec-draft-n-max N``) is still in Draft as of 2026-05-10, awaiting at least 2 approving reviews. **Third-party path considered + rejected for v1:** [MTPLX](https://github.com/youssofal/MTPLX) (221 stars, MIT) wraps native MTP for Apple Silicon but ships as an OpenAI/Anthropic HTTP server — chaining HTTP servers from our FastAPI backend has unwanted latency + retry surface. **Re-evaluate when:** (a) mlx-lm gains a native MTP head loader (track ``ml-explore/mlx-lm`` releases), OR (b) llama.cpp PR #22673 merges, OR (c) MTPLX exposes a programmatic in-process Python API. The user-facing speedup is real (live benchmarks: M4 Pro × Qwen3.5-27B-4bit 15.3 → 23.3 tok/s) so this stays high-priority on the queue. |
| FU-029 | KVTC (NVIDIA ICLR 2026) KV cache strategy | **Deferred 2026-05-10 — CUDA-only upstream, awaiting MLX/Metal port + PyPI release.** | Targeting [OnlyTerp/kvtc](https://github.com/OnlyTerp/kvtc) (Apache 2.0). PCA + adaptive quantization + entropy coding — 8–32× compression vs the dropped ChaosEngine's 3.7×, peer-reviewed at ICLR 2026, beats TurboQuant by 37% at comparable quality on long-context. Upstream blockers: (a) CUDA-only — repo's roadmap mentions MLX/Metal as "planned" but not yet implemented, so the Apple Silicon dev box cannot validate end-to-end; (b) not on PyPI — distributed as a `src.*` repo intended for `git clone`; (c) integration shape is a HuggingFace `DynamicCache` wrapper (not a llama.cpp cache type), so the existing GGUF lane has no path. Re-evaluate when either upstream ships MLX support or a Windows/Linux+CUDA development box becomes available. Apple Silicon users continue on TurboQuant-MLX (also ICLR 2026, native today). |
| ~~FU-030~~ | ~~Drop ChaosEngine + RotorQuant strategy slots~~ | **Shipped 2026-05-10.** | ChaosEngine (cryptopoly/ChaosEngine — 1 commit upstream, eclipsed by KVTC at ICLR 2026 with the same PCA approach but 8–32× compression vs 3.7×) and RotorQuant (shipped as a misleading alias for TurboQuant — same ``--cache-type-k turbo{N}`` flags + same Python module marker) both removed from the registry. Persisted user configs that still reference these ids coerce silently to ``turboquant`` via a new ``CacheStrategyRegistry.resolve_legacy_id`` helper + module-level ``_LEGACY_STRATEGY_ALIASES`` map ([cache_compression/__init__.py](cache_compression/__init__.py)). Mirror coercion in frontend ([src/components/runtimeSupport.ts](src/components/runtimeSupport.ts) ``LEGACY_STRATEGY_ALIASES`` + ``canonicalStrategyId``). Two-level llama.cpp fallback chain (was three-level: requested → ChaosEngine → native; now requested → native) in [backend_service/inference/llama_cpp_engine.py](backend_service/inference/llama_cpp_engine.py). Vendored ChaosEngine bundling stripped from [scripts/stage-runtime.mjs](scripts/stage-runtime.mjs) (3 helper functions removed: ``stageVendoredChaosEngine`` + ``ensureSetuptoolsForPep639`` + ``resolveChaosEngineVendor``). Pre-build probe asserts the legacy-id coercion works in CI. ``[rotorquant]`` extra removed from [pyproject.toml](pyproject.toml). ``CHAOSENGINE_VENDOR_PATH`` env var dropped. Cache strategy speed/quality maps in [helpers/cache.py](backend_service/helpers/cache.py) trimmed to remaining strategies. |
| ~~FU-031~~ | ~~Extend `DRAFT_MODEL_MAP` for new z-lab DFlash drafters + pin TriAttention~~ | **Shipped 2026-05-10.** | z-lab published draft checkpoints for several new families since the last `DRAFT_MODEL_MAP` audit; the upstream `dflash-mlx` 0.1.5 release also added the Gemma4 backend (commit 05cc456). Added entries for `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.5-122B-A10B`, `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.7`, `moonshotai/Kimi-K2.6` (all in [dflash/__init__.py](dflash/__init__.py)) plus `mlx-community/...` aliases for each so Apple Silicon quants resolve. New 7 unit tests in [tests/test_dflash.py](tests/test_dflash.py) pin the mappings. **Same commit also pinned TriAttention** to `c3744ee6a50522a1559a577f85aef2b165a344f2` in [pyproject.toml](pyproject.toml) — previously the `[triattention]` and `[triattention-mlx]` extras pulled `git+...git` HEAD, which made fresh installs non-reproducible whenever the upstream landed unreleased work. Pin matches the v0.2.0 release surface plus the AMD GPU port. |
| FU-032 | TurboQuant+ ([TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)) Apple Silicon Metal kernels (**watch-closely**) | Re-evaluate when upstream tags v1.0 release or beats `turboquant-mlx-full` 0.3.0 on a public M-series benchmark | Same author as our `llama-cpp-turboquant` fork. Adds Walsh-Hadamard rotation (improvement over base TurboQuant's Hadamard-only path) + a sparse-V optimization on M5 Max that achieves 0.93x of q8_0 decode speed at long context while saving 50–64% of KV memory. Reported numbers: turbo3 4.6× compression at +1.06% PPL, turbo4 3.8× compression at +0.23% PPL — comparable to our existing `turboquant-mlx-full` pin but with newer kernels. 326 commits + community tested across M1/M2/M3/M5. **Not on PyPI** (development install via `git clone` + `pip install -e .[dev]`), so adopting it means a vendored or git+url install pattern like dflash-mlx — re-evaluate when upstream publishes a wheel or tags a v1.0. Apple Silicon stays on `turboquant-mlx-full` for now; the underlying llama-server-turbo binary already exposes turbo2/3/4 cache types. |
| ~~FU-033~~ | ~~dflash-mlx pin sync assert in pre-build-check~~ | **Shipped 2026-05-10.** | Caught a real bug: [pyproject.toml](pyproject.toml) and [scripts/stage-runtime.mjs](scripts/stage-runtime.mjs) had drifted to different `dflash-mlx` commit hashes (the dev `.venv` ran 0.1.5.1 while `npm run stage:runtime` was bundling 0.1.4.1 into release builds). Both files manually synced to `fada1eb`; new probe in [scripts/pre-build-check.mjs](scripts/pre-build-check.mjs) and [scripts/pre-build-check.sh](scripts/pre-build-check.sh) regex-extracts the commit hash from both files and fails the build when they diverge. Same probe also took the chance to drop the orphan `vendor/ChaosEngine` staleness check from both runners — that vendored path was dropped in FU-030 and would never resolve again. |
| ~~FU-039~~ | ~~Tool-call `arguments: null` bricks Chat tab forever~~ | **Shipped 2026-05-10.** | Caught by the FU-037 ErrorBoundary: Coder-Next + Tools + `What is 17 * 23 plus sqrt(144)?` triggered `TypeError: Object.entries requires that input parameter not be null or undefined` in `ToolCallCard` (minified `_Y`). Root cause traced through the boundary's component stack (`_Y` → Panel `<section>` → ErrorBoundary → workspace) and the minified source: `src/components/ToolCallCard.tsx:116` did `Object.entries(toolCall.arguments)`, but Coder-Next emits `{"arguments": null}` for tool calls that need no parameters. `backend_service/agent.py::_execute_tool_call` then evaluated `isinstance(None, str) → False` and set `arguments = None`, which serialised into the persisted session. Every subsequent render of that turn crashed the Chat tab — the user could not even read prior history because the boundary fires before any other content renders. Two-layer fix: (1) backend `_execute_tool_call` now coerces `None` / empty-string / non-dict shapes to `{}` at the source so the contract "`arguments` is always a dict" holds for all consumers; (2) frontend `ToolCallCard` adds a defensive guard that defaults to `{}` and renders `(no arguments)` for genuinely corrupt records (so old sessions stop crashing without a manual localStorage wipe). 4 new unit tests in `tests/test_agent.py` pin all four null-ish input shapes. |
| ~~FU-038~~ | ~~Diagnostics cleanup: `_free_bytes` import, MallocStackLogging spam, Qwen3.6-27B alias~~ | **Shipped 2026-05-10.** | Three bugs surfaced by the live ``/api/diagnostics/snapshot`` payload from a Coder-Next + Tools repro. (1) ``backend_service/routes/diagnostics.py`` imported ``_free_bytes`` from ``backend_service.routes.setup``, but the setup package's ``__init__.py`` did not re-export it from ``gpu_bundle.py`` — the snapshot's ``extras`` section reported ``ImportError: cannot import name '_free_bytes'``. Added the re-export. (2) macOS hardened-runtime spawned every Python subprocess with three lines of ``MallocStackLogging: can't turn off malloc stack logging because it was not enabled.`` spam (we ship ``bundle.macOS.hardenedRuntime: true``). Hundreds per minute under the metrics poll, drowning out real INFO/ERROR lines. Fixed at source by ``command.env_remove("MallocStackLogging" / "MallocStackLoggingNoCompact" / "MallocScribble")`` in ``src-tauri/src/backend.rs`` so new builds don't produce the spam. Also added a regex filter (``_LOG_NOISE_PATTERNS`` + ``_filter_log_noise``) in ``diagnostics.py`` so the ``/api/diagnostics/log-tail`` and snapshot endpoints strip the spam from logs produced by older builds too — existing installs see a clean diagnostic surface without rebuilding. Filter reads 4× the requested line window so 200 useful lines survive even when the raw log is 50% spam. (3) Qwen3-Coder-Next was rebranded ``Qwen3.6-27B`` upstream; lmstudio-community MLX conversion's HF metadata reports ``mlx-community/Qwen3.6-27B-4bit`` as the canonical repo. ``model_resolution.resolve_dflash_target_ref`` prefers canonical, so ``DRAFT_MODEL_MAP`` missed and the runtimeNote said *DFLASH unavailable for 'mlx-community/Qwen3.6-27B-4bit': no compatible draft model is registered.* Aliased the three quant variants (4bit / bf16 / 8bit) back to ``Qwen/Qwen3-Coder-Next`` so the existing ``z-lab/Qwen3-Coder-Next-DFlash`` drafter resolves. New unit test pins the mapping. |
| ~~FU-037~~ | ~~Per-tab ErrorBoundary + Tauri devtools in release builds~~ | **Shipped 2026-05-10.** | A tool-call in the Chat tab against `Qwen3-Coder-Next` blanked the entire packaged macOS app — webview reload returned the user to the Dashboard, and any subsequent Chat navigation crashed again. Root cause: the React tree had no error boundary, so a single uncaught render error in one tab tore down the whole `<main>` content frame. Release builds also did not ship the WebKit inspector, so the user could not pull a stack trace without rebuilding via `cargo tauri dev`. (1) New [src/components/ErrorBoundary.tsx](src/components/ErrorBoundary.tsx) — `getDerivedStateFromError` + `componentDidCatch` capture the error, render an inline fallback with the error message, JS stack, component stack, "Try again" reset, and "Copy details" clipboard button. Wrapped around `{content}` in [src/App.tsx](src/App.tsx) keyed by `activeTab` so switching tabs is its own recovery path. (2) `src-tauri/Cargo.toml` `tauri` dep gains the `devtools` Cargo feature so right-click → Inspect Element opens WebKit devtools in release builds. (3) CSS for `.error-boundary` lives next to the existing notice banners in [src/styles.css](src/styles.css) — same colour vocabulary. Unit tests in [src/components/__tests__/ErrorBoundary.test.ts](src/components/__tests__/ErrorBoundary.test.ts) pin the static-derive contract so the boundary cannot silently stop catching errors. Frontend errors land in the webview console; backend errors land in the Diagnostics tab + the in-memory `app.state.chaosengine` log buffer. |
| ~~FU-034~~ | ~~Hide unrecoverable launch-modal options instead of greying them out~~ | **Shipped 2026-05-10.** | The launch settings panel ([src/components/RuntimeControls.tsx](src/components/RuntimeControls.tsx)) used to render every cache-strategy card and the DFlash speculative-decoding toggle for every model + engine combo, with disabled checkboxes + "N/A" badges when an option could not run. That taught users the wrong thing — a disabled card with no install button suggests something they could fix, when the only fix lived outside the app or did not exist at all. New rule: **hide options the user has no in-app path to recover.** (1) Cache-strategy cards now skip render when the strategy is engine-incompatible (e.g. TriAttention selected on the MLX engine — engine mismatch is fundamental, no install button helps) or when the strategy needs the turbo binary on a GGUF backend without `llama-server-turbo` present (only fix is `scripts/build-llama-turbo.sh` outside the app). (2) The DFlash toggle hides entirely when the selected model has no draft in [`DRAFT_MODEL_MAP`](dflash/__init__.py) or the engine is GGUF (DFlash needs MLX/vLLM). The "DFlash package not installed but model would be supported" case stays visible — the install button gets the user to ready in one click. ``native`` always survives. Hardcoded `f825ffb` install hint string in the DFlash help panel was the same drift bug from FU-033 — fixed alongside (now `fada1eb`). The popover-side filter ([src/components/kvStrategyFilter.ts](src/components/kvStrategyFilter.ts)) already followed this rule, so the modal now matches. |

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
| Cross-strategy E2E matrix runner | `test_cache_strategy_matrix_runner.py` | `pytest tests/test_cache_strategy_matrix_runner.py -v` |
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
