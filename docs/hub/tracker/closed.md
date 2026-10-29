---
title: Closed follow-ups
order: 3
summary: 77 shipped, obsoleted or dropped rows, newest first.
---

# Closed follow-ups

Generated from the Follow-Ups Tracker in `CLAUDE.md` by `tools/hub_tracker.py`; edit the row there, not here. Shipped, obsoleted or dropped rows (struck through, or marked shipped with no open points left). Kept as the history of why things are the way they are.


<details>
<summary>FU-125 · Better video model than Wan 2.2 TI2V 5B for a 64 GB Mac (research; nothing removed) — Shipped 2026-10-29 as option 1 (FU-128)</summary>

**Status or trigger:** **Shipped 2026-10-29 as option 1 (FU-128); the Wan 2.2 TI2V 5B family is removed.**

**Notes:** **Why the Wan output is poor:** PyTorch MPS corrupts bf16 output for Wan 2.2 and LTX (several open upstream PyTorch-MPS issue reports describe the same silent corruption we bisected in FU-102), so the diffusers-on-MPS path is a known-bad class and the blocky / pink-strip artifacts are probably not only the model. **Options measured or reported:** (1) **LTX-2.3 / 2.5 through `dgrauet/ltx-2-mlx`** (MIT, pure MLX, text / image / audio to video with stereo audio, distilled and two-stage pipelines, int4 / int8 / bf16 packs `dgrauet/ltx-2.3-mlx-{q4,q8}` (59.7 GB for q4 with the shared text encoder) and `dgrauet/ltx-2.5-mlx-q4` (50.7 GB, gated); a community report of 2026-10-24 gives 121-210 s for a 5 s clip on M4 / M5 Max, unverified here; our mlx-video LTX-2.3 path measured 1130 s for 33 frames at 768x512, 8 steps (FU-110), so this runtime is the faster Mac route; LTX-2 community licence, free under 10 M USD revenue). (2) **Wan 2.2 5B through mlx-video** (the FU-025 convert path; avoids MPS; unmeasured). (3) MiniMax-H3 (best quality reported, 498 GB repo, territory exclusions, not practical on 64 GB). (4) HunyuanVideo 1.5 (diffusers `HunyuanVideo15Pipeline` exists in 0.42, Tencent community licence, GPU-oriented, no Mac data), LongCat-Video (MIT, `mlx-community/LongCat-Video-q4`, needs its own MLX package, unmeasured). **Not done:** nothing downloaded and the Wan 2.2 family stays until a replacement runs here (disk free was 25 GB, the link ran at about 4 MB/s, so the LTX-2.3 q4 pack alone is a 4-5 hour download and needs about 35 GB freed first, e.g. the cached Wan 2.2 5B (24 GB) and the incomplete `prince-canuma/LTX-2.3-distilled` (49 GB, no text encoder)). Removing the family also touches E2E phase 5, `_wan_not_burnt`, the FU-102 scheduler swap, the mlx-video Wan convert panel and the defaults table.

</details>

<details>
<summary>FU-114 · GGUF and MLX chat paths: reasoning, vision, catalog launches, stream drain — Shipped 2026-10-21.</summary>

**Status or trigger:** **Shipped 2026-10-21.**

**Notes:** Found running the new catalog rows live. (1) llama-server parses `<think>` into `reasoning_content`, which the stream ignored: Muse Glimmer (reasons even with `--reasoning off`) showed an empty reply and 0 tokens. (2) A catalog launch without a path reached llama-server as `--hf-repo`, so the pinned quant was skipped and drafter sidecars were never found; a GGUF in the library now loads from its folder. Muse Glimmer Q4_K_M with its `dflash-` sidecar: **29.1 tok/s against 19.5 plain** on a coding prompt, mmproj wired. (3) The loaded model now reports `speculativeDecoding` for GGUF spec lanes (the speed history skips speculative runs). (4) Templates that end the prompt with `<think>` (LFM2.5 always) leaked the plan into the answer on MLX; the filter starts inside the block. (5) MLX models loaded through mlx-vlm now report `visionEnabled`, so images reach Gemma 4 on Apple Silicon. A text-only mlx-lm load of a natively multimodal model (Qwen3.5 / 3.6 / 3.8) still drops images, with a log warning. (6) An abandoned MLX stream (runaway guard, dropped client) left the worker's remaining lines in the queue, and the next request read them ("invalid result payload"); the stream now drains to `done`. (7) A cancelled quant's partial blobs, a library scan taken mid-download, and HF orgs ending in "models" (`meta-models/...`) each made a downloaded model refuse to load. (8) espeak-ng drops a data path of about 160+ characters and then exits the process, so one TTS request could take the backend down; a long path is swapped for a copy under `~/.chaosengine/espeak-ng-data`. Also checked live: Kolibri 1 GGUF Q3_K_M on llama.cpp b11786 (correct German, 25 tok/s on a short reply), Whisper and Parakeet transcribing a Kokoro clip exactly (first live Parakeet run, FU-092).

</details>

<details>
<summary>FU-113 · MLX stack moved to mlx 0.32 for releases — Shipped 2026-10-21 (release constraints)</summary>

**Status or trigger:** **Shipped 2026-10-21 (release constraints); the dev `.venv` still runs 0.31.**

**Notes:** `release-constraints.txt` now pins mlx 0.32.4, mlx-lm 0.32.2, transformers 5.20.2 (mlx-vlm 0.7.9 and diffusers 0.41.0 for the Setup overlay). Checked in a separate venv with this branch: full pytest passes on both stacks; live on the M4 Max: LFM2.5 and Kolibri 1 chat, DFlash, Gemma 4 E2B image input (mlx-vlm 0.4.4 could not load that conversion at all), FLUX.2 klein 4B (diffusers 0.41), LTX-2.3 (mlx-video) and Wan 2.2 5B (diffusers) video, Kokoro TTS (mlx-audio 0.6.1 + misaki), Whisper and Parakeet STT, TurboQuant (turboquant-mlx-full 0.29), and the quick cache-strategy matrix. Release installs resolve for Python 3.12 on macOS arm64, Linux and Windows (pip dry runs). Fixes the upgrade needed: Kolibri's vendored file needs `trust_remote_code` on mlx-lm 0.32; `ArraysCache.state` became a tuple (hybrid prompt-cache checkpoints); the mlx-video companion pin `mlx<0.32` would have failed every install; mlx-vlm 0.7 installs mlx-audio by itself, so TTS now also requires misaki; a streamed reply cut short left lines in the worker queue (FU-114). **Upgraders:** the extras overlay sits ahead of the bundle and older Setup installs left mlx 0.31 in it (this Mac's `cp312` overlay has mlx 0.31.2, mlx-lm 0.31.3, mlx-vlm 0.4.4); packaged builds now move overlay copies of mlx / mlx-metal / mlx-lm that differ from the bundle to `extras/<tag>/stale-core/` at startup ([runtime_paths.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/runtime_paths.py) `guard_overlay_against_bundle`). Overlay mlx-vlm 0.4.4 then runs on mlx 0.32 untested; reinstalling from Setup resolves 0.7.9. **To do after merge:** upgrade the dev `.venv` to the same versions (`.venv/bin/pip install -c release-constraints.txt mlx mlx-lm mlx-vlm transformers diffusers`). **Unlocked, not done:** mlx-vlm 0.7.9 ships `qwen4_exp`, `glm5_next`, `deepseek_v4`, `muse_glimmer` and `bonsai` — routing those `model_type`s to mlx-vlm in the worker would let stock MLX load Qwen3.8 Flash Next, GLM-5.3-Flash and DeepSeek-V4 Flash without TensorFold (FU-088); diffusers 0.41 has `QwenImage21Pipeline`, the Qwen-Image-2.1 diffusers lane for CUDA (FU-096).

</details>

<details>
<summary>FU-109 · Disk usage tab could not look inside a folder or delete from it — Shipped 2026-10-20.</summary>

**Status or trigger:** **Shipped 2026-10-20.**

**Notes:** A model directory such as `AI_Models` showed as one row (`AI_Models/models` = 150 GB of 3D tool trees) with no way to see what filled it or to remove anything from the screen. [diagnostics.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/diagnostics.py) gains `GET /api/diagnostics/storage-children?path=` (direct children of one folder, files and folders, biggest first; 403 for a path outside the enabled model directories) and the Top 20 rows now carry `expandable` / `deletable`. [DiagnosticsPanel.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/settings/DiagnosticsPanel.tsx) `StorageRow` expands lazily and nests, and its Delete button confirms, then calls the existing `POST /api/models/delete` (inside-a-model-directory check, never a model directory itself, unloads a loaded model first). Anything inside the shared Hugging Face `blobs` store is `deletable: false` (many repos link into it). 4 tests in [test_diagnostics_routes.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_diagnostics_routes.py); checked in the dev UI by expanding a folder and deleting a throwaway one. **Open:** the scan counts the same bytes in `blobs` and in the repos that link into it, so the total over-counts.

</details>

<details>
<summary>FU-108 · Chat &gt; My Models listed 3D parts, encoders, voice models and bare weight files as chat models — Shipped 2026-10-20.</summary>

**Status or trigger:** **Shipped 2026-10-20.**

**Notes:** Discovery called everything that was not image / video / draft "text". On the dev Mac that put 35 of 45 entries in the chat list: Trellis 2 / TripoSG / Hunyuan3D tool trees (GGUF stages, `.safetensors` decoders and encoders), DINO, `flux_text_encoders`, Kokoro, and several copies of each from nested folders. [model_classifier.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/model_classifier.py) `_looks_like_non_chat_model` now marks them `modelType: "other"` from structural evidence: a bare `.safetensors` file; a GGUF whose `general.architecture` (read from the header) is not a language model (the Trellis GGUFs report `flux`); a `config.json` whose architecture does not end in CausalLM / ConditionalGeneration / LMHeadModel / ChatModel, or that has no architecture or model type, unless the folder ships a chat template; then path and name keywords for entries with no readable config. The Chat tab filter ([library.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/utils/library.ts)) hides `other`, and `LIBRARY_CACHE_VERSION` is 4 so existing installs re-scan. On the real directories the chat list drops from 45 to the 11 real chat models. 7 tests in [test_non_chat_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_non_chat_models.py), a vitest case and an e2e check. **Open:** an architecture name that ends differently and ships no chat template would be hidden; the keyword lists are not exhaustive (the structural rules are the main defence).

</details>

<details>
<summary>FU-107 · Release bundles shipped without llama-server on every platform — Shipped 2026-10-20.</summary>

**Status or trigger:** **Shipped 2026-10-20.**

**Notes:** [stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) fetched `repos/ggml-org/llama.cpp/releases/latest` for the prebuilt binaries. Upstream's "latest" had become a `v0.7.0` entry holding only `nightly-tag.txt`, so the v0.9.8 build found no asset, warned "Could not auto-download llama.cpp prebuilt binaries" and shipped `llamaServer: null` on macOS, Linux and Windows (v0.9.4 bundled it; the Linux and Windows installers shrank 28-39 MB). GGUF models need the Setup page's llama-server install in such a build. The build now fetches the pinned tag `LLAMA_CPP_TAG` (`b11277`, kept equal to `LLAMA_CPP_REF` in `update-llama-cpp.sh` by a test; override `CHAOSENGINE_LLAMA_CPP_TAG`), and the Linux asset patterns accept the `.tar.gz` upstream now ships. Checked against the real b11277 assets: all four platforms resolve, and the macOS tarball holds `llama-server` / `llama-cli` with `--spec-type`. **Open:** `llama-server-turbo` and `sd` are still not bundled; confirm `llamaServer` is non-null in the next release's `runtime-*.manifest.json`.

</details>

<details>
<summary>FU-106 · Release tag could differ from the app version, so an up-to-date install was offered the same release — Shipped 2026-10-05.</summary>

**Status or trigger:** **Shipped 2026-10-05.**

**Notes:** The updater compares the running app's version (`tauri.conf.json`) with `latest.json`, whose version comes from the release tag. v0.9.4 was tagged 0.9.4 but built as 0.9.3, so every current install kept offering v0.9.4, and the Linux and Windows `latest.json` entries (named from the tag) were dropped. [check-version-sync.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/check-version-sync.mjs) now checks all six places the version lives (package.json, package-lock.json, pyproject.toml, Cargo.toml, Cargo.lock, tauri.conf.json) and, with `--tag`, the tag; [release.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/release.yml) runs it first, before a release is created or anything is built, and both pre-build gates run it. 4 tests in [test_check_version_sync.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_check_version_sync.py). **Open:** the shipped v0.9.4 still reports 0.9.3, so installs on it will keep seeing v0.9.4 until they take a later release.

</details>

<details>
<summary>FU-105 · Tool use was broken on MLX and on the non-streaming chat route — Shipped 2026-10-05.</summary>

**Status or trigger:** **Shipped 2026-10-05.**

**Notes:** Found running Kolibri 1 with the calculator tool. (1) The agent loop returns tool results as `tool` messages, but [mlx_worker_prompt.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_prompt.py) `_build_prompt_text` dropped every role except system / user / assistant, so on MLX the model never saw a result and repeated the same call until the 10-iteration cap (Kolibri did 10 identical calls, then the loop stopped with no answer). Tool turns now render as a user turn wrapping the result in `<tool_response>`, which is what Hermes-style chat templates do; the same run now answers after 3 iterations. (2) `POST /api/chat/generate` with `enableTools` answered 500 (`'_AgentResultProxy' object has no attribute 'to_metrics'`) because [generation.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/state/generation.py) built a stand-in object instead of a `GenerationResult`; fixed, with a route test that fails without the fix. The composer uses the streaming route, whose loop shares problem (1). **Open:** the model still makes one redundant duplicate call before answering; other model families were not re-run live.

</details>

<details>
<summary>FU-104 · Release bundle shipped none of the Setup installer scripts — Shipped 2026-10-05.</summary>

**Status or trigger:** **Shipped 2026-10-05.**

**Notes:** The Install buttons for MTPLX, TensorFold and mflux run `bash <backend>/scripts/install-*.sh`, resolved next to `backend_service` (`Path(__file__).parents[3]`), but [stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) copied only the Python packages: the `backend/` directory of the v0.9.3 bundle holds `backend_service`, `cache_compression`, `turboquant_mlx`, `README.md` and `pyproject.toml` and nothing else. From a source checkout the scripts resolve, so the gap shows only in a packaged app, where each of those installs would have ended with "No such file". `BUNDLED_INSTALL_SCRIPTS` now stages the three scripts into `backend/scripts/`, and [test_stage_runtime_scripts.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_stage_runtime_scripts.py) fails when a route runs a script the list lacks. stable-diffusion.cpp (MIT) was also missing from THIRD_PARTY_NOTICES although `sd` is staged into the bundle; added. **Open:** no packaged build was made here (`npm run release:macos`), so check that `backend/scripts/` is in the archive and run one install from the packaged app; the installers still pick a system Python 3.11-3.13 and do not use the app's embedded one, so a Mac without one gets the script's own preflight message.

</details>

<details>
<summary>FU-102 · Wan 2.2 TI2V (and every Wan row) rendered burnt, over-saturated video on Apple Silicon — Shipped 2026-10-05.</summary>

**Status or trigger:** **Shipped 2026-10-05.**

**Notes:** Root cause found by bisecting one real run (640×352, 9 frames, seed 42): VAE decode is identical on CPU fp32, MPS fp32 and MPS bf16, and an fp32 transformer changes nothing, but the *latents* leave the denoising loop several times too large (std 3.2–3.5, range ±15–19, against 0.9 and ±4.5 for a healthy run) with UniPC and even with guidance off. A model-free run of diffusers' `UniPCMultistepScheduler` on the same noise follows the CPU trajectory for one step, then drifts with the error roughly doubling per step (CPU matches the analytic ODE solution; MPS collapses to a constant on the last step). The three ops I patched to run on CPU (`linalg.solve`, `expm1`, `log`) and `einsum` are not the cause; the exact op is unknown. Fixed by [pipeline_helpers.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime/pipeline_helpers.py) `swap_scheduler`: on MPS a flow-prediction UniPC scheduler (all Wan repos) becomes `FlowMatchEulerDiscreteScheduler` with the repo's `flow_shift`; other devices and non-flow configs keep UniPC. **Measured:** clipped pixels 57-66 % → 0.0 %; the app's default run (20 steps, CFG 7.5) now gives a clean golden retriever on a beach in 65 s. Stepping UniPC on CPU also works (1.6 % clipped) but left artefacts at 20 steps; guidance 5.0 (the official value) is not needed. 4 tests in `UniPCOnMpsTests`. **Open:** report upstream (diffusers / PyTorch MPS) with the model-free repro; the image runtime's UniPC-based samplers were not checked on MPS.

</details>

<details>
<summary>FU-101 · GitHub #55 — local GGUF folders not recognised — Shipped 2026-10-05 (the reporter never confirmed their layout).</summary>

**Status or trigger:** **Shipped 2026-10-05 (the reporter never confirmed their layout).**

**Notes:** [discovery.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/discovery.py): a ModelScope cache folder was listed but as `Qwen3___5-9B-…` (ModelScope writes dots as `___`), which hid the model family from every name-based check — `_directory_model_name` restores the dots; a symlinked model folder was invisible because the walk never followed links — it now does, walking each real directory once. 7 tests in [test_discovery_layouts.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_discovery_layouts.py). **Not done:** a ModelScope downloader or a default ModelScope cache directory (a product call; the reporter asked for the integration). The reporter's folder must still be added under Settings → Model directories.

</details>

<details>
<summary>FU-100 · GitHub #93 — Linux AppImage stuck on "Still loading" — Shipped 2026-10-05</summary>

**Status or trigger:** **Shipped 2026-10-05; the cause is derived from kernel semantics and the code, not reproduced on Linux (no Linux box here).**

**Notes:** Two bugs that together match the report. (1) [lib.rs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/lib.rs) ran `bootstrap` on a short-lived thread, and on Linux the sidecar is forked with `PR_SET_PDEATHSIG` — which fires when the *thread* that forked the child exits (prctl(2)), not the process. Every Linux build since the death signal landed (2026-04-21) therefore had its backend SIGKILLed the moment bootstrap returned. `spawn_sidecar_thread` now parks that thread on Linux. (2) [useWorkspace.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/hooks/useWorkspace.ts) polled the shell for `startupError` but through a cached promise (`getTauriBackendInfo()` without `force`), so the splash never saw a crash, a spawn failure or a port fallback and spun forever. It now re-reads each attempt, and `resolveApiBase` follows the sidecar when it moves to another port. **Shell:** [backend.rs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/backend.rs) stops waiting when the sidecar has exited (the old wait polled a dead process for 12 s), and a slow start is a soft note that `runtime_info` clears once the port answers (`startup_note_soft`). **Splash:** [StartupProgressPanel.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/StartupProgressPanel.tsx) shows a failure only when the process is gone, shows a running sidecar's note under the spinner, and prints the backend log path. 12 Rust tests (`cargo test --manifest-path src-tauri/Cargo.toml`; the Linux-only assertion in `sidecar_thread_tests` runs on Linux) and 7 vitest cases. **Open:** reply to the reporter once a build ships; if the AppImage still hangs, ask for `/tmp/chaosengine-backend-8876.log` (now shown in the splash), `mount | grep /tmp` and `getenforce`; an AppImage-injected `LD_LIBRARY_PATH` reaching the bundled Python is a second untested suspect.

</details>

<details>
<summary>FU-099 · Catalog pins pointed at Hub repos and files that do not exist; LoRAs were never downloaded — Shipped 2026-10-05.</summary>

**Status or trigger:** **Shipped 2026-10-05.**

**Notes:** [scripts/audit-catalog-repos.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/audit-catalog-repos.py) (new; run it by hand, needs network) asks the Hub for every repo and pinned file in the three catalogs. First run found 10 of the 25 GGUF-pinned image and video rows dead, so their Download button 404ed: FLUX.1 schnell / dev "Q4_K_M" (city96 publishes Q4_K_S; the diffusers GGUF rows are now Q4_K_S, the sd.cpp rows use sd.cpp's own `leejet/FLUX.1-{schnell,dev}-gguf` `q4_k`), Wan 2.1 1.3B GGUF (`city96/Wan2.1-T2V-1.3B-gguf` never existed; now `samuelchristlie/Wan2.1-T2V-1.3B-GGUF`), Wan 2.1 14B GGUF (file names are lower-case `14b`; sizes corrected to 10.1 / 12.5 / 15.9 GB + base), the two CausVid LoRA rows (`lightx2v/*-CausVid-LoRA` do not exist; now `Kijai/WanVideo_comfy`, the repo the diffusers docs load; CC BY-NC) and CogVideoX 1.5 (id `THUDM/CogVideoX-1.5-5b` is not a repo; now `THUDM/CogVideoX1.5-5B`, in [repos.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime/repos.py) too). The audit now passes: 147 repos, 200 references. **LoRA lane:** both runtimes loaded catalog LoRAs with `local_files_only=True` and nothing downloaded them, so the Hyper-SD, Turbo-Alpha and CausVid variants quietly ran without their LoRA (only a load note) — [helpers/lora.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/lora.py) `fuse_catalog_lora` now fetches the one file first. **sd.cpp FLUX rows** (FU-096 e): variant repo = the GGUF repo (the Download button fetches the GGUF only, like Qwen-Image-2.1), companion files in `sdcppAux` (VAE + T5-Q8 from `second-state/FLUX.1-schnell-GGUF`, CLIP-L from `comfyanonymous/flux_text_encoders`; all ungated, first-use download), guidance goes to `--guidance` with `--cfg-scale 1` ([sdcpp_image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_image_runtime.py)), a failed run raises instead of falling back to a 24 GB diffusers FLUX. **sd.cpp progress:** image runs reported nothing to the UI — no run was ever begun and this build's `| 2/4 - 21.6s/it` bar did not match the step regex (the video lane had the same blind spot). `helpers/sdcpp_progress.parse_step` now reads that format (and ignores model-loading bars), the image lane begins / finishes the run with loading → diffusing → decoding phases, and a companion file that is not cached yet shows "Downloading … (first run only)". **Measured (M4 Max):** schnell Q4_K 4 steps = 114 s at 1024² and 28 s at 512² through the app, dev Q4_K 8 steps = 58 s at 512²; coherent images. CausVid key conversion checked with diffusers 0.40 (918 / 1218 tensors, all `transformer.*`). **Open:** a CausVid fuse on a real Wan pipeline and CUDA runs of the sd.cpp FLUX rows are untested; T5 Q8 vs fp16 quality is unmeasured; sd.cpp is far slower than mflux for FLUX on a Mac (23-27 s/step), so it is the Windows / Linux lane.

</details>

<details>
<summary>FU-098 · Setup installs of MLX-stack packages polluted the extras overlay — Shipped 2026-10-02.</summary>

**Status or trigger:** **Shipped 2026-10-02.**

**Notes:** The `cp311` extras overlay on the dev box held mlx 0.31.2 and 0.32.3, five `huggingface_hub`, three `transformers`, plus `mlx-vlm` 0.7.4 and `mlx-audio` 0.5.7 — put first on `PYTHONPATH` for every worker, it made video generation die importing `mlx_vlm` and image generation fall back to the placeholder engine. Source: the Setup install of `mlx-video` (git) has an unbounded `mlx-vlm` dependency; `pip --upgrade --target` resolves the newest (0.7.4), which needs mlx >=0.32.2 and transformers >=5.14 (0.6.5 and later already need mlx >=0.32). `_COMPANION_PINS` in [routes/setup/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/__init__.py) now adds `mlx>=0.31.2,<0.32` and `mlx-vlm>=0.6.3,<0.6.5` to that install; `mflux` (which needs mlx >=0.32) moved to its own venv (FU-096). **2026-10-05:** every overlay install now carries a constraint file (`mlx>=X.Y,<X.Y+1`, `mlx-metal` likewise; X.Y is read from the mlx the app itself runs, never from the overlay) — `_write_mlx_constraint` in [_install_helpers.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/_install_helpers.py). Checked with `pip install --dry-run --target` of `mlx-audio`: mlx 0.32.3 + mlx-metal 0.32.3 without it, 0.31.2 with it. **Still open:** mlx-audio 0.5.x needs transformers >=5.14 (the app runs 5.5.3), so a voice install still puts a newer transformers in the overlay ahead of the app's, and mlx-vlm 0.6.4's `<5.13` pin conflicts with it — untested at runtime; and a repair action that moves a corrupt overlay aside (done by hand on the dev box: `extras/cp311` -> `cp311.broken-20261001`).

</details>

<details>
<summary>FU-097 · Pattern-filtered model downloads failed to open offline (huggingface_hub 1.x) — Shipped 2026-10-01.</summary>

**Status or trigger:** **Shipped 2026-10-01.**

**Notes:** `huggingface_hub` 1.x raises `IncompleteSnapshotError` from `snapshot_download(local_files_only=True)` when the cached file listing names files the download allow-list skipped on purpose (duplicate single-file checkpoints, sample images). Every curated image model fetched through the app then failed to load and fell back to the **placeholder SVG engine** — found when FLUX.2 klein 4B (5 files skipped) came back as a placeholder. New [hf_local.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/hf_local.py) `local_snapshot_path` returns the exception's snapshot path; used by the image, video and mlx-video runtimes, which already validate the files they need. 3 tests in [test_hf_local.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_hf_local.py). FLUX.2 klein 4B now renders a 1024² image in 21 s on MPS.

</details>

<details>
<summary>FU-095 · Relabel TurboQuant as a memory saver — Shipped 2026-09-30.</summary>

**Status or trigger:** **Shipped 2026-09-30.**

**Notes:** FU-066 measured TurboQuant 3-bit at 5.6–7.4 tok/s against Native's 25.7–29.7 on Qwen3-4B-bf16 (M4 Max, short context); on MLX it also bypasses prompt-cache reuse ([mlx_worker_prompt_cache.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_prompt_cache.py) is gated on `native`) and is locked out of speculative decoding. The UI no longer implies it is an acceleration option: strategy card + chat-composer chip carry a "Memory saver" tag (`isMemorySaverStrategy` in [runtimeSupport.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/runtimeSupport.ts), legacy ids included), the Discover / Models filter chips read "TurboQuant · memory saver" with an explanatory tooltip, the routing badge's cache tooltip says so, and the Native card now says it is the fastest choice and works with speculative decoding. Docs: "Which one should I pick?" + "A memory saver, not a speed option" in [cache-strategies.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/features/cache-strategies.md). **Follow-ups shipped 2026-10-01:** (a) the locale `runtime.cacheStrategies.turboquant.description` is translated in all nine other locales (`i18n:validate` clean; the key still has no code consumer, and new strings in this codebase ship as inline `defaultValue` English — 855 keys repo-wide are in that state, so there is no per-feature translation gate to satisfy); (b) `_strategy_speed_map("turboquant")` was 0.60× at 3-bit against a measured ~0.2–0.28×, so it is now 0.15 / 0.17 / 0.20 / 0.22 at 1 / 2 / 3 / 4 bits (the 3-bit point is the 0.28.0 measurement; the others keep the old shape scaled to it and are **unmeasured**), pinned by a regression test, and `scripts/calibrate-cache-speed.py` measures native vs TurboQuant at several bit widths and prompt sizes through a running backend and prints the line to paste. **2026-10-29 run (M4 Max, `mlx-community/Qwen3-4B-4bit`, 2 / 3 / 4 bits, 512 / 4096 / 16384 prompt tokens, 128-token decode):** Native 163.0 / 128.1 / 75.0 tok/s; TurboQuant 29.2 / 14.4 / 3.0 (2-bit), 31.0 / 12.5 / 3.1 (3-bit), 34.3 / 15.2 / 3.0 (4-bit), so the ratio is 0.18-0.21, 0.10-0.12 and 0.04: **compressed KV does not narrow the gap at long context, it widens it** (25 times slower at 16K). `_strategy_speed_map` for a typical 2048-token conversation is now 0.13 / 0.13 / 0.13 / 0.15 (1 bit unmeasured), and the launch summary adds the 16K figure past 4K context; raw numbers `~/.chaosengine/test-results/cache-speed-20261029-174554.json`. **Open:** one model and one machine; TurboQuant 0.28 may have a slow path worth reporting upstream (a 25x drop is larger than the KV traffic alone explains); the estimate does not yet take the actual conversation length into account.

</details>

<details>
<summary>FU-092 · Non-Apple STT passes MLX Whisper ids to faster-whisper — Shipped 2026-09-30 (+ Parakeet on Apple Silicon).</summary>

**Status or trigger:** **Shipped 2026-09-30 (+ Parakeet on Apple Silicon).**

**Notes:** [voice_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_runtime.py) `_STT_MODELS` is now one logical model per row with a per-backend source; `list_stt_models` shows only what the host can run and `resolve_stt_model` maps any catalog id to the host's sibling (so a stale MLX id on Linux still works). **CTranslate2 sources:** exactly what faster-whisper 1.2.1 maps its own size tokens to (`faster_whisper/utils.py` `_MODELS`): `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `Systran/faster-whisper-{small,base}` — trusting nothing the library doesn't already trust. **Parakeet:** `mlx-community/parakeet-tdt-0.6b-v3` via `parakeet-mlx>=0.5.2` (Apache-2.0; weights NVIDIA CC-BY-4.0), installed alongside mlx-whisper, becomes the Apple default once installed; Whisper stays for non-European languages. Single-slot model cache for Parakeet/faster-whisper (no reload per utterance). `/api/voice/transcribe` rejects non-catalog ids with 400 (the id reaches loaders that also accept filesystem paths). **Open:** live Windows/Linux transcription + a live Parakeet run on a Mac; the Parakeet download size (2.5 GB) is from memory, not checked against the Hub.

</details>

<details>
<summary>FU-091 · PyPI dflash collides with our in-repo dflash package — Shipped 2026-09-30</summary>

**Status or trigger:** **Shipped 2026-09-30 — kept our module name, dropped the PyPI package.**

**Notes:** vLLM serves DFlash natively (`speculative_config={"method": "dflash"}`), so the PyPI package was never needed, and its exact pins (`tqdm==4.70.0`, `datasets==5.0.1`, `requests==2.34.2`) would land in the shared extras dir. `dflash.is_vllm_available()` now reads vLLM's package metadata (no import) and requires ≥0.28 (DFlash 2, vllm#52816); `accelerators.dflash_cuda_{available,version}` follow. Setup: `"dflash"` moved to `_MANUAL_INSTALL_MESSAGES` (explains vLLM); `"vllm"` installs `vllm>=0.28.0` so an older install upgrades. `[dflash]` extra removed; `[vllm]` / `[triattention]` floor 0.28. Frontend DFlash (CUDA) accelerator + composer hint install `vllm`. Renaming our package was rejected: a bigger diff through every import, for no gain once nothing installs the PyPI one. Note vLLM 0.30 pins `torch==2.13.0` exactly.

</details>

<details>
<summary>FU-090 · MLX prompt-cache reuse is off for hybrid models — Shipped 2026-09-30.</summary>

**Status or trigger:** **Shipped 2026-09-30.**

**Notes:** `ArraysCache.is_trimmable()` is False in mlx-lm 0.31.3, so Qwen3.5/3.6/3.8, Qwen3-Next/Coder-Next, Nemotron-H and Mamba re-prefilled the whole chat every turn. [mlx_worker_prompt_cache.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_prompt_cache.py) now, for caches whose layers are all `ArraysCache`-or-trimmable, prefills all but the last prompt token itself, copies the recurrent layers' state at that boundary (`mx.array` copies; a few small arrays per layer), and hands the final token to `stream_generate`. Next turn, when the common prefix reaches the boundary, attention layers trim back to it, recurrent layers restore the copy, and only the new suffix is prefilled. Any mismatch falls back to a fresh full prefill. No new deps. `HybridCheckpointTests` in [tests/test_mlx_prompt_cache.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_prompt_cache.py) run a tiny random `qwen3_next` model on real MLX (CPU wheel on Linux; skips without mlx): restored-path logits match a full prefill, and a mutation that skips the restore fails. Upstream mlx-lm #1821 would make this redundant.

</details>

<details>
<summary>FU-088 · Catalog: Qwen 3.8, Bonsai, Nemotron 3.5 Lightning, MiniCPM 5, Ornith 1.5 — Shipped 2026-09-28 (runnable rows only).</summary>

**Status or trigger:** **Shipped 2026-09-28** (runnable rows only).

**Notes:** New families `qwen-3-8` (Qwen3.8-27B: config identical to Qwen3.6-27B bar `transformers_version`), `bonsai` (PrismML 1-bit Q1_0 GGUF + ternary MLX 2-bit), `nemotron-3-5-lightning` (nemotron_h), `minicpm-5` (llama arch); Ornith 1.5 appended to `ornith-1`. Live-validated on the M4 Max: Ternary-Bonsai-27B MLX 44.8 tok/s @ 7.9 GB, Ternary-Bonsai-4B MLX 213 tok/s and Bonsai-1.7B Q1_0 GGUF 341 tok/s through the real backend load + chat-stream path on stock mlx-lm 0.31.3 / llama.cpp b9180. **Deliberately not added** (need runtimes we don't ship): Qwen3.8-Flash-Next (`qwen4_exp`: mlx-vlm ≥0.6.17 / llama.cpp ≥b10660), Bonsai 2 (PrismML llama.cpp fork or mlx-vlm ≥0.7.2), Bonsai MLX 1-bit (mlx-vlm 1-bit kernels), Ternary Bonsai Q2_0-g64 GGUF (llama.cpp ≥b9994), Muse Glimmer 30B (≥b10353 / mlx-vlm ≥0.6.12), GLM-5.3-Flash (`glm5_next`), DeepSeek-V4-Flash-0731 (`deepseek_v4` is mlx-vlm-only — the existing `deepseek-v4` MLX rows have the same problem), Kimi K3. Unlock = bump the bundled llama.cpp + route MLX loads by `config.json` `model_type` to mlx-vlm (≥0.7 needs mlx ≥0.32.2 + transformers ≥5.14 — check against the dflash-mlx pin). Qwen 3.7 still has no open weights; no `Wan-AI/Wan3.0-*` repo yet (FU-082).

</details>

<details>
<summary>FU-087 · Text GGUF rows downloaded every quant in the repo and loaded the largest — Shipped 2026-09-28.</summary>

**Status or trigger:** **Shipped 2026-09-28.**

**Notes:** `/api/models/download` snapshot-downloaded whole repos for text models and `_resolve_gguf_path` loaded the largest non-mmproj `.gguf`. So the `lmstudio-community/Qwen3.6-27B-GGUF` "Q4_K_M 16.5 GB" row fetched 68 GB and ran the Q8_0; the unsloth GLM-5.1 / Mistral-Large-3 rows would have pulled ~11 TB each; split quants resolved to the largest shard instead of `-00001-of-`. New `ggufFile` field on text variants (as image/video already had) + [backend_service/helpers/text_gguf.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/text_gguf.py): downloads get an allowlist of the pinned quant (all shards) + `*mmproj*.gguf`, the progress total is sized to those files, and the resolver prefers the pin (repo recovered from the `models--owner--name` or `owner/name` path), else the largest file stepped back to its first shard. Pinned all 18 multi-quant catalog rows (filenames checked against the Hub tree); `unsloth/Kimi-K2.6-GGUF` relabelled `UD-Q4_K_XL` (it has no Q4_K_M). 14 tests in [tests/test_text_gguf.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_text_gguf.py).

</details>

<details>
<summary>FU-086 · Nunchaku / SageAttention install buttons installed the wrong thing (corrects FU-059) — Shipped 2026-09-28.</summary>

**Status or trigger:** **Shipped 2026-09-28.**

**Notes:** PyPI `nunchaku` is an **unrelated** piecewise-linear-segmentation library (author Yu Huo, Edinburgh GitLab) — FU-059's "version reset to 0.16" was that other project. The SVDQuant Nunchaku (nunchux-ai, Apache-2.0) ships torch/CUDA/Python-specific wheels from GitHub releases only. `sageattention==2.2.0` never resolved (PyPI max is the Triton-only 1.0.6, which diffusers' `sage` backend rejects). Fixes: both keys moved from `_INSTALLABLE_PIP_PACKAGES` to `_MANUAL_INSTALL_MESSAGES` (the Install button now shows wheel/build instructions instead of installing an impostor or failing inside pip); `accelerators.nunchaku_available()` checks for the SVDQuant package's `models/` subpackage on disk (import-free) so the impostor no longer flips "Installed ✓"; both dropped from `_TORCH_ABI_DEPENDENT_PACKAGES`, whose torch-upgrade rebuild reinstalled by bare PyPI name — it would have **overwritten a real Nunchaku with the impostor** and downgraded a source-built SageAttention 2 to 1.0.6. Follow-up worth doing: a one-click SageAttention path via diffusers' `sage_hub` backend (`kernels` package), and a Nunchaku wheel-picker keyed on torch/CUDA/Python tags.

</details>

<details>
<summary>FU-085 · Release bundle never shipped the dflash/ package — Shipped 2026-09-28.</summary>

**Status or trigger:** **Shipped 2026-09-28.**

**Notes:** [scripts/stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) copied `backend_service`, `cache_compression`, `turboquant_mlx` into the embedded runtime but never `dflash` (true since the initial release — confirmed by listing `/Applications/ChaosEngineAI.app`'s `runtime-darwin-aarch64.tar.gz`). Every `from dflash import …` (mlx_engine, vllm_engine, accelerators, helpers/system) raised ImportError in packaged builds and was swallowed, so **DFlash / DDTree never engaged in release builds** — only from a source checkout. Added `dflash` to the copy list; `validateBundledProjectImports` now imports `dflash.DRAFT_MODEL_MAP` and runs with `cwd = backendDest` (it ran from the repo root, so `python -c`'s cwd-first `sys.path` masked anything missing from the bundle). Needs a rebuild to reach users.

</details>

<details>
<summary>FU-084 · Voice I/O unit-test pass — Shipped 2026-09-28.</summary>

**Status or trigger:** **Shipped 2026-09-28.**

**Notes:** [tests/test_voice.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_voice.py) grew 35 → 82 cases (voice modules 86 % → 99 % line coverage) plus a new frontend client suite [src/api/voice.test.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/api/voice.test.ts) (10 cases). The pass surfaced and fixed five real bugs: (1) `_hf_snapshot_path` hardcoded `~/.cache/huggingface/hub`, so with Settings → `hfCachePath` (→ `HF_HOME`) every voice model reported "not installed" — now delegates to `helpers.hf_cache_paths._hf_repo_snapshot_dir`; (2) the kokoro-onnx lane imported `soundfile`, which kokoro-onnx doesn't depend on and "Install kokoro-onnx" never pulled in — fresh Windows/Linux TTS died with `No module named 'soundfile'`; now encodes 16-bit PCM WAV with stdlib `wave` + numpy; (3) a failed kokoro download stranded its `NamedTemporaryFile(delete=False)` partial (up to ~310 MB per retry) — unlinked on failure now; (4) faster-whisper transcripts joined segments with `" "` although each segment carries its own leading space (`"hello  world"`) — concatenated like Whisper's own `text`; (5) `voice[:1] in "abefhipjz"` is True for an empty id, so `voice[0]` raised IndexError, and the gallery id regex used `match` + `$`, accepting a trailing newline — `fullmatch` now. Also pins every voice install key against the Setup allowlist. **Open:** see FU-092.

</details>

<details>
<summary>FU-081 · Bump vllm floor &gt;=0.23.0 → &gt;=0.24.0 — Shipped 2026-06-30.</summary>

**Status or trigger:** **Shipped 2026-06-30.**

**Notes:** Routine upstream scan: PyPI `vllm` latest is 0.24.0 (floor was one minor behind). Floor bump only in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) `[vllm]` + `[triattention]` extras; loose `>=` so existing installs already resolved newer wheels — this just keeps the documented floor honest. No code changes; not exercised live on this Apple Silicon box (CUDA-only).

</details>

<details>
<summary>FU-080 · Backend cold start dragged in torch via cache-strategy availability probes — Shipped 2026-05-29.</summary>

**Status or trigger:** **Shipped 2026-05-29.**

**Notes:** `python -X importtime backend_service.app` measured **2.6 s**, of which **1.64 s was `diffusers.hooks`** (→ `torch` → `torch._dynamo` → `sympy`) — blowing the CLAUDE.md "< 2 s backend startup" target. Traced the chain: state init → system snapshot → `_get_cache_strategies()` → `registry.available()` instantiates every strategy and calls `is_available()`, and the 5 diffusion strategies (fbcache / taylorseer / magcache / pab / fastercache) answered availability by **actually importing `diffusers.hooks`** — pulling the whole torch stack onto the cold-start path on every launch. Fix: new [cache_compression/_diffusers_probe.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/_diffusers_probe.py) `diffusers_at_least(major, minor)` reads the installed version via `importlib.metadata.version` (metadata only — never executes `diffusers.__init__`, so no torch). Each `is_available()` now gates on the version (fbcache ≥0.36, the other four ≥0.38); the real `diffusers.hooks` import stays lazy inside each `apply_*` method (still raises a clean NotImplementedError on a broken install). Result: `diffusers` / `torch` / `mlx` are **no longer in `sys.modules` after `import backend_service.app`**, import time dropped **2.6 s → ~0.85 s**, and cold-start → first `/api/health` 200 is **2.34 s** (the native-backend MLX subprocess probe was already async — "detection still running" on first health, never blocked startup). Two subprocess-isolated regression guards in [tests/test_cache_strategies.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_cache_strategies.py) (`StartupImportPurityTests`) assert neither `registry.available()` nor `import backend_service.app` pulls torch/diffusers, so this can't silently regress. All 5 diffusion strategies still report `available=True` against the installed diffusers 0.38.

</details>

<details>
<summary>FU-079 · MTPLX proxy doesn't surface incremental tokens to the chat stream (empty output) — Shipped 2026-09-30 (with MTPLX 2.12).</summary>

**Status or trigger:** **Shipped 2026-09-30 (with MTPLX 2.12).**

**Notes:** Root cause: MTPLX's reasoning parser (qwen3 by default) moves the `<think>` block into `delta.reasoning_content`; [mtplx_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/mtplx_engine.py) only read `delta.content`, so a thinking model that spent its budget reasoning streamed nothing while reporting real tok/s. Reasoning deltas now stream as reasoning chunks with a `reasoning_done` marker before the answer; `thinking_mode=off` sends `chat_template_kwargs.enable_thinking=false`. MTPLX 2.12: installer floor `mtplx>=2.12.0`; dropped the forced `--profile performance-cold --max` (upstream now calls Burst a legacy benchmark mode that pins fans at 100 %, and it bypassed the per-model Turbo profile where the 2.12 gains are); the runtime note carries the in-product "Powered by MTPLX" credit its NOTICE requires. Stub server emits `reasoning_content`; 2 new integration tests. FU-048 (prefer GGUF MTP over MTPLX) is still open — re-bench against 2.12. **Open:** live run on a Mac.

</details>

<details>
<summary>FU-078 · MtplxEngine handed MTPLX a bare repo id instead of the local snapshot path — Shipped 2026-05-29.</summary>

**Status or trigger:** **Shipped 2026-05-29.**

**Notes:** Final MTPLX blocker: `mtplx quickstart` died with "model is not available locally. Run: mtplx pull Qwen/Qwen3.5-4B" — it resolves a model *id* against its own registry/cache, not the HF hub cache. [mtplx_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/mtplx_engine.py) set `model_arg = path or runtime_target or model_ref`, and for raw HF-org repos `path` is None while `runtime_target` is the *repo id* (`Qwen/Qwen3.5-4B`), so MTPLX got an id it couldn't find. Fix: whenever the candidate isn't an existing local directory, resolve the already-downloaded HF snapshot dir via `snapshot_download(model_ref, local_files_only=True)` (no network) and pass that. Live-verified: MTPLX now **loads + engages** (note "MTPLX MTP speculative decoding active (draft tokens: 1, model: Qwen3.5-4B)", reports 17.8 tok/s) instead of failing to start. Also fixed the matrix runner's `0.0 tok/s` (read `done.assistant.metrics.tokS`, not a non-existent top-level `tokensPerSecond`) + captured `dflashAcceptanceRate`. **Verified-genuine after these fixes: DFlash (33.2 tok/s), DDTree (31.4 tok/s), GGUF-MTP (14.7 tok/s), turboquant MLX/GGUF, triattention, native** — all stream real output with real throughput. MTPLX still has one remaining issue → FU-079.

</details>

<details>
<summary>FU-077 · MTPLX isolated venv had a truncated install (missing server deps) — Shipped 2026-05-29.</summary>

**Status or trigger:** **Shipped 2026-05-29.**

**Notes:** After FU-076 routed correctly, `MtplxEngine` startup died: `ModuleNotFoundError: No module named 'numpy'` — and then `safetensors`, `uvicorn`, `fastapi`, `pydantic`, `mlx-lm`, `rich`… The `~/.chaosengine/mtplx-venv` was a *truncated* install (interrupted `pip install mtplx`), but the installer's verify only ran `import mtplx`, which succeeds because the server deps are imported lazily by `mtplx.server.openai` (not at package top level). Fixed the live venv with a full `pip install --upgrade mtplx` (0.3.5 → 0.3.7, pulled all deps). Hardened [scripts/install-mtplx.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/install-mtplx.sh): the verify now imports `mtplx.server.openai` (the real server entrypoint) and auto-retries a full dependency install once before failing loudly, so a truncated install can't pass silently again.

</details>

<details>
<summary>FU-076 · MTP tensor probe missed top-level mtp. keys → MTPLX never selected — Shipped 2026-05-29.</summary>

**Status or trigger:** **Shipped 2026-05-29.**

**Notes:** The matrix MTPLX cell routed to the DFlash path instead of `MtplxEngine`. `RuntimeController._select_engine` gates MTPLX on `has_mtp_heads_strict(repo, path)`, which calls `model_has_mtp_tensors(path)` → scans the safetensors index against `_MTP_TENSOR_HINTS = ('mtp_heads.', 'mtp_decoder.', 'mtp_emb.', 'model.mtp.', '.mtp.')`. Every hint assumes a *nested* key, but Qwen3.5 / Qwen3.6 ship the MTP head as **top-level** `mtp.layers.*` / `mtp.fc.weight` (no leading prefix) — so the probe returned False on a genuinely MTP-bearing model and MTPLX was skipped. Live-confirmed: `model_has_mtp_tensors` returned False on the real `Qwen/Qwen3.5-4B` snapshot. Fix in [_mtp.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_mtp.py): also match `tensor_name.startswith("mtp.")`. New `test_safetensors_index_with_top_level_mtp_keys` in [tests/test_inference.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_inference.py).

</details>

<details>
<summary>FU-075 · MLX spec-dec silently broken — stale configure_full_attention_split import — Shipped 2026-05-29.</summary>

**Status or trigger:** **Shipped 2026-05-29.**

**Notes:** **Highest-impact bug this sweep.** Inspecting the matrix runtimeNotes (not just pass/fail) revealed the MLX DFlash / DDTree / MTPLX cells were *passing the weak non-empty-output check while NOT actually running spec-dec* — `actual_strategy: native`, note `dflash-mlx could not be imported (cannot import name 'configure_full_attention_split' from 'dflash_mlx.runtime')`. Root cause: dflash-mlx 0.1.5 moved the pre-0.1.5 top-level `configure_full_attention_split` onto the per-family `target_ops` adapter (the FU-006 migration that rewrote `ddtree.py` — but [mlx_worker_lifecycle.py:153](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_lifecycle.py) was missed). Python evaluates the whole `from … import a, b` line, so the failed `configure_full_attention_split` symbol killed the co-imported `load_draft_bundle` too → `_dflash_generator` never loaded → **every** MLX spec-dec path fell back to standard generation for all users. Fix: import `load_draft_bundle` + `resolve_target_ops` (both still top-level), resolve the adapter, and call `target_ops.configure_full_attention_split(...)` only for the `hybrid_gdn` family (it's a no-op for pure-attention Qwen3/3.5/3.6 — upstream only calls it there). Live-verified after fix: DFlash note "DFLASH speculative decoding active (draft: z-lab/Qwen3-4B-DFlash-b16)", DDTree "DDTree active (budget=16)".

</details>

<details>
<summary>FU-074 · GGUF MTP speculative decoding had no UI toggle — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** FU-047 wired the GGUF MTP backend (`--spec-type draft-mtp`, gated on the `speculativeDecoding` request flag in [llama_cpp_engine.py:531](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py)) + the `ggufMtpAvailable` capability flag, but never surfaced a UI control. The launch modal's only spec-dec toggles are DFlash (hidden for GGUF — "not supported with llama.cpp models") and MTPLX (Apple-Silicon MLX only), so a user loading `ggml-org/Qwen3.6-27B-MTP-GGUF` had **no way to enable** the lane — only the matrix runner could, by POSTing `speculativeDecoding=true` directly. The button audit (this turn) caught it. Added an `isMtpGgufRepo(repo)` helper in [runtimeSupport.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/runtimeSupport.ts) (mirrors backend `is_mtp_gguf_repo`: MTP-flavoured name on a GGUF repo) + a "GGUF MTP" toggle in [RuntimeControls.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/RuntimeControls.tsx), shown only when `isGgufBackend && isMtpGgufRepo(selectedCanonicalRepo)` (FU-034 hide-when-not-applicable). It binds to the same `speculativeDecoding` flag the backend reads; no cache-strategy lock (GGUF KV cache is orthogonal to MTP draft decode, unlike MLX DFlash which forces native). Also patched the DFlash-availability reset effect (was clearing `speculativeDecoding` for any non-DFlash model — would have instantly un-ticked the GGUF-MTP box) to keep it on for `ggufMtpModelSupported`. Old binaries without `--spec-type` fall back to standard decode + a runtimeNote (backend FU-047 path) — acceptable since the bundled llama-server is current; a future refinement could additionally gate the toggle on the `ggufMtpAvailable` capability for old-binary boxes (needs the flag threaded through the ~8 RuntimeControls call sites). 8 new `isMtpGgufRepo` unit tests in [runtimeSupport.test.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/__tests__/runtimeSupport.test.ts). Verified live: matrix `gguf MTP (Qwen3.6-27B)` cell PASS (sha 74a1eca8b3b4).

</details>

<details>
<summary>FU-073 · Matrix MTPLX cell targeted a non-MTP VL model — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** `scripts/cache-strategy-matrix.py` `MID_MLX_MTPLX_CAPABLE` was `mlx-community/Qwen3.5-4B-bf16` — a VL conversion (ships `video_preprocessor_config.json`) that carries no MTP heads and is absent from both `MTP_MODEL_MAP` and `_MTP_ALIASES`, so the MTPLX cell could never have exercised MTP even with the model on disk (it'd fail the `has_mtp_heads_strict` tensor probe). Switched to the canonical `Qwen/Qwen3.5-4B`, which is a direct `MTP_MODEL_MAP` key (verified `mtp.layers.*` + `mtp.fc.weight` in its safetensors index), a catalog variant (so it passes the `library_refs` check), and downloaded to exercise the lane. Pairs with the FU-070 download-skip classifier so the cell reports honestly on boxes without the model.

</details>

<details>
<summary>FU-072 · Restore vision capability to Qwen3.5 + Qwen3.6 families (reverse FU-040) — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** FU-040 (2026-05-10) removed `vision` from Qwen3.6-27B + family, asserting the dense model was text-only with vision on "a separate `Qwen3.6-27B-VL` we don't ship." Re-checking upstream on 2026-05-28: **every** Qwen3.5/3.6 `config.json` now ships `architectures: [Qwen3_5ForConditionalGeneration]` / `[Qwen3_5MoeForConditionalGeneration]` with `vision_config` + `image_token_id` + `vision_start/end_token_id` — the base models are natively multimodal. `mlx-vlm` ships `qwen3_5` + `qwen3_5_moe` model support, and the `ggml-org/*-GGUF` packs include an `mmproj-*.gguf` sibling (auto-wired by `llama_cpp_engine._resolve_mmproj_path` → `--mmproj`). The catalog was also internally inconsistent (Qwen3.5-9B tagged vision, Qwen3.5-4B not, same arch). Re-added `vision` across both families in [text_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/text_models.py): qwen-3-6 family-level + all 11 variants; qwen-3-5 family-level + `Qwen3.5-4B` (vision+video, matching its 9B sibling) + `lmstudio-community/Qwen3.5-9B-GGUF`. **Safety net (why this can't resurrect the FU-040 broken-button bug):** the composer "Attach image" affordance ([ChatComposer.tsx:129](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/chat/ChatComposer.tsx)) reads the *runtime* `supportsVision`, which [catalog/capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/capabilities.py) demotes to False for the MLX worker (carries no images today) and gates on actual `--mmproj` resolution for GGUF ([llama_cpp_engine.py:737](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py) `visionEnabled=attempt_mmproj_path is not None`). So the catalog `vision` tag now drives only the variant-picker / discover badges (capability-in-principle), while the functional button stays runtime-accurate. `gemma-4` was already correctly vision-tagged (mlx-vlm `gemma4` support) — left untouched. Catalog parses + `test_capabilities` / `test_mmproj_vision` green.

</details>

<details>
<summary>FU-071 · DDTree availability probe checks pre-0.1.5 symbol names — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** The cache-strategy matrix `ddtree spec-dec` cell skipped with *DDTree runtime not available* even though `dflash_mlx` 0.1.5.1 is installed and `backend_service/ddtree.py` works. Root cause: `dflash.is_ddtree_available()` ([dflash/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/dflash/__init__.py)) source-greps the installed `dflash_mlx.runtime` for three required symbols and the list was stale — it required `target_forward_with_hidden_states`, which dflash-mlx 0.1.5 **renamed** to the per-family adapter `target_ops.forward_with_hidden_capture` (the same FU-006 migration that rewrote our `ddtree.py` to call `resolve_target_ops(target_model)`). The probe was never updated alongside that rewrite, so it required a symbol that (a) no longer exists in any modern dflash-mlx build (`grep -c` = 0 in the installed `runtime.py`) and (b) our own code no longer uses. Confirmed the real contract our DDTree path imports: `resolve_target_ops` (ddtree.py adapter entry), `load_draft_bundle` (worker lifecycle), `stream_dflash_generate` (speculative). Updated `required_symbols` to those three; dropped the obsolete name + the unused `load_target_bundle`. `dflash.is_ddtree_available()` now returns `True` on this M4 Max box. 4 new `DDTreeAvailabilityProbeTests` in [tests/test_dflash.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_dflash.py) mock the runtime source so a future rename can't silently regress the probe again. Note: when FU-057 bumps dflash-mlx to 0.1.7 (which removes `configure_full_attention_split` and reshapes `stream_dflash_generate`), this probe + the lifecycle import need re-checking in lockstep.

</details>

<details>
<summary>FU-070 · Matrix runner: classify missing-download as SKIP, not FAIL — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** The full `scripts/cache-strategy-matrix.py` sweep on 2026-05-28 reported the `gguf MTP (Qwen3.6-27B)` cell as **FAIL** — `POST /api/models/load -> 500: Cannot load 'ggml-org/Qwen3.6-27B-MTP-GGUF': No .gguf, .safetensors, or pytorch weights found in HF cache entry.` Root cause: the repo had an empty `~/.cache/huggingface/hub/models--ggml-org--Qwen3.6-27B-MTP-GGUF/` dir (4.0 KB, only `refs/main`, dated May 16 — an interrupted pull), and the runner's `skip_reason` library check uses `caps.library_refs`, which is built from the **catalog** (every variant repo from `/api/workspace`), not from what's actually downloaded. So a catalogued-but-undownloaded model passes the library check and only errors at load — reported as a product FAIL when it's really a missing download (same false-positive class as FU-053). Fix: new pure helper `classify_load_skip(msg)` in [scripts/cache-strategy-matrix.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/cache-strategy-matrix.py) matches the backend's 'no weights found in HF cache entry' markers; `run_cell` now wraps the load call separately and converts that specific error into `skipped=True, skip_reason="weights not downloaded (<ref>)"` instead of a failure. Genuine load errors (OOM, etc.) still surface as fails. 4 unit tests in [tests/test_cache_strategy_matrix_runner.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_cache_strategy_matrix_runner.py) (`ClassifyLoadSkipTests`) pin the classification. The dflash/mtplx cells already skipped correctly because their target models (`mlx-community/Qwen3-4B-bf16` / `Qwen3.5-4B-bf16`) aren't catalog variants so they never entered `library_refs`. **To actually exercise the GGUF-MTP lane (FU-047/FU-052 trip-wire), download `ggml-org/Qwen3.6-27B-MTP-GGUF` first**, then re-run full.

</details>

<details>
<summary>FU-069 · Bump turboquant-mlx-full floor &gt;=0.4.0 → &gt;=0.5.0 — Shipped 2026-05-28.</summary>

**Status or trigger:** **Shipped 2026-05-28.**

**Notes:** Upstream `turboquant-mlx-full` 0.5.0 on PyPI (FU-062 had just floored at 0.4.0 on 2026-05-25). v0.5.0 builds on the v0.4.0 expert-streaming path (FU-062) with **parallel expert prefetch** — the missing MoE experts for each layer are read on a thread pool (`--prefetch-workers`, default `8`) so SSD latency hides behind compute. Upstream-reported **~1.9× faster decode** at a tight cache budget, still bit-identical output. `--prefetch-workers 1` restores the serial v0.4.0 behaviour. No API change at our call surface — runtime still constructs `TurboQuantKVCache` with the same signature; the new flag is converter/runtime-side. Floor bump only in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) `[turboquant]` extra; loose `>=` so existing 0.4.x installs stay satisfied locally. Apple Silicon only. Folds in the spirit of FU-066 (the matrix MoE-turboquant baseline should be captured against 0.5.0 once the wheel is installed on the M4 Max box).

</details>

<details>
<summary>FU-068 · MLX probe timeout 12 s → 20 s — Shipped 2026-05-25 (v0.9.3).</summary>

**Status or trigger:** **Shipped 2026-05-25 (v0.9.3).**

**Notes:** E2E full-sweep Phase 1 surfaced three intermittent fails on a freshly-booted backend — `MLX native cache` / `MLX TurboQuant cache` / `fused attention flag` all returned `MLX backend requested but unavailable: ...mlx_worker probe timed out after 12.0 seconds`. Measured cold-start: `time .venv/bin/python -m backend_service.mlx_worker probe` = **12.43 s** on M4 Max / Python 3.11 against current `mlx 0.31.2` + `mlx-lm 0.31.3` + `mlx-vlm 0.4.4` — 0.4 s past the 12.0 s ceiling. The 12.0 s value was an arbitrary default from the v0.8.0 `capabilities.py` extract (commit `f91709e`), never tuned. Bumped to **20.0 s** in [backend_service/inference/capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/capabilities.py) `_probe_native_backends` — ~60% headroom over today's envelope. Phase 5 video gen + Phase 1 GGUF / DFlash / cache-preview already passed (proves MLX itself works once the probe lands), so this was a pure cold-boot probe timing issue, not a regression from the FU-062 / FU-063 floor bumps (which are loose `>=`, no installed package changed).

</details>

<details>
<summary>FU-064 · Add ggml-org/Qwen3.6-{27B,35B-A3B}-GGUF non-MTP catalog rows — Shipped 2026-05-25 (v0.9.3).</summary>

**Status or trigger:** **Shipped 2026-05-25 (v0.9.3).**

**Notes:** ggml-org published canonical Q8_0 non-MTP companion packs on 2026-05-22 alongside the MTP variants we wired in FU-047. Two new rows in [text_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/text_models.py) `qwen-3-6` family: `ggml-org/Qwen3.6-27B-GGUF` (Q8_0, 29 GB, dense) + `ggml-org/Qwen3.6-35B-A3B-GGUF` (Q8_0, 37 GB, MoE). Catalog note steers users at the MTP siblings when they want spec-dec. No runtime changes — direct `llama.cpp` lane, same as the lmstudio-community Q4_K_M variants already shipping.

</details>

<details>
<summary>FU-063 · Bump mlx-vlm floor &gt;=0.4.0 → &gt;=0.5.0 — Shipped 2026-05-25 (v0.9.3).</summary>

**Status or trigger:** **Shipped 2026-05-25 (v0.9.3).**

**Notes:** Upstream `mlx-vlm` 0.5.0 on PyPI (installed was 0.4.4). Minor bump, no API breakage at our call surface (`mlx_vlm.load` + `mlx_vlm.generate` from [mlx_worker_multimodal.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_multimodal.py)). Floor bump in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) `[mlx-vlm]` extra; loose `>=` semantics mean existing 0.4.x installs are still satisfied locally, but fresh installs pick up the newer wheel which carries the upstream Qwen3.5-VL + GLM-4.5V fixes.

</details>

<details>
<summary>FU-062 · Bump turboquant-mlx-full floor &gt;=0.3.0 → &gt;=0.4.0 — Shipped 2026-05-25 (v0.9.3).</summary>

**Status or trigger:** **Shipped 2026-05-25 (v0.9.3).**

**Notes:** Upstream `turboquant-mlx-full` 0.4.1 on PyPI (installed was 0.3.0, FU-001 pin). v0.4.0 added **expert streaming** — pages router-selected MoE experts from disk per token, runs models whose weights exceed available RAM. Live-validated upstream against `Qwen3.6-35B-A3B` (35B sparse) on a 16 GB Mac mini in under 4 GB RAM, output bit-identical to fully-resident model. Compounds with our existing Hadamard rotation + Lloyd-Max codebook K/V compression. Floor bump only — no API changes required, runtime continues to call `TurboQuantKVCache` with the same signature. Pin lives in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) `[turboquant]` extra. Apple Silicon only (CUDA users stay on the `llama-server-turbo` binary path via FU-001's parallel track).

</details>

<details>
<summary>FU-061 · "Watching upstream" badge + disabled download for tracked-only image seeds — Shipped 2026-05-18.</summary>

**Status or trigger:** **Shipped 2026-05-18.**

**Notes:** User-reported gap: downloaded `baidu/ERNIE-Image-Turbo` from Image Discover (it sits in `LATEST_IMAGE_TRACKED_SEEDS`), expected it in the Studio dropdown, didn't appear. Root cause: tracked seeds are discovery-only — Studio's dropdown is fed by `IMAGE_MODEL_FAMILIES` which requires explicit pipeline routing (flow-match flags, sampler registry, scheduler defaults). ERNIE-Image (+ Nucleus-Image, Z-Image, HiDream, GLM-Image, FLUX.2 family) has no diffusers-routable Studio variant yet. Fix path A picked over path B (full per-family pipeline wiring) — surgical UX disambiguation. **Backend:** new `_is_launchable_image_repo(repo_id)` helper in [backend_service/helpers/images.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/images.py) returns True only when `repo_id` resolves to a curated `IMAGE_MODEL_FAMILIES` variant. Wired into both payload sites — `_tracked_latest_seed_payloads` (line 411) + the live-HF lane (line 622) — so every Discover row carries `trackedOnly: bool`. **Frontend:** new `trackedOnly?: boolean` field on `ImageModelVariant` ([src/types/image.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/types/image.ts)). [ImageDiscoverTab.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/images/ImageDiscoverTab.tsx) chip row gains a "Watching upstream" badge + tooltip when `trackedOnly`. Action column branches first on `trackedOnly` → renders a disabled `IconActionButton` with tooltip "Watching upstream — Studio playback for this family isn't wired yet. Catalog entry is for awareness; download won't unlock Studio." instead of the Generate / Download / Resume CTAs. Backward-compat: existing curated families have `trackedOnly: undefined` → falsy → no UX change. **Tests:** new `TrackedOnlyFlagTests` in [tests/test_image_discover.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_image_discover.py) — 5 cases covering `_is_launchable_image_repo` (FLUX.1-dev + SDXL = true; ERNIE-Image / Nucleus-Image = false; empty = false), `trackedOnly: True` on ERNIE seed payload, and the negative case where a tracked seed that IS in IMAGE_MODEL_FAMILIES must NOT carry the flag (forward-compat for catalog evolution). **Follow-up path B (deferred):** wire ERNIE-Image / Nucleus-Image / Z-Image / HiDream / GLM-Image / FLUX.2 family as real launchable families via per-family pipeline detection in `image_runtime`. Multi-hour per family, gated on diffusers' upstream support landing for each architecture.

</details>

<details>
<summary>FU-060 · Mock memory-pressure gate in route tests — Shipped 2026-05-18.</summary>

**Status or trigger:** **Shipped 2026-05-18.**

**Notes:** The `gate_video_generation` + `gate_image_generation` pre-flight checks in [backend_service/helpers/memory_gate.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/memory_gate.py) read live psutil via `snapshot_memory_signals()`. On a busy dev box (parallel pytest + vitest + Tauri running) the host legitimately sits at >92% memory pressure and the route returns `503 — Memory pressure is 97% — video generation would likely OOM`. The 7 `VideoGenerateRouteTests` cases in [tests/test_video_routes.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_video_routes.py) + the 2 image equivalents in [tests/test_backend_service.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_backend_service.py) (`test_image_generate_unloads_idle_video_runtime_first` + `test_image_generate_rejects_while_video_generation_active`) never mocked this dependency, so a half-full RAM box would surface them as flake. Fix: `setUp` patches `backend_service.helpers.memory_gate.snapshot_memory_signals` with a fixed `(64.0, 10.0)` return — 64 GB available, 10% pressure — so every assertion exercises the actual route logic and FastAPI handler shape, not the host OS state. `tearDown` stops the patch. The function-local `import` in [routes/video.py:335](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/video.py) means the patch lives at the source module rather than `video_routes` namespace; this works for both the image and video gates since they share the same `snapshot_memory_signals` helper. Repro: was deterministic when running `pytest tests/test_mlx_video.py tests/test_video_routes.py` in that order (the mlx-video tests' MLX import allocations briefly pushed the box past the 92% gate).

</details>

<details>
<summary>FU-059 · Nunchaku pin correction: &gt;=1.2.1 → &gt;=0.16.0 — Shipped 2026-05-17</summary>

**Status or trigger:** **Shipped 2026-05-17 — superseded by FU-086: the PyPI `nunchaku` package is an unrelated project.**

**Notes:** The FU-023 pin `nunchaku>=1.2.1` in `_INSTALLABLE_PIP_PACKAGES` ([backend_service/routes/setup/__init__.py:118](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/__init__.py)) was unsatisfiable: `pip index versions nunchaku` returns max `0.16.1` on PyPI (full history `0.1.0 → 0.16.1`, no `1.x` line exists). Either the original FU-023 note had a typo, OR upstream pulled / renumbered the `1.x` release after the FU-023 tracker line was written. The setup-tab "Install Nunchaku" action would have failed with "No matching distribution" for every user running it after the original FU-023 ship. Floor lowered to `>=0.16.0` so the install button actually resolves to PyPI's current 0.16.1 wheel. Comment in [routes/setup/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/__init__.py) updated to reflect the version reset. **What still needs verification:** the FU-023 wrappers in [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) (`NunchakuFluxTransformer2dModel` / `NunchakuQwenImageTransformer2DModel` / `NunchakuSD3Transformer2DModel` / `NunchakuSanaTransformer2DModel` / `NunchakuPixArtSigmaTransformer2DModel`) need a live CUDA run against `nunchaku==0.16.1` to confirm the class API still matches — the original FU-023 work was tested against the (now non-existent) 1.2.1 surface, so the class names may have moved between versions. Can't validate on this M4 Max box; trip-wire stays open until the FU-056 Windows-runner work picks it up.

</details>

<details>
<summary>FU-058 · Bump vLLM floor to &gt;=0.21.0 — Shipped 2026-05-17.</summary>

**Status or trigger:** **Shipped 2026-05-17.**

**Notes:** Upstream [v0.21.0](https://github.com/vllm-project/vllm/releases/tag/v0.21.0) landed 2026-05-15 — 367 commits from 202 contributors. Floor bumped from `>=0.8.0` to `>=0.21.0` in both the `[vllm]` and `[triattention]` extras in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml). **Relevant to our open tracker rows:** (a) **Gemma4 MTP** (#41745) + MTP for MiMo-V2.5 (#41905) + EAGLE for Mistral (#41024) — vLLM gains the MTP heads natively that FU-028 still can't get on the MLX side; (b) **TurboQuant hybrid model + uniform quantization** (#39931) — relevant to FU-001 / cache-strategy CUDA parity; (c) **Spec-dec with thinking budget** (#34668) — reasoning-model spec dec is correct now (Qwen3.5/3.6/R1); (d) **Qwen3.5/Mamba hybrid Model Runner V2** (#35520) — matters for FU-031 + Coder-Next; (e) **Gated DeltaNet attention for Qwen 3.5/3.6** CPU path (#41025). **Breaking build changes upstream** (consumed transparently by us): C++20 required (PyTorch compat) — affects FU-052's Windows test runner toolchain; transformers v4 deprecated → must migrate to transformers v5. We do not exercise vLLM on Apple Silicon so this is a no-op locally; FU-052 vLLM cells against `Qwen/Qwen3-0.6B` + `Qwen/Qwen3.5-4B` should be re-run on the next Windows / Linux CUDA pull to confirm the matrix still passes against the bumped floor. No code changes required; the `[vllm]` extra was the only contract point and it stays loose-bounded (`>=`) so users with newer vLLM are still satisfied.

</details>

<details>
<summary>FU-055 · Storage explorer panel in Diagnostics tab — surface top disk consumers in-app — Shipped 2026-05-17</summary>

**Status or trigger:** **Shipped 2026-05-17.** Complements Stuff Diver's blind spot on HF cache layout: blobs live at `~/.cache/huggingface/hub/models--*/blobs/<sha>` (each a single 5–30 GB safetensors / GGUF shard), but `snapshots/<rev>/<filename>` are symlinks — third-party scanners that don't follow symlinks miss the real bytes. New endpoint `GET /api/diagnostics/storage-top?limit=20` walks every directory under `state.settings.modelDirectories` (HF cache + AI_Models + ~/Models + user dirs), sums per-repo via `du`-equivalent path walk with cycle protection (reuses `_path_size_bytes` from [discovery.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/discovery.py#L49)), and returns sorted `[{path, repoLabel, sizeBytes, lastModified, sourceKind}]`. Frontend renders the top-N table in a new "Disk usage" subsection of the Diagnostics tab with a "Reveal in Finder" + "Delete repo" action per row. Cycle protection prevents the `mlx-video-wan` converted-output dirs (which contain symlinks back to HF cache blobs in some configs) from double-counting. Trip-wire numbers from the M4 Max box on first-run: total `~/.cache/huggingface/hub` = 997 GB, top-3 = LTX-2 dev / LTX-2 distilled / LTX-2.3 distilled at ~87 / 87 / 81 GB respectively.

**Notes:** Trigger / Condition — closed inline.

</details>

<details>
<summary>FU-054 · Same-repo variants: show actual on-disk size + shared-repo badge — Shipped 2026-05-17</summary>

**Status or trigger:** **Shipped 2026-05-17.** Wan 2.2 TI2V 5B GGUF (Q4_K_M + Q6_K + Q8_0) renders as three rows in the Video Models tab, each labelled "31.9 GB". On disk those three share ONE `models--QuantStack--Wan2.2-TI2V-5B-GGUF/` dir totalling 12 GB (Q4_K_M=3.2 GB + Q6_K=3.9 GB + Q8_0=5.0 GB), not 95.7 GB as the per-row "31.9 GB" repetition implies. Two surgical changes: (1) Backend `_video_variant_for_payload` now stat-sizes the specific `ggufFile` (not the whole snapshot dir) for variants that pin a single GGUF, exposing `ggufFileBytes` alongside the existing `onDiskBytes`. (2) UI shows the live on-disk byte size next to the catalog estimate when present, and adds a "shares repo with N other variants" badge when ≥2 catalog variants resolve to the same on-disk repo dir. Avoided a full table restructure — the badge gives the user the "deleting this row does/doesn't affect siblings" signal without dragging in expandable parent rows. Frontend changes confined to [VideoModelsTab.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/video/VideoModelsTab.tsx); CSS additions land in [styles.css](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/styles.css) under `.shared-repo-badge`.

**Notes:** Trigger / Condition — closed inline.

</details>

<details>
<summary>FU-053 · Library status false-positive: distill / sibling variants marked installed when only base repo is on disk — Shipped 2026-05-17</summary>

**Status or trigger:** **Shipped 2026-05-17.** Surfaced live: the Video Models tab showed `Wan 2.2 I2V A14B · Distill 4-step (BF16)` + `(FP8)` rows with a green "installed" tick + 117.5 GB sized, but `du -sh ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers-distill-{bf16,fp8}` returned nothing — neither distill repo was actually present. Root cause: both catalog variants share `repo: "Wan-AI/Wan2.2-I2V-A14B-Diffusers"` (the BASE, non-distill repo) and route their distinguishing weights via the separate `distillTransformerRepo: "lightx2v/Wan2.2-Distill-Models"` field (FU-019 pattern). The validator at [_video_variant_validation_error](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/video.py#L226) only checked the *base* repo via `_video_download_validation_error` + `_video_full_precision_weights_validation_error` — both passed because the base repo IS on disk. The distill-specific files (`distillTransformerHighNoiseFile` + `distillTransformerLowNoiseFile`) were never checked, so any variant that pinned distill weights via the FU-019 swap mechanism was unconditionally marked installed once the base was downloaded. Fix: new `_distill_transformer_validation_error` helper that, when a variant declares `distillTransformerRepo`, requires (a) that repo's HF snapshot dir exists and (b) both the high-noise + low-noise filenames are present inside it. Wired into `_video_variant_validation_error` so distill variants now flip to "Not installed" when the distill weights are missing. New unit test in [tests/test_video.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_video.py) pins both the false-positive regression (base-only → not installed) and the happy path (base + distill snapshot → installed). Live verification: after the fix landed on this M4 Max, the two distill rows correctly dropped out of the "installed" filter.

**Notes:** Trigger / Condition — closed inline.

</details>

<details>
<summary>FU-050 · Cache-strategy matrix runner: reasoning-channel capture + max-tokens default — Shipped 2026-05-17</summary>

**Status or trigger:** **Shipped 2026-05-17.** Surfaced during the M4 Max 64 GB live sweep against current-gen Qwen3-0.6B smoke targets: `_stream_inference` in [scripts/cache-strategy-matrix.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/cache-strategy-matrix.py) only captured SSE events with a `token` key, so reasoning models (Qwen3 / Qwen3.5 / Qwen3.6 / DeepSeek-R1) that emit the `<think>...</think>` block on the separate `reasoning` channel produced empty `full_text` when the entire `max_tokens` budget was spent inside the thinking block. Fix: capture both `token` AND `reasoning` payloads into `full_text` so the SHA-12 output-hash compares cover reasoning + answer; bump `DEFAULT_MAX_TOKENS` from 96 → 512 so a 4B-class reasoning model has room for both the think block and a final answer. Same commit also corrected three stale paths in the runner — endpoint `/api/generate/stream` → `/api/chat/generate/stream` (route moved under `/api/chat/`); `load_resp.loadedModel.cacheStrategy` lookup updated to walk `runtime.loadedModel` (response shape nests under `runtime`); the FU-030 legacy-alias assertion now compares against `actual_strategy` extracted from the new path. Final `--quick` result: 6/6 pass on the runnable cells, 9 skip (all `quick=False` deferrals for vLLM / MTPLX / GGUF-MTP / DFlash / DDTree / TurboQuant-GGUF). Pre-build gate untouched (it doesn't run the matrix).

**Notes:** Trigger / Condition column — closed inline.

</details>

<details>
<summary>FU-047 · GGUF MTP speculative decoding via llama.cpp PR #22673 — Shipped 2026-05-16</summary>

**Status or trigger:** **Shipped 2026-05-16.** All six plan items landed: `--spec-type draft-mtp --spec-draft-n-max <N>` wired into [LlamaCppEngine._build_command](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py) gated on `_llama_server_supports("--spec-type")` + new `is_mtp_gguf_repo` helper in [_mtp.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_mtp.py); 4 aliases added (2 canonical `ggml-org/...` + 2 author preview `am17an/...`); new `ggufMtpAvailable` capability flag in [base.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/base.py) + detection probe in [capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/capabilities.py); catalog entries for `ggml-org/Qwen3.6-27B-MTP-GGUF` (29 GB Q8_0) + `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` (37 GB Q8_0 MoE) in [text_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/text_models.py); runtimeNote surfaces D2H prompt-processing caveat; 5 new unit tests in [test_inference.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_inference.py). FU-028 stays open for the MLX side. Research: [docs/UPSTREAM_RESEARCH_2026-05-16.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/UPSTREAM_RESEARCH_2026-05-16.md).

**Notes:** Original deferral notes: Closes the GGUF half of FU-028 (which was deferred waiting on this exact PR). llama.cpp PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) by am17an merged at 2026-05-16T12:06:24Z (merge commit `2555826`), shipping `--spec-type draft-mtp --spec-draft-n-max N` for any model with baked-in MTP heads. Upstream-reported acceptance ~72% with N=3 on Qwen3.6-27B, ~2× tok/s vs no-spec baseline. Canonical MTP GGUFs published under the `ggml-org/` HF org: `ggml-org/Qwen3.6-27B-MTP-GGUF` and `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` (each with BF16 + Q8_0 + mmproj; both verified HTTP 200 + sibling list via HF API). The MLX lane via MTPLX (separate `mtplx_engine.py` subprocess + `MTP_MODEL_MAP` in [_mtp.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_mtp.py)) was already shipping; this row wires the GGUF lane. Plan: (1) emit `--spec-type draft-mtp` + `--spec-draft-n-max <get_mtp_draft_n>` in `LlamaCppEngine._build_command` when the canonical repo is in `MTP_MODEL_MAP` and the binary supports `--spec-type` (probe via existing `_llama_server_supports` helper); (2) extend `_MTP_ALIASES` with the 4 new GGUF repos (2 canonical `ggml-org/...` + 2 author preview `am17an/...`); (3) add a `ggufMtpAvailable` flag to `inference/capabilities.py` so the UI can show MTP-via-GGUF separately from MTP-via-MTPLX; (4) add catalog entries under the `qwen3.6` family in [text_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/text_models.py); (5) `pre-build-check` asserts the bundled `llama-server` exposes `--spec-type`; (6) E2E matrix runner picks up a new `gguf+mtp` cell. Upstream caveats to surface in runtimeNote: prompt-processing speed takes a hit due to D2H embedding transfers; `--parallel >1` decoding is not yet fully optimised (our default `--parallel 1` is unaffected). FU-028 stays open for the MLX side until upstream `mlx-lm` lands a native MTP head loader OR MTPLX exposes an in-process Python API (the HTTP-server-only path makes it unsuitable for our FastAPI backend chaining). Research write-up at [docs/UPSTREAM_RESEARCH_2026-05-16.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/UPSTREAM_RESEARCH_2026-05-16.md).

</details>

<details>
<summary>FU-041 · Qwen3-Coder-Next-MLX-4bit was mis-canonicalised as Qwen3.6-27B-4bit — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** User-spotted mismatch: their local install at `/Users/dan/AI_Models/lmstudio-community/Qwen3-Coder-Next-MLX-4bit` was surfacing as canonical repo `mlx-community/Qwen3.6-27B-4bit` in the diagnostics snapshot, picking up the wrong catalog row and the wrong DFlash drafter. Inspecting the on-disk `config.json` confirmed the model is **Qwen3-Next** (architectures `Qwen3NextForCausalLM`, `model_type: "qwen3_next"`, sparse MoE with 512 experts, hidden_size 2048, ~3B active per token) — fundamentally different from the dense Qwen3.6-27B (`qwen3` arch, hidden_size 5120). Root cause: there was no catalog variant for the lmstudio-community community MLX 4-bit conversion of Coder-Next, so the fuzzy matcher in `src/utils/library.ts::libraryVariantMatchScore` settled for the closest "MLX + 4-bit + Qwen3" entry, which happened to be the unrelated `mlx-community/Qwen3.6-27B-4bit` row. Fix: (1) added an explicit `lmstudio-community/Qwen3-Coder-Next-MLX-4bit` variant to the `qwen3-coder-next` family in `backend_service/catalog/text_models.py` with the correct params (80B sparse, ~45 GB on disk, qwen3_next family capabilities). (2) Reverted the FU-038 DFlash aliases that wrongly pointed `mlx-community/Qwen3.6-27B-4bit / bf16 / 8bit` at `Qwen/Qwen3-Coder-Next` — those quants are the dense 27B Coder and have no drafter today. (3) Replaced them with the correct `lmstudio-community/Qwen3-Coder-Next-MLX-4bit` alias plus an `-Instruct` sibling for completeness. New regression tests in `tests/test_dflash.py` pin both the new alias resolution and that the dense 27B-4bit MUST NOT alias to the MoE drafter.

</details>

<details>
<summary>FU-040 · Tool-call parser misses open-only &lt;tool_call&gt; + Qwen3.6-27B false-positive vision tag — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** Surfaced by a Coder-Next chat session: tool calls rendered as raw `<tool_call>{"name": "web_search", ...}` text in the assistant bubble with no execution, while in a separate turn the "Attach image" affordance appeared even though Qwen3.6-27B is text-only. Three fixes. (1) **Tool-call parser widened.** Old regex `<tool_call>\s*(\{.*?\})\s*</tool_call>` required a closing tag and only matched objects. Coder-Next emitted three real-world shapes in a single session: canonical (closed + object), open-only (no `</tool_call>`), and array-shaped (model hallucinated a list of pseudo-results). The new parser uses `json.JSONDecoder.raw_decode` on each `<tool_call>` opener so it consumes the next valid JSON value regardless of close tag, dispatches objects with a `name`, drops list payloads silently, and continues scanning so a later well-formed call in the same message still lands. 7 new unit tests in `tests/test_agent.py` pin all three shapes plus the OpenAI-style stringified-arguments path. (2) **`_strip_tool_call_xml` helper** removes the JSON region the parser consumed from `result.text` before the streaming layer hands it to the chat bubble — fixes the "raw XML next to the ToolCallCard" duplication. Applied in both `run_agent_loop` and `run_agent_loop_streaming`. 6 new unit tests pin the strip behaviour. (3) **Qwen3.6-27B + Qwen3.5 catalog cleanup.** Dense Qwen3.6-27B (Coder-Next branding), Qwen3.6-27B-FP8, mlx-community/Qwen3.6-27B-4bit, and the family-level Qwen3.6 + Qwen3.5 entries all carried the `vision` capability — a copy-paste bug from when the catalog was scaffolded. Vision lives on a separate `Qwen3.6-27B-VL` variant we do not yet ship; the stale tag was promoting `supportsVision: true` for every community quant, making `ChatComposer` render the "Attach image" affordance for a text-only model. Dropped the tag from all five entries.

</details>

<details>
<summary>FU-039 · Tool-call arguments: null bricks Chat tab forever — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** Caught by the FU-037 ErrorBoundary: Coder-Next + Tools + `What is 17 * 23 plus sqrt(144)?` triggered `TypeError: Object.entries requires that input parameter not be null or undefined` in `ToolCallCard` (minified `_Y`). Root cause traced through the boundary's component stack (`_Y` → Panel `<section>` → ErrorBoundary → workspace) and the minified source: `src/components/ToolCallCard.tsx:116` did `Object.entries(toolCall.arguments)`, but Coder-Next emits `{"arguments": null}` for tool calls that need no parameters. `backend_service/agent.py::_execute_tool_call` then evaluated `isinstance(None, str) → False` and set `arguments = None`, which serialised into the persisted session. Every subsequent render of that turn crashed the Chat tab — the user could not even read prior history because the boundary fires before any other content renders. Two-layer fix: (1) backend `_execute_tool_call` now coerces `None` / empty-string / non-dict shapes to `{}` at the source so the contract "`arguments` is always a dict" holds for all consumers; (2) frontend `ToolCallCard` adds a defensive guard that defaults to `{}` and renders `(no arguments)` for genuinely corrupt records (so old sessions stop crashing without a manual localStorage wipe). 4 new unit tests in `tests/test_agent.py` pin all four null-ish input shapes.

</details>

<details>
<summary>FU-038 · Diagnostics cleanup: _free_bytes import, MallocStackLogging spam, Qwen3.6-27B alias — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** Three bugs surfaced by the live ``/api/diagnostics/snapshot`` payload from a Coder-Next + Tools repro. (1) ``backend_service/routes/diagnostics.py`` imported ``_free_bytes`` from ``backend_service.routes.setup``, but the setup package's ``__init__.py`` did not re-export it from ``gpu_bundle.py`` — the snapshot's ``extras`` section reported ``ImportError: cannot import name '_free_bytes'``. Added the re-export. (2) macOS hardened-runtime spawned every Python subprocess with three lines of ``MallocStackLogging: can't turn off malloc stack logging because it was not enabled.`` spam (we ship ``bundle.macOS.hardenedRuntime: true``). Hundreds per minute under the metrics poll, drowning out real INFO/ERROR lines. Fixed at source by ``command.env_remove("MallocStackLogging" / "MallocStackLoggingNoCompact" / "MallocScribble")`` in ``src-tauri/src/backend.rs`` so new builds don't produce the spam. Also added a regex filter (``_LOG_NOISE_PATTERNS`` + ``_filter_log_noise``) in ``diagnostics.py`` so the ``/api/diagnostics/log-tail`` and snapshot endpoints strip the spam from logs produced by older builds too — existing installs see a clean diagnostic surface without rebuilding. Filter reads 4× the requested line window so 200 useful lines survive even when the raw log is 50% spam. (3) Qwen3-Coder-Next was rebranded ``Qwen3.6-27B`` upstream; lmstudio-community MLX conversion's HF metadata reports ``mlx-community/Qwen3.6-27B-4bit`` as the canonical repo. ``model_resolution.resolve_dflash_target_ref`` prefers canonical, so ``DRAFT_MODEL_MAP`` missed and the runtimeNote said *DFLASH unavailable for 'mlx-community/Qwen3.6-27B-4bit': no compatible draft model is registered.* Aliased the three quant variants (4bit / bf16 / 8bit) back to ``Qwen/Qwen3-Coder-Next`` so the existing ``z-lab/Qwen3-Coder-Next-DFlash`` drafter resolves. New unit test pins the mapping.

</details>

<details>
<summary>FU-037 · Per-tab ErrorBoundary + Tauri devtools in release builds — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** A tool-call in the Chat tab against `Qwen3-Coder-Next` blanked the entire packaged macOS app — webview reload returned the user to the Dashboard, and any subsequent Chat navigation crashed again. Root cause: the React tree had no error boundary, so a single uncaught render error in one tab tore down the whole `<main>` content frame. Release builds also did not ship the WebKit inspector, so the user could not pull a stack trace without rebuilding via `cargo tauri dev`. (1) New [src/components/ErrorBoundary.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/ErrorBoundary.tsx) — `getDerivedStateFromError` + `componentDidCatch` capture the error, render an inline fallback with the error message, JS stack, component stack, "Try again" reset, and "Copy details" clipboard button. Wrapped around `{content}` in [src/App.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/App.tsx) keyed by `activeTab` so switching tabs is its own recovery path. (2) `src-tauri/Cargo.toml` `tauri` dep gains the `devtools` Cargo feature so right-click → Inspect Element opens WebKit devtools in release builds. (3) CSS for `.error-boundary` lives next to the existing notice banners in [src/styles.css](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/styles.css) — same colour vocabulary. Unit tests in [src/components/__tests__/ErrorBoundary.test.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/__tests__/ErrorBoundary.test.ts) pin the static-derive contract so the boundary cannot silently stop catching errors. Frontend errors land in the webview console; backend errors land in the Diagnostics tab + the in-memory `app.state.chaosengine` log buffer.

</details>

<details>
<summary>FU-034 · Hide unrecoverable launch-modal options instead of greying them out — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** The launch settings panel ([src/components/RuntimeControls.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/RuntimeControls.tsx)) used to render every cache-strategy card and the DFlash speculative-decoding toggle for every model + engine combo, with disabled checkboxes + "N/A" badges when an option could not run. That taught users the wrong thing — a disabled card with no install button suggests something they could fix, when the only fix lived outside the app or did not exist at all. New rule: **hide options the user has no in-app path to recover.** (1) Cache-strategy cards now skip render when the strategy is engine-incompatible (e.g. TriAttention selected on the MLX engine — engine mismatch is fundamental, no install button helps) or when the strategy needs the turbo binary on a GGUF backend without `llama-server-turbo` present (only fix is `scripts/build-llama-turbo.sh` outside the app). (2) The DFlash toggle hides entirely when the selected model has no draft in [`DRAFT_MODEL_MAP`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/dflash/__init__.py) or the engine is GGUF (DFlash needs MLX/vLLM). The "DFlash package not installed but model would be supported" case stays visible — the install button gets the user to ready in one click. ``native`` always survives. Hardcoded `f825ffb` install hint string in the DFlash help panel was the same drift bug from FU-033 — fixed alongside (now `fada1eb`). The popover-side filter ([src/components/kvStrategyFilter.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/kvStrategyFilter.ts)) already followed this rule, so the modal now matches.

</details>

<details>
<summary>FU-033 · dflash-mlx pin sync assert in pre-build-check — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** Caught a real bug: [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) and [scripts/stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) had drifted to different `dflash-mlx` commit hashes (the dev `.venv` ran 0.1.5.1 while `npm run stage:runtime` was bundling 0.1.4.1 into release builds). Both files manually synced to `fada1eb`; new probe in [scripts/pre-build-check.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/pre-build-check.mjs) and [scripts/pre-build-check.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/pre-build-check.sh) regex-extracts the commit hash from both files and fails the build when they diverge. Same probe also took the chance to drop the orphan `vendor/ChaosEngine` staleness check from both runners — that vendored path was dropped in FU-030 and would never resolve again.

</details>

<details>
<summary>FU-031 · Extend DRAFT_MODEL_MAP for new z-lab DFlash drafters + pin TriAttention — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** z-lab published draft checkpoints for several new families since the last `DRAFT_MODEL_MAP` audit; the upstream `dflash-mlx` 0.1.5 release also added the Gemma4 backend (commit 05cc456). Added entries for `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.5-122B-A10B`, `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.7`, `moonshotai/Kimi-K2.6` (all in [dflash/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/dflash/__init__.py)) plus `mlx-community/...` aliases for each so Apple Silicon quants resolve. New 7 unit tests in [tests/test_dflash.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_dflash.py) pin the mappings. **Same commit also pinned TriAttention** to `c3744ee6a50522a1559a577f85aef2b165a344f2` in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) — previously the `[triattention]` and `[triattention-mlx]` extras pulled `git+...git` HEAD, which made fresh installs non-reproducible whenever the upstream landed unreleased work. Pin matches the v0.2.0 release surface plus the AMD GPU port.

</details>

<details>
<summary>FU-030 · Drop ChaosEngine + RotorQuant strategy slots — Shipped 2026-05-10.</summary>

**Status or trigger:** **Shipped 2026-05-10.**

**Notes:** ChaosEngine (cryptopoly/ChaosEngine — 1 commit upstream, eclipsed by KVTC at ICLR 2026 with the same PCA approach but 8–32× compression vs 3.7×) and RotorQuant (shipped as a misleading alias for TurboQuant — same ``--cache-type-k turbo{N}`` flags + same Python module marker) both removed from the registry. Persisted user configs that still reference these ids coerce silently to ``turboquant`` via a new ``CacheStrategyRegistry.resolve_legacy_id`` helper + module-level ``_LEGACY_STRATEGY_ALIASES`` map ([cache_compression/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/__init__.py)). Mirror coercion in frontend ([src/components/runtimeSupport.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/runtimeSupport.ts) ``LEGACY_STRATEGY_ALIASES`` + ``canonicalStrategyId``). Two-level llama.cpp fallback chain (was three-level: requested → ChaosEngine → native; now requested → native) in [backend_service/inference/llama_cpp_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py). Vendored ChaosEngine bundling stripped from [scripts/stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) (3 helper functions removed: ``stageVendoredChaosEngine`` + ``ensureSetuptoolsForPep639`` + ``resolveChaosEngineVendor``). Pre-build probe asserts the legacy-id coercion works in CI. ``[rotorquant]`` extra removed from [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml). ``CHAOSENGINE_VENDOR_PATH`` env var dropped. Cache strategy speed/quality maps in [helpers/cache.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/cache.py) trimmed to remaining strategies.

</details>

<details>
<summary>FU-026 · TaylorSeer + DBCache aggressive cache preset — Obsoleted 2026-05-03 by diffusers 0.38 core.</summary>

**Status or trigger:** **Obsoleted 2026-05-03 by diffusers 0.38 core.**

**Notes:** Diffusers 0.38.0 (2026-05-01) ships ``TaylorSeerCacheConfig``, ``MagCacheConfig``, ``PyramidAttentionBroadcastConfig``, ``FasterCacheConfig`` natively — no ``cache-dit`` dependency required. Wired as registry strategies (ids ``taylorseer``, ``magcache``, ``pab``, ``fastercache``) in [cache_compression/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/__init__.py). Each adapter calls ``pipeline.transformer.enable_cache(<Config>)``. UNet pipelines (SD1.5/SDXL) raise ``NotImplementedError`` into a runtimeNote, matching the FBCache contract. MagCache is FLUX-only without calibration UX (uses ``FLUX_MAG_RATIOS`` from ``diffusers.hooks.mag_cache``); other DiTs raise a "calibration required" message until that UX lands.

</details>

<details>
<summary>FU-025 · mlx-video Wan one-shot convert action — Fully shipped 2026-05-04 (Phase 7 + Phase 8 + Phase 9).</summary>

**Status or trigger:** **Fully shipped 2026-05-04 (Phase 7 + Phase 8 + Phase 9).**

**Notes:** Closes FU-009 Wan branch. **Phase 7 (foundation):** `[mlx-video]` extra in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) flipped to ``git+https://github.com/Blaizzy/mlx-video.git``. Helper [backend_service/mlx_video_wan_convert.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_video_wan_convert.py) wraps the upstream `python -m mlx_video.models.wan_2.convert` subprocess: `slug_for(repo)` / `output_dir_for(repo)` / `status_for(repo)` / `list_converted()` / `run_convert(checkpoint_dir, repo, dtype, quantize, bits, group_size, timeout)`. Output under ``~/.chaosengine/mlx-video-wan/<slug>/`` (override via ``CHAOSENGINE_MLX_VIDEO_WAN_DIR``). **Phase 8 (routing):** [mlx_video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_video_runtime.py) `supported_repos()` returns dynamic union of LTX-2 + converted-on-disk Wan repos. `_REPO_ENTRY_POINTS` adds `"Wan-AI/": "mlx_video.models.wan_2.generate"`. `_build_wan_cmd` produces the Wan-shaped CLI (`--model-dir`, `--guide-scale` string, `--scheduler`, optional `--seed`/`--steps`/`--negative-prompt`; no LTX-2 flags). `generate()` picks `_wan_runtime_note` (flags MoE experts) and skips LTX-2 effective-step / effective-guidance overrides. **Phase 9 (GUI):** Orchestrator [backend_service/mlx_video_wan_installer.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_video_wan_installer.py) drives preflight → download-raw → convert → verify with structured progress events. Setup endpoints in [routes/setup.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup.py): `POST /api/setup/install-mlx-video-wan` (background-job pattern mirroring `/api/setup/install-longlive`), `GET /api/setup/install-mlx-video-wan/status`, `GET /api/setup/mlx-video-wan/inventory`. Frontend client in [src/api.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/api.ts) (`startWanInstall`, `getWanInstallStatus`, `getWanInventory`). UI panel [src/components/WanInstallPanel.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/WanInstallPanel.tsx) lists every supported Wan repo with raw-size hint + converted badge / install button + live `InstallLogPanel` underneath; rendered in [VideoDiscoverTab.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/video/VideoDiscoverTab.tsx) above the variant grid. Supported raw repos: `Wan-AI/Wan2.{1-T2V-1.3B,1-T2V-14B,2-TI2V-5B,2-T2V-A14B,2-I2V-A14B}`. End-to-end UX: user clicks Install → backend downloads + converts in background → runtime auto-detects + routes Wan generate calls through mlx-video. Tests: 21 in [test_mlx_video_wan_convert.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_video_wan_convert.py), 9 Wan-routing in [test_mlx_video.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_video.py), 15 in [test_mlx_video_wan_installer.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_video_wan_installer.py).

</details>

<details>
<summary>FU-022 · LLM-based prompt enhancer — Shipped 2026-05-04 (Apple Silicon path).</summary>

**Status or trigger:** **Shipped 2026-05-04 (Apple Silicon path).**

**Notes:** Replaces the deterministic per-family template-suffix enhancer in `_enhance_prompt`. Helper [backend_service/helpers/prompt_enhancer.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/prompt_enhancer.py) wraps `mlx_lm.load` + `mlx_lm.generate` against a small instruct model (default `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, ~700 MB on disk, ~3s cold load + sub-second per call) — cached in a process-level `_EnhancerSingleton` so the second call onward hits the warm model. Per-family system prompts (`wan` / `ltx` / `hunyuan` / `flux` / `sdxl` / `sd3` / `default`) anchor the rewrite to the DiT's training distribution. `family_for(repo)` matches longest-prefix-wins. Endpoint `POST /api/prompt/enhance` ([routes/prompts.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/prompts.py)) returns `{enhanced, note, modelUsed, family}`. Frontend exposes a "Enhance" pill button next to the Prompt label in both Studio tabs ([components/PromptEnhanceButton.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/PromptEnhanceButton.tsx)) — click triggers the rewrite + replaces the textarea on success or surfaces a tooltip note when the enhancer fell back. Failure modes (non-Apple platform, mlx_lm missing, model not cached, generation crash, shorter-than-input rewrite) all return the original prompt + a runtimeNote so the user sees why. Live smoke 2026-05-04: 6-word "a fluffy cat on a windowsill" → 16-word FLUX rewrite (3.2s cold), 13-word Wan rewrite (0.12s warm), 8-word LTX rewrite (0.11s warm). 16 unit tests covering family-mapping + happy path + load-failure + generation crash + shorter-rewrite reject + quote stripping. CUDA / Linux still get the legacy template suffix; the helper returns the original + a "requires Apple Silicon" runtimeNote on those platforms.

</details>

<details>
<summary>FU-021 · Image-runtime CFG decay parity — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** `cfgDecay` field on `ImageGenerationConfig` + `ImageGenerationRequest`. Linear ramp from initial guidance to 1.5 floor inside the existing `callback_on_step_end` in `generate()`. Gated to flow-match repos (`_is_flow_matching_repo`); SD1.5/SDXL ignore the flag. Default off — opt-in vs. video runtime's default-on.

</details>

<details>
<summary>FU-020 · AYS (Align Your Steps) schedule for SD/SDXL — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** New samplers `ays_dpmpp_2m_sd15` / `ays_dpmpp_2m_sdxl` in `_SAMPLER_REGISTRY` ([image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py)). Private `_ays_family` token stripped from `from_config` kwargs and stashed on `pipeline._chaosengine_ays_timesteps`; `_build_pipeline_kwargs` passes it via `timesteps=` and pops `num_inference_steps`. Hardcoded NVIDIA timestep arrays for SD1.5/SDXL/SVD. Flow-match models continue to be gated out by `_is_flow_matching_repo`.

</details>

<details>
<summary>FU-019 · Distill LoRA support (Hyper-SD, FLUX.1-Turbo, lightx2v Wan CausVid) — Shipped 2026-05-03</summary>

**Status or trigger:** **Shipped 2026-05-03; extended Phase 3 with Wan2.2-Distill.**

**Notes:** LoRA load + fuse path in both [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) and [video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime.py) `_ensure_pipeline`. Catalog variants in [catalog/image_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/image_models.py) (FLUX.1-dev × Hyper-SD-8step + Turbo-Alpha) and [catalog/video_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/video_models.py) (Wan2.1 1.3B/14B × CausVid). **Phase 3 extension: Wan 2.2 A14B I2V × lightx2v 4-step distill.** lightx2v ships full distilled transformers (not LoRAs) for both Wan2.2 MoE experts. New `distillTransformer*` fields on `VideoGenerationConfig` carry repo + high/low-noise filenames + precision (`bf16` / `fp8_e4m3` / `int8`). `_swap_distill_transformers` helper downloads both safetensors via `huggingface_hub.hf_hub_download`, loads via `WanTransformer3DModel.from_single_file`, and reassigns `pipeline.transformer` + `pipeline.transformer_2`. Variant key includes the distill identity so switching variants triggers clean rebuilds. Distill takes precedence over LoRA when both are pinned. Catalog adds: `Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-bf16` + `-distill-fp8`. Schema-default substitution sets `defaultSteps=4` + `cfgOverride=1.0`.

</details>

<details>
<summary>FU-018 · TAEHV / TAESD preview decoder — Fully shipped 2026-05-04 (parts 1 + 2).</summary>

**Status or trigger:** **Fully shipped 2026-05-04 (parts 1 + 2).**

**Notes:** Tiny VAE for cheap preview decode each step. **Part 1 — full-decode VAE swap** ([backend_service/helpers/preview_vae.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/preview_vae.py)) maps repo → preview VAE id (FLUX.1/2 → taef1/taef2, SD3 → taesd3, SDXL incl. sdxl-turbo + SDXL-Lightning → taesdxl, SD1.x/2.x incl. sd-turbo → taesd, Wan2.x → taew2_2, LTX-Video / LTX-2 → taeltx2_3_wide, HunyuanVideo → taehv1_5, CogVideoX → taecogvideox, Mochi → taemochi, Qwen-Image → taeqwenimage). `maybe_apply_preview_vae(pipeline, repo, enabled)` swaps `pipeline.vae` for an `AutoencoderTiny`, mirrors the stock VAE's dtype + device (live-validated against SDXL-Turbo on MPS — without the device mirror the first decoder pass raises `MPSHalfType` vs `torch.HalfTensor`). **Part 2 — live per-step thumbnails** ([backend_service/helpers/preview_thumbnails.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/preview_thumbnails.py)) decodes `callback_kwargs["latents"]` through the swapped tiny VAE inside `callback_on_step_end`, scales to ≤192 px, base64-encodes a PNG, publishes to `IMAGE_PROGRESS.set_thumbnail` / `VIDEO_PROGRESS.set_thumbnail`. Stride caps emit count at ~8 (image) / ~6 (video) per gen so the polled `/api/{images,video}/progress` endpoint stays cheap. Handles both standard 4D `(B, C, H, W)` latents (SD1.5 / SDXL / SD3) and FLUX's packed 3D `(B, seq_len, 64)` shape via `pipeline._unpack_latents` (live-validated against FLUX.1-schnell on MPS — 4 thumbnails captured per 4-step gen, all valid base64 PNGs at 192x192). Frontend reads `snapshot.thumbnail` from `useGenerationProgress`, renders inside `LiveProgress` between the bar and the phase list when present. Errors are best-effort: a decode crash never aborts the actual generation — caller catches and falls back to no-thumbnail. **LTX refiner private-kwarg fix:** the FU-018 part 2 wiring also caught + fixed a pre-existing leak where `_invoke_pipeline_with_ltx_refiner` was passing `__cfg_decay` directly into `LTXPipeline.__call__` (would have started leaking `__preview_vae` too). Both private kwargs now stripped in the refiner path.

</details>

<details>
<summary>FU-017 · SDXL VAE fp16 fix on MPS / CUDA — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** Probes `madebyollin/sdxl-vae-fp16-fix` snapshot via `local_files_only=True` (no surprise download) at pipeline load. When cached, swaps `pipeline.vae` and lets `_preferred_torch_dtype` stay on fp16 for SDXL on MPS — drops the previous fp32 fallback that doubled wall-time on Apple Silicon. Helpers `_is_sdxl_repo` + `_locate_sdxl_vae_fix_snapshot` in [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py). Falls back to stock VAE + fp32 on any failure.

</details>

<details>
<summary>FU-016 · SageAttention CUDA backend wiring — Shipped 2026-05-03 (CUDA-gated).</summary>

**Status or trigger:** **Shipped 2026-05-03 (CUDA-gated).**

**Notes:** Helper at [backend_service/helpers/attention_backend.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/attention_backend.py) (`maybe_apply_sage_attention`). Called from both [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) and [video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime.py) `_ensure_pipeline` after pipeline build. CUDA + sageattention pip wheel + diffusers ≥0.36 + DiT pipeline. No-op on macOS / CPU / UNet / non-DiT pipelines. Stacks multiplicatively with FBCache (community Wan2.1 720P cumulative 54%). Setup-page install action (`pip install sageattention`) follows.

</details>

<details>
<summary>FU-015 · First Block Cache (diffusers 0.36 generic hook) — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** Cross-platform diffusion cache strategy backed by `diffusers.hooks.apply_first_block_cache`. Lives at [cache_compression/firstblockcache.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/firstblockcache.py), registered as id `fbcache` in the strategy registry ([cache_compression/__init__.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/__init__.py)). Applies to image + video DiTs (FLUX, SD3.5, Wan2.1/2.2, HunyuanVideo, LTX-Video, CogVideoX, Mochi). Default threshold 0.12 (≈1.8× speedup on FLUX.1-dev with imperceptible quality drift). Same `apply_diffusion_cache_strategy` hook as TeaCache; UNet pipelines (SD1.5/SDXL) raise NotImplementedError into a runtimeNote. Closes FU-007.

</details>

<details>
<summary>FU-014 · LLM-based prompt enhancer — Closed 2026-05-04 by FU-022.</summary>

**Status or trigger:** **Closed 2026-05-04 by FU-022.**

**Notes:** Replaced by FU-022's MLX-native enhancer (see below).

</details>

<details>
<summary>FU-009 · mlx-video (Blaizzy) Apple Silicon video engine — Fully shipped 2026-05-04</summary>

**Status or trigger:** **Fully shipped 2026-05-04. Live smoke validated end-to-end.**

**Notes:** LTX-2 paths (`prince-canuma/LTX-2-{distilled,dev,2.3-distilled,2.3-dev}`) routed through subprocess engine in [backend_service/mlx_video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_video_runtime.py); Wan-AI paths route via Phase 8 of FU-025 (`_is_wan_repo` + `_build_wan_cmd` + `_REPO_ENTRY_POINTS["Wan-AI/"] = "mlx_video.models.wan_2.generate"`). Live smoke 2026-05-04 against `Wan-AI/Wan2.1-T2V-1.3B` (480×272, 5 frames, 4 steps, unipc): T5 encode 14.1s + transformer load 0.2s (4-bit q) + denoise 2.9s @ 1.4 it/s + VAE decode 1.3s = 19.6s total, 383 KB .mp4 output. The smoke also surfaced + fixed a `status_for` filename gap — mlx-video upstream emits root-level `model.safetensors` + `t5_encoder.safetensors`, not the legacy `transformer*.safetensors` / `text_encoder*.safetensors` patterns the helper originally checked for. Both now match.

</details>

<details>
<summary>FU-008 · stable-diffusion.cpp engine (cross-platform diffusion) — Shipped 2026-05-03 (video) + 2026-05-04 (image).</summary>

**Status or trigger:** **Shipped 2026-05-03 (video) + 2026-05-04 (image).**

**Notes:** Binary build via [scripts/build-sdcpp.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/build-sdcpp.sh) + [scripts/update-sdcpp.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/update-sdcpp.sh) (clones to `/tmp/stable-diffusion.cpp`, cmake `-DSD_METAL=ON` on Darwin or `-DSD_CUBLAS=ON` on Linux+CUDA, installs to `~/.chaosengine/bin/sd`). Build target is `sd-cli` (renamed from `sd` upstream around master-590); installer copies it back to the legacy `sd` filename so downstream resolvers in [sdcpp_video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_video_runtime.py), [sdcpp_image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_image_runtime.py), and [stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) keep working. Path resolution in [src-tauri/src/lib.rs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/lib.rs). **Video lane** (`SdCppVideoEngine.generate`): subprocess spawn → maps `VideoGenerationConfig` → sd.cpp flags (`--diffusion-model`, `-p`, `-W/-H`, `--steps`, `--cfg-scale`, `--seed`, `-o`, `--video-frames`, `--fps`, `--negative-prompt`); regex-parses `step N/M` (or `[N/M]`) into `VIDEO_PROGRESS`; reads `.webm` bytes back (sd.cpp's video output is `.webm`/`.avi`/animated `.webp` — no native `.mp4`). Catalog requires `ggufRepo` + `ggufFile` pin (e.g. `QuantStack/Wan2.2-TI2V-5B-GGUF`). **Image lane** (`SdCppImageEngine.generate`, [sdcpp_image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_image_runtime.py)): mirrors video shape but emits PNG, drops `--video-frames`/`--fps`, batches by looping seeds (sd.cpp renders one image per invocation). Manager dispatch in [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) `ImageRuntimeManager.generate` routes when `config.runtime == "sdcpp"`, falls through to diffusers on probe failure or runtime error. Catalog variants: `FLUX.1-schnell-sdcpp-q4km` + `FLUX.1-dev-sdcpp-q4km` ([catalog/image_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/image_models.py)). Supported image repos: FLUX.1/2 family, SD3.5, SDXL, SD2.1, Qwen-Image (+ 2512), Z-Image (+ Turbo).

</details>

<details>
<summary>FU-007 · TeaCache for Wan2.1/2.2 — Obsoleted 2026-05-03 by FU-015.</summary>

**Status or trigger:** **Obsoleted 2026-05-03 by FU-015.**

**Notes:** TeaCache patches for FLUX + HunyuanVideo + LTX-Video + CogVideoX + Mochi remain under [cache_compression/_teacache_patches/](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/cache_compression/_teacache_patches/). The Wan-specific port that was deferred here is no longer needed: diffusers 0.36 ships a model-agnostic `apply_first_block_cache` hook (FU-015) that operates on `pipeline.transformer` regardless of model, so Wan caches via the same generic strategy without a vendored forward. Pick FBCache for Wan; TeaCache stays available as the alternative for FLUX-family pipelines.

</details>

<details>
<summary>FU-006 · Re-verify dflash-mlx pin — Bumped to fada1eb (HEAD) on 2026-05-10</summary>

**Status or trigger:** **Bumped to `fada1eb` (HEAD) on 2026-05-10. Previously bumped to `8d8545d` = v0.1.5.1 on 2026-05-05 after the ddtree.py rewrite landed.**

**Notes:** 2026-05-10 bump from `8d8545d` to `fada1eb` covers 11 upstream commits including the new Gemma4 DFlash backend (commit 05cc456, "feat: add Gemma4 DFlash backend"), the v0.1.5 serving surface, live server metrics endpoint, prefix-cache survival test gate, async L2 writer fix, long-context runtime diagnostics hardening, benchmark slugging fixes, and a license switch to Apache-2.0. Same fix applied in both [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) (already correct) and [scripts/stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) (was lagging on `f825ffb` v0.1.4.1 — staged release runtime would have shipped the old binary). The two pins now live as the same hex string in both files; CI's pre-build-check should grow a sync assert. No breaking API changes between the pins per upstream commit log. **Earlier bump notes:** Pin advanced from `f825ffb` (v0.1.4.1) to `8d8545d` (v0.1.5.1). 0.1.5+ moved every primitive that [backend_service/ddtree.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ddtree.py) consumed off the runtime top-level onto a per-family `target_ops` adapter — `target_forward_with_hidden_states` → `target_ops.forward_with_hidden_capture`, `extract_context_feature_from_dict` → `target_ops.extract_context_feature`, `make_target_cache` → `target_ops.make_cache`, `_target_embed_tokens` → `target_ops.embed_tokens`, `_target_text_model` → `target_ops.text_model`, `_lm_head_logits` → `target_ops.logits_from_hidden`. `ContextOnlyDraftKVCache` moved to `dflash_mlx.model`; `create_attention_mask` re-imported from `mlx_lm.models.base`; `trim_cache_to` was removed entirely and now lives as a thin local `_trim_cache_to` shim that calls each entry's own `.rollback()` / `.trim()` / `.crop()`. Adapter resolved once at the top of `generate_ddtree_mlx` via `resolve_target_ops(target_model)`. Live smoke 2026-05-05 against `mlx-community/Qwen2.5-0.5B-Instruct-4bit` confirmed adapter resolves (`backend=qwen_gdn`, `family=pure_attention`), forward+capture / embed_tokens / text_model / logits_from_hidden / extract_context_feature / `_trim_cache_to` all working. Gains over 0.1.4.1: draft model quantization with Metal MMA kernels, branchless Metal kernels + fused draft KV projections, long-context runtime diagnostics. Re-check cadence resets to quarterly.

</details>

<details>
<summary>FU-005 · arozanov v_only TurboQuant MLX mode — Dropped 2026-04-24</summary>

**Status or trigger:** **Dropped 2026-04-24**

**Notes:** Our current `turboquant-mlx-full` 0.1.3 path already runs without any mlx-lm fork — uses pip `TurboQuantKVCache` with `QuantizedKVCache` fallback ([turboquant_mlx/__init__.py:174-186](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/turboquant_mlx/__init__.py)). `VOnlyTurboQuantCache` is only in the arozanov fork (we track but don't consume). Value prop already satisfied; entry removed.

</details>

<details>
<summary>FU-002 · Wire TriAttention MLX compressor into mlx_worker — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** Unblocked by triattention 0.2.0's MLX port (RavenX AI, 2026-04-09): `apply_triattention_mlx(model, kv_budget=N)` operates on the model directly, bypassing the `mlx_lm.stream_generate` callback gap. Spike at [scripts/spike_triattention_mlx.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/spike_triattention_mlx.py) confirmed 2.63× speedup with identical output on Qwen2.5-0.5B-Instruct-4bit (norm-only scoring works without calibration stats). Wired into `WorkerState._apply_cache_profile` ([backend_service/mlx_worker.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker.py)) via a new `_apply_triattention_mlx_compressor` branch — when `cacheStrategy == "triattention"` the worker delegates to `cache_compression.registry.get("triattention").apply_mlx_compressor(model, kv_budget=self.kv_budget)`. `kvBudget` request param defaults to 2048; falls back to native cache on any failure (model None, registry missing, strategy unavailable, apply raises).

</details>

<details>
<summary>FU-001 · Bump turboquant to 0.3.x — Shipped 2026-05-03.</summary>

**Status or trigger:** **Shipped 2026-05-03.**

**Notes:** `turboquant-mlx-full` 0.3.0 published to PyPI; `[turboquant]` extra pin bumped from `>=0.1.3` to `>=0.3.0` in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml). Adds asymmetric K/V bits, layer-adaptive precision, `--no-quant` eval flag, NumPy 2.0 + transformers 5.x compat. Verified backward compatible — full ``test_cache_strategies.py`` + ``test_image_runtime.py`` + ``test_video_runtime.py`` (190 tests) pass against 0.3.0. The `turboquant` (HuggingFace) and `turboquant-mlx` (arozanov fork) packages stay on their existing pins; only the active `turboquant-mlx-full` path advances.

</details>
