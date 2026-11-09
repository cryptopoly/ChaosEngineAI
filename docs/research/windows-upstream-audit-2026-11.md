# Windows upstream audit (2026-11-09)

Read-only audit on the RTX 4090 PC (Windows 11, driver 591.86, Python 3.12.10), branch `test/rtx4090`. Sources: PyPI JSON, `download.pytorch.org` indexes, the GitHub API and release pages, `git ls-remote`, and the files installed on this PC. Nothing was installed or changed.


**Checked afterwards on the 4090 PC:** `git ls-remote` lists tag `b12233`; `llama-b12233-bin-win-cuda-13.5-x64.zip` answers 200 (171 MB) and the `13.4` name answers 404, so `llama_cuda.cuda_version_for` (returns "13.4") must change together with any pin bump. The repo dependency audit (`scripts/audit-dependencies.py`, osv.dev) passed the same day: no high or critical advisory in 1,006 pinned packages; 101 behind, 26 by a major version.

## Summary

The Python side of the app is mostly current. diffusers, accelerate, safetensors, tokenizers and pillow are at the latest release, and transformers is two patch releases behind. The native side is not. On Windows, chat runs on the bundled **Vulkan** llama-server, because the CUDA download path (`llama_cuda.py`) is Linux-only. The developer copy in `~/.chaosengine/bin/llama-server.exe` is build 9189 from May and has **no GPU backend** (`--list-devices` lists nothing), so dev runs on this PC chat on the CPU. The TurboQuant fork (May build) and stable-diffusion.cpp (May build) are about 2,000 and 486 commits behind. Both now publish Windows CUDA prebuilts, so no local CUDA build is needed. torch 2.11 is the last release with Windows cu128 wheels. Moving past it gains little on Ada GPUs, and it would break Nunchaku (no Windows wheel past torch 2.11) and torchaudio (frozen at 2.11). Hold torch at 2.11 and spend the effort on four things: a Windows CUDA llama-server, a llama.cpp bump, and one-click installs of the SageAttention and Nunchaku Windows wheels.

## 1. Native binaries

| Component | Where pinned | Our version | Latest | Windows / CUDA build available | What changed that matters | Upgrade? Effort |
|---|---|---|---|---|---|---|
| llama.cpp (bundled) | `scripts/stage-runtime.mjs` `LLAMA_CPP_TAG`, `scripts/update-llama-cpp.sh` `LLAMA_CPP_REF`, `backend_service/helpers/llama_cuda.py`, `scripts/linux-gpu-check.sh` | b11981 (2026-10-29). Windows bundle stages `bin-win-vulkan-x64` | b12233 (2026-11-09), 252 commits later | Yes, every tag: `win-cuda-12.4-x64`, `win-cuda-13.5-x64` (+ `cudart-*` zips), `win-vulkan-x64`, `win-rocm-10.0-x64`, `win-sycl-x64`, `win-openvino`, `win-cpu`. b11981 had `cuda-13.4`. | Windows unbuffered model load (#26014). CUDA: graphs no longer evicted early (#30662), batched tensor reads into mapped host memory (#31391), MoE layout for MMQ (#31047). Speculative decoding: MTP hidden state saved and restored (#30141), MTP reads target states from a device buffer (#31927), DFlash device copies (#31506), spec checkpoint restored for long drafts (#31350), MoE cache not applied to the draft (#31497). Server uses reference-counted checkpoints (#30137). Gemma 4 graph optimisations with "massive gains on pp for E2B and E4B" (#31356); Nemotron 3.5 Bolt is Gemma-4-E4B based. FP8 E4M3 quant (#27512). **Breaking for us:** CUDA 13 assets moved from 13.4 to 13.5 (#31788), and `llama_cuda.cuda_version_for` hard-codes `"13.4"`, so a pin bump breaks the Linux CUDA download unless that string changes too. | Yes. S (bump plus the 13.5 string) |
| llama.cpp CUDA on Windows | none (`llama_cuda.host_arch()` returns None off Linux) | Vulkan only | same tags | Yes, `win-cuda-13.5-x64` for driver 580+, `win-cuda-12.4-x64` for older drivers | On NVIDIA, CUDA is normally faster than Vulkan for prompt processing and supports all KV quant types with flash attention. UNVERIFIED on this PC: `~/.chaosengine/test-llama-b11981/{cuda,vulkan}` are both unpacked, so `llama-bench` both builds to get a number. | Yes. M (port `llama_cuda.py` to Windows .zip assets) |
| llama-server in `~/.chaosengine/bin` (this PC) | not pinned; Tauri dev and the backend resolve `~/.chaosengine/bin` first (`src-tauri/src/binaries.rs`, `inference/binaries.py` `_LLAMA_FALLBACK_DIRS`) | build 9189 (`64b38b561`, 2026-05-16). Only `ggml-cpu-*.dll`, no `ggml-cuda.dll` or `ggml-vulkan.dll`. `--list-devices` is empty. | b12233 | n/a | **Dev runs on this PC load chat models on the CPU.** Release builds are not affected: Tauri sets `CHAOSENGINE_LLAMA_SERVER` to the bundled binary. `~/.chaosengine/prebuilt-llama/win-x64/` also holds 28 old cached tags (b8857 to b9524). | Replace or delete (owner action). S |
| llama-server-turbo (TurboQuant fork) | `scripts/build-llama-turbo.{sh,ps1}`, `update-llama-turbo.{sh,ps1}`: branch `feature/turboquant-kv-cache`, unpinned | `69d8e4b` (built 2026-05-05, from `llama-server-turbo.version`) | Branch head `6439568` (2026-10-22). The branch was rebased: it is 2,007 commits ahead and our commit is no longer on it. Release `tqp-v0.4.0` (2026-09-28). | **Yes, a prebuilt exists:** `turboquant-plus-tqp-v0.4.0-windows-x64-cuda12.4.zip` (also v0.2.0 and v0.3.0). Licence MIT. UNVERIFIED that the `tqp` release is cut from `feature/turboquant-kv-cache`. | New `turbo5` / `turbo6` (TQ5_0 / TQ6_0) KV types with CUDA flash attention. Native turbo routing and width-gated MMA on CUDA. Batched host-to-device copies for deep-spill decode. kv-stream. CI builds CUDA 13 for sm_75 and sm_89 (the 4090). New cache types need registry entries before the UI can offer them. | Yes. M (prefer the prebuilt; pin a tag, not a branch) |
| stable-diffusion.cpp (`sd`) | `scripts/build-sdcpp.{sh,ps1}`, `update-sdcpp.{sh,ps1}`: branch `master`, unpinned | `3d6064b` (built 2026-05-05, from `sd.version`) | `a3c49b1` = release `master-1079-a3c49b1` (2026-11-09), 486 commits ahead | **Yes:** `sd-master-*-bin-win-cuda12-x64.zip` + `cudart-sd-bin-win-cu12-x64.zip`, plus Vulkan, ROCm and CPU. Licence MIT. | INT8 CUDA fusions (#2235). Packed INT4 with CUTLASS on CUDA (#2278). Z-Image fused ops and a faster BF16 GEMM (#2265). Built-in SageAttention CUDA code. FLUX.3 support. SeedVR2 and RelayVSR upscaling. ControlNet Union for SDXL. Fix for layer split across CUDA (#2262). CUTLASS is BSD-3 and must be added to `THIRD_PARTY_NOTICES.md` if we ship it. UNVERIFIED: the binary name inside the zip (`sd.exe` or `sd-cli.exe`). | Yes. S to M (download a pinned prebuilt instead of building master) |

## 2. Python pins: main app

`release-constraints.txt` is what release builds install. The GPU bundle (`routes/setup/gpu_bundle.py`) installs into `%LOCALAPPDATA%\ChaosEngineAI\extras\cp312` with floors only.

| Component | Where pinned | Our version | Latest | Win cp312 CUDA wheel | What changed that matters | Upgrade? Effort |
|---|---|---|---|---|---|---|
| torch | `release-constraints.txt` | 2.11.0 (cu128 on this PC) | 2.15.0 (2026-10-28) | cu128: last is 2.11.0 (2.12 dropped cu128). cu126 and cu130: up to 2.14.1. cu132: up to 2.15.0. cu124: stops at 2.6.0. | 2.12: `torch.accelerator.Graph`, `torch.cond` in CUDA graphs. 2.13: CuTeDSL Inductor backend (prototype). 2.14 and 2.15: NVGEMM, mostly NVFP4 on Blackwell. Little gain for Ada (sm_89). | **Hold at 2.11.** Nunchaku Windows wheels stop at torch 2.11 and torchaudio stops at 2.11. L when it is time. |
| torch in GPU bundle | `gpu_bundle.py` installs `torch>=2.4.0`; `_install_helpers.cuda_torch_indexes` tries cu130 first on driver 580+ | floor only | n/a | n/a | On this driver (591) the bundle would resolve the newest cu130 torch, **2.14.1**, not 2.11.0. That mismatches `release-constraints.txt` and every Nunchaku Windows wheel. This is inferred from the index contents; no install was run to confirm it. | Pin `torch==2.11.0` in the bundle. S |
| torchvision | constraints | 0.26.0 | 0.29.2 | follows torch | Tied to torch. | Hold with torch |
| torchaudio | voice lane, music venv | 2.9.1 (voice) | 2.11.1 PyPI; index wheels stop at 2.11.0 | cu128 2.11.0 yes | The index has no torchaudio for torch 2.12+. Anything needing torchaudio pins torch at 2.11 or lower. | Hold |
| diffusers | constraints | 0.42.1 | 0.42.1 | pure | Current. | No |
| transformers | constraints | 5.20.2 | 5.22.2 (2026-11-06) | pure | Same hub floor (`>=1.31,<3`). | Yes. S |
| accelerate | constraints | 1.15.0 | 1.15.0 | pure | Current. | No |
| huggingface_hub | constraints | 1.33.0 | 2.4.0 (2026-10-29) | pure | 2.0 (2026-09-24) moved to `httpx2` and removed deprecated APIs and CLI commands. 2.2 fixed `snapshot_download` offline with a pinned revision and a race that skipped Xet. transformers and diffusers allow `<3.0`. | Yes, as its own change: grep for removed APIs and `httpx` error types. M |
| hf-xet | constraints | 1.6.0 | 1.8.0 | yes | Download backend. | Yes. S |
| safetensors / tokenizers / pillow | constraints | 0.8.0 / 0.23.3 / 13.0.0 | same | yes | Current. | No |
| numpy | constraints | 2.4.4 | 2.5.4 | yes | Minor release. | Optional. S |
| gguf | constraints, bundle `>=0.10` | 0.18.0 | 0.19.0 | pure | Loader for GGUF DiTs. | Yes. S |
| fastapi / uvicorn / pydantic | constraints | 0.135.3 / 0.44.0 / 2.12.5 | 0.162.2 / 0.54.0 / 2.14.3 | pure | Routine. `scripts/audit-dependencies.py` covers advisories. | Routine. S |
| opencv-python | constraints | 4.13.0.92 | 5.0.0.93 | yes | Major version. | Hold. M |
| imageio-ffmpeg / protobuf / scipy / sentencepiece | constraints | 0.6.0 / 7.34.1 / 1.17.1 / 0.2.1 | 0.8.1 / 7.36.2 / 1.18.1 / 0.2.3 | yes | Routine. | Routine. S |

**Stale files on this PC.** `extras\cp312\site-packages` holds diffusers 0.38.0, transformers 5.7.0, bitsandbytes 0.49.2 and torchao 0.17.0 (2026-05-02). `extras\cp313` holds two `huggingface_hub` dist-infos (1.13 and 1.15), two `safetensors` dist-infos (0.7.0 and 0.8.0rc0) and a `.torch-rollback-2.6.0+cu124`. `extras\site-packages` holds diffusers 0.37.1 and transformers 5.5.4. The repo `.venv` has no `transformers` installed. UNVERIFIED: whether the cp312 extras shadow the bundled 0.42.1 / 5.20.2 at runtime, which depends on `sys.path` order. A clean reinstall of the GPU bundle is the safe fix.

## 3. Accelerators and add-ons (CUDA, Windows)

| Component | Where pinned | Our version | Latest | Win cp312 CUDA wheel | What changed that matters | Upgrade? Effort |
|---|---|---|---|---|---|---|
| SageAttention 2 | `_MANUAL_INSTALL_MESSAGES["sageattention"]` (says to build from source); wired by `helpers/attention_backend.py` (FU-016) | not installed | thu-ml official: v2.0.1 (2025-01), no wheels. Windows builds: `woct0rdho/SageAttention` v2.2.0-windows.post6 (2026-07-17). post7 is ROCm only. | **Yes:** `sageattention-2.2.0+cu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl`, which matches torch 2.11 cu128. A cu130 variant also exists. Apache-2.0. The fork's author also maintains triton-windows; it is not the thu-ml org. | post6 fixes an out-of-bounds bug that gave black or noisy outputs. SageAttention 3 targets Blackwell FP4 and does not apply to the 4090. Needs `triton-windows` (below). Speed-up on our DiT pipelines is UNVERIFIED; measure it. | Yes: one-click install by URL plus sha256. M |
| triton-windows | not pinned (music lock only, cp<3.12) | not installed | 3.9.0.post30 (PyPI, 2026-10-28), MIT | yes | Needed by SageAttention's Triton kernels and by `torch.compile` on Windows. The GitHub repo `woct0rdho/triton-windows` is archived (2026-02), but PyPI releases continue. UNVERIFIED: which triton-windows minor matches torch 2.11. | With SageAttention. S |
| Nunchaku (SVDQuant) | `_MANUAL_INSTALL_MESSAGES["nunchaku"]`; loader in `image_runtime.py` (FU-023) | not installed | `nunchux-ai/nunchaku` stable v1.2.1. Dev v1.3.0dev20260306 has no Windows wheels; the last dev with Windows wheels is v1.3.0dev20260213. Last push 2026-09-06. | **Yes:** `nunchaku-1.3.0.dev20260213+cu12.8torch2.11-cp312-cp312-win_amd64.whl` (and `cu13.0torch2.11`). Apache-2.0. UNVERIFIED whether v1.2.1 ships a torch 2.11 Windows wheel. | 4-bit FLUX, Qwen-Image and SD3.5 transformers. This is the biggest memory and speed lever for FLUX on 24 GB. There are no Windows wheels for torch past 2.11, which is the main reason to hold torch. | Yes: one-click install of the pinned wheel, then run FU-023 on the 4090. M |
| flash-attn | music lock only (cp311 Windows) | not in the main app | 2.8.3.post2 official (Linux only). Windows: `mjun0812/flash-attention-prebuild-wheels` (BSD-3) | **Yes:** `flash_attn-2.8.3+cu128torch2.11-cp312-cp312-win_amd64.whl` (release v0.9.6) | On DiTs, SageAttention usually beats FA2. FA3 and FA4 target Hopper and Blackwell, not Ada. | Low priority. S |
| bitsandbytes | `_INSTALLABLE_PIP_PACKAGES`, bundle `>=0.43` | 0.49.2 installed | 0.50.2 (2026-08-27), MIT | yes (PyPI `win_amd64`) | NF4 FLUX path. | Yes. S |
| torchao | bundle `>=0.6` | 0.17.0 installed | 0.18.0 (2026-08-03) | pure wheel | int8wo / fp8 weight paths. Licence shows NOASSERTION on GitHub; it is BSD-3 by upstream's LICENSE (UNVERIFIED from metadata). | Yes. S |
| kvpress | `"kvpress>=0.5.3"` | floor 0.5.3 | 0.5.5 (2026-09-17), Apache-2.0 | pure | Raise the floor. | Yes. S |
| TriAttention | `pyproject.toml`, setup `__init__.py` | `a4bc3c8` | `a4bc3c8` = HEAD | source | Current. | No |
| vLLM | pyproject `>=0.28.0` | floor | 0.33.0 (2026-11-02) | **No Windows wheels**; WSL only (`routes/setup/vllm_wsl.py`) | Linux CUDA only. | Raise the floor when the WSL path is tested. S |
| xformers | not used | none | 0.0.35 | no PyPI cp312 Windows wheel found | Not needed next to SDPA, SageAttention and FA. | No |
| TensorRT | not used | none | 11.4.0 | no cp312 Windows wheel on PyPI | **Proprietary licence.** Do not add. | No |
| onnxruntime-gpu | not used (voice uses CPU `onnxruntime`) | none | 1.31.1, MIT | yes | A CUDA execution provider for kokoro-onnx and sherpa-onnx. Kokoro is already fast on CPU, so the gain is small. Do not install it next to CPU `onnxruntime`. | Optional. M |

## 4. Python pins: voice venv (`backend_service/voice_env.py`)

| Component | Where pinned | Our version (installed) | Latest | Win cp312 wheel | What changed that matters | Upgrade? Effort |
|---|---|---|---|---|---|---|
| faster-whisper | `>=1.2.1,<1.3` | 1.2.1 | 1.2.1 (2025-10-31); master has later unreleased commits (HEAD `846467d`) | pure | Current release. | No |
| ctranslate2 | via faster-whisper | 4.8.5 | 4.8.5 (2026-11-02), MIT | yes (CUDA 12 on Windows via `nvidia-cublas-cu12` / cuDNN 9) | Current. | No |
| kokoro-onnx | `>=0.6.1,<0.7` | 0.6.1 | 0.6.1 | pure | Current. | No |
| onnxruntime | `>=1.20` | 1.31.1 | 1.31.1 | yes | Current. | No |
| sherpa-onnx | `>=2.0.1,<2.1` | 2.0.1 | 2.0.1 | yes | Current. | No |
| av | `>=13` | 19.0.2 | 19.0.2 | yes | Current. | No |
| SPEECH_GPU_TORCH | `torch==2.9.1`, `torchaudio==2.9.1`, cu128 | 2.9.1+cu128 | 2.11.0 is the newest Windows cu128 | yes | Aligning with the app's 2.11.0 cu128 means one torch line to support. chatterbox-tts 0.1.7 declares `torch==2.6.0`; we install it with `--no-deps`, so test before moving. | Optional. M |
| chatterbox-tts | `SPEECH_GPU_NO_DEPS` `==0.1.7` | 0.1.7 | 0.1.7 is the newest PyPI release. Multilingual V3 (`t3_model="v3"`) exists only on GitHub master (commit `65b1843719`, 2026-06-10). Master HEAD is now `ab55923` (not inspected: API rate limit). MIT. | pure | V3 needs a git install pinned to a commit. | Yes, for V3: `chatterbox-tts @ git+...@<sha>` with `--no-deps`. M |
| voxcpm | `==2.0.3` | 2.0.3 | 2.0.3 (2026-05-11); GitHub pushed 2026-11-09 with unreleased commits. Apache-2.0. | pure | Current release. | No |
| transformers / diffusers / safetensors (speech GPU) | `SPEECH_GPU_PACKAGES` | 5.2.0 / 0.29.0 / 0.5.3 | n/a | pure | Hand-picked for chatterbox and voxcpm. Leave as is. | No |
| torchcodec | `==0.8.1` | 0.8.1 | 0.17.0 | yes | Tied to the torch minor version. Move it only with torch. | Hold |

## 5. Music venv (`backend_service/music_env.py`, `music_requirements.txt`)

| Component | Where pinned | Our version | Latest | Win cp312 CUDA wheel | What changed that matters | Upgrade? Effort |
|---|---|---|---|---|---|---|
| ACE-Step 1.5 | `ACE_REF` | `ca1e85f` | `ca1e85f` = HEAD | source | Current. | No |
| torch (music, Windows) | `music_requirements.txt` (exported from upstream `uv.lock`) | 2.7.1+cu128 on Windows; 2.10.0 on Linux | n/a | yes | Upstream's lock fixes it. | No (upstream-bound) |
| flash-attn / triton-windows (music, Windows) | lock, `python_full_version < '3.12'` only | none on Python 3.12 | n/a | The lock has only cp311 Windows wheels | On this PC (Python 3.12) music runs without flash-attn and Triton. A Python 3.11 music venv on Windows would pick them up. The speed gain is UNVERIFIED. | Optional. M |

## Do next (top 10)

1. **Replace the CPU-only `~/.chaosengine/bin/llama-server.exe` (b9189) on this PC.** Dev runs resolve it before anything else, so chat in dev runs on the CPU. Delete it or drop in the b11981 CUDA build from `test-llama-b11981\cuda`. S, owner action.
2. **Add a Windows CUDA llama-server download.** Mirror `llama_cuda.py` for `llama-<tag>-bin-win-cuda-13.5-x64.zip` plus `cudart-llama-bin-win-cuda-13.5-x64.zip` on driver 580+, and the 12.4 assets on older drivers. Today every Windows NVIDIA user gets Vulkan. Run `llama-bench` on the CUDA and Vulkan builds first to put a number on it. M.
3. **Bump llama.cpp from b11981 to the newest tag with full assets (b12233 today).** In the same change, set `cuda_version_for` to `"13.5"`, or Linux CUDA installs 404. This brings Windows unbuffered load, MTP and DFlash device-side fixes, the spec checkpoint fix, and the Gemma 4 graph gains (Nemotron 3.5 Bolt). S.
4. **Pin torch to 2.11.0 in the GPU bundle.** On driver 580+ it now tries cu130 first with `torch>=2.4.0`, which resolves 2.14.1. That breaks parity with `release-constraints.txt` and with every Nunchaku Windows wheel. S.
5. **Offer the SageAttention 2.2.0 Windows wheel in one click.** Use post6 `cu128torch2.10.0andhigher`, pinned by URL and sha256, together with a matching `triton-windows`. Replace the "build from source" message. It is the cheapest speed-up for image and video DiTs on the 4090; measure it. M.
6. **Offer the Nunchaku Windows wheel in one click** (`1.3.0.dev20260213+cu12.8torch2.11-cp312`), then run the FU-023 live validation on the 4090. It is the biggest VRAM and speed win for FLUX and Qwen-Image. M.
7. **Refresh TurboQuant from the `tqp-v0.4.0` Windows CUDA 12.4 prebuilt.** Our build is from May and its commit is no longer on the branch. The new build brings turbo5/turbo6 KV types and native CUDA flash-attention routing. Pin a tag, not a branch. M.
8. **Switch stable-diffusion.cpp to a pinned prebuilt release** (`master-1079-a3c49b1`, `win-cuda12-x64` plus cudart) instead of building master locally. It brings INT8 and INT4 CUDA fusions, Z-Image and FLUX.3 work, and SeedVR2 upscaling. Add CUTLASS (BSD-3) to `THIRD_PARTY_NOTICES.md`. S to M.
9. **Chatterbox Multilingual V3:** install chatterbox from git at a pinned commit (`65b1843719` or a checked newer one) with `--no-deps` in the speech GPU lane. While there, consider moving that lane to torch 2.11.0 cu128 to match the app. M.
10. **Routine pins and cleanup.** Bump transformers 5.22.2, hf-xet 1.8.0, gguf 0.19.0, bitsandbytes 0.50.2, torchao 0.18.0, kvpress 0.5.5 and imageio-ffmpeg 0.8.1, then re-freeze the constraints. Do huggingface_hub 2.x (the httpx2 move) as a separate change. Clear the stale `extras` dist-infos and the torch 2.6 rollback dir on this PC. S, then M for hub 2.x.

**Hold:** torch 2.12+ (little gain on Ada; breaks Nunchaku and torchaudio on Windows), opencv 5, xformers. **Do not add:** TensorRT (proprietary licence). **Not applicable on Windows:** vLLM (WSL only).
