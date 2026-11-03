# Linux ARM64 and NVIDIA (DGX Spark) research, 2026-11-03

Why: a prospective user asked for Linux ARM64 support on an NVIDIA DGX Spark ([FU-137](hub/tracker/open.md)). This note collects what upstream projects and NVIDIA document about llama.cpp, vLLM, PyTorch, the speech stack and Tauri packaging on Linux ARM64 + CUDA, what we changed because of it, and what is still unknown.

Method: three web-research passes (search results plus some primary pages), then the decisive claims checked by hand against the source (marked **checked**). Everything else is as reported, with the confidence the source deserves. Nothing here was run on a Spark or any NVIDIA GPU.

## The hardware

- DGX Spark: GB10 Grace Blackwell, compute capability 12.1 (sm_121), 128 GB shared CPU/GPU memory (about 119.7 GiB usable), DGX OS 7 (Ubuntu based), NVIDIA driver 580.x, CUDA Toolkit 13.0 ([release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)). The two NVIDIA release-note pages disagree on the exact driver (580.126.09 against 580.95.05).
- `nvidia-smi` works but reports **"Memory-Usage: Not Supported"** because there is no dedicated framebuffer; `cudaMemGetInfo` under-reports allocatable memory; NVIDIA suggests `/proc/meminfo` `MemAvailable` ([known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).
- CUDA 13 drops Maxwell, Pascal and Volta (Turing is the floor) and needs driver 580 or newer; CUDA 12.8 needs 570.26 or newer ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

## llama.cpp

- **Checked:** the b11981 CUDA build's default architecture list includes `120a-real` and `121a-real` for CUDA 12.9+ (`ggml/src/ggml-cuda/CMakeLists.txt`, lines 62-67), and the release workflow builds `ubuntu-cuda-13.4-arm64` without overriding it. So the official prebuilt carries native Spark kernels; no source build needed.
- **Checked (on our ARM VM):** the binaries use `RUNPATH $ORIGIN`; `libggml-cuda.so` needs `libcudart.so.13`, `libcublas.so.13` (shipped in the separate `cudart-` archive) and `libcuda.so.1` (from the driver). Without a GPU, `--list-devices` prints none and the CPU backend still runs.
- Community numbers (llama.cpp discussion [16578](https://github.com/ggml-org/llama.cpp/discussions/16578)): gpt-oss-120b MXFP4 with flash attention, about 1.7-2k tok/s prefill and 46-61 tok/s decode at short context. Loading through mmap is slow on the Spark; `--no-mmap` is the common advice. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` gave no gain.
- A llama.cpp issue ([22561](https://github.com/ggml-org/llama.cpp/issues/22561)) reports the CUDA 13 image silently running on the CPU when the driver is too old. Our `--list-devices` check catches that case.

## vLLM

- Older reports (late 2025) say PyPI vLLM wheels were CUDA 12 builds and failed on the Spark with `libcudart.so.12`, so people used the cu130 nightly index or NVIDIA's container.
- **Checked:** vLLM 0.33.0 on PyPI ships `manylinux_2_28_aarch64` wheels and pins `torch==2.14.0` plus `nvidia-cutlass-dsl[cu13]`, i.e. a CUDA 13 stack. The old failure should not apply to the vLLM versions we install (floor 0.28), but this is not run.
- One report of `'sm_121a' is not defined for option 'gpu-name'` under `torch.compile`; the workaround is eager mode. Unverified, single report.

## PyTorch, diffusers, extras

- **Checked:** torch 2.11.0 (our release pin) on PyPI has `manylinux_2_28_aarch64` wheels whose dependencies are the CUDA 13.0.2 toolkit, cuDNN 9.19 (cu13) and NCCL (cu13). Since 2.11 the default PyPI Linux build is CUDA 13 on both architectures.
- **Checked:** download.pytorch.org has `+cu130` aarch64 wheels (2.9.0 to 2.14.1) and **no** `+cu124` aarch64 wheels.
- sm_120 kernels reportedly run on sm_121, and the "outside supported range" warning is described as harmless. Sources disagree on whether some tensor-core paths need sm_121a. Unverified.
- No official aarch64 or Spark builds found for flash-attn, xformers, SageAttention or Nunchaku; FP4 kernels may not run on sm_121. bitsandbytes reportedly works. All unverified.
- No diffusers numbers for FLUX, Wan or LTX on the Spark were found. NVIDIA says FLUX.2 runs at full precision on it ([NVIDIA blog](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark)).

## Speech

- PyPI's aarch64 CTranslate2 wheel has no CUDA, so faster-whisper falls back to the CPU on ARM ([speaches issue 620](https://github.com/speaches-ai/speaches/issues/620)).
- No prebuilt onnxruntime-gpu for Linux aarch64 on PyPI; community builds exist ([NVIDIA forum](https://forums.developer.nvidia.com/t/onnx-runtime-gpu-inference-on-dgx-spark-gx10-build-guide-and-prebuilt-binaries/366157)). sherpa-onnx presumably follows.
- The Spark has only CUDA 13 runtime libraries; CUDA 12 pip libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) would not match a cu13 torch on the same library path.

## Packaging (Tauri)

- Tauri docs: ARM AppImages cannot be cross-compiled; native `ubuntu-22.04-arm` / `ubuntu-24.04-arm` runners work ([AppImage docs](https://v2.tauri.app/distribute/appimage/)). Build on the oldest Ubuntu we support (glibc baseline).
- The ARM AppImage is named `_aarch64.AppImage` (tauri-action once looked for `arm64`, [issue 798](https://github.com/tauri-apps/tauri-action/issues/798)); the deb follows Debian's `_arm64.deb`. Our manifest generator accepts both spellings for both.
- Updater keys are `OS-ARCH` (`linux-aarch64`), and the updater validates every entry before it compares versions ([updater docs](https://v2.tauri.app/plugin/updater/)), so a half-written ARM entry would break updates on every platform. Our generator writes an ARM entry only when the file and its signature both exist.
- Ubuntu 24.04+ has no `libfuse2` by default; an AppImage then fails with `libfuse.so.2`. Fix: `sudo apt install libfuse2t64`, or use the deb.

## Beta channel

- A pre-release `0.9.10-beta.1` sorts below `0.9.10` under SemVer, so beta testers would get the final release as an update (inferred from the "greater than" rule; not tested).
- Windows MSI accepts only a numeric pre-release (`-1`), not `-beta.1`; the release workflow builds NSIS only, whose handling is untested.
- GitHub's `releases/latest` is widely understood to skip pre-releases and drafts; the docs page does not say so. Test on the real pre-release before announcing.
- A good call-for-testers pattern (Ubuntu's arm64 call for testing): state the tested hardware and driver, list known failures, ask for device, driver and logs.

## What we changed because of this

| Finding | Change |
|---|---|
| Spark reports GPU memory as "Not Supported" | `_parse_nvidia_smi` takes system memory for a GPU with no memory of its own; `nvidia_shared_memory()` detects it ([gpu.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/gpu.py)) |
| mmap loading is slow on shared-memory GPUs | llama-server gets `--no-mmap` there (`CHAOSENGINE_LLAMA_MMAP=1` keeps mmap) |
| GPU runtime installer tried cu124 first, which has no ARM64 build and no Blackwell kernels | `cuda_torch_indexes()`: Linux ARM64 tries cu130 then cu128; a 580+ driver starts at cu130, 570+ at cu128 (this also fixes RTX 50-series on x86_64) |
| CTranslate2 aarch64 is CPU-only | the voice pack no longer installs CUDA 12 libraries on Linux ARM64 |
| CUDA llama-server not bundled for ARM | Settings > Accelerators installs ggml-org's CUDA build and uses it only when it lists a CUDA device |
| Old/missing driver makes the CUDA build fall back to CPU silently | the `--list-devices` check keeps the bundled build in that case |

## Still unknown (needs a real Spark)

- Whether torch 2.11 cu13 runs diffusers image and video models correctly on sm_121, and how fast.
- vLLM on the Spark through our Setup install (torch 2.14 lands in the extras overlay).
- Real `nvidia-smi --query-gpu` output on the Spark (we assume "[Not Supported]" or "[N/A]"; both parse).
- Load time and tok/s of a large GGUF through the app with `--no-mmap`.
- The ARM64 AppImage and deb on a real Spark desktop (WebKitGTK, FUSE).
