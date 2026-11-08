# ChaosEngineAI v0.9.10-beta.1 (Linux beta)

This is a **pre-release for Linux testers**, mainly for **Linux ARM64 with an NVIDIA GPU** (DGX Spark and similar GB10 / Grace machines). It is not offered through the in-app updater: existing installs stay on v0.9.9. When v0.9.10 ships, beta installs will be offered it as a normal update.

macOS and Windows users: stay on [v0.9.9](https://github.com/cryptopoly/ChaosEngineAI/releases/tag/v0.9.9). This beta has no macOS or Windows packages.

## What is new for Linux

- **Linux ARM64 packages** (`_aarch64.AppImage`, `_arm64.deb`) for the first time. They include the Python runtime and a CPU build of llama.cpp, so GGUF chat works out of the box on the CPU.
- **CUDA llama-server for NVIDIA GPUs** (Linux ARM64 and x86_64): Settings > Accelerators shows an Install button when `nvidia-smi` is present. It downloads llama.cpp's CUDA build (about 680 MB, including the CUDA runtime) into `~/.chaosengine/bin/llama-cuda/`. The app switches to it only when it can see a CUDA device, so a missing or too-old driver falls back to the CPU build.
- **Shared-memory GPUs** (DGX Spark GB10, GH200): `nvidia-smi` reports their memory as "Not Supported"; the app now reads system memory instead, so the GPU is detected and memory estimates use the 128 GB pool. llama-server loads with `--no-mmap` on these machines (set `CHAOSENGINE_LLAMA_MMAP=1` to keep mmap).
- **GPU runtime install** (image and video models) picks the PyTorch build that matches the CPU and driver: CUDA 13.0 first on ARM64 and on drivers 580+, CUDA 12.8 on drivers 570+. This also helps RTX 50-series cards on x86_64.
- Hugging Face Hub telemetry is now off by default (`HF_HUB_DISABLE_TELEMETRY=1`; set it to `0` yourself to opt back in).

## Install

**deb** (Ubuntu 22.04 or newer, Debian 12 or newer):

```bash
sudo apt install ./ChaosEngineAI_0.9.10-beta.1_arm64.deb
```

**AppImage**:

```bash
chmod +x ChaosEngineAI_0.9.10-beta.1_aarch64.AppImage
./ChaosEngineAI_0.9.10-beta.1_aarch64.AppImage
```

If the AppImage fails with `libfuse.so.2`, install `libfuse2t64` (Ubuntu 24.04 and newer) or `libfuse2`, or use the deb.

x86_64 Linux packages are attached as well (`_amd64`).

## What has been tested

- Built on GitHub's Ubuntu 22.04 ARM runners (glibc 2.35 baseline).
- On an Ubuntu 26.04 ARM64 virtual machine without a GPU: the deb and the AppImage install and start, and a GGUF model (LFM3 1.7B Q4_K_M) loads and chats on the CPU at about 29 tokens per second.
- The CUDA llama.cpp download was installed and run on that machine; with no GPU it correctly keeps the CPU build.

## What has not been tested (this is why we need you)

- **Any NVIDIA GPU on Linux ARM64**, including a real DGX Spark. GPU detection, the CUDA llama-server, `--no-mmap` and the PyTorch CUDA 13 install are built from NVIDIA's and llama.cpp's documentation, not from a run on the hardware.
- Image and video generation (diffusers on PyTorch CUDA 13) on the GB10's sm_121.
- vLLM installed from the Setup page on ARM64.
- Voice transcription runs on the CPU on ARM64 (the ARM build of its speech engine has no CUDA support).

## How to help

1. Install the beta, open Settings > Accelerators and install the CUDA llama-server.
2. Run the GPU check script and attach its report to a GitHub issue:

```bash
curl -fsSLO https://raw.githubusercontent.com/cryptopoly/ChaosEngineAI/v0.9.10-beta.1/scripts/linux-gpu-check.sh
bash linux-gpu-check.sh
```

It writes `~/chaosengine-gpu-report.md` (machine, driver, the app's status and a short llama.cpp benchmark on a 1.1 GB model). Nothing is uploaded; read it before you post it.

3. Tell us what you tried, what worked and what did not. Include the backend log (`/tmp/chaosengine-backend-8876.log`) for anything that failed.

## Checksums

See `SHA256SUMS.txt` attached to this release.
