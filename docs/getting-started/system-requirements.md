# System requirements

ChaosEngineAI supports three classes of inference hardware. Each one ships
with a different default engine, supports a different set of cache strategies,
and exposes a different speculative-decoding path.

## Apple Silicon (recommended)

The reference platform — every feature ships and is exercised here first.

- **Hardware:** M1 / M2 / M3 / M4 / M5 Mac with at least 16 GB unified memory.
  32 GB unlocks 14B-class models at f16; 64 GB+ comfortably runs 27B / 35B-A3B
  MLX quants and Wan 2.1 / 2.2 video diffusion.
- **OS:** macOS 13 Ventura or newer. Notarized builds run on macOS 14+.
- **Engines:** MLX (default), `llama.cpp` GGUF (via Homebrew or bundled binary).
- **Cache strategies:** Native f16, TurboQuant (MLX + llama.cpp fork),
  TriAttention (CPU fallback — slow), FBCache, TeaCache, TaylorSeer, MagCache,
  PAB, FasterCache for diffusion.
- **Speculative decoding:** DFlash (via `dflash-mlx`), DDTree (built on
  DFlash), MTPLX (native Multi-Token Prediction, isolated venv).
- **Image / video:** Diffusers on MPS, mlx-video subprocess engine for LTX-2
  and Wan, optional stable-diffusion.cpp via Metal.

## Linux + CUDA

The performance platform for large dense models.

- **Hardware:** NVIDIA GPU with SM 8.0+ (Ampere or newer). FP8 layerwise
  casting requires SM 8.9+ (Ada / Hopper). At least 24 GB VRAM for 14B-class
  models at fp16.
- **OS:** Ubuntu 22.04 LTS / Debian 12 / any recent x86_64 Linux with a
  matching NVIDIA driver.
- **Engines:** `llama.cpp` GGUF with CUDA backend, vLLM (`pip install vllm`),
  Diffusers on CUDA.
- **Cache strategies:** Native f16, TurboQuant (via the `llama-server-turbo`
  fork), TriAttention (native vLLM integration), all diffusion caches.
- **Speculative decoding:** DFlash (via `dflash`), DDTree.
- **Image / video:** Diffusers + CUDA, optional SageAttention
  (`pip install sageattention`), Nunchaku SVDQuant transformers, FP8 layerwise
  casting for Wan / HunyuanVideo / FLUX, optional stable-diffusion.cpp via
  cuBLAS.

MTPLX is **not** supported on Linux — its forked `mlx` is Metal-only. Use
DFlash for speculative decoding on CUDA.

## Windows

- **Hardware:** Same NVIDIA GPU requirements as Linux for accelerated paths.
  CPU-only is supported but slow.
- **OS:** Windows 11. Windows 10 may work but isn't routinely tested.
- **Engines:** `llama.cpp` GGUF (CPU / CUDA), Diffusers (CPU / CUDA).
  vLLM is not packaged for Windows by upstream.
- **Cache strategies:** Native f16, TurboQuant (CUDA llama.cpp fork), all
  diffusion caches.
- **Speculative decoding:** DFlash on CUDA. No MTPLX.

## CPU-only

Everything works in principle, but in practice you'll want at least 32 GB of
system RAM and patience. CPU inference is useful for development and CI; it's
not a daily driver.

## Disk

| Workload | Working size |
|---|---|
| App install | ~250 MB |
| One small LLM (4-8B 4-bit GGUF) | 2 – 6 GB |
| One medium LLM (14-27B 4-bit) | 8 – 20 GB |
| Mid-quality SDXL pipeline | ~7 GB |
| FLUX.1-dev or FLUX.1-schnell | ~24 GB |
| Wan 2.1 T2V 14B | ~28 GB |
| Wan 2.2 TI2V 5B (mlx-video converted) | ~12 GB |

Plan for at least 100 GB free if you're going to seriously explore image and
video diffusion models. Models live under `~/AI_Models/` by default; the
location is configurable from **Settings → Storage**.

## Network

- Hugging Face access (login token recommended for gated models — set it in
  Settings).
- Optional outbound HTTPS to a remote OpenAI-compatible provider, if you
  configure one in Settings.

The backend itself binds to `127.0.0.1` by default and never reaches out on
its own. Set `CHAOSENGINE_HOST=0.0.0.0` if you want LAN exposure of the
OpenAI-compatible API.
