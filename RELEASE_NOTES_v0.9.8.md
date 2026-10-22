## What's new in v0.9.8

The first release since v0.9.4. Versions 0.9.5 to 0.9.7 were internal builds, so everything below is new to anyone on 0.9.4 or earlier.

### Highlights

- **DFlash on Apple Silicon finally runs.** Since dflash-mlx 0.1.5, every DFlash reply had quietly fallen back to normal decoding while the load note still said DFlash was active. Fixed: Qwen3-4B now writes at 113 to 138 tokens per second against 53 without it, with identical output.
- **Newer MLX underneath.** Releases now ship mlx 0.32, mlx-lm 0.32 and transformers 5.20. Setup installs resolve mlx-vlm 0.7.10 and diffusers 0.41. Current Gemma 4 conversions load (the old mlx-vlm could not read them), and every release build pins the exact versions it was tested with.
- **Images in chat on Apple Silicon.** Vision models loaded through mlx-vlm (Gemma 4) now see attached images. Before, the image was always dropped.
- **Honest memory and speed estimates.** The launch screen works out the real KV-cache size for each model's architecture and estimates speed from your own past runs or your machine's memory bandwidth. It warns before you load a chat model that may not fit. Image and video estimates know which engine they run on and count a chat model that is already loaded.

### New models

**Chat**
- **Kolibri 1** (Aleph Alpha, Apache 2.0): a 78B mixture-of-experts model with 3.5B active per token, built for German and English, with a 256K context.
  - Runs on Apple Silicon through community MLX builds (4-bit, 3-bit, 2-bit) at about 60 tokens per second (3-bit, M4 Max).
  - A new llama.cpp Q3_K_M build (37.6 GB) also runs on Windows and Linux.
  - The Thinking control sets its reasoning level.
- **LFM2.5 24B-A2B** (Liquid AI): MLX 4-bit and 8-bit, and a GGUF build. About 100 tokens per second (MLX 4-bit) and 140 (GGUF) on an M4 Max.
- **Flint** (Recursive, Apache 2.0): a 39.8B coding-agent mixture-of-experts model with 2.5B active per token and a 256K context. Runs through llama.cpp: the official INT4 GGUF (26 GB, Apple Silicon with 48 GB or more) and a community Q4_K_M for Windows and Linux. About 112 tokens per second on an M4 Max, with reasoning and tool calls working.
- **Muse Glimmer 30B** GGUF, with image input and its DFlash 2 drafter: 29 tokens per second against 19.5 without the drafter.
- **Qwen3.8 Flash Next** GGUF (111 GB, for machines with 128 GB or more) and **Qwen3.6 35B-A3B with MTP** through TensorFold.
- Also new since 0.9.4: **Kimi K2.6**, **Llama 4 Scout**, **MiniMax M3**, **Mistral Large 3**, **Ornith 1**, **Qwen 3.8**, **Bonsai** (1-bit and ternary builds that fit a 27B model in 4 to 9 GB), **Nemotron 3.5 Lightning** and **MiniCPM 5**.

**Image**
- **Qwen-Image-2.1 Turbo** through mflux: 1024x1024 in about 95 seconds on an M4 Max (8 steps).
- **FLUX 3 Image Klein** (distilled and base) through mflux: 1024x1024 in about 83 seconds on an M4 Max (16 steps, 23.5 GB peak). Black Forest Labs' licence must be accepted on Hugging Face for each repo, and your token needs read access to gated repos.
- **Qwen-Image-2.1** (7B) on Apple Silicon through MLX, with a stable-diffusion.cpp variant for other platforms. Qwen-Image-2.1 and its Turbo are under the Qwen Research licence (non-commercial).
- **FLUX.2 klein 4B**, plus Qwen-Image, Z-Image Turbo and FLUX.1 through the diffusers lane.

### Faster generation

- **DFlash** (Apple Silicon) fixed, as above.
- **Thinking Off now works** for Qwen 3 / 3.5 / 3.6 / 3.8 on Apple Silicon. Before, they reasoned anyway and spent the reply budget on it.
- **TensorFold 0.6.6** (Apple Silicon): exact speculative decoding with no change in output. Install it from the launch settings.
- **llama.cpp b11810** bundled: DFlash 2, EAGLE-3 and draft-free `ngram-mod`, picked automatically from the drafter files next to a GGUF.
- **MTPLX 2.12**: reasoning now streams correctly.
- **Hybrid models** (Qwen 3.5 / 3.6 / 3.8, Qwen3-Next, Nemotron-H, Mamba) reuse the chat history between turns instead of re-reading the whole conversation.
- **TurboQuant** is labelled a memory saver, not a speed option.

### Image and video

- **mflux engine** (Apple Silicon) now at 0.23, in its own isolated environment. The Image Studio banner offers an update when an older version is installed.
- **stable-diffusion.cpp lane**: FLUX.1 schnell and dev fetch their text-encoder and VAE files on first use, and image runs show live progress.
- **Wan videos** no longer come out burnt and over-saturated on Apple Silicon.
- Catalog **LoRA variants** (Hyper-SD, Turbo-Alpha, CausVid) now download their LoRA on first use.
- Image and video Discover list every variant once. Variants that share a download are marked.

### Voice I/O

A Voice group in the sidebar: speech-to-text and text-to-speech with Studio, Models and Gallery tabs. Whisper and Parakeet (Apple Silicon) or faster-whisper (Windows and Linux) for transcription, Kokoro for speech.

### Chat and Settings

- **Chat > My Models lists chat models only.** 3D tool parts, text encoders, voice models and bare weight files no longer show up as chat models.
- **Settings > Diagnostics is split into three tabs**: Diagnostics, Accelerators and Disk usage.
- **Disk usage rows expand and can be deleted.** The shared Hugging Face `blobs` store is protected.

### Fixes

- **GGUF chat**: models that reason even with reasoning off (Muse Glimmer) showed an empty reply. Launching a catalog model from Discover skipped its chosen quant and its drafter file.
- **Apple Silicon chat**: LFM2.5's reasoning leaked into the answer. A reply cut short (for example by the repeat guard) made the next action fail with "invalid result payload".
- **Downloads that would not load**: a cancelled download of another quant in the same repo, a library scan taken mid-download, or an organisation name ending in "models" each blocked a finished model.
- **Text to speech** said it was ready when only part of the voice pack was installed, then produced no audio. On an install with a very long folder path, one speech request could shut down the backend.
- **Linux AppImage** stuck on "Still loading". Fixed; the log path now appears on the loading screen. Not yet tested on Fedora.
- **Local GGUF folders** from ModelScope and symlinked model folders are now listed.
- **GGUF downloads** fetch only the quant a catalog row names (one 16 GB row used to pull 68 GB).
- **Dead catalog links** repaired; an audit script checks every row.
- **Release builds** include the `dflash` package, the Setup installers and a pinned llama.cpp build, so GGUF models work out of the box.
- **Tool use on Apple Silicon** no longer loops.
- The memory check counts memory macOS can reclaim, so large models are no longer refused with plenty free.

### Removed

- The DeepSeek-V4 8-bit, BF16 and Pro MLX rows, which nothing in the app could load. The 4-bit row stays and runs through TensorFold.
- The DDTree tree-budget slider on Apple Silicon. Its tree verification was never finished; DFlash runs instead.

### Known limits

- stable-diffusion.cpp and `llama-server-turbo` are not bundled in the release build. The sd.cpp image variants need the `sd` binary built from a source checkout.
- MLX models that are multimodal but load as text only (Qwen 3.5 / 3.6 / 3.8 on MLX) do not see images yet.
- Not tested on real hardware: the Linux fix on Fedora, sd.cpp FLUX on CUDA, the CausVid LoRA on a real Wan pipeline, Qwen3.8 Flash Next GGUF (needs more memory than the test machine has).
- The CausVid LoRA is CC BY-NC; Qwen-Image-2.1 and its Turbo are non-commercial.
- Kolibri 1 chat history is re-read each turn once it passes about 500 tokens.

### Downloads

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `ChaosEngineAI_0.9.8_aarch64.dmg` |
| Windows | `ChaosEngineAI_0.9.8_x64-setup.exe` |
| Linux | `ChaosEngineAI_0.9.8_amd64.AppImage`, `ChaosEngineAI_0.9.8_amd64.deb` |

The in-app updater reads `latest.json`; the `.sig` files and `ChaosEngineAI_aarch64.app.tar.gz` next to the installers are its inputs.

### Upgrade notes

- If your install keeps offering v0.9.4 although you are on it: v0.9.4 was built as 0.9.3 by mistake. Installing this release ends it.
- No breaking changes: settings, sessions and APIs carry forward.
- On first start the app moves older copies of mlx and mlx-lm that earlier Setup installs left in its extras folder to `extras/<python>/stale-core/`, so they cannot shadow the newer versions it ships. If the video runtime, voice pack or DFlash then misbehave, reinstall them from Setup.
- TensorFold, MTPLX and mflux each install into their own environment under `~/.chaosengine/` on first use. TensorFold and mflux need Python 3.11 to 3.13 on the Mac.
