## What's new in v0.9.8

The first release since v0.9.4. Versions 0.9.5 to 0.9.7 were internal builds, so everything below is new to anyone on 0.9.4 or earlier.

### New models

**Chat**
- **Kolibri 1** (Aleph Alpha, Apache 2.0): a 78B mixture-of-experts model with 3.5B active per token, built for German and English, with a 256K context. It runs on Apple Silicon through community MLX conversions (4-bit, 3-bit and 2-bit); the 3-bit build was run end to end, with correct German answers at every effort level and a working tool call. The Thinking control drives its reasoning level: Off turns reasoning off, Low / Med / High set the effort. No GGUF or CUDA build yet, because neither llama.cpp nor vLLM supports the architecture.
- **Kimi K2.6**, **Llama 4 Scout**, **MiniMax M3**, **Mistral Large 3**, **Ornith 1** (agentic coding), **Qwen 3.8**, **Bonsai** (1-bit and ternary builds that fit a 27B model in 4 to 9 GB), **Nemotron 3.5 Lightning** and **MiniCPM 5**.
- The duplicate Gemma 4 card is gone; all variants sit in one family.

**Image**
- **Qwen-Image-2.1** (7B, text-to-image) on Apple Silicon through MLX: 1024x1024 in about 6 minutes on an M4 Max. A stable-diffusion.cpp variant covers other platforms. Qwen-Image-2.1 is under the Qwen Research licence, which is non-commercial only.
- **FLUX.2 klein 4B** (26 s at 1024x1024 on an M4 Max), plus Qwen-Image, Z-Image Turbo and FLUX.1 through the existing diffusers lane.

### Faster generation

- **TensorFold** (Apple Silicon): exact speculative decoding with no change in output. Install it from the launch settings. Optional image-input and structured-output extras.
- **llama.cpp speculative lanes**: DFlash 2, EAGLE-3 and draft-free `ngram-mod`, picked automatically from the drafter files next to a GGUF. Needs a llama-server that supports them; older binaries fall back to standard decoding with a note.
- **MTPLX 2.12**: reasoning now streams correctly (previously a thinking model could look silent), and the forced benchmark profile is gone.
- **Hybrid models** (Qwen 3.5 / 3.6 / 3.8, Qwen3-Next, Nemotron-H, Mamba) now reuse the chat history between turns instead of re-reading the whole conversation every time.
- **TurboQuant** is labelled a memory saver, not a speed option, and its speed estimate now uses measured M4 Max numbers.

### Image and video

- **mflux engine** (Apple Silicon): image models run on MLX in their own isolated environment. Install it from the banner in Image Studio.
- **stable-diffusion.cpp lane**: FLUX.1 schnell and dev now fetch their text-encoder and VAE files on first use, and image runs show live progress.
- **Wan videos** no longer come out burnt and over-saturated on Apple Silicon (the sampler diverged on the GPU; it now uses a stable one there).
- Catalog **LoRA variants** (Hyper-SD, Turbo-Alpha, CausVid) now download their LoRA on first use; before, they quietly ran without it.
- Image models too big for free memory are refused up front with an explanation instead of crashing the app.

### Voice I/O

A Voice group in the sidebar: speech-to-text and text-to-speech with Studio, Models and Gallery tabs. Whisper and Parakeet (Apple Silicon) or faster-whisper (Windows and Linux) for transcription, Kokoro for speech.

### Fixes

- **Linux AppImage** stuck on "Still loading": the backend was killed as soon as the app finished starting it, and the loading screen never showed the error. Both fixed; the log path now appears on the loading screen. Not yet tested on Fedora.
- **Local GGUF folders** from ModelScope (`Qwen3___5` style names) and symlinked model folders are now listed.
- **GGUF downloads** fetch only the quant a catalog row names, not every quant in the repo (one 16 GB row used to pull 68 GB).
- **Dead catalog links**: 13 catalog rows pointed at Hugging Face repos or files that do not exist, so their Download button failed. All repaired, and a new audit script checks every row.
- **Release builds** now include the `dflash` package, so DFlash and DDTree actually engage in the packaged app, and the Setup installers for MTPLX, TensorFold and mflux.
- **Tool use on Apple Silicon (MLX)** no longer loops: tool results were never passed back to the model, so it repeated the same call until the iteration cap. The non-streaming chat API with tools enabled no longer answers 500.
- Installing optional packages from Setup no longer pulls a newer MLX over the one the app runs.
- Models whose download skipped some files on purpose no longer open as a placeholder image engine.
- A model already on disk outside the Hugging Face cache is loaded from there instead of being downloaded again.
- The memory check counts the cache macOS can reclaim, so large models are no longer refused with plenty of memory free.

### Known limits

- stable-diffusion.cpp and `llama-server-turbo` are not bundled in the release build (as in earlier releases). The sd.cpp image variants need the `sd` binary built from a source checkout with `scripts/build-sdcpp.sh`.
- Not tested on real hardware: the Linux fix on Fedora, sd.cpp FLUX on CUDA, the CausVid LoRA on a real Wan pipeline.
- The CausVid LoRA is CC BY-NC; Qwen-Image-2.1 is non-commercial.
- Kolibri 1 chat history is re-read each turn once it passes about 500 tokens.

### Downloads

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `ChaosEngineAI_0.9.8_aarch64.dmg` |
| Windows | `ChaosEngineAI_0.9.8_x64-setup.exe` |
| Linux | `ChaosEngineAI_0.9.8_amd64.AppImage`, `ChaosEngineAI_0.9.8_amd64.deb` |

The in-app updater reads `latest.json`; the `.sig` files and `ChaosEngineAI_aarch64.app.tar.gz` next to the installers are its inputs.

### Upgrade notes

- If your install keeps offering v0.9.4 although you are on it: v0.9.4 was built as 0.9.3 by mistake. Installing this release ends it, and releases now fail early if the tag and the app version differ.
- No breaking changes: settings, sessions and APIs carry forward.
- Speculative decoding on llama.cpp needs a recent llama-server; the Setup tab shows what is installed.
- TensorFold, MTPLX and mflux each install into their own environment under `~/.chaosengine/` on first use. TensorFold and mflux need Python 3.11 to 3.13 on the Mac.
