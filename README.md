<p align="center">
  <img src="./ChaosEngineAI_AppIcon.svg" alt="ChaosEngineAI" width="200" />
</p>

<h1 align="center">ChaosEngineAI</h1>

<p align="center">
  <strong>The local AI model runner for serious tinkerers.</strong><br/>
  Discover, convert, serve, chat with, benchmark, and generate images from open-weight models — all on your own machine.
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-work%20in%20progress-f59e0b?style=flat-square" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-1f2937?style=flat-square" />
  <img alt="Shell" src="https://img.shields.io/badge/shell-Tauri%202-24c8db?style=flat-square" />
  <img alt="Backend" src="https://img.shields.io/badge/backend-Python%20%2B%20llama.cpp-3776ab?style=flat-square" />
  <img alt="Acceleration" src="https://img.shields.io/badge/Apple%20Silicon-MLX-000000?style=flat-square" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-4b5563?style=flat-square" />
</p>

> ⚠️ **Work in progress.** ChaosEngineAI is under active development. Expect rough edges, breaking changes between versions, and features that appear (and occasionally disappear) from one release to the next. Feedback and issue reports are very welcome.

<p align="center">
  <img src="./docs/tour.gif" alt="ChaosEngineAI tour" width="900" />
</p>

---

## Why ChaosEngineAI

ChaosEngineAI is a desktop control plane for running large language models locally. It pairs a fast Tauri + React shell with a Python backend that drives `llama.cpp` and a custom MLX runtime, so you get a single window for everything from "I want to try this Hugging Face model" to "show me tokens-per-second across three quantizations on this exact prompt."

- **One app, the whole pipeline.** Discover models, download them, convert to MLX, load into a warm pool, serve over an OpenAI-compatible API, chat, benchmark, and generate images.
- **Real local performance.** First-class support for `llama.cpp` GGUF and Apple Silicon MLX for LLMs, plus local Stable Diffusion for image generation — with a pluggable cache strategy system supporting optional third-party compression backends.
- **Built for power users.** Live runtime telemetry, structured logs, side-by-side benchmark history, and granular launch preferences (FP16 layers, fused attention, context window, KV cache strategy).
- **Polished, fast UI.** A focused dark workspace that gets out of the way and never blocks on the backend.
- **Self-updating.** Ships with a signed in-app updater — no more manual re-downloads when a new release drops.

---

## Feature Highlights

- 🧭 **Discover** open-weight models from curated catalogs and pull them with one click
- 💾 **My Models** library with format, size, context, and modified-date sorting
- 🔁 **Conversion** pipeline that turns Hugging Face checkpoints into MLX (Apple Silicon)
- 🚀 **Server** mode exposing an OpenAI-compatible REST API for your other tools
- 💬 **Chat** with the loaded model, including document attachments for RAG and image inputs for vision-capable models
- 🎨 **Image Discover** curated catalog of Stable Diffusion models with one-click download from Hugging Face
- 🖼️ **Image Models** library showing installed image models ready for generation
- 🖌️ **Image Studio** for prompt-based image generation with aspect ratio, quality presets, and negative prompts
- 🏛️ **Image Gallery** to browse, filter, and reuse saved outputs — compare models and re-run with the same settings
- 📊 **Benchmarks** with reproducible runs, live token/sec streaming, and a history view for A/B comparisons
- 📜 **Logs** streaming straight from the Python runtime
- ⚙️ **Settings** for directories, default launch preferences, and runtime tuning
- 🧠 **Warm pool** keeps recently-used models hot so subsequent loads are instant
- 🔄 **In-app updates** — signed, verified, cross-platform. Launch, prompt, relaunch.
- 🍎 **Notarized macOS builds** with a fully embedded, hardened-runtime Python runtime
- 🐧 **AppImage + .deb** for Linux, **NSIS installer** for Windows
- 🎚️ **Adaptive runtime controls** for balancing speed, memory footprint, and quality on local hardware
- 📈 **Live telemetry** on the dashboard: backend health, engine, loaded model, hardware, warm-pool state

---

## A Tour of the App

### Dashboard — *System overview*
![Dashboard](./desktop/Screenshots/1.%20Dashboard_v2.png)

The launchpad. Surfaces backend health, the engine in use, the currently loaded model, hardware (platform, arch, memory), and quick stats from the warm pool. Big colored badges tell you instantly whether the runtime is online and what cache mode it will use on the next launch.

### Discover — *Browse and download AI models*
![Discover](./desktop/Screenshots/2.%20Discover_v2.png)

Browse curated model families with capability tags (chat, coding, vision, reasoning, tools, multilingual, video, thinking). Expand a family to see every variant with its format, parameter count, and quant level, then queue downloads. Filter by format and capability across the whole catalog.

### Model Selection — *Configure and launch*
![Model Selection](./desktop/Screenshots/4.%20Model_Selection_Modal_v2.png)

The launch modal. Pick a variant, set context length, choose the engine, and tune runtime strategy knobs in one place, pre-populated from your defaults.

### My Models — *Models on this machine*
![My Models](./desktop/Screenshots/3.%20My_Models_v2.png)

Your local library. Sort by name, format, size, context length, or last-modified date. Each entry shows where it lives on disk, its quant scheme, and a one-click **Launch** that opens the launch modal with the right defaults pre-populated.

### Image Discover — *Browse and download image models*
![Image Discover](./desktop/Screenshots/5.%20Image%20Discover.png)

Curated catalog of local image models. Discover Stable Diffusion models optimized for ChaosEngineAI, scout new releases from Hugging Face, and install them with one click. Filter by compatibility tags and see metadata at a glance.

### Image Models — *Installed image models*
![Image Models](./desktop/Screenshots/6.%20Image%20Models.png)

Your installed image model library. See which Stable Diffusion models are ready for generation — each card shows size, diffusion pipeline, resolution, and a one-click **Generate** to jump straight into Image Studio.

### Image Studio — *Generate images from text*
![Image Studio](./desktop/Screenshots/7.%20Image%20Studio.png)

Prompt-based image generation with full control. Choose a model, set aspect ratio and quality presets (square, portrait, landscape, wide), write positive and negative prompts, and generate. Recent outputs appear in the right panel with metadata and re-run options.

![Image Processing](./desktop/Screenshots/8.%20Image%20Processing.png)

Live progress while the diffusion pipeline runs — step-by-step denoising visualization, elapsed time, and a running status log.

![Image Completion](./desktop/Screenshots/9.%20Image%20Completion.png)

Completed image with full generation metadata — model, prompt, seed, steps, resolution, and timing. Open, reveal on disk, clone settings, or save directly from the completion modal.

### Image Gallery — *Browse and reuse generated images*
![Image Gallery](./desktop/Screenshots/10.%20Image%20Gallery.png)

All your generated images in one place. Search by prompt, model, or runtime; filter by frame size and sort order. Each card shows the source model, generation settings, and quick actions to re-run with the same seed or open in Image Studio.

### Chat — *Local AI chat*
![Chat](./desktop/Screenshots/12.%20Chat_v2.png)

A focused chat surface with multi-thread sessions in the left rail, streaming responses, document and image attachments, and inline thread renaming. Threads persist across launches and are scoped to the model that produced them.

### Server — *OpenAI-compatible local API*
![Server](./desktop/Screenshots/11.%20Server_v2.png)

Start, stop, and inspect a local OpenAI-compatible HTTP server backed by the loaded model. Shows the bind address, current model, request count, and a remote-test panel for firing a sample completion against `/v1/chat/completions` without leaving the app.

### Benchmarks — *Run a new benchmark*
![Benchmarks](./desktop/Screenshots/13.%20Benchmarks_v2.png)

Configure a benchmark run: choose a model, prompt set, token budget, and decoding parameters, then watch live progress as the runner streams tokens-per-second, time-to-first-token, and memory usage.

![Benchmark running](./desktop/Screenshots/14.%20Benchmark_Running_v2.png)

Live progress while a run is in flight — token/sec, TTFT, current prompt, and memory pressure all update in real time.

![Benchmark complete](./desktop/Screenshots/15.%20Benchmark_Complete_v2.png)

Final report card: throughput, latency percentiles, generation samples, and the exact run parameters — saved automatically to history.

### History — *Compare saved runs*
![Benchmark History](./desktop/Screenshots/16.%20Benchmark_History_v2.png)

Every benchmark you've ever run, side-by-side. Pick two runs and the page diffs them across throughput, latency, and quality metrics — perfect for proving that your new quant actually pays its keep.

### Conversion — *Convert models to MLX format* (macOS only)
![Conversion](./desktop/Screenshots/17.%20Conversion_v2.png)

Apple Silicon only. Point at a Hugging Face checkpoint or local directory and convert it to MLX with optional runtime-specific compression settings. A conversion picker surfaces eligible source models.

![Conversion running](./desktop/Screenshots/18.%20Conversion_Running_v2.png)

Layer-by-layer live progress while the conversion runs — bit budget per block, memory footprint, and a running log tail.

### Logs — *Runtime events*
![Logs](./desktop/Screenshots/19.%20Logs_v2.png)

A live tail of the backend log stream — load events, server requests, errors, and runtime warnings — with level filtering. The first place to look when something feels off.

### Settings — *Directories and defaults*
![Settings](./desktop/Screenshots/20.%20Settings_v2.png)

Configure model and cache directories, default launch preferences (cache strategy, FP16 layers, fused attention, context tokens, fit-in-memory toggle), and advanced runtime knobs. Every default in this panel is reused as the starting state for the launch modal.

---

## Download & Install

Head to the [Releases](https://github.com/cryptopoly/ChaosEngineAI/releases/latest) page for signed builds:

| Platform | File | Notes |
|---|---|---|
| **macOS** (Apple Silicon) | `ChaosEngineAI_*_aarch64.dmg` | Signed + notarized |
| **Linux** | `ChaosEngineAI_*_amd64.AppImage` | Portable, in-app updates supported |
| **Linux** (Debian/Ubuntu) | `ChaosEngineAI_*_amd64.deb` | Install via `dpkg`, update via apt |
| **Windows** | `ChaosEngineAI_*_x64-setup.exe` | Unsigned for now — SmartScreen will warn on first run |

From v0.4.21 onward, every install auto-updates from GitHub Releases on launch. Updates are cryptographically signed.

---

## Quick Start (from source)

Prereqs: Rust toolchain, Node 20+, Python 3.11+, and (on macOS) Xcode command-line tools.

```bash
# 1. Install JS dependencies
cd desktop
npm install

# 2. Stage the bundled Python runtime + llama.cpp binaries into src-tauri/
npm run stage:runtime

# 3. Launch the desktop app in dev mode (Tauri shell + Vite HMR)
npm run tauri:dev
```

That's it — the Python backend is spawned by the Tauri shell, the React UI hot-reloads, and you can start exploring.

---

## Architecture

ChaosEngineAI is three cooperating layers:

```
┌─────────────────────────────────────────────────────────┐
│  Tauri shell  (Rust)                                    │
│  ├─ React + TypeScript UI  (desktop/src)                │
│  ├─ In-app updater (signed releases from GitHub)        │
│  └─ Spawns and supervises the Python backend            │
└─────────────────────────────────────────────────────────┘
                          │  HTTP  /  IPC
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Python backend  (backend_service/)                     │
│  ├─ Workspace + library + warm pool state               │
│  ├─ OpenAI-compatible HTTP server                       │
│  └─ Engine adapters                                     │
└─────────────────────────────────────────────────────────┘
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
        ┌──────────────┐    ┌──────────────────┐
        │  llama.cpp   │    │   MLX runtime    │
        │  (GGUF)      │    │  (Apple Silicon) │
        └──────────────┘    └──────────────────┘
```

- **`desktop/`** — Tauri 2 + React 18 + TypeScript UI. Single-window workspace with a sidebar nav covering LLM and image generation screens.
- **`backend_service/`** — Python service that owns model lifecycle, the warm pool, the OpenAI-compatible API, and the benchmark runner.
- **`backend_service/cache_strategies/`** — Pluggable cache/compression strategy system. Ships with a native f16 strategy and optional adapters for TriAttention, RotorQuant, and MegaKernel.

---

## Building a Release

Releases are tag-driven. Push `vX.Y.Z` and the GitHub Actions release workflow builds signed bundles for macOS, Linux, and Windows in parallel, generates the `latest.json` updater manifest, and stages a draft release.

```bash
# locally, if you want to cut a build by hand:
cd desktop
npm run stage:runtime:release
npm run tauri:build
```

Release artifacts land in `desktop/src-tauri/target/release/bundle/`.

For an unsigned local macOS app + DMG without Apple signing/notarization or Tauri updater signing configured:

```bash
cd desktop
npm run release:macos -- --skip-sign --skip-notarize
```

That writes the local app + DMG to `desktop/releases/macos/`.

---

## Project Layout

```
ChaosEngineAI/
├── desktop/              Tauri + React desktop app
│   ├── src/              React UI (App.tsx is the workspace shell)
│   ├── src-tauri/        Rust shell + bundled runtime
│   ├── scripts/          Release + runtime staging
│   └── Screenshots/      UI screenshots used by this README
├── backend_service/      Python backend (engine adapters + HTTP server)
├── backend_service/cache_strategies/  Pluggable cache/compression strategy adapters
├── tests/                Backend integration tests
├── docs/                 Tour GIF + supporting docs
└── ChaosEngineAI_AppIcon.svg
```

---

## License & Credits

ChaosEngineAI is currently distributed under the Apache-2.0 license. See [`LICENSE`](./LICENSE).

See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for optional third-party dependency notes.

Built on the shoulders of [`llama.cpp`](https://github.com/ggerganov/llama.cpp), [Apple MLX](https://github.com/ml-explore/mlx), [Tauri](https://tauri.app/), and the broader open-weights community.
