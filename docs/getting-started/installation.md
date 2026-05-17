# Installation

ChaosEngineAI ships as a signed desktop application for macOS, Linux, and
Windows, plus a self-contained source tree that can run headlessly on any
platform that supports Python 3.11+.

## Option 1 — Signed desktop installer

The fastest path is the release page:

[Latest release on GitHub](https://github.com/cryptopoly/ChaosEngineAI/releases/latest)

| Platform | File | Notes |
|---|---|---|
| **macOS** (Apple Silicon) | `ChaosEngineAI_*_aarch64.dmg` | Signed + notarized |
| **Linux** (portable) | `ChaosEngineAI_*_amd64.AppImage` | In-app updates supported |
| **Linux** (Debian / Ubuntu) | `ChaosEngineAI_*_amd64.deb` | Install via `dpkg`, update via apt |
| **Windows** | `ChaosEngineAI_*_x64-setup.exe` | Unsigned for now — SmartScreen will warn on first run |

From v0.4.21 onward every install auto-updates from GitHub Releases on launch.
Updates are cryptographically signed.

## Option 2 — Build from source

Use this path if you want to hack on the app, ship custom builds, or run on a
platform without a prebuilt installer.

### Prerequisites

- Rust toolchain (`rustup`, stable channel)
- Node.js 20+ and `npm`
- Python 3.11+ (3.13 also tested)
- On macOS: Xcode command-line tools (`xcode-select --install`)
- On Linux: standard build tools (`build-essential`, `libssl-dev`,
  `libwebkit2gtk-4.1-dev`, `librsvg2-dev`)

### Bootstrap

```bash
git clone https://github.com/cryptopoly/ChaosEngineAI.git
cd ChaosEngineAI

# 1. Python backend
python3 -m venv .venv
.venv/bin/pip install -e .

# 2. JS dependencies
npm install

# 3. Stage the bundled Python runtime + llama.cpp binaries into src-tauri/
npm run stage:runtime

# 4. Launch the desktop app in dev mode (Tauri shell + Vite HMR)
npm run tauri:dev
```

The Python backend is spawned by the Tauri shell, the React UI hot-reloads,
and the dashboard tab should turn green within a few seconds.

### Optional extras

| Capability | Install command | Notes |
|---|---|---|
| MLX inference on Apple Silicon | Already included via `pip install -e .` | Bundled by default; nothing extra to do. |
| TurboQuant MLX cache | `.venv/bin/pip install turboquant-mlx-full` | Apple Silicon only. |
| TurboQuant llama.cpp cache | `scripts/build-llama-turbo.sh` | Builds the `llama-server-turbo` fork into `~/.chaosengine/bin/`. |
| TriAttention | `.venv/bin/pip install triattention vllm` | Linux + CUDA only (via vLLM). |
| MTPLX speculative decoding | Setup tab → **Install MTPLX**, or `scripts/install-mtplx.sh` | Apple Silicon only, isolated venv at `~/.chaosengine/mtplx-venv/`. |
| DFlash speculative decoding | `.venv/bin/pip install dflash-mlx` (Apple Silicon) or `dflash` (CUDA) | A draft checkpoint is auto-resolved per target model. |
| stable-diffusion.cpp image / video | `scripts/build-sdcpp.sh` | Builds `sd` into `~/.chaosengine/bin/`. |

After installing an extra, restart the backend (or the whole Tauri app) so
the capabilities probe picks up the new feature.

## Verify

Once the app is open:

1. The **Dashboard** should report a green backend status, a detected
   Python runtime path, and the platform / arch.
2. **Settings → Diagnostics** should list every installed extra under
   "Capabilities".
3. From a terminal, `./scripts/chaosengine-cli health` should print
   `{"status": "ok", ...}`.

If any of the above fail, jump to [Troubleshooting](../troubleshooting/faq.md).
