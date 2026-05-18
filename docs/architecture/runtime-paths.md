# Runtime paths

Where ChaosEngineAI keeps its state on disk. Most paths are configurable
from Settings → Storage; the defaults are listed here.

## App data root — `~/.chaosengine/`

The app's managed root. Lives under your home directory on every
platform.

```
~/.chaosengine/
├── bin/                          Managed binaries
│   ├── llama-server-turbo        TurboQuant fork (built by scripts/build-llama-turbo.sh)
│   ├── llama-cli-turbo           CLI companion (same fork)
│   ├── sd                        stable-diffusion.cpp (built by scripts/build-sdcpp.sh)
│   └── mtplx.version             Version marker written by the MTPLX installer
├── mtplx-venv/                   Isolated venv for MTPLX (Apple Silicon)
│   └── bin/                      mtplx executable + forked mlx
├── mlx-video-wan/                Converted Wan 2.1 / 2.2 checkpoints
│   ├── wan-2-1-t2v-1-3b/         (one subdir per converted Wan repo)
│   └── wan-2-2-ti2v-5b/
└── test-results/                 E2E suite reports
    ├── e2e-YYYYMMDD-HHMMSS.json  Machine-readable
    └── e2e-YYYYMMDD-HHMMSS.md    Human-readable summary
```

Override the `mlx-video-wan/` location with `CHAOSENGINE_MLX_VIDEO_WAN_DIR`.

## Model directory — `~/AI_Models/`

The default location for downloaded model weights. Configurable from
**Settings → Storage**; the data-directory migration helper copies
existing models to the new location before switching over.

```
~/AI_Models/
├── Qwen/
│   ├── Qwen3-4B/                 Safetensors checkpoints
│   └── Qwen3.5-14B/
├── mlx-community/
│   └── Qwen3.6-27B-4bit/         MLX quantised checkpoints
├── lmstudio-community/
│   └── Qwen3-Coder-Next-MLX-4bit/
├── black-forest-labs/
│   └── FLUX.1-schnell/
└── Wan-AI/
    └── Wan2.1-T2V-1.3B/          Raw video diffusion checkpoints
```

The scanner also picks up Hugging Face's standard cache at
`~/.cache/huggingface/hub/` so downloads triggered by the `transformers`
or `huggingface_hub` libraries surface in the library view too.

## Settings — platform-specific

Settings persist alongside the OS-conventional config dir:

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/com.chaosengineai.desktop/settings.json` |
| Linux | `~/.config/com.chaosengineai.desktop/settings.json` |
| Windows | `%APPDATA%\com.chaosengineai.desktop\settings.json` |

Settings include: default cache strategy, default context length,
default fused-attention state, the storage directory list, Hugging
Face tokens (for gated models), remote provider configs, UI scale,
integrations metadata, plugin enable / disable state.

Hugging Face tokens are written to the file in plaintext — there's no
keyring integration yet. If multi-user / shared-disk security matters
for your setup, keep tokens out of the persisted settings and inject
them via `HF_TOKEN` at backend launch time.

## Runtime staging — `src-tauri/resources/embedded/`

In release builds, Tauri bundles a Python runtime + `llama-server` +
optional `llama-server-turbo` + optional `sd` (stable-diffusion.cpp).
`scripts/stage-runtime.mjs` populates this directory before the build.
At first launch, the Tauri shell extracts the staged runtime to a
workspace directory.

Dev builds (`npm run tauri:dev`) skip the bundled runtime and use the
editable `.venv` at the repo root instead.

## Environment variable overrides

| Variable | Default | Purpose |
|---|---|---|
| `CHAOSENGINE_HOST` | `127.0.0.1` | Backend bind host. |
| `CHAOSENGINE_PORT` | `8876` | Backend port. |
| `CHAOSENGINE_LLAMA_SERVER` | auto | Override standard `llama-server` path. |
| `CHAOSENGINE_LLAMA_SERVER_TURBO` | `~/.chaosengine/bin/llama-server-turbo` | Override turbo binary path. |
| `CHAOSENGINE_LLAMA_CLI` | auto | Override `llama-cli` path. |
| `CHAOSENGINE_MLX_PYTHON` | `.venv/bin/python` | Python interpreter for MLX. |
| `CHAOSENGINE_LLAMA_BIN_DIR` | `../llama.cpp/build/bin/` | Build directory for staging script. |
| `CHAOSENGINE_VENDOR_PATH` | (removed in FU-030) | Was for ChaosEngine vendor; no longer used. |
| `CHAOSENGINE_MLX_VIDEO_WAN_DIR` | `~/.chaosengine/mlx-video-wan/` | Wan conversion output root. |

These take precedence over the binary-resolution chain (PATH, Homebrew,
bundled, `~/.chaosengine/bin/`).

## Logs

The backend log buffer is in-memory (capped) and exposed via the
`/api/diagnostics/log-tail` endpoint. Persistent logs are not written
to disk by default — if you need them on disk, redirect the backend's
stdout / stderr:

```bash
./scripts/chaosengine-cli serve > /var/log/chaosengine.log 2>&1
```

The macOS spam suppression in
[`backend_service/routes/diagnostics.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/diagnostics.py)
strips `MallocStackLogging` / `MallocScribble` noise from the log tail
+ snapshot so the diagnostics surface stays clean even on hardened-
runtime builds.

## See also

- [System requirements](../getting-started/system-requirements.md) — disk
  sizing.
- [Environment variables](../reference/env-vars.md) — full reference.
