# First run

The first time ChaosEngineAI launches it sets up local state, probes the
machine for inference capabilities, and registers itself with the in-app
updater. Here's what happens, in order, so you know what to expect.

## What the app does

1. **Extract runtime.** On macOS / Windows / Linux signed builds, the Tauri
   shell stages a bundled Python runtime + `llama-server` + (when available)
   `llama-server-turbo` + `sd` (stable-diffusion.cpp) under a workspace
   directory. Dev builds (`npm run tauri:dev`) use the editable `.venv` at
   the repo root instead.
2. **Spawn the FastAPI backend.** The Tauri Rust shell starts the Python
   backend on `127.0.0.1:8876` and waits for `/api/health` to return 200.
3. **Probe capabilities.** The backend introspects which engines are usable
   on this machine — `mlx`, `mlx_lm`, `gguf` (Homebrew or bundled
   llama-server), `llama-server-turbo`, `mtplx`, `vllm` — and caches the
   result for the lifetime of the process.
4. **Read settings.** Settings live in
   `~/Library/Application Support/com.chaosengineai.desktop/settings.json`
   on macOS (analogous paths on Linux / Windows). On a fresh install the
   defaults are written here.
5. **Open the Dashboard tab.** This is the launchpad. Big colored badges
   tell you backend state, the engine selected for the next load, hardware
   summary, and warm-pool stats.

## Where things land on disk

| Path | Purpose |
|---|---|
| `~/.chaosengine/bin/` | Local binaries — `llama-server-turbo`, `sd` (stable-diffusion.cpp), MTPLX runner. |
| `~/.chaosengine/mtplx-venv/` | Isolated MTPLX virtualenv (ships its own forked `mlx`). |
| `~/.chaosengine/mlx-video-wan/<slug>/` | Converted Wan 2.1 / 2.2 mlx-video checkpoints. |
| `~/.chaosengine/test-results/` | E2E suite JSON + Markdown reports. |
| `~/AI_Models/` | Default model directory (configurable in Settings). |
| `~/.cache/huggingface/` | Standard Hugging Face cache used for downloads. |

You can move the data directory from **Settings → Storage**; the backend will
copy existing models to the new path before switching over.

## First model

The Dashboard's empty state nudges you to the **Discover** tab. Pick a model
family, expand to see quant variants, and queue a download. Smaller models
(Qwen3-4B 4-bit MLX, ~2.3 GB) finish in a couple of minutes on a normal
connection and are a good way to verify the full pipeline.

When the download lands, the launch modal pre-populates with sensible defaults
for the model's architecture — engine, context length, cache strategy,
sampling presets. Hit **Launch**, wait for the warm-pool entry to flip green,
and head over to the **Chat** tab.

## First-launch troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Dashboard shows red "Backend offline" | Bundled Python failed to extract | Settings → Diagnostics → **Re-extract runtime**. |
| No GPU detected on a known-GPU machine | Capabilities probe ran before drivers loaded | Restart the app, or Settings → Diagnostics → **Refresh capabilities**. |
| Model load fails with "MLX is not available" | The bundled Python's `mlx` didn't import | Confirm Apple Silicon hardware; on Intel Macs / x86 Linux, use GGUF models with `llama.cpp` instead. |
| Updater banner won't dismiss | An update check is still in flight | Wait ~30 s for it to finish, or restart the app — the banner is sticky during the check. |

See [Troubleshooting](../troubleshooting/faq.md) for the longer list.

## Next steps

- [System requirements](system-requirements.md) — what each platform / runtime
  combination supports.
- [Chat walk-through](../usage/chat.md) — the most-used tab.
- [CLI overview](../cli/overview.md) — drive the same backend from a terminal.
