# Environment variables

ChaosEngineAI reads a small number of `CHAOSENGINE_*` environment variables
to override defaults. None are required for normal operation; they're
escape hatches for development, packaging, and multi-host deployments.

## Backend host + port

| Variable | Default | Purpose |
|---|---|---|
| `CHAOSENGINE_HOST` | `127.0.0.1` | Bind host for the FastAPI backend. Set to `0.0.0.0` to expose on the LAN. |
| `CHAOSENGINE_PORT` | `8876` | Bind port. |

Both are honoured by `chaosengine-cli serve` (the backend) and by the CLI
itself for outgoing calls.

## Binary paths

| Variable | Default | Purpose |
|---|---|---|
| `CHAOSENGINE_LLAMA_SERVER` | auto | Override the standard `llama-server` path. Auto-resolution order: env override → `~/.chaosengine/bin/` → bundled runtime → `PATH`. |
| `CHAOSENGINE_LLAMA_SERVER_TURBO` | `~/.chaosengine/bin/llama-server-turbo` | Override the TurboQuant fork binary. |
| `CHAOSENGINE_LLAMA_CLI` | auto | Override the `llama-cli` path (used for cache-type probing). |
| `CHAOSENGINE_LLAMA_BIN_DIR` | `../llama.cpp/build/bin/` | Build directory used by the runtime staging script. |
| `CHAOSENGINE_MLX_PYTHON` | `.venv/bin/python` | Python interpreter for the MLX worker. Set this when you're running with a non-standard venv layout. |

## Storage paths

| Variable | Default | Purpose |
|---|---|---|
| `CHAOSENGINE_MLX_VIDEO_WAN_DIR` | `~/.chaosengine/mlx-video-wan/` | Output root for the Wan one-shot convert pipeline. |
| `HF_HOME` | `~/.cache/huggingface/` | Standard Hugging Face cache root. Honoured by `huggingface_hub` directly. |
| `HF_TOKEN` | unset | Hugging Face access token for gated models. Persisted from Settings → Hugging Face token; can also be set in the environment for non-interactive runs. |

## Notes

- Environment variables take precedence over `Settings → Storage` values
  for paths. If you set `CHAOSENGINE_LLAMA_SERVER_TURBO`, the launch modal
  won't second-guess you — the binary at that exact path is what's used.
- The backend re-reads env vars on every process startup, **not** on
  every request. After changing an env var, restart the backend.
- Removed in FU-030: `CHAOSENGINE_VENDOR_PATH`. The deprecated ChaosEngine
  vendor path is no longer used.

## See also

- [Runtime paths](../architecture/runtime-paths.md) — where each path
  resolves to in practice.
- [Headless install](../getting-started/headless-install.md) — common
  env-var patterns.
