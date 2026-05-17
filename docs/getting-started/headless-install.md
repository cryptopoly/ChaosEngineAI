# Headless install (CLI only)

You don't need the Tauri desktop shell to use ChaosEngineAI. The backend is
self-sufficient — install it into a Python virtualenv, run the FastAPI server,
and drive every feature through [`chaosengine-cli`](../cli/overview.md).

This is the recommended setup for servers, CI, and remote development boxes.

## Prerequisites

- Python 3.11 or newer
- `git`
- ~5 GB free disk (more if you plan to download models)

You do **not** need `rustc`, `node`, `npm`, or `cargo` for headless mode.

## Install

```bash
git clone https://github.com/cryptopoly/ChaosEngineAI.git
cd ChaosEngineAI
python3 -m venv .venv
.venv/bin/pip install -e .
```

The `-e .` editable install pulls every direct dependency listed in
`pyproject.toml`. Optional extras (MTPLX, DFlash, turboquant-mlx-full, etc.)
are installed separately on demand.

## Run the backend

```bash
./scripts/chaosengine-cli serve
```

This binds FastAPI to `127.0.0.1:8876` and runs in the foreground until you
hit `Ctrl-C`. To run it in the background you can use `nohup`, `tmux`, or a
systemd unit — the backend doesn't fork on its own.

Override host or port via environment variables or flags:

```bash
CHAOSENGINE_HOST=0.0.0.0 CHAOSENGINE_PORT=9000 ./scripts/chaosengine-cli serve

# or per-command
./scripts/chaosengine-cli --host 0.0.0.0 --port 9000 health
```

## Smoke test

```bash
./scripts/chaosengine-cli health
./scripts/chaosengine-cli routes | jq '.count'
./scripts/chaosengine-cli status | jq '.runtime.state'
```

A healthy install reports `"ok"`, `125` routes (as of v0.8.0), and a runtime
state of `"idle"`.

## Add a model

```bash
# Catalog search
./scripts/chaosengine-cli search "qwen3"

# Direct Hugging Face download
./scripts/chaosengine-cli download Qwen/Qwen3-4B

# Load + prompt
./scripts/chaosengine-cli load Qwen/Qwen3-4B
./scripts/chaosengine-cli prompt "Write a Rust quicksort." --max-tokens 256 --stream
```

## OpenAI-compatible endpoint

Once a model is loaded, the same backend exposes
`/v1/chat/completions`, `/v1/models`, and `/v1/embeddings`:

```bash
curl http://127.0.0.1:8876/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "loaded",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

Point any OpenAI client at `http://127.0.0.1:8876/v1` to use it.

## What you give up

A headless install skips:

- The Tauri shell (no system tray, no in-app updater, no signed bundle).
- The React UI (Image Studio, Video Studio, the Dashboard, etc. all live
  there — the headless backend still exposes the underlying HTTP endpoints).
- The cross-platform signed installer pipeline.

Every feature in the desktop app is reachable through the same FastAPI
endpoints — see the [CLI reference](../cli/reference.md) for the per-route
mapping and the [API reference](../reference/api.md) for the raw HTTP surface.
