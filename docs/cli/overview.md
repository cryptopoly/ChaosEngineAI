# CLI overview

`scripts/chaosengine-cli` is a Python 3 wrapper (stdlib only, zero new
dependencies) that talks to the same FastAPI backend the Tauri shell uses.
It covers 100% of the 125 backend routes through a generic `call <METHOD>
<PATH>` dispatcher plus 95 ergonomic typed shortcuts.

JSON is written to stdout; errors to stderr. Exit 0 on success, non-zero on
failure. The tool is composable with `jq` or any other pipeline tool.

## Quick start

```bash
# Make sure the backend is running (the Tauri app launches it for you, or
# you can run it standalone)
./scripts/chaosengine-cli serve &

# Probe
./scripts/chaosengine-cli health
./scripts/chaosengine-cli status | jq '.runtime.state'

# Discover + load
./scripts/chaosengine-cli search "qwen3"
./scripts/chaosengine-cli load Qwen/Qwen3-4B --spec

# Generate
./scripts/chaosengine-cli prompt "Write a Rust quicksort" \
    --max-tokens 256 --stream --metrics
```

Add a PATH symlink so you can invoke it from anywhere:

```bash
ln -s /path/to/ChaosEngineAI/scripts/chaosengine-cli ~/.local/bin/chaosengine-cli
```

## Two layers

### 1. Generic dispatcher — `call`

Every route registered with the backend is reachable through `call`. No
matter what new endpoints land in `backend_service/routes/`, they're
available without a CLI update:

```bash
./scripts/chaosengine-cli call GET /api/health
./scripts/chaosengine-cli call POST /api/models/load \
    --body '{"modelRef":"Qwen/Qwen3-4B"}'
./scripts/chaosengine-cli call PATCH /api/settings \
    --body '{"defaultContext": 32768}'
```

The dispatcher takes `--body` (JSON string), `--body-file` (read JSON from
a file), `--raw-body` (raw bytes from stdin), and `--content-type` for
non-JSON payloads.

### 2. Typed shortcuts

Ergonomic subcommands for common workflows. Each one wraps a specific
endpoint with a friendly flag set:

| Category | Subcommands |
|---|---|
| Lifecycle | `serve`, `status`, `health`, `runtime`, `load`, `unload` |
| Catalog | `search`, `hub-search`, `hub-files`, `list-weights`, `quantized-variants` |
| Library | `download`, `download-status`, `download-cancel`, `download-delete`, `convert`, `reveal`, `delete-model` |
| Chat | `prompt`, `bench`, `benchmark-run`, `chat-cancel`, `compare` |
| Sessions | `session-create`, `session-rename`, `session-delete`, `session-fork`, `session-variant`, `session-delve`, `session-documents`, `session-document-upload`, `session-document-delete` |
| HTML Challenge | `challenges-list`, `challenges-get`, `challenges-file`, `challenges-create`, `challenges-open-file`, `challenges-repair`, `challenges-retry`, `challenges-validate`, `challenges-delete` |
| Server | `server-status`, `server-shutdown`, `server-logs` |
| Image Studio | `image-generate`, `image-progress`, `image-cancel`, `image-outputs`, `image-output-get`, `image-output-delete`, `image-runtime`, `image-unload`, `image-library`, `image-catalog`, `image-preload`, `image-download`, `image-download-status`, `image-download-cancel`, `image-download-delete` |
| Video Studio | `video-generate`, `video-progress`, `video-cancel`, `video-outputs`, `video-output-get`, `video-output-file`, `video-output-delete`, `video-runtime`, `video-library`, `video-catalog`, `video-mlx-runtime`, `video-longlive`, `video-download`, `video-download-status`, `video-download-cancel`, `video-download-delete` |
| Setup | `mtplx-install`, `mtplx-status`, `longlive-install`, `longlive-status`, `wan-install`, `wan-status`, `wan-inventory`, `cuda-torch-install`, `gpu-bundle-install`, `gpu-bundle-status`, `gpu-bundle-info`, `setup-install-package`, `setup-install-system-package`, `setup-refresh`, `turbo-update-check` |
| Diagnostics | `diagnostics-snapshot`, `diagnostics-log-tail`, `diagnostics-reextract`, `gpu-status`, `metrics-gpu`, `cache-preview` |
| Other | `prompts-list`, `prompts-create`, `prompts-delete`, `prompts-enhance`, `settings-get`, `settings-patch`, `settings-storage-get`, `settings-storage-set`, `settings-storage-move`, `settings-storage-move-status`, `workspaces-list`, `workspaces-rename`, `workspaces-delete`, `workspaces-document-delete`, `plugins-list`, `plugins-enable`, `plugins-disable`, `tools-list`, `adapters-list`, `finetuning-status`, `finetuning-start`, `auth-session`, `routes`, `openapi`, `v1-models` |

Run `./scripts/chaosengine-cli <subcommand> --help` for the flags a specific
shortcut accepts.

## Host + port

The CLI defaults to `127.0.0.1:8876`. Override via env vars or flags:

```bash
# Env (persists for the shell)
export CHAOSENGINE_HOST=0.0.0.0
export CHAOSENGINE_PORT=9000

# Per-command
./scripts/chaosengine-cli --host 192.168.1.5 --port 9000 status
```

`serve` honours the same overrides.

## Output

- **stdout** — structured JSON (sometimes raw text for streaming subcommands
  like `prompt --stream`). Pipe to `jq` for filtering.
- **stderr** — human-readable error messages, phase logs from long-running
  installers, and the streaming phase events from `image-progress` /
  `video-progress` polling helpers.
- **exit code** — 0 on success; non-zero on backend error (the body of
  the error response goes to stderr).

## See also

- [CLI reference](reference.md) — every endpoint, grouped by prefix.
- [CLI recipes](recipes.md) — common workflows end-to-end.
- [Automation](automation.md) — CI patterns and tips.
