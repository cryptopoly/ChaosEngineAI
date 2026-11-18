# Server mode (OpenAI-compatible)

The backend always exposes an OpenAI-compatible HTTP surface — it's the same
endpoints the Tauri shell uses internally, and they're reachable to any
OpenAI client that lets you change the base URL.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Standard OpenAI chat completion, streaming or non-streaming. |
| `GET /v1/models` | List the currently loadable models (warm pool + library). |
| `POST /v1/embeddings` | Run `llama-embedding` against text inputs. |

Plus the native ChaosEngineAI surface (`/api/chat/generate`, `/api/runtime`,
etc.) for anything OpenAI doesn't model — speculative decoding state, cache
strategy reporting, runtime telemetry. See the
[API reference](../reference/api.md) for the full list.

## Starting the server

The Server tab in the desktop app has two views, Overview and Activity
(the view you last used is remembered).

**Overview:**

- **Connect your app**: one line with the base URL and Copy; open it for the
  Python, JavaScript and config snippets (it stays open or closed as you left it).
- **Recent activity**: the last few model events, with a way into Activity.
- **Bind address** (default `127.0.0.1:8876`).
- **Preferred port** override.
- **LAN exposure** toggle — flips the bind to `0.0.0.0`.
- **Auto-start** flag — start the server on app launch.
- **Warm pool** state — which models are hot and instant-loadable.
- **Live counters** — request count, active connections.
- **Remote test panel** — copyable curl commands for `/health`, `/models`,
  and `/chat/completions`.

**Activity** (this was the Logs page):

- **Model timeline**: one bar per model with its loading, ready, generating,
  idle and warm phases, over 15 minutes, an hour, a day or everything kept.
- **Events**: requested, resolving, loading (weights size), loaded (seconds,
  memory), kept warm, first token (prefill time), generation (tokens per
  second), idle, unloaded (by you, another model, to free memory) and
  failures, each with its request id, engine, cache strategy and speculative
  method. Filter by model, level, request id or text; open a row for the full
  command line, the engine's environment and its last output.
- **Copy selected**, **Export selected** (JSON), **Pause** the live tail,
  **Latest failure**.
- **Log lines**: the plain backend log, searchable.

Events are kept in memory (the newest 1000) and in
`<data dir>/logs/model-events.jsonl` (rotated at 1 MB, three old files kept).
Tokens, API keys, secret file paths and the home folder are hidden before an
event is stored. Read them with `GET /api/server/events` (filters `model`,
`level`, `kind`, `requestId`, `since`, `limit`), the stream
`GET /api/server/events/stream`, or `chaosengine-cli server-events`.

Headless install? `./scripts/chaosengine-cli serve` runs the same backend
without the Tauri shell.

## Sampler parity

The OpenAI-compatible endpoint accepts the full ChaosEngineAI sampler chain:

- `temperature`, `top_p`, `top_k`, `min_p`
- `repeat_penalty`, `frequency_penalty`, `presence_penalty`
- `seed`, `mirostat` (mode + tau + eta), `reasoning_effort`
- `response_format` with JSON schema for constrained output

Pass them in the request body — they pass through to the engine without
mangling.

## Integration snippets

The Settings tab carries copy-paste snippets for common downstream tools:

```bash
# Continue.dev / Cursor / Goose / Claude Code — point at http://127.0.0.1:8876/v1
```

```python
# OpenAI Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8876/v1",
    api_key="not-used-locally",
)

resp = client.chat.completions.create(
    model="loaded",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

```bash
# curl streaming
curl -N http://127.0.0.1:8876/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "loaded",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

## Security

By default the server binds to `127.0.0.1` — only your own machine can reach
it. Set `CHAOSENGINE_HOST=0.0.0.0` (or flip the LAN exposure toggle in the
Server tab) to expose it on the network. There's currently no built-in auth;
put it behind a reverse proxy with TLS + basic auth if you expose it beyond
your own machine.

The backend itself never reaches outbound on its own. The only outbound
traffic is Hugging Face for model downloads (when you trigger one) and
optional remote OpenAI-compatible providers (when you configure one in
Settings).
