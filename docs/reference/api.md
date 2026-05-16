# HTTP API reference

The ChaosEngineAI backend exposes 125 routes across two API surfaces:

- **`/api/*`** — native ChaosEngineAI endpoints used by the Tauri frontend
  and the CLI. The richest surface; everything the desktop app does is
  reachable here.
- **`/v1/*`** — OpenAI-compatible shim. `chat/completions`, `models`,
  `embeddings`. Point any OpenAI client at `http://127.0.0.1:8876/v1/`.

## Live schema

The backend serves its own OpenAPI document at:

```
GET http://127.0.0.1:8876/openapi.json
```

Pretty-print it:

```bash
./scripts/chaosengine-cli openapi | jq '.paths | keys'
```

The interactive Swagger UI lives at `http://127.0.0.1:8876/docs` and the
ReDoc rendering at `http://127.0.0.1:8876/redoc` when the backend is
running.

## Endpoint inventory

For the full per-route table, see the [CLI reference](../cli/reference.md)
— it's generated from the same OpenAPI document and groups every endpoint
by prefix.

Quick links to the key surfaces:

| Surface | Key endpoints |
|---|---|
| Health + workspace | `GET /api/health`, `GET /api/workspace`, `GET /api/runtime` |
| Models | `GET /api/models/search`, `POST /api/models/load`, `POST /api/models/unload`, `POST /api/models/download`, `POST /api/models/convert` |
| Chat | `POST /api/chat/generate`, `POST /api/chat/generate/stream`, `POST /api/chat/compare`, `POST /api/chat/sessions` |
| HTML Challenge | `GET/POST /api/chat/html-challenges`, `POST .../slots/{slot_id}/repair`, `POST .../slots/{slot_id}/retry` |
| OpenAI shim | `POST /v1/chat/completions`, `GET /v1/models`, `POST /v1/embeddings` |
| Image | `POST /api/images/generate`, `GET /api/images/progress`, `GET /api/images/outputs`, `GET /api/images/catalog`, `GET /api/images/library` |
| Video | `POST /api/video/generate`, `GET /api/video/progress`, `GET /api/video/outputs`, `GET /api/video/catalog`, `GET /api/video/mlx-runtime` |
| Setup | `POST /api/setup/install-mtplx`, `POST /api/setup/install-mlx-video-wan`, `POST /api/setup/install-longlive`, `POST /api/setup/refresh-capabilities` |
| Diagnostics | `GET /api/diagnostics/snapshot`, `GET /api/diagnostics/log-tail`, `POST /api/diagnostics/reextract-runtime` |
| Settings | `GET/PATCH /api/settings`, `GET/POST /api/settings/storage` |
| Plugins / tools | `GET /api/plugins`, `POST /api/plugins/{id}/enable`, `GET /api/tools`, `GET /api/adapters` |
| Server | `GET /api/server/status`, `POST /api/server/shutdown`, `GET /api/server/logs/stream` |

## Authentication

The backend doesn't require auth — it binds to `127.0.0.1` by default and
trusts every request. The `/api/auth/session` endpoint exists for the UI's
own state-tracking; it's not a security boundary.

If you expose the backend on the network (`CHAOSENGINE_HOST=0.0.0.0` or the
LAN exposure toggle in the Server tab), put it behind a reverse proxy with
TLS and basic auth. There's no built-in auth surface.

## Long-running endpoints

Three patterns:

**Synchronous request / response.** Most endpoints. The request blocks
until the backend has a final answer; the response carries the result.
Examples: `POST /api/chat/generate`, `GET /api/runtime`.

**Server-Sent Events (SSE).** Streaming generation + log streaming. The
response is a `text/event-stream` of JSON-encoded events. Examples:
`POST /api/chat/generate/stream`, `GET /api/server/logs/stream`.

**Background job + polling.** Long installers + downloads + conversions
+ video generation. The POST endpoint kicks off the job and returns
immediately with a job ID; the GET endpoint reports progress. Examples:
`POST /api/setup/install-mtplx` + `GET /api/setup/install-mtplx/status`,
`POST /api/images/generate` + `GET /api/images/progress`.

## Error shape

All errors are wrapped in a consistent envelope:

```json
{
  "detail": "Human-readable error message.",
  "runtimeNote": "engine + binary + cache strategy at the point of failure"
}
```

`runtimeNote` is present on inference-path errors; not present on
validation / not-found errors. The HTTP status code follows REST
conventions — 4xx for caller bugs, 5xx for backend issues.

## Rate limiting

None. The backend trusts every caller. If you need rate limiting, put
it in the reverse proxy in front.

## Versioning

The native `/api/*` surface is **not** versioned — backwards compatibility
is on a best-effort basis. The OpenAI-compatible `/v1/*` surface is
versioned by virtue of the prefix; new OpenAI features land additively.

Breaking changes to either surface land in `CHANGELOG.md` with a
migration note.

## See also

- [CLI reference](../cli/reference.md) — endpoint-by-endpoint listing.
- [Server mode](../usage/server-mode.md) — OpenAI-compatible workflow.
- [Architecture overview](../architecture/overview.md) — how routes map
  to backend modules.
