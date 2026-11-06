# Agent bridge: MCP tools for the Image, Video and Voice APIs

Status: scoped 2026-11-06, not started. Follows the agent-made advert demo (PR 127,
`docs/examples/agent-ad-demo/`) and the FU-126 idea. Decision: `docs/hub/decisions.md` item 7.

## Goal

Any user's agent (Claude Code, Claude Desktop, Cursor, any MCP client) can make images, video clips,
speech and transcripts with ChaosEngineAI through typed tools, without reading our source or docs, and
without crashing the machine by running heavy jobs at the same time.

## What the demo showed

The HTTP API already works for an agent: 10 posters (FLUX 3 Klein), a 30 s advert from LTX-2.5 clips and
a Kokoro voiceover, all from `POST /api/images/generate`, `/api/video/generate` and `/api/voice/synthesize`.
The gaps were not missing features but friction and safety:

1. **Discovery.** The agent read source files to learn request fields, model ids, valid frame counts
   (LTX 8k+1) and voice ids.
2. **Response shapes differ.** Images return `{"artifacts": [...]}` with a base64 data URI; video returns
   `{"artifact": {...}}` and the file comes from a second call; TTS returns raw WAV bytes.
3. **Long blocking calls.** A 4 s LTX clip holds one HTTP request for about 2 minutes.
4. **No job serialisation.** Six images, five clips and browser renders queued together crashed the Mac
   (cause not confirmed). One at a time, with a free-memory check, was stable.
5. **Auth was unclear.** Bearer auth is on by default, yet the demo's calls worked without a token (not
   checked why). An agent needs one clear way in.

## Options

| | What | For | Against |
|---|---|---|---|
| A | MCP endpoint inside the backend (`/mcp`, streamable HTTP) | One process; the tools see the real job queue, memory gate and loaded models; nothing extra to install | Needs the app (or headless backend) running; needs the token in the client config |
| B | stdio MCP server (`chaosengine-cli mcp`) that forwards to the HTTP API | Works with clients that only start stdio servers; can read the token from the app's settings itself | A second process; duplicates schemas unless it reads them from A |
| C | A Claude skill only (`SKILL.md` + the `chaos_api.py` helper) | Cheapest (half a day); no protocol code | Claude-only; no typed schemas; no shared queue; the agent still assembles requests by hand |

**Recommendation: A, plus a thin B, plus a short skill that points at the tools.** A carries the real
work (schemas, queue, guards). B is about 100 lines: a stdio-to-HTTP relay for clients that cannot do HTTP
MCP. The skill is one page that tells Claude when to use the tools and the memory rule. C alone does not
fix points 3 and 4.

## v1 tool set

| Tool | Does | Notes |
|---|---|---|
| `system_status` | Free memory, platform, loaded models, running job, whether agent access is on | Lets the agent decide before it asks |
| `recommend_models` | For a kind (`image`, `video`, `voice`), the best not-installed models for this machine, with size, licence, gated flag and fit | Read-only; uses the fit ratings (FU-133). The agent relays it to the user, who downloads from the Models page |
| `list_models` | Installed models by kind (`image`, `video`, `voice`, `stt`) with id, size, fit, sizes / frame rules | Only installed and runnable on this host; no downloads in v1 |
| `list_voices` | TTS voices with language and accent | |
| `generate_image` | Prompt, model, size, seed, steps (optional) → job id | Sizes snapped as the UI does; notes returned |
| `generate_video` | Prompt, model, size, frames or seconds, seed → job id | Seconds converted to the model's frame rule |
| `speak` | Text, voice, speed → job id | Short text may finish before the first poll |
| `transcribe` | Path to an audio or video file → job id | Path must be readable by the backend user |
| `job_status` | Job id → state, progress, position in queue, notes, output | |
| `wait_for_job` | Job id, timeout ≤ 60 s → same as `job_status` | Polling without busy loops; the agent calls it again |
| `cancel_job` | Job id | Uses the existing cancel routes |

**Outputs are file paths, not bytes.** Each finished job returns the absolute path of the PNG, MP4 or WAV
under the app's output folders plus width, height, duration and `hasAudio`. Base64 only on request
(`include_preview: true`, images only). A remote client cannot read the path; that is a v2 concern.

**Out of scope for v1:** model downloads, chat (the OpenAI-compatible API already covers it), editing
(ffmpeg stays the agent's job), image-to-video.

**Why no downloads (decided 2026-11-06).** One call could start a 30 to 60 GB download, gated models need
the licence accepted on Hugging Face by the user, some licences are non-commercial, and nothing checks the
disk first. A user is expected to set the app up before pointing an agent at it; when an agent finds no
suitable model it calls `recommend_models` and tells the user which one to get and why. v2 may add a
`download_model` tool that shows size and licence and asks the user to confirm in the agent's chat, and
refuses gated or non-commercial models whose licence the user has not accepted.

## Guards built in

- **One heavy job at a time.** A single in-process queue for image, video and voice jobs started through
  the bridge. A new job returns its queue position instead of starting next to another. Whether the UI's
  own jobs join the same queue is an open question (see below).
- **Memory gate first.** The existing image and video gates run before a job leaves the queue; a refusal
  comes back as a readable tool error with the numbers, not a crash.
- **Same corrections as the UI.** Size snapping, frame rules, minimum sizes and step caps from
  `generation_limits.py` and `request_guards.py`; every correction is listed in the job's notes.

## Auth and setup

- Loopback only by default (as the API today). The bridge uses the same bearer token.
- **Off until the user turns it on (decided 2026-11-06).** When off, `/mcp` still answers: `initialize` and
  `tools/list` work, and every tool call returns one clear error, "Agent access is off. Turn it on in
  ChaosEngineAI: Settings > Integrations > Agent access.", so the agent can tell the user what to do instead
  of seeing a connection failure. The stdio relay does the same when the app is not running ("start
  ChaosEngineAI").
- Settings > Integrations gets an "Agent access" card: on/off switch, copy buttons for the
  `claude mcp add --transport http chaosengine http://127.0.0.1:8876/mcp --header "Authorization: Bearer …"`
  line and a JSON block for other clients, and a token rotate button.
- Why the demo worked without a token: the author's settings had `requireApiAuth: false`. Not a bug. `/mcp` is outside the `/api/` and `/v1/` auth middleware, so the bridge must check the token itself.

## Implementation sketch

- `backend_service/mcp_bridge/`: tool definitions, the job queue, result shaping. Calls the same internal
  functions the routes call, not the HTTP routes. Lazy imports; nothing heavy at startup.
- Protocol: the official `mcp` Python SDK (MIT; check the version and licence before adding it, and add it to
  `THIRD_PARTY_NOTICES.md`), mounted as a streamable HTTP app at `/mcp`. The fallback if the SDK fights
  FastAPI: hand-written JSON-RPC for `initialize`, `tools/list`, `tools/call` (we already have a JSON-RPC
  helper in `backend_service/inference/jsonrpc.py`).
- `scripts/chaosengine-cli mcp`: stdio relay that reads the port and token from the running app.
- `docs/features/agent-bridge.md`, plus a `SKILL.md` users can drop into `~/.claude/skills/`.

## Related work

- **Generation settings inside the output files (v1, standalone, about half a day).** Write prompt, model,
  seed, steps, guidance, size and app version into PNG text chunks and MP4 metadata tags, so any output can
  be reproduced by a person or an agent, and dropping a file back into a Studio could restore its settings
  later. Nothing writes this today.
- **Recipes (v2).** A saved, replayable list of steps (image, then video, then voice) as a small data file,
  so a multi-step job like the advert demo can be rerun with new prompts. Not a node editor.
- **Reusing unchanged work (later, measure first).** Keeping a text encoder's output between runs that
  change only the seed could cut repeat-generation time; measure before building.
- **No arbitrary code from agents.** The tool set stays fixed and reviewed; agents cannot load plugins or
  run code inside the app.

## Rollout (v1)

Built on a feature branch (`feat/agent-bridge`) cut from `release/v0.9.10`, merged when the gate below
passes, shipped in the next release after v0.9.10. The shared queue changes how the Studios start jobs,
so it should not ride into a release that is already in beta.

| Step | What | Days | Done when |
|---|---|---|---|
| 0 (done 2026-11-06) | Find why the demo's calls worked without a token; write generation settings into PNG and MP4 files | 0.5 | Auth behaviour explained (fixed if it is a bug); outputs carry their settings |
| 1 (done 2026-11-06) | One job queue for image, video and voice, used by the Studios and by agents | 1 | Two jobs started together run one after the other; the Studio shows "waiting for an agent job" |
| 2 (done 2026-11-06) | `/mcp` endpoint with the v1 tools, off by default, the "turn it on" answer, `recommend_models` | 1 | `tools/list` and every tool work from an MCP client; off answers with the Settings path |
| 3 (done 2026-11-06) | `chaosengine-cli mcp` stdio relay; Settings > Integrations "Agent access" card | 0.5 | One copied line adds the server to Claude Code; the relay says "start ChaosEngineAI" when the app is closed |
| 4 (done 2026-11-06) | `SKILL.md`, `docs/features/agent-bridge.md`, locale strings | 0.25 | A new user can set it up from the docs alone |
| 5 (done 2026-11-06; Claude Code run passed; Windows still open) | Tests: unit, an E2E phase check, a live run | 0.5 to 1 | Claude Code remakes a poster, a 3 s clip and a voice line from one prompt on this Mac; Windows checked for the relay and endpoint only |

The user's flow once shipped: install the app and download a model or two, turn on Agent access in
Settings, paste the copied line into the agent, then ask the agent for an image, clip or voiceover. The
agent checks `system_status`, calls `recommend_models` if nothing suitable is installed (and tells the user
what to download), starts jobs, waits for them one at a time, and gets file paths back to edit or share.

## Effort

About 3 days: queue and tools 1 day, MCP wiring 0.5, stdio relay 0.5, Settings card and docs 0.5,
tests and a live agent run 0.5. The FU-126 estimate (1 to 2 days) did not include the queue or the relay.

## Tests

- Unit: every tool's input schema rejects bad input; the queue runs jobs one at a time and reports positions;
  results carry paths that exist.
- E2E (new feature gate): a phase check that calls `tools/list`, then `generate_image` with the smallest
  installed image model and `wait_for_job` until done.
- Live: add the server to Claude Code on this Mac and have it make one poster, one 3 s clip and one voice line
  from a single prompt. Windows and Linux: only the stdio relay and the HTTP endpoint, no CUDA generation.

## Risks

- **MCP client timeouts.** Client tool-call timeouts vary and are not all documented; the job id plus
  `wait_for_job` pattern keeps every call short whatever the client does.
- **The demo crash cause is unknown.** The queue removes concurrency, which was the common factor, but the
  same crash on a single job would need its own investigation.
- **Exposure.** Any local process with the token can spend minutes of GPU and fill the disk with outputs. The
  switch defaults to off until the user turns it on.

## Open questions for the owner

1. ~~Should UI-started jobs and agent jobs share one queue?~~ **Decided 2026-11-06: yes**, so an agent job never
   starts while the Studio is rendering. Cost: a queued UI job waits too, so the UI shows "waiting for an
   agent job".
2. ~~Agent access on or off by default?~~ **Decided 2026-11-06: off**, one switch in Settings; tools explain
   how to turn it on.
3. ~~Include model downloads in v1?~~ **Decided 2026-11-06: no**; `recommend_models` instead.
4. Go ahead and build v1? See decision D-21.
