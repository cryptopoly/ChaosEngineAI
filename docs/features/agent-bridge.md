# Agent bridge: let an AI agent make images, video and voice

The agent bridge lets an AI agent that speaks MCP (the Model Context Protocol), such as Claude Code,
make images, video clips, speech, music, transcripts and short films with ChaosEngineAI. The agent calls typed tools and
gets file paths back. It does not need to read our source or docs, and it cannot start two heavy jobs
at once.

## Set it up (about a minute)

1. Install ChaosEngineAI and download at least one model you want the agent to use (Image, Video or
   Voice, from the **Models** page). The agent cannot download models.
2. Open **Settings > Integrations > Agent access** and switch on **Allow agents to use ChaosEngineAI**.
3. Copy the line for your agent and run it once. For Claude Code:

   ```bash
   claude mcp add --transport http chaosengine http://127.0.0.1:8876/mcp --header "Authorization: Bearer <token>"
   ```

   The card shows the line with your token filled in. It also shows a JSON block for other MCP clients
   and, for clients that can only start a local program, a stdio command.
4. Ask the agent for something: "make a poster of a lighthouse at dusk, then a 3 second clip of waves
   and a one-line voiceover".

Optional: copy [`SKILL.md`](agent-bridge/SKILL.md) into `~/.claude/skills/chaosengine/` so Claude knows
when to use the tools and the one rule below.

## The one rule

Jobs run **one at a time**. An agent starts a job, then calls `wait_for_job` until it is done. This is
the same queue the Studios use, so an agent job never starts while a Studio is rendering, and a Studio
job waits for a running agent job (the progress panel says "Waiting for an agent job to finish").

## Tools

| Tool | What it does |
|---|---|
| `system_status` | Free memory, platform, the running job, the queue. |
| `recommend_models` | For `image`, `video`, `voice`, `stt` or `music`, the best models **not installed** that fit this machine, with size and gated / non-commercial flags. The agent tells you what to download. |
| `list_models` | Installed models with id, size, fit, size limits and frame rules. |
| `list_voices` | Voices that are ready, with language. |
| `generate_image` | Prompt, model, size, steps, seed, optional `reference_images` and `image_strength`. Returns a job id. Sizes are snapped to what the model supports; corrections appear in the job notes. |
| `generate_video` | Prompt, model, size, `seconds` (or `frames`), fps, seed, optional `start_image`, `end_image`, `image_strength`. Returns a job id. Seconds are converted to the model's frame rule. |
| `upload_image` | A small image as base64. Returns a handle (`upload:...`) for the image inputs above; for an agent on another machine. |
| `speak` | Text, voice, speed. Returns a job id. |
| `transcribe` | Path to an audio or video file the app can read. Returns a job id. |
| `make_music` | Style `caption`, `lyrics` (or `instrumental`), `seconds` 10 to 600, `bpm`, `key`, `seed`, `variations` 1, 2 or 4, `model`. Returns a job id; the result has the FLAC path of each take in the Music Library and a `download`. `cancel_job` stops it through the music worker. See [Music](music.md). |
| `job_status`, `wait_for_job`, `cancel_job` | Poll, wait up to 60 s, cancel. |
| `storyboard_project` | Film projects: `create`, `list`, `get` (shots, takes, timeline, revision), `edit` (name, defaults, export), `undo`, `redo`, `delete`, `export` (a `.chaosfilm` bundle with a `download`), `import`. |
| `storyboard_shot` | `add`, `edit`, `remove`, `reorder`, `lock`, `unlock`, `pick_take`, `redo` (marks a shot for the next generate), `import` (a clip or still file). |
| `storyboard_audio` | `add`, `edit`, `remove` audio clips; anchor one to a shot so it follows the shot. |
| `storyboard_generate` | New takes for the shots that need work (or `shot_ids`; `what` is `still`, `clip` or `both`): all stills, then all clips. Locked shots are skipped; naming one needs `force`. Returns a job id; `cancel_job` stops the whole batch. |
| `storyboard_render` | Render the film to MP4 (`preview` is small and fast). Returns a job id; the finished output has the path and `download`. |

A finished job returns the **absolute path** of the PNG, MP4, WAV or FLAC, plus width, height, duration and
`hasAudio`. `generate_image` can also return the PNG as base64 (`include_preview`). Every PNG and MP4
carries its prompt, model, seed, steps, guidance, size and app version inside the file.

### Input images (for consistency)

Give a file path, not the image data. `list_models` marks the models that take one (`acceptsImageInput`;
images also report `maxReferenceImages`). A model that cannot take an image answers with a clear error;
it never ignores the image.

- **Video:** `start_image` (and `end_image`) on LTX-2.3 / 2.5. The clip begins on the start image and ends
  on the end image. `image_strength` 1.0 follows the image closely; lower lets it drift.
- **Image:** `reference_images` on FLUX.1, FLUX.2 Klein and Qwen Image 2.1 (one start image; higher
  `image_strength` stays closer to it) and on FLUX 3 Klein (up to four images, edited or re-drawn from the
  prompt: the way to keep a subject or a style across a set).
- With no width and height, the output takes the shape of the first image.
- Voice cloning and sound input are not supported: the speech engine has no reference-audio input.

Every storyboard edit carries the project's `revision`; an old one is refused with the current number, so
the agent and the app can edit the same project. See [Storyboard](storyboard.md).

Not in the first version: model downloads, chat (the OpenAI-compatible API covers it) and media editing tools.

## Using ChaosEngineAI on another machine

An agent can drive a ChaosEngineAI running on a different computer, for example a PC with an NVIDIA GPU on
your home network. The paths in a job result belong to that PC, so files move over HTTP:

- **Send an image in.** `upload_image` takes a small image as base64; larger files: `POST` the raw bytes to
  `/api/agent/uploads` with the Agent access token. Either returns a handle like `upload:ab12cd34ef56`. Use the
  handle wherever a tool takes an image path (`start_image`, `end_image`, `reference_images`). The type is read
  from the bytes, the size cap is 30 MB, and handles are removed after 24 hours.
- **Get an output back.** A finished job's `output.download` has `url` (relative to the app's address), `bytes`
  and `sha256`. `GET` that URL with the token. Only files inside the app's own image, video and voice folders can
  be fetched, plus each storyboard project's `exports/` folder (films and bundles); any other path answers 404.
- **Reach the app.** The backend must listen on the network (`--host`, for example `0.0.0.0`) and the firewall
  must allow its port on private networks. The traffic is plain HTTP with a bearer token, which is fine on a home
  network nobody else can join; use a VPN or SSH tunnel anywhere else.
- Audio for `transcribe` is not uploaded yet.

## Safety

- **Off until you turn it on.** While off, the endpoint still answers and every tool call returns
  "Agent access is off. Turn it on in ChaosEngineAI: Settings > Integrations > Agent access."
- **Token.** The agent token is created once and kept between restarts, so the copied line keeps
  working. **Make a new token** in the card stops the old one at once. The token is stored in
  `agent_access.json` next to `settings.json`.
- **This computer only.** The endpoint is on the local address like the rest of the API.
- **Memory checks.** The same checks as the Studios run before a job starts. A refusal comes back as a
  readable error with the numbers.
- **Fixed tool set.** An agent cannot run code inside the app or load plugins through the bridge. Anything
  with the token can spend GPU time and fill the output folders, so keep it private.

## Clients that only start local programs

The stdio relay is `chaosengine-cli mcp`. It forwards each message to the running app. If the app is
closed it answers "ChaosEngineAI is not running. Start ChaosEngineAI, then try again." The card shows
the exact command for your install. It reads the token from `CHAOSENGINE_API_TOKEN` if set, otherwise
from the running app.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Agent access is off" | Switch it on in Settings > Integrations > Agent access. |
| HTTP 401 from `/mcp` | The token changed. Copy the line again from the card. |
| "No image model is installed" | Download one from Models. `recommend_models` suggests which. |
| A job stays queued | Another job (yours or the Studio's) is running. Check `system_status`. |
| "ChaosEngineAI is not running" | Start the app. |
