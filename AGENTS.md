# ChaosEngineAI agent guide

You are an AI agent and the person you work for has ChaosEngineAI on their computer. Read this once, then
use the tools. It is also served as the MCP resource `chaosengine://guide`, by the MCP prompt `start_here`,
and at `GET /api/agent/guide`.

## What the app is

ChaosEngineAI is a desktop app that runs AI models on the person's own hardware: chat, images, video clips
(some with sound), speech, transcripts, music, sound effects and short films (the Storyboard). Everything
runs locally. You drive the media tools through MCP; you do not run models yourself and you cannot run code
inside the app.

You can: make media, edit film projects, read what is installed and what the queue is doing.
You cannot: download models (tell the person which to get), create a cloned voice, or delete a film project
or track unless the person asks.

## Connect

1. The person opens **Settings > Integrations > Agent access** and switches on **Allow agents to use
   ChaosEngineAI**. It is off by default. While off, every tool answers "Agent access is off" and says where
   to turn it on: pass that on to the person, do not retry.
2. Endpoint: `POST http://127.0.0.1:8876/mcp` (streamable HTTP, JSON-RPC 2.0; the port is the app's port, shown
   on the card). Send the Agent access token as `Authorization: Bearer <token>`. The card shows a ready line, for
   Claude Code:
   `claude mcp add --transport http chaosengine http://127.0.0.1:8876/mcp --header "Authorization: Bearer <token>"`
3. A client that can only start local programs uses the stdio relay `chaosengine-cli mcp` (the card shows the
   exact command for the install). It forwards each message to the running app and reads the token from
   `CHAOSENGINE_API_TOKEN` or from the app. If the app is closed it answers "ChaosEngineAI is not running."
4. A 401 means the token changed ("Make a new token" stops the old one at once). Ask the person to copy the line again.
5. An agent on another computer: the app must listen on the network (`--host`), traffic is plain HTTP with a
   bearer token, so use a private network or a tunnel. Paths in job results belong to the app's machine; see Files below.

### Scopes

A token carries scopes and `tools/list` only shows what the token may use.

| Scope | Gives |
|---|---|
| `read` | status, model and voice lists, job status, `storyboard_project` list and get, `music_library` list and get |
| `generate` | every make tool, film edits, `cancel_job` |
| `files` | `upload_file`, `/api/agent/uploads`, `/api/agent/files`, film export and SRT and OTIO downloads |
| `clone` | `list_cloned_voices`, `speak_cloned` (never in a token unless the person ticks it) |

A second, read-only token exists for dashboards. A refused call names the missing scope: tell the person to change the
token's permissions in the same Settings card.

## First calls, every session

1. `system_status`: free memory, platform and the whole job queue with estimated times.
2. `list_models` for each kind you need (`image`, `video`, `voice`, `stt`, `music`). Each row has id, size, `fit`
   (skip a `wont-fit` model), size limits, frame rules and `acceptsImageInput`.
3. If a kind is empty, call `recommend_models`, tell the person which model to download from the Models page and why,
   and stop. You cannot download. Gated models need the person to accept a licence first.

## The one rule: jobs run one at a time

Every make tool returns a `jobId`. Heavy jobs share one queue with the Studios, so a job waits while the person
renders in the app (and the other way round). The loop is always: start a job, `wait_for_job` until `state` is
`done`, `failed` or `cancelled`, use the output, then start the next. Do not start a second heavy job while one is
running, and do not poll other tools in a tight loop while the queue is busy: `wait_for_job` waits up to 60 s on the
server, call it again if the job is not finished.

- Each token may hold **20 unfinished jobs**; the 21st is refused until some finish.
- Jobs are kept 7 days and survive an app restart as `interrupted`. `list_jobs` finds a job id you lost.
- `cancel_job` stops a queued or running job (the whole batch for `storyboard_generate`).
- A finished job's `output` has an absolute `path` (width, height, duration, `hasAudio`) and, for another machine,
  `output.download` (`url`, `bytes`, `sha256`). Read `notes`: every size or frame correction is listed there.
- Memory checks run before every job. A refusal comes back with the numbers; report them, do not retry in a loop.

## Tools

One line per tool: what it does and the arguments that matter. `*` marks a required argument.

### Look

- `system_status`: free memory, platform, the job queue with positions and `estimatedSeconds`. Call it first.
- `list_models`: installed models of a `kind`* with fit and limits.
- `recommend_models`: models not installed that fit this machine (`kind`*, `limit`). You cannot download them.
- `list_voices`: ready text-to-speech voices with language.
- `list_jobs`: recent jobs, newest first (`state`, `limit`).
- `job_status`: state, progress, queue position, notes and output of one job (`job_id`*).
- `wait_for_job`: like `job_status`, waits up to `timeout_seconds` (max 60).
- `cancel_job`: cancel a queued or running job (`job_id`*).

### Make

- `generate_image`: `prompt`*, `model`, `width`, `height`, `steps`, `seed`, `reference_images`, `image_strength`. Sizes snap to
  what the model supports. Reference images only work on models with `acceptsImageInput`; others refuse, they never ignore.
- `generate_video`: `prompt`*, `model`, `seconds` (not frames), `width`, `height`, `fps`, `seed`, `start_image`, `end_image`,
  `image_strength`. Takes minutes. `start_image` and `end_image` are for LTX-2 models.
- `speak`: text to speech (`text`*, `voice`, `speed`, `style`, `pitch`). Cancellable.
- `speak_cloned`: speech in a cloned voice (`voice_id`*, `text`*, `permission_confirmed`* must be `true`). See Cloned voices.
- `list_cloned_voices`: cloned voices the person made in the app, with ids. Needs the `clone` scope.
- `make_music`: a song or `instrumental`; `caption`* (style), `lyrics`, `seconds` 10 to 600, `bpm`, `key`, `seed`, `variations` 1, 2 or 4.
  FLAC into the Music Library.
- `make_sound`: a sound effect or loop, 1 to 11 s (up to 47 s with the 1.0 model `modelId`); `caption`* in English, `loop`. Not songs or speech.
  Needs a gated Stable Audio Open model.
- `transcribe`: speech to text for a file on the app's machine (`path`*, `language`). Local paths only, no `upload:` handles.

### Films (the Storyboard)

Every film edit carries the project `revision` from the last answer. A stale one is refused with the current number:
call `storyboard_project` get and retry.

- `storyboard_project`: `action`* is `create` (`name`, `aspect` 16:9, 9:16 or 1:1, `resolution` 480p to 2160p default 720p, `fps` 24 to 60),
  `list`, `get`, `edit`, `undo`, `redo`, `delete`, `export`, `import`, and the cast actions `add_ref`, `edit_ref`, `remove_ref`.
  `edit` with `fields.export` sets what a render writes next to the film: `captionsBurn`, `captionsSrt`, `stems`.
- `storyboard_shot`: `action`* is `add`, `edit`, `remove`, `reorder`, `lock`, `unlock`, `pick_take`, `redo`, `import`.
  Fields: `title`, `stillPrompt`, `motionPrompt`, `seconds`, `seed`, `startFrom` (`generate`, `upload` or `previous-shot`),
  `referenceImages` (cast ids), `trimIn`, `trimOut`, `speed`, `transition`, `nativeSoundDb`.
- `storyboard_audio`: `action`* is `add`, `edit`, `remove`; `kind` is `music`, `sfx`, `voice` or `import`; `path` is an `upload:` handle or a file
  in the app's output folders. `fields`: `anchor {shotId, offset}` (sound follows the shot), `start`, `gainDb`, `fadeIn`, `fadeOut`,
  `duck`, `loop`, `trimIn`, `trimOut`, `label`.
- `storyboard_generate`: new takes, all stills then all clips, for shots that need work (`shot_ids`, `what` is `still`, `clip` or `both`,
  `force` to include locked shots). Returns a job id.
- `storyboard_render`: render to MP4 (`preview` is small and fast, `notify` shows a banner). A failed render names how many shots are
  done; call again to resume.
- `storyboard_captions`: `generate` (a job; transcribes the voice clips), `export_srt`, `set`, `add`, `edit`, `remove`.
- `storyboard_export_otio`: the cut as an OpenTimelineIO file for Resolve or Premiere.

### Library and files

- `music_library`: `list` (`query`, `tag`, paging), `get` (adds a download), `rename`, `delete` (permanent: ask the person first).
- `upload_file`: a small image, sound or video as base64 (about 8 MB); returns an `upload:` handle valid 24 hours. Larger files: `POST` the
  raw bytes to `/api/agent/uploads` with the same token.

## Make a short film: the order of calls

1. `system_status`, then `list_models` for `image` and `video` (and `voice`, `music` if you want sound).
2. `storyboard_project` create (`name`, `aspect`, `resolution`). Keep the returned `projectId` and `revision`.
3. Cast (optional, for the same face or place across shots): `storyboard_project` add_ref with `kind` `character` or `place`, a
   `description` and up to 4 `images`. Shots use the ref id in `fields.referenceImages`. A film can have no cast.
4. Shots: `storyboard_shot` add once per beat with `title`, `stillPrompt`, `motionPrompt` (one clear motion), `seconds`. Use
   `startFrom: "previous-shot"` to continue from the last frame.
5. Voice: `list_voices`, then `speak` for each line, `wait_for_job`, then `storyboard_audio` add with `kind: "voice"`, the output `path`, and
   `fields.anchor` set to its shot so it moves with the shot. Optionally `storyboard_captions` generate.
6. Music: `make_music` (`instrumental: true`, `seconds` close to the film length), `wait_for_job`, then `storyboard_audio` add with `kind: "music"`,
   the track path from `output.tracks[0].path`, `fields.duck: true`, `fields.fadeOut`. Ask clips for "no music" in the motion prompt so
   they do not fight the soundtrack. Sound effects: `make_sound`, then `storyboard_audio` add `kind: "sfx"`.
7. Generate: `storyboard_generate`, `wait_for_job`, `storyboard_project` get to check each shot's take. For a weak shot: `storyboard_shot` redo
   then `storyboard_generate` again. Other shots are untouched; shots that continue from a redone one go stale. `lock` the good ones.
8. Mix: `storyboard_audio` edit (`gainDb`, `fadeIn`, `fadeOut`, `duck`) and `storyboard_shot` edit (`nativeSoundDb`, `trimIn`, `trimOut`, `transition`).
9. Preview then final: `storyboard_render` with `preview: true`, `wait_for_job`, look at it, then `preview: false`. Give the person the file name
   and `download`.
10. Export: `storyboard_project` export (a `.chaosfilm` bundle), `storyboard_export_otio`, or captions with `export_srt`.

`storyboard_project` undo and redo step the project history.

## Chat's studio tools: the plan-approval gate

The Chat tab has a Studio switch. When it is on, the chat model gets a subset of these tools, in process, with scopes `read` and `generate` only
(no files, downloads or cloning; export, import and delete actions are removed). The first time the model calls a tool that renders
(`generate_image`, `generate_video`, `speak`, `make_music`, `make_sound`, `storyboard_generate`, `storyboard_render`, `storyboard_captions`)
the call does not run. The app shows a plan (what, model, size, estimated time and memory) and waits for the person to click approve.
Only that click counts, never chat text. Rules for a model in Chat:

- **Batch every render for one job into ONE plan.** Put all the render calls in the same reply so the person approves once.
- If a call is declined, nothing was made. Do not call it again; ask what to change. If the person chose to edit the plan, stop calling tools
  and ask.
- The plan waits up to 30 minutes. "Always allow renders in this chat" is the person's choice, off by default.
- Read and edit tools (status, lists, shot edits) never wait for approval.

## Memory and time limits

Numbers below were measured on an Apple M4 Max with 64 GB unless said otherwise. A figure marked unmeasured has not been timed on any machine
in this repo; do not quote it as fact. `list_models` `fit` and `system_status` free memory are the truth for the machine you are on.

| Kind | Model | Time | Memory |
|---|---|---|---|
| Speech | Kokoro 82M (MLX) | 0.17 s warm for 6 s of speech | model file about 330 MB; peak unmeasured |
| Speech | Kokoro (portable lane) | 0.47 s for the same line | unmeasured |
| Speech | Qwen3-TTS 0.6B CustomVoice | 2.65 s per 10 s of speech, 2.4 s cold start | peak 2.9 GB |
| Speech | Qwen3-TTS 1.7B CustomVoice | 2.45 s per 10 s, 2.4 s cold | peak 3.5 GB |
| Speech | Qwen3-TTS 1.7B VoiceDesign | 2.46 s per 10 s, 2.3 s cold | peak 3.5 GB |
| Speech | Qwen3-TTS 0.6B Base (clone) | 3.2 s per 10 s, 3.0 s cold | peak 4.3 GB |
| Speech | Qwen3-TTS 1.7B Base (clone) | 3.6 s per 10 s, 3.6 s cold | peak 6.5 GB |
| Transcript | Parakeet (MLX) | 0.09 s for a 6 s clip, first call 1.9 s | unmeasured |
| Transcript | faster-whisper base (portable) | 0.6 s for the same clip | unmeasured |
| Music | ACE-Step 1.5 turbo | about 1 s per 10 s of audio (60 s in 5 to 6 s, 600 s in 82 s), first track +25 s | about 22 GB generating, 15 GB idle |
| Music | ACE-Step 1.5 XL turbo | about 2 s per 10 s | about 47 GB; 64 GB Macs only, unloads other models first |
| Video | LTX-2.3 q4 with sound | 4 s clip at 768 by 512 in about 155 s | about 17 GB; download 28 GB |
| Video | LTX-2.5 with sound | same clip in 128 s | gated model; memory unmeasured separately |
| Video | other models | unmeasured; `system_status` shows an estimate labelled as estimated | read `list_models` |
| Image | all models | unmeasured here; first run is slower than the estimate | read `list_models` |

Speech notes: a short line costs about 2 s even though it is short, because the model must start; batch several lines of the
same voice rather than many tiny jobs. The 0.6B CustomVoice model ignores `style`. Cloning needs the Base models.

Size rules: sizes snap to what the model supports and the correction is in the job `notes`. Image models default to a 4096 px side cap (several are
unverified); FLUX.2 allows 4 megapixels in multiples of 16. LTX video sizes are multiples of 64, Wan sizes multiples of 16. `generate_video` takes
`seconds`; frames follow the model's rule. Ask for 3 to 4 seconds per clip.

Film and upload limits: 500 shots, 500 audio clips, 2000 captions, 50 takes per shot, 50 characters and 50 places, 4 pictures per cast entry, prompts
up to 4000 characters, titles and lines up to 200. Uploads: images 30 MB, sound 200 MB, video 1 GB; `upload_file` base64 about 8 MB. Handles last 24 hours.

## Files and another machine

- Local agent: use the `path` in a job output directly.
- Remote agent: send inputs with `upload_file` or `POST /api/agent/uploads` (type read from the bytes) and use the `upload:` handle wherever a tool wants
  a path (image handles for `start_image`, `end_image`, `reference_images`). Fetch results from `output.download.url` with the same token. Only files in the
  app's own image, video and voice folders and each film's `exports/` folder can be fetched.
- Film imports (`storyboard_shot` import, `storyboard_audio` add, `storyboard_project` import) accept only an `upload:` handle or a file inside the
  app's output folders. Any other path is refused.

## Cloned voices

You may only use a voice the person already made in the app. Three locks must all be open: the token has the `clone` scope, the Voice setting
"Allow agents to use cloned voices" is on (only the app window can change it), and the voice has a stored permission record. A caller on another computer also
needs "Also allow cloning from other computers". `speak_cloned` requires `permission_confirmed: true` on every call; set it only when the voice is the
person's own or its recorded permission covers this use. The call is written to the audit log and the output is tagged and watermarked as AI-made. If
refused, tell the person what the error says. Never try to create a voice.

## Prompts that work

- Images: subject, style, light, framing in plain words. Keep one seed to keep a set matching; change only composition.
- Video: one clear motion per clip; describe sound too for LTX models. Poster to clip: `start_image` set to the poster path.
- Same subject in new scenes: `generate_image` with `reference_images` (FLUX 3 Klein takes up to four; FLUX.1, FLUX.2 Klein and Qwen Image take one).
- Music: `caption` describes the sound (genre, instruments, mood), not an artist. Write your own lyrics with `[verse]` and `[chorus]` tags.
- Sound effects: English, concrete ("wooden door closing, small room").

## Common mistakes

- **Starting a second job before the first finished.** Always `wait_for_job` first. Check `system_status` when a job sits queued: the person may be rendering.
- **Many small renders instead of one batch.** In Chat, put every render of one task in a single reply so one plan covers it. For films, one
  `storyboard_generate` makes all stills and clips; do not call `generate_image` per shot.
- **A stale `revision`.** Use the revision from your last answer; on refusal call `storyboard_project` get.
- **Cloning without all three locks, the permission tick, or `permission_confirmed: true`.** See Cloned voices.
- **Passing frames to video.** Pass `seconds`.
- **Ignoring `notes`.** Size and frame corrections are listed there; the file may not be the size you asked for.
- **Reference images on the wrong model.** Check `acceptsImageInput` in `list_models` first.
- **Trying to download a model, or looping on a memory refusal.** Report the numbers and what to download; stop.
- **Local paths from another machine.** Upload first, then use the handle.
- **Deleting.** `music_library` delete and `storyboard_project` delete are permanent. Ask the person first.
- **Calling tools your token does not list.** A scope error names the scope; ask the person to change it.
- **Waiting without bound.** Video takes minutes; tell the person what you are waiting for and the estimate from `system_status`.

## Resources and prompts

Resources (every token): `chaosengine://guide` (this guide), `chaosengine://system/status`, `chaosengine://music/library`, `chaosengine://music/{id}`,
`chaosengine://gallery/recent`, `chaosengine://storyboard/{id}`. Prompts: `start_here` (read this guide first), `short_film_from_brief`, `poster_set`,
`soundtrack_for_project`. When a `tools/call` carries `_meta.progressToken`, `wait_for_job` streams progress events.
