---
name: chaosengine
description: Make images, video clips, speech, music and transcripts with the local ChaosEngineAI app through its MCP tools (generate_image, generate_video, speak, make_music, transcribe). Use when the user asks for a picture, poster, clip, voiceover, song, soundtrack or transcript made locally, or mentions ChaosEngineAI.
---

# ChaosEngineAI tools

The `chaosengine` MCP server makes media on this computer. If its tools are missing, tell the user to
open ChaosEngineAI > Settings > Integrations > Agent access, switch it on, and run the copied line.

## Rules

1. Call `system_status` first. Note free memory and whether a job is running.
2. Run **one job at a time** (each token may hold 20 unfinished jobs). Start a job, call `wait_for_job` with its `jobId`, repeat until `state` is
   `done`, `failed` or `cancelled`. Do not start the next job before that.
3. Call `list_models` for the kind you need. If it is empty, call `recommend_models`, tell the user which
   model to download from ChaosEngineAI > Models and why, and stop. Never try to download anything.
4. Use `seconds` for video, not frames. Read the job `notes`: they list every size or frame correction.
5. A finished job gives a file path. Use that path for editing, sharing or placing in a project. Tell the
   user the path.
6. Video takes minutes. Image takes seconds to a minute. Speech is quick. Music takes about 1 s per 10 s of audio once loaded (about 25 s more the first time).
7. If a job fails with a memory message, report the numbers to the user. Do not retry in a loop.

## Typical flow

1. `list_models` kind `image`, `generate_image` with the prompt, `wait_for_job`.
2. `list_models` kind `video`, `generate_video` with `seconds` 3 or 4, `wait_for_job`.
3. `list_voices`, `speak` with the text, `wait_for_job`.

Prompts: be concrete about subject, style, light and framing. Keep video prompts to one clear motion.

## Keeping a subject or style consistent

Pass the path of an earlier output as an input image. Only models with `acceptsImageInput` in `list_models` take one; the others return an error.

- Poster to clip: `generate_video` with `start_image` set to the poster path (LTX-2 models). Add `end_image` to land on a second image.
- Same subject in new scenes: `generate_image` with `reference_images` set to the earlier image (FLUX 3 Klein takes up to four; FLUX.1, FLUX.2 Klein and Qwen take one, with `image_strength` where higher stays closer).
- Leave width and height out and the output takes the image's shape.
- Cloned voices: `list_cloned_voices` then `speak_cloned` (needs the `clone` scope and the person's "Allow agents to use cloned voices" switch). You can only use a voice the person already made in the app with their spoken consent; you cannot create one. The output is tagged and watermarked as AI-made, and a film clip made from it is marked automatically. If refused, tell the user what the error says.
- On another machine, paths in a job result are not readable from here: send inputs with `upload_file` (use the `upload:` handle as the path) and fetch a result from `output.download.url` with the same token.

## Making a short film (storyboard)

`storyboard_project` create, then `storyboard_shot` add one shot per beat (`stillPrompt`, `motionPrompt`, `seconds`; `startFrom: previous-shot` continues from the last frame). Every edit needs the `revision` from the last answer; if it is refused, call `storyboard_project` get and retry. `storyboard_generate` makes new takes (stills first, then clips); `wait_for_job`, then `get` to check statuses. To redo shots, `storyboard_shot` redo them and generate again: other shots are untouched, and shots that continue from a redone one turn `stale`. Add sound with `storyboard_audio` (anchor voice to its shot), then `storyboard_render` with `preview` true; render the final with `preview` false and give the user the path.

## Captions, stems and editor export

`storyboard_project` edit with `fields.export` `{captionsSrt, captionsBurn, stems}` (booleans) sets what a render writes next to the film; the render output lists them in `extras` (each with a `download`) and any `warnings`. `storyboard_captions`: `generate` (a job: transcribes the voice clips), `set` / `add` / `edit` / `remove` to fix cues by hand (times in seconds, `revision` needed), `export_srt` for a download. `storyboard_export_otio` gives an OpenTimelineIO file for Resolve or Premiere. Pass `notify: true` to `storyboard_render` for a system banner when a long render ends. A failed or interrupted render says how many shots are done: call `storyboard_render` again to resume.

## Music

`list_models` kind `music`, then `make_music` with a style `caption` (genre, instruments, mood, voice) and `lyrics` with `[verse]` / `[chorus]` tags, or `instrumental: true`; `seconds` 10 to 600; the same `seed` gives the same track. `variations` 2 or 4 gives takes with seeds `seed`, `seed+1`, ... in `output.tracks`. The result is a FLAC in the Music Library; `music_library` lists, gets (with a download), renames or tags, and deletes tracks (delete is permanent: ask the user first). Write your own lyrics; do not name an artist in the caption, describe the sound. When a storyboard has a music lane, ask for clips with "no music" in the prompt so the clip's own sound does not fight the soundtrack.

## Film with music

1. `make_music` (instrumental or with lyrics), `wait_for_job`; note `output.tracks[0].path`.
2. Build the film with `storyboard_*`, asking clips for "no music" in the prompt.
3. `storyboard_audio` add with `kind: "music"` and that path (or an `upload:` handle), then `storyboard_render`.

## Resources, prompts and progress

`resources/list` shows the system status, the music library, recent outputs and each film (`chaosengine://storyboard/{id}`); `resources/read` reads one. `prompts/list` offers `short_film_from_brief`, `poster_set` and `soundtrack_for_project`. If your client sends a `progressToken` with `wait_for_job`, it gets `notifications/progress` while waiting; without one the answer is plain JSON.

## Errors and recovery

- **Revision refused** ("The project is at revision N now"): call `storyboard_project` get, redo the edit with the new `revision`. Do not loop blind.
- **"does not have the 'X' scope"**: the token is limited (for example read-only). Tell the user to change it in Settings > Integrations > Agent access; do not try another route.
- **"unfinished jobs (the limit)"**: wait for or `cancel_job` some jobs first.
- **Lost job id, or the app restarted**: `list_jobs`. A job that was running shows `interrupted`; start it again.
- **A job fails on memory**: report the numbers to the user; no retry loop.

## Using an app on another machine

Paths in results belong to that machine. Send inputs with `upload_file`, fetch outputs from `output.download.url` with the same token, and check `output.download.sha256`. Transcribing needs a path on that machine. Models cannot be downloaded by you: use `recommend_models` and tell the user. A request-and-approve download flow is planned; until it exists, never try to install anything.

## Estimating the wait

`system_status` lists the whole queue with positions and estimated times (labelled estimated; a first run of a model loads weights and is slower). Tell the user the wait before starting a long job when others are queued.
