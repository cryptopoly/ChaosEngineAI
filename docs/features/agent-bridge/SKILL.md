---
name: chaosengine
description: Make images, video clips, speech and transcripts with the local ChaosEngineAI app through its MCP tools (generate_image, generate_video, speak, transcribe). Use when the user asks for a picture, poster, clip, voiceover or transcript made locally, or mentions ChaosEngineAI.
---

# ChaosEngineAI tools

The `chaosengine` MCP server makes media on this computer. If its tools are missing, tell the user to
open ChaosEngineAI > Settings > Integrations > Agent access, switch it on, and run the copied line.

## Rules

1. Call `system_status` first. Note free memory and whether a job is running.
2. Run **one job at a time**. Start a job, call `wait_for_job` with its `jobId`, repeat until `state` is
   `done`, `failed` or `cancelled`. Do not start the next job before that.
3. Call `list_models` for the kind you need. If it is empty, call `recommend_models`, tell the user which
   model to download from ChaosEngineAI > Models and why, and stop. Never try to download anything.
4. Use `seconds` for video, not frames. Read the job `notes`: they list every size or frame correction.
5. A finished job gives a file path. Use that path for editing, sharing or placing in a project. Tell the
   user the path.
6. Video takes minutes. Image takes seconds to a minute. Speech is quick.
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
- Voice cloning is not available.
- On another machine, paths in a job result are not readable from here: send inputs with `upload_image` (use the `upload:` handle as the path) and fetch a result from `output.download.url` with the same token.

## Making a short film (storyboard)

`storyboard_project` create, then `storyboard_shot` add one shot per beat (`stillPrompt`, `motionPrompt`, `seconds`; `startFrom: previous-shot` continues from the last frame). Every edit needs the `revision` from the last answer; if it is refused, call `storyboard_project` get and retry. `storyboard_generate` makes new takes (stills first, then clips); `wait_for_job`, then `get` to check statuses. To redo shots, `storyboard_shot` redo them and generate again: other shots are untouched, and shots that continue from a redone one turn `stale`. Add sound with `storyboard_audio` (anchor voice to its shot), then `storyboard_render` with `preview` true; render the final with `preview` false and give the user the path.
