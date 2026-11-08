# Image and video input for the Studios and the agent bridge

Status: started 2026-11-06 on `feat/media-input` (cut from `feat/agent-bridge`). Decision: D-23 on `docs/hub/decisions.md`.

## Goal

Make an image or a clip **from** an image, so an agent (or a person) can keep a subject, a look or a frame
consistent across outputs: a poster, then a video that starts from the poster, then the same character in a
new scene.

## What the engines can do (checked 2026-11-06 against the installed tools)

| Engine | Input it takes | Flags |
|---|---|---|
| LTX-2.3 / 2.5 (`ltx-2-mlx`) | Start image, end image, or images anchored at chosen frames (image-to-video). Sound-to-video exists upstream. | `--image PATH [FRAME_IDX STRENGTH]`, repeatable |
| mflux, FLUX.1 (`mflux-generate`), FLUX.2 Klein (`-flux2`), Qwen-Image-2.1 (`-qwen-2.1`) | Init image with a strength (image-to-image) | `--image PATH [STRENGTH]` |
| mflux edit commands: `-flux2-edit`, `-flux3-edit` (FLUX 3 Klein), `-qwen-2.1-edit` | One or more reference images plus an edit prompt (keeps the subject) | `--image-paths P1 P2 ...` |
| mflux FLUX 3 Klein generate, Iris-3B | None on the generate command | use the `-flux3-edit` command for FLUX 3 |
| Diffusers video models (Wan, Hunyuan, LTX-Video) | Not wired here | out of scope |
| Voice | Kokoro has no cloning or reference audio | out of scope; `transcribe` already takes a file |

Today none of the request models has an image field, so the Studios cannot do this either. The app, not the
bridge, is the gap.

## Order

1. **Video start / end image** (LTX only; **done 2026-11-06**): request fields `startImagePath`, `endImagePath`, `imageStrength`; the
   engine adds `--image`; other video models answer a clear 400. MCP `generate_video` gets `start_image`,
   `end_image`. Done when a poster turns into a clip that starts on it (live run).
2. **Image reference** (mflux; **done 2026-11-06**): `referenceImagePaths` and `imageStrength` on the image request; FLUX.1, FLUX.2 Klein
   and Qwen use init-image; FLUX 3 Klein uses its edit command. MCP `generate_image` gets `reference_images`.
3. **Studio UI** (**done 2026-11-06**): an image picker in Image Studio and Video Studio (upload, or pick from the gallery), shown only
   for models that take it. Strings and locales.
4. Docs, SKILL.md and E2E checks for each step.

## Rules

- Inputs are local file paths for now (the backend user must be able to read them), checked the same way as
  `transcribe`: resolved, a regular file, an image extension, at most 30 MB.
- A model that cannot take an image refuses with a message naming models that can; it never ignores the image.
- The output file records which input image it used, so a run can be repeated.

## Not in scope

Voice cloning, sound-to-video, ControlNet / depth / fill, multi-keyframe timelines beyond start and end,
remote clients that cannot share a path (upload tool is a later step).
