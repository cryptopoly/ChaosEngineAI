# Image Generation MVP

## Summary

Add a first-class local image-generation workflow to ChaosEngineAI that feels like the rest of the app:

- discover a small curated set of image models
- download them with one click
- generate images from a prompt without touching Python environments or external workflow tools
- keep outputs, prompts, seeds, and settings inside the app

The key product principle is simplicity over maximal flexibility. This should not feel like ComfyUI-in-a-window. It should feel like "Discover, Download, Generate".

## Why This Fits ChaosEngineAI

ChaosEngineAI already has several pieces we can reuse:

- Hugging Face repo discovery and one-click downloads in the backend
- Hub file inspection and lightweight repo metadata
- local library scanning and per-model actions
- a Tauri + React shell that already handles long-running jobs, logs, and media previews
- a Python backend with clear request/response boundaries

What is missing today is an image-generation runtime and image-specific data model. Current image support is input-only for multimodal chat models, not local image synthesis.

## Product Goals

### MVP goals

- Let users browse a curated catalog of image models
- Make model download/setup dead simple
- Generate images from text with a small, opinionated set of controls
- Save outputs and metadata locally
- Make results easy to iterate on: re-run, tweak seed, tweak prompt, change preset

### Non-goals for MVP

- Node-graph workflow editing
- Full ComfyUI compatibility
- LoRA training
- Video generation
- Inpainting, ControlNet, and multi-stage upscaler pipelines
- Full model-family autodetection across the whole Hub

## Recommended UX

### New top-level tab

Add a new tab:

- `Image Studio` or `Images`

This should be a dedicated mode, not hidden inside Chat.

### Core screens

#### 1. Discover Images

Parallel to the current Discover screen:

- curated image model families
- tags like `photoreal`, `illustration`, `fast`, `high-quality`, `anime`, `general`
- one-click download
- visible disk footprint and approximate generation speed

#### 2. My Image Models

Parallel to My Models:

- installed image models
- format/runtime
- size on disk
- compatible tasks: `txt2img`, later `img2img`, `inpaint`
- quick action: `Generate`

#### 3. Image Studio

The actual generation workspace:

- model picker
- prompt
- negative prompt
- aspect ratio preset
- image count
- steps
- guidance / CFG
- seed and randomize toggle
- quality preset: `Fast`, `Balanced`, `High Quality`
- generate button
- output gallery with prompt metadata

### UX constraints

- no raw scheduler dropdown in MVP
- no VAE dropdown in MVP
- no separate checkpoint / text encoder / VAE assembly in MVP
- hide most runtime complexity behind curated presets

## Technical Recommendation

### Runtime strategy

Use a separate image runtime path instead of trying to force image generation into the current LLM runtime controller.

Recommended structure:

- keep current LLM runtime untouched for chat/server/benchmark flows
- add a dedicated image-generation service layer in `backend_service`
- add a dedicated image model catalog and image job API

This keeps the mental model clean and avoids polluting LLM-specific types with diffusion-specific concerns.

### Recommended MVP backend choice

Use Python-based diffusion execution first, behind a small internal adapter layer.

Why:

- it fits the existing Python backend
- it keeps download, local state, and job orchestration in one place
- it is easier to make approachable than shelling out to an external app
- we can evolve the underlying engine later without rewriting the UI contract

The app should define its own internal engine abstraction, for example:

- `ImageGenerationEngine`
- `generate(prompt, options) -> ImageGenerationResult`
- `probe_capabilities()`
- `list_local_models()`

The implementation can start with one engine and add more later.

### Runtime options

### Option A: Diffusers-style backend

Pros:

- straightforward Python integration
- broad model compatibility
- easiest to reason about for an MVP
- good fit for cross-platform long term

Cons:

- Apple Silicon performance may be less compelling than a specialized path
- dependency footprint is larger

### Option B: Apple-Silicon-optimized image runtime

Pros:

- more aligned with ChaosEngineAI's Apple Silicon story
- potentially better local ergonomics for Mac users

Cons:

- likely narrower model compatibility
- higher implementation risk unless we lock to a small model set

### Recommendation

Abstract the engine from day one, but ship the MVP with a single supported path and a curated model list. The product value is the UX, not exposing every backend immediately.

## Data Model Additions

Add image-specific types instead of overloading the current LLM ones.

### New catalog types

- `ImageModelFamily`
- `ImageModelVariant`
- `ImageModelCapability`

Suggested fields:

- `id`
- `name`
- `provider`
- `repo`
- `link`
- `runtime`
- `taskSupport`
- `sizeGb`
- `recommendedResolution`
- `styleTags`
- `availableLocally`
- `downloadedFiles`

### New local library types

- `LocalImageModel`

Suggested fields:

- `name`
- `path`
- `runtime`
- `sourceKind`
- `sizeGb`
- `lastModified`
- `tasks`
- `broken`
- `brokenReason`

### New generation types

- `ImageGenerationRequest`
- `ImageGenerationResult`
- `SavedImageArtifact`

Suggested request fields:

- `modelId`
- `prompt`
- `negativePrompt`
- `width`
- `height`
- `steps`
- `guidance`
- `seed`
- `batchSize`
- `qualityPreset`

Suggested result fields:

- `artifactId`
- `imagePath`
- `thumbnailPath`
- `seed`
- `durationSeconds`
- `modelId`
- `prompt`
- `negativePrompt`
- `width`
- `height`
- `steps`
- `guidance`
- `createdAt`

## Storage Plan

Store generated images under the app data directory, not inside arbitrary model folders.

Suggested layout:

```text
<data-dir>/
  images/
    outputs/
      2026-04-09/
        <artifact-id>.png
        <artifact-id>.json
    thumbnails/
      <artifact-id>.jpg
```

Each generated image should have a sidecar metadata file so the gallery can be rebuilt from disk.

## Backend API Plan

Add image-specific endpoints instead of mixing them into `/api/chat`.

### Catalog / library

- `GET /api/images/catalog`
- `GET /api/images/library`
- `POST /api/images/download`
- `GET /api/images/download/status`

### Generation

- `POST /api/images/generate`
- `GET /api/images/jobs/{job_id}`
- `POST /api/images/jobs/{job_id}/cancel`
- `GET /api/images/outputs`
- `GET /api/images/outputs/{artifact_id}`
- `DELETE /api/images/outputs/{artifact_id}`

### Optional streaming

If we want a polished feel, expose progress for cold starts and generation:

- model loading
- pipeline warmup
- step progress
- image encode/write

This can mirror the app's existing long-running action patterns.

## Frontend Plan

### Tabs

Add:

- `image-discover`
- `image-models`
- `image-studio`

If we want to keep nav tight, `Image Studio` can combine the latter two in MVP.

### UI components

Likely reusable patterns:

- `Panel`
- status badges
- filter chips
- download progress badges
- existing long-running task modals / progress surfaces

New components:

- `ImagePromptForm`
- `ImagePresetPicker`
- `ImageModelCard`
- `ImageOutputGallery`
- `ImageOutputDetailModal`

## Recommended MVP flow

1. User opens `Image Studio`
2. User chooses a curated model or is prompted to download one
3. User enters prompt
4. User chooses `Fast`, `Balanced`, or `High Quality`
5. User clicks `Generate`
6. App shows progress
7. Result appears in a gallery

Quick actions on each result:

- `Open`
- `Reveal`
- `Use same settings`
- `Vary seed`
- `Delete`

## Curated model strategy

Do not start with open-ended Hub search for all image models.

Start with a hand-picked catalog of maybe 4-6 variants:

- one fast general model
- one high-quality general model
- one illustration/anime model
- one photoreal model

Selection criteria:

- stable repo structure
- clear licensing
- reasonable local hardware expectations
- predictable inference setup

The app can still expose a later "advanced import from Hugging Face" path, but MVP should prioritize known-good experiences.

## Hardware and platform considerations

The app should be explicit about support tiers.

Suggested MVP support:

- Apple Silicon first-class
- other platforms best-effort or phase 2

Even if the underlying runtime is cross-platform, the UX should set expectations:

- approximate RAM requirement
- first-run warmup time
- recommended resolution
- expected speed tier

## Error handling

The current app already benefits from human-readable failure messages. Keep that standard here.

Important cases:

- missing backend dependencies
- unsupported model repo shape
- insufficient memory
- gated/private Hugging Face repo
- cancelled generation
- corrupt local model files

Errors should always tell the user what to do next.

## Implementation Phases

### Phase 0: design and plumbing

- add image-generation types
- add image tab scaffolding
- add backend config and storage paths
- define internal engine abstraction
- add stub API routes with mock data

### Phase 1: curated download + local generation MVP

- curated image catalog
- one-click download
- installed image model list
- `Image Studio` page
- prompt -> generate -> save output
- output gallery

### Phase 2: quality-of-life

- seed history
- rerun with same settings
- multiple images per job
- better metadata inspection
- image delete / reveal / open actions

### Phase 3: advanced controls

- `img2img`
- inpainting
- LoRA support
- advanced resolution and sampler controls

## Codebase Touchpoints

Expected files/modules to touch:

- `src/App.tsx`
- `src/types.ts`
- `src/api.ts`
- `src/styles.css`
- `backend_service/app.py`
- `backend_service/inference.py`

Possible later additions:

- `backend_service/image_runtime.py`
- `backend_service/image_catalog.py`
- `src/components/Image*.tsx`

## Risks

### Biggest technical risk

The runtime choice. Image generation libraries and model layouts are less uniform than current LLM flows.

### Biggest product risk

Trying to support too much too early and ending up with a confusing, semi-working "advanced" UI.

### Mitigation

- curated model list first
- one engine first
- strict MVP surface
- architecture abstraction before engine sprawl

## Recommendation

Build this.

But build it as a focused, curated local image studio, not as a general-purpose workflow editor.

If we keep the first version opinionated, this could become one of the strongest parts of ChaosEngineAI:

- easier than ComfyUI
- more local/private than hosted apps
- more polished than most self-hosted image frontends
- naturally aligned with the app's existing discover/download/run model

## Suggested next implementation step

Start with Phase 0 on this branch:

- add the new tab IDs and image types
- add mock image catalog data
- scaffold `Image Studio` with a fake generate flow
- add backend stub endpoints that return deterministic placeholder results

That gives us a visible vertical slice before we commit to a specific runtime.
