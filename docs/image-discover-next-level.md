# Image Discover Next-Level Plan

Last updated: April 9, 2026

## Goal

Take `Image Discover` from a small static starter catalog into a living, trustworthy model browser that:

- surfaces newer image models quickly
- shows accurate live metadata instead of stale hardcoded values
- explains what will actually work on this machine
- keeps the simple ChaosEngineAI UX instead of turning into raw Hugging Face search

## Current Gap

Today the image catalog is still a handcrafted Phase 0 list in [backend_service/app.py](../backend_service/app.py):

- `FLUX.1 Schnell`
- `Stable Diffusion XL Base 1.0`
- `SDXL Turbo`
- `Stable Diffusion v1.5`

That got the vertical slice shipped, but it now has clear limits:

- sizes are hardcoded
- recency is hardcoded
- latest official image releases are missing
- `txt2img` and editing models are mixed together conceptually
- users cannot sort by newest, smallest, fastest, open-license, or "works on my machine"
- one `sizeGb` number is too simplistic for modern repos

## Market Reality

The official diffusers and Hugging Face ecosystem is now broader than our current catalog shape.

### Strong current candidates

These are good fits for a curated ChaosEngineAI image catalog refresh:

| Family | Primary task | Approx repo/core size | Notes |
| --- | --- | --- | --- |
| `black-forest-labs/FLUX.1-schnell` | txt2img | `23.8 GB` main weight | 12B model, fast 1-4 step generation |
| `black-forest-labs/FLUX.1-dev` | txt2img | `23.8 GB` main weight | better quality, gated, non-commercial |
| `stabilityai/stable-diffusion-3.5-medium` | txt2img | `5.11 GB` main weight | much more modern than SDXL, gated |
| `stabilityai/stable-diffusion-3.5-large-turbo` | txt2img | `16.5 GB` main weight | newer turbo option, gated |
| `stabilityai/stable-diffusion-xl-base-1.0` | txt2img | `6.94 GB` fp16 core weight, `76.9 GB` repo | still useful baseline, repo size is misleading if shown as one number |
| `stabilityai/sdxl-turbo` | txt2img | `6.94 GB` fp16 core weight, `55.5 GB` repo | still useful for fast drafts |
| `Qwen/Qwen-Image` | txt2img | `57.7 GB` repo | new diffusers-native text-to-image family |
| `Qwen/Qwen-Image-Edit` | img2img/edit | `57.7 GB` repo | strong signal that editing should become a first-class catalog task |
| `HiDream-ai/HiDream-I1-Full` | txt2img | `47.2 GB` repo | 17B open model, quality-oriented |
| `HiDream-ai/HiDream-I1-Dev` | txt2img | `47.2 GB` repo | same family, dev variant |
| `HiDream-ai/HiDream-I1-Fast` | txt2img | `47.2 GB` repo | same family, speed-oriented |
| `zai-org/GLM-Image` | txt2img + img2img | `35.8 GB` repo | unified generation and editing path |
| `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | txt2img | `7.7 GB` repo | compelling fast/small lane |
| `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | txt2img | `9.74 GB` repo | larger fast model |

### Important ecosystem shift

The diffusers pipeline list now includes newer families such as:

- `QwenImage`
- `HiDream-I1`
- `HunyuanImage2.1`
- `GLM-Image`
- `Sana`
- `Sana Sprint`
- `Flux` / `Flux Kontext`
- `Kandinsky 5.0 Image`
- `Lumina 2.0`
- `Ovis-Image`
- `Z-Image`

Inference: our current static four-model catalog is now too narrow for the pace of the upstream image ecosystem.

## Product Direction

### Keep two layers, not one

The best UX is not "show every repo on Hugging Face".

Instead, split Discover into:

1. `Curated`
   - a compact, verified set we know works well in ChaosEngineAI
   - opinionated recommendations like `Best Quality`, `Fastest`, `Smallest`, `Editing`

2. `Latest`
   - a live feed of newer diffusers-compatible image repos
   - ranked and filtered so it still feels curated, not chaotic

This preserves simplicity while letting the app stay current.

## Metadata Plan

### Replace static catalog-only metadata with live enrichment

Build an image-discover metadata service that combines:

- our curated catalog entries
- Hugging Face model metadata
- Hugging Face file-tree summaries
- local runtime compatibility heuristics

### New metadata fields

Extend image model records with:

- `license`
- `gated`
- `pipelineTag`
- `pipelineClass`
- `tasks`
- `downloadsLastMonth`
- `likes`
- `lastModified`
- `repoSizeBytes`
- `coreWeightsBytes`
- `componentSizes`
- `recommendedDtype`
- `recommendedSteps`
- `minDiffusersVersion`
- `supportsMps`
- `supportsCpuFallback`
- `estimatedRamGb`
- `estimatedVramGb`
- `verifiedByChaosEngine`
- `verificationNotes`
- `source: curated | latest | experimental`

### Key size fix

Do not show a single size number anymore.

Show both:

- `Core weights`
- `Full repo download`

Reason:

- SDXL Base shows `6.94 GB` for the fp16 checkpoint but `76.9 GB` for the full repo
- SDXL Turbo shows `6.94 GB` for the fp16 checkpoint but `55.5 GB` for the full repo

That distinction will make the UI feel much more honest.

## UX Plan

### Discover layout upgrade

Rework `Image Discover` into sections:

1. `Recommended For This Mac`
   - ranked by likely success on current hardware
   - badge examples: `MPS verified`, `Small download`, `Fast first image`

2. `Curated Picks`
   - `Fast`
   - `Balanced`
   - `Best Quality`
   - `Smallest`
   - `Editing`

3. `Latest Releases`
   - live list from Hugging Face model metadata
   - sorted by `Newest`, `Most downloaded`, or `Most liked`

4. `Explore More`
   - broader search results with stronger warnings and compatibility labels

### Filters

Add filter chips / selects for:

- task: `txt2img`, `img2img`, `edit`, `inpaint`, `variation`
- license: `commercial`, `non-commercial`, `apache`, `community`
- access: `open`, `gated`
- size: `<10 GB`, `10-25 GB`, `25-50 GB`, `50+ GB`
- speed: `fast`, `balanced`, `quality`
- provider
- runtime support: `MPS-ready`, `CPU-only fallback`, `experimental`
- source: `curated`, `latest`

### Card improvements

Every card should show:

- model name and provider
- task badges
- exact access state: `Open` or `Needs Hugging Face approval`
- `Core weights` and `Full repo` size
- last updated date
- downloads / likes
- recommended resolution
- likely first-run load time class
- "works on this Mac" badge when verified

### Details drawer

Open a richer side panel or modal with:

- model summary
- pipeline class
- tasks supported
- required files/components
- exact file tree summary
- license and gating
- quality vs speed notes
- known ChaosEngineAI caveats
- direct actions: `Download`, `Open Model Card`, `Copy Repo Id`

## Runtime Compatibility Plan

### Add an image model verifier

Before a model is marked as `Recommended` or `Verified`, run a lightweight verification pass:

- confirm `model_index.json` exists
- confirm all sharded weights referenced by `*.index.json` exist
- detect pipeline class
- detect whether our installed diffusers version supports it
- classify runtime support on this machine
- optionally perform a tiny preload smoke test

### Compatibility labels

Use product wording like:

- `Verified on this Mac`
- `Large model, slower first load`
- `Needs newer diffusers runtime`
- `Gated model`
- `Experimental in ChaosEngineAI`

## Catalog Scope Plan

### Phase 1 curated refresh

Refresh the curated list to something like:

- `FLUX.1 Schnell`
- `FLUX.1 Dev`
- `Stable Diffusion 3.5 Medium`
- `Stable Diffusion 3.5 Large Turbo`
- `SDXL Base 1.0`
- `Sana Sprint 0.6B`
- `Sana Sprint 1.6B`
- `Qwen-Image`
- `HiDream-I1 Fast`
- `GLM-Image`

### Phase 2 task-based expansion

Split by task family:

- `Text to Image`
- `Editing and Variations`
- `Control / Structure`

Candidates:

- `Qwen-Image-Edit`
- `FLUX.1 Kontext-dev`
- `FLUX.1 Redux-dev`

### Phase 3 advanced / experimental lane

Only after the metadata and compatibility system exists:

- `HiDream-I1 Full`
- `HunyuanImage-2.1`
- newer diffusers pipelines that exceed normal laptop-friendly footprints

## Backend Plan

### New service layer

Add an image-discover service in `backend_service` responsible for:

- querying Hugging Face metadata
- querying file tree summaries
- caching responses
- normalizing licenses, tasks, and pipeline names
- calculating size summaries
- calculating compatibility scores
- merging live metadata with curated overrides

### Caching strategy

Use a persistent cache with:

- short TTL for search results, e.g. `6h`
- longer TTL for file-tree summaries, e.g. `24h`
- manual `Refresh Metadata` action in UI

This keeps Discover feeling fresh without making the app depend on constant live calls.

### Ranking heuristics

Score models using:

- verified compatibility on current platform
- recency
- popularity
- size
- gating friction
- curated priority

Example:

- a `7.7 GB` Sana Sprint model may outrank a `57.7 GB` Qwen model on an Apple laptop
- a verified open model may outrank a gated model unless the user explicitly filters for quality

## Frontend Delivery Plan

### Milestone A: Live metadata

- add backend image-discover metadata service
- keep current UI mostly intact
- replace hardcoded `updatedLabel` and size text with live metadata

### Milestone B: New Discover information architecture

- add `Curated` vs `Latest`
- add filter bar
- add richer model cards
- add details drawer

### Milestone C: Compatibility and trust

- add `Verified on this Mac`
- add exact gating and license messaging
- add core-size vs full-repo-size display

### Milestone D: Task-aware catalog

- add `txt2img` vs `edit` vs `variation`
- surface `Qwen-Image-Edit`, `FLUX Kontext`, `FLUX Redux`

### Milestone E: Personalization

- rank models against local hardware and free disk
- recommend small or verified models first

## Success Criteria

We should consider this successful when:

- users can find newer official image models without leaving ChaosEngineAI
- size and access expectations are accurate before download starts
- Discover clearly separates safe recommendations from experimental models
- users can filter to "models that will probably work here"
- the catalog can keep up with new diffusers-compatible releases without hand-editing every card

## Recommended Build Order

1. Live metadata enrichment for the existing curated catalog
2. Core-size vs full-repo-size display
3. Latest model feed
4. Compatibility labels
5. Task filters and editing-model lane
6. Verification-based recommendations

## Reference Sources

- Hugging Face Diffusers pipelines overview: https://huggingface.co/docs/diffusers/api/pipelines/overview
- Hugging Face Flux docs: https://huggingface.co/docs/diffusers/api/pipelines/flux
- Hugging Face QwenImage docs: https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- Hugging Face HiDream docs: https://huggingface.co/docs/diffusers/api/pipelines/hidream
- Hugging Face HunyuanImage2.1 docs: https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuanimage21
- Hugging Face SANA-Sprint docs: https://huggingface.co/docs/diffusers/api/pipelines/sana_sprint
- FLUX.1 Schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- FLUX.1 Dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
- FLUX.1 Kontext-dev: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- Stable Diffusion 3.5 Medium: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
- Stable Diffusion 3.5 Large Turbo: https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
- SDXL Base 1.0: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- SDXL Turbo: https://huggingface.co/stabilityai/sdxl-turbo
- Qwen-Image: https://huggingface.co/Qwen/Qwen-Image
- Qwen-Image-Edit: https://huggingface.co/Qwen/Qwen-Image-Edit
- HiDream-I1 Full: https://huggingface.co/HiDream-ai/HiDream-I1-Full
- GLM-Image: https://huggingface.co/zai-org/GLM-Image
- Sana Sprint 0.6B: https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
- Sana Sprint 1.6B: https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers
