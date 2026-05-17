# Catalog vs library

ChaosEngineAI distinguishes between two model registries:

- **Catalog** — curated lists of recommended models the app ships
  metadata for. Lives in `backend_service/catalog/`.
- **Library** — actual model weights on disk. Discovered by scanning
  configured model directories.

The launch modal pulls candidates from both: a model in your library
plus matching catalog metadata gives you a one-click launch with
pre-populated defaults; a model in the catalog you haven't downloaded
yet shows up with a Download button.

## Catalog

Three submodules:

| File | Purpose |
|---|---|
| `backend_service/catalog/text_models.py` | LLM families + variants. |
| `backend_service/catalog/image_models.py` | Diffusion image models. |
| `backend_service/catalog/video_models.py` | Diffusion video models. |

Each entry is a typed dict carrying:

- **`repo`** — canonical Hugging Face repo id.
- **`family`** — group key for the launch modal (`qwen3-coder-next`,
  `flux`, `wan-2-1`, etc.).
- **`name`**, **`description`**, **`tags`**.
- **`params`** — parameter count (`"14B"`, `"35B-A3B"`).
- **`format`** — `"safetensors"`, `"gguf"`, `"mlx"`.
- **`capabilities`** — typed flags (`chat`, `code`, `vision`,
  `reasoning`, `tools`, `multilingual`, `thinking`).
- **`defaultSteps`** + **`cfgOverride`** for diffusion models.
- **`ggufRepo` / `ggufFile`** for GGUF variants pinned to a specific
  quant.
- **`distillRepo` / `distillTransformer*`** for distill LoRA / fp8
  transformer combos.
- **`runtime`** — `"mlx-video"`, `"sdcpp"`, `"diffusers"`.

The catalog is **append-only** for stable model refs; once a row ships
in a release, removing it would break existing user sessions that
reference it. New variants land as additional rows.

## Library

The library is whatever's on disk under the configured model
directories. Settings → Storage exposes the directory list; the
defaults are:

- `~/AI_Models/` — the app's primary model root.
- Hugging Face cache (`~/.cache/huggingface/hub/`) — symlinked /
  resolved through `huggingface_hub`.
- Any custom directories you add (Ollama, LM Studio, network shares).

The library scanner walks each directory, identifies safetensors / GGUF
/ MLX checkpoints, and surfaces a `LibraryEntry` carrying:

- Source ref (`<org>/<repo>` reconstructed from the path).
- Format detection (safetensors / GGUF / MLX-quantised).
- On-disk size and last-modified time.
- Resolved `runtimeTarget` path used by the loader.

## Matching catalog ↔ library

The frontend joins the two views by repo id. The fuzzy matcher in
`src/utils/library.ts::libraryVariantMatchScore` handles cases where
the library entry's name doesn't exactly match a catalog repo — useful
for community quants whose paths don't carry the canonical org prefix.

**Watch-out:** when no exact catalog match exists for a community
quant, the matcher picks the closest variant by family + format +
bits, which can silently mis-assign a model to the wrong catalog row.
The FU-041 fix added an explicit
`lmstudio-community/Qwen3-Coder-Next-MLX-4bit` catalog variant for
exactly this reason (it was being mis-canonicalised as the unrelated
dense Qwen3.6-27B).

When you add a new community quant entry, add an explicit catalog row
rather than relying on fuzzy matching. The DFlash / MTPLX registries
do the same — community aliases are first-class in `_ALIASES` and
`_MTP_ALIASES`.

## Resolving for inference

`backend_service/model_resolution.py` handles the canonical-repo
lookup chain:

1. Caller supplies `model_ref` (might be an MLX community quant, a
   GGUF path, or a canonical repo id).
2. `resolve_dflash_target_ref(model_ref)` follows `_ALIASES` to find
   the canonical target for the DFlash registry lookup.
3. The same canonical repo is fed into `has_mtp_heads(repo)` for the
   MTPLX gate.
4. The on-disk path is resolved against the library and the HF cache.

This layered resolution means a user can load
`mlx-community/Qwen3.6-27B-4bit` and the backend correctly:

- Probes for MTP heads via `Qwen/Qwen3.6-27B` (the canonical).
- Probes for a DFlash drafter via `Qwen/Qwen3.6-35B-A3B` if applicable
  (depends on the alias map).
- Loads the actual MLX 4-bit weights from disk.

## Adding a model

Concrete steps for landing a new model family:

1. **Catalog entry.** Add a row to `text_models.py` / `image_models.py` /
   `video_models.py`. Pin every quant variant the family ships.
2. **DFlash / MTPLX registry.** Add the canonical repo to
   `DRAFT_MODEL_MAP` (with drafter ref) or `MTP_MODEL_MAP` (with
   draft-n) if applicable. Add community aliases.
3. **Tests.** Update `tests/test_dflash.py` / `tests/test_mtp.py` with
   pinned-mapping assertions. Add a happy-path test under the relevant
   E2E phase.
4. **Docs.** Cross-reference the new entry from
   [features/mtplx.md](../features/mtplx.md) / [features/dflash.md](../features/dflash.md)
   / [usage/image-studio.md](../usage/image-studio.md) /
   [usage/video-studio.md](../usage/video-studio.md).

## See also

- [Runtime paths](runtime-paths.md).
- [Inference engines](inference-engines.md).
- [Contributing → Adding a feature](../contributing/adding-a-feature.md).
