# Reference pictures for stills on NVIDIA (B1309)

Status: wired and unit-tested with stubs. **Not run on a GPU yet.**

## What was wired

**Model:** FLUX 3 Image Klein · sd.cpp Q8_0 (`leejet/flux-3-image-klein-GGUF-q8`, file
`flux3-image-klein-Q8_0.gguf`). It is already the Storyboard's default still model on the 4090, so no new
catalogue row, download or licence is needed (THIRD_PARTY_NOTICES.md unchanged).

**Why this model and not Qwen-Image-Edit-2511:** the pinned stable-diffusion.cpp (master-1122, `3b12781`)
documents reference editing for FLUX.3 Klein directly: `docs/models/flux3.md` shows a `-r` edit example
and says to "repeat `-r` to use multiple reference images, up to 10". `docs/guides/edit.md` lists
"FLUX.2 / FLUX.3 Klein: VAE reference latents with increasing position indexing" as the model default. So
the existing row, its companion files (Qwen3-VL-4B text encoder, FLUX.2 VAE) and its flags work as they
are. Qwen-Image-Edit-2511 would need a new GGUF row, a 20B-class transformer, its own text encoder with a
vision tower, and new VRAM measurements. That is more work and slower on a 24 GB card.

**Flags (checked against `sd.exe --help` of the pinned build):**

- `-r, --ref-image`: "reference image for Flux Kontext or MiniMax-H3 Ref2VA (can be used multiple times)".
  The help text is older than the docs. The FLUX.3 doc and the edit guide both cover Klein.
- `--increase-ref-index` and `--auto-resize-ref-image` **no longer exist**. `--ref-image-args` was removed
  as well (edit guide). The generated DGL uses each model's default reference policy. For FLUX.3 Klein
  that is increasing position indexing, with each picture scaled to at most the output's area, capped at
  1 MP and kept on a 16 px grid (`src/api/generation_image.cpp`). Custom resizing now needs a DGL script.
  The app passes nothing extra.
- Unchanged: `--disable-image-metadata`, `--sampling-method euler`, `--fa`, `--cfg-scale 1`, `--vae`,
  `--llm`.

**Code:**

| File | Change |
| --- | --- |
| `backend_service/sdcpp_image_runtime.py` | `_REFERENCE_IMAGE_LIMITS` (Klein GGUF: 4) and `reference_image_limit(repo)`. `_reference_args` adds `-r PATH` per picture after the companion files and before the catalogue's extra arguments. Each path goes through `resolve_input_image` again (real file, PNG/JPEG/WebP, 30 MB cap). It refuses more than 4, or a model without reference support. Arguments are in list form, with no shell. |
| `backend_service/helpers/media_input.py` | `reference_image_limit(variant)`: mflux family table, sd.cpp table, or 0 on any other engine. `checked_reference_images` uses it. The refusal message now names the NVIDIA option. |
| `backend_service/helpers/images.py` | The Studio inputs (`maxReferenceImages`) for sd.cpp rows come from the same limit, so the Studio shows the reference picker for this row. |
| `backend_service/mcp_bridge/models.py` | The agent model list (`acceptsImageInput`, `maxReferenceImages`) uses the same limit. |
| `backend_service/image_runtime/__init__.py` | When references are given and sd.cpp is missing or fails, the request fails instead of falling back to diffusers, which would drop the pictures without telling anyone (the same rule mflux follows). |
| `backend_service/catalog/image_models.py` | The row's note says it takes up to 4 reference pictures. |

**Storyboard:** characters reach the still through `storyboard/generators.reference_paths`, then
`referenceImagePaths`, then `routes/images._checked_reference_images`, then `checked_reference_images`.
That check now passes for this row, so character pictures reach `sd` as `-r`.
`storyboard/style.py::picture_slots` still returns 0 for every engine except mflux, so a project's *style*
pictures still go into the prompt as words on this lane. Making them pictures takes a one-line change in
`storyboard/style.py` (`return reference_image_limit(variant)` from `helpers.media_input`). That change
was left to the Storyboard stream because `backend_service/storyboard/*` is outside this change.

## Real-GPU test to run later (RTX 4090)

Run it when the GPU is free. Resolve the three cached files first. They are already on disk if this row
has rendered before:

```powershell
$py = "E:\ChaosEngineAI\.venv\Scripts\python.exe"
$dit = & $py -c "from huggingface_hub import hf_hub_download as d; print(d('leejet/flux-3-image-klein-GGUF','flux3-image-klein-Q8_0.gguf'))"
$llm = & $py -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen3-VL-4B-Instruct-GGUF','Qwen3VL-4B-Instruct-Q4_K_M.gguf'))"
$vae = & $py -c "from huggingface_hub import hf_hub_download as d; print(d('black-forest-labs/flux-3-image-klein','ae.safetensors'))"
```

Then run the command line the app builds, with one character picture (`char.png`, a clear front-facing
portrait) at the Storyboard size:

```powershell
E:\tmp\sdcpp-new\bin\sd.exe --diffusion-model $dit -p "the same woman from the reference picture sitting in a rainy cafe at night, side light, film still" -W 1280 -H 720 --steps 8 --cfg-scale 1 --seed 42 -o E:\tmp\b1309\ref1.png --disable-image-metadata --vae $vae --llm $llm -r E:\tmp\b1309\char.png --sampling-method euler --fa -v
```

Repeat with 2 and 4 `-r` pictures (for example `char.png`, `char_side.png`, `outfit.png`, `style.png`).
Also run once with no `-r` and the same seed as a control. Then run the same request through the app, so
the route, the check and the runtime are covered end to end:
`POST /api/images/generate` with `modelId = leejet/flux-3-image-klein-GGUF-q8`, width 1280, height 720 and
`referenceImagePaths: ["E:\\tmp\\b1309\\char.png"]`. Finally render a three-shot Storyboard with one
character.

**Expected outputs**

- Exit code 0 and a 1280x720 PNG for each run. With `-v`, the log shows the reference being encoded
  (VAE encode of `reference_0` and so on) before sampling.
- Note wall time and peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -lms 200` in another
  window) for 0, 1, 2 and 4 references.

**How to judge identity**

- Put the reference picture and the shots side by side.
- Pass: the same face shape, hairline, hair colour and length, eye colour, any distinctive marks, and
  (when asked for) the same clothing, across all three Storyboard shots. The scene, pose and lighting
  follow the prompt.
- Fail (known for FLUX.3 on diffusers, FU-273): the output is a near copy of the reference, with the
  same framing and background and the prompt ignored. Fail too: a different person, which shows the
  reference was not used. Compare against the no-`-r` control at the same seed. If the face matches the
  control and not the reference, the pictures did not reach the model.
- If the output copies the reference, try a prompt that names a change ("in a different place, new pose")
  and 16 steps (the documented edit setting) before calling the model unusable.

## Risks

- **Not verified on a GPU.** Everything above is unit-tested with stubs only. The `-r` path for FLUX.3
  Klein is documented upstream, but it has not been run in this app.
- **Copying instead of keeping identity.** FU-273 found the diffusers FLUX.3 row returned the reference
  nearly unchanged without BFL's prompt upsampler. sd.cpp uses a different reference path (DGL default
  policy), but the same model behaviour could show up.
- **VRAM at 1280x720.** Without references the row measured 11.1 GB above an idle card at 1024 px. Each
  reference is scaled to at most the output area (about 0.92 MP at 1280x720), which adds about 3,600 image
  tokens per picture to the 3,600 of the output. With 4 references the transformer sees about 18,000 image
  tokens instead of 3,600. Flash attention (`--fa`) keeps attention memory roughly linear. Activations and
  the VAE encode of each picture still add memory: expect roughly 12 to 16 GB with 4 references. sd.cpp's
  auto-fit spills weights to RAM rather than failing, and gets slower when it does. The app's memory gate
  uses the row's single footprint (11.5 GB) and does not count references.
- **Speed.** The row takes 5.5 s at 1024 px for 8 steps without references. Joint attention cost grows
  with the square of the token count, so 1 reference could take about 2 to 3 times as long and 4 references
  about 5 to 10 times as long (rough figures, not measured). Catalogue time estimates (`estimateRef`) do not
  count references yet, so Storyboard time estimates will be low for shots with characters.
- **Style pictures.** As noted above, they stay as prompt words on this lane until `storyboard/style.py`
  uses the shared limit.
