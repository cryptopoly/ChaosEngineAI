# v0.10.0: stable-diffusion.cpp pin bump (task B step 0, follow-up)

Branch `v0100/bump-sdcpp`, worktree `/Users/dan/ChaosEngineAI-v0100-bump-sdcpp`, cut from `release/v0.10.0` at `bbdd72f2`.

## Goal

Move `SDCPP_REF` from `3f8527a` (master-929, 2026-09-27) to the newest `master` named by `scripts/check-upstreams.py`, and prove it breaks nothing with one real FLUX.1 schnell image and one Wan 2.1 T2V 1.3B clip on the new build.

## State

- [x] `check-upstreams.py` rerun 2026-11-14: newest `master` is `5dcfa79801213f23e27b3e83617350ebb8ea93aa` (2026-11-14, 175 commits ahead of `3f8527a`, 1104 commits on master).
- [x] Docs read at the new pin (`docs/models/`): `wan.md`, `flux.md`, `minimax_h3.md`, `minimax_h3_image.md` (exists).
- [ ] Downloads (owner-approved this session, see below).
- [ ] Baseline FLUX.1 schnell image on the old binary.
- [ ] Build new pin, FLUX.1 schnell image, Wan 2.1 1.3B clip.
- [ ] Tests.
- [ ] CLAUDE.md row, check-upstreams note.

## Change of plan (owner, 2026-11-14, mid-task)

FLUX.1 schnell and Wan 2.1 are too old to be the regression test. Primary image check is now the `leejet/Qwen-Image-2.1-GGUF-q4k` row through the app's own arguments, old binary and new binary, same seed, size and steps. FLUX.1 schnell only if its files were already complete (they were not: download stopped at 5.3 of 6.9 GB for the GGUF, `.incomplete` left in the HF cache, nothing else of FLUX fetched). Wan clip dropped (no app path, FU-176); no Wan file was fetched. The only sd.cpp video row in the app is MiniMax H3, which is never run, so there is no video run.

Qwen-Image 2.1 files (Hub list, 2026-11-14): `leejet/Qwen-Image-2.1-GGUF/qwen_image_2.1-Q4_K.gguf` (4.20 GB), `Comfy-Org/Qwen-Image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors` (0.68 GB, already cached), `Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf` (5.03 GB). Supported at both pins (`docs/qwen_image_2.1.md` at `3f8527a`, `docs/models/qwen_image_2.1.md` at `5dcfa79`).

## Original download plan (superseded)

FLUX.1 schnell sd.cpp row (`image_models.py`): `leejet/FLUX.1-schnell-gguf/flux1-schnell-q4_k.gguf` (6.88 GB), `second-state/FLUX.1-schnell-GGUF/ae.safetensors` (0.34 GB) and `t5xxl-Q8_0.gguf` (5.20 GB), `comfyanonymous/flux_text_encoders/clip_l.safetensors` (already cached).

Wan 2.1 T2V 1.3B, files exactly as `docs/models/wan.md` at `5dcfa79` names them. The doc lists no GGUF for the 1.3B, only the fp16 safetensors, so that is what was fetched (smaller than the approved 15 GB): `Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors` (2.84 GB), `.../split_files/vae/wan_2.1_vae.safetensors` (0.25 GB), `city96/umt5-xxl-encoder-gguf/umt5-xxl-encoder-Q8_0.gguf` (6.04 GB). Total for both models about 21.6 GB.

## Findings

- CLI flags: every flag the app passes still exists at `5dcfa79` (`--diffusion-model -p -W -H --steps --cfg-scale --guidance --seed -o --vae --clip_l --t5xxl --negative-prompt --sampling-method -M --audio-vae --llm --rng --fps --video-frames --offload-to-cpu`). `--fa` and `--diffusion-fa` are now deprecated aliases (they log a warning and set `--flash-attn on` / `--flash-attn off,diffusion=on`). Kept as they are: `--flash-attn` does not exist at `3f8527a`, so switching would break anyone still on the old build, and the pinned H3 doc itself still uses `--diffusion-fa`.
- New default: `--auto-fit on` (places weights on GPU, RAM or disk by available memory). Not passed by the app.
- H3 video flags in `docs/models/minimax_h3.md` at the new pin match `_build_h3_cli_args` (no rename), so the H3 command line and `tests/test_licence_gate.py` need no change.

## H3-Image (not built, record only)

`docs/models/minimax_h3_image.md` exists at `5dcfa79`. An image row would need:
- Files: `--diffusion-model` a single-file checkpoint with the six-step PDD heads (`minimax_h3_image_pruned_bf16.safetensors` or its INT8 ConvRot version; the doc links only the `Comfy-Org/MiniMax-H3-Image` safetensors, no GGUF), `--vae` the dedicated H3-Image VAE (`minimax_h3_image_vae_fp16.safetensors`; the H3 video VAE is incompatible), `--llm` the H3 Qwen3-VL-32B text encoder (shared with the video row, `leejet/MiniMax-H3-GGUF` Q4_K_M 18.2 GB) or the much smaller P-H3-Image text encoder (Qwen3-VL-4B with connector, about 5 GB INT8 ConvRot, `PrunaAI/P-H3-Image-Comfy`). No audio VAE.
- Args: `--cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image --rng cpu --diffusion-fa --vae-tiling`, plus `--offload-to-cpu`; start at 1024x1024; sides aligned up to 32. Reference editing adds `-r` (up to 16, `<Picture N>` in the prompt).
- App work: `engine: "sdcpp"` row with `ggufRepo`/`ggufFile`/`sdcppAux`/`sdcppArgs` like FLUX.1 schnell, but the image runtime has no `--llm` / `--sigmas` handling beyond `sdcppArgs` and `sdcppAux` would carry `--vae` and `--llm`; the licence gate and `H3_RELEASED` switch apply as for the video row; exact file names and sizes to be read from the Hub lists. Not added (not asked).
