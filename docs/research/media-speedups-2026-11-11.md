# Image, video and audio speed-ups beyond what we ship (2026-11-11)

Status: **desk notes from model knowledge, not measured and not re-verified online.** Check each repo and licence before building anything (FU-167).

Already in the repo (`cache_compression/`): fbcache, teacache, taylorseer, magcache, pab, fastercache. Nunchaku (SVDQuant) is already referenced.

## Ranked by expected win

1. **Step distillation.** Fewer steps beats cheaper steps. Hyper-SD, Lightning and Turbo LoRAs for image models; Wan self-forcing / CausVid style checkpoints and distilled LTX checkpoints for video. Typical 20-50 steps down to 4-8. Risk: quality and prompt-adherence drop, and each LoRA is per base model.
2. **Sparse attention for video.** Sparse VideoGen, Radial Attention, FastVideo (VSA). Helps long clips most. Mostly CUDA; check MLX/MPS support.
3. **Quantised attention kernels (CUDA).** SageAttention. Drop-in on supported GPUs; verify output drift.
4. **More caching.** DeepCache (UNet models only), ToCa / DuCa (token-level caching for DiT models). Overlaps with what we have; try only if a model is not covered by the existing six.
5. **Audio, speech-to-text.** distil-whisper with speculative decoding, faster-whisper, lightning-whisper-mlx on Mac.
6. **Audio, TTS and music.** Autoregressive TTS is LLM-shaped, so speculative decoding and the MLX / llama.cpp paths apply. ACE-Step turbo variants are already step-distilled.

## Next step

Pick one image model and one video model, measure baseline seconds per output on the 4090 and the M4 Max (`scripts/perf-baseline.py`), then trial 1 and 3 first. Log the numbers here.
