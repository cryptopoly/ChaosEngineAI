# Newest models for a 24 GB RTX 4090 against the catalogue (2026-11-19)

Two read-only research agents (web search, Hugging Face API and model cards) compared the newest open models with the catalogue; I read the summaries and checked the points marked "checked". Everything else is the agents' reading and is a lead to test, not a fact. This adds to `cuda-model-landscape-2026-11.md` and `minimax-h3-and-new-video-image-2026-11-10.md`; it does not repeat them.

## Corrections to earlier notes

- **Qwen-Image 2.1 is non-commercial** (its LICENSE, section 2a: commercial use needs a separate licence from Qwen). The catalogue row note already says "Qwen Research licence: non-commercial use only" (checked); the 11-09 note called it unclear.
- **Stable Audio 3** (released 2026-05-17: Medium 2B, Small-SFX 0.6B, Small-Music) is in no earlier note and not in the catalogue (the catalogue has Stable Audio Open and Small).
- **Nunchaku FLUX.1 on Windows**: the older "does not load" result was a wrong wheel pick (checked, see the 2026-11-19 Accelerators section of `rtx4090-progress.md`).

## Image, video, music

| Area | Model | Why it matters | Catalogue today | Tested here |
|---|---|---|---|---|
| Image | FLUX.3 Image Klein 5.6B (Oct 2026; edit, 10 refs; gated, free under US$5M revenue) | Third on the open-weights arena; successor to FLUX.2 Klein | Mac (mflux) row only, no CUDA row | no |
| Image | Qwen-Image 2.1 (and Turbo) | Second on the arena; 21 s at 1024 px, 40 steps through sd.cpp CUDA (measured) | Q4_K sd.cpp row; Turbo on Mac only | PASS |
| Image | Krea 2 Turbo (12B, 8 steps, `Krea2Pipeline` in diffusers; gated) | Most popular new non-Chinese open image model | missing | no |
| Image | MOSS-Image SFT (5B, Apache 2.0, 50 steps; needs remote code) | Only fully open 2026 image model | missing | no |
| Image | FLUX.2 Klein 4B | Default; about 6.9 s warm (measured) | row | PASS |
| Video | LTX-2.5 (22B, audio; 4-bit on load) | Flagship; 63 s for 4 s at 768x512 (measured) | 2 CUDA rows | PASS |
| Video | LTX-2.3 nf4 row | Rests on one community upload (12 downloads) | row | PASS |
| Video | Kandinsky 6.0 Lite (3B, MIT, 44 kHz audio, 10 steps) | MIT video with audio | missing | no |
| Video | Wan 2.2 TI2V 5B | Works; start frame only | row | PASS |
| Video | MiniMax H3 family | Top of the video arena, territory-locked | hidden on purpose | no |
| Music | ACE-Step 1.5 (and XL turbo), MIT | Fits easily; the CUDA lane is not built (FU-139) | row (Mac) | SKIP |
| Music | Stable Audio 3 (Small-SFX 0.6B, Medium 2B) | Replaces a 2024 model on the existing `stable-audio-tools` worker | missing | no |
| Music | HeartMuLa-oss-3B, MOSS-SoundEffect v2.0, MiniMax-Music3 | Alternatives; licences differ | missing | no |

Not downloadable: Wan 2.5 to 3.0 (sources disagree whether open weights exist). Skip: Ideogram 4 (non-commercial), HiDream-O1 (no diffusers path), Apple FastFrames (research only), Sulphur-2 (no licence stated).

## Chat and voice

| Area | Model | Note | Catalogue today | Tested here |
|---|---|---|---|---|
| Chat | Qwen3.8-27B (Apache 2.0) | Default; own runs 49.6 tok/s plain, 132 to 206 with TensorFold drafting, llama.cpp DFlash2 1.8 to 2.6x | rows on every lane | PASS |
| Chat | Gemma 4 26B-A4B and 12B QAT GGUF | CUDA gaps (the 26B has MLX rows only, the 12B only BF16) | partly | no |
| Chat | Qwen3.6-35B-A3B, Nemotron 3.5 Lightning 30B-A3B | Speed picks for 24 GB; no Windows tok/s yet | rows | no (Bolt E4B PASS 172 tok/s) |
| Chat | Qwen3.8-Flash-Next 125B-A6B | Does not fit 24 GB; do not offer as a 4090 default | row | no |
| TTS | Kokoro, Chatterbox (clone), VoxCPM2, Orpheus, Qwen3-TTS | All PASS here; Qwen3-TTS is slow (0.58x real time) | rows | PASS |
| TTS | Ming-3.0-tts-0.5B (MIT; clone and design), ZONOS3-Mini (Apache; clone, 43 languages) | New, small; Windows support untested | missing | no |
| STT | Whisper large-v3-turbo (faster-whisper) | The only STT offered on Windows and Linux (`voice_runtime._lane_backends`) | row | PASS |
| STT | Qwen3-ASR 1.7B (Apache 2.0, 52 languages) | Top pick to add | missing | no |
| STT | Parakeet TDT 0.6B v3, Nemotron 3.5 ASR streaming 0.6B | Rows exist but run on the MLX lane only | Mac only | no |
| STT | Granite Speech 5.0 2B | Second option, 6 languages | missing | no |

## What to do, ranked for a Windows 4090 user

1. STT on CUDA: Qwen3-ASR-1.7B first, then Parakeet v3 and Nemotron 3.5 ASR 0.6B (rows exist, the CUDA lane does not run them).
2. ACE-Step 1.5 on CUDA (music is skipped on Windows today).
3. FLUX.3 Image Klein CUDA row.
4. Gemma 4 26B-A4B and 12B QAT GGUF rows.
5. Stable Audio 3 rows.
6. Measure Qwen3.6-35B-A3B and Nemotron 3.5 Lightning here (no Windows speeds).
7. Test Kandinsky 6.0 Lite, Krea 2 Turbo, MOSS-Image SFT, Ming-3.0-tts, ZONOS3-Mini.
8. Back up the LTX-2.3 nf4 row; speed up or flag Qwen3-TTS.

## Could not verify

4090 speeds for FLUX.3 Klein, Krea 2, MOSS-Image, Kandinsky 6, Stable Audio 3, HeartMuLa, ACE-Step, Gemma 4, Qwen3.6-35B-A3B and Lightning; the Krea 2, FLUX.3 and NVIDIA "other" licence texts; whether Ming-3.0-tts and Triton-based models run on Windows; whether the Gemma 4 `-assistant` repos are drafters; the Wan 2.5 to 3.0 question; vendor-reported benchmark numbers. `mobiuslabsgmbh/faster-whisper-large-v3-turbo` now redirects to `dropbox-dash/faster-whisper-large-v3-turbo`: check the pinned name still downloads.
