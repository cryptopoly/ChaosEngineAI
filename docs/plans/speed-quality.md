# Speed and quality plan: image, video, music, voice

Status: draft 1 (2026-11-07), branch `research/speed-quality-plan` (from `feat/home-studio`). Research only; no code changed.
Method: four web research passes (one per area plus cross-cutting), key claims re-opened by hand, then a read of our code so the plan does not repeat finished work.

**How to read the numbers.** "Measured" means a number on the cited page; it was measured by the page's authors on their hardware, almost always a CUDA GPU, not on our Mac. "Est." means my estimate, to be measured with the protocol in section 9. "Unverified" means I could not open the primary page; the claim came from a search snippet or a secondary write-up. Re-read every licence on the model card when pinning.

## 1. Summary

The biggest wins are not new models. They are in how we run the ones we have:

1. **Image and video re-load their model for every picture and every clip.** `mflux_engine.py` and `ltx2_mlx_runtime.py` start one CLI process per job, so each still pays Python imports, the weight load and the MLX first-call compile. The music spike measured that overhead at about 20 s per cold job (imports 6 s, load 4 to 6 s, first-call compile about 9 s). The lighthouse film spent about 85 s per still and 125 s per clip; we do not know how much of that is load. **Measure it, then keep the model warm in a worker** like the music and voice workers already do.
2. **Video frame interpolation is a linear blend** (`video_runtime/defaults.py::_interpolate_frames`), and it is not wired into the LTX or mlx-video paths at all. A real interpolator (RIFE 4.25, MIT, has an MLX port) lets us render half the frames and interpolate the rest.
3. **There is no image upscaler.** SeedVR2 (Apache 2.0) ships inside the mflux release we already pin.
4. **Our six cache methods only reach the diffusers path.** mflux and ltx-2-mlx (our main Mac engines) get none of them. A community MLX TeaCache plugin covers several mflux base models.

Top 8 recommendations (details in section 8):

| # | Item | Expected gain | Basis |
|---|---|---|---|
| 1 | Timing split (load / encode / denoise / decode) in every image and video job | Tells us where the 85 s and 125 s go | Prerequisite |
| 2 | Warm worker for ltx-2-mlx (Python API, pipeline kept between clips) | Est. 15 to 30 % per clip in a storyboard batch | Music spike cold-start numbers; ltx-2-mlx has a Python pipeline API |
| 3 | Warm worker for mflux (Python API, e.g. `ZImageTurbo(quantize=8)`) | Est. 15 to 40 % per still, more for fast models | Same; mflux README shows the Python API |
| 4 | RIFE 4.25 MLX interpolation, plus a "fast motion" mode (render 12 fps, interpolate to 24) | Quality: no ghosting at 2x. Speed est. 1.5 to 2x per clip | Port verified (MIT); speed is an estimate |
| 5 | `LTX2_COMPUTE_DTYPE=float16` A/B | Measured 701 s to 563 s (about 20 %) on M1 Max; smaller expected on M3/M4 | ltx-2-mlx README |
| 6 | SeedVR2 3B image upscaler (int8, not int4) | Quality: 2 to 4x size; make at about 1 MP, upscale | In mflux; weights Apache 2.0 |
| 7 | Z-Image-Turbo through mflux (MLX) instead of diffusers on MPS | Est. 1.5 to 3x for that model | mflux supports it; our catalog runs it on diffusers |
| 8 | Voice: smaller first chunk, cached speaker reference, loudness match and short crossfades at joins | First audio sooner; steadier voice over long reads | Standard practice, unverified figures |

## 2. What we already have (do not rebuild)

| Area | Already done | Where |
|---|---|---|
| Image | mflux families incl. FLUX.2 Klein 4B (4 steps), FLUX 3 Klein distilled, Qwen-Image 2.1 Turbo (8 steps), Iris-3B Turbo (4 steps); `--quantize` passed through | `image_runtime/mflux_engine.py` |
| Image | Diffusers path: TAESD preview decoder (FU-018), distill LoRAs (FU-019), attention slicing and VAE tiling on low memory, LLM prompt enhancer (FU-022), Nunchaku and FP8 foundations for CUDA (FU-023, FU-024) | `image_runtime/__init__.py`, `helpers/` |
| Image/video | Cache methods FBCache, TeaCache, TaylorSeer, MagCache, PAB, FasterCache (diffusers hooks only) | `cache_compression/` |
| Video | ltx-2-mlx distilled pipeline (8 passes half size, 2x neural upscale, 3 refine), LTX-2.3 and 2.5 q4 packs, start and end images | `ltx2_mlx_runtime.py` |
| Video | mlx-video (Wan on MLX), stable-diffusion.cpp video, LTX two-stage spatial upscale on diffusers, linear-blend interpolation (diffusers path only) | `mlx_video_runtime.py`, `video_runtime/` |
| Music | ACE-Step 1.5 turbo, 8 steps, shift 3; spike memory settings (VAE chunk 256, MLX cache limit 2 GB, torch decoder dropped: peak 66 GB to 21 GB); planner LM off by default; warm worker with idle stop; cancel | `music_worker_acestep.py`, `music_client.py` |
| Voice | Warm voice worker with idle stop; Parakeet TDT v3 and Whisper; Kokoro (mlx-audio or ONNX); sentence-by-sentence read-aloud stream (`/api/voice/speak/stream`); Silero VAD; diarisation | `voice_worker.py`, `routes/voice.py` |
| Storyboard | Batches by kind (all stills, then all clips) with an unload between kinds; one heavy job at a time | `storyboard/batch.py` |

The gap that matters most: the storyboard batches by kind so the model *could* stay loaded across shots, but both image and video engines exit after each job, so it never does.

## 3. Image

| Idea | Expected gain (claim + source, or est.) | Effort | Risk | Licence | Fits? | Rec. |
|---|---|---|---|---|---|---|
| I1. Warm mflux worker: load once through the Python API, keep between images, idle stop, `mx.set_cache_limit` and `mx.clear_cache` between jobs | Est. 15 to 40 % per still (removes imports, load, first-call compile; music measured about 20 s of that per cold job) | M | Holds 10 to 30 GB until idle stop; must release before video | mflux MIT | Yes: copy the music worker pattern (JSON lines, cancel, idle stop) | **Do now** |
| I2. Z-Image-Turbo on mflux (`mflux-generate-z-image-turbo` / `ZImageTurbo`) instead of diffusers on MPS | Est. 1.5 to 3x for that model; 8 steps, guidance 0 (negative prompt ignored) | S to M | Output differs slightly from the diffusers build | Apache 2.0 (card, unverified) | Yes: one `_FAMILIES` row plus catalog engine switch | **Do now** |
| I3. SeedVR2 3B upscaler (in mflux; MLX weights `mlx-community/SeedVR2-3B-mlx`, int8 for 7B `benc0/SeedVR2-7B-sharp-mlx-int8`) | Quality: big. Lets fast models run at about 1 MP then upscale, instead of slow native 2 MP. A 4-bit card reports 2.89 GiB peak on M5 Max (unverified) | M | int4 degrades this family badly (card); keep int8 or bf16 | Apache 2.0 weights, MIT mflux | Yes: new "Upscale" action on gallery items and as a storyboard still step | **Do now** |
| I4. mlx-teacache plugin for mflux base models (FLUX.1 dev, Klein base, Z-Image base, Qwen-Image) | TeaCache papers report about 1.5 to 2x at small quality cost; plugin gives no numbers (est.) | M | Community plugin, may lag mflux; little gain on 4 to 8 step models | Unknown, check repo | Partly: needs the warm worker (I1) | Next |
| I5. EasyCache as a 7th cache method (diffusers path) | Paper: 2.1 to 3.3x (abstract, https://arxiv.org/abs/2507.02860); FLUX 4.64x vs TeaCache 3.27x (secondary, unverified) | M | Code licence not stated | Check repo | Yes: same adapter shape as `teacache.py` | Next |
| I6. mflux 8-bit by default on 32 GB Macs and below; 4-bit only where tested | mflux publishes no speed or memory numbers; measure | S | 4-bit hurts fine detail | MIT | Yes: `mfluxQuantize` exists | Next (measure) |
| I7. `--compute-precision float16` on M1/M2 for FLUX.2, Klein, Z-Image, Qwen 2.1 | No number published (mflux README) | S | Possible artefacts | MIT | Yes: flag per chip | Next |
| I8. Draft then final: fast model or lower size for picking, then the chosen seed re-rendered at full size or upscaled | Est. 2 to 4x less time spent on rejected stills | M | UX work | n/a | Yes: storyboard takes already exist | Next |
| I9. NAG (negative prompt for distilled models) | Restores negatives without CFG cost (https://chendaryen.github.io/NAG.github.io, unverified numbers) | M | Port needed for mflux | Unknown | Not yet | Later |
| I10. Face and hand refiner (detect, crop, re-run with an edit model) | Quality for portraits | L | Common detectors are AGPL (Ultralytics) | Need a permissive detector | Partly (Klein edit can inpaint) | Later |
| I11. `mx.compile` of the DiT step | No published gain on Mac; Mac is bandwidth-bound (est. 0 to 10 %) | M | Compile time per shape | n/a | Only with I1 | Later (measure) |
| I12. Text-encoder output cache (same prompt, new seed) | Est. 1 to 3 s per rerun | S | Memory | n/a | Only with I1 | Later |
| I13. More FLUX.1 speed LoRAs (Hyper-SD, Turbo-Alpha), DMD2 | Already have FU-019 | n/a | FLUX.1-dev non-commercial; DMD2 CC-BY-NC-SA | Non-commercial | n/a | Skip |
| I14. 4x-UltraSharp | Popular upscaler | n/a | CC-BY-NC-SA (https://openmodeldb.info/models/4x-UltraSharp) | Non-commercial | n/a | Skip |
| I15. Nunchaku / SVDQuant, FP8 | Measured 3x on a 16 GB laptop 4090 (https://hanlab.mit.edu/blog/svdquant) | (exists) | CUDA only | Apache 2.0 code (unverified) | Foundations shipped (FU-023/024) | Later, CUDA lane (P7) |

Notes: Qwen-Image-Lightning (4 and 8 step, https://github.com/ModelTC/Qwen-Image-Lightning, Apache 2.0, "12 to 25x" vs 100 NFE) is covered by the Qwen-Image 2.1 Turbo row we already run; no new work unless we bring back the original Qwen-Image.

## 4. Video

| Idea | Expected gain (claim + source, or est.) | Effort | Risk | Licence | Fits? | Rec. |
|---|---|---|---|---|---|---|
| V1. Warm ltx-2-mlx worker: build the pipeline once through its Python API (e.g. `TI2VidTwoStagesPipeline(model_dir=...)`), reuse per clip, idle stop | Est. 15 to 30 % per clip (load of a 12 to 47 GB pack plus Gemma encoder plus compile). README does not say weights persist between calls: spike first | S spike, M build | Pins an internal API; RAM held | MIT | Yes: same worker pattern; storyboard batches clips already | **Do now** |
| V2. `LTX2_COMPUTE_DTYPE=float16` | Measured 701 s to 563 s, about 20 %, M1 Max, 2.5 q8, 576x1024x241 (https://github.com/dgrauet/ltx-2-mlx). M3+ have fast bf16, so expect less | S | Possible artefacts; A/B by eye | MIT | Yes: one env var | **Do now (A/B)** |
| V3. RIFE 4.25 MLX interpolation (`pip install rife-mlx`, weights `mlx-community/RIFE-4.25`, about 23 MB) replacing the linear blend, and wired into the LTX and mlx-video paths | Quality: real motion, no ghosting. Speed: "fast motion" mode renders half the frames and doubles them; est. 1.5 to 2x per clip (attention cost grows faster than frame count). No published Mac fps | M | Fast action can warp; LTX sound must keep its length (frame count halves, duration does not) | MIT (Practical-RIFE) | Yes: `_interpolate_frames` already takes RGB frames; add an ffmpeg round trip for CLI engines | **Do now** |
| V4. LTX temporal 2x upsampler (built into ltx-2-mlx) as the alternative to V3 | Same idea in latent space; no numbers published | S spike | May need the two-stage pipeline | LTX-2 Community License (we carry it) | Yes | Next (compare with V3) |
| V5. Quality mode: `--two-stages-hq` (or `--two-stage`) as "Final" next to distilled "Draft" | Quality: full guided pipeline; slower (no number published) | M | Longer runs; more memory | Same | Yes: one more pipeline row | Next |
| V6. `--nag` for the distilled pipeline, so the negative prompt works | Quality for "no text", "no extra people" | S | Marked experimental | Same | Yes: flag | Next |
| V7. Native `extend` and `retake` for long shots and fixing one second of a clip | Quality: smoother than last-frame chaining; distilled variants claim "about 10 to 20x faster" (README, experimental) | M | Experimental upstream | Same | Yes: fits storyboard takes | Next |
| V8. `--low-ram` and `--tile-frames` / `--tile-spatial` for 16 to 32 GB Macs and HD | Fit, not speed: q8 on 16 GB, bf16 on 32 GB (README) | S | Slower (streams blocks from disk) | Same | Yes: tie to fit gate | Next |
| V9. Prompt Relay (timed prompts in one clip) | Quality: action changes inside one shot | S | New UI | Same | Yes | Later |
| V10. EasyCache for diffusers video (Wan, LTX on diffusers) | Measured HunyuanVideo 2.21x at 32.66 PSNR vs TeaCache 1.67x at 23.85 (paper, https://arxiv.org/abs/2507.02860) | M | Shared with I5 | Check repo | Yes | Next (with I5) |
| V11. Wan 2.2 lightx2v 4-step LoRAs for mlx-video | Vendor: combined distill + NVFP4 + sparse attention "over 50x" on RTX 5090 (https://github.com/ModelTC/LightX2V); LoRA-only gain not stated | M | Needs two-expert LoRA support in mlx-video (unverified) | Apache 2.0 | Partly | Later (CUDA lane first) |
| V12. Video upscale with SeedVR2 | Quality big; official runs need an H100 for 720p (https://github.com/ByteDance-Seed/SeedVR); MLX video use unproven | L | Memory, time | Apache 2.0 | Not yet | Later |
| V13. FlashVSR | Measured about 17 fps at 768x1408 on A100 (https://github.com/OpenImagingLab/FlashVSR) | M | CUDA only (block-sparse kernels) | Apache 2.0 | CUDA lane | Later |
| V14. Sparse attention (Radial Attention 1.9x, STA 1.36 to 3.53x) | https://github.com/mit-han-lab/radial-attention, https://huggingface.co/papers/2502.04507 | L | CUDA kernels; no Mac path found | Unverified | No | Skip on Mac |
| V15. Colour drift fix when chaining clips (match the next clip's histogram to the last frame) | Est. visible on long chains; no sourced method | S | Can flatten intended light changes | n/a | Yes: ffmpeg filter in the Mixer | Later |
| V16. Video-to-sound: ThinkSound | Apache 2.0 except a Stable Audio Open VAE under Stability terms (unverified); MMAudio weights CC-BY-NC | M | Licence mix; Mac untested | Mixed | Partly | Later (P5 already lists it) |

Wan 2.5 / 2.6 weights: not open as far as I could find; Wan 2.2 is the last open release (unverified, check https://huggingface.co/Wan-AI).

## 5. Music

| Idea | Expected gain (claim + source, or est.) | Effort | Risk | Licence | Fits? | Rec. |
|---|---|---|---|---|---|---|
| M1. Speed: nothing big left on the Mac. 60 s in 5 to 6 s warm (about 10x real time) | Spike (`music-spike.md`) | n/a | n/a | MIT | n/a | Keep |
| M2. Repaint, extend and cover in the Studio | Spike measured cover 6.8 s and repaint 7.3 s on a 60 s track; repaint windows 3 to 90 s, chainable to extend (Tutorial, https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) | M | sft does not support cover / repaint (README) | MIT | Yes: worker change (`task_type`), plus UI | **Next (quality)** |
| M3. XL turbo as the "Best" render: keep seed and settings, re-render a turbo draft on XL | Est. better arrangement; no published delta. HF card rates turbo "very high", sft "high", base "medium" (vendor) | S | XL needs a 64 GB Mac; peak not measured with spike settings | MIT | Yes: XL row exists | Next |
| M4. Lyrics helper: 6 to 10 syllables per line, matching line counts per section, short section tags, BPM and key as fields not words | Tutorial guidance; quality of vocals and alignment | S | None | n/a | Yes: client-side hints | Next |
| M5. Make N variations, rank by the DiT lyrics alignment score, show the best first | Tutorial mentions the score; batch 4 at 60 s takes 23 s (spike) | M | Score may not match taste | MIT | Yes | Next |
| M6. "Smart" mode with planner LM 1.7B (Tutorial's recommended default size) | Infers BPM, key, structure; +1 to 4 s per run, about 23 s load, about 1.3 GB (0.6B measured), not seed-reproducible | S | Can rewrite a vocal prompt into instrumental (community claim) | MIT | Yes: switch exists | Next (owner) |
| M7. Stems: mlx-audio-separator (BS-RoFormer, Mel-RoFormer, htdemucs) | Measured 1.4 to 2.5x vs the torch package on an M4 mini, median 1.85x (https://pypi.org/project/mlx-audio-separator/, v0.1.20, 2026-10-02) | M | Each community checkpoint has its own licence, often unstated | Code MIT; weights per checkpoint | Yes: own venv or the music venv | Next (owner licence check) |
| M8. Mastering: ffmpeg only (EQ, `acompressor`, `alimiter`, two-pass `loudnorm`) | Est. consistent level and polish | S | Over-processing | Our ffmpeg build | Yes: Mixer already does loudness | Later |
| M9. LoRA styles (load only; training later) | README: about 8 songs trained in 1 h on a 3090 | M | Users bring their own data | MIT | Partly | Later |
| M10. Watch `acestep-v15-turbo-rl` ("to be released" on the HF card) | Unknown | S | n/a | MIT | Yes | Watch |
| M11. matchering, pedalboard | Reference mastering, effects | n/a | GPL-3.0 (pedalboard README; matchering per aggregator) | GPL | No (bundling) | Skip |
| M12. AudioSR (bandwidth repair) | 50 DDIM steps by default, no MPS note | n/a | Slow; weight licence unstated | Code MIT | No | Skip |
| M13. Other song models (DiffRhythm, HeartMuLa, SongBloom, YuE2, MiniMax Music 3) | No speed or quality win over ACE-Step established; several licences unclear or non-commercial | n/a | Licence | Mixed | n/a | Skip for this plan |

## 6. Voice

| Idea | Expected gain (claim + source, or est.) | Effort | Risk | Licence | Fits? | Rec. |
|---|---|---|---|---|---|---|
| Vo1. First chunk of 5 to 10 words, then full sentences; synthesise chunk n+1 while n plays | Est. first audio about halved for long first sentences | S | Odd prosody on the first chunk | n/a | Yes: `speech_sentences` already splits | **Do now** |
| Vo2. Speaker consistency for long reads: compute the voice reference once and reuse, fixed seed per session, loudness match per chunk, 20 to 50 ms crossfades | Est. steadier voice and level (practice, unverified) | S | None | n/a | Yes | **Do now** (with Qwen3-TTS) |
| Vo3. Qwen3-TTS (planned for cloning): 0.6B for speed, 1.7B for quality | Paper: first packet about 97 ms (0.6B) and 101 ms (1.7B) on GPU (https://arxiv.org/html/2601.15621). Community: MLX 1.7B 8-bit about real time on an M1 Pro 16 GB (https://pypi.org/project/qwen-tts-cli/0.4.0/, unverified) | M | Cloning needs the Base checkpoint; 0.6B adds lead-in silence (third-party) | Apache 2.0 | Yes: mlx-audio lists Base, CustomVoice, VoiceDesign | Next (P5) |
| Vo4. Dia 1.6B for two-speaker dialogue (`[S1]` / `[S2]`), mlx-audio | Quality for dialogue; Dia2 streams but has no MLX port found | M | English only | Apache 2.0 | Yes | Later |
| Vo5. Chatterbox (MIT; 16 languages listed in mlx-audio) | Turbo "up to 6x faster than real time" on GPU, adds a watermark (https://www.resemble.ai/learn/models/chatterbox-turbo) | M | Our earlier research called it retired; status conflicts; re-check | MIT | Yes | Later (re-check) |
| Vo6. STT: stay on Parakeet TDT v3 | Leaderboard: 3333 RTFx vs Whisper large-v3 145 RTFx, GPU (https://github.com/huggingface/open_asr_leaderboard) | n/a | No Mac RTFx published | CC-BY 4.0 (from memory) | Already shipped | Keep; measure on Mac |
| Vo7. Canary-Qwen 2.5B (best WER 5.63) | No MLX port found | n/a | Speed (418 RTFx GPU) | Unverified | No | Skip |
| Vo8. F5-TTS / E2-TTS | Weights CC-BY-NC | n/a | Licence | Non-commercial | n/a | Skip |
| Vo9. VibeVoice TTS | Microsoft pulled the TTS code (https://github.com/microsoft/VibeVoice) | n/a | Provenance | MIT fork only | n/a | Skip |
| Vo10. Kyutai TTS / STT, Sesame CSM, Higgs Audio v2, VoxCPM2 | Streaming or quality claims, all unverified here; Higgs licence has user caps | n/a | Unverified | Mixed | n/a | Later (re-research) |

## 7. Cross-cutting

| Idea | Expected gain | Effort | Risk | Rec. |
|---|---|---|---|---|
| C1. Timing split in every image and video job (import, load, text encode, denoise, decode, write), logged and returned in the job result | Shows where time goes; needed before claiming any win | S | None | **Do now** |
| C2. Keep-warm policy across kinds: one warm heavy model at a time; the queue unloads the previous kind before loading the next (as music does); idle stop (default 10 min, as voice) | Removes reloads inside storyboard batches; no 2-model RAM spikes | M | Bad policy can swap; reuse the memory gate | **Do now** (with V1, I1) |
| C3. MLX memory hygiene in every worker: `mx.set_cache_limit`, `mx.clear_cache()` between jobs, never raise the wired limit | Music: peak 66 GB to 21 GB and faster decode under pressure (spike) | S | None | **Do now** (with the workers) |
| C4. Text-encoder sharing: LTX-2.3 uses Gemma 3 12B 4-bit, LTX-2.5 Gemma 4; FLUX.2 / Z-Image use Qwen3 encoders. Sharing across engines is not realistic (different models and processes); cache embeddings per prompt instead | Est. small | M | n/a | Later |
| C5. Storyboard: with warm workers, the batch keeps one model for all stills, then one for all clips (already ordered by kind) | Est. 20 to 30 % off a lighthouse-size batch (est. from V1 and I1) | S once V1, I1 land | None | **Do now** (follows V1, I1) |
| C6. M5 Neural Accelerators | Apple: FLUX-dev 4-bit 1024x1024 "more than 3.8x faster" on M5 than M4, needs macOS 26.2+ (https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | S | Only for M5 owners | Keep MLX pins current in each venv; record chip in benchmarks |
| C7. Speculative diffusion (draft model proposes steps, verifier checks): ASDSV 1.77 to 3.01x on FLUX.1-dev and Wan 2.1 (https://papers.nips.cc/paper_files/paper/2025/hash/83c82e4af3146a12e456a68d9b3414b0-Abstract-Conference.html); FlowCast over 2.5x (https://arxiv.org/pdf/2602.01329) | Large in papers | L | Research code, no MLX, licences unchecked | Later (watch; the DFlash idea does not transfer directly) |
| C8. Core ML / Neural Engine for diffusion | No FLUX Core ML vs MLX comparison found; ANE suits large DiTs poorly (unverified study) | L | n/a | Skip |
| C9. Parallel jobs (image while video runs) | Est. negative on one GPU and shared memory | n/a | Swap | Skip (one heavy job at a time stays) |

## 8. Ranked roadmap (top 12)

Ranks weigh gain per day of work for the storyboard use case (many stills and clips in one go) on the owner's Mac.

| Rank | Item | Gain | Effort | Depends on | Done when |
|---|---|---|---|---|---|
| 1 | C1 timing split + benchmark script (section 9) | Measurement | S | none | Load vs denoise vs decode known for Klein edit, Z-Image-Turbo and LTX-2.5 q4 |
| 2 | V1 ltx-2-mlx warm worker (spike: does the pipeline object keep weights?) | Est. 15 to 30 % per clip | S + M | 1 | Second clip in a batch skips load; numbers in `PERF_BASELINE.md` |
| 3 | I1 mflux warm worker (+ C3 memory hygiene) | Est. 15 to 40 % per still | M | 1 | Same, for stills; cancel and idle stop as music |
| 4 | C2 + C5 keep-warm policy and storyboard reuse | Est. 20 to 30 % off a batch | S | 2, 3 | Lighthouse batch re-timed (was about 25 min) |
| 5 | V3 RIFE MLX interpolation + "fast motion" mode | Quality now; est. 1.5 to 2x per clip | M | 1 (for timing) | Side-by-side vs linear blend; LTX audio length intact |
| 6 | V2 float16 compute A/B | Up to about 20 % | S | 1 | Kept only if frames look the same to the owner |
| 7 | I3 SeedVR2 upscaler | Quality | M | 3 (shares the mflux venv) | Upscale action in gallery and storyboard; peak memory known |
| 8 | I2 Z-Image-Turbo on mflux | Est. 1.5 to 3x for that model | S to M | 3 | Same prompt, same seed, timed against diffusers |
| 9 | Vo1 + Vo2 voice first-chunk and consistency | Latency and steadiness | S | none | First audio time measured; long read A/B |
| 10 | M2 + M4 repaint / extend / cover + lyrics helper | Quality and control | M | none | Studio actions, E2E check |
| 11 | V5 + V6 + V7 LTX quality mode, NAG, extend / retake | Quality | M | 2 | Draft / Final switch; retake in storyboard |
| 12 | I4 + I5 / V10 caches for MLX (mlx-teacache) and EasyCache | Est. 1.5 to 2x on base (non-distilled) models | M | 3 | Speed and quality numbers per model; off by default for 4 to 8 step models |

Then: M7 stems (after licence check), M3 XL "Best", M5 variation ranking, V8 low-RAM, I8 draft-then-final, Vo3 Qwen3-TTS (already P5). CUDA lane items (I15, V11, V13) wait for the remote 4090 (P7).

Each shipped item needs an E2E check in `scripts/e2e_test_suite.py` (CLAUDE.md feature gate) and a FU row.

## 9. Benchmark protocol

Follows `scripts/perf-baseline.py` (each run in a fresh subprocess, models must already be cached, no downloads) and `scripts/perf-gate.py` (median of 3, 5 % tolerance, floors in `PERF_BASELINE.md`).

**Add to `perf-baseline.py`** (one `--only` label each): `image-mflux` (FLUX 3 Klein edit with one reference, Z-Image-Turbo, 1024x1024), `video-ltx2mlx` (LTX-2.5 q4 distilled, 768x448, 97 frames, start image), `music` (ACE-Step turbo, 60 s with lyrics), `voice` (Kokoro, 400-word paragraph; Parakeet on a 5 min file), `upscale` once I3 lands.

**Per case, record:**

1. Machine: chip, memory, macOS, MLX / mflux / ltx-2-mlx versions, power adapter on, Low Power Mode off, no other heavy jobs (note used memory before the run, as the music spike did).
2. Cold run (fresh worker) and then 3 warm runs; report the median of the warm runs and the cold value separately.
3. Time split from C1 (load, encode, denoise, decode, write) plus wall time.
4. Peak memory as `phys_footprint` sampled every 0.5 s (the music spike method; RSS misses GPU memory), cross-checked with `/usr/bin/time -l`.
5. Fixed prompts and seeds from a checked-in JSON file, so before and after are the same job.
6. Batch case: 4 stills then 4 clips through the storyboard batch route, total wall time (this is what the owner feels).

**Quality checks (no automatic score is good enough on its own):**

- Image and video: a contact sheet (same seed, before / after) for the owner to judge blind (labels hidden, order shuffled). For upscalers and interpolation, also a 100 % crop.
- Video: check frame count, duration and audio length match (`ffprobe`); for interpolation, scrub fast motion for warping.
- Music: integrated loudness and true peak with ffmpeg `ebur128` (as the spike); owner listens; for lyrics, Parakeet transcript vs the lyric sheet as a rough alignment check.
- Voice: time to first audio byte, real-time factor, and a 2-minute read checked by ear for voice drift.

**Pass rule:** a speed change ships when the warm median improves and the owner cannot pick the "after" contact sheet as worse. Update the floors in `PERF_BASELINE.md` only for a validated win.

## 10. Decisions for the owner (ELI5)

1. **Keep models in memory between jobs?** Faster second and later pictures and clips, but the app holds 10 to 47 GB until it goes idle (default 10 minutes). Recommend yes, with the existing idle stop.
2. **"Fast motion" video mode** (make half the frames, fill the rest in)? About twice as fast, slight risk of wobbly fast action. Recommend yes, off by default until you have looked at samples.
3. **Download the SeedVR2 upscaler** (3B model, several GB) as an optional extra? Recommend yes, optional.
4. **"Smart" music mode** (a small helper model plans tempo, key and structure)? Better songs from short prompts, but the same seed no longer gives the same song. Recommend offer it, off by default.
5. **Stem splitting** (vocals / drums / bass apart): the code is MIT, but each separation model has its own licence and many do not state one. Recommend only ship checkpoints with a clear permissive licence.
6. **No GPL audio tools in the bundle** (matchering, pedalboard). Recommend ffmpeg-only mastering.
7. **Draft and Final buttons** in Video Studio (fast distilled vs slower full-quality LTX)? Recommend yes.
8. **CUDA speed-ups** (Nunchaku, Wan 4-step LoRAs, FlashVSR) wait for the remote 4090 phase? Recommend yes, wait.

## 11. Not verified

- No Mac measurements exist for: mflux 4/8-bit, mx.compile, mlx-teacache, RIFE MLX fps, SeedVR2 on video, Qwen3-TTS on an M4, Parakeet v3 RTFx. All speed gains marked "est." depend on C1.
- Whether ltx-2-mlx's Python pipeline keeps weights between calls (README is silent): the V1 spike answers it.
- Licences read only from snippets or memory: Z-Image-Turbo, FLUX.2 Klein 4B (Apache 2.0 per BFL blog snippet), EasyCache, DiCache, mlx-teacache, Parakeet v3, Kyutai, Sesame CSM, Higgs Audio, Orpheus, ThinkSound's VAE, BS-RoFormer community checkpoints.
- Model-card and paper speed numbers are the authors' own, mostly on CUDA GPUs.
- ACE-Step: nothing newer than the XL release (2026-04-02) on the README; check commits after our pin (`ca1e85f`) before bumping.
- Chatterbox status conflicts between this research (MIT, mlx-audio lists a multilingual build) and the home-studio research ("retired").

## Sources (opened while writing this plan)

- ltx-2-mlx README (modes, float16 number, low-RAM, tiles, upsamplers, retake / extend, NAG, Python API): https://github.com/dgrauet/ltx-2-mlx
- mflux README (Python API, Z-Image-Turbo, SeedVR2, mlx-teacache plugin, quantize): https://github.com/filipstrand/mflux ; plugin https://github.com/IonDen/mlx-teacache (not opened)
- Apple M5 and MLX: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- RIFE 4.25 MLX: https://huggingface.co/mlx-community/RIFE-4.25 (package `rife-mlx`, https://github.com/xocialize/rife-mlx)
- mlx-audio-separator: https://pypi.org/project/mlx-audio-separator/
- EasyCache: https://arxiv.org/abs/2507.02860, code https://github.com/H-EmbodVis/EasyCache
- Qwen-Image-Lightning: https://github.com/ModelTC/Qwen-Image-Lightning
- SeedVR / SeedVR2: https://github.com/ByteDance-Seed/SeedVR, MLX weights https://huggingface.co/mlx-community/SeedVR2-3B-mlx
- ACE-Step 1.5 README and Tutorial: https://github.com/ace-step/ACE-Step-1.5, https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md
- Qwen3-TTS report: https://arxiv.org/html/2601.15621
- Open ASR leaderboard: https://github.com/huggingface/open_asr_leaderboard
- Others as cited inline (research agents' reads; treat as second-hand).
