# Lip sync for v0.10.0: model landscape and plan (2026-11-20)

Research for D-98 (a full local music video in which a character sings a generated song, lip-synced).
Read-only research: nothing was installed, downloaded or run. Every speed and memory figure below is a
vendor or third-party claim unless it says **measured**; none was measured on our 4090 or Mac.

## Headline

- **Primary, both platforms: LTX-2.3 audio-to-video (A2V), one performance shot at a time.** It is the
  same model family the Storyboard already uses, it takes a start image (the shot's reference still) plus
  an audio slice, and keeps that audio fixed while it makes the picture. On the Mac our pinned engine
  (`ltx-2-mlx`) already has an `a2v` command, so the Mac side is mostly wiring. On Windows, diffusers 0.42
  has **no** audio-to-video path, so we need either the Lightricks `ltx_pipelines` package in its own venv
  or a small custom denoise loop around our existing 4-bit LTX-2.3 pipeline. Use **2.3, not 2.5**, for
  singing (community reports, see risks).
- **Fallback / quality tier on Windows: InfiniteTalk** (Apache 2.0, Wan 2.1 14B 480p, fp8 single file,
  image-to-video *and* video-to-video dubbing, streaming long takes). Purpose-built for talking and singing,
  but its own old dependency stack (torch 2.4.1, xformers, flash-attn 2.7.4) and no Mac path.
- **Fallback on Mac: LTX-2.3 LipDub IC-LoRA** (video-to-video re-sync of a clip we already made), via the
  `lipdub` command of the same `ltx-2-mlx` engine (experimental, 2.3 only). Same engine also works on CUDA.
- **Watch, do not build for v0.10.0: LongCat-Video-Avatar 1.6** (MIT, released 2026-11-13/15, best long-form
  and singing claims) and **Kandinsky S2V 1.0** (MIT, Oct 2026, 2B Lite). Both are newer and possibly better,
  but neither has a tested 24 GB or Mac path yet.
- **Vocal stem first.** Every audio-driven model syncs better on an isolated vocal than on a full mix. Feed
  the model the vocal stem of the shot's slice, keep the full mix for the final film.
- **Effort:** about 15 to 18 working days for the primary pipeline on both platforms, plus 5 to 7 days for
  the InfiniteTalk tier. Tight but possible for v0.10.0 only if the two spikes in "Plan" pass in week 1.

## Comparison table

Legend: I2V = single image + audio in, V2V = existing video + audio in. "s/s" = seconds of compute per
second of output. UNVERIFIED = no source we could check for that number.

| Model (date) | Licence (commercial?) | Repo / weights | Input | Singing | Clip length, long songs | Res / fps | 4090 VRAM and s/s | Windows | Mac | Dep. conflicts | Face detector |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LTX-2.3 A2V** (Lightricks, 2026-03) | LTX-2 Community: free below US$10M yearly revenue | `Lightricks/LTX-2` (`A2VidPipelineTwoStage`); our CUDA row `OzzyGT/LTX-2.3-Distilled-bnb-nf4`; Mac `dgrauet/ltx-2.3-mlx-q4/q8` | I2V (+ text), audio held fixed | Yes, widely used; best results with a vocal stem | about 20 s per generation; chain shots | 768x512 typical, up to 1080p; 24 fps, frames 8k+1 | Our T2V/I2V row **measured** 22.3 GB peak, ~70 s per 4 s at 768x512 (8 steps). A2V uses the dev model with CFG, two stages: expect 2-4x slower (UNVERIFIED). Third-party: 10 s 480p A2V under 4 min on a 4090 | Lightricks code runs on SDPA; `natten` extra skipped on Windows. **diffusers 0.42 has no A2V** | **Yes: `ltx-2-mlx a2v` (our pinned engine)**, validated on 2.5 by the author; 2.3 implied | own venv for `ltx_pipelines`; none if custom diffusers loop | none |
| LTX-2.5 A2V (2026-08) | LTX-2.x Community, gated | `Lightricks/LTX-2.5`, `dgrauet/ltx-2.5-mlx-q4` | I2V | Inconsistent: users report 2.3 is better for singing, mouth sometimes ignores external vocals | as 2.3 | as 2.3 | as 2.3 | as 2.3 | `a2v` e2e-validated on 2.5 | as 2.3 | none |
| **LTX-2.3 LipDub IC-LoRA** (2026-05) | LTX-2 Community, gated | `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (`ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors`), `lipdub.py` / `DubItPipeline` | V2V | Not stated | per clip | matches source | ~16 GB+ (third-party, UNVERIFIED) | Lightricks code | `ltx-2-mlx lipdub` (experimental, not on 2.5) | as above | none |
| **InfiniteTalk** (MeiGen/Meituan, 2025-08) | Apache 2.0 (code and weights) | `MeiGen-AI/InfiniteTalk` + `Wan-AI/Wan2.1-I2V-14B-480P` + `TencentGameMate/chinese-wav2vec2-base` | I2V **and** V2V dubbing | Yes (vendor listings say talking or singing; widely used for songs) | 81-frame chunks, 9 motion frames carried over; streaming to 1000 frames (40 s) default, longer possible; colour shift beyond ~1 min | 480p / 720p, 25 fps | fp8 single-GPU file + `--num_persistent_param_in_dit 0` offload; 4-step lightx2v LoRA. One third-party: 28 GB+ and 16 min per 40 s (hosted vendor, conflicts with low-VRAM claims). ~70 s for a short V2V clip on a 4090 (length not given). UNVERIFIED | Not documented; flash-attn, xformers, `xfuser`, `decord` needed | No port | torch 2.4.1, transformers >=4.49 (own venv needed) | none (scene-cut detector only) |
| **LongCat-Video-Avatar 1.6** (Meituan, 2026-11-13/15) | MIT code and weights. **But** its 1.6 page says generated content is "for academic use only"; contradiction to resolve | `meituan-longcat/LongCat-Video-Avatar-1.6` (1.5, original also) | I2V, text, continuation; 1-2 speakers | Singing is a listed scenario (1.5 benchmark); no singing numbers | segments chained by continuation; >10 min claimed | 480p / 720p | Official: 2 GPUs. INT8 DiT ~15 GB. Community PR (1.5, unmerged): one 4090, ~2 min per 3.7 s segment after a 7 min compile, 70 min for 12 s at first; claims later retracted in part. UNVERIFIED | Not documented; WSL2 only report | MLX port of **1.5** (`mlx-community/LongCat-Video-Avatar-1.5-bf16`, q8 31 GB, q4 24 GB): 105 s for 29 frames at 256x432 on an M5 Max (too slow for a song) | torch 2.6, flash-attn 2.7.4, `audio-separator` 0.30.2 | none named |
| **Kandinsky S2V 1.0** Lite / Pro (Kandinsky Lab, 2026-10) | MIT (code, weights) | `kandinskylab/Kandinsky-S2V-1.0-Lite(-distill)(-Diffusers)`, `-Pro…`; uses `Qwen/Qwen2.5-Omni-7B` audio tower and HunyuanVideo VAE | I2V + text | Not mentioned | 5 s chunks with three-scale memory; length follows audio | SD 512x768, HD 1280x768, 25 fps | Vendor (H100): Lite-distill 8.7 s/s; peak 57 GiB, ~28 GB with VAE tiling. Does not fit 24 GB as documented | Not documented | No port (2B, MPS plausible, UNVERIFIED) | needs diffusers from `main` (modular pipeline) | none named |
| Wan 2.2 S2V 14B (Alibaba, 2025-08) | Apache 2.0 | `Wan-AI/Wan2.2-S2V-14B` | I2V (+ optional pose video) | Yes (singing examples) | auto from audio, `--num_clip` | 480p / 720p | Official single GPU: 80 GB; offload flags exist. Not in diffusers (fork PR only) | flash-attn | No | own repo | none |
| LiveAvatar (Alibaba Quark, 2025-12 / v1.1 2026-01) | Apache 2.0 (research preview) | `Quark-Vision/Live-Avatar` LoRA on Wan2.2-S2V-14B | I2V, streaming | Listed scenario | 10,000 s+ streaming | area-set | 80 GB, or 48 GB with fp8. Not a 4090 option | Linux scripts | No | torch 2.8, flash-attn 2.8.3 | none |
| SkyReels V3 A2V 19B (Skywork, 2026-01) | Skywork licence (custom; terms not checked) | `Skywork/SkyReels-V3-A2V-19B` | I2V | Claimed | audio up to 200 s | 720p (540p/480p low-VRAM), 24 fps | `--low_vram` fp8 + block offload for <24 GB. UNVERIFIED | Not documented | No | py 3.12, CUDA 12.8 | not named |
| EchoMimic V3 (Ant Group, 2025-07) | Apache 2.0 | `BadToBest/EchoMimicV3` (Wan2.1-Fun-1.3B-InP) | I2V | Karaoke listed in paper | 138 frames, longer via "Long Video CFG" | not stated (sub-720p) | tested on 4090D 24 GB | Linux-tested | MLX weights `Avdpro/EchoMimicV3-MLX` (weights only, no runner) | xformers | not named |
| HuMo 1.7B / 17B (ByteDance, 2025-09) | Apache 2.0 (most sources) | `bytedance-research/HuMo` | I2V + text | Not stated | short clips | 480p / 720p | 17B needs a large card; 1.7B fits | Not documented | No | Wan stack | not named |
| HunyuanVideo-Avatar (Tencent, 2025-05) | Tencent Hunyuan community licence; sibling models exclude EU, UK, South Korea (verify) | `tencent/HunyuanVideo-Avatar` | I2V | Yes (talk or sing) | 129 frames | 704 px | fp8 + `--cpu-offload` "very low VRAM"; slow | Linux scripts | No | flash-attn | not named |
| OmniAvatar, FantasyTalking, MultiTalk (2025) | Apache 2.0 | Wan 2.1 based | I2V | Partly | short / streaming | 480p | 14B heavy | Linux | No | Wan stack | — |
| **LatentSync 1.6** (ByteDance, 2025-06) | Apache 2.0 code/weights; **insightface default pack (buffalo_l) is non-commercial** | `ByteDance/LatentSync-1.6` | V2V (mouth region) | Not stated | any length (per-frame windows) | 512x512 face crop, 25 fps | 18 GB minimum (vendor); slow (diffusion, 20-50 steps) | Bash setup; Windows not documented | No port | pins diffusers 0.32.2, transformers 4.48 (own venv) | **insightface 0.7.3** + mediapipe + face-alignment |
| MuseTalk 1.5 (Tencent Music, 2025-03) | MIT code and weights; DWPose Apache, BiSeNet face-parsing MIT, S3FD/face-alignment BSD | `TMElyralab/MuseTalk` | V2V | Not stated | any length | **256x256 mouth region** (soft at HD), 25 fps | 4 GB laptop works; 30 fps+ on V100 | Windows demo tested | **MLX port** `mlx-community/MuseTalk-1.5-fp16` (~34 faces/s, 7 GB; preprocessing stays upstream) | mmcv 2.0.1 / mmpose 1.1 / torch 2.0.1 (hard on Windows, own venv) | S3FD, DWPose |
| TBDub (Taobao, 2026-09) | Apache 2.0 | `TaoLiveAIGC/TBDub` (teacher 30 steps, student 2 steps; Wan2.2 VAE from `KlingTeam/X-Dub`) | V2V | Not stated | tested to 376 frames | 512x512 | 15-19.5 GiB peak with `--cpu-offload` on a 72 GB card, 21 GiB host RAM; "not a 4090 validation" | **Linux only** | No | own | mediapipe (Apache) |
| Wav2Lip (2020) | **Non-commercial** (research only) | — | V2V | Poor | any | 96x96 mouth | tiny | yes | yes | old | S3FD |

Not open weights (excluded): WanSong (paper only, July 2026), Wan-Streamer (code only), Wan 3.0 (not
released). Wan-Dancer-14B (Apache, July 2026) makes dance from music, not lip sync; interesting for
B-roll later.

## Why LTX-2.3 A2V as the primary

1. **Already ours on both platforms.** Mac: `ltx-2-mlx` (pinned `895201d`) ships `a2v`
   (`ltx-2-mlx a2v --prompt … --audio vocals.wav --frame-rate 24 -o out.mp4`, default 97 frames, `--low-ram`,
   `--no-regen-audio`). Windows: our LTX-2.3 NF4 row is **measured** at 22.3 GB peak on the 4090.
2. **Shot-shaped.** The Storyboard already makes 2-8 s shots from a reference still. A2V adds one input
   (the audio slice) to the shot we already render. Long-take models (InfiniteTalk, LongCat) shine at
   one continuous 3-minute take, which a cut music video does not need.
3. **Same look as the B-roll.** Performance shots and B-roll come from the same model, so colour and
   texture match across cuts.
4. **Licence is known** and already accepted for the catalog (free below US$10M revenue).

Weak points (all in the risk list): sync quality is "good for creators", not studio grade; 2.5 is worse
than 2.3 for external vocals; diffusers has no A2V on Windows; a two-stage pipeline that re-noises the
audio latent in stage 2 makes the mouth drift (a third-party project hit exactly this and had to hold the
audio fixed through both stages, PortOS PR #11243).

## Windows 4090: two ways to get LTX A2V

- **(a) Lightricks `ltx_pipelines` in its own venv** (`python -m ltx_pipelines.a2vid_two_stage`,
  `--audio-path`, `--audio-start-time`, `--audio-max-duration`, image conditioning, `--quantization
  fp8-cast`, `--offload cpu`). Reference implementation, audio frozen by design. Needs the **dev** 22B
  transformer plus the distilled LoRA and Gemma 3 12B: about 22 GB fp8 + 12 GB encoder, so CPU offload and
  plenty of system RAM. Another large download. Code Apache 2.0.
- **(b) Custom loop on our diffusers NF4 distilled pipeline.** Encode the vocal slice with the LTX audio VAE,
  pass it as `audio_latents` with `noise_scale=0`, and keep the audio tokens clean at every step (audio
  timestep 0, or re-inject the clean latent after each step). `LTX2Pipeline` today re-noises and denoises
  whatever `audio_latents` it gets, and the visible code does not encode input audio, so this is our own
  code against diffusers internals. Smaller download (re-uses the 31 GB pack), but untested quality with
  the distilled model (the reference uses dev + CFG).

**Spike both for one day each and keep the one that lip-syncs.** If (b) syncs, it is cheaper to ship and
keeps one model on disk.

## Integration plan with the Storyboard

The Storyboard (on `test/rtx4090` / `feat/home-studio`) already has: song attach and analysis
(`storyboard/song.py`: tempo, beats, sections with energy), lyric timing by sections or speech-to-text
(`storyboard/lyrics.py`), reference images (`ref_ops.py`), shot planner and batch queue, and the assembler
(cuts, crossfades, music under the film, loudness).

1. **Vocal stem once per song.** ACE-Step 1.5 makes the song, and ACE-Step 1.5 has an "extract" task that
   pulls the vocal track out of a mix, so the stem can come from the engine we ship. If that is too slow or
   unclean, use a dedicated separator (Demucs, MIT; or `audio-separator`, MIT wrapper whose UVR model
   licences must be checked). Store `vocals.wav` next to the song; lip sync always uses the stem, the film
   always uses the full mix.
2. **Split the 3-minute song into shots.**
   - Start from lyric lines (already timed). A **performance shot** covers one or two lyric lines.
   - Cut on a beat (`track_beats`) at or just before the line start; keep 0.3-0.5 s of pre-roll so the mouth
     can open before the first syllable.
   - Length 3-8 s (LTX frames 8k+1 at 24 fps: 73, 97, 121, 145, 169, 193). Lines longer than 8 s split at a
     word gap (STT word times) or a beat.
   - Instrumental parts (no lyric) become B-roll shots (existing image-to-video, no lip sync).
   - Aim for 40-60 % performance shots. A 3-minute song at ~4.5 s per shot is about 40 shots, ~20 sung.
3. **Keep the face the same.** One **character reference** image per singer (made in Image Studio, stored as
   a project ref). Each performance shot gets its own start still made **from that ref** (image edit for a
   new angle, framing or outfit-consistent pose), then A2V animates that still. The still carries the
   identity; LTX keeps it within the shot. Prefer medium close-ups with the face at least ~1/4 of frame
   height; wide shots lose the mouth. Later: a per-character LoRA (not for v0.10.0).
4. **Render a performance shot.** Inputs: start still, vocal slice (with pre-roll), prompt describing look,
   light and camera only (not "singing loudly", which fights the audio), seed. Output video is muxed with
   the **full-mix** slice for preview; the assembler already lays the full song under the film, so the
   shot's own audio is dropped at assembly.
5. **Join.** Hard cuts on beats (no crossfade on sung shots: a crossfade shows two mouths). The existing
   assembler trims each shot to its slot; keep frame-exact alignment (slot start = slice start minus
   pre-roll).
6. **Retakes.** Lip sync fails per shot, not per film. Show 2 takes per sung shot (two seeds) in the batch,
   let the user pick, and allow "re-sync this shot" with LipDub (V2V) on the chosen take.
7. **Optional automatic check.** A SyncNet-style confidence score (LatentSync's SyncNet is Apache) could flag
   bad takes. Not for v0.10.0 unless time allows.
8. **Agent bridge.** A `lip_sync_shot` / `make_music_video` tool reuses the same job queue.

## Effort (working days, one engineer)

| Step | Days |
|---|---|
| Vocal stem (ACE-Step extract or separator), store, tests | 1.5 |
| Shot splitter from lyrics + beats, performance vs B-roll, 8k+1 lengths, tests | 2 |
| Mac: `ltx-2-mlx a2v` wiring (audio flag, frames, 2.3 pack, mux), live check | 1.5 |
| Windows: spike (a) and (b), pick one, implement, 24 GB recipe | 4-5 |
| Storyboard: performance shot type, per-shot slice, character ref to per-shot still, batch, takes | 3 |
| LipDub V2V re-sync (both platforms, experimental) | 1.5 |
| UI strings (en + null placeholders), E2E phase check, MCP tool, docs | 1.5 |
| **Primary total** | **15-17** |
| InfiniteTalk tier on Windows (own venv, fp8, offload, streaming, tests) | +5-7 |

## Risks, ranked

1. **Sync quality on sung vocals is not good enough** (high likelihood, high impact). Community reports say
   LTX lip sync is creator-grade, inconsistent on some shots, and 2.5 sometimes ignores external vocals.
   Mitigation: 2.3 not 2.5, vocal stem, front-facing medium close-ups, two takes per shot, LipDub re-sync,
   InfiniteTalk tier on Windows.
2. **No A2V in diffusers on Windows** (certain, medium impact): extra engine or custom loop; schedule risk.
   Mitigation: day-1 spike of both routes.
3. **Audio latent re-noised in stage 2 / distilled path** (medium): the mouth drifts. Mitigation: hold the
   audio latent fixed in every stage; test with a click track and a sung line.
4. **24 GB and system RAM on the 4090** (medium): dev 22B + Gemma 12B with offload may need 64 GB+ system
   RAM and be slow. Mitigation: measure; fall back to route (b) on the NF4 pack.
5. **Render time for a whole song** (medium): ~20 sung shots x 2 takes; at 2-4 min per take (UNVERIFIED) that
   is 1.5-5 h on the 4090 and more on a 64 GB Mac. Mitigation: preview at 512 px, final at 768 px; queue
   overnight; show the ETA (FU-153 estimate).
6. **Face drift between shots** (medium): each shot is re-anchored on a still made from the ref, but the image
   edit step can drift. Mitigation: one ref per character, keep the edit light, user picks stills first.
7. **Licence surprises** (low-medium): LongCat 1.6 page "academic use only" wording vs MIT; LatentSync's
   insightface pack non-commercial; UVR separator model licences; LTX gated repos (click-through for 2.5 and
   LipDub). Mitigation: primary path uses none of the risky items; Demucs for the stem.
8. **Python stack conflicts** (low for the primary, high for fallbacks): InfiniteTalk (torch 2.4.1,
   flash-attn), LongCat (torch 2.6, flash-attn), LatentSync (diffusers 0.32 / transformers 4.48), MuseTalk
   (mmcv 2.0.1) all need their own venv like the music and voice venvs. Windows flash-attn wheels are the
   usual blocker.
9. **Newest models move fast** (low): LongCat 1.6 is 5 days old and Kandinsky S2V a month old; a better
   option may appear before release. Re-check in two weeks; do not re-plan around it.

## What must be tested on the 4090

1. Route (a) `a2vid_two_stage` with `--quantization fp8-cast --offload cpu`: peak VRAM, system RAM, seconds
   for a 4 s and an 8 s 768x512 shot, sync on (i) a spoken line, (ii) a sung line from an ACE-Step song,
   (iii) the same line as full mix vs vocal stem.
2. Route (b) custom loop on `OzzyGT/LTX-2.3-Distilled-bnb-nf4`: same three clips; confirm the audio track
   in the output equals the input (no regeneration) and the mouth follows it.
3. LTX-2.3 vs 2.5 on the same sung line and still (decides which pack to wire).
4. LipDub IC-LoRA V2V on a generated clip: VRAM and sync.
5. InfiniteTalk fp8 single-GPU on Windows: does it install (flash-attn, xformers, decord) and run; s/s at
   480p with the 4-step LoRA; a 30 s sung take for identity and colour drift.
6. Assembly: 10 sung shots cut on beats against the full mix, played back to check A/V alignment within
   one frame at each cut.
7. Mac (M4 Max 64 GB): `ltx-2-mlx a2v` on the 2.3 q4/q8 pack, same clips, peak memory and time.

## Sources

- InfiniteTalk repo (licence, fp8, streaming, LoRAs, deps): https://github.com/MeiGen-AI/InfiniteTalk ;
  defaults from `generate_infinitetalk.py` and `requirements.txt` in that repo
- LongCat-Video (Avatar 1.5/1.6 flags, INT8, MIT, deps): https://github.com/meituan-longcat/LongCat-Video ;
  1.6 page: https://meigen-ai.github.io/LongCat-Video-Avatar-1.6-Page/ ; 4090 low-memory PR (open):
  https://github.com/meituan-longcat/LongCat-Video/pull/115 ; MLX port:
  https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16 ; 1.6 dates:
  https://casrai.org/news/longcat-video-avatar-1-6-open-weights-2026
- Kandinsky S2V: https://github.com/kandinskylab/kandinsky-s2v ,
  https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-distill , https://arxiv.org/abs/2610.19290
- LTX-2 pipelines, A2V, DubIt, Windows notes, licences: https://github.com/Lightricks/LTX-2 ;
  LipDub IC-LoRA: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub ;
  LTX-2 paper: https://arxiv.org/abs/2601.03233 ; A2V tutorial: https://ltx.io/blog/how-to-generate-video-from-audio
- LTX-2.5 vs 2.3 for singing (community): https://huggingface.co/Lightricks/LTX-2.5/discussions/57 ,
  https://huggingface.co/Lightricks/LTX-2.5/discussions/44
- Stage-2 audio re-noising pitfall: https://github.com/atomantic/PortOS/pull/11243
- ltx-2-mlx `a2v` / `lipdub`: https://github.com/dgrauet/ltx-2-mlx
- diffusers LTX-2 pipelines (no audio input): https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ltx2
- Wan 2.2 S2V: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B ; LiveAvatar: https://github.com/Alibaba-Quark/LiveAvatar
- SkyReels V3 A2V: https://huggingface.co/Skywork/SkyReels-V3-A2V-19B
- EchoMimic V3: https://huggingface.co/BadToBest/EchoMimicV3 , https://huggingface.co/Avdpro/EchoMimicV3-MLX
- HunyuanVideo-Avatar: https://huggingface.co/tencent/HunyuanVideo-Avatar
- LatentSync: https://github.com/bytedance/LatentSync (requirements, `latentsync/utils/face_detector.py`),
  https://huggingface.co/ByteDance/LatentSync-1.6
- MuseTalk: https://github.com/TMElyralab/MuseTalk ; MLX port: https://huggingface.co/mlx-community/MuseTalk-1.5-fp16
- TBDub: https://huggingface.co/TaoLiveAIGC/TBDub , https://arxiv.org/abs/2609.06144
- Wan 3.0 signals (WanSong, Wan-Dancer, Wan-Streamer): https://huggingface.co/blog/ResterChed/wan-3-0
- ACE-Step 1.5 extract task: https://github.com/ACE-Step/ACE-Step-1.5 , https://arxiv.org/abs/2602.00744
- Recent paper list (Kandinsky S2V, Hi-Singers, LiveSync etc.): https://github.com/liutaocode/talking-face-arxiv-daily
- In-repo: `docs/research/cuda-model-landscape-2026-11.md`, `docs/research/windows-model-landscape-2026-11-19.md`
  and `backend_service/catalog/video_models.py` on `test/rtx4090` (measured LTX-2.3 NF4 footprint).
