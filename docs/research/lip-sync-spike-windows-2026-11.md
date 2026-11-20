# Lip-sync spike on Windows: LTX-2.3 audio-to-video on the RTX 4090 (2026-11-20)

Spike for D-98 (lip-synced music video in v0.10.0), following
[lip-sync-landscape-2026-11-20.md](lip-sync-landscape-2026-11-20.md). Question: can LTX-2.3 take one start still
plus an audio clip and make a 768x512, 121-frame (5.04 s, 24 fps) shot whose mouth follows the audio, including
singing, on Windows 11 + RTX 4090 24 GB (64 GB RAM)? Everything below is **measured** on that machine unless it
says otherwise. I cannot listen to audio: every statement about sync comes from a face-landmark measurement and
from looking at frame strips, and every statement about what is sung comes from speech-to-text.

## Verdict

| Route | Runs on Windows? | Speech sync | Singing sync | Wall time per 5 s shot | GPU peak | Host RAM peak | Download |
|---|---|---|---|---|---|---|---|
| **(b) our own loop on the app's 4-bit distilled pack** (`OzzyGT/LTX-2.3-Distilled-bnb-nf4`, diffusers 0.42.1) | **Yes, no patches** | **Good** (r 0.50, chance 0.16) | **Partial**: follows phrase starts and stops, misses some lines | **110-126 s** (load 35 s + 64-69 s denoise + decode) | 21.8 GB whole card (torch 17.2 GiB) | 36-37 GB | 31 GB (the existing row, nothing new) |
| (a) Lightricks `ltx_pipelines` A2Vid two-stage, dev 22B fp8 + Gemma 3 12B, `--offload cpu` | **Yes, after 2 fixes** (a POSIX-only `mmap` flag; mono audio crashes) | Good (r 0.45, chance 0.17) | Weak on the vocal stem (mouth mostly closed), partial on the full mix | **270-284 s** | **9.0 GB** | 29.8 GB | 62 GB new |

**Recommendation: build route (b).** On Windows it is the same lip sync for speech, at least as good for singing
in these tests, 2.3x faster, and it needs no new download, no new venv and no upstream patches. The one thing
route (a) does better is GPU memory (9 GB, through its streamed offload), which matters for 12-16 GB cards, not
for the 24 GB card this spike targets. Keep route (a) as the reference to compare quality against, and as the
low-VRAM option later (FU-283).

**Singing is the open risk, not the engine.** Both routes lip-sync a spoken line well. On a sung line the mouth
opens and closes with the phrases but a sustained or hummed passage can stay closed while the singer is still
audible; this held across seeds and across both routes (the same passage both times). Two takes per sung shot
(already in the plan) and a pre-roll are needed; this spike does not show "studio grade" singing sync.

The InfiniteTalk fallback was not needed and not tried.

## Inputs (all made with the app's own tools)

Throwaway backend: worktree code, dev venv, port 8952, `HOME`/`USERPROFILE` in `E:\tmp\lipsync\home`,
`HF_HUB_CACHE=E:\AI Models`, the existing music and voice venvs pointed at with `CHAOSENGINE_MUSIC_VENV` /
`CHAOSENGINE_VOICE_VENV`. GPU work held `E:\tmp\gpu.lock`.

| Input | How | Result |
|---|---|---|
| Spoken line | `POST /api/voice/synthesize`, Kokoro `af_heart`: "I never thought the city lights would look this bright tonight, but here we are." | 3.73 s, 24 kHz mono; padded to 5.04 s with 0.4 s silence before and ~0.9 s after (so the mouth must also stay shut). STT gives the exact sentence back. |
| Sung line (main) | `POST /api/music/generate`, **ACE-Step 1.5 XL turbo**, 30 s, our lyrics, seed 11, 2 variations (75 s for both) | Variation `4490aaa8`: Whisper reads back the lyrics word for word. Window 16.40-21.44 s: "Sing it slowly. We will be home soon. Every heart...". |
| Sung line (second) | ACE-Step 1.5 **turbo** (not XL), three tries | All three sang something other than the lyrics (STT: "Ooh, ooh", "I'm not alone", "I'm not afraid of the dark"). The `ooh` window was kept as a hummed/sustained singing case. The XL row followed the lyrics, turbo did not, in 3 of 3 tries. |
| Vocal stem | Demucs `htdemucs` (MIT) in a tools venv | 0.7-0.9 s per 30 s song on the GPU, 0.65 GB. ACE-Step's own extract task is not exposed by the music route, so it was not tried. |
| Start still | `POST /api/images/generate`, FLUX.2 Klein 4B, 768x512, 4 steps, seed 78 | Head-and-shoulders, frontal, studio background (12.8 s with load). |

## How sync was measured

`sync_check.py` (spike tool, not in the repo): MediaPipe Face Landmarker (Apache-2.0; no insightface) on every
frame; mouth opening = inner-lip gap (landmarks 13-14) / face height (10-152). Audio envelope = RMS of the
driving audio in one-frame windows after a 100-4000 Hz band-pass. Reported:

- **r** = Pearson correlation of mouth opening and envelope at lag 0, with a **chance level**: the 95th percentile
  of r over 200 circular shifts of the envelope (12+ frames). r above chance means the mouth follows this audio.
- **VAD agreement**: share of frames where "voiced" (envelope > 15 % of max) equals "mouth open" (> 0.02), plus
  "closed while voiced" and "open while silent". Gaps between words count as silent, so speech never scores 1.0.
- A 12-frame strip with the two curves under it, which I looked at for every run.

Limits: envelope correlation suits speech; sustained sung notes have a flat envelope, so r is less meaningful for
singing (chance levels of 0.3-0.7 show that) and the VAD numbers and the strips matter more there. Nothing here
checks that the mouth makes the right *shape* for each sound (a SyncNet-style score would, FU-284).

## Route (b): our loop around the 4-bit diffusers pipeline

**What it does.** The clip becomes a log-mel spectrogram exactly as Lightricks' `ltx_core` makes it (16 kHz,
n_fft 1024, hop 160, 64 slaney mel bins, `log(clamp(x, 1e-5))`, stereo), is encoded with the pack's own audio VAE
(posterior mode, 126 latent frames for 5.04 s), and goes into `LTX2ImageToVideoPipeline` as `audio_latents` with
`noise_scale=0`. A forward pre-hook on the transformer then, on every call, puts the clean audio tokens back and
sets `audio_timestep` and `audio_sigma` to 0. That is what `ltx_core` does for a frozen modality (denoise mask 0,
sigma 0, so video cross-attends to clean audio and the prompt AdaLN for audio sees sigma 0). The pipeline's own
decoded audio is thrown away; the input clip is muxed in. Distilled schedule, 8 steps, `guidance_scale=1`,
`audio_guidance_scale=1` (no CFG pass), STG 1, modality guidance 3: three transformer passes per step.

**Control.** The same run without the hold (`--no-freeze`: audio tokens re-noised and denoised as in plain
text-to-video) gives a face that lurches, zooms and keeps its mouth open through the silent tail (open while
silent 0.92 vs 0.45). The hold is what makes the sync.

| Run (seed 42 unless noted) | r | chance | VAD agree | closed while voiced | open while silent | Strip |
|---|---|---|---|---|---|---|
| Speech | **0.50** | 0.16 | 0.74 | 0.04 | 0.45 | Mouth shut on the 0.4 s lead-in, talks through the line, shuts when it ends and stays shut. Identity and framing stable. |
| Speech, no hold (control) | 0.47 | 0.19 | 0.53 | 0.00 | **0.92** | Big head moves, camera push-in, mouth open the whole clip. |
| Sung (XL), vocal stem | 0.23 | 0.36 | 0.63 | 0.30 | 0.64 | Follows the first two phrases; closed through "home soon"; at the very end hands come up over the face (artifact). |
| Sung (XL), vocal stem, seed 7 | 0.24 | 0.41 | 0.64 | 0.28 | 0.64 | Same phrase pattern, no hand artifact. |
| Sung (XL), full mix | **0.41** | 0.22 | 0.71 | 0.29 | 0.00 | Same pattern; best of the sung runs. |
| Sung (ooh), vocal stem | 0.04 | 0.64 | 0.69 | 0.34 | 0.19 | Clip starts mid-phrase: mouth stays shut for that phrase, opens before the next one and holds an "oo" shape through the held note. |
| Sung (ooh), stem seed 7 / full mix | -0.35 / 0.05 | 0.61 / 0.74 | 0.60 / 0.60 | 0.35 / 0.40 | 0.56 / 0.00 | Same behaviour. |
| Sung (XL), modality guidance 5 instead of 3 (stem / mix) | 0.21 / 0.40 | 0.39 / 0.24 | 0.63 / 0.74 | 0.30 / 0.26 | 0.64 / 0.00 | No visible change from modality 3; stronger audio-to-video guidance does not fix the closed passage. Same cost (62 s denoise). |

Stem vs full mix: no win for the stem in these runs (XL: mix 0.41 vs stem 0.23 against their own envelopes; the
mix video against the stem envelope scored 0.16). Two takes are not enough to say the stem is worse; keep the
plan's stem default but test both on more shots (FU-281).

**Cost per 5.04 s shot** (each run its own process, weights read from the page cache): load 34.8 s, denoise
63.6-69.3 s (6.1-6.4 s per step, three passes), whole process 110-126 s including VAE decode and mux. Whole-card
GPU peak 21.7-21.8 GB above an idle 1.6 GB (torch reserved 17.2 GiB: `enable_model_cpu_offload` moves the text
encoder, transformer and VAE on and off). Host RSS peak 36.4-37.0 GB. A warm worker that keeps the pipeline
loaded would save the 35 s load on every shot after the first.

## Route (a): Lightricks `ltx_pipelines` (A2VidPipelineTwoStage)

Setup: `Lightricks/LTX-2` at commit `daa6b16` (2026-11-19), `ltx-core` / `ltx-pipelines` 1.6.2 (Apache-2.0)
installed into its own venv (`E:\tmp\lipsync\venv-ltx`, Python 3.12, torch 2.11.0+cu128, transformers 5.14.1;
upstream caps transformers below 5.15). No triton on Windows, so fp8-cast uses its eager path; the `natten` extra
is Linux-only and was skipped. Weights (all ungated):

| File | Repo | Size |
|---|---|---|
| `ltx-2.3-22b-dev-fp8.safetensors` (transformer + VAEs, one file) | `Lightricks/LTX-2.3-fp8` | 29.15 GB |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (stage 2) | `Lightricks/LTX-2.3` | 7.61 GB |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `Lightricks/LTX-2.3` | 1.00 GB |
| Gemma 3 12B (`--gemma-root`) | `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` | 24.4 GB |

`google/gemma-3-12b-it-qat-q4_0-unquantized` (the repo the LTX docs link) is **gated** and the app's saved token
has not accepted it (HTTP 403). Lightricks publishes an ungated copy of the same files, which is what was used.
The 46 GB bf16 dev checkpoint was not needed: `--quantization fp8-cast` folds the fp8 checkpoint's scales.

Command (stage 1 at 384x256 with CFG 3, STG 1, modality 3 for 30 steps; stage 2 upsamples x2 and refines 3
distilled steps; audio frozen in both stages; the input audio is muxed into the output):

```
python -m ltx_pipelines.a2vid_two_stage --checkpoint-path ltx-2.3-22b-dev-fp8.safetensors \
  --gemma-root <gemma snapshot dir> --distilled-lora ltx-2.3-22b-distilled-lora-384-1.1.safetensors 0.8 \
  --spatial-upsampler-path ltx-2.3-spatial-upscaler-x2-1.1.safetensors --quantization fp8-cast --offload cpu \
  --prompt "..." --image still.png 0 1.0 --audio-path clip.wav --num-frames 121 --frame-rate 24 \
  --height 512 --width 768 --num-inference-steps 30 --seed 42 --output-path out.mp4
```

**Two Windows blockers found and patched in the throwaway venv** (to report upstream, FU-282):

1. `--offload cpu` crashes at the first weight load: `ltx_core/loader/buffers.py` `host_bytes()` calls
   `mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)`; neither flag exists on Windows
   (`AttributeError: module 'mmap' has no attribute 'MAP_PRIVATE'`). Patch: use plain `mmap.mmap(-1, n)` when the
   flags are missing (anonymous on Windows already). After that the streamed offload ran fine.
2. A **mono** WAV (Kokoro's output) crashes the audio VAE (`expected input[1, 1, 507, 66] to have 2 channels`):
   the pipeline does not up-mix. Workaround: pass a stereo file (`ffmpeg -ac 2`). An app wrapper must always do it.

| Run (seed 42) | r | chance | VAD agree | closed while voiced | open while silent | Strip |
|---|---|---|---|---|---|---|
| Speech | **0.45** | 0.17 | 0.69 | 0.12 | 0.47 | Calm, natural; smaller mouth movements than (b), shut before and after the line. |
| Sung (XL), vocal stem | -0.10 | 0.32 | **0.22** | **0.86** | 0.44 | Mouth mostly closed, eyes closing, reads as listening; opens only at the end. |
| Sung (XL), full mix | 0.33 | 0.25 | 0.56 | 0.44 | 0.00 | Eyes closed, head down, camera drifts; mouth opens on the louder phrases. |

**Cost per 5.04 s shot**: 270-284 s wall (one process per shot: text encoder, two streamed transformer builds,
upsampler, decoder). Whole-card GPU peak **9.0 GB** above idle; host RSS peak **29.8 GB**. The offload streams
weights through pinned host buffers, which is why VRAM is low and the run is slow.

## Model files and sizes (route b, the one to build)

Nothing new: the existing catalog row `OzzyGT/LTX-2.3-Distilled-bnb-nf4` (31.0 GB on disk, Apache-2.0
conversion of LTX-2.3 under the LTX-2 Community licence: free below US$10M yearly revenue). Files it uses for A2V:
`transformer/` 14.4 GB NF4, `text_encoder/` 8.4 GB Gemma 3 NF4, `connectors/` 6.3 GB, `audio_vae/` 0.11 GB
(the encoder half is what A2V adds), `vae/` 1.45 GB, `vocoder/` 0.26 GB (unused for A2V). Pins: diffusers
0.42.1, transformers 5.20.2, torch 2.11.0+cu128, bitsandbytes 0.50.2, accelerate 1.15.0 (the dev venv as
tested), plus librosa for the mel filterbank (already in the venv; torchaudio is not needed).

## Runtime module sketch (route b, not wired)

A sketch of what the app would call, for the person who builds it. It is not in `backend_service/` and has no
tests. It reuses the existing video worker's loaded `LTX2ImageToVideoPipeline` (same row, same load path).

```python
# backend_service/video_runtime/ltx2_a2v.py  (sketch)
"""LTX-2.3 audio-to-video on the diffusers pipeline: the shot's audio slice is encoded and held fixed."""
import librosa, numpy as np, soundfile as sf, torch

FPS = 24.0
LATENTS_PER_SECOND = 16000 / 160 / 4  # audio VAE: 16 kHz, hop 160, 4x temporal compression -> 25 / s


def encode_audio_clip(pipe, wav_path: str, seconds: float) -> torch.Tensor:
    """[1, 8, L, 16] audio-VAE latent of the clip, made exactly like ltx_core (stereo, log-mel, posterior mode)."""
    wav, sr = sf.read(wav_path, always_2d=True, dtype="float32")
    wav = wav.T if wav.shape[1] > 1 else np.repeat(wav.T, 2, axis=0)  # mono -> stereo (the VAE has 2 inputs)
    wav = librosa.resample(wav[:2], orig_sr=sr, target_sr=16000) if sr != 16000 else wav[:2]
    n = int(round(seconds * 16000))
    wav = np.pad(wav[:, :n], ((0, 0), (0, max(0, n - wav.shape[1]))))
    spec = torch.stft(torch.from_numpy(wav), 1024, 160, 1024, torch.hann_window(1024), center=True,
                      pad_mode="reflect", return_complex=True).abs()
    fb = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64, fmax=8000.0, norm="slaney")).float()
    mel = torch.log(torch.einsum("mf,cft->cmt", fb, spec).clamp(min=1e-5)).permute(0, 2, 1)[None]
    with torch.no_grad():
        lat = pipe.audio_vae.encode(mel.to(pipe.audio_vae.device, pipe.audio_vae.dtype)).latent_dist.mode()
    return lat[:, :, : round(seconds * LATENTS_PER_SECOND)].float()


def hold_audio_fixed(transformer):
    """Pre-hook: every transformer call sees the clean audio tokens at timestep 0 and sigma 0 (ltx_core 'frozen')."""
    clean = {}

    def hook(_module, args, kwargs):
        ah = kwargs["audio_hidden_states"]
        clean.setdefault("x", ah[:1].detach().clone())  # first call: noise_scale=0, so this is the clean latent
        kwargs["audio_hidden_states"] = clean["x"].to(ah.device, ah.dtype).expand_as(ah).contiguous()
        zeros = torch.zeros(ah.shape[0], device=ah.device, dtype=kwargs["timestep"].dtype)
        kwargs["audio_timestep"] = zeros
        kwargs["audio_sigma"] = zeros
        return args, kwargs

    return transformer.register_forward_pre_hook(hook, with_kwargs=True)  # .remove() after the shot


def generate_a2v_shot(pipe, *, image, wav_path, prompt, frames=121, width=768, height=512, seed=0):
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    seconds = frames / FPS
    lat = encode_audio_clip(pipe, wav_path, seconds).to("cuda")  # cuda: the pipeline noises it with a cuda generator
    handle = hold_audio_fixed(pipe.transformer)
    try:
        out = pipe(image=image, prompt=prompt, width=width, height=height, num_frames=frames, frame_rate=FPS,
                   sigmas=list(DISTILLED_SIGMA_VALUES), num_inference_steps=8, guidance_scale=1.0,
                   audio_guidance_scale=1.0, stg_scale=1.0, modality_scale=3.0, audio_latents=lat, noise_scale=0.0,
                   generator=torch.Generator("cuda").manual_seed(seed), image_crf=0, output_type="np")
    finally:
        handle.remove()
    return out.frames[0]  # mux with the shot's full-mix slice (helpers/video_audio), never the decoded audio
```

Wiring notes: a `audioPath` (+ start, duration) field on the img2video request for LTX-2.3 diffusers rows only;
frames from the slice length snapped to 8k+1; the start still re-compressed at CRF 33 as today; the existing
audio mux path; Storyboard performance shots call it with the vocal slice plus 0.3-0.5 s pre-roll; a new E2E check
(speech clip, mouth-open correlation above its shuffled baseline) per the build checklist.

## What was not verified

- **Listening.** I cannot hear the clips; sync is a landmark measure plus my reading of frame strips.
- **Mouth shapes.** No viseme or SyncNet check; "opens with the audio" is all that is shown.
- **Sample size.** One spoken line, two sung windows, 1-2 seeds each, one face, one framing (head and
  shoulders, frontal). No side angles, no wide shots, no men, no fast rap, no 8 s shots.
- **Route (a) knobs.** Default A2V guidance only; no fewer-step or `--offload none` runs; no LipDub (V2V) re-sync.
- **LTX-2.5** was not tried for A2V (the landscape page says 2.3 is better for singing; not checked here).
- **Mac** (`ltx-2-mlx a2v`) not touched.
- **Warm timing.** Every run was a fresh process; per-shot time in a warm worker was not measured.
- **Shared card.** Other agents used the GPU between my runs; each run held the GPU lock and started from a
  1.6-1.8 GB idle baseline, so the peaks are this job's.

## Files

Spike scripts and outputs (not in the repo): `E:\tmp\lipsync\tools\` (`route_b.py`, `route_a.ps1`,
`sync_check.py`, `monitor.py`, `prep_audio.py`, `pdl.py`), clips and strips in `E:\tmp\lipsync\out\`, inputs in
`E:\tmp\lipsync\inputs\`. The patched upstream file is `E:\tmp\lipsync\venv-ltx\Lib\site-packages\ltx_core\loader\buffers.py`.
New weights in `E:\AI Models`: `Lightricks/LTX-2.3-fp8` (dev file only), `Lightricks/LTX-2.3` (LoRA, upscaler),
`Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`, `OzzyGT/LTX-2.3-Distilled-bnb-nf4`.
