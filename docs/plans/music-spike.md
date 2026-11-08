# P0 music spike: ACE-Step 1.5 on Apple Silicon

Date: 2026-11-06. Machine: this Mac (Apple Silicon, 64 GB unified memory, macOS, Darwin 27.3). Plan: [home-studio.md](home-studio.md) sections 3.1, 5.1, 7 (P0), 8.
Throwaway environment: `~/.chaosengine/spikes/acestep/` (venv, weights, harness, raw logs). Samples: `~/ChaosEngineAI-demos/music-spike/`.
Nothing in the repo was changed except this file.

## 1. Decision

**GO-WITH-CAVEATS** for ACE-Step 1.5 2B turbo (`acestep-v15-turbo`) as the first music engine on Apple Silicon.

- Speed is not the risk. A 60 s instrumental or song takes **about 5 to 6 s** warm (10 to 12 x real time); 600 s takes 82 s.
- **Memory is the risk, and it is fixable from the worker without patching the library.** With the library's defaults the process peaks at **about 66 GB footprint** for a 60 s track on this 64 GB Mac (it swaps). With three worker-side settings (section 4) the peak is **20 to 23 GB** for every case tried (60 s to 600 s, batch 4), output bit-identical.
- Caveats: no cancel API (kill the worker, or raise from a step hook, section 6); diffusion progress is only 8 coarse steps and the other half of the time (VAE decode) has no progress; the planner LM is not seed-reproducible; quality could not be judged by ear (owner must listen).

## 2. Versions and pins

| Item | Value |
|---|---|
| Repo | `https://github.com/ace-step/ACE-Step-1.5`, branch `main` |
| **Pinned commit** | **`ca1e85fe9430179831e6bc6be790c332190a3866`** (2026-08-29, "fix(jetson): remove GPL video codecs..."), 21 commits after tag `v0.1.8` (`dce6214`, 2026-05-18). Package version in `pyproject.toml`: `ace-step 1.5.0` |
| Python | `requires-python = ">=3.11,<3.13"`. Measured on 3.11.2 (uv venv, `uv sync --frozen`); pip-only venv on 3.12.5 also works with identical output (section 9) |
| Key packages (from `uv.lock`) | torch 2.10.0 (MPS), torchaudio 2.10.0, torchcodec 0.10.0, mlx 0.30.6, mlx-lm 0.29.1, transformers 4.57.6, diffusers 0.37.1, numpy 2.3.5, huggingface-hub 0.36.0, gradio 6.2.0, lightning 2.6.1, peft 0.18.1, torchao 0.16.0, modelscope 1.34.0 |
| Weights, main | `ACE-Step/Ace-Step1.5` rev `19671f406d` (10.09 GB: 2B turbo DiT 4.79 GB, VAE 0.34 GB, Qwen3-Embedding-0.6B text encoder 1.19 GB, planner LM 1.7B 3.71 GB) |
| Weights, planner LM 0.6B | `ACE-Step/acestep-5Hz-lm-0.6B` rev `148d8ea022` (1.37 GB) |
| Weights, XL turbo | `ACE-Step/acestep-v15-xl-turbo` rev `d4a0b288b8` (19.95 GB, bf16, 4 shards) |
| Other sizes (not downloaded) | `acestep-5Hz-lm-4B` 8.43 GB; `acestep-v15-base` / `-sft` / `-turbo-shift3` 4.79 GB each |
| All repos above | `license: mit`, `gated: false` (HF API) |

## 3. Measurements

All runs: one process, MLX DiT and MLX VAE active (log line `MLX DiT: Active (native MLX)`, `MLX VAE: Active (native MLX)`), torch on `mps` in float32, 8 steps, shift 3.0, ODE (Euler) sampler, no planner LM unless stated. "Footprint" is macOS `phys_footprint` of the process (the number Activity Monitor shows; includes Metal buffers), sampled every 0.5 s; `/usr/bin/time -l` "peak memory footprint" agrees. RSS alone is misleading here (it misses most GPU memory). "x RT" = seconds of audio per second of wall time. Other light jobs were running on the Mac (system used memory before each run 11 to 25 GB), so absolute times can move by about 20 %.

### 3.1 Library defaults (what you get out of the box)

| Case | Wall | x RT | Library timings (diffusion / VAE decode) | Peak footprint |
|---|---|---|---|---|
| (a) 30 s lo-fi, first generation in the process (cold) | 14.6 s | 2.1 | 1.3 s / 4.7 s (rest is MLX first-call compile and warm-up) | not sampled (RSS 10.8 GB; `time -l` 67.4 GB for the whole run) |
| (e) same, second run (warm) | 5.0 s | 6.0 | 1.6 / 2.3 | |
| (b) 60 s orchestral, 90 bpm | 10.7 s | 5.6 | 2.0 / 7.2 | 66.1 GB (repeat run) |
| (c) 60 s indie pop, female vocal, lyrics | 9.8 s | 6.1 | 2.4 / 5.9 | about 66 GB |
| 120 s orchestral | 19.0 s | 6.3 | 4.5 / 13.4 | |
| 180 s orchestral | 32.6 s | 5.5 | 9.3 / 17.0 | |
| 30 s, batch 4 | 8.5 s | 14.2 | 3.4 / 3.5 | 47.3 GB |
| 60 s, batch 4 | 25.0 s | 9.6 | 7.9 / 15.8 | 66.1 GB |
| Idle after load | | | | 20.4 GB |

The peak comes from the MLX buffer cache growing during VAE decode (timeline: 22.7 GB after diffusion, 31.6, 49.1, 58.3, 66.1 GB during decode, back to 21.3 GB after). `mx.get_peak_memory()` reports only 15 to 19 GB of live arrays; the rest is cache. Under this pressure VAE decode is also 2 to 3 times slower than it needs to be.

### 3.2 Recommended worker settings (section 4)

| Case | Wall | x RT | Diffusion / VAE | Peak footprint |
|---|---|---|---|---|
| Idle after load | | | | 17.1 GB, 15.2 GB after the first generation |
| 30 s, reference audio given | 3.6 s | 8.5 | 1.0 / 1.6 | 18.8 GB |
| **60 s orchestral, 5 back-to-back runs (seeds 100 to 104)** | **5.5, 5.7, 5.9, 6.1, 6.1 s** | 9.8 to 10.9 | 1.7 to 2.1 / 2.8 to 3.6 | 20.3 to 21.3 GB, idle back to 15.2 GB after each |
| **60 s song with vocals (2 seeds)** | **6.2 s** | 9.7 | 2.1 / 3.4 | 21.6 GB |
| 180 s orchestral | 18.1 s | 9.9 | 7.0 / 10.1 | 21.8 GB |
| 300 s orchestral | 33.5 s | 9.0 | 14.5 / 17.5 | 39.9 GB with VAE chunk 256 only (setting 1 of 3); not rerun with all three |
| 600 s orchestral (the library maximum) | 82.4 s | 7.3 | 40.8 / 39.6 | 22.7 GB |
| 60 s, batch 4 | 23.0 s | 10.4 | 8.4 / 13.8 | 22.3 GB |
| Cover of the 60 s track (strength 0.5) | 6.8 s | 8.9 | 1.7 / 2.9 (+1.1 s VAE encode) | 21.0 GB |
| Repaint 20 to 35 s of the 60 s track | 7.3 s | 8.3 | 2.0 / 3.4 | 21.6 GB |
| Whole process, `time -l` peak (8 cases incl. 600 s) | | | | 22.9 GB |
| After in-process unload (`del` handlers, `gc`, `mx.clear_cache`, `torch.mps.empty_cache`) | | | | 1.06 GB (process exit frees everything) |

Effect of each setting on the 60 s case (peak footprint / wall): defaults 66.1 GB / 9.6 s; VAE chunk 256 only 35.7 GB / 5.1 s; MLX cache limit 2 GB only 30.4 GB / 6.3 s; both 26.7 GB / 5.2 s; both plus dropping the unused torch DiT decoder 20.4 GB / 5.9 s. Chunk 256 and chunk 1024 outputs are bit-identical; so are outputs with and without the decoder drop.

### 3.3 Start-up costs

| Step | Time |
|---|---|
| uv venv install (125 packages, empty uv cache) | 43 s, 1.2 GB venv |
| Weights download (main + LM 0.6B, 11.5 GB) | about 20 min on this connection (2.6 to 10 MB/s, varied) |
| Python imports (`torch`, `acestep.handler`, `acestep.inference`) | 24 s first ever, 5.7 to 6.5 s after |
| DiT + VAE + text encoder load and MLX conversion | 4.3 to 6.2 s |
| First generation in a fresh process | about +9 s over a warm one (MLX compile of the VAE and DiT) |
| Planner LM 0.6B load (MLX) | 22.7 s (10.7 s of it "MLX model loaded") |
| So: worker cold start to first 60 s track | about 25 s; warm, 5 to 6 s |

### 3.4 Planner LM (case d)

30 s lo-fi, seed 42, LM `acestep-5Hz-lm-0.6B` on MLX:

| Variant | Wall | LM time | Peak footprint | Notes |
|---|---|---|---|---|
| No LM loaded | 3.4 to 5.0 s | 0 | 20.7 GB | deterministic |
| LM loaded, `thinking=True` | 7.0 s, then 5.4 s | 3.7 s, then 2.6 s | 23.2 GB | wrote a 400+ character caption and chose 108 bpm, then 106 bpm on the identical second call |
| LM loaded, `thinking=False` | 4.0 s | 1.0 s | 20.7 GB | still rewrites the caption (`use_cot_caption` defaults to True) and reported bpm 300 in its metadata |

The LM adds 1 to 4 s and about 1.3 GB, costs 23 s to load, and makes the output **not reproducible from the seed** (two identical calls gave different audio). Not tried: LM 1.7B (downloaded, part of the main repo) and 4B.

### 3.5 Output format and measured audio facts

- The library returns a float32 tensor `[2, samples]` at **48 kHz stereo** plus a file. `audio_format="wav"` and `"flac"` write **16-bit**; `"wav32"` writes 32-bit float.
- **Peak-normalised to -1.0 dBFS by default** (`enable_normalization=True`, `normalization_db=-1.0`): every file has a sample peak of 0.8912. No clipped samples in any file. True peak -0.2 to -1.0 dBTP.
- **Duration is exact**: requested 17, 30, 45, 60, 120, 180, 600 s gave exactly that many samples.
- Integrated loudness varies a lot per track: -13.0 to -18.8 LUFS (LRA 1.6 to 12.1 LU). The mixer must loudness-normalise (`loudnorm`), as planned.
- Bandwidth (highest frequency within 60 dB of the spectral maximum): 10 to 12.6 kHz for the lo-fi prompts (plausible for the genre), 15 to 20 kHz for orchestral and pop. 99 % of energy below 0.9 to 7.9 kHz.
- Stereo correlation 0.39 to 0.94 (real stereo, not dual mono).
- Analysis used the bundled ffmpeg 7.1 (`ebur128=peak=true`, `astats`) plus a numpy FFT script (`harness/analyze.py`). **I could not judge musical quality, vocal intelligibility or lyric alignment by ear.**

### 3.6 Controls

| Control | Passed as | Effect measured |
|---|---|---|
| Seed | `GenerationParams.seed` plus `GenerationConfig(use_random_seed=False, seeds=[...])` | Same seed twice: **byte-identical** WAV (also across processes). Not true when the planner LM runs |
| BPM | `bpm` (30 to 300) | Same seed, bpm 70 vs 120: different tracks (correlation with the no-bpm track 0.11 and 0.16). Rough onset-tempo estimate: 119.7 for bpm 120; 93.8 for bpm 70 (a 4:3 relation, so 70 is not confirmed). The LM's chosen 106/108 bpm measured 106.1. Tempo estimator is crude; owner should check by ear |
| Key | `keyscale` ("A minor", "F# major") | Strongest pitch classes: A minor run A, B, C, F; F# major run B, F#, F (E#), A#, all in the asked scale. Consistent with the key being followed, not proof |
| Duration | `duration` (10 to 600) | Exact (above) |
| Lyrics / instrumental | `lyrics` with `[verse]`, `[chorus]` tags, or `instrumental=True` with `lyrics="[Instrumental]"`; `vocal_language="en"` | Vocal case ran; content not verifiable without listening |
| Variations | `GenerationConfig.batch_size` 2 or 4 with a seed list | Batch 2 x 30 s 6.9 s, batch 4 x 30 s 8.5 s, batch 4 x 60 s 23 s. Batching is about 2x cheaper per track than separate runs for short tracks, no gain at 60 s |
| Turbo guidance | `guidance_scale` | Ignored: the library forces 1.0 for turbo (log: "overriding guidance_scale 7.0 -> 1.0") |
| Steps / shift | `inference_steps` (turbo 1 to 20, default 8), `shift` (docs: set 3.0 for turbo, default 1.0) | Not swept |

### 3.7 Other tasks on the MLX path

All three ran on MLX DiT (log: "DiT backend: MLX (native)") with the 2B turbo model:

- **Cover** (`task_type="cover"`, `src_audio`, `audio_cover_strength=0.5`): ran, 6.8 s; output correlation with the source 0.02 (it is a new rendition, not a copy).
- **Repaint** (`task_type="repaint"`, `repainting_start=20`, `repainting_end=35`): ran, 7.3 s; outside the window the audio is identical to the source (correlation 1.000 for 0 to 18 s and 37 to 60 s), inside it is new (0.05).
- **Reference audio** (`reference_audio=path` with text2music): ran, 3.6 s; log "Processing reference audio". Whether the style transfers needs listening.
- Lego, extract, complete and stems need the base model (docs); not tried.

## 4. Recommended defaults

Worker environment and calls (all outside the library, no patch):

1. `ACESTEP_MLX_VAE_CHUNK=256` (the library picks 1024 on a 64 GB Mac).
2. `mx.set_cache_limit(2 * 1024**3)` right after importing MLX.
3. After `initialize_service`, when `handler.mlx_decoder is not None`: `handler.model.decoder.to("meta")`, `gc.collect()`, `torch.mps.empty_cache()`. The torch copy of the DiT decoder is only used by lyric scoring and timestamps (`lyric_score.py`, `lyric_timestamp.py`) and by the silent PyTorch fallback (section 7). Saves about 4 to 5 GB.
4. `TOKENIZERS_PARALLELISM=false`, `ACESTEP_LM_BACKEND=mlx`, `HF_HUB_OFFLINE=1` after install, `ACESTEP_CHECKPOINTS_DIR=<music data dir>/checkpoints` (keeps weights out of the code folder).
5. Call `initialize_service(project_root=<code dir>, config_path="acestep-v15-turbo", device="auto")` (resolves to `mps`; `use_mlx_dit=True` is the default).

Generation defaults: `acestep-v15-turbo`, `inference_steps=8`, `shift=3.0`, `infer_method="ode"`, `thinking=False` and no LM loaded, `batch_size=1`, fixed seed shown in the UI, `enable_normalization=False` in the worker (keep the raw level; the mixer normalises) or keep the -1 dBFS peak for stand-alone tracks, take the returned tensor and write **FLAC 24-bit 48 kHz** ourselves (the library's own FLAC/WAV are 16-bit).

### 4.1 Minimum UI controls

Style prompt (caption, max 512 chars), lyrics (max 4096 chars) with an Instrumental switch, length (10 to 600 s), seed, variations (1, 2, 4). Optional, collapsed: BPM, key, vocal language, time signature. Later: "let the planner fill in details" (LM 0.6B; warn that it breaks seed reproducibility), reference audio, cover with a strength slider, repaint a time range. Do not expose guidance (ignored for turbo), steps, shift or sampler in the first version.

### 4.2 Run estimate strip (this Mac, recommended settings, warm worker)

- Time: about **1 s per 10 s of audio** up to 3 minutes (60 s -> 6 s, 180 s -> 18 s), about 1.4 s per 10 s at 5 to 10 minutes (600 s -> 82 s). Batch 4 at 60 s: 23 s. Add about 25 s when the worker is cold, plus 23 s if the planner LM must load.
- Memory: show **"about 22 GB while generating, 15 GB while idle"**; +1.3 GB with the planner LM 0.6B. Measured on the 2B turbo only.
- Disk: 10.1 GB for the main pack (it includes the 1.7B LM, 3.7 GB, whether used or not), +1.4 GB for LM 0.6B.

### 4.3 Alongside a loaded 20 GB model

Facts only (no other model was loaded): 64 GB total; with recommended settings the worker peaks at 20.3 to 22.9 GB and idles at 15.2 GB; other processes on this Mac used 11 to 25 GB during the spike. So 20 GB (other model) + 23 GB (music) + 11 to 25 GB (everything else) = 54 to 68 GB: it fits only when the rest of the system is light, and it will not fit at all with the library defaults (66 GB alone). The memory gate should budget 23 GB for a music generation and 15 GB for an idle music worker, and the queue should unload the worker when another kind needs the room.

## 5. XL 4B turbo

Downloaded (19.95 GB, 1,421 s) and run with the recommended settings (`config_path="acestep-v15-xl-turbo"`, MLX DiT active):

| Case | Wall | x RT | Diffusion / VAE | Peak footprint |
|---|---|---|---|---|
| Load | 11.5 s | | | **40.2 GB idle** (MLX arrays 17.4 GB) |
| 60 s orchestral | 12.4 s | 4.8 | 5.7 / 5.0 | 46.5 GB |
| 60 s song with vocals | 11.3 s | 5.3 | 5.9 / 4.6 | 46.6 GB |
| 30 s lo-fi | 5.4 s | 5.6 | 3.0 / 1.8 | 44.4 GB |
| Whole process (`time -l`) | | | | 47.0 GB |

Card claims: weights bf16 about 18.8 GB; "≥20 GB without offload (recommended)", "≥12 GB with CPU offload + INT8 quantization"; the repo's GPU table says "XL models use ~9GB VRAM for weights". On MPS the library loads float32, which is why it idles at 40 GB here.

Verdict: it runs, about 2x slower than the 2B (still 5x real time), but at **47 GB peak it cannot share a 64 GB Mac with anything else**. Offer it on 64 GB Macs only as an opt-in "large" model that unloads everything else first; do not offer it below 64 GB. Whether it sounds better needs the owner's ears (samples 13 to 15). Not tried: bf16 on MPS, offload, quantisation.

## 6. Integration: drive it from a JSON-lines worker using the Python API

Options looked at:

| Option | Verdict |
|---|---|
| REST server (`acestep-api`, port 8001) | Works on MLX (tried: `/release_task` then poll `/query_result`, 10.4 s for a 30 s track including lazy model load). But: no cancel endpoint for generation (only training has stop); polling only, progress is a log line string; it lazy-loads models on the first request even without `--no-init`; it pulls in training, dataset, LoRA, OpenAI-compatible chat and Gradio routes; it writes outputs under `<code>/.cache/acestep/tmp/api_audio`; another port to own. Same seed gave a different file than the Python API (different defaults). `/v1/audio?path=/etc/hosts` was refused (403), good |
| CLI (`cli.py`) | Wizard or TOML only, one generation per process: pays imports + load (about 12 to 30 s) per track. No |
| **Python API in our own worker** | **Recommended.** `AceStepHandler.initialize_service` + `generate_music(dit, llm, GenerationParams, GenerationConfig, save_dir, progress=cb)`. Gives the tensor directly, lets the worker apply the memory settings in section 4, a step hook and cooperative cancel, and matches `voice_worker.py` |

Worker shape: `backend_service/music_worker.py`, run by the music venv's Python, imports nothing from `backend_service`, swaps `sys.stdout` for stderr at start (needed: the LM path prints "Using precomputed LM hints" to stdout, which broke my own JSON lines), one JSON object per line each way. Unlike the voice worker, a generation runs in a thread so the stdin loop can still read `cancel` and `status`.

### 6.1 Progress

- The library's `progress(value, desc)` callback is mostly a **time estimate**: a thread ticks every 0.5 s from the previous run's per-step time (`progress.py`, `_start_diffusion_progress_estimator`); values seen: 0.51 "Preparing inputs", 0.52 "Generating music", 0.8 "Decoding audio", 0.99. 10 to 29 calls per track.
- Real step progress: the MLX loop is `for step_idx in tqdm(range(num_steps), desc="MLX DiT diffusion")` in `acestep/models/mlx/dit_generate.py`. Replacing the module's `tqdm` with a wrapper gave real per-step events (8 per track, 0.2 to 0.3 s apart at 60 s, 5 s apart at 600 s). This is a monkeypatch on a private name; pin the commit and cover it with a test.
- VAE decode (40 to 60 % of the time) has no per-chunk hook on the MLX path; report it as one stage with an estimate (VAE seconds are roughly 0.05 to 0.07 x audio seconds).

### 6.2 Cancel

- No cancel in the library.
- Cooperative: raising an exception from the step hook stops diffusion at the next step. The library catches it ("MLX diffusion failed (...); falling back to PyTorch"); with the torch decoder dropped (section 4) the fallback fails at once with "Tensor on device meta ...", `generate_music` returns `success=False`, the worker stays healthy, and the next generation is byte-identical to a clean run. The worker must map this to `cancelled` using its own flag, not the error text. Without the decoder drop, a cancel would start a slow float32 PyTorch diffusion instead.
- Hard: SIGTERM during diffusion exited in 0.12 s and system available memory went from 39.3 to 52.8 GB; no files were left in the spike folder, weights or cache (checked with `find -newer`). Next start costs about 25 s.
- Proposed: `cancel` sets the flag; if the job has not ended within 3 s (it is in VAE decode), the client kills the worker, as planned for "running-cancel by killing the worker".

### 6.3 Message protocol (proposal)

Requests (stdin), each with an `id`:

```json
{"id": 1, "cmd": "ping"}
{"id": 2, "cmd": "status"}
{"id": 3, "cmd": "load", "engine": "acestep", "model": "acestep-v15-turbo", "lm": null}
{"id": 4, "cmd": "generate", "job": "mus_7f3a", "engine": "acestep", "model": "acestep-v15-turbo",
 "task": "text2music", "caption": "indie pop, female vocal", "lyrics": "[verse]\n...", "instrumental": false,
 "vocal_language": "en", "seconds": 60, "bpm": null, "key": null, "time_signature": null,
 "seed": 11, "variations": 1, "steps": 8, "shift": 3.0, "planner": false,
 "src_audio": null, "reference_audio": null, "cover_strength": null, "repaint": null,
 "out_dir": "/abs/music/tmp/mus_7f3a", "format": "flac24", "normalize_peak_db": null}
{"id": 5, "cmd": "cancel", "job": "mus_7f3a"}
{"id": 6, "cmd": "unload"}
```

Replies (stdout): first line `{"id": null, "ok": true, "ready": true, "pid": 123, "engine_version": "ca1e85f"}`. Then for `generate`, zero or more events and one final reply, all carrying the request `id` and `job`:

```json
{"id": 4, "job": "mus_7f3a", "event": "stage", "stage": "load|encode|diffusion|decode|write", "estimate_s": 3.4}
{"id": 4, "job": "mus_7f3a", "event": "step", "step": 3, "total": 8, "elapsed_s": 1.1}
{"id": 4, "job": "mus_7f3a", "ok": true, "seconds": 6.2,
 "tracks": [{"path": "/abs/.../mus_7f3a_0.flac", "seed": 11, "sample_rate": 48000, "channels": 2,
             "duration_s": 60.0, "bpm": null, "key": null}],
 "timings": {"diffusion_s": 2.1, "vae_decode_s": 3.4}, "peak_footprint_gb": 21.6}
{"id": 4, "job": "mus_7f3a", "ok": false, "cancelled": true}
{"id": 4, "job": "mus_7f3a", "ok": false, "error": "ValueError: ..."}
```

`cancel` answers at once with `{"id": 5, "ok": true, "job": "mus_7f3a", "state": "cancelling"}`. Rules: one generation at a time (a second `generate` while busy gets `ok: false, error: "busy"`); paths must be absolute and under the music data dir (the backend checks before sending; the worker checks again); `seed` is always resolved before generation and returned; `status` returns `{loaded model, lm, busy job, footprint_gb}` from `proc_pid_rusage` (ctypes, no psutil needed). The client keeps the voice client's idle stop and timeout-kill; timeout for `generate` = 30 s + 0.3 x seconds x variations.

## 7. Failure modes and surprises seen

1. **Library defaults nearly exhaust 64 GB** (66 GB footprint for 60 s). Fixed by section 4; without it expect swapping and a 2 to 3x slower decode.
2. **Silent PyTorch fallback**: any exception in the MLX diffusion is logged as a warning and the run continues on PyTorch float32. A real MLX failure (memory) would look like a sudden slow run, not an error. The decoder drop turns it into an error, which is what we want.
3. **Planner LM not reproducible** and its metadata can be odd (bpm 300 with `thinking=False`).
4. **stdout pollution** from the LM path; the worker must redirect stdout.
5. **Remote code**: the DiT loads with `trust_remote_code=True` from the checkpoint folder, but the library first copies its own model code from `acestep/models/<variant>/` into the checkpoint (`model_downloader.py`, "Model Code File Sync"), so the code that runs is from the pinned commit. `silence_latent.pt` loads with `torch.load(weights_only=True)`. The install should still pin HF revisions.
6. **Downloader network probe**: `acestep-download` tests whether Google is reachable and otherwise downloads from ModelScope. Our installer should call `huggingface_hub.snapshot_download` with pinned revisions instead (as done here).
7. Importing `acestep.handler` also imports the trainer (log: "bitsandbytes not installed. Using standard AdamW"); 6 s warm imports, 24 s the very first time.
8. Turbo ignores guidance; `shift` default 1.0 is wrong for turbo (set 3.0).
9. The upstream repo ships a `.claude/` folder and an `AGENTS.md`; irrelevant to us, not read or followed.

## 8. Dependencies

- Repo `LICENSE`: MIT. Model cards (`Ace-Step1.5`, `acestep-5Hz-lm-0.6B`, `acestep-v15-xl-turbo`): YAML `license: mit`, not gated.
- Suggested UI line: "Music made with ACE-Step 1.5 (MIT). Check that a track is original before you publish it, and say it was made with AI."
- Dependencies on the Mac path: MIT, BSD or Apache except `pytorch-wavelets` 1.3.0, used only for the DWT (Haar) correction; the MLX path runs Haar natively. Dropped from the pack (D-37).
- Heavy optional deps pulled in by the lock but not needed for inference: gradio, lightning, tensorboard, peft, lycoris-lora, modelscope, torchao, matplotlib. They add install size; trimming them is possible only with our own requirements list (untested).

## 9. Install recipe (proposed `scripts/install-music.sh` steps)

Tested two ways: `uv sync --frozen` on Python 3.11 (43 s, 125 packages, 1.2 GB), and the pip-only block below on Python 3.12 (result at the end of this section). The block follows `voice_env.py` (venv from the backend's own Python, no uv needed). Disk for the whole spike: 35 GB, of which 29 GB weights (main 10.1, LM 0.6B 1.4, XL 20.0).

```bash
set -euo pipefail
MUSIC_DIR="${CHAOSENGINE_MUSIC_DIR:-$HOME/.chaosengine/music}"
ACE_REF=ca1e85fe9430179831e6bc6be790c332190a3866
PY="${CHAOSENGINE_MUSIC_PYTHON:-python3}"          # backend's own Python, 3.11 or 3.12
"$PY" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), sys.version'
mkdir -p "$MUSIC_DIR"
git clone --filter=blob:none https://github.com/ace-step/ACE-Step-1.5.git "$MUSIC_DIR/acestep-src"
git -C "$MUSIC_DIR/acestep-src" checkout --detach "$ACE_REF"
"$PY" -m venv "$MUSIC_DIR/venv"
"$MUSIC_DIR/venv/bin/python" -m pip install --upgrade pip
# requirements exported once from the pinned uv.lock and committed with the script:
#   uv export --frozen --no-hashes --no-dev --no-emit-project --python 3.12 | grep -v '^\./'
"$MUSIC_DIR/venv/bin/python" -m pip install -r scripts/music-requirements.txt
"$MUSIC_DIR/venv/bin/python" -m pip install --no-deps "$MUSIC_DIR/acestep-src"
"$MUSIC_DIR/venv/bin/python" -I - <<'EOF'
import os
from huggingface_hub import snapshot_download
ck = os.path.join(os.environ.get("CHAOSENGINE_MUSIC_DIR", os.path.expanduser("~/.chaosengine/music")), "checkpoints")
snapshot_download("ACE-Step/Ace-Step1.5", revision="19671f406d", local_dir=ck)            # 10.1 GB
snapshot_download("ACE-Step/acestep-5Hz-lm-0.6B", revision="148d8ea022",
                  local_dir=os.path.join(ck, "acestep-5Hz-lm-0.6B"))                       # 1.4 GB, optional
EOF
"$MUSIC_DIR/venv/bin/python" -c "import mlx.core, torch, acestep.handler; print(torch.backends.mps.is_available())"
```

**Result of the pip-only path (python.org-style `python3 -m venv` on Homebrew Python 3.12.5, no uv):** exit 0, 128 packages, 1.6 GB venv, 1,151 s (slow because the XL download shared the line; the uv install took 43 s). A 30 s generation from this venv was **byte-identical** to the 3.11 uv venv. No CUDA-only wheel, flash-attn or compiler was needed on macOS; torch comes from PyPI.

Use full 40-character HF revisions in the real script (short hashes were used here). The export has 601 lines with platform markers (Windows and Linux CUDA wheels come from the PyTorch index and need `--extra-index-url https://download.pytorch.org/whl/cu128` there; untested). `nano-vllm` (vendored, CUDA only) is excluded on macOS by its marker.

## 10. Risks

- Memory settings rely on a private detail (MLX cache growth in VAE decode) and an undocumented env var; a library update could undo them. Mitigation: pin the commit; an E2E check that asserts peak footprint under 26 GB for a 60 s track.
- The step hook and the decoder drop touch private names (`dit_generate.tqdm`, `handler.model.decoder`). Same mitigation.
- Quality unknown until the owner listens; vocal and lyric quality are the biggest unknown.
- The 10.1 GB main pack carries a 3.7 GB LM we do not use by default; downloading per subfolder (`allow_patterns`) would save it but was not tried.
- Upstream moves fast (last commit 2026-08-29, 21 commits after the last tag); expect churn when bumping.
- Windows and Linux (CUDA) paths were not tested.

## 11. Unknowns and things the owner must check by hand

1. Listen to `~/ChaosEngineAI-demos/music-spike/` (list below). Judge: vocals intelligible and on the lyrics (03, 03b), whether 70 bpm sounds like 70 (08), whether the reference audio steered the style (12), cover (10), the repaint seams at 20 s and 35 s (11), the 600 s track for repetition or drift (06).
2. `pytorch-wavelets` (dropped, section 8).
3. Whether to keep the library's -1 dBFS peak normalisation for stand-alone tracks.
4. XL 4B quality vs 2B (section 5).
5. LM 1.7B and 4B: not measured.
6. Steps above 8 or other shifts: not swept.
7. Behaviour on 16 / 32 GB Macs: not measured (the library has `MAX_MPS_VRAM` to simulate tiers; the footprint numbers here suggest 32 GB is fine with the recommended settings, 16 GB is doubtful).

Samples (all WAV 16-bit 48 kHz stereo, written by the library):

| File | What |
|---|---|
| 01_lofi_30s_instrumental_seed42 | case (a), "warm lo-fi hip hop beat, vinyl crackle, mellow piano" |
| 02_orchestral_60s_instrumental_90bpm_seed7 | case (b), "cinematic orchestral build, strings and brass, epic, 90 bpm", bpm 90 |
| 03 / 03b_indiepop_60s_female_vocal_lyrics_seed11 / seed12 | case (c), "indie pop, female vocal", lyrics below |
| 04 / 05 / 06_orchestral_120s / 180s / 600s | long instrumentals |
| 07_lofi_30s_with_planner_LM_0.6B_seed42 | case (d) with the LM |
| 08 / 08b / 08c | 30 s lo-fi seed 42: bpm 70 A minor, bpm 120 A minor, bpm 70 F# major |
| 09_lofi_30s_batch4_variation1..4 | batch of 4, seeds 1 to 4 |
| 10_cover_of_02_as_lofi_strength0.5 | cover |
| 11_repaint_of_02_20s_to_35s | repaint |
| 12_lofi_30s_with_reference_audio_02 | reference audio = file 02 |
| 13 / 14 / 15_XL4B_... | XL 4B turbo: same prompts and seeds as 02, 03 and 01 |

Lyrics used for case (c) (written for this spike):

```
[verse]
Morning light on an empty street
Coffee cold and the radio sweet
I keep your letters in a paper box
Counting days on the kitchen clock

[chorus]
Oh we were young and the summer was gold
Every secret that we never told

[verse]
Now the city hums a different tune
I still look for you under the moon

[chorus]
Oh we were young and the summer was gold
Every secret that we never told
```

Raw logs and the harness (`bench.py`, `analyze.py`, case files): `~/.chaosengine/spikes/acestep/harness/` and `~/.chaosengine/spikes/acestep/runs/`.
