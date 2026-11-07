# Sound effects: first real run (Stable Audio Open Small)

Date 2026-11-07. Machine: Apple M4 Max, 64 GB. Branch `feat/sound-real-test`, backend on port 8908 from the repo venv.
Model: `stabilityai/stable-audio-open-small` at the pinned revision `ea2772170fc0`, plus `google-t5/t5-base`.
The Hugging Face token already saved in the app was used only through the app's own download path; it was not printed or copied.

## What was installed (through the app's own routes)

- Sounds pack (`POST /api/music/sound/install`): about 53 s, Python 3.11.2, torch 2.7.1, transformers 5.22.2, stable-audio-tools 0.0.20, MPS available. Verify step passed first time.
- Weights (`POST /api/music/download`): accepted with no 403, 1.56 GB model files + 0.86 GB T5 = 2.4 GB in the Hub cache (catalog says 2.57 GB, fine). Slow part (about 10 minutes, network bound, partly overlapping the pip install).
- The runtime reports `ready: true` for the row without any manual mirroring; the T5 encoder is mirrored on first use.

## Clips (warm worker unless stated, 8 steps, pingpong sampler, seed 7, default peak -1 dBFS)

| Prompt | Asked | Got | Job time | Peak footprint (worker) | File | Format |
|---|---|---|---|---|---|---|
| door creaking | 4 s | 4.00 s | 3.31 s (first call of a cold worker; includes load) | 4.99 GB | 536 KB | FLAC PCM_24, 44.1 kHz, 2 ch |
| rain on a tin roof, loop | 10 s | 9.92 s | 0.74 s | 5.01 GB | 1.83 MB | same |
| footsteps on gravel | 6 s | 6.00 s | 0.73 s | 5.02 GB | 1.10 MB | same |
| glass bottle rolling on a wooden floor, via bridge `make_sound` | 3 s | 3.00 s | 1.13 s (worker started inside the call; whole job incl. spawn + load about 5 s) | n/a | 419 KB | same |

- Whole first HTTP call (spawn worker, import torch, load 1.7 GB + T5, generate): 8.2 s.
- Explicit model load on an unloaded backend (`/api/music/preload`): 3.7 s, footprint 3.77 GB.
- Loop clip is 0.08 s shorter than asked by design (the tail is cross-faded into the head and dropped).
- Peak memory is the footprint the worker reports through the app (`peakFootprintGb`). I did not get an independent RSS sample (my `ps` sampler matched no process name), so treat it as the app's own number.

## Objective checks (numbers only; I cannot listen)

Measured with numpy/soundfile on the FLAC files.

| Clip | RMS | Peak | Clipped samples | Notes |
|---|---|---|---|---|
| door creaking | -23.2 dBFS | -1.0 dBFS | 0 | L/R correlation 1.0 (effectively mono). Final ~0.5 s is near silent (-37 then -90 dB per half second). |
| rain loop | -20.9 dBFS | -1.0 dBFS | 0 | Level steady within 1 dB across the whole clip (-20 to -22 per half second), as rain should be. Loop seam step (last sample to first) 0.027, small against a 0.89 peak. L/R corr 0.999. |
| footsteps on gravel | -24.6 dBFS | -1.0 dBFS | 0 | Final ~0.5 s near silent (-78 dB). L/R corr -0.58 (wide, partly out of phase stereo). |
| bottle rolling | -30.2 dBFS | -1.0 dBFS | 0 | Final half second near silent. L/R corr 0.84. |

Verdict: none are silent, none clip, the peak normaliser lands on -1.0 dBFS, durations and sample rate are right. Whether they sound like a door, rain or gravel is unverified: nobody listened. Two things worth a human ear: the trailing near-silence on the non-loop clips (the model seems to fade out before the end of the window; consider trimming or telling users), and the out-of-phase stereo on footsteps (check mono-compatibility).

## Licence and attribution

- `GET /api/music/catalog` returns the licence block for the row: "Stability AI Community Licence", gated, commercial "limited", revenue cap US$1,000,000, attribution "Powered by Stability AI", five plain-language summary lines, a note that weights are never bundled, and the LICENSE URL pinned to the revision.
- Every library item carries `"licence": "Stability AI Community Licence; Powered by Stability AI"`; the FLAC `COMMENT` tag holds the same string plus prompt, seed, model, engine version, `loop` and `machine_generated: true`; `ENCODER` tag names the model.
- The bridge tool `make_sound` returns a note "keep 'Powered by Stability AI' where you publish with it", the licence string, and a download link with size and sha256.
- Not checked: the licence screen in the desktop UI (API text only).

## Bridge tool

`make_sound` was called through `/mcp` with `tools/call`, then `wait_for_job` (60 s max per wait). To avoid turning on the owner's real Agent access switch or touching the token, the call ran in a Python test client with the access check patched on; the tool handler, job queue, worker and output are the real ones. A live external client with a real token is not tested.

## Defects found and fixed

1. Download progress for the sound (and any music) row used the whole repo size (6.0 GB) instead of the 1.68 GB of files actually fetched, so the bar showed about 26 % when the download was complete. Fixed in `backend_service/state/downloads.py`: size the bar from the row's `downloadPatterns`, as is done for pinned text GGUFs. Checked against the Hub file list: 1.68 GB.
2. `tests/test_music_sound.py::test_text_encoder_is_mirrored_from_the_cache` read the real Hub cache for its "not ready" assertion, so it failed as soon as T5 was downloaded. Test now points the cache at an empty temp folder first.
3. Catalog estimates for the sound row were guesses (peak 6 GB, cold start 20 s, 5 s per 10 s of audio). Replaced with the measured values above (basis "measured"); the two test assertions on `basis` updated.

## Open points, not fixed

- Trailing silence on non-loop clips (above).
- The first call after a cold worker spends about 3 s in the job on MPS warm-up besides the load; the estimate's cold start of 8 s covers spawn, load and that.
- One bridge job (`job-1`) from my first, argument-error attempt stays `running` in the in-process job table only; it died with that test process. Not a product issue.

## Files

- Only the bottle clip (419 KB) is saved in `docs/research/media/sound-samples/`: the repo's pre-commit hook rejects files over 500 KB.
- The others stay in the Music Library: door creaking (536 KB) `/Users/dan/.chaosengine/music/library/1794062989_fe35e693.flac`, rain loop (1.83 MB) `/Users/dan/.chaosengine/music/library/1794062990_7a6af2b3.flac`, footsteps (1.10 MB) `/Users/dan/.chaosengine/music/library/1794062991_dfa24268.flac`.
