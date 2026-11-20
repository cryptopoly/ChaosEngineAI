# Singing lip sync: what improves it, measured (2026-11-20)

Follow-up to [lip-sync-spike-windows-2026-11.md](lip-sync-spike-windows-2026-11.md) and D-103 (owner: singing sync
must work fully for v0.10.0). Machine: Windows 11, RTX 4090 24 GB, 64 GB RAM. Engine: the app's phase-1 lip-sync
worker (`backend_service/lipsync_worker.py`, LTX-2.3 distilled 4-bit, audio held at sigma 0), driven by a warm
harness that opens its knobs. **I cannot listen to audio**: every statement about sync comes from the automatic
score below and from frame strips I looked at; what is sung comes from speech-to-text on the vocal stem.

_Status: in progress; sections are added as each experiment finishes._

## The automatic sync score (FU-334, `backend_service/lip_sync_score.py`)

Per frame: mouth opening = inner-lip gap (MediaPipe Face Landmarker landmarks 13/14, Apache-2.0) over face height
(10/152), 3-frame median; head turn = nose-tip offset from the eye-corner midpoint over the eye distance (about 0
frontal, 0.7-1.2 in profile; above 0.35 counts as turned away). Voice = band-passed (100-4000 Hz) RMS of the
**vocal stem of the shot's window** per video frame, whatever audio the model was given, so every variant is judged
against the same truth. Reported:

| Field | Meaning |
|---|---|
| `r` | Pearson r of mouth vs voice, best lag in +-3 frames |
| `chance_p95` | 95th percentile of the same best-lag r over 200 circular shifts of the voice; `beats_chance` = r above it |
| `closed_while_voiced` (cwv) | share of voiced frames with the mouth shut: the singing failure |
| `open_while_silent` (ows) | share of silent frames with the mouth open (gaps between words count, so even good speech gets 0.3-0.5) |
| `balanced` | mean of (1 - cwv) and (1 - ows); 0.5 = ignores the audio |
| `motion` | 0-1, how far the mouth really opens (its 95th percentile over 0.04); r is scaled by it, because a shut mouth's landmark jitter can still correlate with the audio |
| `turned_away` | share of frames turned away or with no face |
| `score` | `(0.5 * balanced + 0.5 * max(0, r) * motion) * (1 - turned_away)`: ranks takes **of one shot**; not a grade across songs |

Checks on the spike's clips: speech with the audio hold r 0.50 (chance 0.17), score 0.63; the same speech without
the hold (mouth open through the silent tail) 0.50; Lightricks' route (a) speech 0.59. Unit tests:
`tests/test_lip_sync_score.py` (13, synthetic series, no MediaPipe needed). The maths needs only numpy and scipy;
reading a video needs `cv2` + `mediapipe`, which the app venv does not have (the experiments ran them from a tools
venv). The `motion` and `turned_away` terms were added after looking at the first strips: a closed-mouth take scored
0.39 on r alone, and a take where the singer turns to profile scored as well as a frontal one.

## Inputs

- **Three sung lines**, all made through the app's own `POST /api/music/generate` on a throwaway backend (port 8954,
  `HOME` in `E:\tmp\lipsync\home`), **ACE-Step 1.5 XL turbo**, 30 s, `vocalLanguage=en`: `ballad_f` (slow female
  ballad, 76 bpm, seed 11), `soul_m` (male soul, 96 bpm, seed 23), `pop_f` (upbeat female pop, 118 bpm, seed 37).
  Whisper Large v3 Turbo (`/api/voice/transcribe`) on the Demucs stem reads back the lyrics in all three (one word
  changed in two: "a river" for "the river", "for the sunshine" for "pour the sunshine").
- **Vocal stem**: Demucs `htdemucs` (MIT). **Windows** cut from the stem's activity map so each shot starts 0.4 s
  before a phrase: ballad 7.2 s ("We were young and we were stronger", then 0.7 s silence), soul 10.8 s ("Oh baby
  come along with me", one long phrase), pop 3.7 s (a phrase, a 0.6 s gap, the next phrase).
- **Stills**: FLUX.2 Klein 4B via `/api/images/generate`, 768x512, seed 78: female and male head-and-shoulders
  ("close") and waist-up ("medium"), frontal, mouth closed.
- **Default recipe under test** (the app's today): 5.04 s (121 frames, 24 fps), 768x512, distilled 8 steps, no CFG,
  STG 1, modality guidance 3, guidance rescale 0.7 (the diffusers default, which the worker does not override), audio
  at sigma 0, prompt "A {young woman | man} sings in a music studio, natural head movement, soft warm light, static
  camera.", seeds 42 and 7. One warm process: 55-57 s per shot (8 steps at 5.7-6.0 s plus decode), load 35 s once.

## Experiment 2: what audio the model hears

Each line x 2 seeds x 5 versions of the same window: the Demucs stem as-is (`stem`, -17 to -20 dBFS RMS), the full
mix (`mix`, -12 to -17), stem + 30 % instrumental (`stem30`), the stem normalised to -16 dBFS RMS (`stemn`), and the
stem + 50 % instrumental normalised to -16 dBFS (`boost`). Scored against the stem.

| Audio | mean score | mean r | beats chance | Clips where the mouth shut through the line |
|---|---|---|---|---|
| **stem** | **0.44** | **0.35** | **4 / 6** | 1 (ballad seed 42) |
| stemn | 0.42 | 0.33 | 4 / 6 | 1 (same) |
| stem30 | 0.37 | 0.28 | 3 / 6 | 1 (same) |
| boost | 0.37 | 0.24 | 2 / 6 | 1 (same) |
| mix | 0.37 | 0.22 | 2 / 6 | 1 (same) |

**The stem is the best input; keep it as the default.** Every bit of instrumental added lowered r. Loudness is not
the reason: the normalised stem is no better than the plain one. (The spike's "mix beat stem" came from scoring each
clip against its own audio; scored against the same vocal truth the mix loses.)

**The seed matters far more than the audio.** For a given line and seed the five audio versions give the same
performance: ballad seed 42 never opens her mouth in any of the five (reads as listening, cwv 1.00); soul seed 7
turns to profile half-way in all five; ballad seed 7 sings the line well (stem r 0.63) and then also turns to profile.
The start noise decides *whether* the face performs and where it looks; the audio decides *when* the mouth moves once
it does. That points at levers that act on the performance (prompt, framing, more takes) rather than on the audio.

Frame strips looked at: all 30. Other things the strips show and the score half-catches: a grin with the mouth open
after the phrase ends (soul seed 42; counts as open while silent), a hand coming up to the face (pop seed 7), light
streaks in the male close-up background.

## Experiment 4: prompt wording and framing

Stem audio, same windows and seeds. "Explicit" prompt: "Close-up of a {young woman | man} singing, mouth opening and
closing with each syllable, expressive face, in a music studio, soft warm light, static camera." "Medium" = the
waist-up stills instead of head-and-shoulders.

| Variant | mean score | mean r | beats chance | Never sang (cwv > 0.9) | Turned away > 30 % |
|---|---|---|---|---|---|
| close, neutral (today) | 0.44 | 0.35 | 4 / 6 | 1 | 2 |
| **close, explicit** | **0.49** | **0.44** | **5 / 6** | **0** | 1 |
| medium, neutral | 0.40 | 0.30 | 2 / 6 | 2 | 1 |
| medium, explicit | 0.45 | 0.35 | 3 / 6 | 0 | **0** |

**The explicit singing prompt is the first real lever**: it removed the "listens instead of singing" failure (ballad
seed 42 went from a shut mouth through the whole line, cwv 1.00, to singing it, cwv 0.07) and raised r. Its cost is
the opposite error: the face now often keeps its mouth open or grins after the line ends (open while silent 0.7-0.9
on several takes), which costs r. **Medium framing** kept every singer facing the camera (0 of 6 turned away with the
explicit prompt) but follows the vocal less closely (mouth smaller in the frame, r 0.35 vs 0.44); with the neutral
prompt it was worse than the close-up (two of six never sang). Recommendation so far: head-and-shoulders still +
explicit singing wording; medium framing only when the shot needs it, with the explicit wording.

Strips: all 18 looked at. The fixed ballad take sings the line with visible open-close on the syllables, then smiles
with the mouth open in the 0.7 s tail. Soul seed 7 (explicit) has the highest r (0.66) but looks down and away and its
last frame smears the face; `turned_away` (yaw only) caught 22 %, the downward look is not measured (no pitch check).

## Experiment 1: shot length and where the shot starts

Explicit prompt (experiment 4's winner), stem, close-up, seeds 42 and 7. Each shot starts 0.4 s before a phrase
(`lead`), except `5s_mid`, which starts 1.5 s *into* the same phrase (no lead-in). The windows hold different audio,
so compare the pattern, not single numbers.

| Shot | Frames | mean score | mean r | beats chance | mean open while silent | Turned away > 30 % | s per shot |
|---|---|---|---|---|---|---|---|
| **3 s, lead-in** | 73 | **0.52** | **0.41** | **5 / 6** | 0.58 | 0 | 37 |
| 5 s, lead-in | 121 | 0.49 | 0.44 | 5 / 6 | 0.63 | 1 | 56 |
| 8 s, lead-in | 193 | 0.45 | 0.34 | 2 / 6 | 0.76 | 0 | 88 |
| 5 s, starts mid-phrase | 121 | 0.37 | 0.30 | 2 / 6 | 0.87 | 1 | 56 |

**Start the shot just before a phrase.** Starting mid-phrase is the worst case: the start still has a shut mouth and
the face needs about a second to start singing, so the opening words are sung with a closed mouth (strip: frames 0-21
shut while the vocal is loud), and the takes that recover then mouth on through the gaps. A 0.4 s lead-in avoids it.
**Shorter is better, 8 s is worse**: in an 8 s shot with a 1-2 s gap between phrases the face keeps singing (or
grimaces with an open mouth) through the gap (open while silent 0.92 on both soul takes), so r falls. 3-5 s shots cut
to one phrase plus its lead-in are the sweet spot; this matches the phase-1 rule of 3 to 8 s but argues for cutting
at phrase boundaries and keeping sung shots at 3-5 s.

Strips: all 18 looked at (the 8 s ones show the open-mouth grimace through the gap; the mid-phrase ones the late
start; hands come up to the face in one mid-phrase take).

## Experiment 6: LipDub re-sync second pass (not run: gated)

`Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (model card title "LTX-2.3 22B IC-LoRA Dub-It", last change 2026-10-22):
one file, `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors`, 2.47 GB (the card still names it `...-lipdub-0.9`), an IC-LoRA
on LTX-2.3 22B, video and audio control, reference at output resolution; licence LTX-2 Community (same as LTX-2.3:
free below US$10M yearly revenue). Pipeline: `ltx_pipelines.dubit` (Lightricks' LTX-2 repo), which is already
installed in the route (a) venv from the spike (`E:\tmp\lipsync\venv-ltx`, ltx-pipelines 1.6.2), so the code side is
ready. **Blocked by the gate**: the repo is gated (`auto`, "Agree and Access"), and the click-through also asks the
account holder to consent to marketing offers. The saved Hugging Face token gets HTTP 403 on the file. Accepting is
the owner's call (I do not accept terms on the account's behalf). Once accepted, the test is one command per clip
(the take from the main recipe as the reference video, the stem window as the audio) on the existing venv, about
4-5 minutes per 5 s shot by analogy with route (a) (same 22B fp8 checkpoint and streamed offload). FU row added.

## Experiment 3: denoise settings

Explicit prompt, stem, 5 s lead-in windows, seeds 42 and 7; each knob changed alone from the default (8 distilled
steps, STG 1, modality guidance 3, guidance rescale 0.7, audio at sigma 0). Paired against the same song and seed
with the default; "better in" counts the pairs whose score rose.

| Knob | mean score (default 0.485) | paired change | better in | mean r | s per shot |
|---|---|---|---|---|---|
| audio sigma 0.1 x the video's (instead of 0) | **0.516** | **+0.03** | 4 / 6 | 0.46 | 56 |
| audio sigma 0.3 x | 0.488 | +0.00 | 4 / 6 | 0.43 | 56 |
| modality 3, guidance rescale 0 | 0.465 | -0.02 | 3 / 6 | 0.41 | 56 |
| STG 0 (two passes a step) | 0.462 | -0.02 | 4 / 6 | 0.39 | **40** |
| modality 5, rescale 0 | 0.454 | -0.03 | 2 / 6 | 0.38 | 56 |
| modality 5 | 0.435 | -0.05 | 3 / 6 | 0.42 | 56 |
| audio-to-video attention x 1.5 | 0.423 | -0.06 | 1 / 6 | 0.34 | 56 |
| modality 1 (off) | 0.423 | -0.06 | 3 / 6 | 0.31 | 42 |
| 12 steps (distilled schedule stretched) | 0.398 | -0.09 | 2 / 6 | 0.31 | 78 |
| modality 8 | 0.387 | -0.10 | 1 / 6 | 0.36 | 56 |
| 16 steps | 0.387 | -0.10 | 1 / 6 | 0.29 | 101 |
| STG 2 | 0.383 | -0.10 | 1 / 6 | 0.29 | 56 |

**No denoise knob fixes singing.** Every change is inside the take-to-take noise except three that clearly hurt:
more steps (the distilled model is trained on its 8 sigmas; stretching the schedule to 12 or 16 steps costs 40-80 %
more time and lowers r), STG 2, and modality 8. Stronger audio guidance (modality 5 or 8, with or without the 0.7
rescale that diffusers applies by default, or a 1.5x gain on the audio-to-video attention) does not make the mouth
follow the vocal more closely; it mostly makes the face mouth through the gaps (open while silent up to 0.80). The
one small positive, telling the model the audio is at 10 % of the video's noise level instead of clean, is +0.03 on
6 pairs: too small and too uncertain to change the shipped default without more takes. The cheap end is worth
knowing: STG 0 or modality 1 drop a transformer pass (40-42 s instead of 56 s) at a small cost.

The two-stage, non-distilled path (Lightricks' route (a): dev 22B fp8, CFG, 30 + 3 steps) is below.
