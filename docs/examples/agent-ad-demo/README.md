# Agent demo: a 30 second burger advert and 10 posters, made through the local API

An AI coding agent (Claude Code) produced a launch advert and a poster set for a fictional-style local burger brand,
**The Butcher's Best** ("Made, not assembled."), using only the HTTP API of a running ChaosEngineAI backend. No app
source was needed to *call* it; the source was read only to learn request fields. The finished media lives outside
this repo (about 30 MB); this folder keeps the method, prompts and helper so it can be repeated.

## What was used

| Job | Model / route | Cost on an M4 Max 64 GB |
|---|---|---|
| Stills for posters | FLUX 3 Image Klein (mflux), `POST /api/images/generate`, 1024x1024, 4 steps | about 20 s each |
| Video clips | LTX-2.5 distilled q4, `POST /api/video/generate`, 768x512, 24 fps, with sound | 3 s clip about 94 s; 4 s clip about 2 min |
| Voiceover | Kokoro TTS, `POST /api/voice/synthesize`, voice `bm_george` (British male) | a few seconds a line |
| Edit, captions, posters | ffmpeg and headless Chrome (outside the app) | light |

Voices with a British accent: `bm_george`, `bm_fable`, `bm_lewis`, `bm_daniel`, `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`.

## Lessons

- **Run one job at a time.** The first attempt queued 6 images, 5 video clips and Chrome renders in one batch and the Mac
  crashed (cause not confirmed; the LTX step is the main suspect). Sequential runs with a free-memory check between clips
  (stop under 40 percent free) were stable: free memory stayed about 92 to 93 percent.
- **Headless backend, no window.** With the UI closed, start it as a long-lived process (a plain background shell that
  exits will shut it down):
  `PYTHONPATH=<runtime>/python/site-packages <runtime>/python/home/bin/python3 -m backend_service.app` from the repo root,
  where `<runtime>` is the `chaosengine-embedded-runtime/...` folder the app extracts under the temp directory.
  The default port is 8876 (`CHAOSENGINE_PORT`). Bearer auth defaults to on; set `CHAOSENGINE_API_TOKEN` and send it,
  or `CHAOSENGINE_REQUIRE_AUTH=0` for local scripts. (On the author's Mac the calls worked with no token; why was not checked.)
- **Response shapes that tripped the first script:** images return `{"artifacts": [...]}` with a base64 `previewUrl`;
  video returns `{"artifact": {...}}` and the file is fetched from `/api/video/outputs/{artifactId}/file`.
- **Prompts that gave the wrong subject.** A "biting a burger" clip rendered a crumbly pastry; naming every layer and
  "the burger stays fully visible in frame" is the fix tried second.
- **AI visuals are a poor fit for a "100% real, all homemade" brand.** Use them for concepts and storyboards; shoot the real
  burger for launch material.

## Repeat it

```bash
python3 chaos_api.py image hero.png black-forest-labs/flux-3-image-klein-mflux 1024 1024 11 "Professional food photography, a hand-built cheeseburger on a butcher's block, warm morning light"
python3 chaos_api.py video clip.mp4 dgrauet/ltx-2.5-mlx-q4 97 22 "Cinematic close-up: a baker pulls golden sesame buns from an oven, steam rising"
python3 chaos_api.py voice line.wav bm_george "Six a.m. Same butcher. Same hands."
```

Then cut with ffmpeg (slow clips 1.15x, caption PNGs overlaid, VO delayed with `adelay`, 8 s end card with `zoompan`).
ffmpeg here has no `drawtext`, so captions were rendered to transparent PNGs with headless Chrome.

Files here: `chaos_api.py` (image, video and voice calls), `build_ads.sh` (the ffmpeg edit for 16:9 and 9:16, paths are
relative to the author's working folder), `make_posters.py` (poster HTML; render each to PNG with headless Chrome).
The first cut used the macOS `say` voice; the app's own Kokoro voice replaced it, which is the better route.

## Next step this points at

A thin MCP server or skill over `/api/images`, `/api/video` and `/api/voice`, so any user's agent gets typed tools
and the memory rules above built in. See `docs/hub/ideas.md` and FU-126 in `CLAUDE.md`.
