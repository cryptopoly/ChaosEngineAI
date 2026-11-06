# Lighthouse keeper: a 32 second, eight-shot film through the agent bridge

Made 2026-11-06 on an M-series Mac with 64 GB, using only the `/mcp` tools (no paid tokens, no chat model).
`make_demo.py` is the whole recipe.

| Step | Tool | Model |
|---|---|---|
| Keeper portrait | `generate_image` | FLUX 3 Klein (mflux) |
| One still per shot, with the lighthouse and the keeper as `reference_images` | `generate_image` | FLUX 3 Klein edit |
| One clip per still, `start_image` = that still, with sound | `generate_video` | LTX-2.5 distilled q4 |
| Voiceover | `speak` | Kokoro |
| Join, mix the voiceover under the clip sound | ffmpeg | |

Eight shots: establishing, morning with a mug, painting in the sun, weeding in rain, storm shutters, polishing the lamp, sweeping snow at night, dawn on the rocks.

Result: 768x448, 24 fps, 32.4 s, with sound. About 25 minutes end to end (stills about 85 s each, clips about 125 s each).
The video is not in the repo (about 9 MB); run the script to remake it:

```bash
python3 docs/examples/lighthouse-keeper-demo/make_demo.py LIGHTHOUSE_IMAGE OUT_DIR
```

Turn on Settings > Integrations > Agent access first. The script is resumable.
