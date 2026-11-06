#!/usr/bin/env python3
"""30 second, eight-shot short film made only through the agent bridge (/mcp tools).

Keeps one character and one place consistent across shots:
  1. a keeper portrait (text to image),
  2. one still per shot from FLUX 3 Klein with BOTH the lighthouse and the keeper as reference images,
  3. one clip per still (LTX-2.5, start_image = that still, with sound),
  4. a Kokoro voiceover,
  5. ffmpeg joins the clips (hard cuts) and mixes the voiceover under the clip sound.

Usage:  make_demo.py LIGHTHOUSE_IMAGE OUT_DIR   (backend on 127.0.0.1:8876, Agent access on)
Resumable: a step whose output file already exists is skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8876"
IMAGE_MODEL = "black-forest-labs/flux-3-image-klein-mflux"
VIDEO_MODEL = "dgrauet/ltx-2.5-mlx-q4"
SIZE = (1024, 576)        # stills, 16:9
CLIP = (768, 448)         # clips, multiples of 64
FRAMES = 97               # 4.04 s at 24 fps; 8 shots is about 32 s

KEEPER = (
    "Portrait of a weathered lighthouse keeper in his sixties, white beard, kind tired eyes, bright yellow "
    "oilskin raincoat and a dark wool cap, standing outdoors, soft overcast light, photorealistic"
)
LOOK = "cinematic photograph, natural colours, 35mm, shallow haze"

SHOTS = [
    ("establishing", "Wide shot of the red and white striped lighthouse on its rocky headland at golden dusk, calm sea, no people.",
     "Slow push-in on the lighthouse at dusk. Gentle waves on the rocks, seagulls, soft wind."),
    ("morning", "The keeper in his yellow raincoat steps out of the lighthouse door with a steaming tin mug, early morning light.",
     "He steps out, takes a breath of sea air and looks at the sky. Door creak, footsteps on stone, waves, gulls."),
    ("painting", "The keeper in the yellow raincoat, sleeves rolled up, stands on a wooden ladder painting the lighthouse tower white in bright summer sun, blue sky.",
     "He brushes long strokes of white paint down the tower. Brush strokes, bright wind, distant waves."),
    ("weeding", "The keeper in the yellow raincoat crouches at the foot of the lighthouse pulling weeds from the grass in heavy rain.",
     "He pulls a weed and drops it in a bucket, rain falling hard. Rain on grass, wind, a distant rumble."),
    ("storm", "The keeper in the yellow raincoat fights the wind to close a wooden storm shutter on the lighthouse in a violent storm, dark sky, spray.",
     "He slams the shutter and bolts it as wind whips his coat. Howling gale, crashing waves, thunder."),
    ("lamp", "Inside the lighthouse lantern room the keeper in his yellow raincoat polishes the great brass lens, warm lamplight at dusk.",
     "He wipes the lens in slow circles and the light flares. Soft cloth squeak, a low mechanical hum, wind outside."),
    ("snow", "The keeper in the yellow raincoat sweeps fresh snow off the lighthouse steps at night, blue moonlight, lantern in hand, the beam crossing the snowfall.",
     "He sweeps the snow aside, the beam passes overhead. Quiet snowfall, scraping broom, a low foghorn."),
    ("dawn", "The keeper in the yellow raincoat sits on a rock beside the lighthouse with a tin mug, looking at a calm sea at sunrise, warm gold light.",
     "He sips from the mug and watches the sun rise, the lighthouse beam fading. Calm waves, a gull, a long exhale."),
]

NARRATION = (
    "Forty years on the rock. Paint in the sun. Weeds in the rain. Shutters in the storm. "
    "And every night, the same promise: the light stays on."
)


def _token() -> str:
    """The Agent access token: CHAOSENGINE_AGENT_TOKEN, else the app's agent_access.json."""
    if os.environ.get("CHAOSENGINE_AGENT_TOKEN"):
        return os.environ["CHAOSENGINE_AGENT_TOKEN"]
    return json.loads((Path.home() / ".chaosengine" / "agent_access.json").read_text())["token"]


def rpc(method: str, params: dict | None = None, timeout: float = 120.0) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params else {})}).encode()
    request = urllib.request.Request(BASE + "/mcp", body, {"content-type": "application/json", "authorization": f"Bearer {_token()}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["result"]


def tool(name: str, arguments: dict) -> dict:
    result = rpc("tools/call", {"name": name, "arguments": arguments})
    text = result["content"][0]["text"]
    if result.get("isError"):
        raise RuntimeError(f"{name}: {text}")
    return json.loads(text)


def run_job(name: str, arguments: dict, label: str) -> dict:
    job = tool(name, arguments)
    started = time.time()
    while True:
        state = tool("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 30})
        if state["state"] in ("done", "failed", "cancelled"):
            break
    if state["state"] != "done":
        raise RuntimeError(f"{label} {state['state']}: {state.get('error')}")
    print(f"[{time.time() - started:6.0f}s] {label}: {state['output'].get('path')}", flush=True)
    for note in state.get("notes") or []:
        print(f"           note: {note}", flush=True)
    return state["output"]


def keep(out: Path, step: str, make) -> Path:
    marker = out / f"{step}.json"
    if marker.exists():
        return Path(json.loads(marker.read_text())["path"])
    path = make()
    marker.write_text(json.dumps({"path": str(path)}))
    return Path(path)


def main() -> None:
    lighthouse = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    keeper = keep(out, "keeper", lambda: run_job("generate_image", {
        "prompt": KEEPER, "model": IMAGE_MODEL, "width": 1024, "height": 576, "seed": 7}, "keeper portrait")["path"])

    clips: list[Path] = []
    for index, (slug, still_prompt, motion) in enumerate(SHOTS, start=1):
        tag = f"{index:02d}-{slug}"
        still = keep(out, f"{tag}-still", lambda: run_job("generate_image", {
            "prompt": f"{still_prompt} The man is the keeper from the second reference image and the tower is the lighthouse "
                      f"from the first reference image. {LOOK}",
            "model": IMAGE_MODEL, "reference_images": [str(lighthouse), str(keeper)],
            "width": SIZE[0], "height": SIZE[1], "seed": 100 + index}, f"{tag} still")["path"])
        clip = keep(out, f"{tag}-clip", lambda: run_job("generate_video", {
            "prompt": motion, "model": VIDEO_MODEL, "start_image": str(still),
            "frames": FRAMES, "fps": 24, "width": CLIP[0], "height": CLIP[1], "seed": 200 + index}, f"{tag} clip")["path"])
        clips.append(clip)

    voice = keep(out, "voice", lambda: run_job("speak", {"text": NARRATION, "voice": "am_michael"}, "voiceover")["path"])

    listing = out / "clips.txt"
    listing.write_text("".join(f"file '{c}'\n" for c in clips))
    joined = out / "joined.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(joined)], check=True)
    final = out / "lighthouse-keeper.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-i", str(voice),
                    "-filter_complex", "[0:a]volume=0.55[bed];[1:a]adelay=1500|1500,volume=1.4[vo];[bed][vo]amix=inputs=2:duration=first:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(final)], check=True)
    print("DONE", final, flush=True)


if __name__ == "__main__":
    main()
