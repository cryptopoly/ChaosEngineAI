"""Tiny helpers an agent (or a person) can use to drive the local ChaosEngineAI API.

Usage:
  python3 chaos_api.py image  out.png  <modelId> <w> <h> <seed> "<prompt>"
  python3 chaos_api.py video  out.mp4  <modelId> <frames> <seed> "<prompt>"
  python3 chaos_api.py voice  out.wav  <voiceId> "<text>"

Env: CHAOSENGINE_URL (default http://127.0.0.1:8876), CHAOSENGINE_API_TOKEN (sent as a Bearer token when set).
Run one job at a time: each model load is large (LTX-2.5 q4 is about 17 GB resident).
"""
import base64, json, os, sys, urllib.error, urllib.request

BASE = os.environ.get("CHAOSENGINE_URL", "http://127.0.0.1:8876")
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("CHAOSENGINE_API_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["CHAOSENGINE_API_TOKEN"]


def call(path, body=None, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode() if body is not None else None, HEADERS)
    try:
        return urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {exc.read().decode()[:600]}")


def image(out, model, w, h, seed, prompt):
    r = json.loads(call("/api/images/generate", {"modelId": model, "prompt": prompt, "width": w, "height": h, "seed": seed, "steps": 4}))
    art = r["artifacts"][0]  # key is "artifacts" (a list); the PNG comes back as a data URI in previewUrl
    open(out, "wb").write(base64.b64decode(art["previewUrl"].split(",", 1)[1]))


def video(out, model, frames, seed, prompt):
    # LTX-2.5: frames must be 8k+1 (73 = 3 s, 97 = 4 s, 121 = 5 s at 24 fps); steps/guidance are ignored by the engine.
    r = json.loads(call("/api/video/generate", {"modelId": model, "prompt": prompt, "width": 768, "height": 512,
                                                "numFrames": frames, "fps": 24, "steps": 8, "guidance": 1.0, "seed": seed}))
    aid = r["artifact"]["artifactId"]
    open(out, "wb").write(call(f"/api/video/outputs/{aid}/file"))  # mp4 with the generated stereo sound


def voice(out, voice_id, text):
    open(out, "wb").write(call("/api/voice/synthesize", {"text": text, "voice": voice_id, "speed": 0.95}))  # WAV


if __name__ == "__main__":
    kind, a = sys.argv[1], sys.argv[2:]
    if kind == "image": image(a[0], a[1], int(a[2]), int(a[3]), int(a[4]), a[5])
    elif kind == "video": video(a[0], a[1], int(a[2]), int(a[3]), a[4])
    elif kind == "voice": voice(a[0], a[1], a[2])
    else: sys.exit(__doc__)
    print("saved", a[0])
