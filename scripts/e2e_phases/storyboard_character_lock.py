"""Phase 26: the Storyboard character lock on the NVIDIA lane (B1315), FLUX 3 Klein through stable-diffusion.cpp.

A character with ``inEveryShot`` has its reference picture in every shot's still, except the shots that list it in
``castOut``; the NVIDIA lane's still model (FLUX 3 Klein · sd.cpp Q8_0) takes up to four reference pictures, so the
picture goes into the request itself and not only the character's words.

Two parts:

* **In process, always runs** (a temporary project store, the real catalogue row, the studio's real still builder with
  only the image route's inner call replaced by a stand-in that records the request and writes a PNG): the data rules
  (a shot's effective cast: ticked references first, then every-shot characters it does not leave out; ticking and
  leaving out stay apart; only a character can be in every shot) and the request of each shot's still on the FLUX 3
  Klein row: the picture of the character is in ``referenceImagePaths`` and its words are in the prompt; the shot that
  leaves the character out has neither; a still model that takes no pictures gets the words alone and a note.
* **Live, skips with its reason** unless the backend answers, the FLUX 3 Klein weights are downloaded and the ``sd``
  binary is staged: a throw-away project with one character (a stand-in picture) in every shot and three shots, the
  stills made through ``POST .../batches`` at the smallest usable size (512 x 320). Asserts the batch is done, each still
  is a PNG of that size, each take records ``cast: pictures``; then the same three shots with the character left out
  (same seeds) come out with no ``cast`` record and different pixels, so the picture, not the seed, changed the render.

What it cannot show: that the person looks the same. The stand-in picture is flat colour bands, so the run proves the
request and the render path, not identity. Whether the face, hair and clothes carry over is judged by looking at real
stills with a real portrait as the reference (the 4090 identity test, B1315). The live part needs a GPU and the project folder the
backend uses; it deletes the project it made.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

PHASE_KEY = 26
ROOT = Path(__file__).resolve().parents[2]
KLEIN = "leejet/flux-3-image-klein-GGUF-q8"
LIVE_SIZE = (512, 320)  # the smallest size that keeps the model's multiples of 64 and a legible picture
PNG = b"\x89PNG\r\n\x1a\n"
SHOTS = [
    ("Cafe at night", "Mara at a rain-streaked cafe window at night, neon light on her face, side view", 101),
    ("Market at noon", "Mara laughing in a sunlit market square, low camera looking up at her", 102),
    ("Library", "High angle on Mara reading a book in a dim library, cool light from a tall window", 103),
]


def _setup_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _png(path: Path, width: int = 64, height: int = 64) -> Path:
    from e2e_phases.phase_20 import png

    path.write_bytes(png(width, height, [(205, 110, 40), (40, 120, 130), (230, 190, 40)]))
    return path


def _in_process_store(base: Path) -> Any:
    _setup_path()
    from backend_service.storyboard.store import ProjectStore

    return ProjectStore(base / "storyboards")


def _png_size(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Storyboard character lock (NVIDIA lane)")
    _setup_path()
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)

    def _data_rules():
        import backend_service.app  # noqa: F401 - the route modules import each other; the app module loads them in order
        from backend_service.storyboard import cast
        from backend_service.storyboard.ops import OpError, run_op

        store = _in_process_store(base)
        p = store.create("E2E lock rules", None)

        def do(op_name: str, /, **args: Any) -> Any:
            return run_op(store, p.id, p.revision, op_name, args)

        p = do("add_reference", kind="character", name="Mara", description="copper curls, round glasses",
               images=[str(_png(base / "mara.png"))], inEveryShot=True)
        p = do("add_reference", kind="character", name="Ben", description="tall, grey beard",
               images=[str(_png(base / "ben.png"))])
        p = do("add_reference", kind="place", name="Pier", images=[str(_png(base / "pier.png"))])
        for title in ("a", "b", "c"):
            p = do("add_shot", title=title, stillPrompt=f"shot {title}")
        mara, ben = (r.id for r in p.references.characters)
        pier = p.references.places[0].id
        a, b, c = (s.id for s in p.shots)
        p = do("edit_shot", shotId=b, referenceImages=[pier, ben])
        p = do("edit_shot", shotId=c, castOut=[mara])
        got = [cast.effective_references(p, p.shot(s)) for s in (a, b, c)]
        if got != [[mara], [pier, ben, mara], []]:
            return "fail", f"effective references {got}", {}
        p = do("edit_shot", shotId=c, referenceImages=[mara])  # ticked again: no longer left out
        if (p.shot(c).referenceImages, p.shot(c).castOut) != ([mara], []):
            return "fail", f"ticking did not clear castOut: {p.shot(c).referenceImages} / {p.shot(c).castOut}", {}
        p = do("edit_shot", shotId=c, castOut=[mara])  # left out again: unticked
        if (p.shot(c).referenceImages, p.shot(c).castOut) != ([], [mara]):
            return "fail", f"leaving out did not clear the tick: {p.shot(c).referenceImages} / {p.shot(c).castOut}", {}
        try:
            do("edit_reference", refId=pier, inEveryShot=True)
        except OpError:
            pass
        else:
            return "fail", "a place was accepted as in every shot", {}
        p = do("edit_reference", refId=mara, inEveryShot=False)
        if cast.effective_references(p, p.shot(a)) != []:
            return "fail", "the lock is off but shot a still has the character", {}
        return "pass", "", {"effective": got}

    def _request_on_the_klein_row():
        import backend_service.app  # noqa: F401
        from backend_service.helpers.images import _find_image_variant
        from backend_service.helpers.media_input import checked_reference_images, reference_image_limit
        from backend_service.storyboard import generators
        from backend_service.storyboard.ops import run_op

        variant = _find_image_variant(KLEIN)
        if variant is None:
            return "fail", f"{KLEIN} is not in the image catalogue", {}
        if variant.get("engine") != "sdcpp" or reference_image_limit(variant) < 1:
            return "fail", f"the row's engine {variant.get('engine')!r}, reference pictures {reference_image_limit(variant)}", {}

        store = _in_process_store(base)
        p = store.create("E2E lock request", {"imageModel": KLEIN, "width": LIVE_SIZE[0], "height": LIVE_SIZE[1]})
        p = run_op(store, p.id, p.revision, "add_reference", {"kind": "character", "name": "Mara",
                                                             "description": "copper curls, round glasses",
                                                             "images": [str(_png(base / "mara-req.png"))],
                                                             "inEveryShot": True})
        mara = p.references.characters[0].id
        for title, prompt, seed in SHOTS:
            p = run_op(store, p.id, p.revision, "add_shot", {"title": title, "stillPrompt": prompt, "seed": seed})
        p = run_op(store, p.id, p.revision, "edit_shot", {"shotId": p.shots[2].id, "castOut": [mara]})
        folder = store.folder(p.id)
        sent: list[Any] = []

        def stand_in(_request: Any, body: Any) -> dict[str, Any]:
            sent.append(body)
            return {"artifacts": [{"imagePath": str(_png(base / f"out-{len(sent)}.png")), "modelName": "stand-in",
                                   "seed": body.seed, "width": body.width, "height": body.height}]}

        with mock.patch.object(generators, "_variant", return_value=variant), \
                mock.patch("backend_service.helpers.job_queue.HEAVY_JOBS.slot", return_value=nullcontext()), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=stand_in):
            make_still = generators.studio_generators(SimpleNamespace(), "b_e2e", folder)["make_still"]
            infos = [make_still(shot, p)[1]["settings"] for shot in p.shots]
        ref_path = str((folder / p.references.characters[0].images[0].path).resolve())
        for i in (0, 1):
            body, info = sent[i], infos[i]
            if body.modelId != KLEIN or body.referenceImagePaths != [ref_path]:
                return "fail", f"shot {i + 1}: model {body.modelId}, pictures {body.referenceImagePaths}, want [{ref_path}]", {}
            if "Characters: Mara" not in body.prompt or SHOTS[i][1] not in body.prompt:
                return "fail", f"shot {i + 1}: prompt lost the character's words or the shot: {body.prompt!r}", {}
            if info.get("cast") != "pictures" or "castNote" in info or body.seed != SHOTS[i][2]:
                return "fail", f"shot {i + 1}: take settings {info}, seed {body.seed}", {}
            if not Path(body.referenceImagePaths[0]).is_file():
                return "fail", f"shot {i + 1}: the picture {body.referenceImagePaths[0]} is not a file", {}
        if checked_reference_images(sent[0].referenceImagePaths, variant) != [ref_path]:
            return "fail", "the image route would not take the picture the shot sent", {}
        body, info = sent[2], infos[2]
        if body.referenceImagePaths or "Characters:" in body.prompt or "cast" in info:
            return "fail", f"the shot that left Mara out still has her: {body.referenceImagePaths}, {body.prompt!r}, {info}", {}
        # A still model that takes no pictures: the character goes in as words, with a note, and the request is not refused.
        flux1 = {"id": "stand-in", "name": "Stand-in", "engine": "diffusers", "repo": "black-forest-labs/FLUX.1-schnell"}
        sent.clear()
        with mock.patch.object(generators, "_variant", return_value=flux1), \
                mock.patch("backend_service.helpers.job_queue.HEAVY_JOBS.slot", return_value=nullcontext()), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=stand_in):
            info = generators.studio_generators(SimpleNamespace(), "b_e2e", folder)["make_still"](p.shots[0], p)[1]["settings"]
        if sent[0].referenceImagePaths or info.get("cast") != "words" or "words only" not in info.get("castNote", ""):
            return "fail", f"a model without pictures: {sent[0].referenceImagePaths}, {info}", {}
        return "pass", "", {"pictures": 1, "model": KLEIN}

    def api(method: str, path: str, payload: Any = None, timeout: float = 120.0) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, data = suite._voice_http(method, path, body=body, content_type="application/json" if body else None,
                                       timeout=timeout)
        try:
            return code, json.loads(data.decode("utf-8")) if data.strip() else None
        except (ValueError, UnicodeDecodeError):
            return code, data

    made: list[str] = []

    def _live_three_stills():
        if not cap.backend_reachable:
            return "skip", "backend not reachable", {}
        code, catalog = api("GET", "/api/images/catalog")
        if code != 200 or not isinstance(catalog, dict):
            return "skip", f"image catalogue not readable (HTTP {code})", {}
        row = next((v for f in catalog.get("families", []) for v in f.get("variants", []) if v.get("id") == KLEIN), None)
        if row is None:
            return "skip", f"{KLEIN} is not in this build's catalogue", {}
        if not row.get("availableLocally"):
            return "skip", "the FLUX 3 Klein sd.cpp weights are not downloaded", {}

        def op(body: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
            project = body["project"]
            code, out = api("POST", f"/api/storyboard/projects/{project['id']}/ops",
                            {"revision": project["revision"], "op": name, "args": args})
            if code != 200:
                raise RuntimeError(f"{name}: HTTP {code} {str(out)[:240]}")
            return out

        def stills(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
            """Make every still (forced), wait for the batch. Returns (project, skip reason or '', batch)."""
            project = body["project"]
            code, out = api("POST", f"/api/storyboard/projects/{project['id']}/batches",
                            {"revision": project["revision"], "what": "still", "force": True})
            if code != 200:
                raise RuntimeError(f"batch: HTTP {code} {str(out)[:240]}")
            deadline = time.time() + 300
            while True:
                code, info = api("GET", f"/api/storyboard/projects/{project['id']}/batches/{out['batchId']}")
                if code == 200 and info["batch"]["state"] not in ("running", "queued", "pending"):
                    break
                if time.time() > deadline:
                    raise RuntimeError("the batch was still running after 300 s")
                time.sleep(1.0)
            failed = [r for r in info["batch"]["results"] if r["state"] == "failed"]
            if failed:
                text = str(failed[0].get("error") or "")
                if "binary not staged" in text:  # the backend's own words for no sd binary
                    return None, text, info["batch"]
                raise RuntimeError(f"a still failed: {text[:240]}")
            code, again = api("GET", f"/api/storyboard/projects/{project['id']}")
            return again, "", info["batch"]

        def fetch(project: dict[str, Any], rel: str) -> bytes:
            code, data = suite._voice_http("GET", f"/api/storyboard/projects/{project['id']}/files?path={rel}")
            if code != 200 or not data.startswith(PNG):
                raise RuntimeError(f"{rel}: HTTP {code}, {data[:24]!r}, not a PNG")
            return data

        code, body = api("POST", "/api/storyboard/projects", {"name": "E2E character lock"})
        if code != 200:
            raise RuntimeError(f"create: HTTP {code} {str(body)[:200]}")
        made.append(body["project"]["id"])
        body = op(body, "edit_project", {"defaults": {"imageModel": KLEIN, "width": LIVE_SIZE[0], "height": LIVE_SIZE[1]}})
        body = op(body, "add_reference", {"kind": "character", "name": "Mara", "description": "a woman in her forties",
                                          "image": str(_png(base / "mara-live.png", 256, 256)), "inEveryShot": True})
        for title, prompt, seed in SHOTS:
            body = op(body, "add_shot", {"title": title, "stillPrompt": prompt, "seed": seed, "seconds": 3})
        mara = body["project"]["references"]["characters"][0]["id"]

        started = time.time()
        with_cast, reason, batch = stills(body)
        if with_cast is None:
            return "skip", reason, {}
        cast_seconds = round(time.time() - started, 1)
        pictures: list[bytes] = []
        for shot in with_cast["project"]["shots"]:
            take = next(t for t in shot["takes"] if t["id"] == shot["activeTake"])
            data = fetch(with_cast["project"], take["still"])
            if _png_size(data) != LIVE_SIZE:
                raise RuntimeError(f"a still is {_png_size(data)}, asked for {LIVE_SIZE}")
            if take["settings"].get("cast") != "pictures" or "castNote" in take["settings"]:
                raise RuntimeError(f"take settings {take['settings']}: the picture did not go in")
            pictures.append(data)
        if len(pictures) != 3 or len(set(pictures)) != 3:
            return "fail", f"{len(pictures)} stills, {len(set(pictures))} different", {}

        left_out = with_cast
        for shot in with_cast["project"]["shots"]:
            left_out = op(left_out, "edit_shot", {"shotId": shot["id"], "castOut": [mara]})
        control, reason, _ = stills(left_out)
        if control is None:
            return "skip", reason, {}
        for i, shot in enumerate(control["project"]["shots"]):
            take = next(t for t in shot["takes"] if t["id"] == shot["activeTake"])
            if "cast" in take["settings"]:
                return "fail", f"shot {i + 1} left Mara out but the take records cast {take['settings']['cast']!r}", {}
            if fetch(control["project"], take["still"]) == pictures[i]:
                return "fail", f"shot {i + 1}: the same pixels with and without the character (same seed)", {}
        return "pass", "", {"size": list(LIVE_SIZE), "secondsForThree": cast_seconds, "batch": batch["id"]}

    try:
        for name, fn in [
            ("character lock rules: effective cast, ticking against leaving out, characters only", _data_rules),
            ("a shot's still request on FLUX 3 Klein sd.cpp carries the character's picture and words; the opted-out shot "
             "has neither; a model without pictures gets words and a note (in process)", _request_on_the_klein_row),
            ("live: three stills of one character at 512 x 320 on the NVIDIA lane, then the control with the character left "
             "out (skips without the backend, the Klein weights or the sd binary)", _live_three_stills),
        ]:
            phase.checks.append(suite._check(name, fn))
    finally:
        for project_id in made:
            code, body = api("GET", f"/api/storyboard/projects/{project_id}")
            if code == 200:
                suite._voice_http("DELETE", f"/api/storyboard/projects/{project_id}?revision={body['project']['revision']}")
        tmp.cleanup()
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
