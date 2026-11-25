"""Storyboard contracts for the guided flow (UX audit stream A): failure codes, lip-sync coverage, quality labels,
export readiness and the stage rail, plus the ``guide`` block on every project payload. No model runs."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service import lipsync_runtime
from backend_service.storyboard import coverage as cov_mod
from backend_service.storyboard import failure, guide, quality, song_vocals as sv
from backend_service.storyboard.batch import Batch
from backend_service.storyboard.ffmpeg_bin import FfmpegMissing
from backend_service.storyboard.models import BatchResult
from backend_service.storyboard.stages import STAGE_IDS
from backend_service.storyboard.timeline import compute_timeline
from tests.storyboard_fixtures import FakeGenerators, StoreCase, fake_mp4, fake_wav, png
from tests.test_storyboard_song import song_args

H = {"Authorization": "Bearer app-token"}


class FailureCodeTests(unittest.TestCase):
    def test_known_errors_get_their_code_and_action(self):
        cases = {
            "CUDA out of memory. Tried to allocate 2.00 GiB": ("out-of-memory", "smaller-size"),
            "LTX needs about 26 GB of free memory to make a clip, but only 20.1 GB is free.": ("out-of-memory", "smaller-size"),
            "No video model is installed. Call recommend_models": ("model-missing", "open-models"),
            "FLUX.1 is not installed locally yet.": ("model-missing", "open-models"),
            "Wan 2.2 cannot lip-sync. Pick LTX-2.3 distilled": ("lipsync-model", "open-models"),
            "Video tools are needed. Open Setup and choose Install video tools": ("video-tools-missing", "install-video-tools"),
            "Wan cannot start a clip from an image, so it would ignore": ("references-unsupported", "open-models"),
            "SD 3.5 allows at most 1536 px per side": ("size-over-cap", "smaller-size"),
            "the shot has no still prompt": ("no-prompt", "edit-shot"),
            "[Errno 28] No space left on device": ("disk-full", "free-disk"),
            "cancelled": ("cancelled", "retry"),
            "401 Client Error: gated repo": ("download-blocked", "open-models"),
            "KeyError: 'transformer'": ("unknown", "retry"),
        }
        for text, (code, action) in cases.items():
            with self.subTest(text=text):
                got = failure.describe(failure.classify(text))
                self.assertEqual((got["code"], got["action"]), (code, action))
                self.assertTrue(got["message"].endswith("."))

    def test_the_exception_type_wins_over_the_text(self):
        self.assertEqual(failure.classify("anything", FfmpegMissing()), "video-tools-missing")
        self.assertEqual(failure.classify("boom", MemoryError()), "out-of-memory")
        self.assertEqual(failure.classify("boom", OSError(28, "full")), "disk-full")

    def test_every_code_has_a_known_action_and_unknown_codes_read_as_unknown(self):
        self.assertTrue(all(action in failure.ACTIONS for _, action in failure.FAILURES.values()))
        self.assertEqual(failure.describe("from-the-future")["code"], "unknown")

    def test_an_old_result_without_a_code_is_classified_from_its_text(self):
        old = BatchResult(shotId="s1", kind="clip", state="failed", error="CUDA out of memory")
        self.assertEqual(failure.result_failure(old)["code"], "out-of-memory")
        self.assertEqual(failure.result_failure(old)["error"], "CUDA out of memory")


class FailingGenerators(FakeGenerators):
    def make_still(self, shot, project):
        raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")


class BatchFailureTests(StoreCase):
    def test_a_failed_step_stores_the_code_beside_the_raw_text(self):
        project = self.new(shots=1)
        gens = FailingGenerators(self.src)
        batch = Batch(self.store, project.id, "still", gens.make_still, gens.make_clip, gens.unload, gens.last_frame)
        batch.start(project.revision)
        batch.run()
        project = self.store.load(project.id)
        result = next(r for r in project.batches[-1].results if r.state == "failed")
        self.assertEqual(result.code, "out-of-memory")
        self.assertIn("CUDA out of memory", result.error)
        failures = guide.failures(project)
        self.assertEqual([(f["shotId"], f["code"], f["action"]) for f in failures],
                         [(project.shots[0].id, "out-of-memory", "smaller-size")])
        self.assertEqual(failures[0]["batchId"], project.batches[-1].id)

    def test_each_half_records_its_own_size(self):
        project = self.new(shots=1)
        gens = FakeGenerators(self.src)

        def still(shot, p):
            return gens.make_still(shot, p), {"settings": {"width": 640, "height": 384}}

        def clip(shot, p, image):
            path, info = gens.make_clip(shot, p, image)
            return path, {**info, "settings": {"width": 1280, "height": 720, "fps": 24}}

        batch = Batch(self.store, project.id, "both", still, clip, gens.unload, gens.last_frame)
        batch.start(project.revision)
        batch.run()
        take = self.store.load(project.id).shots[0].active()
        self.assertEqual({k: take.settings[k] for k in ("stillWidth", "stillHeight", "clipWidth", "clipHeight", "clipFps")},
                         {"stillWidth": 640, "stillHeight": 384, "clipWidth": 1280, "clipHeight": 720, "clipFps": 24})


class CoverageTests(StoreCase):
    def project(self, lengths):
        project = self.new(shots=len(lengths))
        for shot, seconds in zip(project.shots, lengths):
            project = self.op(project, "edit_shot", shotId=shot.id, seconds=seconds, trimOut=seconds)
        return self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav")))

    def test_rules_match_the_lip_sync_runtime(self):
        self.assertEqual(cov_mod.CLIP_MAX_SECONDS, lipsync_runtime.MAX_SECONDS)
        self.assertEqual(sv.MIN_SHOT_SECONDS, 2.0)  # 2 s and up lip-syncs; a 2 to 3 s shot gets silent room

    def test_skipped_shots_carry_a_code_and_a_sentence(self):
        project = self.project([1.5, 9.0, 4.0, 3.0])
        project = self.op(project, "lock", shotId=project.shots[3].id, locked=True)
        plan = sv.plan_vocals(project)
        self.assertEqual([(s["number"], s["code"]) for s in plan["skipped"]], [(1, "too-short"), (2, "too-long"), (4, "locked")])
        self.assertIn("Shot 1 lasts 1.5 s", plan["skipped"][0]["reason"])
        self.assertIn("at least 2 s", plan["skipped"][0]["reason"])

    def test_no_singing_comes_from_the_stem_and_is_kept_in_the_stored_record(self):
        project = self.project([4.0, 4.0])
        plan = {"items": [{"shotId": project.shots[1].id, "number": 2, "songStart": 3.6, "lead": 0.4, "length": 4.0}],
                "skipped": [{"shotId": project.shots[0].id, "number": 1, "code": "no-singing",
                             "reason": "Shot 1 has no singing in the vocal stem.", "songStart": 0.0, "length": 4.0}]}
        record = cov_mod.record(plan, "stem", time.time())
        self.assertEqual([s["shotId"] for s in record["shots"]], [s.id for s in project.shots])
        items = [{"shotId": project.shots[1].id, "audio": str(fake_wav(self.src / "v.wav")), "lead": 0.4, "length": 4.0}]
        project = self.op(project, "set_shot_vocals", items=items, coverage=record)
        self.assertEqual(project.song.coverage.source, "stem")
        table = cov_mod.coverage(project)
        self.assertEqual([(r["state"], r["code"], r["fix"]) for r in table["shots"]],
                         [("skipped", "no-singing", None), ("waiting", None, "make-lipsync")])
        self.assertEqual(table["counts"]["sings"], 1)
        undone = self.store.undo(project.id, project.revision)
        self.assertIsNone(undone.song.coverage)  # one op, one Undo, coverage included

    def test_a_run_with_no_vocals_still_stores_its_coverage(self):
        project = self.project([1.0])
        record = cov_mod.record(sv.plan_vocals(project), "song", 1.0)
        project = self.op(project, "set_shot_vocals", coverage=record)
        self.assertEqual(project.song.coverage.shots[0].code, "too-short")

    def test_live_reasons_follow_the_shot_and_offer_the_fix(self):
        project = self.project([1.5, 4.0])
        table = cov_mod.coverage(project)
        self.assertEqual([(r["state"], r["code"], r["fix"]) for r in table["shots"]],
                         [("skipped", "too-short", "lengthen"), ("unassigned", "not-assigned", "assign-vocals")])
        project = self.op(project, "edit_shot", shotId=project.shots[0].id, seconds=2.5, trimOut=2.5)
        self.assertEqual(cov_mod.coverage(project)["shots"][0]["code"], "not-assigned")  # fixed: no longer too short
        self.assertEqual(table["shots"][1]["maxSeconds"], 7.6)

    def test_a_shot_with_a_current_clip_is_synced(self):
        project = self.project([4.0])
        sid = project.shots[0].id
        items = [{"shotId": sid, "audio": str(fake_wav(self.src / "v.wav")), "lead": 0.0, "length": 4.0}]
        project = self.op(project, "set_shot_vocals", items=items)
        project = self.op(project, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "c.mp4")))
        self.assertEqual(project.shots[0].status, "done")
        self.assertEqual(cov_mod.coverage(project)["shots"][0]["state"], "synced")

    def test_no_song_no_coverage(self):
        self.assertIsNone(cov_mod.coverage(self.new(shots=1)))


class QualityTests(StoreCase):
    def test_drafts_sizes_and_stale_for_the_format(self):
        project = self.new(shots=2)
        a, b = project.shots[0].id, project.shots[1].id
        gens = FakeGenerators(self.src)

        def still(shot, p):
            return gens.make_still(shot, p), {"settings": {"width": p.defaults.width, "height": p.defaults.height}}

        def clip(shot, p, image):
            path, info = gens.make_clip(shot, p, image)
            return path, {**info, "settings": {"width": p.defaults.width, "height": p.defaults.height, "fps": 24}}

        batch = Batch(self.store, project.id, "both", still, clip, gens.unload, gens.last_frame, shot_ids=[a],
                      draft=True, draft_sizes={"still": (640, 384), "clip": (640, 384)})
        batch.start(project.revision)
        batch.run()
        project = self.store.load(project.id)
        batch = Batch(self.store, project.id, "both", still, clip, gens.unload, gens.last_frame, shot_ids=[b])
        batch.start(project.revision)
        batch.run()
        project = self.store.load(project.id)
        q = quality.quality(project)
        self.assertEqual(q["target"], {"width": 1280, "height": 720, "fps": 24})
        self.assertEqual(q["draftSize"], {"width": 640, "height": 384})
        self.assertEqual(q["draftShots"], [a])
        label = q["takes"][a][project.shot(a).activeTake]
        self.assertEqual((label["pass"], label["draft"], label["still"], label["clip"]),
                         ("draft", True, {"width": 640, "height": 384}, {"width": 640, "height": 384, "fps": 24}))
        self.assertEqual(q["staleShots"], [])
        project = self.op(project, "edit_project", defaults={"width": 1920, "height": 1080})
        self.assertEqual(quality.quality(project)["staleShots"], [b])  # the final take is the old size; the draft's shape holds
        project = self.op(project, "edit_project", defaults={"exportPreset": "9:16"})
        self.assertEqual(set(quality.quality(project)["staleShots"]), {a, b})

    def test_an_uploaded_take_has_no_size_and_is_not_stale(self):
        project = self.new(shots=1)
        project = self.op(project, "import_take", shotId=project.shots[0].id, still=str(png(self.src / "p.png")))
        label = next(iter(quality.quality(project)["takes"][project.shots[0].id].values()))
        self.assertEqual((label["still"], label["stillStale"], label["draft"]), (None, False, False))


class ReadinessAndStageTests(StoreCase):
    def guide(self, project):
        return guide.build(project, compute_timeline(project), self.folder(project))

    def test_an_empty_project_stops_the_export_and_starts_at_shots(self):
        g = self.guide(self.new())
        self.assertEqual([i["code"] for i in g["readiness"]["stops"]], ["no-shots"])
        self.assertFalse(g["readiness"]["ready"])
        self.assertEqual([s["id"] for s in g["stages"]["stages"]], list(STAGE_IDS))
        self.assertEqual(g["stages"]["current"], "shots")
        song = g["stages"]["stages"][0]
        self.assertEqual((song["applies"], song["action"]), (False, "attach-song"))

    def test_missing_clips_failed_shots_and_drafts_are_worth_fixing(self):
        project = self.new(shots=3)
        a, b, c = (s.id for s in project.shots)
        project = self.op(project, "import_take", shotId=a, clip=str(fake_mp4(self.src / "a.mp4")))
        g = self.guide(project)
        self.assertEqual(g["readiness"]["stops"], [])
        worth = {i["code"]: i for i in g["readiness"]["worthFixing"]}
        self.assertEqual(worth["missing-clips"]["shotIds"], [b, c])
        self.assertEqual(worth["missing-clips"]["action"], "make-clips")
        self.assertEqual(g["readiness"]["forInfo"][0]["code"], "format")
        stages = {s["id"]: s for s in g["stages"]["stages"]}
        self.assertEqual((stages["drafts"]["status"], stages["drafts"]["action"]), ("in-progress", "make-missing"))
        self.assertEqual(g["stages"]["current"], "drafts")
        self.assertEqual(stages["check"]["status"], "not-started")  # not before the shots are made

    def test_song_length_clip_sound_and_lip_sync_gaps(self):
        project = self.new(shots=2)
        a, b = (s.id for s in project.shots)
        project = self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav"), seconds=40.0))
        for sid in (a, b):
            project = self.op(project, "edit_shot", shotId=sid, seconds=4.0, trimOut=4.0)
            project = self.op(project, "import_take", shotId=sid, clip=str(fake_mp4(self.src / f"{sid}.mp4")))
        project = self.op(project, "edit_shot", shotId=a, nativeSoundDb=-60.0)  # shots made before the song are not muted
        project = self.op(project, "edit_shot", shotId=b, nativeSoundDb=0.0, seconds=1.5, trimOut=1.5)
        record = cov_mod.record(sv.plan_vocals(project), "song", 1.0)
        project = self.op(project, "set_shot_vocals", coverage=record)
        g = self.guide(project)
        codes = [i["code"] for i in g["readiness"]["worthFixing"]]
        self.assertIn("song-length", codes)
        self.assertIn("clip-sound-audible", codes)
        gap = next(i for i in g["readiness"]["worthFixing"] if i["code"] == "lipsync-gap")
        self.assertEqual((gap["shotIds"], gap["action"], gap["data"]["reason"]), ([b], "lengthen", "too-short"))
        song_length = next(i for i in g["readiness"]["worthFixing"] if i["code"] == "song-length")
        self.assertEqual(song_length["message"], "The film is 0:06, the song 0:40.")
        self.assertIn("clip-sound-muted", [i["code"] for i in g["readiness"]["forInfo"]])
        stages = {s["id"]: s for s in g["stages"]["stages"]}
        self.assertEqual(stages["song"]["status"], "done")
        self.assertEqual((stages["lipsync"]["status"], stages["lipsync"]["action"]), ("needs-attention", "open-coverage"))
        self.assertEqual((stages["sound"]["status"], stages["sound"]["action"]), ("needs-attention", "open-mixer"))
        self.assertEqual(stages["check"]["status"], "needs-attention")

    def test_a_clean_film_is_ready_and_export_is_next(self):
        project = self.new(shots=1)
        project = self.op(project, "import_take", shotId=project.shots[0].id, clip=str(fake_mp4(self.src / "a.mp4")))
        g = self.guide(project)
        self.assertTrue(g["readiness"]["ready"])
        self.assertEqual(g["readiness"]["worthFixing"], [])
        self.assertEqual(g["stages"]["current"], "export")
        exports = self.folder(project) / "exports"
        exports.mkdir(exist_ok=True)
        (exports / "Lighthouse.mp4").write_bytes(b"x")
        self.assertIsNone(self.guide(project)["stages"]["current"])


class GuideRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(Path(self.tmp.name) / "storyboards")})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app())

    def test_every_payload_has_the_guide_and_the_guide_route_reads_it(self):
        response = self.client.post("/api/storyboard/projects", json={"name": "Night"}, headers=H)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body["guide"]), {"quality", "coverage", "failures", "readiness", "stages"})
        pid = body["project"]["id"]
        response = self.client.get(f"/api/storyboard/projects/{pid}/guide", headers=H)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["guide"]["stages"]["current"], "shots")
        self.assertEqual(response.json()["revision"], body["project"]["revision"])


def create_app():
    from backend_service.app import create_app as make

    return make(api_token="app-token")


if __name__ == "__main__":
    unittest.main()
