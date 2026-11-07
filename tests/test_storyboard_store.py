"""Storyboard project store: layout, revisions, history and undo, hostile input, media, deps, batches, bundles."""

import json
import os
import threading
import time
import unittest
import zipfile
from unittest import mock

from backend_service.storyboard import bundle, store as store_module
from backend_service.storyboard.batch import Batch, BatchRefused
from backend_service.storyboard.deps import stale_shots
from backend_service.storyboard.media import sha256_file
from backend_service.storyboard.models import SCHEMA_VERSION
from backend_service.storyboard.ops import OpError
from backend_service.storyboard.store import RevisionConflict, StoreError, migrate, parse_project
from tests.storyboard_fixtures import FakeGenerators, StoreCase, fake_mp4, fake_wav, png


def _minimal(**extra):
    data = {"schemaVersion": SCHEMA_VERSION, "revision": 1, "id": "0123456789abcdef", "name": "x"}
    data.update(extra)
    return json.dumps(data).encode()


class LayoutAndRevisionTests(StoreCase):
    def test_create_makes_a_self_contained_folder(self):
        project = self.new()
        folder = self.folder(project)
        for sub in ("media/audio", "cache", "history", "exports"):
            self.assertTrue((folder / sub).is_dir(), sub)
        data = json.loads((folder / "project.json").read_text())
        self.assertEqual((data["schemaVersion"], data["revision"]), (SCHEMA_VERSION, 1))
        self.assertRegex(project.id, r"^[0-9a-f]{16}$")
        self.assertEqual(self.store.list()[0]["name"], "Lighthouse")

    def test_project_ids_are_never_path_pieces(self):
        for bad in ("../x", "/etc", "ABC", "", "0123456789abcdeg"):
            with self.assertRaises(StoreError):
                self.store.folder(bad)

    def test_every_change_bumps_revision_writes_atomically_and_logs_history(self):
        project = self.new(shots=2)
        self.assertEqual(project.revision, 3)
        folder = self.folder(project)
        self.assertEqual(len(list((folder / "history").glob("*.json"))), 2)
        self.assertEqual([p.name for p in folder.iterdir() if p.name.startswith(".tmp-")], [])
        entry = json.loads((folder / "history" / "00000003.json").read_text())
        self.assertEqual(entry["op"], "add_shot")
        self.assertIsNone(next(iter(entry["before"]["shots"]["items"].values())))

    def test_old_revision_is_refused_with_the_current_project(self):
        project = self.new(shots=1)
        self.op(project, "edit_shot", shotId=project.shots[0].id, title="new")
        with self.assertRaises(RevisionConflict) as caught:
            self.op(project, "edit_shot", shotId=project.shots[0].id, title="stale write")
        self.assertEqual(caught.exception.current.revision, project.revision + 1)
        self.assertEqual(caught.exception.current.shots[0].title, "new")

    def test_concurrent_writers_with_the_same_revision_one_wins(self):
        project = self.new(shots=1)
        outcomes = []

        def write(title):
            try:
                self.op(project, "edit_shot", shotId=project.shots[0].id, title=title)
                outcomes.append("ok")
            except RevisionConflict:
                outcomes.append("conflict")

        threads = [threading.Thread(target=write, args=(f"t{i}",)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(sorted(outcomes), ["conflict"] * 5 + ["ok"])
        self.assertEqual(self.store.load(project.id).revision, project.revision + 1)

    def test_bad_op_args_are_readable(self):
        project = self.new(shots=1)
        for op_name, args, needle in (
            ("edit_shot", {"shotId": "nope", "title": "x"}, "Unknown shot"),
            ("edit_shot", {"shotId": project.shots[0].id, "speed": 9}, "speed"),
            ("edit_shot", {"shotId": project.shots[0].id, "bogus": 1}, "bogus"),
            ("reorder", {"order": ["x"]}, "every shot"),
            ("nope", {}, "Unknown op"),
        ):
            with self.assertRaises(OpError) as caught:
                self.op(project, op_name, **args)
            self.assertIn(needle, str(caught.exception))

    def test_restart_resets_unfinished_work(self):
        project = self.new(shots=2)
        path = self.folder(project) / "project.json"
        data = json.loads(path.read_text())
        data["shots"][0]["status"] = "running"
        data["shots"][1]["status"] = "queued"
        data["batches"] = [{"id": "b_1234abcd", "what": "both", "state": "running"}]
        path.write_text(json.dumps(data))
        store_module._RECOVERED.discard(str(self.folder(project).resolve()))
        loaded = self.store.load(project.id)
        self.assertEqual([s.status for s in loaded.shots], ["idle", "idle"])
        self.assertEqual(loaded.batches[0].state, "interrupted")
        self.assertEqual(json.loads(path.read_text())["shots"][0]["status"], "idle")


class UndoRedoTests(StoreCase):
    def _undo_round_trip(self, project, op, check_after, check_before, **args):
        after = self.op(project, op, **args)
        check_after(after)
        undone = self.store.undo(after.id, after.revision)
        check_before(undone)
        redone = self.store.redo(undone.id, undone.revision)
        check_after(redone)
        return redone

    def test_shot_edit_reorder_lock_add_remove(self):
        p = self.new(shots=3)
        a, b, c = (s.id for s in p.shots)
        p = self._undo_round_trip(p, "edit_shot", lambda x: self.assertEqual(x.shot(a).title, "A!"),
                                  lambda x: self.assertEqual(x.shot(a).title, "shot 0"), shotId=a, title="A!")
        p = self._undo_round_trip(p, "reorder", lambda x: self.assertEqual([s.id for s in x.shots], [c, a, b]),
                                  lambda x: self.assertEqual([s.id for s in x.shots], [a, b, c]), order=[c, a, b])
        p = self._undo_round_trip(p, "lock", lambda x: self.assertTrue(x.shot(b).locked),
                                  lambda x: self.assertFalse(x.shot(b).locked), shotId=b)
        p = self._undo_round_trip(p, "remove_shot", lambda x: self.assertIsNone(x.shot(b)),
                                  lambda x: self.assertEqual(x.shots[2].id, b), shotId=b)
        self._undo_round_trip(p, "add_shot", lambda x: self.assertEqual(len(x.shots), 3),
                              lambda x: self.assertEqual(len(x.shots), 2), title="new")

    def test_pick_take_and_audio_edits(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "one.mp4")))
        first = p.shots[0].activeTake
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "two.mp4")))
        p = self._undo_round_trip(p, "pick_take", lambda x: self.assertEqual(x.shot(sid).activeTake, first),
                                  lambda x: self.assertNotEqual(x.shot(sid).activeTake, first), shotId=sid, takeId=first)
        p = self.op(p, "add_audio", kind="voice", path=str(fake_wav(self.src / "v.wav")), anchor={"shotId": sid, "offset": 0.5})
        cid = p.audio[0].id
        p = self._undo_round_trip(p, "edit_audio", lambda x: self.assertEqual(x.clip(cid).gainDb, -6),
                                  lambda x: self.assertEqual(x.clip(cid).gainDb, 0), clipId=cid, gainDb=-6)
        self._undo_round_trip(p, "remove_audio", lambda x: self.assertEqual(x.audio, []),
                              lambda x: self.assertEqual(x.audio[0].id, cid), clipId=cid)

    def test_new_change_clears_redo_and_undo_keeps_later_takes(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self.op(p, "edit_shot", shotId=sid, title="x")
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(len(p.redoStack), 1)
        p = self.op(p, "edit_shot", shotId=sid, stillPrompt="changed")
        self.assertEqual(p.redoStack, [])
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "k.mp4")))
        p = self.store.undo(p.id, p.revision)  # undo the import: the take stays, the pick goes
        self.assertEqual(len(p.shots[0].takes), 1)
        self.assertIsNone(p.shots[0].activeTake)
        with self.assertRaises(StoreError):
            fresh = self.new()
            self.store.undo(fresh.id, fresh.revision)

    def _run_batch(self, p, what="clip"):
        gens = FakeGenerators(self.src)
        batch = Batch(self.store, p.id, what, gens.make_still, gens.make_clip, gens.unload, gens.last_frame,
                      shot_ids=[s.id for s in p.shots], force=True)
        batch.start(p.revision)
        batch.run()
        return self.store.load(p.id)

    def test_undo_of_an_edit_keeps_the_pick_a_later_batch_made(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self._run_batch(p)
        p = self.op(p, "edit_shot", shotId=sid, title="renamed")
        p = self._run_batch(p)
        second = p.shot(sid).activeTake
        self.assertEqual(len(p.shot(sid).takes), 2)
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.shot(sid).title, "shot 0")
        self.assertEqual(p.shot(sid).activeTake, second)
        self.assertEqual(len(p.shot(sid).takes), 2)

    def test_undo_of_an_earlier_edit_keeps_a_relinked_clip(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "c.mp4")))
        p = self.op(p, "edit_shot", shotId=sid, title="renamed")
        (self.folder(p) / p.shots[0].active().clip).unlink()
        p = self.store.load(p.id)
        p = self.op(p, "relink", id=sid, path=str(fake_mp4(self.src / "found.mp4")))
        relinked = p.shot(sid).active().clip
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.shot(sid).title, "shot 0")
        self.assertEqual(p.shot(sid).active().clip, relinked)
        self.assertEqual(p.shot(sid).status, "done")

    def test_undo_of_an_edit_keeps_a_relinked_audio_source(self):
        p = self.new(shots=1)
        p = self.op(p, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")))
        cid = p.audio[0].id
        p = self.op(p, "edit_audio", clipId=cid, gainDb=-6)
        p = self.op(p, "relink", id=cid, path=str(fake_wav(self.src / "m2.wav")))
        relinked = p.clip(cid).source.path
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.clip(cid).gainDb, 0)
        self.assertEqual(p.clip(cid).source.path, relinked)

    def test_undo_then_redo_of_add_shot_brings_back_its_batch_takes(self):
        p = self.new()
        p = self.op(p, "add_shot", title="one")
        sid = p.shots[0].id
        p = self._run_batch(p)
        take = p.shot(sid).activeTake
        self.assertIsNotNone(take)
        p = self.store.undo(p.id, p.revision)
        self.assertIsNone(p.shot(sid))
        p = self.store.redo(p.id, p.revision)
        self.assertEqual(len(p.shot(sid).takes), 1)
        self.assertEqual(p.shot(sid).activeTake, take)
        self.assertEqual(p.shot(sid).status, "done")
        self.assertEqual(p.shot(sid).title, "one")


class HostileInputTests(StoreCase):
    def test_bad_paths_in_project_json_are_refused(self):
        bad = ["/etc/passwd", "../x.png", "media/../../x.png", "C:/x.png", "c:x.png", "file:///etc/x", "http://x/y.mp4",
               "-rf.mp4", "media/-x.mp4", "media/a\nb.mp4", "media\\a.mp4", "other/a.mp4", "media//a.mp4", "media/./a.mp4"]
        for path in bad:
            raw = _minimal(audio=[{"id": "a_1", "kind": "voice", "source": {"path": path}, "start": 0}])
            with self.assertRaises(StoreError, msg=path):
                parse_project(raw)

    def test_other_hostile_shapes_are_refused(self):
        for raw, needle in (
            (b"not json", "not valid JSON"),
            (b"[1]", "object"),
            (_minimal(extra=1), "extra"),
            (_minimal(schemaVersion=SCHEMA_VERSION + 1), "newer"),
            (_minimal(schemaVersion="1"), "schemaVersion"),
            (_minimal(id="../../x"), "id"),
            (_minimal(shots=[{"id": "-x"}]), "shots"),
            (_minimal(shots=[{"id": "s_1"}, {"id": "s_1"}]), "duplicate"),
            (_minimal(shots=[{"id": "s_1", "activeTake": "t_9"}]), "active take"),
            (_minimal(name="a\x00b"), "name"),
            (b" " * (store_module.MAX_PROJECT_BYTES + 1), "too large"),
        ):
            with self.assertRaises(StoreError) as caught:
                parse_project(raw)
            self.assertIn(needle, str(caught.exception), raw[:60])

    def test_symlink_out_of_the_project_reads_as_missing(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "c.mp4")))
        clip = self.folder(p) / p.shots[0].active().clip
        outside = fake_mp4(self.base / "outside.mp4")
        clip.unlink()
        clip.symlink_to(outside)
        self.assertEqual(self.store.load(p.id).shots[0].status, "missing")

    def test_migration_registry_runs_older_formats(self):
        def rename_duration(data):
            for shot in data.get("shots", []):
                shot["seconds"] = shot.pop("duration")
            return data

        old = {"schemaVersion": 1, "revision": 1, "id": "0123456789abcdef", "name": "x", "shots": [{"id": "s_1", "duration": 3}]}
        with mock.patch.object(store_module, "SCHEMA_VERSION", 2), mock.patch.dict(store_module.MIGRATIONS, {1: rename_duration}):
            migrated = migrate(dict(old))
            self.assertEqual((migrated["schemaVersion"], migrated["shots"][0]["seconds"]), (2, 3))
        with mock.patch.object(store_module, "SCHEMA_VERSION", 3), self.assertRaises(StoreError):
            migrate({"schemaVersion": 1})


class MediaTests(StoreCase):
    def test_import_hard_links_on_the_same_disk_with_a_hash(self):
        p = self.new(shots=1)
        src = fake_mp4(self.src / "clip.mp4")
        p = self.op(p, "import_take", shotId=p.shots[0].id, clip=str(src), still=str(png(self.src / "s.png")))
        take = p.shots[0].active()
        self.assertTrue(take.clip.startswith(f"media/{p.shots[0].id}/{take.id}/"))
        real = self.folder(p) / take.clip
        self.assertEqual(real.stat().st_ino, src.stat().st_ino)
        self.assertEqual(take.hashes["clip"], sha256_file(src))
        self.assertEqual(p.shots[0].status, "done")
        with mock.patch("backend_service.storyboard.media.os.link", side_effect=OSError("cross-device")):
            p = self.op(p, "import_take", shotId=p.shots[0].id, clip=str(src))
        self.assertNotEqual((self.folder(p) / p.shots[0].active().clip).stat().st_ino, src.stat().st_ino)

    def test_bad_sources_are_refused(self):
        p = self.new(shots=1)
        (self.src / "x.txt").write_text("hi")
        (self.src / "empty.mp4").write_bytes(b"")
        for path in (str(self.src / "x.txt"), str(self.src / "nope.mp4"), str(self.src / "empty.mp4"), str(self.src)):
            with self.assertRaises(OpError, msg=path):
                self.op(p, "import_take", shotId=p.shots[0].id, clip=path)

    def test_missing_file_then_relink(self):
        p = self.new(shots=1)
        sid = p.shots[0].id
        p = self.op(p, "import_take", shotId=sid, clip=str(fake_mp4(self.src / "c.mp4")))
        p = self.op(p, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")))
        (self.folder(p) / p.shots[0].active().clip).unlink()
        (self.folder(p) / p.audio[0].source.path).unlink()
        p = self.store.load(p.id)
        self.assertEqual((p.shots[0].status, p.audio[0].status), ("missing", "missing"))
        replacement = fake_mp4(self.src / "found.mp4")
        p = self.op(p, "relink", id=sid, path=str(replacement))
        p = self.op(p, "relink", id=p.audio[0].id, path=str(fake_wav(self.src / "m2.wav")))
        self.assertEqual((p.shots[0].status, p.audio[0].status), ("done", "ok"))
        self.assertEqual(p.shots[0].active().hashes["clip"], sha256_file(replacement))

    def test_a_real_97_frame_clip_measures_4_0417_seconds(self):
        import subprocess

        from backend_service.storyboard.media import ffmpeg_exe, probe_seconds

        exe = ffmpeg_exe()
        if not exe:
            self.skipTest("bundled ffmpeg not installed")
        clip = self.src / "97.mp4"
        subprocess.run([exe, "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24", "-frames:v", "97",
                        "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=60)
        self.assertAlmostEqual(probe_seconds(clip), 4.0417, places=3)
        p = self.new(shots=1)
        p = self.op(p, "import_take", shotId=p.shots[0].id, clip=str(clip))
        self.assertAlmostEqual(p.shots[0].active().measuredSeconds, 4.0417, places=3)


class DepsAndBatchTests(StoreCase):
    def _batch(self, p, gens, **kwargs):
        batch = Batch(self.store, p.id, kwargs.pop("what", "both"), gens.make_still, gens.make_clip, gens.unload,
                      gens.last_frame, **kwargs)
        batch.start(p.revision)
        batch.run()
        return batch, self.store.load(p.id)

    def _film(self):
        """A..F; D continues from C and E from D."""
        p = self.new(shots=6)
        ids = [s.id for s in p.shots]
        for chained in (ids[3], ids[4]):
            p = self.op(p, "edit_shot", shotId=chained, startFrom="previous-shot")
        return p, ids

    def test_batch_orders_by_kind_and_unloads_between(self):
        p, ids = self._film()
        gens = FakeGenerators(self.src)
        batch, p = self._batch(p, gens)
        kinds = [c[0] for c in gens.calls]
        self.assertEqual(kinds, ["still"] * 4 + ["unload"] + ["clip"] * 6)
        self.assertEqual([c[1] for c in gens.calls if c[0] == "clip"], ids)
        self.assertEqual(gens.unloads, ["image"])
        self.assertTrue(all(s.status == "done" and len(s.takes) == 1 for s in p.shots))
        record = p.batches[-1]
        self.assertEqual(record.state, "done")
        self.assertEqual(sum(r.state == "done" for r in record.results), 10)
        self.assertNotIn("/Users/me", json.dumps(p.shots[0].active().settings))

    def test_keep_a_b_e_f_redo_c_and_d(self):
        p, ids = self._film()
        gens = FakeGenerators(self.src)
        _, p = self._batch(p, gens)
        folder = self.folder(p)
        kept = {s.id: (s.activeTake, sha256_file(folder / s.active().clip)) for s in p.shots if s.id not in ids[2:4]}
        p = self.op(p, "mark_redo", shotId=ids[2])
        p = self.op(p, "mark_redo", shotId=ids[3])
        gens.calls.clear()
        _, p = self._batch(p, gens)
        self.assertEqual({c[1] for c in gens.calls if c[0] != "unload"}, {ids[2], ids[3]})
        for shot_id, (take_id, digest) in kept.items():
            shot = p.shot(shot_id)
            self.assertEqual(shot.activeTake, take_id)
            self.assertEqual(sha256_file(folder / shot.active().clip), digest)
        self.assertEqual([len(p.shot(i).takes) for i in ids], [1, 1, 2, 2, 1, 1])
        self.assertEqual(stale_shots(p), {ids[4]})  # E continues from D, which changed
        self.assertEqual([s.status for s in p.shots], ["done"] * 4 + ["stale", "done"])

    def test_repick_and_reference_or_default_changes_mark_stale(self):
        p, ids = self._film()
        gens = FakeGenerators(self.src)
        _, p = self._batch(p, gens, shot_ids=[ids[2], ids[3], ids[4]])
        _, p = self._batch(p, gens, shot_ids=[ids[2]])  # a new take of C: D and E (chained) go stale
        self.assertEqual(stale_shots(p), {ids[3], ids[4]})
        _, p = self._batch(p, gens, shot_ids=[ids[3], ids[4]])
        self.assertEqual(stale_shots(p), set())
        p = self.op(p, "pick_take", shotId=ids[2], takeId=p.shot(ids[2]).takes[0].id)  # re-pick C's first take
        self.assertEqual(stale_shots(p), {ids[3], ids[4]})
        self.assertEqual([s.status for s in p.shots[2:5]], ["done", "stale", "stale"])
        p = self.op(p, "add_reference", kind="character", name="Keeper", description="old man")
        ref = p.references.characters[0].id
        _, p = self._batch(p, gens, shot_ids=ids)
        self.assertEqual(stale_shots(p), set())
        p = self.op(p, "edit_shot", shotId=ids[0], referenceImages=[ref])
        p = self.op(p, "edit_reference", refId=ref, description="young woman")
        self.assertIn(ids[0], stale_shots(p))
        p = self.op(p, "edit_project", defaults={"videoModel": "other"})
        self.assertEqual(stale_shots(p), set(ids))

    def test_locks_skip_refuse_or_force(self):
        p = self.new(shots=2)
        a, b = (s.id for s in p.shots)
        p = self.op(p, "lock", shotId=a)
        gens = FakeGenerators(self.src)
        _, p = self._batch(p, gens, what="still")
        self.assertEqual(gens.calls, [("still", b)])
        with self.assertRaises(BatchRefused):
            Batch(self.store, p.id, "still", gens.make_still, gens.make_clip, shot_ids=[a]).start(p.revision)
        _, p = self._batch(p, gens, what="still", shot_ids=[a], force=True)
        self.assertEqual(len(p.shot(a).takes), 1)
        with self.assertRaises(BatchRefused):  # nothing left that needs work
            Batch(self.store, p.id, "still", gens.make_still, gens.make_clip).start(p.revision)

    def test_one_cancel_stops_the_batch_and_leaves_others_alone(self):
        from backend_service.helpers.job_queue import HEAVY_JOBS

        p = self.new(shots=3)
        gens = FakeGenerators(self.src)
        started = threading.Event()
        other = HEAVY_JOBS.enqueue("image", "ui", "studio job")
        self.addCleanup(HEAVY_JOBS.release, other)
        batch = Batch(self.store, p.id, "still", gens.make_still, gens.make_clip)

        def waiting_still(shot, project):
            started.set()
            ticket = HEAVY_JOBS.enqueue("image", "storyboard", shot.id, batch=batch.id)
            try:
                HEAVY_JOBS.wait_turn(ticket)  # the studio job holds the queue head
            finally:
                HEAVY_JOBS.release(ticket)
            raise AssertionError("should have been cancelled")

        batch.make_still = waiting_still
        batch.start(p.revision)
        thread = threading.Thread(target=batch.run)
        thread.start()
        self.assertTrue(started.wait(5))
        time.sleep(0.1)
        self.assertTrue(batch.cancel())
        thread.join(5)
        self.assertIn(other, HEAVY_JOBS._waiting)  # the Studio's waiting job is untouched
        p = self.store.load(p.id)
        self.assertEqual(p.batches[-1].state, "cancelled")
        self.assertEqual([r.state for r in p.batches[-1].results], ["cancelled"] * 3)
        self.assertTrue(all(s.status == "idle" for s in p.shots))

    def test_a_video_model_without_image_input_skips_stills_and_chains_with_a_reason(self):
        p, ids = self._film()  # A..F, D and E continue from the shot before
        p = self.op(p, "edit_shot", shotId=ids[5], startFrom="upload")
        gens = FakeGenerators(self.src)
        reason = "Wan 2.1 cannot start a clip from an image; pick LTX-2.3 or LTX-2.5 in the project defaults."
        batch = Batch(self.store, p.id, "both", gens.make_still, gens.make_clip, gens.unload, gens.last_frame,
                      clip_image_refusal=lambda project: reason)
        with self.assertRaises(BatchRefused) as caught:  # nothing it could do would honour the shots
            batch.start(p.revision)
        self.assertIn("LTX-2.3", str(caught.exception))
        self.assertEqual(gens.calls, [])
        batch = Batch(self.store, p.id, "still", gens.make_still, gens.make_clip, gens.unload, gens.last_frame,
                      clip_image_refusal=lambda project: reason)
        batch.start(p.revision)  # stills alone are fine: no clip will ignore them
        batch.run()
        self.assertEqual([c[0] for c in gens.calls], ["still"] * 3)  # A, B, C; D and E chain, F is an upload
        p = self.store.load(p.id)
        gens.calls.clear()
        batch = Batch(self.store, p.id, "clip", gens.make_still, gens.make_clip, gens.unload, gens.last_frame,
                      clip_image_refusal=lambda project: reason, shot_ids=ids[:5], force=True)
        with self.assertRaises(BatchRefused):
            batch.start(p.revision)
        record = self.store.load(p.id)
        self.assertEqual(gens.calls, [])
        self.assertTrue(all(s.status != "queued" for s in record.shots))
        steps = Batch(self.store, p.id, "both", gens.make_still, gens.make_clip,
                      clip_image_refusal=lambda project: reason).plan(record)
        self.assertTrue(all(r == reason for s, k, r in steps if s in ids[:5]), steps)

    def test_studio_wiring_refuses_image_input_for_models_without_it(self):
        from backend_service.storyboard import generators

        p = self.new(shots=1)
        wan = {"id": "wan", "name": "Wan 2.1", "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
        ltx = {"id": "ltx", "name": "LTX-2.3", "repo": "dgrauet/ltx-2.3-mlx-q4"}
        with mock.patch.object(generators, "_variant", return_value=wan), \
                mock.patch("backend_service.mcp_bridge.tools._recommended", return_value={}):
            wiring = generators.studio_generators(mock.Mock(), "b_1", self.folder(p))
            self.assertIn("Wan 2.1 cannot start a clip from an image", wiring["clip_image_refusal"](p))
            with self.assertRaises(RuntimeError) as caught:  # the step itself refuses too, before any model runs
                wiring["make_clip"](p.shots[0], p, None)
            self.assertIn("LTX-2.3", str(caught.exception))
        with mock.patch.object(generators, "_variant", return_value=ltx):
            self.assertIsNone(generators.studio_generators(mock.Mock(), "b_1", self.folder(p))["clip_image_refusal"](p))

    def test_a_failed_step_is_recorded_without_paths(self):
        p = self.new(shots=2)
        gens = FakeGenerators(self.src)

        def broken(shot, project):
            if shot.id == p.shots[0].id:
                raise RuntimeError("model not found at /Users/me/models/x.safetensors")
            return gens.make_still(shot, project)

        batch = Batch(self.store, p.id, "still", broken, gens.make_clip)
        batch.start(p.revision)
        batch.run()
        p = self.store.load(p.id)
        self.assertEqual([s.status for s in p.shots], ["failed", "done"])
        error = p.batches[-1].results[0].error
        self.assertIn("x.safetensors", error)
        self.assertNotIn("/Users/me", error)
        self.assertEqual(p.batches[-1].state, "failed")


class BundleTests(StoreCase):
    def _project_with_media(self):
        p = self.new(shots=1)
        p = self.op(p, "import_take", shotId=p.shots[0].id, clip=str(fake_mp4(self.src / "c.mp4")), still=str(png(self.src / "s.png")))
        return self.op(p, "add_audio", kind="voice", path=str(fake_wav(self.src / "v.wav")))

    def test_export_and_import_round_trip(self):
        p = self._project_with_media()
        path = bundle.export_bundle(self.store, p.id)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            text = zf.read("project.json").decode()
        self.assertEqual(names, {"project.json", p.shots[0].active().clip, p.shots[0].active().still, p.audio[0].source.path})
        self.assertNotIn(str(self.base), text)
        imported = bundle.import_bundle(self.store, path)
        self.assertNotEqual(imported.id, p.id)
        self.assertEqual(imported.shots[0].status, "done")
        take = imported.shots[0].active()
        self.assertEqual(sha256_file(self.folder(imported) / take.clip), take.hashes["clip"])

    def _zip(self, entries, project=None):
        path = self.base / f"bad-{len(list(self.base.glob('bad-*')))}.chaosfilm"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("project.json", project if project is not None else _minimal())
            for name, data in entries:
                if isinstance(name, zipfile.ZipInfo):
                    zf.writestr(name, data)
                else:
                    zf.writestr(name, data)
        return path

    def test_hostile_bundles_are_refused_and_leave_nothing(self):
        link = zipfile.ZipInfo("media/link.mp4")
        link.external_attr = 0o120777 << 16
        cases = [
            self._zip([("../evil.txt", b"x")]),
            self._zip([("/abs/evil.txt", b"x")]),
            self._zip([("C:/evil.txt", b"x")]),
            self._zip([("media\\..\\evil.txt", b"x")]),
            self._zip([("cache/x.bin", b"x")]),
            self._zip([(link, b"/etc/passwd")]),
            self._zip([], project=_minimal(schemaVersion=SCHEMA_VERSION + 1)),
            self._zip([], project=b'{"schemaVersion": 1}'),
        ]
        no_project = self.base / "nop.chaosfilm"
        with zipfile.ZipFile(no_project, "w") as zf:
            zf.writestr("media/a.mp4", b"x")
        cases.append(no_project)
        (self.base / "notzip.chaosfilm").write_bytes(b"hello")
        cases.append(self.base / "notzip.chaosfilm")
        for path in cases:
            with self.assertRaises(StoreError, msg=path.name):
                bundle.import_bundle(self.store, path)
        self.assertFalse((self.base / "evil.txt").exists())
        self.assertEqual([p.name for p in self.root.iterdir()] if self.root.exists() else [], [])

    def test_size_and_count_caps(self):
        path = self._zip([("media/a.mp4", b"x" * 1000), ("media/b.mp4", b"y")])
        with mock.patch.object(bundle, "MAX_TOTAL_BYTES", 500), self.assertRaises(StoreError):
            bundle.import_bundle(self.store, path)
        with mock.patch.object(bundle, "MAX_FILES", 2), self.assertRaises(StoreError):
            bundle.import_bundle(self.store, path)

    def test_a_file_with_the_wrong_hash_comes_in_missing(self):
        p = self._project_with_media()
        path = bundle.export_bundle(self.store, p.id)
        tampered = self.base / "tampered.chaosfilm"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as out:
            for info in src.infolist():
                data = src.read(info)
                out.writestr(info.filename, b"evil bytes" if info.filename.endswith(".mp4") else data)
        imported = bundle.import_bundle(self.store, tampered)
        self.assertEqual(imported.shots[0].status, "missing")

    def test_moved_folder_still_opens(self):
        p = self._project_with_media()
        moved_root = self.base / "elsewhere"
        moved_root.mkdir()
        os.rename(self.folder(p), moved_root / p.id)
        other = store_module.ProjectStore(moved_root)
        loaded = other.load(p.id)
        self.assertEqual((loaded.shots[0].status, loaded.audio[0].status), ("done", "ok"))


if __name__ == "__main__":
    unittest.main()
