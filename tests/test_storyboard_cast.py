"""Cast pictures (D-65): up to four pictures per character or place, the format 2 -> 3 migration, bundles that
older apps can still read, shot stills as pictures, the picture upload route and the agent cast actions."""

import json
import zipfile

from backend_service.storyboard import bundle
from backend_service.storyboard.deps import fingerprint
from backend_service.storyboard.generators import reference_paths
from backend_service.storyboard.models import Project
from backend_service.storyboard.ops import OpError
from backend_service.storyboard.store import migrate, parse_project
from tests.storyboard_fixtures import StoreCase, png
from tests.test_storyboard_routes import H, StoryboardRouteCase
import unittest

from tests import test_storyboard_tools as tool_tests


def _v2_project(image=None, digest=None):
    ref = {"id": "r_1", "name": "Keeper", "description": "old man", "image": image, "hash": digest}
    return {"schemaVersion": 2, "revision": 1, "id": "0123456789abcdef", "name": "x",
            "references": {"characters": [ref], "places": []}}


class CastPictureTests(StoreCase):
    def pic(self, name, color=(10, 20, 30)):
        return str(png(self.src / name, color))

    def test_format_2_reference_reads_as_a_list_of_one(self):
        digest = "a" * 64
        migrated = migrate(_v2_project("media/refs/r_1.png", digest))
        ref = migrated["references"]["characters"][0]
        self.assertEqual(migrated["schemaVersion"], 3)
        self.assertEqual(ref["images"], [{"path": "media/refs/r_1.png", "hash": digest}])
        self.assertNotIn("image", ref)
        empty = parse_project(json.dumps(_v2_project()).encode())
        self.assertEqual(empty.references.characters[0].images, [])

    def test_add_with_pictures_append_remove_and_the_limit(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Mia", images=[self.pic("a.png"), self.pic("b.png", (1, 2, 3))])
        ref = p.references.characters[0]
        self.assertEqual(len(ref.images), 2)
        self.assertTrue(all((self.folder(p) / img.path).is_file() and img.hash for img in ref.images))
        p = self.op(p, "edit_reference", refId=ref.id, addImage=self.pic("c.png", (4, 5, 6)))
        p = self.op(p, "edit_reference", refId=ref.id, addImage=self.pic("d.png", (7, 8, 9)))
        with self.assertRaises(OpError):
            self.op(p, "edit_reference", refId=ref.id, addImage=self.pic("e.png", (9, 9, 9)))
        with self.assertRaises(OpError):  # five at once is refused by the argument model
            self.op(p, "add_reference", kind="place", images=[self.pic(f"x{i}.png") for i in range(5)])
        second = p.references.characters[0].images[1].path
        p = self.op(p, "edit_reference", refId=ref.id, removeImage=0)
        self.assertEqual(len(p.references.characters[0].images), 3)
        self.assertEqual(p.references.characters[0].images[0].path, second)
        p = self.op(p, "edit_reference", refId=ref.id, image=self.pic("f.png", (3, 3, 3)))  # format 2 meaning: replace all
        self.assertEqual(len(p.references.characters[0].images), 1)
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(len(p.references.characters[0].images), 3)

    def test_a_project_can_have_no_cast(self):
        p = self.new(shots=2)
        self.assertEqual(p.references.characters + p.references.places, [])
        self.assertEqual(reference_paths(p, p.shots[0], self.folder(p)), [])

    def test_shot_still_becomes_a_picture(self):
        p = self.new(shots=1)
        p = self.op(p, "import_take", shotId=p.shots[0].id, still=self.pic("still.png", (50, 60, 70)))
        still = p.shots[0].active().still
        p = self.op(p, "add_reference", kind="character", name="From still", images=[still])
        img = p.references.characters[0].images[0]
        self.assertTrue(img.path.startswith("media/refs/"))
        self.assertEqual(img.hash, p.shots[0].active().hashes["still"])
        with self.assertRaises(OpError):
            self.op(p, "add_reference", kind="place", images=["media/../../etc/passwd.png"])

    def test_one_picture_keeps_the_format_2_fingerprint(self):
        p = self.new(shots=1)
        p = self.op(p, "add_reference", kind="character", name="K", image=self.pic("k.png"))
        ref = p.references.characters[0]
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[ref.id])
        before = fingerprint(p, p.shots[0])
        old = dict(json.loads(p.model_dump_json()))
        old["schemaVersion"] = 2
        old["references"]["characters"][0] = {"id": ref.id, "name": "K", "description": "", "image": ref.images[0].path,
                                              "hash": ref.images[0].hash}
        self.assertEqual(fingerprint(Project.model_validate(migrate(old)), p.shots[0]), before)
        p = self.op(p, "edit_reference", refId=ref.id, addImage=self.pic("k2.png", (1, 1, 1)))
        self.assertNotEqual(fingerprint(p, p.shots[0]), before)  # a new picture marks the shot stale

    def test_reference_paths_take_first_pictures_first_and_stop_at_four(self):
        p = self.new(shots=1)
        p = self.op(p, "add_reference", kind="character", name="A", images=[self.pic(f"a{i}.png", (i, 0, 0)) for i in range(3)])
        p = self.op(p, "add_reference", kind="place", name="B", images=[self.pic(f"b{i}.png", (0, i, 0)) for i in range(3)])
        a, b = p.references.characters[0], p.references.places[0]
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[a.id, b.id])
        paths = reference_paths(p, p.shots[0], self.folder(p))
        expected = [a.images[0], b.images[0], a.images[1], b.images[1]]
        self.assertEqual(paths, [str((self.folder(p) / img.path).resolve()) for img in expected])


class CastBundleTests(StoreCase):
    def _bundle_json(self, path):
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("project.json")), set(zf.namelist())

    def test_one_picture_each_writes_format_2_for_older_apps(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Mia", image=str(png(self.src / "m.png")))
        data, names = self._bundle_json(bundle.export_bundle(self.store, p.id))
        self.assertEqual(data["schemaVersion"], 2)
        ref = data["references"]["characters"][0]
        self.assertEqual(ref["image"], p.references.characters[0].images[0].path)
        self.assertNotIn("images", ref)
        self.assertIn(ref["image"], names)

    def test_bundle_name_keeps_readable_characters(self):
        p = self.store.create("Petal & Stem", None)
        self.assertTrue(bundle.export_bundle(self.store, p.id).name.startswith("Petal-Stem-"))

    def test_several_pictures_write_format_3_and_round_trip(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Mia",
                    images=[str(png(self.src / "m1.png")), str(png(self.src / "m2.png", (5, 5, 5)))])
        path = bundle.export_bundle(self.store, p.id)
        data, names = self._bundle_json(path)
        self.assertEqual(data["schemaVersion"], 3)
        self.assertTrue({img.path for img in p.references.characters[0].images} <= names)
        imported = bundle.import_bundle(self.store, path)
        self.assertEqual([i.hash for i in imported.references.characters[0].images],
                         [i.hash for i in p.references.characters[0].images])

    def test_an_old_single_picture_bundle_imports_as_a_list_of_one(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="place", name="Pier", image=str(png(self.src / "p.png")))
        imported = bundle.import_bundle(self.store, bundle.export_bundle(self.store, p.id))  # written as format 2
        self.assertEqual(len(imported.references.places[0].images), 1)
        self.assertTrue((self.folder(imported) / imported.references.places[0].images[0].path).is_file())


class PictureUploadRouteTests(StoryboardRouteCase):
    def setUp(self):
        super().setUp()
        from unittest import mock

        from backend_service.mcp_bridge import files

        inbox = self.base / "inbox"
        inbox.mkdir()
        patcher = mock.patch.object(files, "inbox_dir", lambda: inbox)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upload_then_add_a_character_with_it(self):
        raw = png(self.src / "u.png").read_bytes()
        response = self.client.post("/api/storyboard/uploads", content=raw, headers=H)
        self.assertEqual(response.status_code, 200, response.text)
        handle = response.json()["handle"]
        self.assertTrue(handle.startswith("upload:"))
        pid, data = self.create(shots=0)
        data = self.op(pid, data["project"]["revision"], "add_reference", {"kind": "character", "images": [handle]})
        self.assertEqual(len(data["project"]["references"]["characters"][0]["images"]), 1)

    def test_upload_refuses_what_is_not_a_picture(self):
        for body in (b"RIFF\x00\x00\x00\x00WAVE" + b"x" * 64, b"hello world, not a picture at all"):
            response = self.client.post("/api/storyboard/uploads", content=body, headers=H)
            self.assertEqual(response.status_code, 400, response.text)

    def test_upload_refuses_an_empty_body_and_one_over_the_cap(self):
        from unittest import mock

        from backend_service.mcp_bridge import files

        self.assertEqual(self.client.post("/api/storyboard/uploads", content=b"", headers=H).status_code, 400)
        raw = png(self.src / "big.png").read_bytes()
        with mock.patch.object(files, "max_upload_bytes", lambda ext: len(raw) - 1):
            response = self.client.post("/api/storyboard/uploads", content=raw, headers=H)
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual([p for p in files.inbox_dir().iterdir() if p.name.startswith(".part")], [])

    def test_thumbnails_are_small_jpegs_and_never_scaled_up(self):
        from PIL import Image

        big = self.src / "big.png"
        Image.new("RGB", (1600, 900), (10, 20, 30)).save(big)
        small = png(self.src / "small.png")  # 16 x 16
        pid, data = self.create(shots=0)
        data = self.op(pid, data["project"]["revision"], "add_reference", {"kind": "place", "images": [str(big), str(small)]})
        paths = [img["path"] for img in data["project"]["references"]["places"][0]["images"]]
        from io import BytesIO

        sizes = []
        for rel in paths:
            response = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": rel, "thumb": 256}, headers=H)
            self.assertEqual((response.status_code, response.headers["content-type"]), (200, "image/jpeg"))
            sizes.append(Image.open(BytesIO(response.content)).size)
        self.assertEqual(sizes, [(455, 256), (16, 16)])
        bad = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": paths[0], "thumb": 300}, headers=H)
        self.assertEqual(bad.status_code, 400)


class StillOnlyBatchTests(StoreCase):
    """A shot with a still and no clip is missing its clip: Make missing makes only the clip, from that still."""

    def test_both_batch_makes_only_the_clip_of_a_still_only_shot(self):
        from backend_service.storyboard.batch import Batch, needs_work
        from tests.storyboard_fixtures import FakeGenerators

        p = self.new(shots=2)
        a, b = (s.id for s in p.shots)
        p = self.op(p, "import_take", shotId=a, still=str(png(self.src / "imported.png")))
        still = p.shot(a).active().still
        self.assertTrue(needs_work(p.shot(a)))
        self.assertFalse(needs_work(p.shot(a), "still"))
        gens = FakeGenerators(self.src)
        batch = Batch(self.store, p.id, "both", gens.make_still, gens.make_clip, gens.unload, gens.last_frame)
        batch.start(p.revision)
        batch.run()
        p = self.store.load(p.id)
        self.assertEqual([c[:2] for c in gens.calls if c[0] != "unload"], [("still", b), ("clip", a), ("clip", b)])
        shot = p.shot(a)
        self.assertEqual(len(shot.takes), 1)  # the clip completed the imported take
        self.assertEqual((shot.active().still, bool(shot.active().clip), shot.status), (still, True, "done"))


class CastEdgeTests(StoreCase):
    def test_a_repeated_source_is_one_picture_and_a_refused_add_leaves_no_file(self):
        p = self.new()
        pic = str(png(self.src / "same.png"))
        p = self.op(p, "add_reference", kind="place", image=pic, images=[pic, pic])
        self.assertEqual(len(p.references.places[0].images), 1)
        refs = self.folder(p) / "media" / "refs"
        before = sorted(refs.iterdir())
        with self.assertRaises(OpError):
            self.op(p, "add_reference", kind="place", images=[str(png(self.src / f"m{i}.png", (i, 2, 2))) for i in range(5)])
        self.assertEqual(sorted(refs.iterdir()), before)

    def test_image_replaces_every_picture_and_remove_out_of_range_is_refused(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", images=[str(png(self.src / f"{i}.png", (i, 1, 1))) for i in range(3)])
        ref = p.references.characters[0].id
        with self.assertRaises(OpError):
            self.op(p, "edit_reference", refId=ref, removeImage=3)
        p = self.op(p, "edit_reference", refId=ref, image=str(png(self.src / "one.png", (9, 9, 9))))
        self.assertEqual(len(p.references.characters[0].images), 1)


class CastToolTests(unittest.TestCase):
    # The tool tests' helpers, without collecting their tests a second time here.
    setUp = tool_tests.StoryboardToolTests.setUp
    _inbox = tool_tests.StoryboardToolTests._inbox
    call = tool_tests.StoryboardToolTests.call
    project = tool_tests.StoryboardToolTests.project

    def test_agent_uses_a_shot_still_as_a_cast_picture(self):
        info = self.project(shots=1)
        pid, sid = info["projectId"], info["shots"][0]["id"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": sid, "path": str(png(self.src / "still.png"))})
        got = self.call("storyboard_project", {"action": "get", "project_id": pid})
        still = got["project"]["shots"][0]["takes"][0]["still"]
        info = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character", "name": "From the shot", "images": [still]})
        self.assertEqual(info["cast"][0]["pictures"], 1)
        text = self.call("storyboard_project", {"action": "edit_ref", "project_id": pid, "revision": info["revision"],
                                                "ref_id": info["refId"], "add_image": "media/../../secret.png"}, error=True)
        self.assertTrue(text)

    def test_agent_adds_edits_and_removes_cast(self):
        info = self.project(shots=1)
        pid = info["projectId"]
        pics = [str(png(self.src / f"c{i}.png", (i, i, i))) for i in range(2)]
        info = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character", "name": "Mia", "images": pics})
        self.assertEqual(info["cast"], [{"id": info["refId"], "kind": "character", "name": "Mia", "pictures": 2}])
        ref = info["refId"]
        info = self.call("storyboard_project", {"action": "edit_ref", "project_id": pid, "revision": info["revision"],
                                                "ref_id": ref, "remove_image": 0})
        self.assertEqual(info["cast"][0]["pictures"], 1)
        got = self.call("storyboard_project", {"action": "get", "project_id": pid})
        self.assertEqual(len(got["project"]["references"]["characters"][0]["images"]), 1)
        text = self.call("storyboard_project", {"action": "edit_ref", "project_id": pid, "revision": info["revision"],
                                                "ref_id": ref, "add_image": "/etc/hosts"}, error=True)
        self.assertIn("output folders", text)
        info = self.call("storyboard_project", {"action": "remove_ref", "project_id": pid, "revision": info["revision"],
                                                "ref_id": ref})
        self.assertEqual(info["cast"], [])


class FormatUpgradeHistoryTests(StoreCase):
    """B1: undo and redo of cast changes saved before format 3 still work after the upgrade."""

    def _to_format_2(self, p):
        from backend_service.storyboard.store import refs_to_single

        folder = self.folder(p)
        data = json.loads((folder / "project.json").read_text())
        (folder / "project.json").write_text(json.dumps(refs_to_single(data)))
        for path in (folder / "history").glob("*.json"):
            entry = json.loads(path.read_text())
            for side in ("before", "after"):
                if "references" in entry[side]:
                    entry[side] = {k: v for k, v in refs_to_single({**entry[side]}).items() if k != "schemaVersion"}
            path.write_text(json.dumps(entry))

    def test_undo_and_redo_of_an_old_cast_change(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Keeper", image=str(png(self.src / "k.png")))
        ref = p.references.characters[0].id
        p = self.op(p, "edit_reference", refId=ref, description="old man")
        self._to_format_2(p)
        loaded = self.store.load(p.id)
        self.assertEqual(len(loaded.references.characters[0].images), 1)
        undone = self.store.undo(p.id, loaded.revision)
        self.assertEqual(undone.references.characters[0].description, "")
        undone = self.store.undo(p.id, undone.revision)  # the add itself
        self.assertEqual(undone.references.characters, [])
        redone = self.store.redo(p.id, undone.revision)
        self.assertEqual(len(redone.references.characters[0].images), 1)
        redone = self.store.redo(p.id, redone.revision)
        self.assertEqual(redone.references.characters[0].description, "old man")
