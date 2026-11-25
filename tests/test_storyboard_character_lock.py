"""Characters kept the same across a film's shots (B1309): ``Reference.inEveryShot`` and ``Shot.castOut``, the one
resolver of a shot's cast, the still request (pictures when the model takes them, words always, a note when words
only), the stale fingerprint, the planner's per-shot ``characters``, the cut plan and the agent tool switches."""

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import backend_service.app  # noqa: F401  (loads the routes package, so routes.images can be patched)
from backend_service.storyboard import cast, cutplan, planner
from backend_service.storyboard.deps import fingerprint
from backend_service.storyboard.generators import reference_paths
from backend_service.storyboard.models import Song
from backend_service.storyboard.ops import OpError
from backend_service.storyboard.store import parse_project
from tests.storyboard_fixtures import StoreCase, png
from tests.test_storyboard_plan import MFLUX, SDCPP, FakeRuntime, fake_state
from tests import test_storyboard_tools as tool_tests


class CastCase(StoreCase):
    def cast_project(self, shots=3):
        p = self.new(shots=shots)
        p = self.op(p, "add_reference", kind="character", name="Mia", description="red-haired woman in a yellow raincoat",
                    images=[str(png(self.src / "mia.png", (200, 40, 40)))], inEveryShot=True)
        p = self.op(p, "add_reference", kind="character", name="Tom", description="tall man, grey beard",
                    images=[str(png(self.src / "tom.png", (40, 40, 200)))])
        p = self.op(p, "add_reference", kind="place", name="Pier", images=[str(png(self.src / "pier.png", (40, 200, 40)))])
        return p

    def ids(self, p):
        return p.references.characters[0].id, p.references.characters[1].id, p.references.places[0].id


class ModelAndOpsTests(CastCase):
    def test_old_project_loads_unchanged(self):
        raw = {"schemaVersion": 3, "revision": 1, "id": "0123456789abcdef", "name": "x",
               "references": {"characters": [{"id": "r_1", "name": "Keeper"}]},
               "shots": [{"id": "s_1", "referenceImages": ["r_1"]}]}
        p = parse_project(json.dumps(raw).encode())
        self.assertFalse(p.references.characters[0].inEveryShot)
        self.assertEqual(p.shots[0].castOut, [])
        self.assertEqual(cast.effective_references(p, p.shots[0]), ["r_1"])

    def test_in_every_shot_is_persisted_undoable_and_for_characters_only(self):
        p = self.cast_project()
        mia, tom, pier = self.ids(p)
        self.assertTrue(self.store.load(p.id).references.characters[0].inEveryShot)
        p = self.op(p, "edit_reference", refId=tom, inEveryShot=True)
        self.assertTrue(p.reference(tom).inEveryShot)
        p = self.store.undo(p.id, p.revision)
        self.assertFalse(p.reference(tom).inEveryShot)
        with self.assertRaises(OpError):
            self.op(p, "edit_reference", refId=pier, inEveryShot=True)
        with self.assertRaises(OpError):
            self.op(p, "add_reference", kind="style", name="Look", inEveryShot=True)

    def test_effective_cast_every_shot_minus_opt_outs_plus_ticks(self):
        p = self.cast_project()
        mia, tom, pier = self.ids(p)
        a, b, c = (s.id for s in p.shots)
        p = self.op(p, "edit_shot", shotId=b, referenceImages=[pier, tom])
        p = self.op(p, "edit_shot", shotId=c, castOut=[mia])
        self.assertEqual(cast.effective_references(p, p.shot(a)), [mia])
        self.assertEqual(cast.effective_references(p, p.shot(b)), [pier, tom, mia])
        self.assertEqual(cast.effective_references(p, p.shot(c)), [])
        self.assertEqual([r.name for r in cast.shot_cast(p, p.shot(b))], ["Tom", "Mia"])
        with self.assertRaises(OpError):  # a place is never in every shot, so it cannot be left out
            self.op(p, "edit_shot", shotId=c, castOut=[pier])

    def test_ticking_and_leaving_out_stay_apart(self):
        p = self.cast_project(shots=1)
        mia, _, _ = self.ids(p)
        shot = p.shots[0].id
        p = self.op(p, "edit_shot", shotId=shot, castOut=[mia])
        p = self.op(p, "edit_shot", shotId=shot, referenceImages=[mia])  # ticked again: no longer left out
        self.assertEqual((p.shots[0].referenceImages, p.shots[0].castOut), ([mia], []))
        p = self.op(p, "edit_shot", shotId=shot, castOut=[mia])  # left out again: unticked
        self.assertEqual((p.shots[0].referenceImages, p.shots[0].castOut), ([], [mia]))
        self.assertEqual(cast.effective_references(p, p.shots[0]), [])

    def test_removing_a_character_clears_it_from_opt_outs(self):
        p = self.cast_project(shots=1)
        mia, _, _ = self.ids(p)
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, castOut=[mia])
        p = self.op(p, "remove_reference", refId=mia)
        self.assertEqual(p.shots[0].castOut, [])

    def test_fingerprint_follows_the_effective_cast_and_keeps_old_ones(self):
        p = self.new(shots=1)
        p = self.op(p, "add_reference", kind="character", name="K", images=[str(png(self.src / "k.png"))])
        k = p.references.characters[0].id
        plain = fingerprint(p, p.shots[0])
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[k])
        ticked = fingerprint(p, p.shots[0])
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[])
        p = self.op(p, "edit_reference", refId=k, inEveryShot=True)
        self.assertEqual(fingerprint(p, p.shots[0]), ticked)  # in every shot = ticked, for the take's inputs
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, castOut=[k])
        self.assertEqual(fingerprint(p, p.shots[0]), plain)


class StillRequestTests(CastCase):
    def test_reference_paths_include_every_shot_characters_first(self):
        p = self.cast_project(shots=1)
        mia, _, pier = self.ids(p)
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[pier])
        folder = self.folder(p)
        paths = reference_paths(p, p.shots[0], folder)
        expected = [p.reference(mia).images[0], p.reference(pier).images[0]]
        self.assertEqual(paths, [str((folder / img.path).resolve()) for img in expected])
        self.assertEqual(reference_paths(p, p.shots[0], folder, 1), paths[:1])
        self.assertEqual(reference_paths(p, p.shots[0], folder, 0), [])

    def test_words_always_pictures_only_with_slots(self):
        p = self.cast_project(shots=1)
        paths, prompt, mode, note = cast.apply_to_still(p, p.shots[0], ["/m.png"], 0, "a pier at dawn")
        self.assertEqual((paths, mode), ([], "words"))
        self.assertEqual(prompt, "a pier at dawn Characters: Mia, red-haired woman in a yellow raincoat.")
        self.assertIn("takes no reference pictures", note)
        self.assertIn("Mia", note)
        paths, prompt2, mode, note = cast.apply_to_still(p, p.shots[0], ["/m.png"], 2, "a pier at dawn")
        self.assertEqual((paths, mode, note, prompt2), (["/m.png"], "pictures", "", prompt))
        again = cast.apply_to_still(p, p.shots[0], [], 0, prompt)[1]
        self.assertEqual(again, prompt)  # the words are not added twice

    def test_no_cast_no_change(self):
        p = self.new(shots=1)
        self.assertEqual(cast.apply_to_still(p, p.shots[0], [], 4, "x"), ([], "x", "none", ""))
        self.assertEqual(cast.cast_mode(p, MFLUX), "none")

    def test_cast_mode_reads_the_still_model(self):
        p = self.cast_project(shots=1)
        self.assertEqual(cast.cast_mode(p, SDCPP), "words")
        with mock.patch("backend_service.image_runtime.mflux_engine.reference_image_limit", return_value=2):
            self.assertEqual(cast.cast_mode(p, MFLUX), "pictures")

    def _make_still(self, p, variant, limit=0):
        from backend_service.storyboard import generators

        sent = {}

        def fake_generate(request, body):
            sent["body"] = body
            return {"artifacts": [{"imagePath": str(png(self.src / "out.png")), "modelName": "M", "seed": 1}]}

        with mock.patch.object(generators, "_variant", return_value=variant), \
                mock.patch("backend_service.mcp_bridge.tools._recommended", return_value={}), \
                mock.patch("backend_service.image_runtime.mflux_engine.reference_image_limit", return_value=limit), \
                mock.patch("backend_service.helpers.job_queue.HEAVY_JOBS.slot", return_value=nullcontext()), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=fake_generate):
            make_still = generators.studio_generators(SimpleNamespace(), "b1", self.folder(p))["make_still"]
            _, info = make_still(p.shots[0], p)
        return sent["body"], info["settings"]

    def test_still_request_on_a_model_without_reference_pictures(self):
        p = self.cast_project(shots=1)
        body, settings = self._make_still(p, SDCPP)
        self.assertIsNone(body.referenceImagePaths)  # the request is not refused for a picture it cannot take
        self.assertIn("Characters: Mia, red-haired woman in a yellow raincoat.", body.prompt)
        self.assertEqual(settings["cast"], "words")
        self.assertIn("as words only", settings["castNote"])

    def test_still_request_on_a_model_with_reference_pictures(self):
        p = self.cast_project(shots=1)
        mia, _, _ = self.ids(p)
        body, settings = self._make_still(p, MFLUX, limit=2)
        self.assertEqual(body.referenceImagePaths, [str((self.folder(p) / p.reference(mia).images[0].path).resolve())])
        self.assertIn("Characters: Mia", body.prompt)
        self.assertEqual(settings["cast"], "pictures")
        self.assertNotIn("castNote", settings)

    def test_the_take_keeps_the_cast_note(self):
        from backend_service.storyboard.batch import _clean_settings

        kept = _clean_settings({"cast": "words", "castNote": "No pictures of Mia yet, so they went in as words only."})
        self.assertEqual(kept["cast"], "words")
        self.assertTrue(kept["castNote"].startswith("No pictures of Mia"))


class PlanAndCutTests(CastCase):
    def test_planner_maps_names_to_ids_and_leaves_out_unknown_ones(self):
        p = self.cast_project(shots=0)
        p = self.op(p, "set_plan", brief="A rainy pier")
        mia, tom, _ = self.ids(p)
        reply = {"shots": [
            {"title": "Arrive", "stillPrompt": "Mia on the pier", "motionPrompt": "walk", "seconds": 4, "characters": ["mia", "Bob"]},
            {"title": "Meet", "stillPrompt": "Tom waves", "motionPrompt": "wave", "seconds": 4, "characters": ["Tom"]},
            {"title": "Sea", "stillPrompt": "Empty sea", "motionPrompt": "waves", "seconds": 4, "characters": []},
            {"title": "End", "stillPrompt": "Rain", "motionPrompt": "rain", "seconds": 4},
        ], "note": "Rain."}
        runtime = FakeRuntime([reply])
        out = planner.propose(fake_state(runtime), p, self.folder(p), p.plan.brief, None, 4)
        shots = out["shots"]
        self.assertEqual((shots[0]["referenceImages"], shots[0]["castOut"]), ([mia], []))
        self.assertEqual((shots[1]["referenceImages"], shots[1]["castOut"]), ([tom], [mia]))  # the plan left Mia out
        self.assertEqual((shots[2]["referenceImages"], shots[2]["castOut"]), ([], [mia]))
        self.assertNotIn("castOut", shots[3])  # no list: the shot keeps the every-shot cast
        self.assertIn("Bob", out["note"])
        call = runtime.calls[0]
        self.assertIn("Mia (in every shot)", call["prompt"])
        self.assertIn("characters", call["system_prompt"])
        self.assertIn("characters", planner.PROPOSE_SCHEMA["properties"]["shots"]["items"]["properties"])
        applied = self.op(p, "apply_plan", mode="replace", shots=shots)
        self.assertEqual([[r.name for r in cast.shot_cast(applied, s)] for s in applied.shots],
                         [["Mia"], ["Tom"], [], ["Mia"]])

    def test_direct_edits_change_a_shots_cast(self):
        p = self.cast_project(shots=2)
        mia, tom, pier = self.ids(p)
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, referenceImages=[pier])
        reply = {"reply": "Tom joins.", "edits": [{"shot": 1, "characters": ["Tom", "Mia"]}],
                 "add": [{"after": 2, "title": "Alone", "stillPrompt": "the sea", "characters": []}]}
        out = planner.direct(fake_state(FakeRuntime([reply])), p, self.folder(p), "add Tom to shot 1")
        self.assertEqual(out["changes"]["edits"][0]["referenceImages"], [pier, tom, mia])
        applied = self.op(p, "plan_changes", **out["changes"])
        self.assertEqual([r.name for r in cast.shot_cast(applied, applied.shots[0])], ["Tom", "Mia"])
        self.assertEqual(cast.shot_cast(applied, applied.shots[2]), [])
        self.assertIn("characters: Mia", planner.shots_text(applied))

    def test_cut_plan_shots_carry_every_shot_characters(self):
        p = self.cast_project(shots=2)
        mia, _, _ = self.ids(p)
        p = self.op(p, "edit_shot", shotId=p.shots[1].id, castOut=[mia])
        p = p.model_copy(update={"song": Song(path="media/song/a.wav", seconds=12.0)})
        plan = cutplan.plan_cuts(p, 4.0)
        self.assertEqual([s["castOut"] for s in plan["shots"]], [[], [mia], []])
        saved = self.store.load(p.id)
        applied = self.op(saved, "apply_plan", mode="replace", shots=plan["shots"])
        self.assertEqual([cast.effective_references(applied, s) for s in applied.shots], [[mia], [], [mia]])


class AgentToolTests(unittest.TestCase):
    # The storyboard tool test harness, borrowed without inheriting (and so re-running) its tests.
    setUp = tool_tests.StoryboardToolTests.setUp
    _inbox = tool_tests.StoryboardToolTests._inbox
    call = tool_tests.StoryboardToolTests.call
    project = tool_tests.StoryboardToolTests.project

    def test_in_every_shot_and_cast_out_through_the_tools(self):
        info = self.project(shots=2)
        pid = info["projectId"]
        pic = str(png(self.src / "mia.png"))
        info = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character", "name": "Mia", "images": [pic], "in_every_shot": True})
        ref = info["refId"]
        self.assertTrue(info["cast"][0]["inEveryShot"])
        self.assertIn("castOut", info["next"])
        shot = info["shots"][1]["id"]
        info = self.call("storyboard_shot", {"action": "edit", "project_id": pid, "revision": info["revision"],
                                             "shot_id": shot, "fields": {"castOut": [ref]}})
        info = self.call("storyboard_project", {"action": "edit_ref", "project_id": pid, "revision": info["revision"],
                                                "ref_id": ref, "in_every_shot": False})
        self.assertNotIn("inEveryShot", info["cast"][0])
        got = self.call("storyboard_project", {"action": "get", "project_id": pid})
        self.assertEqual(got["project"]["shots"][1]["castOut"], [ref])
        self.assertFalse(got["project"]["references"]["characters"][0]["inEveryShot"])


if __name__ == "__main__":
    unittest.main()
