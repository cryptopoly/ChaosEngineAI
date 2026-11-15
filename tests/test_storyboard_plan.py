"""Film from an idea (S8) and the style and brand kit (S15): plan and brand ops with Undo and Redo, the palette
reader, style references as pictures or words, the planner against a fake chat model, and the plan routes."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.storyboard import palette, planner, style
from backend_service.storyboard.models import Project
from backend_service.storyboard.ops import OpError
from backend_service.storyboard.store import parse_project
from tests.storyboard_fixtures import StoreCase, png
from tests.test_storyboard_routes import H, StoryboardRouteCase


def two_tone(path: Path, left=(20, 90, 40), right=(240, 230, 200), size=(80, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, right)
    image.paste(Image.new("RGB", (size[0] // 2, size[1]), left), (0, 0))
    image.save(path, "PNG")
    return path


class FakeRuntime:
    """Stands in for ``state.runtime``: records each call and answers with the next scripted reply."""

    def __init__(self, replies, vision=False):
        self.replies = list(replies)
        self.calls = []
        self.loaded_model = SimpleNamespace(name="Tiny Chat 1B", ref="tiny/chat", visionEnabled=vision)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply if isinstance(reply, str) else json.dumps(reply))


def fake_state(runtime):
    return SimpleNamespace(runtime=runtime, add_log=lambda *a, **k: None)


PROPOSAL = {"shots": [
    {"title": "Steam rises", "stillPrompt": "A teapot on a wooden counter, morning light", "motionPrompt": "steam curls up", "seconds": 4},
    {"title": "The shop", "stillPrompt": "A small tea shop front, green awning", "motionPrompt": "slow push in", "seconds": 6.2},
    {"title": "Pour", "stillPrompt": "Tea poured into a cup", "motionPrompt": "the pour", "seconds": 99},
    {"title": "Smile", "stillPrompt": "A customer smiles\nover the cup", "motionPrompt": "", "seconds": "x"},
], "note": "Morning ritual."}


class PaletteTests(unittest.TestCase):
    def test_two_colours_most_used_first(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pic = two_tone(Path(tmp) / "a.png", size=(90, 30))
            colours = palette.palette_of([pic])
            self.assertEqual(len(colours), 2)
            self.assertTrue(all(c.startswith("#") and len(c) == 7 for c in colours))
            self.assertEqual(palette.palette_of([pic]), colours)  # the same pictures give the same palette
            names = {palette.colour_name(c) for c in colours}
            self.assertIn("green", " ".join(names))

    def test_unreadable_picture_gives_no_colours(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.png"
            bad.write_bytes(b"not a picture")
            self.assertEqual(palette.palette_of([bad]), [])

    def test_colour_names(self):
        self.assertEqual(palette.colour_name("#000000"), "black")
        self.assertEqual(palette.colour_name("#ffffff"), "white")
        self.assertEqual(palette.colour_name("#808080"), "grey")
        self.assertEqual(palette.colour_name("#0a3d1a"), "deep green")
        self.assertEqual(palette.palette_words(["#0a3d1a", "#000000", "#010101"]), "deep green and black")


class PlanOpsTests(StoreCase):
    def test_old_project_without_plan_fields_loads(self):
        p = self.new(shots=1)
        data = json.loads((self.folder(p) / "project.json").read_text())
        for key in ("plan", "brand", "planChat"):
            data.pop(key)
        data["references"].pop("styles")
        loaded = parse_project(json.dumps(data).encode())
        self.assertEqual(loaded.plan.brief, "")
        self.assertEqual(loaded.brand.styleUse, "auto")
        self.assertEqual(loaded.references.styles, [])

    def test_brief_is_an_op_with_undo(self):
        p = self.new()
        p = self.op(p, "set_plan", brief="A one-minute advert for a tea shop", filmSeconds=60)
        self.assertEqual((p.plan.brief, p.plan.filmSeconds), ("A one-minute advert for a tea shop", 60))
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.plan.brief, "")
        p = self.store.redo(p.id, p.revision)
        self.assertEqual(p.plan.filmSeconds, 60)
        p = self.op(p, "set_plan", clearFilmSeconds=True)
        self.assertIsNone(p.plan.filmSeconds)

    def test_apply_plan_is_one_undo_step(self):
        p = self.new(shots=2)
        shots = [{"title": f"t{i}", "stillPrompt": f"p{i}", "motionPrompt": "m", "seconds": 3} for i in range(4)]
        p = self.op(p, "apply_plan", mode="append", shots=shots)
        self.assertEqual(len(p.shots), 6)
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(len(p.shots), 2)
        p = self.store.redo(p.id, p.revision)
        self.assertEqual([s.title for s in p.shots[2:]], ["t0", "t1", "t2", "t3"])
        replaced = self.op(p, "apply_plan", mode="replace", shots=shots[:2])
        self.assertEqual([s.title for s in replaced.shots], ["t0", "t1"])
        back = self.store.undo(replaced.id, replaced.revision)
        self.assertEqual(len(back.shots), 6)

    def test_replace_refuses_a_locked_shot(self):
        p = self.new(shots=1)
        p = self.op(p, "lock", shotId=p.shots[0].id)
        with self.assertRaises(OpError):
            self.op(p, "apply_plan", mode="replace", shots=[{"title": "x", "stillPrompt": "y"}])

    def test_plan_changes_edit_add_remove_in_one_step(self):
        p = self.new(shots=3)
        a, b, c = (s.id for s in p.shots)
        p = self.op(p, "plan_changes", edits=[{"shotId": a, "stillPrompt": "new a", "seconds": 5}],
                    add=[{"after": a, "title": "after a 1", "stillPrompt": "x"}, {"after": a, "title": "after a 2", "stillPrompt": "y"},
                         {"after": None, "title": "first", "stillPrompt": "z"}],
                    remove=[c])
        self.assertEqual([s.title for s in p.shots], ["first", "shot 0", "after a 1", "after a 2", "shot 1"])
        self.assertEqual(p.shot(a).stillPrompt, "new a")
        p = self.store.undo(p.id, p.revision)
        self.assertEqual([s.id for s in p.shots], [a, b, c])
        self.assertEqual(p.shot(a).stillPrompt, "still 0")

    def test_plan_changes_refuses_unknown_and_locked(self):
        p = self.new(shots=2)
        with self.assertRaises(OpError):
            self.op(p, "plan_changes", edits=[{"shotId": "s_nothere", "title": "x"}])
        p = self.op(p, "lock", shotId=p.shots[1].id)
        with self.assertRaises(OpError):
            self.op(p, "plan_changes", remove=[p.shots[1].id])
        with self.assertRaises(OpError):
            self.op(p, "plan_changes")

    def test_revision_check_holds(self):
        from backend_service.storyboard.ops import run_op
        from backend_service.storyboard.store import RevisionConflict

        p = self.new()
        self.op(p, "set_plan", brief="one")
        with self.assertRaises(RevisionConflict):
            run_op(self.store, p.id, p.revision, "set_plan", {"brief": "two"})

    def test_chat_log_survives_undo(self):
        p = self.new()
        p = self.op(p, "set_plan", brief="tea")
        p = self.op(p, "add_plan_message", role="user", text="Make it warmer")
        p = self.op(p, "add_plan_message", role="assistant", text="Warmer light in every shot.")
        p = self.store.undo(p.id, p.revision)  # undoes the brief, not the chat
        self.assertEqual(p.plan.brief, "")
        self.assertEqual([m.role for m in p.planChat], ["user", "assistant"])
        p = self.op(p, "clear_plan_chat")
        self.assertEqual(p.planChat, [])

    def test_style_reference_reads_a_palette_and_rereads_on_new_pictures(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="style", name="Shop look", description="hand-drawn, warm",
                    images=[str(two_tone(self.src / "s.png"))])
        ref = p.references.styles[0]
        self.assertGreaterEqual(len(ref.palette), 2)
        self.assertEqual(p.reference(ref.id).id, ref.id)
        p = self.op(p, "edit_reference", refId=ref.id, palette=[ref.palette[0]])
        self.assertEqual(len(p.references.styles[0].palette), 1)
        p = self.op(p, "edit_reference", refId=ref.id, addImage=str(png(self.src / "red.png", (220, 20, 20))))
        self.assertGreaterEqual(len(p.references.styles[0].palette), 2)  # read again from all pictures
        p = self.op(p, "remove_reference", refId=ref.id)
        self.assertEqual(p.references.styles, [])

    def test_palette_only_on_a_style(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Mia")
        with self.assertRaises(OpError):
            self.op(p, "edit_reference", refId=p.references.characters[0].id, palette=["#ffffff"])

    def test_brand_logo_caption_look_and_undo(self):
        p = self.new()
        p = self.op(p, "set_brand", logo=str(png(self.src / "logo.png", (1, 2, 3))), logoCorner="top-left",
                    captions={"font": "Helvetica", "colour": "#ffcc00", "position": "top"})
        self.assertTrue((self.folder(p) / p.brand.logo.path).is_file())
        self.assertTrue(p.brand.logo.path.startswith("media/brand/"))
        self.assertEqual((p.brand.captions.font, p.brand.captions.colour, p.brand.captions.position), ("Helvetica", "#ffcc00", "top"))
        p = self.op(p, "set_brand", captions={"colour": "#000000"})
        self.assertEqual(p.brand.captions.font, "Helvetica")  # untouched fields stay
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.brand.captions.colour, "#ffcc00")
        p = self.op(p, "set_brand", clearLogo=True, captions={"clearFont": True})
        self.assertIsNone(p.brand.logo)
        self.assertIsNone(p.brand.captions.font)
        with self.assertRaises(OpError):
            self.op(p, "set_brand", captions={"colour": "red"})

    def test_bundle_carries_style_pictures_and_logo(self):
        from backend_service.storyboard.bundle import referenced

        p = self.new()
        p = self.op(p, "add_reference", kind="style", images=[str(two_tone(self.src / "s.png"))])
        p = self.op(p, "set_brand", logo=str(png(self.src / "logo.png")))
        files = referenced(p)
        self.assertIn(p.references.styles[0].images[0].path, files)
        self.assertIn(p.brand.logo.path, files)


MFLUX = {"id": "flux2", "name": "FLUX.2 Klein", "engine": "mflux", "repo": "black-forest-labs/FLUX.2-klein-4B"}
SDCPP = {"id": "iris", "name": "Iris-3B Turbo", "engine": "sdcpp", "repo": "x/iris"}


class StyleUseTests(StoreCase):
    def styled(self, use="auto"):
        p = self.new(shots=1)
        p = self.op(p, "add_reference", kind="style", description="watercolour shop front",
                    images=[str(two_tone(self.src / "s.png"))])
        if use != "auto":
            p = self.op(p, "set_brand", styleUse=use)
        return p

    def test_words_when_the_model_takes_no_pictures(self):
        p = self.styled()
        paths, prompt, mode = style.apply_to_still(p, p.shots[0], self.folder(p), SDCPP, [], "a teapot")
        self.assertEqual(mode, "words")
        self.assertEqual(paths, [])
        self.assertIn("watercolour shop front", prompt)
        self.assertIn("colour palette of", prompt)
        self.assertTrue(prompt.startswith("a teapot"))

    def test_pictures_when_the_model_takes_them(self):
        p = self.styled()
        with mock.patch("backend_service.image_runtime.mflux_engine.reference_image_limit", return_value=2):
            paths, prompt, mode = style.apply_to_still(p, p.shots[0], self.folder(p), MFLUX, ["/cast.png"], "a teapot")
            self.assertEqual(mode, "pictures")
            self.assertEqual(len(paths), 2)
            self.assertEqual(prompt, "a teapot")
            full = style.apply_to_still(p, p.shots[0], self.folder(p), MFLUX, ["/a.png", "/b.png"], "x")
            self.assertEqual(full[2], "words")  # the cast fills every slot: words instead

    def test_words_setting_and_off(self):
        p = self.styled("words")
        with mock.patch("backend_service.image_runtime.mflux_engine.reference_image_limit", return_value=4):
            self.assertEqual(style.apply_to_still(p, p.shots[0], self.folder(p), MFLUX, [], "x")[2], "words")
        off = self.op(p, "set_brand", styleUse="off")
        self.assertEqual(style.apply_to_still(off, off.shots[0], self.folder(off), SDCPP, [], "x"), ([], "x", "none"))

    def test_text_clip_gets_the_words(self):
        p = self.styled()
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, startFrom="text")
        self.assertIn("watercolour", style.apply_to_motion(p, p.shots[0], "pan left"))
        p = self.op(p, "edit_shot", shotId=p.shots[0].id, startFrom="generate")
        self.assertEqual(style.apply_to_motion(p, p.shots[0], "pan left"), "pan left")

    def test_no_style_no_change(self):
        p = self.new(shots=1)
        self.assertEqual(style.apply_to_still(p, p.shots[0], self.folder(p), SDCPP, [], "x"), ([], "x", "none"))
        self.assertEqual(style.style_mode(p, SDCPP), "none")


class PlannerTests(StoreCase):
    def test_propose_cleans_the_reply(self):
        p = self.op(self.new(), "set_plan", brief="A one-minute advert for a tea shop", filmSeconds=60)
        runtime = FakeRuntime(["<think>hmm</think>```json\n" + json.dumps(PROPOSAL) + "\n```"])
        out = planner.propose(fake_state(runtime), p, self.folder(p), p.plan.brief, 60, None)
        self.assertEqual(len(out["shots"]), 4)
        self.assertEqual(out["shots"][2]["seconds"], 20.0)  # 99 clamped
        self.assertEqual(out["shots"][3]["seconds"], 4.0)  # unreadable: the default
        self.assertEqual(out["shots"][1]["seconds"], 6.0)
        self.assertEqual(out["model"], "Tiny Chat 1B")
        call = runtime.calls[0]
        self.assertIn("tea shop", call["prompt"])
        self.assertIn("about 60", call["prompt"])
        self.assertEqual(call["json_schema"], planner.PROPOSE_SCHEMA)
        self.assertIsNone(call["images"])

    def test_pictures_go_to_a_vision_model(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="style", images=[str(two_tone(self.src / "s.png"))])
        runtime = FakeRuntime([PROPOSAL], vision=True)
        out = planner.propose(fake_state(runtime), p, self.folder(p), "tea", None, 4)
        self.assertEqual(len(runtime.calls[0]["images"]), 1)
        self.assertTrue(out["sawPictures"])
        self.assertIn("exactly 4", runtime.calls[0]["prompt"])
        self.assertIn("palette", runtime.calls[0]["prompt"])

    def test_no_model_no_brief_bad_reply_and_busy(self):
        p = self.new()
        state = fake_state(SimpleNamespace(loaded_model=None))
        with self.assertRaises(planner.PlanError) as ctx:
            planner.propose(state, p, self.folder(p), "tea", None, None)
        self.assertEqual(ctx.exception.code, "no-model")
        with self.assertRaises(planner.PlanError) as ctx:
            planner.propose(fake_state(FakeRuntime([])), p, self.folder(p), "  ", None, None)
        self.assertEqual(ctx.exception.code, "no-brief")
        with self.assertRaises(planner.PlanError) as ctx:
            planner.propose(fake_state(FakeRuntime(["I cannot do that."])), p, self.folder(p), "tea", None, None)
        self.assertEqual(ctx.exception.code, "bad-reply")
        planner._BUSY.acquire()
        try:
            with self.assertRaises(planner.PlanError) as ctx:
                planner.propose(fake_state(FakeRuntime([PROPOSAL])), p, self.folder(p), "tea", None, None)
            self.assertEqual(ctx.exception.code, "busy")
        finally:
            planner._BUSY.release()

    def test_rework_one_shot(self):
        p = self.new(shots=3)
        runtime = FakeRuntime([{"stillPrompt": "A close-up of the cup, golden hour", "why": "Closer and warmer."}])
        out = planner.rework(fake_state(runtime), p, self.folder(p), p.shots[1], "closer and warmer",
                             [{"role": "user", "text": "earlier ask"}, {"role": "assistant", "text": "earlier answer"}])
        self.assertEqual(out["shot"]["stillPrompt"], "A close-up of the cup, golden hour")
        self.assertEqual(out["shot"]["title"], "shot 1")  # kept from the shot
        self.assertEqual(out["why"], "Closer and warmer.")
        self.assertIn("Rework shot 2", runtime.calls[0]["prompt"])
        self.assertEqual(len(runtime.calls[0]["history"]), 2)

    def test_direct_maps_numbers_to_ids_and_skips_locked(self):
        p = self.new(shots=3)
        p = self.op(p, "lock", shotId=p.shots[2].id)
        reply = {"reply": "Warmer, and a closing shot.",
                 "edits": [{"shot": 1, "stillPrompt": "warmer"}, {"shot": 3, "title": "locked"}, {"shot": 9, "title": "nope"}],
                 "add": [{"after": 2, "title": "Logo", "stillPrompt": "the shop sign"}, {"after": 0, "title": "Open", "stillPrompt": "dawn"}],
                 "remove": [2, 3, 42]}
        out = planner.direct(fake_state(FakeRuntime([reply])), p, self.folder(p), "make it warmer")
        ids = [s.id for s in p.shots]
        self.assertEqual(out["changes"]["edits"], [{"shotId": ids[0], "stillPrompt": "warmer"}])
        self.assertEqual(out["changes"]["remove"], [ids[1]])
        self.assertEqual(out["changes"]["add"][0]["after"], ids[0])  # shot 2 is removed: after the shot before it
        self.assertIsNone(out["changes"]["add"][1]["after"])
        applied = self.op(p, "plan_changes", **out["changes"])
        self.assertEqual([s.title for s in applied.shots], ["Open", "shot 0", "Logo", "shot 2"])


class PlanRouteTests(StoryboardRouteCase):
    def state(self):
        return self.client.app.state.chaosengine

    def test_status_without_a_chat_model_and_propose_says_so(self):
        pid, data = self.create(shots=0)
        with mock.patch.object(self.state().runtime, "loaded_model", None):
            status = self.get(f"/projects/{pid}/plan/status")
            self.assertIsNone(status["chatModel"])
            self.assertEqual(status["style"]["mode"], "none")
            out = self.post(f"/projects/{pid}/plan/propose", {"brief": "tea"}, code=409)
            self.assertEqual(out["detail"]["code"], "no-model")

    def test_propose_apply_and_direct(self):
        pid, data = self.create(shots=0)
        runtime = FakeRuntime([PROPOSAL, {"reply": "Shorter.", "edits": [{"shot": 1, "seconds": 2}]}])
        with mock.patch.object(self.state(), "runtime", runtime):
            out = self.post(f"/projects/{pid}/plan/propose", {"brief": "tea shop advert", "filmSeconds": 30})
            self.assertEqual(len(out["shots"]), 4)
            data = self.op(pid, data["project"]["revision"], "apply_plan", {"mode": "append", "shots": out["shots"]})
            self.assertEqual(len(data["project"]["shots"]), 4)
            told = self.post(f"/projects/{pid}/plan/direct", {"message": "make the first shot shorter"})
            self.assertEqual(told["reply"], "Shorter.")
            self.assertEqual([m["role"] for m in told["project"]["planChat"]], ["user", "assistant"])
            first = told["project"]["shots"][0]["id"]
            self.assertEqual(told["changes"]["edits"], [{"shotId": first, "seconds": 2.0}])
            applied = self.op(pid, told["project"]["revision"], "plan_changes", told["changes"])
            self.assertEqual(applied["project"]["shots"][0]["seconds"], 2.0)

    def test_rework_unknown_shot_and_fonts(self):
        pid, _ = self.create(shots=1)
        self.post(f"/projects/{pid}/plan/rework", {"shotId": "s_missing", "message": "x"}, code=404)
        with mock.patch("backend_service.storyboard.fonts.installed_fonts", return_value=["Avenir", "Georgia"]):
            self.assertEqual(self.get("/fonts")["fonts"], ["Avenir", "Georgia"])


class FontTests(unittest.TestCase):
    def test_reads_families_from_a_folder(self):
        import tempfile

        from backend_service.storyboard import fonts

        found = fonts.font_dirs()
        self.assertTrue(found)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "junk.ttf").write_bytes(b"nope")
            self.assertEqual(fonts.installed_fonts([Path(tmp)]), [])


if __name__ == "__main__":
    unittest.main()
