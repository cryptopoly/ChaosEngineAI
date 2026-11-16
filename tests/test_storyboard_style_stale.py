"""A change to the project's style or the brand kit's style switch marks the shots that used the style as stale
(S15), and only those: never a regeneration, never a shot that took no style, never for the logo or caption look."""

from __future__ import annotations

from backend_service.storyboard import deps
from tests.storyboard_fixtures import StoreCase, png


class StyleStaleTests(StoreCase):
    def setUp(self):
        super().setUp()
        p = self.new(shots=3)
        p = self.op(p, "add_reference", kind="style", name="Shop", description="warm wood",
                    images=[str(png(self.src / "s1.png", (120, 60, 20)))])
        self.style_id = p.references.styles[0].id
        # Three generated takes: the style went in as words, as pictures, and not at all.
        for shot, used in zip(p.shots, ("words", "pictures", None)):
            p = self.op(p, "import_take", shotId=shot.id, still=str(png(self.src / f"{shot.id}.png", (9, 9, 9))))
            p = self._generated(p, shot.id, used)
        self.p = p
        self.words, self.pictures, self.plain = (s.id for s in p.shots)

    def _generated(self, p, shot_id, used):
        def change(project, folder):
            shot = project.shot(shot_id)
            take = shot.active()
            take.origin = "batch"
            if used:
                take.settings = {**take.settings, "style": used}
            take.basis = deps.fingerprint(project, shot)

        return self.store.mutate(p.id, p.revision, change, "test")

    def statuses(self, p):
        return {s.id: s.status for s in p.shots}

    def test_takes_start_fresh(self):
        self.assertEqual(set(self.statuses(self.p).values()), {"done"})

    def test_new_style_words_mark_only_the_words_shot(self):
        p = self.op(self.p, "edit_reference", refId=self.style_id, description="cold steel")
        self.assertEqual(self.statuses(p), {self.words: "stale", self.pictures: "done", self.plain: "done"})

    def test_new_style_picture_marks_the_pictures_shot_and_the_palette_the_words_shot(self):
        p = self.op(self.p, "edit_reference", refId=self.style_id, image=str(png(self.src / "s2.png", (10, 200, 10))))
        status = self.statuses(p)
        self.assertEqual(status[self.pictures], "stale")
        self.assertEqual(status[self.words], "stale")  # new pictures, new palette, new words
        self.assertEqual(status[self.plain], "done")

    def test_adding_or_removing_a_style_marks_both(self):
        p = self.op(self.p, "add_reference", kind="style", description="neon")
        self.assertEqual(self.statuses(p), {self.words: "stale", self.pictures: "stale", self.plain: "done"})
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(set(self.statuses(p).values()), {"done"})  # undo brings the shots back

    def test_brand_style_switch_marks_them(self):
        p = self.op(self.p, "set_brand", styleUse="off")
        self.assertEqual(self.statuses(p), {self.words: "stale", self.pictures: "stale", self.plain: "done"})

    def test_logo_and_caption_look_do_not(self):
        p = self.op(self.p, "set_brand", logoCorner="top-left", captions={"colour": "#ff0000", "position": "top"})
        self.assertEqual(set(self.statuses(p).values()), {"done"})

    def test_nothing_is_regenerated(self):
        before = {s.id: (s.activeTake, len(s.takes)) for s in self.p.shots}
        p = self.op(self.p, "edit_reference", refId=self.style_id, description="cold steel")
        self.assertEqual({s.id: (s.activeTake, len(s.takes)) for s in p.shots}, before)
        self.assertEqual(p.batches, [])
