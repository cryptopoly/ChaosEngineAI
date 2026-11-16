"""The freestyle board (S-M2): card positions with Undo and Redo, the revision check, refusals, a picture dropped
as a new shot in one step, and an old project without a board."""

import json

from backend_service.storyboard.ops import OpError
from backend_service.storyboard.store import RevisionConflict, parse_project
from tests.storyboard_fixtures import StoreCase, png


class BoardOpsTests(StoreCase):
    def test_old_project_without_board_loads(self):
        p = self.new(shots=1)
        data = json.loads((self.folder(p) / "project.json").read_text())
        data.pop("board")
        self.assertEqual(parse_project(json.dumps(data).encode()).board.pos, {})

    def test_place_cards_with_undo_and_redo(self):
        p = self.new(shots=2)
        a, b = (s.id for s in p.shots)
        p = self.op(p, "place_cards", positions={a: {"x": 10.04, "y": -20}, b: {"x": 300, "y": 0}})
        self.assertEqual((p.board.pos[a].x, p.board.pos[a].y), (10.0, -20.0))
        p = self.op(p, "place_cards", positions={a: {"x": 50, "y": 60}})
        self.assertEqual((p.board.pos[a].x, p.board.pos[b].x), (50.0, 300.0))
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.board.pos[a].x, 10.0)
        p = self.store.undo(p.id, p.revision)
        self.assertEqual(p.board.pos, {})
        p = self.store.redo(p.id, p.revision)
        self.assertEqual(p.board.pos[b].x, 300.0)

    def test_reference_cards_can_be_placed(self):
        p = self.new()
        p = self.op(p, "add_reference", kind="character", name="Mara", image=str(png(self.src / "mara.png")))
        ref = p.references.characters[0].id
        p = self.op(p, "place_cards", positions={ref: {"x": 1, "y": 2}})
        self.assertEqual(p.board.pos[ref].y, 2.0)

    def test_unknown_id_and_far_position_refused(self):
        p = self.new(shots=1)
        with self.assertRaises(OpError):
            self.op(p, "place_cards", positions={"s_gone": {"x": 0, "y": 0}})
        with self.assertRaises(OpError):
            self.op(p, "place_cards", positions={p.shots[0].id: {"x": 1e9, "y": 0}})
        with self.assertRaises(OpError):
            self.op(p, "place_cards", positions={})

    def test_stale_revision_is_refused(self):
        from backend_service.storyboard.ops import run_op

        p = self.new(shots=1)
        moved = self.op(p, "place_cards", positions={p.shots[0].id: {"x": 5, "y": 5}})
        with self.assertRaises(RevisionConflict):
            run_op(self.store, p.id, p.revision, "place_cards", {"positions": {p.shots[0].id: {"x": 9, "y": 9}}})
        self.assertEqual(self.store.load(p.id).board.pos[p.shots[0].id].x, moved.board.pos[p.shots[0].id].x)

    def test_positions_of_removed_shots_drop_on_the_next_move(self):
        p = self.new(shots=2)
        a, b = (s.id for s in p.shots)
        p = self.op(p, "place_cards", positions={a: {"x": 1, "y": 1}, b: {"x": 2, "y": 2}})
        p = self.op(p, "remove_shot", shotId=b)
        self.assertIn(b, p.board.pos)  # kept so an Undo of the removal puts the card back where it was
        p = self.op(p, "place_cards", positions={a: {"x": 3, "y": 3}})
        self.assertNotIn(b, p.board.pos)

    def test_picture_as_new_shot_is_one_step(self):
        p = self.new(shots=1)
        first = p.shots[0].id
        p = self.op(p, "board_add_shot", still=str(png(self.src / "pic.png")), title="Dropped", index=0, x=40, y=80)
        self.assertEqual(len(p.shots), 2)
        shot = p.shots[0]
        self.assertEqual((shot.title, shot.status), ("Dropped", "done"))
        self.assertTrue(shot.active().still.startswith(f"media/{shot.id}/"))
        self.assertEqual((p.board.pos[shot.id].x, p.board.pos[shot.id].y), (40.0, 80.0))
        p = self.store.undo(p.id, p.revision)
        self.assertEqual([s.id for s in p.shots], [first])
        self.assertNotIn(shot.id, p.board.pos)

    def test_new_shot_refuses_a_file_that_is_not_a_picture(self):
        p = self.new()
        text = self.src / "notes.png"
        text.write_text("not a picture")
        with self.assertRaises(OpError):
            self.op(p, "board_add_shot", still=str(text))
        self.assertEqual(self.store.load(p.id).shots, [])
