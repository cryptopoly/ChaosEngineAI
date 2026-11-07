"""The project's Film list: finished films in exports/, newest first, no paths, nothing unfinished."""

import os

from tests.test_storyboard_routes import StoryboardRouteCase


class FilmListTests(StoryboardRouteCase):
    def test_lists_only_finished_films_newest_first(self):
        pid, _ = self.create(shots=0)
        exports = self.base / "storyboards" / pid / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        (exports / "old-film.mp4").write_bytes(b"a")
        (exports / "new-film.mp4").write_bytes(b"bb")
        (exports / "notes.srt").write_text("x")
        (exports / "share.chaosfilm").write_bytes(b"z")
        (exports / ".new-film.partial-abcdef.mp4").write_bytes(b"p")
        os.utime(exports / "old-film.mp4", (1_000_000, 1_000_000))
        os.utime(exports / "new-film.mp4", (2_000_000, 2_000_000))
        films = self.get(f"/projects/{pid}/films")["films"]
        self.assertEqual([f["file"] for f in films], ["exports/new-film.mp4", "exports/old-film.mp4"])
        self.assertEqual(films[0]["bytes"], 2)
        self.assertNotIn(str(self.base), str(films))

    def test_empty_when_nothing_was_exported_and_404_for_unknown_project(self):
        pid, _ = self.create(shots=0)
        self.assertEqual(self.get(f"/projects/{pid}/films"), {"films": []})
        self.get("/projects/nope-nope/films", code=404)

    def test_listed_file_is_served_by_the_existing_files_route(self):
        pid, _ = self.create(shots=0)
        exports = self.base / "storyboards" / pid / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        (exports / "film.mp4").write_bytes(b"data")
        rel = self.get(f"/projects/{pid}/films")["films"][0]["file"]
        response = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": rel}, headers={"Authorization": "Bearer app-token"})
        self.assertEqual(response.content, b"data")
