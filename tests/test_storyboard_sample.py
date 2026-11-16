"""The bundled sample film (S12): what ships, how it installs, and that the release staging carries it."""

import re
from pathlib import Path

from backend_service.storyboard import deps, sample
from backend_service.storyboard.models import Project
from tests.storyboard_fixtures import StoreCase
from tests.test_storyboard_routes import StoryboardRouteCase

ROOT = Path(__file__).resolve().parents[1]
MAX_BUNDLE_BYTES = 15 * 1024 * 1024
MAX_FILE_BYTES = 500 * 1024  # the repository's large-file check


def _bundled() -> Project:
    return Project.model_validate_json((sample.FILM / "project.json").read_text(encoding="utf-8"))


class BundledSampleTests(StoreCase):
    def test_four_finished_shots_with_every_file_present_and_small(self):
        project = _bundled()
        self.assertEqual(len(project.shots), 4)
        for shot in project.shots:
            take = shot.active()
            self.assertIsNotNone(take, shot.id)
            self.assertTrue(take.still and take.clip, shot.id)
            self.assertEqual(shot.status, "done")
        self.assertEqual(deps.stale_shots(project), set())
        total = 0
        for rel in sample.files():
            path = sample.FILM / rel
            self.assertTrue(path.is_file(), rel)
            self.assertLessEqual(path.stat().st_size, MAX_FILE_BYTES, rel)
            total += path.stat().st_size
        self.assertLess(total, MAX_BUNDLE_BYTES)

    def test_no_model_is_pinned_so_a_redo_works_on_any_install(self):
        d = _bundled().defaults
        self.assertEqual((d.imageModel, d.finalImageModel, d.videoModel), (None, None, None))
        self.assertEqual((d.width, d.height, d.fps), (854, 480, 24))

    def test_no_history_batches_or_local_paths_ship(self):
        project = _bundled()
        self.assertEqual((project.batches, project.undoStack, project.redoStack), ([], [], []))
        raw = (sample.FILM / "project.json").read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw)
        self.assertNotIn("\\\\", raw)

    def test_info_describes_the_card(self):
        info = sample.info()
        self.assertTrue(info["available"])
        self.assertEqual(info["shots"], 4)
        self.assertTrue(info["poster"])
        self.assertGreater(info["seconds"], 4)


class InstallTests(StoreCase):
    def test_install_copies_an_ordinary_project_with_a_new_id_each_time(self):
        before = {p.name: p.stat().st_mtime for p in sample.FILM.rglob("*") if p.is_file()}
        first = sample.install(self.store, "Mein Leuchtturm")
        second = sample.install(self.store)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.name, "Mein Leuchtturm")
        self.assertEqual(second.name, _bundled().name)
        loaded = self.store.load(first.id)
        self.assertEqual([s.status for s in loaded.shots], ["done"] * 4)  # every file arrived with its hash
        for rel in sample.files():
            self.assertTrue((self.store.folder(first.id) / rel).is_file(), rel)
        self.assertEqual({p.name: p.stat().st_mtime for p in sample.FILM.rglob("*") if p.is_file()}, before)
        self.assertEqual(list(self.root.glob(".sample-*")), [])  # no temp bundle left behind
        # An ordinary project: it can be edited and deleted.
        edited = self.op(loaded, "edit_shot", shotId=loaded.shots[0].id, title="Mine now")
        self.store.delete(first.id, edited.revision)
        self.assertEqual([row["id"] for row in self.store.list()], [second.id])


class SampleRouteTests(StoryboardRouteCase):
    def test_info_install_and_poster(self):
        info = self.get("/sample")
        self.assertTrue(info["available"])
        made = self.post("/sample", {"name": "The Lighthouse"})
        self.assertEqual(made["project"]["name"], "The Lighthouse")
        self.assertEqual(len(made["timeline"]["shots"]), 4)
        listed = self.get("/projects")["projects"]
        self.assertEqual([row["id"] for row in listed], [made["project"]["id"]])
        poster = self.client.get("/api/storyboard/sample/poster", headers={"Authorization": "Bearer app-token"})
        self.assertEqual(poster.status_code, 200)
        self.post("/sample", {"name": "x" * 201}, code=422)


class StagingTests(StoreCase):
    def test_the_release_staging_copies_and_checks_the_sample(self):
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        self.assertIn('const BUNDLED_SAMPLE_FILM = "backend_service/storyboard/sample/film";', source)
        self.assertRegex(source, r'for \(const relativePath of \["backend_service",')
        self.assertIn("verifyBundledSampleFilm();", source)
        self.assertEqual(sample.FILM, ROOT / "backend_service" / "storyboard" / "sample" / "film")

    def test_no_ignore_rule_of_the_staging_drops_a_sample_file(self):
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        body = re.search(r"function shouldIgnorePath\(currentPath\) \{(.*?)\n\}", source, re.S).group(1)
        names = set(re.findall(r'base === "([^"]+)"', body))
        suffixes = re.findall(r'base\.endsWith\("([^"]+)"\)', body)
        prefixes = re.findall(r'base\.startsWith\("([^"]+)"\)', body)
        self.assertIn("__pycache__", names)  # the pattern this test reads is still there
        for rel in ["project.json", *sample.files()]:
            for part in Path("backend_service/storyboard/sample/film", rel).parts:
                self.assertNotIn(part, names, rel)
                self.assertFalse(any(part.endswith(s) for s in suffixes), rel)
                self.assertFalse(any(part.startswith(p) for p in prefixes), rel)
