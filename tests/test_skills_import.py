"""D-62: skill import validation (folder and .zip) and the on-disk store. No network, no models."""

import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from backend_service.skills import importer, store
from backend_service.skills.importer import SkillImportError

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples" / "skills"
SKILL_MD = "---\nname: Test Skill\ndescription: Does a test thing.\n---\n\n# Body\n\nStep one.\n"


class SkillsTempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_SKILLS_DIR": str(self.base / "skills")})
        env.start()
        self.addCleanup(env.stop)

    def folder(self, files: dict[str, bytes | str], name="skill") -> Path:
        root = self.base / name
        for rel, data in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data.encode() if isinstance(data, str) else data)
        return root

    def zip(self, entries, name="skill.zip") -> Path:
        path = self.base / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in entries:
                if isinstance(entry, zipfile.ZipInfo):
                    zf.writestr(entry, b"target")
                else:
                    rel, data = entry
                    zf.writestr(rel, data)
        return path


class FolderImportTests(SkillsTempCase):
    def test_sample_skills_import(self):
        general = importer.load(str(EXAMPLES / "plain-summaries"))
        self.assertEqual(general.name, "plain-summaries")
        self.assertIn("plain English", general.description)
        house = importer.load(str(EXAMPLES / "storyboard-house-style"))
        self.assertEqual(sorted(house.files), ["SKILL.md", "palette.md"])

    def test_binaries_and_hidden_files_are_skipped_and_listed(self):
        root = self.folder({"SKILL.md": SKILL_MD, "logo.png": b"\x89PNG\x00\x01", ".DS_Store": b"x",
                            "notes.txt": b"\xff\xfe\x00bad", "scripts/run.sh": "echo hi\n"})
        skill = importer.from_folder(root)
        self.assertEqual(sorted(skill.files), ["SKILL.md", "scripts/run.sh"])
        reasons = {s["path"]: s["reason"] for s in skill.skipped}
        self.assertEqual(reasons["logo.png"], "not a text file")
        self.assertEqual(reasons[".DS_Store"], "hidden file")
        self.assertEqual(reasons["notes.txt"], "not a text file")

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlink_inside_folder_rejects_the_import(self):
        outside = self.base / "secret.txt"
        outside.write_text("secret")
        root = self.folder({"SKILL.md": SKILL_MD})
        os.symlink(outside, root / "link.md")
        with self.assertRaisesRegex(SkillImportError, "symbolic link"):
            importer.from_folder(root)

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlinked_folder_is_rejected(self):
        root = self.folder({"SKILL.md": SKILL_MD})
        link = self.base / "linked"
        os.symlink(root, link)
        with self.assertRaisesRegex(SkillImportError, "symbolic link"):
            importer.from_folder(link)

    def test_oversize_file_rejects(self):
        root = self.folder({"SKILL.md": SKILL_MD, "big.md": "a" * (importer.MAX_FILE_BYTES + 1)})
        with self.assertRaisesRegex(SkillImportError, "larger than"):
            importer.from_folder(root)

    def test_total_size_cap(self):
        chunk = "b" * (importer.MAX_FILE_BYTES - 10)
        files = {"SKILL.md": SKILL_MD, **{f"part{i}.md": chunk for i in range(9)}}
        with self.assertRaisesRegex(SkillImportError, "in all"):
            importer.from_folder(self.folder(files))

    def test_file_count_cap(self):
        files = {"SKILL.md": SKILL_MD, **{f"f{i}.md": "x" for i in range(importer.MAX_FILES)}}
        with self.assertRaisesRegex(SkillImportError, "text files"):
            importer.from_folder(self.folder(files))

    def test_missing_skill_md(self):
        with self.assertRaisesRegex(SkillImportError, "No SKILL.md"):
            importer.from_folder(self.folder({"README.md": "hi"}))

    def test_missing_path(self):
        with self.assertRaisesRegex(SkillImportError, "does not exist"):
            importer.load(str(self.base / "nope"))
        with self.assertRaises(SkillImportError):
            importer.load("")


class FrontMatterTests(unittest.TestCase):
    def test_bad_front_matter(self):
        cases = {
            "# no front matter": "must start with front matter",
            "---\nname: x\n": "no closing",
            "---\nname: [unclosed\n---\n": "not valid YAML",
            "---\n- a\n- b\n---\n": "key: value",
            "---\ndescription: d\n---\n": "needs a 'name'",
            "---\nname: x\n---\n": "needs a 'description'",
            "---\nname: '!!!'\ndescription: d\n---\n": "letters or digits",
            "---\nname: " + "n" * 80 + "\ndescription: d\n---\n": "1 to 64",
            "---\nname: ok\ndescription: " + "d" * 1100 + "\n---\n": "longer than",
        }
        for text, message in cases.items():
            with self.subTest(text=text[:30]), self.assertRaisesRegex(SkillImportError, message):
                importer.parse_skill_md(text)

    def test_folded_description_and_body(self):
        text = "---\nname: Multi\ndescription: >\n  first line\n  second line\nlicense: MIT\n---\nBody here\n"
        self.assertEqual(importer.parse_skill_md(text), ("Multi", "first line second line"))
        self.assertEqual(importer.skill_body(text), "Body here\n")

    def test_slugify(self):
        self.assertEqual(importer.slugify("Storyboard House Style!"), "storyboard-house-style")
        self.assertEqual(importer.slugify("../../etc"), "etc")


class ZipImportTests(SkillsTempCase):
    def test_zip_with_top_folder(self):
        skill = importer.load(str(self.zip([("my-skill/SKILL.md", SKILL_MD), ("my-skill/ref/a.md", "A"),
                                            ("__MACOSX/my-skill/._SKILL.md", b"\x00\x01")])))
        self.assertEqual(sorted(skill.files), ["SKILL.md", "ref/a.md"])
        self.assertTrue(any(s["path"].startswith("__MACOSX") for s in skill.skipped))

    def test_zip_traversal_and_absolute_entries_reject(self):
        for bad in ("../evil.md", "a/../../evil.md", "/etc/passwd", "C:/win.md", "a\\..\\b.md"):
            with self.subTest(bad=bad), self.assertRaises(SkillImportError):
                importer.from_zip(self.zip([("SKILL.md", SKILL_MD), (bad, "x")], name="t.zip"))

    def test_zip_symlink_entry_rejects(self):
        link = zipfile.ZipInfo("link.md")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(SkillImportError, "symbolic link"):
            importer.from_zip(self.zip([("SKILL.md", SKILL_MD), link]))

    def test_zip_bomb_rejects(self):
        bomb = self.zip([("SKILL.md", SKILL_MD), ("bomb.md", "0" * (importer.MAX_FILE_BYTES - 1))])
        with self.assertRaisesRegex(SkillImportError, "zip bomb"):
            importer.from_zip(bomb)

    def test_zip_declared_oversize_rejects(self):
        path = self.zip([("SKILL.md", SKILL_MD), ("big.md", os.urandom(1024).hex() * 200)])
        with self.assertRaisesRegex(SkillImportError, "larger than"):
            importer.from_zip(path)

    def test_zip_lying_header_is_caught_while_reading(self):
        real_infolist = zipfile.ZipFile.infolist

        def small_sizes(zf):
            infos = real_infolist(zf)
            for info in infos:
                if info.filename == "big.md":
                    info.file_size = 10  # header claims 10 bytes
            return infos
        path = self.zip([("SKILL.md", SKILL_MD), ("big.md", os.urandom(1024).hex() * 200)])
        with mock.patch.object(zipfile.ZipFile, "infolist", small_sizes), self.assertRaisesRegex(SkillImportError, "damaged"):
            importer.from_zip(path)

    def test_too_many_entries(self):
        entries = [("SKILL.md", SKILL_MD)] + [(f"x{i}.bin", b"") for i in range(importer.MAX_ENTRIES)]
        with self.assertRaisesRegex(SkillImportError, "entries"):
            importer.from_zip(self.zip(entries))

    def test_not_a_zip(self):
        bad = self.base / "fake.zip"
        bad.write_text("not a zip")
        with self.assertRaisesRegex(SkillImportError, "not a valid"):
            importer.from_zip(bad)

    def test_names_differing_only_in_case_reject(self):
        with self.assertRaisesRegex(SkillImportError, "appears twice"):
            importer.from_zip(self.zip([("SKILL.md", SKILL_MD), ("Notes.md", "a"), ("notes.md", "b")]))

    def test_no_skill_md_in_zip(self):
        with self.assertRaisesRegex(SkillImportError, "No SKILL.md"):
            importer.from_zip(self.zip([("a/README.md", "x"), ("b/SKILL.md", SKILL_MD)]))


class StoreTests(SkillsTempCase):
    def test_save_list_enable_read_delete(self):
        meta = store.save(importer.load(str(EXAMPLES / "storyboard-house-style")), source="/x/storyboard-house-style")
        self.assertEqual(meta["id"], "storyboard-house-style")
        self.assertFalse(meta["enabled"])  # off by default
        self.assertEqual([s["id"] for s in store.list_skills()], ["storyboard-house-style"])
        self.assertTrue(store.set_enabled("storyboard-house-style", True)["enabled"])
        self.assertIn("Teal", store.read_file("storyboard-house-style", "palette.md"))
        for bad in ("../skill.json", "/etc/passwd", "missing.md", "files/../skill.json"):
            with self.subTest(bad=bad), self.assertRaises(KeyError):
                store.read_file("storyboard-house-style", bad)
        self.assertTrue(store.delete("storyboard-house-style"))
        self.assertEqual(store.list_skills(), [])
        self.assertFalse(store.delete("storyboard-house-style"))

    def test_duplicate_id_is_refused_and_leaves_no_staging(self):
        store.save(importer.load(str(EXAMPLES / "plain-summaries")))
        with self.assertRaises(store.SkillExistsError):
            store.save(importer.load(str(EXAMPLES / "plain-summaries")))
        self.assertEqual(sorted(p.name for p in store.skills_dir().iterdir()), ["plain-summaries"])

    def test_bad_ids_never_touch_the_disk(self):
        for bad in ("..", "../x", "A", "-x", "x/y", ""):
            self.assertIsNone(store.get(bad))
            self.assertIsNone(store.set_enabled(bad, True))
            self.assertFalse(store.delete(bad))
            with self.assertRaises(KeyError):
                store.read_file(bad)

    def test_unsafe_path_in_files_is_refused_at_save(self):
        skill = importer.ImportedSkill(name="evil", description="d", files={"SKILL.md": SKILL_MD, "../x.md": "x"})
        with self.assertRaises(SkillImportError):
            store.save(skill)
        self.assertEqual([p.name for p in store.skills_dir().iterdir()], [])
        self.assertFalse((self.base / "x.md").exists())


if __name__ == "__main__":
    unittest.main()
