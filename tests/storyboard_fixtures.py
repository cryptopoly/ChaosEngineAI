"""Shared helpers for the storyboard tests: a temp store, small media files and fake generators."""

import itertools
import tempfile
import unittest
from pathlib import Path

from backend_service.storyboard.ops import run_op
from backend_service.storyboard.store import ProjectStore

_COUNTER = itertools.count(1)


def png(path: Path, color=(200, 30, 30)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, "PNG")
    return path


def fake_mp4(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + f"clip-{next(_COUNTER)}".encode() * 64)
    return path


def fake_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + f"audio-{next(_COUNTER)}".encode() * 64)
    return path


class FakeGenerators:
    """Records calls; makes a fresh file per call so every take has new bytes."""

    def __init__(self, src: Path, seconds: float = 4.0417) -> None:
        self.src, self.seconds, self.calls, self.unloads = src, seconds, [], []

    def make_still(self, shot, project):
        self.calls.append(("still", shot.id))
        n = next(_COUNTER)
        return png(self.src / f"gen-{shot.id}-{n}.png", (n % 255, 10, 10))

    def make_clip(self, shot, project, still):
        self.calls.append(("clip", shot.id, still))
        return fake_mp4(self.src / f"gen-{shot.id}-{next(_COUNTER)}.mp4"), {
            "measuredSeconds": self.seconds, "settings": {"model": "Lightricks/LTX-2.3", "fps": 24, "where": "/Users/me/x"}}

    def unload(self, kind):
        self.calls.append(("unload", kind))
        self.unloads.append(kind)

    def last_frame(self, clip):
        return png(self.src / f"last-{next(_COUNTER)}.png")


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "storyboards"
        self.src = self.base / "src"
        self.src.mkdir()
        self.store = ProjectStore(self.root)

    def new(self, shots=0, **defaults):
        project = self.store.create("Lighthouse", defaults or None)
        for index in range(shots):
            project = self.op(project, "add_shot", title=f"shot {index}", stillPrompt=f"still {index}", motionPrompt="waves")
        return project

    def op(self, project, op_name, **args):
        return run_op(self.store, project.id, project.revision, op_name, args)

    def folder(self, project):
        return self.store.folder(project.id)
