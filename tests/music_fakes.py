"""Shared fakes for the music tests: a fake engine for the real worker, a data folder, a FLAC fixture.

The fake engine is loaded by ``backend_service/music_worker.py`` through ``CHAOSENGINE_MUSIC_ENGINE_MODULE``
exactly like the ACE-Step engine, so the worker's JSON-lines loop, threads, cancel, busy, events and path
rules run for real without ACE-Step. Words in the caption steer it: ``delay=0.2`` (seconds per step),
``stubborn`` (ignores cancel and sits in "decode"), ``fail``, ``escape`` (returns a path outside the data
folder), ``noisy`` (prints to stdout, which must not reach the protocol).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

FAKE_ENGINE = r'''
import os, shutil, sys, time
from pathlib import Path


class Engine:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.model = None

    def unavailable_reason(self):
        return os.environ.get("FAKE_MUSIC_UNAVAILABLE") or None

    def loaded(self):
        return {"model": self.model, "lm": None}

    def load(self, model, lm=None):
        time.sleep(float(os.environ.get("FAKE_MUSIC_LOAD_S") or 0))
        self.model = model
        return {"model": model, "step_hook": True}

    def unload(self):
        self.model = None

    def generate(self, request, hooks):
        caption = str(request.get("caption") or "")
        if self.model != request.get("model"):
            hooks.stage("load")
            self.load(request.get("model"))
        if "noisy" in caption:
            print("library chatter on stdout")
            sys.stdout.write("more chatter\n")
            sys.stdout.flush()
        hooks.stage("encode")
        delay = float(caption.split("delay=")[1].split()[0]) if "delay=" in caption else 0.0
        total = int(request.get("steps") or 8)
        for index in range(total):
            if "stubborn" not in caption:
                hooks.check_cancel()
            time.sleep(delay)
            hooks.step(index + 1, total)
        hooks.stage("decode", 0.1)
        if "stubborn" in caption:
            time.sleep(60)
        if "fail" in caption:
            raise RuntimeError("boom from the fake engine")
        hooks.stage("write")
        out_dir = Path(request["out_dir"])
        if "escape" in caption:
            out_dir = Path(os.path.expanduser("~"))
        tracks = []
        fixture = os.environ.get("FAKE_MUSIC_FLAC")
        for index in range(int(request.get("variations") or 1)):
            path = out_dir / (request["job"] + "_" + str(index) + ".flac")
            if "escape" not in caption:
                if fixture:
                    shutil.copyfile(fixture, path)
                else:
                    path.write_bytes(b"fLaC-not-really")
            tracks.append({"path": str(path), "seed": int(request["seed"]) + index, "sample_rate": 48000,
                           "channels": 2, "duration_s": float(request["seconds"]), "bpm": request.get("bpm"),
                           "key": request.get("key")})
        return {"tracks": tracks, "timings": {"diffusion_s": 0.1, "vae_decode_s": 0.1}}


def create_engine(data_dir):
    return Engine(data_dir)
'''


def make_flac(path: Path, seconds: float = 1.0) -> Path:
    """A real 24-bit 48 kHz stereo FLAC from the pinned ffmpeg."""
    from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=48000",
         "-ac", "2", "-sample_fmt", "s32", "-c:a", "flac", str(path)],
        check=True, capture_output=True, timeout=60,
    )
    return path


class MusicSandbox:
    """A temporary music data folder, the fake engine, and patches that point the music modules at them."""

    def __init__(self, *, flac: bool = False, installed: bool = True) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name).resolve()
        self.data = self.root / "music"
        for name in ("checkpoints", "library", "tmp"):
            (self.data / name).mkdir(parents=True)
        self.engine_file = self.root / "fake_engine.py"
        self.engine_file.write_text(FAKE_ENGINE, encoding="utf-8")
        self.env: dict[str, str] = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "CHAOSENGINE_MUSIC_DATA": str(self.data),
            "CHAOSENGINE_MUSIC_ENGINE_MODULE": str(self.engine_file),
            "CHAOSENGINE_MUSIC_ENGINE_VERSION": "fake123",
        }
        if flac:
            self.env["FAKE_MUSIC_FLAC"] = str(make_flac(self.root / "fixture.flac"))
        self._stack = ExitStack()
        self.installed = installed

    def __enter__(self) -> "MusicSandbox":
        from backend_service import music_env

        self._stack.enter_context(mock.patch.object(music_env, "MUSIC_DATA_DIR", self.data))
        self._stack.enter_context(mock.patch.object(music_env, "is_installed", lambda: self.installed))
        self._stack.enter_context(mock.patch.object(music_env, "venv_python", lambda: Path(sys.executable)))
        self._stack.enter_context(mock.patch.object(music_env, "worker_env", lambda: dict(self.env)))
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()
        self.tmp.cleanup()

    def patch(self, target: Any, name: str, value: Any) -> None:
        self._stack.enter_context(mock.patch.object(target, name, value))
