"""The music worker's JSON-lines protocol (spike 6.3) against a fake engine, and the ACE-Step step hook.

The worker runs as a real subprocess with this interpreter; the engine is the fake from
``tests/music_fakes.py``. The step-hook tests use fake ``acestep`` modules in ``sys.modules``.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_worker_acestep
from tests.music_fakes import MusicSandbox

WORKER = Path(__file__).resolve().parents[1] / "backend_service" / "music_worker.py"


class Proc:
    """The worker process with a line reader that never blocks the test."""

    def __init__(self, env: dict[str, str], cwd: Path) -> None:
        self.p = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1, env=env, cwd=str(cwd))
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stdout:  # type: ignore[union-attr]
            self.lines.put(line)

    def send(self, payload: dict) -> None:
        self.p.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
        self.p.stdin.flush()  # type: ignore[union-attr]

    def next(self, timeout: float = 20.0) -> dict:
        raw = self.lines.get(timeout=timeout)
        return json.loads(raw)  # every stdout line must be JSON

    def until(self, request_id: int, timeout: float = 30.0) -> tuple[list[dict], dict]:
        """Events and the final reply for one request id."""
        events, deadline = [], time.monotonic() + timeout
        while True:
            line = self.next(max(0.1, deadline - time.monotonic()))
            if line.get("id") != request_id:
                continue
            if line.get("event"):
                events.append(line)
                continue
            return events, line

    def close(self) -> str:
        self.p.stdin.close()  # type: ignore[union-attr]
        self.p.wait(timeout=10)
        return self.p.stderr.read()  # type: ignore[union-attr]


def _gen(job: str, data: Path, **extra) -> dict:
    return {"cmd": "generate", "job": job, "engine": "acestep", "model": "acestep-v15-turbo", "task": "text2music",
            "caption": "x", "lyrics": "[Instrumental]", "instrumental": True, "seconds": 30, "seed": 5, "variations": 1,
            "steps": 4, "out_dir": str(data / "tmp" / job), **extra}


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.box = MusicSandbox().__enter__()
        self.proc = Proc(self.box.env, self.box.data)

    def tearDown(self) -> None:
        if self.proc.p.poll() is None:
            self.proc.p.kill()
            self.proc.p.wait(timeout=10)
        self.box.__exit__(None, None, None)

    def test_ready_line_ping_and_status(self) -> None:
        ready = self.proc.next()
        self.assertEqual(ready["id"], None)
        self.assertTrue(ready["ok"] and ready["ready"])
        self.assertEqual(ready["engine_version"], "fake123")
        self.proc.send({"id": 1, "cmd": "ping"})
        self.assertEqual(self.proc.until(1)[1], {"id": 1, "ok": True, "pong": True})
        self.proc.send({"id": 2, "cmd": "status"})
        status = self.proc.until(2)[1]
        self.assertEqual((status["model"], status["busy"]), (None, None))
        self.assertIn("footprint_gb", status)

    def test_generate_events_reply_and_stdout_swap(self) -> None:
        self.proc.next()
        self.proc.send({"id": 3, **_gen("mus_a1", self.box.data, caption="noisy delay=0.01")})
        events, reply = self.proc.until(3)
        stages = [e["stage"] for e in events if e["event"] == "stage"]
        self.assertEqual(stages, ["load", "encode", "diffusion", "decode", "write"])
        steps = [(e["step"], e["total"]) for e in events if e["event"] == "step"]
        self.assertEqual(steps, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertTrue(all(e["job"] == "mus_a1" for e in events))
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["job"], "mus_a1")
        self.assertEqual(reply["tracks"][0]["seed"], 5)
        self.assertIn("peak_footprint_gb", reply)
        self.assertIn("diffusion_s", reply["timings"])
        stderr = self.proc.close()
        self.assertIn("library chatter on stdout", stderr)  # the library's print went to stderr

    def test_busy_cancel_and_status_while_running(self) -> None:
        self.proc.next()
        self.proc.send({"id": 4, **_gen("mus_b1", self.box.data, caption="delay=0.3", steps=20)})
        time.sleep(0.5)
        self.proc.send({"id": 5, **_gen("mus_b2", self.box.data)})
        self.proc.send({"id": 6, "cmd": "status"})
        self.proc.send({"id": 7, "cmd": "cancel", "job": "mus_b1"})
        seen: dict[int, dict] = {}
        deadline = time.monotonic() + 20
        while 4 not in seen and time.monotonic() < deadline:
            line = self.proc.next()
            if not line.get("event"):
                seen[line["id"]] = line
        self.assertEqual(seen[5], {"id": 5, "job": "mus_b2", "ok": False, "error": "busy"})
        self.assertEqual(seen[6]["busy"], "mus_b1")
        self.assertEqual(seen[7], {"id": 7, "ok": True, "job": "mus_b1", "state": "cancelling"})
        self.assertEqual(seen[4], {"id": 4, "job": "mus_b1", "ok": False, "cancelled": True})
        # The worker stays healthy: the next job runs.
        self.proc.send({"id": 8, **_gen("mus_b3", self.box.data)})
        self.assertTrue(self.proc.until(8)[1]["ok"])
        self.proc.send({"id": 9, "cmd": "cancel", "job": "mus_none"})
        self.assertEqual(self.proc.until(9)[1]["state"], "not-running")

    def test_paths_must_be_absolute_and_inside_the_data_folder(self) -> None:
        self.proc.next()
        cases = {
            10: _gen("mus_c1", self.box.data, out_dir="relative/dir"),
            11: _gen("mus_c2", self.box.data, out_dir=str(self.box.root / "elsewhere")),
            12: _gen("mus_c3", self.box.data, out_dir=str(self.box.data / "tmp" / ".." / ".." / "x")),
            13: _gen("mus_c4", self.box.data, reference_audio="/etc/hosts"),
            14: _gen("../evil", self.box.data),
        }
        for request_id, request in cases.items():
            self.proc.send({"id": request_id, **request})
            reply = self.proc.until(request_id)[1]
            self.assertFalse(reply["ok"], request)
            self.assertIn("ValueError", reply["error"])
        self.proc.send({"id": 15, **_gen("mus_c5", self.box.data, caption="escape")})
        reply = self.proc.until(15)[1]
        self.assertFalse(reply["ok"])
        self.assertIn("inside the music data folder", reply["error"])

    def test_validation_and_unknown_command(self) -> None:
        self.proc.next()
        bad = {
            20: _gen("mus_d1", self.box.data, seconds=5),
            21: _gen("mus_d2", self.box.data, variations=3),
            22: _gen("mus_d3", self.box.data, task="cover"),
            23: _gen("mus_d4", self.box.data, seed=None),
            24: {"cmd": "explode"},
        }
        for request_id, request in bad.items():
            self.proc.send({"id": request_id, **request})
            self.assertFalse(self.proc.until(request_id)[1]["ok"])
        self.proc.p.stdin.write("not json\n")  # type: ignore[union-attr]
        self.proc.p.stdin.flush()  # type: ignore[union-attr]
        self.assertFalse(self.proc.next()["ok"])

    def test_engine_failure_and_unload(self) -> None:
        self.proc.next()
        self.proc.send({"id": 30, **_gen("mus_e1", self.box.data, caption="fail")})
        reply = self.proc.until(30)[1]
        self.assertEqual((reply["ok"], reply["job"]), (False, "mus_e1"))
        self.assertIn("boom from the fake engine", reply["error"])
        self.proc.send({"id": 31, "cmd": "load", "engine": "acestep", "model": "acestep-v15-turbo", "lm": None})
        self.assertTrue(self.proc.until(31)[1]["ok"])
        self.proc.send({"id": 32, "cmd": "status"})
        self.assertEqual(self.proc.until(32)[1]["model"], "acestep-v15-turbo")
        self.proc.send({"id": 33, "cmd": "unload"})
        self.assertTrue(self.proc.until(33)[1]["ok"])
        self.proc.send({"id": 34, "cmd": "status"})
        self.assertIsNone(self.proc.until(34)[1]["model"])


class EngineUnavailableTests(unittest.TestCase):
    def test_reports_the_reason_and_keeps_running(self) -> None:
        with MusicSandbox() as box:
            env = {**box.env, "FAKE_MUSIC_UNAVAILABLE": "MLX is missing on this machine."}
            proc = Proc(env, box.data)
            try:
                self.assertTrue(proc.next()["ready"])
                proc.send({"id": 1, "cmd": "load", "engine": "acestep", "model": "m", "lm": None})
                reply = proc.until(1)[1]
                self.assertFalse(reply["ok"])
                self.assertIn("engine unavailable: MLX is missing", reply["error"])
                proc.send({"id": 2, "cmd": "ping"})
                self.assertTrue(proc.until(2)[1]["ok"])
            finally:
                proc.close()

    def test_the_acestep_engine_explains_itself_off_apple_silicon(self) -> None:
        engine = music_worker_acestep.create_engine(Path("/tmp"))
        with mock.patch.object(music_worker_acestep.platform, "system", return_value="Linux"):
            reason = engine.unavailable_reason()
        self.assertIn("Apple Silicon", reason)
        self.assertIn("CUDA", reason)


class WorkerImportTests(unittest.TestCase):
    def test_the_worker_imports_nothing_from_the_backend(self) -> None:
        for name in ("music_worker.py", "music_worker_acestep.py"):
            source = (WORKER.parent / name).read_text(encoding="utf-8")
            self.assertNotIn("backend_service", source.replace('"""', "").split("from __future__")[1], name)

    def test_its_own_folder_is_not_on_sys_path(self) -> None:
        # Running a script puts its folder first on sys.path; the worker must take it off again.
        code = (f"import json, runpy, sys; sys.path.insert(0, {str(WORKER.parent)!r}); "
                f"runpy.run_path({str(WORKER)!r}, run_name='not_main'); "
                "print(json.dumps(sys.path), file=sys.__stderr__)")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertNotIn(str(WORKER.parent), json.loads(result.stderr.strip().splitlines()[-1]))


class StepHookTests(unittest.TestCase):
    """The pinned private name ``acestep.models.mlx.dit_generate.tqdm``, faked."""

    def _fake(self, **attrs) -> types.ModuleType:
        module = types.ModuleType("fake_dit_generate")
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def tearDown(self) -> None:
        music_worker_acestep._ACTIVE.update({"before": None, "after": None})

    def test_wraps_tqdm_and_reports_steps(self) -> None:
        module = self._fake(tqdm=lambda iterable=None, *a, **k: iter(iterable))
        with mock.patch.dict(sys.modules, {"fake_dit_generate": module}):
            self.assertTrue(music_worker_acestep.install_step_hook("fake_dit_generate"))
            self.assertTrue(music_worker_acestep.install_step_hook("fake_dit_generate"))  # idempotent
            seen: list[tuple[int, int]] = []
            music_worker_acestep._ACTIVE.update({"before": None, "after": lambda d, t: seen.append((d, t))})
            self.assertEqual(list(module.tqdm(range(3), desc="MLX DiT diffusion")), [0, 1, 2])
        self.assertEqual(seen, [(1, 3), (2, 3), (3, 3)])

    def test_before_hook_can_cancel(self) -> None:
        module = self._fake(tqdm=lambda iterable=None, *a, **k: iter(iterable))

        def before(index: int, _total: int) -> None:
            if index == 2:
                raise RuntimeError("cancelled")

        with mock.patch.dict(sys.modules, {"fake_dit_generate": module}):
            music_worker_acestep.install_step_hook("fake_dit_generate")
            music_worker_acestep._ACTIVE.update({"before": before, "after": None})
            items = []
            with self.assertRaises(RuntimeError):
                for item in module.tqdm(range(5)):
                    items.append(item)
        self.assertEqual(items, [0, 1])

    def test_unexpected_layouts_degrade_to_stage_only_progress(self) -> None:
        with mock.patch.dict(sys.modules, {"fake_dit_generate": self._fake()}):
            self.assertFalse(music_worker_acestep.install_step_hook("fake_dit_generate"))  # no tqdm attribute
        with mock.patch.dict(sys.modules, {"fake_dit_generate": self._fake(tqdm="not callable")}):
            self.assertFalse(music_worker_acestep.install_step_hook("fake_dit_generate"))
        self.assertFalse(music_worker_acestep.install_step_hook("no.such.module.anywhere"))

    def test_nothing_listening_falls_through_to_the_original(self) -> None:
        calls = []
        module = self._fake(tqdm=lambda iterable=None, *a, **k: calls.append(k) or iter(iterable))
        with mock.patch.dict(sys.modules, {"fake_dit_generate": module}):
            music_worker_acestep.install_step_hook("fake_dit_generate")
            self.assertEqual(list(module.tqdm(range(2), disable=True)), [0, 1])
        self.assertEqual(calls, [{"disable": True}])

    def test_progress_descriptions_map_to_stages(self) -> None:
        stage = music_worker_acestep._stage_for
        self.assertEqual(stage("Preparing inputs..."), "encode")
        self.assertEqual(stage("Generating music (batch size: 1)..."), "diffusion")
        self.assertEqual(stage("Decoding audio..."), "decode")
        self.assertIsNone(stage("Preparing audio data..."))


if __name__ == "__main__":
    unittest.main()
