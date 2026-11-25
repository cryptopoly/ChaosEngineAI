"""Video upscale quality tier (FlashVSR+ Turbo, D0145): the add-on installer's plan and pinned fetches, the worker
request (warm-up, the 128 grid), the worker's segment loop with a fake model and real ffmpeg, the jobs and routes
wiring, and the fit gates. No torch, no weights, no GPU: the real run is the E2E phase ``video_upscale_quality``."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.catalog import licences
from backend_service.catalog import video_upscale_models as catalog
from backend_service.helpers.job_queue import HeavyJobQueue
from backend_service.routes import video_upscale as route
from backend_service.routes.setup import video_upscale_quality as setup_route
from backend_service.video_upscale import jobs, pipeline, quality, quality_addon
from backend_service.video_upscale import quality_worker
from tests.video_upscale_fakes import count_frames, find_ffmpeg, make_clip

FFMPEG = find_ffmpeg()


def hw(kind: str, usable: float, *, unified: bool = False, total: float = 64.0) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, usable_gb=usable, total_gb=total, unified=unified, bucket=f"{kind}-x",
                           label=kind)


class _Home(unittest.TestCase):
    """A throwaway home, so the add-on folder is never the user's."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        patcher = mock.patch.object(Path, "home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)


# ---------------------------------------------------------------------------
# catalogue, licence and fit gates
# ---------------------------------------------------------------------------


class CatalogueTests(unittest.TestCase):
    def test_the_row_is_apache_and_pinned(self):
        row = catalog.find_quality_model("flashvsr-plus-turbo")
        self.assertIsNotNone(row)
        self.assertIs(row["licence"], licences.APACHE_2_0)
        self.assertEqual(row["licence"]["commercial"], "yes")
        self.assertFalse(row["licence"]["gated"])
        self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(row["sizeBytes"], sum(f["sizeBytes"] for f in row["files"]))
        for entry in row["files"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(catalog.hf_file_url(row, entry["path"]).startswith(
                f"https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/{row['revision']}/"))
        self.assertEqual({f["path"] for f in row["files"]},
                         {"turbo/dit.safetensors", "turbo/lq_proj_in.safetensors", "TCDecoder.ckpt"})

    def test_the_fast_list_never_carries_it(self):
        self.assertIsNone(catalog.find_model("flashvsr-plus-turbo"))
        self.assertTrue(catalog.is_quality_model("flashvsr-plus-turbo"))
        self.assertFalse(catalog.is_quality_model(catalog.DEFAULT_MODEL_ID))

    def test_tiers_from_the_spike(self):
        need = catalog.quality_tier_need_gb
        self.assertEqual(need(1280, 1024), 10.0)
        self.assertEqual(need(1024, 1280), 10.0)  # portrait
        self.assertEqual(need(1920, 1080), 12.0)
        self.assertEqual(need(1350, 1080), 12.0)
        self.assertEqual(need(2560, 1440), 20.0)
        self.assertEqual(need(1440, 2560), 20.0)
        self.assertIsNone(need(3840, 2160))  # 4K: not offered
        self.assertIsNone(need(4096, 400))  # a long strip passes the pixel count, not the side

    def test_mac_and_cpu_are_not_offered(self):
        for machine in (hw("apple", 64.0), hw("cpu", 32.0), hw("cuda", 120.0, unified=True)):
            gate = catalog.quality_fit(machine)
            self.assertFalse(gate["offered"])
            self.assertEqual(gate["code"], "notCuda")

    def test_cards_under_12_gb_are_not_offered(self):
        for vram in (8.0, 10.0, 11.0):
            gate = catalog.quality_fit(hw("cuda", vram))
            self.assertFalse(gate["offered"], vram)
            self.assertEqual(gate["code"], "smallCard")
            self.assertIn(f"{vram:.0f} GB", gate["reason"])

    def test_card_size_sets_the_largest_output(self):
        self.assertEqual(catalog.quality_fit(hw("cuda", 11.99))["maxOutput"], [1920, 1088])  # a "12 GB" card
        self.assertEqual(catalog.quality_fit(hw("cuda", 16.0))["maxOutput"], [1920, 1088])
        self.assertEqual(catalog.quality_fit(hw("cuda", 24.0))["maxOutput"], [2560, 1440])
        self.assertEqual(catalog.quality_fit(hw("cuda", 24.0, total=128.0))["cardGb"], 24.0)  # VRAM, not system RAM

    def test_size_refusals_say_why(self):
        gate = catalog.quality_fit(hw("cuda", 16.0), (2560, 1440))
        self.assertTrue(gate["offered"])
        self.assertFalse(gate["sizeOk"])
        self.assertEqual(gate["code"], "sizeNeedsMore")
        gate = catalog.quality_fit(hw("cuda", 24.0), (3840, 2160))
        self.assertEqual(gate["code"], "tooBig")
        self.assertIn("2560x1440", gate["reason"])
        self.assertTrue(catalog.quality_fit(hw("cuda", 24.0), (1920, 1080))["sizeOk"])

    def test_catalog_with_fit_marks_a_small_card(self):
        rows = catalog.quality_catalog_with_fit(hw("cuda", 10.0))
        self.assertEqual(rows[0]["fit"]["status"], "wont-fit")
        self.assertEqual(rows[0]["qualityFit"]["code"], "smallCard")
        rows = catalog.quality_catalog_with_fit(hw("cuda", 24.0))
        self.assertEqual(rows[0]["fit"]["status"], "fits")
        self.assertEqual(rows[0]["fit"]["basis"], "measured")


# ---------------------------------------------------------------------------
# the installer
# ---------------------------------------------------------------------------


class PlanTests(_Home):
    def test_plan_order_and_pins(self):
        plan = quality_addon.install_plan("py")
        self.assertEqual([step["phase"] for step in plan], ["venv", "packages", "code", "weights", "check"])
        pip = plan[1]["command"]
        self.assertIn("--only-binary=:all:", pip)
        self.assertEqual(pip[pip.index("--extra-index-url") + 1], "https://download.pytorch.org/whl/cu128")
        self.assertEqual(Path(pip[pip.index("-r") + 1]), quality_addon.REQUIREMENTS)
        self.assertEqual(plan[2]["url"], f"https://github.com/OpenImagingLab/FlashVSR_Plus/archive/"
                                         f"{quality_addon.FLASHVSR_REF}.zip")
        self.assertTrue(str(plan[2]["target"]).startswith(str(self.home)))
        self.assertEqual(len(plan[3]["files"]), 3)
        self.assertTrue(all(str(f["target"]).startswith(str(self.home)) for f in plan[3]["files"]))
        self.assertIn("-I", plan[4]["command"])

    def test_requirements_pin_torch_and_the_windows_triton(self):
        lines = [line.split("#")[0].strip() for line in quality_addon.REQUIREMENTS.read_text().splitlines()]
        pins = [line for line in lines if line]
        self.assertIn("torch==2.7.1+cu128", pins)
        self.assertIn('triton-windows==3.3.1.post21; sys_platform == "win32"', pins)
        self.assertTrue(all("==" in pin for pin in pins), pins)
        self.assertIn("packaging==26.4", pins)  # safetensors.torch imports it; the venv lacked it on the 4090

    def test_lane(self):
        self.assertEqual(quality_addon.lane("Windows", "AMD64", True), "cuda")
        self.assertEqual(quality_addon.lane("Linux", "x86_64", True), "cuda")
        self.assertEqual(quality_addon.lane("Darwin", "arm64", True), "unsupported")
        self.assertEqual(quality_addon.lane("Windows", "AMD64", False), "unsupported")
        self.assertEqual(quality_addon.lane("Linux", "aarch64", True), "unsupported")

    def test_unsupported_computer_starts_nothing(self):
        with mock.patch.object(quality_addon, "lane", return_value="unsupported"), \
                mock.patch.object(quality_addon, "_JOB", quality_addon._Job()):
            started = threading.Event()
            job = quality_addon.start_install(runner=started.set)
            self.assertEqual(job["phase"], "error")
            self.assertIn("NVIDIA", job["error"])
            self.assertFalse(started.wait(0.2))
        self.assertFalse(quality_addon.addon_dir().exists())

    def test_not_installed_without_a_marker(self):
        self.assertFalse(quality_addon.installed())

    def test_paths_live_under_the_addon_folder(self):
        self.assertEqual(quality_addon.addon_dir(), self.home / ".chaosengine" / "addons" / "video-upscale-quality")
        for value in quality_addon.model_paths().values():
            self.assertTrue(Path(value).is_relative_to(quality_addon.addon_dir()))
        with self.assertRaises(ValueError):
            quality_addon.weight_path({"path": "../evil.bin"})


def _zip(entries: dict[str, bytes], top: str | None = None, link: str | None = None) -> bytes:
    top = top or f"FlashVSR_Plus-{quality_addon.FLASHVSR_REF}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(f"{top}/", b"")
        for name, data in entries.items():
            archive.writestr(f"{top}/{name}", data)
        if link:
            info = zipfile.ZipInfo(f"{top}/{link}")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, b"target")
    return buf.getvalue()


class ArchiveTests(_Home):
    FILES = {"inference/__init__.py": b"x = 1\n", "LICENSE": b"Apache\n"}

    def test_digest_is_order_free_and_content_bound(self):
        self.assertEqual(quality_addon.tree_digest(self.FILES), quality_addon.tree_digest(dict(reversed(self.FILES.items()))))
        self.assertNotEqual(quality_addon.tree_digest(self.FILES), quality_addon.tree_digest({**self.FILES, "LICENSE": b"MIT"}))

    def test_unpack_writes_the_tree_when_the_digest_matches(self):
        data = _zip(self.FILES)
        quality_addon.unpack_code(data, quality_addon.tree_digest(self.FILES))
        self.assertEqual((quality_addon.code_dir() / "inference" / "__init__.py").read_bytes(), b"x = 1\n")
        self.assertTrue(quality_addon.code_installed())

    def test_a_wrong_digest_keeps_nothing(self):
        with self.assertRaises(RuntimeError):
            quality_addon.unpack_code(_zip(self.FILES), "0" * 64)
        self.assertFalse(quality_addon.code_dir().exists())

    def test_hostile_members_are_refused(self):
        for bad in (_zip({"../escape.py": b""}), _zip(self.FILES, top="other"), _zip(self.FILES, link="ln")):
            with self.assertRaises(ValueError):
                quality_addon.archive_members(bad)
        for name in ("/abs", "C:/x", "a\\b"):
            with self.assertRaises(ValueError):
                quality_addon._member_parts(name, "top")

    def test_the_pinned_digest_is_a_sha256(self):
        self.assertRegex(quality_addon.CODE_TREE_SHA256, r"^[0-9a-f]{64}$")


class _Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DownloadTests(_Home):
    DATA = bytes(range(256)) * 12000  # about 3 MB: three of the download's 1 MB blocks

    def setUp(self):
        super().setUp()
        self.sha = hashlib.sha256(self.DATA).hexdigest()
        self.target = self.home / "w" / "file.bin"
        self.seen: list = []

    def opener(self, honour_range=True):
        def fake(request, timeout=None):
            self.seen.append(dict(request.header_items()))
            ranged = request.get_header("Range")
            if ranged and honour_range:
                start = int(ranged.split("=")[1].rstrip("-"))
                return _Response(self.DATA[start:], 206)
            return _Response(self.DATA, 200)
        return fake

    def test_a_good_file_is_kept(self):
        counts = []
        quality_addon.download_file("https://x/y", self.target, len(self.DATA), self.sha, opener=self.opener(),
                                    on_bytes=counts.append)
        self.assertEqual(self.target.read_bytes(), self.DATA)
        self.assertEqual(counts[-1], len(self.DATA))
        self.assertFalse(self.target.with_name("file.bin.part").exists())

    def test_wrong_size_or_hash_keeps_nothing(self):
        for size, sha in ((len(self.DATA) + 1, self.sha), (len(self.DATA), "0" * 64)):
            with self.assertRaises(RuntimeError):
                quality_addon.download_file("https://x/y", self.target, size, sha, opener=self.opener())
            self.assertFalse(self.target.exists())
            self.assertFalse(self.target.with_name("file.bin.part").exists())

    def test_a_cancelled_part_resumes_with_a_range(self):
        cancel = threading.Event()
        calls = [0]

        def on_bytes(count):
            calls[0] += 1
            if count > 1_500_000:
                cancel.set()

        with self.assertRaises(quality_addon.InstallCancelled):
            quality_addon.download_file("https://x/y", self.target, len(self.DATA), self.sha, opener=self.opener(),
                                        on_bytes=on_bytes, cancel=cancel)
        part = self.target.with_name("file.bin.part")
        have = part.stat().st_size
        self.assertTrue(0 < have < len(self.DATA))
        quality_addon.download_file("https://x/y", self.target, len(self.DATA), self.sha, opener=self.opener())
        self.assertEqual(self.seen[-1].get("Range"), f"bytes={have}-")
        self.assertEqual(self.target.read_bytes(), self.DATA)

    def test_a_server_that_ignores_the_range_starts_again(self):
        part = self.target.with_name("file.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(self.DATA[:1000])
        quality_addon.download_file("https://x/y", self.target, len(self.DATA), self.sha,
                                    opener=self.opener(honour_range=False))
        self.assertEqual(self.target.read_bytes(), self.DATA)

    def test_a_copied_in_file_is_hashed_not_downloaded(self):
        entry = {"path": "TCDecoder.ckpt", "sizeBytes": len(self.DATA), "sha256": self.sha}
        path = quality_addon.weight_path(entry)
        path.parent.mkdir(parents=True)
        path.write_bytes(self.DATA)
        self.assertEqual(quality_addon.ensure_weight(entry, opener=self.opener()), "checked")
        self.assertEqual(self.seen, [])
        self.assertEqual(quality_addon.ensure_weight(entry, opener=self.opener()), "kept")  # remembered
        path.write_bytes(b"\0" * len(self.DATA))  # same size, wrong content
        quality_addon._remember_verified("other")
        (quality_addon.addon_dir() / "weights.json").write_text(json.dumps({"verified": []}))
        self.assertEqual(quality_addon.ensure_weight(entry, opener=self.opener()), "downloaded")
        self.assertEqual(path.read_bytes(), self.DATA)


# ---------------------------------------------------------------------------
# the worker request
# ---------------------------------------------------------------------------


class RequestTests(unittest.TestCase):
    def req(self, **kw):
        base = {"ffmpeg": "ffmpeg", "input": "in.mp4", "first_frame": 0, "frames": 100, "last": False, "fps": "25",
                "sar": "1/1", "src_width": 720, "src_height": 576, "width": 1350, "height": 1080,
                "deinterlace": False, "batch": 1, "codec_args": None, "output": "seg.part.mp4", "weights": "w",
                "tile": 0, "model_id": "flashvsr-plus-turbo", "avtb": False}
        return {**base, **kw}

    def test_padded_size(self):
        self.assertEqual(quality.padded_size(1280, 1024), (1280, 1024))
        self.assertEqual(quality.padded_size(1920, 1080), (1920, 1152))
        self.assertEqual(quality.padded_size(1350, 1080), (1408, 1152))

    def test_first_segment_has_no_warm_up(self):
        out = quality.worker_request(self.req(), paths={"dit": "d"})
        self.assertEqual(out["warmup"], 0)
        self.assertEqual(out["decode"][out["decode"].index("-frames:v") + 1], "100")
        self.assertEqual((out["pad_width"], out["pad_height"]), (1408, 1152))
        self.assertTrue(out["color_correction"])

    def test_later_segments_decode_the_warm_up_first(self):
        out = quality.worker_request(self.req(first_frame=200), paths={"dit": "d"})
        self.assertEqual(out["warmup"], quality.WARMUP_FRAMES)
        self.assertEqual(out["frames"], 100)
        self.assertEqual(out["decode"][out["decode"].index("-frames:v") + 1], str(100 + quality.WARMUP_FRAMES))
        # the seek lands on frame 192 (200 - 8), half a frame early
        vf = out["decode"][out["decode"].index("-vf") + 1]
        self.assertIn(f"trim=start={(192 - 0.5) / 25:.6f}", vf)
        self.assertEqual(quality.worker_request(self.req(first_frame=3), paths={})["warmup"], 3)

    def test_the_encoder_gets_the_exact_size_without_scaling(self):
        out = quality.worker_request(self.req(), paths={})
        enc = out["encode"]
        self.assertEqual(enc[enc.index("-s") + 1], "1350x1080")
        self.assertNotIn("scale=", enc[enc.index("-vf") + 1])

    def test_tiny_results_are_refused(self):
        with self.assertRaises(ValueError):
            quality.worker_request(self.req(width=200, height=100), paths={})

    def test_segments_are_capped_by_pixels(self):
        self.assertEqual(quality.segment_frames(1280, 1024, "25"), 192)  # 200 with the warm-up, as in the spike
        self.assertEqual(quality.segment_frames(320, 240, "25"), 250)  # 10 s
        self.assertEqual(quality.segment_frames(2560, 1440, "24"), 58)
        self.assertGreaterEqual(quality.segment_frames(4096, 4096, "24"), quality.MIN_SEGMENT_FRAMES)

    def test_worker_env_drops_the_backend_allocator_switches(self):
        from backend_service.helpers.malloc_release import MIMALLOC_ENV

        with mock.patch.dict(os.environ, MIMALLOC_ENV):
            env = quality._worker_env()
        self.assertFalse(set(MIMALLOC_ENV) & set(env))

    def test_worker_command_runs_isolated_from_the_venv(self):
        command = quality.worker_command()
        self.assertEqual(command[0], str(quality_addon.venv_python()))
        self.assertEqual(command[1], "-I")
        self.assertTrue(Path(command[2]).is_file())

    def test_expected_chunks(self):
        self.assertEqual(quality_worker.FlashVSRModel.expected_chunks(125), 8)
        self.assertEqual(quality_worker.FlashVSRModel.expected_chunks(1), 1)


# ---------------------------------------------------------------------------
# the worker's segment loop with a fake model, and the job around it
# ---------------------------------------------------------------------------


class FakeModel:
    """Nearest-neighbour resize to the target; records what it was given."""

    def __init__(self):
        self.calls: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    def __call__(self, frames, size, padded, progress=None):
        self.calls.append((int(frames.shape[0]), size, padded))
        if progress is not None:
            progress(0.5)
        width, height = size
        ys = np.arange(height) * frames.shape[1] // height
        xs = np.arange(width) * frames.shape[2] // width
        return frames[:, ys][:, :, xs].copy()


def fake_send(model: FakeModel, log: list):
    """Stands in for ``WarmWorker.request``: runs the worker's segment loop in this process."""

    def send(cmd, *, timeout, stall_s, on_event, cancel_check, job, **req):
        log.append(req)

        def step(done, total):
            on_event({"event": "step", "step": done, "total": total})

        def check():
            if cancel_check():
                raise pipeline.Cancelled("cancelled")

        return quality_worker.process_segment(req, model, step, check)

    return send


@unittest.skipUnless(FFMPEG, "ffmpeg is not available")
class SegmentTests(_Home):
    def setUp(self):
        super().setUp()
        self.src = make_clip(FFMPEG, self.home / "in.mp4", seconds=2.0, fps=10)

    def spec(self, name="up.mp4", size=(192, 128), seconds=0.5):
        spec = pipeline.make_spec(FFMPEG, self.src, self.home / name, model_id="flashvsr-plus-turbo",
                                  weights=self.home / "w", size=size, segment_seconds=seconds)
        return spec

    def test_warm_up_frames_go_to_the_model_and_not_to_the_film(self):
        model, log = FakeModel(), []
        result = pipeline.run_job(self.spec(), quality.runner(send=fake_send(model, log)), exe=FFMPEG,
                                  work=self.home / "work")
        self.assertEqual(result["frames"], 20)
        self.assertEqual(count_frames(FFMPEG, self.home / "up.mp4"), (20, 192, 128))
        self.assertEqual([c[0] for c in model.calls], [5, 10, 13, 13])  # 5, then 5 + 5 warm-up, then 5 + 8
        self.assertEqual([r["warmup"] for r in log], [0, 5, 8, 8])
        self.assertEqual(model.calls[0][2], (256, 128))  # padded to the 128 grid

    def test_frames_match_a_single_pass(self):
        """The warm-up drop keeps every frame in place: the film equals the fake model run on the whole clip."""
        model = FakeModel()
        pipeline.run_job(self.spec(), quality.runner(send=fake_send(model, [])), exe=FFMPEG, work=self.home / "work")
        whole = pipeline.run_job(self.spec("one.mp4", seconds=10.0), quality.runner(send=fake_send(FakeModel(), [])),
                                 exe=FFMPEG, work=self.home / "work2")
        self.assertEqual(whole["segments"], 1)
        a = _decode(self.home / "up.mp4", 192, 128)
        b = _decode(self.home / "one.mp4", 192, 128)
        self.assertEqual(a.shape, b.shape)
        self.assertLess(float(np.abs(a.astype(int) - b.astype(int)).mean()), 3.0)  # encoder noise only

    def test_a_model_of_the_wrong_shape_stops_the_segment(self):
        def bad(frames, size, padded, progress=None):
            return np.zeros((frames.shape[0], 10, 10, 3), np.uint8)

        req = quality.worker_request(pipeline.segment_request(self.spec(), FFMPEG, {"index": 0, "start": 0,
                                                                                    "frames": 5, "last": False},
                                                              self.home / "s.mp4"), paths={})
        with self.assertRaises(RuntimeError):
            quality_worker.process_segment(req, bad, lambda *a: None, lambda: None)
        self.assertFalse((self.home / "s.mp4").exists())

    def test_cancel_inside_the_model_removes_the_part(self):
        def cancelling(frames, size, padded, progress=None):
            progress(0.1)
            raise AssertionError("not reached")

        req = quality.worker_request(pipeline.segment_request(self.spec(), FFMPEG, {"index": 0, "start": 0,
                                                                                    "frames": 5, "last": False},
                                                              self.home / "s.mp4"), paths={})

        def check():
            raise pipeline.Cancelled("cancelled")

        calls = [0]

        def check_after_decode():
            calls[0] += 1
            if calls[0] > 2:
                check()

        with self.assertRaises(pipeline.Cancelled):
            quality_worker.process_segment(req, cancelling, lambda *a: None, check_after_decode)
        self.assertFalse((self.home / "s.mp4").exists())

    def test_progress_only_grows(self):
        steps = []
        req = quality.worker_request(pipeline.segment_request(self.spec(), FFMPEG, {"index": 1, "start": 5,
                                                                                    "frames": 5, "last": False},
                                                              self.home / "s.mp4"), paths={})
        quality_worker.process_segment(req, FakeModel(), lambda done, total: steps.append(done), lambda: None)
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(steps[-1], 5)

    def test_jobs_send_quality_rows_to_the_quality_runner_and_resume(self):
        model, log = FakeModel(), []
        fast_used = []
        manager = jobs.UpscaleJobs(runner_factory=lambda: fast_used.append(1),
                                   quality_runner_factory=lambda: quality.runner(send=fake_send(model, log)),
                                   root=lambda: self.home / "cache", queue=HeavyJobQueue())
        spec = self.spec()
        job = manager.start(spec)
        job.thread.join(timeout=120)
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(fast_used, [])
        self.assertEqual(count_frames(FFMPEG, Path(spec.output)), (20, 192, 128))

        # a cancel after the first segment keeps it; resume runs only the rest
        Path(spec.output).unlink()
        cancel_after = [0]
        manager2 = jobs.UpscaleJobs(runner_factory=lambda: None, root=lambda: self.home / "cache2", queue=HeavyJobQueue())

        def cancelling_send(cmd, **kw):
            result = fake_send(model, log)(cmd, **kw)
            cancel_after[0] += 1
            if cancel_after[0] == 1:
                manager2.cancel(job2.id)
            return result

        manager2.quality_runner_factory = lambda: quality.runner(send=cancelling_send)
        job2 = manager2.start(spec)
        job2.thread.join(timeout=120)
        self.assertEqual(job2.state, "cancelled")
        with mock.patch.object(jobs, "check_ready", return_value={}):  # the real check needs the installed add-on
            resumed = manager2.resume(job2.id)
        resumed.thread.join(timeout=120)
        self.assertEqual(resumed.state, "done", resumed.error)
        self.assertEqual(resumed.resumed_segments, 1)
        self.assertEqual(count_frames(FFMPEG, Path(spec.output)), (20, 192, 128))


def _decode(path: Path, width: int, height: int) -> np.ndarray:
    import subprocess

    raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


class RouteTests(_Home):
    def setUp(self):
        super().setUp()
        app = FastAPI()
        app.include_router(route.router)
        app.include_router(setup_route.router)
        self.client = TestClient(app)

    def test_models_route_hides_the_tier_where_it_cannot_run(self):
        with mock.patch.object(quality_addon, "lane", return_value="unsupported"), \
                mock.patch.object(route, "_deinterlace_available", return_value=True):
            body = self.client.get("/api/video/upscale/models").json()
        self.assertEqual(body["qualityModels"], [])
        self.assertNotIn("flashvsr-plus-turbo", [m["id"] for m in body["models"]])

    def test_models_route_lists_it_on_an_nvidia_card(self):
        with mock.patch.object(quality_addon, "lane", return_value="cuda"), \
                mock.patch.object(route, "_deinterlace_available", return_value=True), \
                mock.patch("backend_service.helpers.hardware_class.host_hardware_class", return_value=hw("cuda", 24.0)):
            body = self.client.get("/api/video/upscale/models").json()
        row = body["qualityModels"][0]
        self.assertEqual(row["id"], "flashvsr-plus-turbo")
        self.assertFalse(row["installed"])
        self.assertEqual(row["qualityFit"]["maxOutput"], [2560, 1440])
        self.assertEqual(row["licence"]["id"], "apache-2.0")

    def test_start_refuses_until_installed(self):
        with mock.patch.object(quality_addon, "lane", return_value="cuda"):
            response = self.client.post("/api/video/upscale", json={"inputPath": "x.mp4", "modelId": "flashvsr-plus-turbo"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("not installed", response.json()["detail"])

    def test_start_refuses_on_a_mac(self):
        with mock.patch.object(quality_addon, "lane", return_value="unsupported"):
            response = self.client.post("/api/video/upscale", json={"inputPath": "x.mp4", "modelId": "flashvsr-plus-turbo"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("NVIDIA", response.json()["detail"])

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_start_refuses_4k(self):
        src = make_clip(FFMPEG, self.home / "in.mp4", seconds=0.5, fps=10)
        with mock.patch.object(quality_addon, "lane", return_value="cuda"), \
                mock.patch.object(quality_addon, "installed", return_value=True), \
                mock.patch("backend_service.helpers.hardware_class.host_hardware_class", return_value=hw("cuda", 24.0)):
            response = self.client.post("/api/video/upscale", json={"inputPath": str(src), "modelId": "flashvsr-plus-turbo",
                                                                    "width": 3840, "height": 2560})
        self.assertEqual(response.status_code, 409)
        self.assertIn("at most 2560x1440", response.json()["detail"])

    def test_setup_routes(self):
        with mock.patch.object(quality_addon, "lane", return_value="unsupported"), \
                mock.patch.object(quality_addon, "_JOB", quality_addon._Job()):
            status = self.client.get("/api/setup/video-upscale-quality/status").json()
            self.assertFalse(status["available"])
            self.assertFalse(status["installed"])
            started = self.client.post("/api/setup/install-video-upscale-quality").json()
            self.assertEqual(started["job"]["phase"], "error")
            cancelled = self.client.post("/api/setup/video-upscale-quality/cancel").json()
            self.assertFalse(cancelled["job"]["running"])
        self.assertFalse(quality_addon.addon_dir().exists())


if __name__ == "__main__":
    unittest.main()
