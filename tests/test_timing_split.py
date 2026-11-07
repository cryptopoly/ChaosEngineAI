"""C1 timing split: the progress tracker's phase seconds, the artifact field and the output metadata."""

from __future__ import annotations

import time
import unittest
from unittest import mock

from backend_service import progress
from backend_service.helpers.image_artifacts import _hydrate_image_artifact
from backend_service.helpers.output_metadata import embed_png_settings, generation_settings, read_png_settings
from backend_service.helpers.video_artifacts import _hydrate_video_artifact
from backend_service.progress import (
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    ProgressTracker,
)

def _png() -> bytes:
    """A valid 1x1 RGB PNG."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b"")


_PNG = _png()


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TrackerTests(unittest.TestCase):
    def test_each_phase_adds_up_and_total_covers_the_run(self) -> None:
        clock = _Clock()
        tracker = ProgressTracker(kind="image")
        with mock.patch.object(progress.time, "time", clock):
            tracker.begin(phase=PHASE_LOADING)
            clock.now += 4.0
            tracker.set_phase(PHASE_ENCODING)
            clock.now += 1.0
            tracker.set_phase(PHASE_DIFFUSING)
            clock.now += 10.0
            tracker.set_phase(PHASE_DECODING)
            clock.now += 2.0
            tracker.finish()
        self.assertEqual(tracker.last_timings(), {
            "loadSeconds": 4.0, "encodeSeconds": 1.0, "denoiseSeconds": 10.0, "decodeSeconds": 2.0, "totalSeconds": 17.0,
        })

    def test_a_phase_entered_twice_accumulates(self) -> None:
        clock = _Clock()
        tracker = ProgressTracker(kind="video")
        with mock.patch.object(progress.time, "time", clock):
            tracker.begin(phase=PHASE_DIFFUSING)
            clock.now += 3.0
            tracker.set_phase(PHASE_DIFFUSING)  # a second stage restarts the bar
            clock.now += 2.0
            tracker.finish()
        self.assertEqual(tracker.last_timings()["denoiseSeconds"], 5.0)

    def test_nothing_before_a_first_run(self) -> None:
        self.assertIsNone(ProgressTracker(kind="image").last_timings())

    def test_a_second_finish_keeps_the_split(self) -> None:
        tracker = ProgressTracker(kind="image")
        tracker.begin()
        tracker.finish()
        first = tracker.last_timings()
        tracker.finish()
        self.assertEqual(tracker.last_timings(), first)


class AppTimingsTests(unittest.TestCase):
    def test_only_a_run_that_just_finished_counts(self) -> None:
        from backend_service.app import _last_timings

        progress.IMAGE_PROGRESS.begin()
        progress.IMAGE_PROGRESS.finish()
        self.assertIsNotNone(_last_timings("image"))
        with mock.patch.object(progress.IMAGE_PROGRESS, "snapshot", return_value={"active": False, "updatedAt": time.time() - 120}):
            self.assertIsNone(_last_timings("image"))


class ArtifactTests(unittest.TestCase):
    TIMINGS = {"loadSeconds": 1.0, "denoiseSeconds": 5.0, "totalSeconds": 6.5, "warm": True}

    def test_artifacts_carry_the_split(self) -> None:
        self.assertEqual(_hydrate_image_artifact({"timings": self.TIMINGS})["timings"], self.TIMINGS)
        self.assertEqual(_hydrate_video_artifact({"timings": self.TIMINGS})["timings"], self.TIMINGS)
        self.assertIsNone(_hydrate_image_artifact({})["timings"])

    def test_the_split_goes_into_the_file_metadata(self) -> None:
        settings = generation_settings({"prompt": "p", "seed": 1, "timings": self.TIMINGS,
                                        "referenceImagePaths": ["/Users/someone/ref.png"]})
        self.assertEqual(settings["timings"], self.TIMINGS)
        self.assertEqual(settings["referenceImagePaths"], ["ref.png"])
        self.assertEqual(read_png_settings(embed_png_settings(_PNG, settings))["timings"], self.TIMINGS)


if __name__ == "__main__":
    unittest.main()
