"""Room after the vocal (FU-609): when a vocal ends on a syllable, how much silence the model gets, how the clip is
cut back, how the audio is padded, and that the score never counts that room against a take's mouth or head."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from backend_service import lip_sync_score as S
from backend_service import lipsync_room as room
from backend_service import lipsync_runtime as lip


def _wav(path: Path, sung: list[tuple[float, float]], seconds: float, rate: int = 48000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for a, b in sung:
        x[(t >= a) & (t < b)] = 0.3 * np.sin(2 * np.pi * 330 * t[(t >= a) & (t < b)])
    pcm = (np.stack([x, x], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return path


class EndsVoicedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_a_vocal_sung_to_its_last_frame_ends_voiced(self) -> None:
        self.assertTrue(room.ends_voiced(_wav(self.folder / "a.wav", [(0.4, 4.04)], 4.04)))

    def test_a_syllable_ending_within_0_2_s_of_the_end_counts(self) -> None:
        self.assertTrue(room.ends_voiced(_wav(self.folder / "a.wav", [(0.4, 3.9)], 4.04)))

    def test_silence_after_the_phrase_is_room_already(self) -> None:
        self.assertFalse(room.ends_voiced(_wav(self.folder / "a.wav", [(0.4, 3.5)], 4.04)))
        self.assertFalse(room.ends_voiced(_wav(self.folder / "b.wav", [], 4.04)))

    def test_an_unreadable_file_gets_no_room(self) -> None:
        self.assertFalse(room.ends_voiced(self.folder / "missing.wav"))
        (self.folder / "bad.wav").write_bytes(b"not a wav")
        self.assertFalse(room.ends_voiced(self.folder / "bad.wav"))


class RoomFramesTests(unittest.TestCase):
    def test_about_half_a_second_in_whole_8k_steps(self) -> None:
        self.assertEqual(room.room_frames(97, 24, 193), 113)  # 0.33 or 0.67 s: the tie goes to the longer
        self.assertEqual(room.room_frames(121, 30, 241), 137)  # 0.53 s

    def test_never_past_the_8_s_limit(self) -> None:
        self.assertEqual(room.room_frames(185, 24, 193), 193)  # only the 0.33 s step fits
        self.assertIsNone(room.room_frames(193, 24, 193))

    def test_the_trim_keeps_the_asked_frames_and_drops_the_sound(self) -> None:
        args = room.trim_command("ffmpeg", Path("in.mp4"), Path("out.mp4"), 97, ["-c:v", "libx264"])
        self.assertEqual(args[args.index("-frames:v") + 1], "97")
        self.assertIn("-protocol_whitelist", args)
        self.assertIn("-an", args)

    def test_the_audio_keeps_the_vocal_then_pads_silence(self) -> None:
        args = lip.prepare_audio_command("ffmpeg", Path("in.wav"), Path("out.wav"), 2.0, 113 / 24, keep=97 / 24)
        limits = [i for i, arg in enumerate(args) if arg == "-t"]
        self.assertLess(limits[0], args.index("-i"))  # an input limit: the rest is apad's silence
        self.assertEqual([args[i + 1] for i in limits], [f"{97 / 24:.6f}", f"{113 / 24:.6f}"])
        plain = lip.prepare_audio_command("ffmpeg", Path("in.wav"), Path("o.wav"), 0.0, 4.0)
        self.assertEqual(plain.count("-t"), 1)


class ShownFramesTests(unittest.TestCase):
    """The score reads the room for r but never counts a grin or a turn there against the take."""

    def series(self):
        fps, n, shown = 24, 121, 105  # 0.4 s lead, sung to 4.4 s, room to 5.04 s
        env = np.full(n, 0.002)
        for a, b in ((0.4, 1.6), (2.0, 3.0), (3.4, 4.36)):
            env[int(a * fps): int(b * fps)] = 0.2
        mouth = np.where(env > 0.05, 0.08, 0.005)
        mouth[shown:] = 0.06  # grins open in the room the film cuts off
        yaw = np.zeros(n)
        yaw[shown:] = 0.8  # and turns away there
        return mouth, env, yaw, shown

    def test_the_room_counts_for_r_only(self) -> None:
        mouth, env, yaw, shown = self.series()
        full = S.score_series(mouth, env, yaw=yaw)
        seen = S.score_series(mouth, env, yaw=yaw, shown=shown)
        self.assertEqual((seen.r, seen.chance_p95, seen.frames), (full.r, full.chance_p95, full.frames))
        self.assertGreater(full.open_while_silent, seen.open_while_silent)
        self.assertEqual(seen.open_while_silent, 0.0)
        self.assertGreater(full.turned_away, 0.1)
        self.assertEqual(seen.turned_away, 0.0)
        self.assertGreater(seen.score, full.score)

    def test_shown_past_the_end_is_the_whole_clip(self) -> None:
        mouth, env, yaw, _ = self.series()
        self.assertEqual(S.score_series(mouth, env, yaw=yaw, shown=999), S.score_series(mouth, env, yaw=yaw))


if __name__ == "__main__":
    unittest.main()
