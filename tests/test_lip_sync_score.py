"""The automatic lip-sync score (FU-334): the maths on synthetic mouth and voice series, no video or MediaPipe."""

import unittest

import numpy as np

from backend_service import lip_sync_score as S

FPS = 24.0
N = 121


def _voice() -> np.ndarray:
    """Phrases on and off: silent 0-0.4 s, sung 0.4-1.6, silent, sung 2.0-3.5, silent, sung 4.0-5.0."""
    env = np.full(N, 0.002)
    for a, b in ((0.4, 1.6), (2.0, 3.5), (4.0, 5.0)):
        env[int(a * FPS): int(b * FPS)] = 0.2
    rng = np.random.default_rng(1)
    return env * (1 + 0.3 * rng.random(N))


def _following(env: np.ndarray, lead: int = 0) -> np.ndarray:
    mouth = np.where(env > 0.05, 0.08, 0.005) + 0.004 * np.random.default_rng(2).random(N)
    return np.roll(mouth, -lead) if lead else mouth


class ScoreSeriesTests(unittest.TestCase):
    def test_a_mouth_that_follows_the_voice_beats_chance(self):
        env = _voice()
        result = S.score_series(_following(env), env)
        self.assertTrue(result.beats_chance)
        self.assertGreater(result.r, 0.9)
        self.assertEqual(result.closed_while_voiced, 0.0)
        self.assertEqual(result.open_while_silent, 0.0)
        self.assertAlmostEqual(result.balanced, 1.0)
        self.assertGreater(result.score, 0.9)

    def test_a_mouth_shut_through_a_sung_phrase_is_counted(self):
        env = _voice()
        mouth = _following(env)
        mouth[int(2.0 * FPS): int(3.5 * FPS)] = 0.005  # the held line sung with a closed mouth
        result = S.score_series(mouth, env)
        voiced = int((env > S.VOICED_FRACTION * np.percentile(env, 95)).sum())
        self.assertAlmostEqual(result.closed_while_voiced, round(36 / voiced, 4), places=3)
        self.assertLess(result.score, S.score_series(_following(env), env).score)

    def test_a_mouth_open_all_the_time_scores_half_on_balance(self):
        env = _voice()
        result = S.score_series(np.full(N, 0.08) + 0.001 * np.random.default_rng(3).random(N), env)
        self.assertEqual(result.closed_while_voiced, 0.0)
        self.assertEqual(result.open_while_silent, 1.0)
        self.assertAlmostEqual(result.balanced, 0.5)
        self.assertFalse(result.beats_chance)

    def test_a_mouth_that_ignores_the_audio_does_not_beat_chance(self):
        env = _voice()
        rng = np.random.default_rng(4)
        # chance_p95 is a 5 % false-positive line: over 40 unrelated mouths about 2 should pass, never many
        passed = sum(S.score_series(rng.random(N) * 0.08, env).beats_chance for _ in range(40))
        self.assertLessEqual(passed, 6)

    def test_a_mouth_two_frames_early_is_found_at_that_lag(self):
        env = _voice()
        result = S.score_series(_following(env, lead=2), env)
        self.assertEqual(result.lag, 2)
        self.assertGreater(result.r, result.r_lag0)

    def test_frames_without_a_face_are_left_out(self):
        env = _voice()
        mouth = _following(env)
        mouth[:10] = np.nan
        result = S.score_series(mouth, env)
        self.assertEqual(result.face_frames, N - 10)
        self.assertTrue(result.beats_chance)

    def test_a_shut_mouth_whose_jitter_follows_the_audio_gets_no_credit_for_r(self):
        env = _voice()
        mouth = 0.001 + 0.002 * (env > 0.05)  # tiny landmark movement in time with the voice, never open
        result = S.score_series(mouth, env)
        self.assertGreater(result.r, 0.9)
        self.assertEqual(result.closed_while_voiced, 1.0)
        self.assertLess(result.motion, 0.1)
        self.assertLess(result.score, 0.3)

    def test_a_head_turned_away_lowers_the_ranking_score(self):
        env = _voice()
        mouth = _following(env)
        front = S.score_series(mouth, env, yaw=np.zeros(N))
        yaw = np.zeros(N)
        yaw[60:] = 0.9  # profile for the second half
        turned = S.score_series(mouth, env, yaw=yaw)
        self.assertEqual(front.turned_away, 0.0)
        self.assertAlmostEqual(turned.turned_away, round(61 / N, 4))
        self.assertLess(turned.score, front.score)
        self.assertEqual(turned.balanced, front.balanced)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            S.score_series(np.zeros(10), np.zeros(11))

    def test_silence_only_never_divides_by_zero(self):
        result = S.score_series(np.full(N, 0.005), np.zeros(N))
        self.assertEqual(result.voiced_frames, 0)
        self.assertEqual(result.closed_while_voiced, 0.0)


class EnvelopeTests(unittest.TestCase):
    def test_envelope_is_loud_where_the_tone_is(self):
        rate = 48000
        t = np.arange(int(rate * N / FPS)) / rate
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        tone[: rate] = 0.0  # first second silent
        env = S.envelope(np.stack([tone, tone], axis=1), rate, N, FPS)
        self.assertEqual(env.shape, (N,))
        self.assertLess(env[:20].max(), 0.01)
        self.assertGreater(env[40:].min(), 0.15)

    def test_out_of_band_rumble_is_ignored(self):
        rate = 16000
        t = np.arange(int(rate * N / FPS)) / rate
        env = S.envelope(0.3 * np.sin(2 * np.pi * 30 * t), rate, N, FPS)
        self.assertLess(env[10:-10].max(), 0.02)


def _take(score, *, beats=True, closed=0.1, turned=0.0):
    return S.SyncScore(frames=N, face_frames=N, voiced_frames=60, r=0.5, r_lag0=0.5, lag=0, chance_p95=0.3,
                       chance_mean=0.0, beats_chance=beats, closed_while_voiced=closed, open_while_silent=0.5,
                       balanced=0.7, motion=1.0, turned_away=turned, score=score, mouth_mean=0.03)


class PickBestTests(unittest.TestCase):
    def test_usable_needs_chance_singing_and_facing_the_camera(self):
        self.assertTrue(_take(0.5).usable)
        self.assertFalse(_take(0.5, beats=False).usable)
        self.assertFalse(_take(0.5, closed=0.6).usable)
        self.assertFalse(_take(0.5, turned=0.5).usable)
        self.assertIn("usable", _take(0.5).as_dict())

    def test_a_usable_take_beats_a_higher_scoring_profile_take(self):
        takes = [_take(0.6, turned=0.5), _take(0.45), _take(0.4)]
        self.assertEqual(S.pick_best(takes), 1)

    def test_highest_score_wins_when_none_is_usable(self):
        self.assertEqual(S.pick_best([_take(0.3, beats=False), _take(0.35, beats=False)]), 1)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            S.pick_best([])


class SmoothTests(unittest.TestCase):
    def test_a_single_frame_spike_is_removed_and_gaps_stay(self):
        out = S.smooth([0.01, 0.01, 0.2, 0.01, float("nan"), 0.01])
        self.assertAlmostEqual(out[2], 0.01)
        self.assertTrue(np.isnan(out[4]))


if __name__ == "__main__":
    unittest.main()
