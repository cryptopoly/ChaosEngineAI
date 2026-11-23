"""Lip-sync takes (D-110): take counts, seeds, the pick, the scorer lookup and process, and the time estimate.

No model and no MediaPipe: the scoring process is a stand-in script that writes the JSON ``lip_sync_score`` writes."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from backend_service import lipsync_runtime as lip
from backend_service import lipsync_takes as takes
from backend_service.lip_sync_score import SyncScore
from tests.test_lipsync import CUDA_REPO, MLX_REPO, _variant


def _score(value: float, usable: bool) -> SyncScore:
    return SyncScore(frames=121, face_frames=121, voiced_frames=80, r=0.5, r_lag0=0.4, lag=1, chance_p95=0.3,
                     chance_mean=0.1, beats_chance=usable, closed_while_voiced=0.1, open_while_silent=0.5,
                     balanced=0.7, motion=1.0, turned_away=0.0, score=value, mouth_mean=0.03)


class TakeCountTests(unittest.TestCase):
    def test_defaults_three_sung_one_spoken(self) -> None:
        self.assertEqual(takes.take_count(None, True), 3)
        self.assertEqual(takes.take_count(None, False), 1)

    def test_a_request_wins_and_is_held_to_one_to_four(self) -> None:
        self.assertEqual(takes.take_count(4, False), 4)
        self.assertEqual(takes.take_count(1, True), 1)
        self.assertEqual(takes.take_count(9, True), 4)

    def test_each_take_has_its_own_seed(self) -> None:
        self.assertEqual(takes.take_seeds(42, 3), [42, 43, 44])
        self.assertEqual(takes.take_seeds(2**31 - 1, 2), [2**31 - 1, 0])


class PickTests(unittest.TestCase):
    def test_usable_first_then_score(self) -> None:
        self.assertEqual(takes.pick([_score(0.6, False), _score(0.4, True), _score(0.5, True)]), 2)

    def test_unscored_takes_are_skipped_and_none_scored_keeps_the_first(self) -> None:
        self.assertEqual(takes.pick([None, _score(0.3, False), None]), 1)
        self.assertEqual(takes.pick([None, None]), 0)

    def test_summary_names_the_kept_take(self) -> None:
        text = takes.summary([_score(0.3, False), _score(0.5, True)], [7, 8], 1, None)
        self.assertIn("kept take 2 (seed 8, score 0.50, usable)", text)
        self.assertEqual(takes.summary([None], [7], 0, None), "")
        self.assertIn("were not scored (no MediaPipe)", takes.summary([None, None], [7, 8], 0, "no MediaPipe"))


class ScorerTests(unittest.TestCase):
    def test_no_addon_no_modules_and_no_override_says_so(self) -> None:
        with mock.patch.dict("os.environ", {takes.SCORE_PYTHON_ENV: "", takes.FACE_MODEL_ENV: ""}), \
                mock.patch.object(takes.lipsync_score_addon, "site_installed", return_value=False), \
                mock.patch.object(takes, "_has_score_modules", return_value=False):
            self.assertEqual(takes.scorer(), takes.NOT_INSTALLED)

    def test_a_missing_face_model_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {takes.SCORE_PYTHON_ENV: sys.executable,
                                               takes.FACE_MODEL_ENV: str(Path(tmp) / "none.task")}):
            self.assertIn("face model that does not exist", takes.scorer())

    def test_the_override_and_model_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "face.task"
            model.write_bytes(b"x")
            with mock.patch.dict("os.environ", {takes.SCORE_PYTHON_ENV: sys.executable, takes.FACE_MODEL_ENV: str(model)}):
                self.assertEqual(takes.scorer(), takes.Scorer(sys.executable, model, None))

    def test_the_installed_addon_is_used_without_any_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lipsync-score"
            (root / "site").mkdir(parents=True)
            (root / "face_landmarker.task").write_bytes(b"x")
            with mock.patch.dict("os.environ", {takes.SCORE_PYTHON_ENV: "", takes.FACE_MODEL_ENV: ""}), \
                    mock.patch.object(takes.lipsync_score_addon, "addon_dir", return_value=root), \
                    mock.patch.object(takes.lipsync_score_addon, "site_installed", return_value=True):
                self.assertEqual(takes.scorer(),
                                 takes.Scorer(sys.executable, root / "face_landmarker.task", root / "site"))

    def test_the_addon_folder_reaches_the_score_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-abc"
            fake = mock.Mock(returncode=1, stderr="stopped", stdout="")
            with mock.patch.object(takes.subprocess, "run", return_value=fake) as run:
                value, problem = takes.score(takes.Scorer(sys.executable, Path("face.task"), site), Path("v.mp4"),
                                             Path("a.wav"))
            command = run.call_args.args[0]
        self.assertIsNone(value)
        self.assertIn("stopped", problem)
        self.assertEqual(command[command.index("--site") + 1], str(site))
        self.assertIn("-I", command)

    def test_the_score_process_result_is_read(self) -> None:
        fields = {**_score(0.42, True).as_dict(), "mouth": [0.01], "env": [0.1]}
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_score.py"
            script.write_text(textwrap.dedent(f"""
                import json, sys
                out = sys.argv[sys.argv.index("--json") + 1]
                open(out, "w").write({json.dumps(fields)!r})
            """), encoding="utf-8")
            with mock.patch.object(takes, "_SCRIPT", script):
                value, problem = takes.score(takes.Scorer(sys.executable, Path("face.task")), Path("v.mp4"),
                                             Path("a.wav"))
        self.assertIsNone(problem)
        self.assertEqual((value.score, value.usable), (0.42, True))

    def test_a_failing_score_process_is_a_reason_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fail.py"
            script.write_text("import sys\nsys.stderr.write('no face found\\n')\nsys.exit(2)\n", encoding="utf-8")
            with mock.patch.object(takes, "_SCRIPT", script):
                value, problem = takes.score(takes.Scorer(sys.executable, Path("face.task")), Path("v.mp4"),
                                             Path("a.wav"))
        self.assertIsNone(value)
        self.assertIn("no face found", problem)

    def test_the_real_score_script_runs_isolated(self) -> None:
        # ``python -I lip_sync_score.py --help`` must work without the backend on sys.path (the scorer's Python is
        # often another venv).
        import subprocess

        run = subprocess.run([sys.executable, "-I", str(takes._SCRIPT), "--help"], capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--model", run.stdout)
        self.assertIn("--site", run.stdout)


class EstimateTests(unittest.TestCase):
    def test_one_take_is_the_measured_shot(self) -> None:
        self.assertAlmostEqual(lip.estimate_seconds(_variant(CUDA_REPO), 768, 512, 121), 118.0)

    def test_three_takes_add_warm_takes_and_scoring(self) -> None:
        # cold 118 s + two warm takes of 70 s + 3 x 10 s scoring
        self.assertAlmostEqual(lip.estimate_seconds(_variant(CUDA_REPO), 768, 512, 121, 3), 288.0)
        self.assertAlmostEqual(lip.estimate_seconds(_variant(CUDA_REPO), 768, 512, 121, 3, warm=True), 240.0)

    def test_no_measurement_is_none(self) -> None:
        self.assertIsNone(lip.estimate_seconds(_variant(MLX_REPO), 768, 512, 121, 3))


if __name__ == "__main__":
    unittest.main()
