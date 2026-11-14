"""Territory gate (D-76, FU-166): MiniMax H3 is hidden and refused until the user confirms, and never before release.

No weights are downloaded and nothing runs: downloads are refused before they start, and the sd.cpp command line
is checked against a fake ``sd`` script that only records its arguments.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service import licence_gate
from backend_service.app import create_app
from backend_service.catalog.video_minimax_h3 import (
    H3_AUDIO_VAE_FILE,
    H3_DIT_FILE,
    H3_GGUF_REPO,
    H3_ID,
    H3_TEXT_ENCODER_FILE,
    H3_VAE_REPO,
    H3_VIDEO_VAE_FILE,
)

H = {"Authorization": "Bearer app-token"}
KEY = licence_gate.H3_KEY
SENTENCE = licence_gate.TERRITORY_MODELS[KEY]["confirmSentence"]
ROOT = Path(__file__).resolve().parent.parent


def _ids(families):
    return {str(v.get("id")) for f in families for v in f.get("variants") or []}


class GateCase(unittest.TestCase):
    """A throwaway data directory; ``released`` decides the switch for the test."""

    released = True

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        for patcher in (
            mock.patch.object(licence_gate, "_data_dir", lambda: self.data),
            mock.patch.object(licence_gate, "H3_RELEASED", self.released),
            # The catalog payload fetches live Hub metadata; keep the tests offline.
            mock.patch("backend_service.helpers.video._image_repo_live_metadata", lambda repo: {}),
            mock.patch("backend_service.helpers.images._image_repo_live_metadata", lambda repo: {}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))

    def video_payload_ids(self):
        from backend_service.helpers.video import _video_model_payloads

        return _ids(_video_model_payloads([]))

    def confirm(self, code=200, headers=None, **overrides):
        body = {"key": KEY, "confirmed": True, "sentence": SENTENCE, **overrides}
        response = self.client.post("/api/licence-gate/confirm", json=body, headers=headers or H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()


class ReleaseSwitchTests(GateCase):
    released = False

    def test_the_committed_switch_is_off(self):
        source = (ROOT / "backend_service" / "licence_gate.py").read_text(encoding="utf-8")
        self.assertRegex(source, r"(?m)^H3_RELEASED = False$")

    def test_unreleased_hides_everything_and_nothing_can_be_confirmed(self):
        self.assertEqual(licence_gate.status(), {"models": []})
        self.assertNotIn(H3_ID, self.video_payload_ids())
        # Even a record left on disk does not count while the switch is off.
        (self.data / "licence-gate.json").write_text(json.dumps({KEY: {"sentence": SENTENCE}}), encoding="utf-8")
        self.assertFalse(licence_gate.is_confirmed(KEY))
        self.assertNotIn(H3_ID, self.video_payload_ids())
        detail = self.confirm(code=403)["detail"]
        self.assertIn("not available in this version", detail)
        response = self.client.post("/api/video/download", json={"repo": H3_GGUF_REPO, "modelId": H3_ID}, headers=H)
        self.assertEqual(response.status_code, 403)

    def test_flipping_the_switch_lists_the_rows_after_confirmation(self):
        with mock.patch.object(licence_gate, "H3_RELEASED", True):
            self.assertEqual([m["key"] for m in licence_gate.status()["models"]], [KEY])
            self.assertNotIn(H3_ID, self.video_payload_ids())
            self.confirm()
            self.assertIn(H3_ID, self.video_payload_ids())
        self.assertNotIn(H3_ID, self.video_payload_ids())


class ConfirmTests(GateCase):
    def test_hidden_by_default_with_one_neutral_notice(self):
        self.assertNotIn(H3_ID, self.video_payload_ids())
        with mock.patch("backend_service.state.ChaosEngineState._library", return_value=[]):
            payload = self.client.get("/api/video/catalog", headers=H).json()
        self.assertNotIn(H3_ID, _ids(payload["families"]))
        self.assertEqual(payload["territoryLimited"], [KEY])
        status = self.client.get("/api/licence-gate", headers=H).json()
        self.assertEqual(status["models"][0]["confirmed"], False)
        self.assertEqual(status["models"][0]["confirmSentence"], SENTENCE)

    def test_confirm_records_the_statement_and_withdraw_hides_again(self):
        status = self.confirm()
        self.assertTrue(status["models"][0]["confirmed"])
        self.assertIn(H3_ID, self.video_payload_ids())
        record = json.loads((self.data / "licence-gate.json").read_text(encoding="utf-8"))[KEY]
        self.assertEqual(record["key"], KEY)
        self.assertEqual(record["sentence"], SENTENCE)
        self.assertTrue(record["appVersion"])
        self.assertRegex(record["confirmedAt"], r"^\d{4}-\d\d-\d\dT")
        response = self.client.post("/api/licence-gate/withdraw", json={"key": KEY}, headers=H)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["models"][0]["confirmed"])
        self.assertNotIn(H3_ID, self.video_payload_ids())
        audit = [json.loads(line) for line in (self.data / "licence-gate-audit.jsonl").read_text().splitlines()]
        self.assertEqual([entry["event"] for entry in audit], ["confirm", "withdraw"])

    def test_confirm_needs_the_tick_and_the_exact_sentence(self):
        self.confirm(code=400, confirmed=False)
        self.confirm(code=400, sentence="I am fine.")
        self.confirm(code=404, key="someone-else")
        self.assertFalse(licence_gate.is_confirmed(KEY))

    def test_a_changed_sentence_needs_a_new_tick(self):
        self.confirm()
        changed = {**licence_gate.TERRITORY_MODELS[KEY], "confirmSentence": "A new statement."}
        with mock.patch.dict(licence_gate.TERRITORY_MODELS, {KEY: changed}):
            self.assertFalse(licence_gate.is_confirmed(KEY))

    def test_only_the_app_window_can_confirm(self):
        from backend_service.mcp_bridge import access

        with mock.patch.object(access, "token_ok", lambda token: token == "agent-token"):
            self.confirm(code=403, headers={"Authorization": "Bearer agent-token"})
            response = self.client.post("/api/licence-gate/withdraw", json={"key": KEY},
                                        headers={"Authorization": "Bearer agent-token"})
            self.assertEqual(response.status_code, 403)
        remote = TestClient(create_app(api_token="app-token"), client=("192.168.1.20", 5000))
        response = remote.post("/api/licence-gate/confirm", json={"key": KEY, "confirmed": True, "sentence": SENTENCE},
                               headers=H)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(licence_gate.is_confirmed(KEY))


class RefusalTests(GateCase):
    def assert_refused(self, response):
        self.assertEqual(response.status_code, 403, response.text)
        self.assertIn("limited by territory", response.json()["detail"])

    def test_every_route_refuses_without_confirmation(self):
        post = self.client.post
        self.assert_refused(post("/api/video/download", json={"repo": H3_GGUF_REPO, "modelId": H3_ID}, headers=H))
        self.assert_refused(post("/api/video/download", json={"repo": H3_GGUF_REPO}, headers=H))
        self.assert_refused(post("/api/video/preload", json={"modelId": H3_ID}, headers=H))
        self.assert_refused(post("/api/video/generate", json={"modelId": H3_ID, "prompt": "a cat"}, headers=H))
        # The image download takes a bare repo: the H3 repos are refused there too.
        self.assert_refused(post("/api/images/download", json={"repo": H3_GGUF_REPO}, headers=H))
        self.assert_refused(post("/api/images/download", json={"repo": H3_VAE_REPO}, headers=H))

    def test_any_download_of_the_weights_is_refused_whichever_route_asks(self):
        from fastapi import HTTPException

        from backend_service.state.downloads import start_download

        for repo in (H3_GGUF_REPO, H3_VAE_REPO):
            with self.assertRaises(HTTPException) as caught:
                start_download(mock.Mock(_downloads={}), repo)
            self.assertEqual(caught.exception.status_code, 403)

    def test_after_confirming_the_gate_lets_the_route_continue(self):
        self.confirm()
        # Past the gate: the weights are not here, so the route says so (409), not 403.
        response = self.client.post("/api/video/preload", json={"modelId": H3_ID}, headers=H)
        self.assertEqual(response.status_code, 409, response.text)


class BridgeTests(GateCase):
    def test_no_agent_tool_can_confirm(self):
        from backend_service.mcp_bridge.tools import HANDLERS, TOOLS

        names = set(HANDLERS) | {str(t.get("name")) for t in TOOLS}
        self.assertFalse([n for n in names if re.search(r"licen|territor|confirm", n)])

    def test_agents_neither_see_nor_get_the_row_by_default(self):
        from backend_service.mcp_bridge import models as bridge_models
        from backend_service.mcp_bridge.tools import ToolError, _pick_variant

        state = mock.Mock(_library=lambda: [])
        rows = bridge_models.recommend(state, "video", limit=50)["models"]
        self.assertNotIn(H3_ID, {r["id"] for r in rows})
        ctx = mock.Mock(app=mock.Mock(state=mock.Mock(chaosengine=state)))
        with self.assertRaises(ToolError):
            _pick_variant(ctx, "video", H3_ID)

    def test_a_confirmed_row_is_flagged_and_never_the_default(self):
        from backend_service.mcp_bridge import models as bridge_models

        self.confirm()
        family = {"name": "MiniMax H3"}
        row = bridge_models._row(family, {"id": H3_ID, "territoryGate": KEY}, "video")
        self.assertTrue(row["territoryRestricted"])
        rows = [{"id": H3_ID, "fit": "fits", "territoryRestricted": True}]
        with mock.patch.object(bridge_models, "installed_models", return_value=rows):
            self.assertIsNone(bridge_models.default_model(mock.Mock(), "video"))

    def test_best_fit_never_picks_a_gated_row(self):
        from backend_service.helpers.fit import best_fit_variant_id

        fits = {"status": "fits", "needGb": 1}
        variants = [
            {"id": "gated", "territoryGate": KEY, "releaseDate": "2026-08", "fit": fits},
            {"id": "open", "releaseDate": "2025-01", "fit": fits},
        ]
        self.assertEqual(best_fit_variant_id(variants, "video"), "open")
        self.assertIsNone(best_fit_variant_id(variants[:1], "video"))


class H3CommandLineTests(unittest.TestCase):
    """The exact argument list, recorded by a fake ``sd`` script (never the real binary, never real weights)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.record = self.dir / "args.json"
        fake = self.dir / "sd"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, pathlib\n"
            f"pathlib.Path({str(self.record)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            "out = sys.argv[sys.argv.index('-o') + 1]\n"
            "print('step 1/20', flush=True)\n"
            "pathlib.Path(out).write_bytes(b'webm')\n",
            encoding="utf-8",
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.files = {}
        for flag, name in (("--diffusion-model", H3_DIT_FILE), ("--llm", H3_TEXT_ENCODER_FILE),
                           ("--vae", H3_VIDEO_VAE_FILE), ("--audio-vae", H3_AUDIO_VAE_FILE)):
            path = self.dir / Path(name).name
            path.write_bytes(b"")
            self.files[name] = str(path)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_SDCPP_BIN": str(fake)})
        env.start()
        self.addCleanup(env.stop)

    def _fake_download(self, repo_id, filename, local_files_only=False):
        assert local_files_only, "generate must never download"
        return self.files[filename]

    def test_h3_arguments_follow_the_pinned_doc(self):
        from backend_service.sdcpp_video_runtime import SdCppVideoEngine
        from backend_service.video_runtime import VideoGenerationConfig

        config = VideoGenerationConfig(
            modelId=H3_ID, modelName="MiniMax H3", repo=H3_GGUF_REPO, prompt="a cat surfs", negativePrompt="blur",
            width=860, height=470, numFrames=48, fps=16, guidance=5.0, steps=20, seed=7,
        )
        with mock.patch("huggingface_hub.hf_hub_download", self._fake_download), \
                mock.patch("backend_service.sdcpp_video_runtime.platform.system", return_value="Darwin"):
            video = SdCppVideoEngine().generate(config)
        args = json.loads(self.record.read_text())
        out = args[args.index("-o") + 1]
        self.assertEqual(args, [
            "-M", "vid_gen",
            "--diffusion-model", self.files[H3_DIT_FILE],
            "--vae", self.files[H3_VIDEO_VAE_FILE],
            "--audio-vae", self.files[H3_AUDIO_VAE_FILE],
            "--llm", self.files[H3_TEXT_ENCODER_FILE],
            "-p", "a cat surfs",
            "--cfg-scale", "1",
            "-W", "864", "-H", "480",
            "--steps", "20",
            "--seed", "7",
            "--diffusion-fa",
            "--rng", "cpu",
            "--fps", "24",
            "--video-frames", "56",
            "-o", out,
        ])
        self.assertEqual((video.width, video.height, video.frameCount, video.fps), (864, 480, 56, 24))
        self.assertEqual(video.bytes, b"webm")

    def test_offload_only_off_macos(self):
        from backend_service.sdcpp_video_runtime import SdCppVideoEngine
        from backend_service.video_runtime import VideoGenerationConfig

        config = VideoGenerationConfig(
            modelId=H3_ID, modelName="MiniMax H3", repo=H3_GGUF_REPO, prompt="p", negativePrompt="",
            width=864, height=480, numFrames=56, fps=24, guidance=1.0, steps=8, seed=1,
        )
        with mock.patch("huggingface_hub.hf_hub_download", self._fake_download), \
                mock.patch("backend_service.sdcpp_video_runtime.platform.system", return_value="Linux"):
            SdCppVideoEngine().generate(config)
        self.assertEqual(json.loads(self.record.read_text())[-1], "--offload-to-cpu")

    def test_a_missing_file_is_an_error_not_a_download(self):
        from backend_service.sdcpp_video_runtime import SdCppVideoEngine
        from backend_service.video_runtime import VideoGenerationConfig

        def missing(repo_id, filename, local_files_only=False):
            assert local_files_only
            raise FileNotFoundError(filename)

        config = VideoGenerationConfig(
            modelId=H3_ID, modelName="MiniMax H3", repo=H3_GGUF_REPO, prompt="p", negativePrompt="",
            width=864, height=480, numFrames=56, fps=24, guidance=1.0, steps=8, seed=1,
        )
        with mock.patch("huggingface_hub.hf_hub_download", missing):
            with self.assertRaisesRegex(RuntimeError, "Download it from Video > Models"):
                SdCppVideoEngine().generate(config)
        self.assertFalse(self.record.exists())

    def test_frame_and_size_alignment(self):
        from backend_service.sdcpp_video_runtime import h3_shape

        self.assertEqual(h3_shape(864, 480, 1), (864, 480, 5))
        self.assertEqual(h3_shape(864, 480, 22), (864, 480, 22))
        self.assertEqual(h3_shape(865, 481, 23), (896, 512, 39))

    def test_the_manager_sends_h3_to_sdcpp(self):
        from backend_service.sdcpp_video_runtime import requires_sdcpp

        self.assertTrue(requires_sdcpp(H3_GGUF_REPO))
        self.assertFalse(requires_sdcpp("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))


class CatalogRowTests(unittest.TestCase):
    def test_row_carries_licence_gate_and_no_time(self):
        from backend_service.helpers.video import _find_video_variant, _video_repo_allow_patterns

        row = _find_video_variant(H3_ID)
        self.assertEqual(row["territoryGate"], KEY)
        self.assertTrue(row["licence"]["territoryRestricted"])
        self.assertEqual(row["licence"]["url"], licence_gate.H3_LICENCE_URL)
        self.assertNotIn("estimatedGenerationSeconds", row)
        self.assertNotIn("estimateRef", row)
        self.assertIn("not tested by the maintainers", row["note"].lower())
        self.assertIn("US$20 million", row["note"])
        # Pinned files only (about 35 GB), never the whole repo.
        self.assertEqual(_video_repo_allow_patterns(H3_GGUF_REPO), [H3_DIT_FILE, H3_TEXT_ENCODER_FILE, "README.md"])
        self.assertAlmostEqual(row["sizeGb"], 35.5, places=1)


if __name__ == "__main__":
    unittest.main()
