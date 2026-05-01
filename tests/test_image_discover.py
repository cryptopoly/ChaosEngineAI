"""Tests for Image Discover latest-model selection."""

from __future__ import annotations

import json
import urllib.error
import unittest
from unittest import mock

from backend_service.helpers.images import (
    _clear_image_discover_caches,
    _image_repo_live_metadata,
    _is_latest_image_candidate,
    _tracked_latest_seed_payloads,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class ImageDiscoverLatestTests(unittest.TestCase):
    def setUp(self):
        _clear_image_discover_caches()

    def tearDown(self):
        _clear_image_discover_caches()

    def test_tracked_latest_fallback_includes_post_august_2025_models(self):
        payloads = _tracked_latest_seed_payloads([])
        repos = [str(item.get("repo") or "") for item in payloads]

        self.assertEqual(payloads[0].get("releaseDate"), "2026-04")
        self.assertIn("baidu/ERNIE-Image", repos)
        self.assertIn("black-forest-labs/FLUX.2-dev", repos)
        self.assertIn("Qwen/Qwen-Image-Edit-2511", repos)

    def test_tracked_latest_fallback_has_size_metadata_for_ram_estimates(self):
        payloads = _tracked_latest_seed_payloads([])

        missing = [
            str(item.get("repo") or item.get("id") or "")
            for item in payloads
            if not (float(item.get("sizeGb") or 0) > 0)
        ]
        missing_weights = [
            str(item.get("repo") or item.get("id") or "")
            for item in payloads
            if not (
                float(item.get("coreWeightsGb") or 0) > 0
                or float(item.get("repoSizeGb") or 0) > 0
            )
        ]

        self.assertEqual(missing, [])
        self.assertEqual(missing_weights, [])

    def test_tracked_latest_fallback_has_runtime_metadata_for_ram_estimates(self):
        payloads = _tracked_latest_seed_payloads([])

        missing_runtime = [
            str(item.get("repo") or item.get("id") or "")
            for item in payloads
            if not (
                float(item.get("runtimeFootprintGb") or 0) > 0
                or float(item.get("runtimeFootprintMpsGb") or 0) > 0
                or float(item.get("runtimeFootprintCudaGb") or 0) > 0
                or float(item.get("runtimeFootprintCpuGb") or 0) > 0
            )
        ]

        self.assertEqual(missing_runtime, [])

    def test_latest_candidate_accepts_current_official_diffusers_repos(self):
        self.assertTrue(
            _is_latest_image_candidate(
                {
                    "id": "baidu/ERNIE-Image",
                    "tags": ["diffusers", "text-to-image"],
                    "pipeline_tag": "text-to-image",
                    "downloads": 4100,
                    "likes": 488,
                },
                curated_repos=set(),
            )
        )

    def test_latest_candidate_filters_low_signal_loras(self):
        self.assertFalse(
            _is_latest_image_candidate(
                {
                    "id": "random-user/current-model-lora",
                    "tags": ["diffusers", "text-to-image", "lora"],
                    "pipeline_tag": "text-to-image",
                    "downloads": 50_000,
                    "likes": 500,
                },
                curated_repos=set(),
            )
        )

    def test_live_metadata_cache_is_token_aware_for_gated_repos(self):
        gated_error = urllib.error.HTTPError(
            url="https://huggingface.co/api/models/gated/model",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )
        live_payload = {
            "downloads": 12,
            "likes": 3,
            "gated": "auto",
            "siblings": [
                {"rfilename": "transformer/model.safetensors", "size": 2 * 1024 ** 3},
            ],
        }

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(
                "backend_service.helpers.images.urllib.request.urlopen",
                side_effect=[gated_error, _FakeResponse(live_payload)],
            ) as urlopen:
                anonymous = _image_repo_live_metadata("gated/model")

                self.assertIn("HTTP 403", anonymous["metadataWarning"])

                with mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test_token"}, clear=True):
                    tokened = _image_repo_live_metadata("gated/model")

                self.assertEqual(tokened["downloads"], 12)
                self.assertEqual(tokened["coreWeightsGb"], 2.0)
                self.assertEqual(urlopen.call_count, 2)
                authed_request = urlopen.call_args_list[1].args[0]
                self.assertEqual(authed_request.get_header("Authorization"), "Bearer hf_test_token")


if __name__ == "__main__":
    unittest.main()
