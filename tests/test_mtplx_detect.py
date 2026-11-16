"""MTPLX detection: a pre-2.12 venv is "outdated", not available; MTP-weight probe on MLX library items."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.discovery import _mlx_mtp_heads
from backend_service.inference import capabilities
from backend_service.inference._mtp import MTP_MODEL_MAP


class MtplxVersionTests(unittest.TestCase):
    def _venv(self, version_text: str | None) -> tuple[Path, Path]:
        root = Path(tempfile.mkdtemp())
        python = root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("")
        version_file = root / "mtplx.version"
        if version_text is not None:
            version_file.write_text(version_text)
        return root / "venv", version_file

    def _detect(self, version_text: str | None):
        venv, version_file = self._venv(version_text)
        with mock.patch.object(capabilities, "_MTPLX_VENV", venv), mock.patch.object(capabilities, "_MTPLX_VERSION_FILE", version_file):
            return capabilities._detect_mtplx()[0], capabilities.mtplx_outdated()

    def test_old_venv_is_outdated_and_not_available(self) -> None:
        self.assertEqual(self._detect("0.3.5\n2026-05-14T11:30:22Z\n"), (False, True))

    def test_current_venv_is_available(self) -> None:
        self.assertEqual(self._detect("2.12.3\n2026-11-16T00:00:00Z\n"), (True, False))
        self.assertEqual(self._detect("2.12.0\n"), (True, False))

    def test_just_below_the_floor_is_outdated(self) -> None:
        self.assertEqual(self._detect("2.11.3\n"), (False, True))

    def test_missing_or_unparsable_version_file(self) -> None:
        self.assertEqual(self._detect(None), (False, False))
        self.assertEqual(self._detect("unknown\n"), (True, False))


class MtpWeightProbeTests(unittest.TestCase):
    def _checkpoint(self, keys: list[str]) -> str:
        root = Path(tempfile.mkdtemp())
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-1.safetensors" for k in keys}}))
        return str(root)

    def test_stock_conversion_has_no_mtp_weights(self) -> None:
        self.assertIs(_mlx_mtp_heads(self._checkpoint(["model.layers.0.mlp.up_proj.weight"])), False)

    def test_checkpoint_with_mtp_keys(self) -> None:
        self.assertIs(_mlx_mtp_heads(self._checkpoint(["model.layers.0.mlp.up_proj.weight", "mtp.layers.0.weight"])), True)

    def test_unreadable_path_is_unknown(self) -> None:
        self.assertIsNone(_mlx_mtp_heads(str(Path(tempfile.mkdtemp()) / "missing")))


class MtplxCatalogTests(unittest.TestCase):
    def test_qwen38_mtplx_checkpoint_is_known_and_in_the_catalog(self) -> None:
        repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
        self.assertEqual(MTP_MODEL_MAP[repo], 3)
        from backend_service.catalog.text_models import MODEL_FAMILIES

        repos = {variant["repo"] for family in MODEL_FAMILIES for variant in family.get("variants", [])}
        self.assertIn(repo, repos)


if __name__ == "__main__":
    unittest.main()
