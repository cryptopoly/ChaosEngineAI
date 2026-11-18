"""Adapter discovery and the dataset helper behind the Fine-Tuning tab (backend_service/helpers/finetuning.py)."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.finetuning import _read_adapter_meta, list_adapters, prepare_dataset
from backend_service.routes import finetuning as finetuning_routes


class AdapterMetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _adapter(self, name: str, config: dict) -> Path:
        folder = self.root / name
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        (folder / "adapters.safetensors").write_bytes(b"\0" * 4096)
        return folder

    def test_row_has_the_fields_the_tab_reads(self) -> None:
        # The tab reads name, path, base_model, size_bytes and created; it used to read id and size_mb,
        # which no row ever had, and crashed as soon as one adapter was on disk.
        meta = _read_adapter_meta(self._adapter("peft", {"base_model_name_or_path": "org/base", "r": 8}))
        for key in ("name", "path", "base_model", "size_bytes", "created"):
            self.assertIn(key, meta)
        self.assertEqual(meta["base_model"], "org/base")
        self.assertGreaterEqual(meta["size_bytes"], 4096)
        self.assertRegex(meta["created"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")

    def test_base_model_of_an_mlx_lm_adapter(self) -> None:
        # mlx-lm writes its training arguments: the base model is under "model", the rank under "lora_parameters".
        meta = _read_adapter_meta(self._adapter("mlx", {"model": "/models/qwen-0.5b", "lora_parameters": {"rank": 8}}))
        self.assertEqual(meta["base_model"], "/models/qwen-0.5b")

    def test_unreadable_config_still_lists(self) -> None:
        folder = self._adapter("broken", {})
        (folder / "adapter_config.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(_read_adapter_meta(folder)["base_model"], "unknown")

    def test_list_finds_nested_adapters_once(self) -> None:
        self._adapter("one", {"model": "a"})
        names = [row["name"] for row in list_adapters(self.root) if row["path"].startswith(str(self.root))]
        self.assertEqual(names, ["one"])

    def test_adapters_route_runs_off_the_event_loop(self) -> None:
        # The scan walks the whole model cache (seconds on a large one); a coroutine route would stall every request.
        self.assertFalse(inspect.iscoroutinefunction(finetuning_routes.get_adapters))


class PrepareDatasetTests(unittest.TestCase):
    def test_prompt_completion_rows_become_text_with_a_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw.jsonl"
            rows = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(10)]
            source.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n\n", encoding="utf-8")
            stats = prepare_dataset(source, Path(tmp) / "out")
            self.assertEqual((stats["total_records"], stats["train_records"], stats["valid_records"]), (10, 9, 1))
            first = json.loads((Path(tmp) / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first, {"text": "q0\na0"})

    def test_empty_or_missing_dataset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                prepare_dataset(Path(tmp) / "none.jsonl", Path(tmp) / "out")
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text('{"other": 1}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_dataset(empty, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
