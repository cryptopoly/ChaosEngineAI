"""LoRA / fine-tuning utilities.

Provides adapter discovery, dataset preparation, and a configuration
dataclass for fine-tuning runs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FineTuneConfig:
    """All parameters needed to kick off a fine-tuning run."""

    model_path: str
    dataset_path: str
    output_path: str
    learning_rate: float = 1e-5
    epochs: int = 1
    lora_rank: int = 8
    batch_size: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    warmup_steps: int = 0
    save_every: int = 100
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FineTuneConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------

def list_adapters(data_dir: Path) -> list[dict[str, Any]]:
    """Scan *data_dir* (and common cache locations) for LoRA adapter dirs.

    An adapter directory is any folder that contains an
    ``adapter_config.json`` file.  Returns a list of metadata dicts.
    """
    results: list[dict[str, Any]] = []
    search_roots: list[Path] = [data_dir]

    # Also search common model cache locations
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    mlx_cache = Path.home() / ".cache" / "mlx"
    home_models = Path.home() / "Models"
    for extra in (hf_cache, mlx_cache, home_models):
        if extra.is_dir() and extra not in search_roots:
            search_roots.append(extra)

    seen: set[str] = set()

    for root in search_roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "adapter_config.json" in filenames:
                adapter_dir = Path(dirpath)
                resolved = str(adapter_dir.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)

                meta = _read_adapter_meta(adapter_dir)
                results.append(meta)

                # Don't recurse into adapter directories
                dirnames.clear()

    return results


def _read_adapter_meta(adapter_dir: Path) -> dict[str, Any]:
    """Read adapter_config.json and build a summary dict."""
    config_path = adapter_dir / "adapter_config.json"
    config: dict[str, Any] = {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass

    # Derive size on disk
    total_bytes = 0
    for f in adapter_dir.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size

    return {
        "name": adapter_dir.name,
        "path": str(adapter_dir),
        "base_model": config.get("base_model_name_or_path", "unknown"),
        "lora_rank": config.get("r", config.get("lora_rank")),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules", []),
        "size_bytes": total_bytes,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Convert a JSONL file to mlx-lm fine-tuning format.

    Expects each input line to have at least ``"prompt"`` and
    ``"completion"`` fields (or ``"instruction"`` / ``"output"``).
    Writes train/valid splits to *output_path*.

    Returns summary stats.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Normalize to {"text": ...} which mlx-lm expects
            text = _record_to_text(obj)
            if text:
                records.append({"text": text})

    if not records:
        raise ValueError("No usable records found in dataset")

    # 90/10 train/valid split
    split_idx = max(1, int(len(records) * 0.9))
    train = records[:split_idx]
    valid = records[split_idx:] if split_idx < len(records) else records[-1:]

    output_path.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(output_path / name, "w", encoding="utf-8") as out:
            for rec in subset:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "total_records": len(records),
        "train_records": len(train),
        "valid_records": len(valid),
        "output_path": str(output_path),
    }


def _record_to_text(obj: dict[str, Any]) -> str | None:
    """Best-effort conversion of a single JSONL object to a text string."""
    # Already has "text"
    if "text" in obj and obj["text"]:
        return str(obj["text"])

    # prompt / completion style
    prompt = obj.get("prompt") or obj.get("instruction") or ""
    completion = obj.get("completion") or obj.get("output") or obj.get("response") or ""
    if prompt and completion:
        return f"{prompt}\n{completion}"

    # messages array (chat format)
    messages = obj.get("messages") or obj.get("conversations")
    if messages and isinstance(messages, list):
        parts = []
        for msg in messages:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))
            if role and content:
                parts.append(f"<|{role}|>\n{content}")
        if parts:
            return "\n".join(parts)

    return None
