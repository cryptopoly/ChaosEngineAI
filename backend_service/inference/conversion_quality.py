"""Measured quality of a converted model: its perplexity against the source's.

The Conversion page used to show a "Quality" percentage that came from the
KV-cache preview and a table of constants; neither measured the converted
model. This module measures it: the source and the converted model each read
the same short WikiText-2 sample (same tokenizer, same seeded windows) and the
change in perplexity is the result.

* ``quality_plan`` decides whether the measurement can run at all (a local
  source folder, both models fit in free memory one after the other).
* ``measure_perplexity`` is the child-process side: load one model, print one
  JSON line. One process per model, so the first is out of memory before the
  second loads and a stuck load never takes the backend with it.
* ``quality_summary`` turns the two numbers into what the UI shows.

Nothing here estimates: when it cannot measure, it says why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from backend_service.inference.conversion import _bytes_to_gb, _path_size_bytes


QUALITY_DATASET = "wikitext-2"
QUALITY_SAMPLES = 8
QUALITY_SEQ_LENGTH = 256
# A model is loaded only when its files take at most this share of the free memory.
_MEMORY_SHARE = 0.7


def quality_skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def quality_plan(
    source_dir: str | None,
    output_dir: str,
    available_bytes: int | None,
) -> dict[str, Any] | None:
    """None when both models can be measured, else the ``skipped`` result that says why not."""
    if not source_dir or not Path(source_dir).is_dir():
        return quality_skipped(
            "The source is not a model folder on this disk, so it cannot be loaded for a comparison."
        )
    if available_bytes is None:
        return quality_skipped("Free memory could not be read, so the models were not loaded for a comparison.")
    largest = max(_path_size_bytes(source_dir), _path_size_bytes(output_dir))
    if largest > available_bytes * _MEMORY_SHARE:
        return quality_skipped(
            f"The comparison loads the source once ({_bytes_to_gb(largest)} GB) and "
            f"{_bytes_to_gb(available_bytes)} GB of memory is free. Unload a model and run a "
            f"Perplexity benchmark on both to compare."
        )
    return None


def quality_summary(source_perplexity: float, converted_perplexity: float) -> dict[str, Any]:
    """The measured result. ``changePercent`` above zero means the converted model predicts the text worse."""
    change = (converted_perplexity - source_perplexity) / source_perplexity * 100.0 if source_perplexity > 0 else 0.0
    return {
        "status": "measured",
        "reason": None,
        "dataset": QUALITY_DATASET,
        "samples": QUALITY_SAMPLES,
        "seqLength": QUALITY_SEQ_LENGTH,
        "sourcePerplexity": round(source_perplexity, 3),
        "convertedPerplexity": round(converted_perplexity, 3),
        "changePercent": round(change, 2),
    }


def parse_measurement(stdout: str) -> float | None:
    """The perplexity from the child's output: the last JSON line with a ``perplexity`` number."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line).get("perplexity")
        except (ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def measure_perplexity(model_path: str) -> float:
    """Load one model and return its perplexity on the fixed sample (child process only: loads MLX)."""
    from mlx_lm import load

    from backend_service.mlx_worker_eval import load_perplexity_data, perplexity_of

    model, tokenizer = load(model_path)
    data = load_perplexity_data(tokenizer, QUALITY_DATASET, QUALITY_SAMPLES, QUALITY_SEQ_LENGTH)
    perplexity, _standard_error = perplexity_of(model, data, batch_size=2)
    return perplexity


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m backend_service.inference.conversion_quality <model folder>", file=sys.stderr)
        return 2
    print(json.dumps({"perplexity": measure_perplexity(args[0])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
