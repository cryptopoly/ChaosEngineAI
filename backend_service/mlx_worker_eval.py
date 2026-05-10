"""Evaluation entrypoints for the MLX worker — perplexity + task accuracy.

Two stateless helpers lifted out of ``WorkerState``:

* ``eval_perplexity`` — compute negative log-likelihood + perplexity over
  a tokenised dataset (default ``wikitext-2``) batched through the loaded
  model. Returns the perplexity, standard error, eval throughput, and
  the dataset/sample shape echoed back to the UI.
* ``eval_task_accuracy`` — run the model against a benchmark task suite
  (MMLU / HellaSwag / etc., shape defined by ``task_datasets``), score
  each greedy completion via ``score_answer`` and return aggregate
  accuracy.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. ``WorkerState.eval_perplexity`` / ``eval_task_accuracy`` thin-
wrap these so the IPC contract is unchanged.
"""

from __future__ import annotations

import time
from typing import Any

from backend_service.mlx_worker_io import emit_progress


def eval_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    import math
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    dataset = request.get("dataset", "wikitext-2")
    num_samples = int(request.get("numSamples", 64))
    seq_length = int(request.get("seqLength", 512))
    batch_size = int(request.get("batchSize", 4))

    dataset_map = {
        "wikitext-2": "wikitext/wikitext-2-raw-v1",
    }
    data_path = dataset_map.get(dataset, dataset)

    emit_progress("loading_data", 10.0, "Loading evaluation dataset...")
    from mlx_lm.perplexity import load_data
    np.random.seed(123)
    data = load_data(tokenizer, data_path, num_samples, seq_length)

    emit_progress("evaluating", 20.0, f"Evaluating perplexity on {len(data)} samples...")
    start = time.monotonic()

    all_losses: list[mx.array] = []
    num_batches = (len(data) + batch_size - 1) // batch_size
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(losses)
        all_losses.append(losses.flatten())

        pct = 20.0 + (i / len(data)) * 70.0
        emit_progress("evaluating", pct, f"Batch {i // batch_size + 1}/{num_batches}")

    all_losses_cat = mx.concatenate(all_losses)
    mean_loss = all_losses_cat.mean().item()
    ppl = math.exp(mean_loss)
    std_dev = mx.sqrt(mx.var(all_losses_cat, ddof=1)).item()
    se_ppl = ppl * (std_dev / math.sqrt(all_losses_cat.size))

    elapsed = time.monotonic() - start
    tokens_eval = data.shape[0] * (data.shape[1] - 1)

    emit_progress("done", 100.0, f"Perplexity: {ppl:.2f}")
    return {
        "perplexity": round(ppl, 3),
        "standardError": round(se_ppl, 3),
        "evalSeconds": round(elapsed, 2),
        "evalTokensPerSecond": round(tokens_eval / elapsed, 1),
        "numSamples": len(data),
        "seqLength": seq_length,
        "dataset": dataset,
    }


def eval_task_accuracy(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.sample_utils import make_sampler
    from backend_service.task_datasets import load_task_data, score_answer

    task_name = request.get("taskName", "mmlu")
    limit = int(request.get("limit", 100))
    num_shots = int(request.get("numShots", 5))

    emit_progress("loading_tasks", 10.0, f"Loading {task_name} task data...")
    tasks = load_task_data(task_name, limit, num_shots)

    sampler = make_sampler(temp=0.0)  # greedy for accuracy
    correct = 0
    total = len(tasks)
    start = time.monotonic()

    for idx, task in enumerate(tasks):
        text_parts: list[str] = []
        for resp in mlx_stream_generate(
            model,
            tokenizer,
            task["prompt"],
            max_tokens=task.get("max_tokens", 3),
            sampler=sampler,
        ):
            if resp.text:
                text_parts.append(resp.text)

        answer = "".join(text_parts).strip()
        if score_answer(task_name, answer, task["correct_answer"], task.get("choices")):
            correct += 1

        pct = 10.0 + ((idx + 1) / total) * 85.0
        emit_progress(
            "evaluating", pct,
            f"Question {idx + 1}/{total} — {correct}/{idx + 1} correct",
        )

    elapsed = time.monotonic() - start
    accuracy = round(correct / total, 4) if total > 0 else 0.0
    emit_progress("done", 100.0, f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    return {
        "taskName": task_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "numShots": num_shots,
        "evalSeconds": round(elapsed, 2),
    }
