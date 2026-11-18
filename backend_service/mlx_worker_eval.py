"""Evaluation entrypoints for the MLX worker — perplexity + task accuracy.

Two stateless helpers lifted out of ``WorkerState``:

* ``eval_perplexity`` — compute negative log-likelihood + perplexity over
  a tokenised dataset (default ``wikitext-2``) batched through the loaded
  model. Returns the perplexity, standard error, eval throughput, and
  the dataset/sample shape echoed back to the UI.
* ``eval_task_accuracy`` — run the model against a benchmark task suite
  (MMLU / HellaSwag, shape defined by ``task_datasets``), pick the answer
  letter the model scores highest as the next token and return aggregate
  accuracy.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. ``WorkerState.eval_perplexity`` / ``eval_task_accuracy`` thin-
wrap these so the IPC contract is unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Any

from backend_service.mlx_worker_io import emit_progress


# Dataset name the UI sends -> (Hub repo, config, split). ``wikitext/wikitext-2-raw-v1``,
# the id used before, is not a Hub repo, so every perplexity run failed with
# "doesn't exist on the Hub".
PERPLEXITY_DATASETS: dict[str, tuple[str, str, str]] = {
    "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
}


CANCELLED_MESSAGE = "Benchmark cancelled."


def raise_if_cancelled(request: dict[str, Any]) -> None:
    """Stop between two items when the app has asked for it.

    The worker reads one request at a time from stdin, so a cancel cannot
    arrive as a message while an eval runs. The app names a file in the
    request (``cancelFile``) and creates it to cancel; the model stays loaded.
    """
    cancel_file = request.get("cancelFile")
    if cancel_file and os.path.exists(str(cancel_file)):
        raise RuntimeError(CANCELLED_MESSAGE)


def perplexity_dataset_spec(dataset: str) -> tuple[str, str, str]:
    spec = PERPLEXITY_DATASETS.get(dataset)
    if spec is None:
        supported = ", ".join(sorted(PERPLEXITY_DATASETS))
        raise RuntimeError(f"Unknown perplexity dataset: {dataset}. Supported: {supported}.")
    return spec


def perplexity_windows(
    tokens: list[int],
    num_samples: int,
    seq_length: int,
    *,
    seed: int = 123,
) -> list[list[int]]:
    """Cut a token stream into ``seq_length`` windows and pick ``num_samples`` of them.

    The pick is seeded, so two runs on the same model score the same text.
    """
    import random

    total = len(tokens) // seq_length
    if total < 1:
        raise RuntimeError(
            f"The evaluation text is shorter than one sample of {seq_length} tokens."
        )
    order = list(range(total))
    random.Random(seed).shuffle(order)
    picked = sorted(order[: max(1, num_samples)])
    return [tokens[i * seq_length : (i + 1) * seq_length] for i in picked]


def load_perplexity_data(tokenizer: Any, dataset: str, num_samples: int, seq_length: int) -> Any:
    """Tokenised evaluation windows as an ``mx.array`` of shape (samples, seq_length)."""
    import mlx.core as mx
    from datasets import load_dataset

    repo, config, split = perplexity_dataset_spec(dataset)
    rows = load_dataset(repo, config, split=split)
    text = "\n\n".join(line for line in rows["text"] if line.strip())
    tokens = list(tokenizer.encode(text))
    return mx.array(perplexity_windows(tokens, num_samples, seq_length))


def perplexity_of(
    model: Any,
    data: Any,
    batch_size: int,
    on_batch: Any = None,
) -> tuple[float, float]:
    """Perplexity and its standard error for ``data`` (an ``mx.array`` of token windows).

    ``on_batch(done_samples)`` is called before each batch (with the samples done so far)
    and after the last one; it may raise to stop the run.
    """
    import math
    import mlx.core as mx
    import mlx.nn as nn

    all_losses: list[mx.array] = []
    total = len(data)
    for i in range(0, total, batch_size):
        if on_batch is not None:
            on_batch(i)
        batch = data[i : i + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(losses)
        all_losses.append(losses.flatten())
    if on_batch is not None:
        on_batch(total)

    all_losses_cat = mx.concatenate(all_losses)
    ppl = math.exp(all_losses_cat.mean().item())
    std_dev = mx.sqrt(mx.var(all_losses_cat, ddof=1)).item()
    return ppl, ppl * (std_dev / math.sqrt(all_losses_cat.size))


def eval_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    dataset = request.get("dataset", "wikitext-2")
    num_samples = int(request.get("numSamples", 64))
    seq_length = int(request.get("seqLength", 512))
    batch_size = int(request.get("batchSize", 4))

    emit_progress("dataset", None, "Loading the evaluation text")
    data = load_perplexity_data(tokenizer, dataset, num_samples, seq_length)
    raise_if_cancelled(request)

    total_samples = len(data)
    start = time.monotonic()

    def on_batch(done: int) -> None:
        if done < total_samples:
            raise_if_cancelled(request)
        emit_progress(
            "evaluating",
            done / total_samples * 100.0,
            f"Sample {done}/{total_samples}",
            step=done,
            total=total_samples,
        )

    ppl, se_ppl = perplexity_of(model, data, batch_size, on_batch)

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


# How a task-accuracy run picks the model's answer. Saved on the run, so history can tell
# runs apart: runs from before this had no field and used SCORING_GENERATED.
SCORING_NEXT_TOKEN = "next_token"
SCORING_GENERATED = "generated"


def choice_token_ids(tokenizer: Any, letters: tuple[str, ...]) -> list[list[int]]:
    """Token ids that stand for each answer letter straight after ``Answer:``.

    A tokenizer has one token for the letter with a leading space and one
    without; which of the two follows the colon depends on the tokenizer, so
    both count for the letter.
    """
    ids: list[list[int]] = []
    for letter in letters:
        found: list[int] = []
        for text in (f" {letter}", letter):
            encoded = list(tokenizer.encode(text))
            if encoded and encoded[-1] not in found:
                found.append(int(encoded[-1]))
        ids.append(found)
    return ids


def pick_choice(scores: list[float], letters: tuple[str, ...]) -> str:
    """The letter with the highest score; the first one on a tie."""
    return letters[max(range(len(scores)), key=lambda index: scores[index])]


def _wired_memory(model: Any) -> Any:
    """mlx-lm's wired-memory context for a run of forward passes; a no-op where it does not exist."""
    try:
        from mlx_lm.generate import wired_limit
    except ImportError:
        import contextlib

        return contextlib.nullcontext()
    return wired_limit(model)


def next_token_scores(model: Any, prompt_tokens: list[int], token_ids: list[int]) -> list[float]:
    """The model's scores for ``token_ids`` as the token that follows the prompt.

    The prompt is read into a cache first and only its last token goes through
    the output layer: scoring every position of a few-shot prompt against the
    whole vocabulary made a run three times slower than generating the answer.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    tokens = mx.array(prompt_tokens)[None]
    cache = make_prompt_cache(model)
    if tokens.shape[1] > 1:
        model(tokens[:, :-1], cache=cache)
        mx.eval([entry.state for entry in cache])
    logits = model(tokens[:, -1:], cache=cache)[0, -1]
    return logits[mx.array(token_ids)].tolist()


def eval_task_accuracy(
    *,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Multiple-choice accuracy by next-token scoring.

    The prompt ends in ``Answer:``. The model's scores for the next token are
    read once and the answer is the letter (A to D) that scores highest. The
    scorer before this generated three tokens and searched them for a letter:
    with no worked examples in the prompt a small model starts a sentence
    instead ("Let's analyze"), which scored 9/100 on MMLU, below the 25 % of
    guessing. Reading the scores gives 29/100 there and the same result as
    before with five examples.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    from backend_service.task_datasets import CHOICE_LETTERS, load_task_data

    task_name = request.get("taskName", "mmlu")
    limit = int(request.get("limit", 100))
    num_shots = int(request.get("numShots", 5))

    emit_progress("dataset", None, f"Loading the {task_name} questions")
    tasks = load_task_data(task_name, limit, num_shots)
    raise_if_cancelled(request)

    letter_ids = choice_token_ids(tokenizer, CHOICE_LETTERS)
    flat_ids = [token_id for ids in letter_ids for token_id in ids]
    # Where each letter's ids sit in ``flat_ids``.
    letter_positions: list[list[int]] = []
    offset = 0
    for ids in letter_ids:
        letter_positions.append(list(range(offset, offset + len(ids))))
        offset += len(ids)
    correct = 0
    total = len(tasks)
    start = time.monotonic()

    emit_progress("evaluating", 0.0, f"Question 0/{total}", step=0, total=total, correct=0)
    # The same memory setting mlx-lm uses while it generates; without it the scoring ran about a fifth slower.
    with _wired_memory(model):
        for idx, task in enumerate(tasks):
            raise_if_cancelled(request)
            scores = next_token_scores(model, tokenizer.encode(task["prompt"]), flat_ids)
            by_letter = [max(scores[position] for position in positions) for positions in letter_positions]
            if pick_choice(by_letter, CHOICE_LETTERS) == task["correct_answer"]:
                correct += 1

            emit_progress(
                "evaluating",
                (idx + 1) / total * 100.0,
                f"Question {idx + 1}/{total}, {correct} correct",
                step=idx + 1,
                total=total,
                correct=correct,
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
        "scoringMethod": SCORING_NEXT_TOKEN,
    }
