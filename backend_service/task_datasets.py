"""Lightweight task dataset loaders for MMLU and HellaSwag benchmarks.

Downloads from Hugging Face datasets on first use. No external eval
framework required — just prompt→answer→score.
"""

from __future__ import annotations

import random
import re
from typing import Any


CHOICE_LETTERS = ("A", "B", "C", "D")


def load_task_data(task_name: str, limit: int, num_shots: int) -> list[dict[str, Any]]:
    if task_name == "mmlu":
        return _load_mmlu(limit, num_shots)
    elif task_name == "hellaswag":
        return _load_hellaswag(limit, num_shots)
    else:
        raise ValueError(f"Unknown task: {task_name}. Supported: mmlu, hellaswag")


def score_answer(
    task_name: str,
    model_output: str,
    correct: str,
    choices: list[str] | None = None,
) -> bool:
    """Extract the answer letter from model output and compare."""
    output = model_output.strip()
    # Try to find a letter in common patterns: "A", "(A)", "A.", "A)", "The answer is A"
    match = re.search(r"\b([A-D])\b", output)
    if match:
        return match.group(1).upper() == correct.upper()
    # Fallback: check if the output starts with the correct letter
    if output and output[0].upper() in CHOICE_LETTERS:
        return output[0].upper() == correct.upper()
    return False


def _load_mmlu(limit: int, num_shots: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=False)
    dev = load_dataset("cais/mmlu", "all", split="dev", trust_remote_code=False)

    # Group dev examples by subject for few-shot
    dev_by_subject: dict[str, list] = {}
    for item in dev:
        subj = item.get("subject", "general")
        dev_by_subject.setdefault(subj, []).append(item)

    items = list(ds)
    random.seed(42)
    random.shuffle(items)
    items = items[:limit]

    tasks = []
    for item in items:
        subject = item.get("subject", "general")
        choices = item["choices"]
        correct_idx = int(item["answer"])
        correct_letter = CHOICE_LETTERS[correct_idx]

        # Build few-shot examples from dev split
        shots = dev_by_subject.get(subject, [])[:num_shots]
        shot_text = ""
        for shot in shots:
            shot_choices = shot["choices"]
            shot_answer = CHOICE_LETTERS[int(shot["answer"])]
            shot_text += _format_mmlu_question(shot["question"], shot_choices, shot_answer)

        prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
        prompt += shot_text
        prompt += _format_mmlu_question(item["question"], choices, answer=None)

        tasks.append({
            "prompt": prompt,
            "correct_answer": correct_letter,
            "choices": choices,
            "max_tokens": 3,
        })

    return tasks


def _format_mmlu_question(
    question: str, choices: list[str], answer: str | None = None
) -> str:
    text = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        text += f"{CHOICE_LETTERS[i]}. {choice}\n"
    text += "Answer:"
    if answer is not None:
        text += f" {answer}\n\n"
    return text


def _load_hellaswag(limit: int, num_shots: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)

    items = list(ds)
    random.seed(42)
    random.shuffle(items)

    # Reserve first few for shots, rest for eval
    shot_items = items[:num_shots]
    eval_items = items[num_shots : num_shots + limit]

    shot_text = ""
    for shot in shot_items:
        endings = shot["endings"]
        correct_idx = int(shot["label"])
        shot_text += _format_hellaswag_question(
            shot["ctx"], endings, CHOICE_LETTERS[correct_idx]
        )

    tasks = []
    for item in eval_items:
        endings = item["endings"]
        correct_idx = int(item["label"])
        correct_letter = CHOICE_LETTERS[correct_idx]

        prompt = "Choose the most plausible continuation.\n\n"
        prompt += shot_text
        prompt += _format_hellaswag_question(item["ctx"], endings, answer=None)

        tasks.append({
            "prompt": prompt,
            "correct_answer": correct_letter,
            "choices": endings,
            "max_tokens": 3,
        })

    return tasks


def _format_hellaswag_question(
    context: str, endings: list[str], answer: str | None = None
) -> str:
    text = f"Context: {context}\n"
    for i, ending in enumerate(endings):
        text += f"{CHOICE_LETTERS[i]}. {ending}\n"
    text += "Answer:"
    if answer is not None:
        text += f" {answer}\n\n"
    return text
