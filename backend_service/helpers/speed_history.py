"""This machine's own measured decode speed for a model, from past runs.

Every chat reply records tok/s with the model's ref, path and launch
settings, and so does every benchmark run. When enough of them exist for the
model and cache strategy being previewed, their median is a better number
than any estimate.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Iterable

# Short replies are dominated by per-request overhead; ignore them.
_MIN_COMPLETION_TOKENS = 32
_MIN_SAMPLES = 2
_MAX_SAMPLES = 20


def _samples(chat_sessions: Iterable[dict[str, Any]], benchmark_runs: Iterable[dict[str, Any]]):
    for session in chat_sessions or []:
        for message in session.get("messages") or []:
            metrics = message.get("metrics")
            if isinstance(metrics, dict):
                yield metrics.get("generatedAt") or "", metrics
    for run in benchmark_runs or []:
        if isinstance(run, dict):
            yield run.get("measuredAt") or "", run


def measured_decode_tok_s(
    chat_sessions: Iterable[dict[str, Any]],
    benchmark_runs: Iterable[dict[str, Any]],
    *,
    model_path: str | None,
    model_repo: str | None,
    strategy: str,
) -> float | None:
    """Median tok/s of recent real runs of this model with ``strategy``, or None."""
    keys = {k for k in (model_path, model_repo) if k}
    if not keys:
        return None
    matched: list[tuple[str, float]] = []
    for when, record in _samples(chat_sessions, benchmark_runs):
        if record.get("backend") in (None, "mock"):
            continue
        if record.get("speculativeDecoding"):
            continue  # a draft model changes the speed; the preview is for plain decode
        if (record.get("cacheStrategy") or "native") != strategy:
            continue
        if not keys & {record.get("modelPath"), record.get("modelRef"), record.get("canonicalRepo"), record.get("path")}:
            continue
        tok_s = record.get("tokS")
        if not isinstance(tok_s, (int, float)) or tok_s <= 0:
            continue
        if int(record.get("completionTokens") or 0) < _MIN_COMPLETION_TOKENS:
            continue
        matched.append((str(when), float(tok_s)))
    if len(matched) < _MIN_SAMPLES:
        return None
    matched.sort(key=lambda item: item[0], reverse=True)
    return round(median(tok for _, tok in matched[:_MAX_SAMPLES]), 1)
