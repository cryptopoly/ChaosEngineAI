#!/usr/bin/env python3
"""Performance gate — compare a fresh perf-baseline.py run against PERF_BASELINE.md.

Reads the JSON blob produced by ``perf-baseline.py`` and validates that no
metric has regressed beyond the configured tolerance. Used at the end of
Phase 5 to block merges that silently sandbag throughput.

Usage:
    # Generate a baseline run
    .venv/bin/python scripts/perf-baseline.py --output /tmp/baseline.json

    # Compare against the captured baseline (non-zero exit on regression)
    .venv/bin/python scripts/perf-gate.py /tmp/baseline.json

    # Custom tolerance (default 5%)
    .venv/bin/python scripts/perf-gate.py /tmp/baseline.json --tolerance 0.10

Locked-in floors (matching PERF_BASELINE.md) live below as the
``BASELINES`` table. Update them deliberately when a refactor produces a
real, validated win — never when a regression squeezes the gate the other
way. Each entry names the gate label, the metric to read, and the floor
the comparator measures against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Captured from the v0.7.6 → v0.8.0 baseline (PERF_BASELINE.md, 2026-05-09).
# ``key``: dot-path inside the perf-baseline.py JSON output.
# ``floor``: the minimum acceptable value when ``higher_is_better`` is True,
# the maximum acceptable value when False.
# ``higher_is_better``: True for tokens-per-second, False for wall-time.
BASELINES: list[dict] = [
    {
        "label": "text_throughput",
        "key": "text.tokens_per_second",
        "floor": 297.0,
        "higher_is_better": True,
        "unit": "tok/s",
    },
    # Image + video gates flip on once PERF_BASELINE.md captures their
    # initial numbers. Adding them here without floors silently passes;
    # we want a hard error instead so the gate stays meaningful.
]


DEFAULT_TOLERANCE = 0.05  # 5%


def _read_metric(payload: dict, key: str) -> float | None:
    """Read a metric from the perf-baseline.py JSON payload.

    Keys are dotted: ``<label>.<metric>`` (e.g. ``text.tokens_per_second``)
    where ``<label>`` matches an entry's ``label`` inside the
    ``results`` array and ``<metric>`` is a numeric field on that entry.
    Returns ``None`` when the entry is absent, skipped, errored, or the
    field isn't numeric.
    """
    if "." not in key:
        return None
    label, metric = key.split(".", 1)
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("label") != label:
            continue
        if not entry.get("ok"):
            return None
        value = entry.get(metric)
        if isinstance(value, (int, float)):
            return float(value)
        return None
    return None


def _check(payload: dict, baseline: dict, tolerance: float) -> tuple[bool, str]:
    """Return (passed, message)."""
    label = baseline["label"]
    key = baseline["key"]
    floor = float(baseline["floor"])
    higher_is_better = bool(baseline["higher_is_better"])
    unit = baseline.get("unit", "")

    actual = _read_metric(payload, key)
    if actual is None:
        return False, f"{label}: missing metric ``{key}`` in run output"

    if higher_is_better:
        # actual must be >= floor * (1 - tolerance)
        threshold = floor * (1.0 - tolerance)
        passed = actual >= threshold
        delta_pct = ((actual - floor) / floor) * 100.0 if floor else 0.0
    else:
        # actual must be <= floor * (1 + tolerance)
        threshold = floor * (1.0 + tolerance)
        passed = actual <= threshold
        delta_pct = ((actual - floor) / floor) * 100.0 if floor else 0.0

    arrow = "↑" if delta_pct >= 0 else "↓"
    line = (
        f"{'PASS' if passed else 'FAIL'} {label}: {actual:.2f} {unit} "
        f"vs floor {floor:.2f} {unit} ({arrow}{abs(delta_pct):.1f}% "
        f"@ ±{tolerance * 100:.0f}% gate)"
    )
    return passed, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Perf gate against captured baselines")
    parser.add_argument("run", type=Path, help="JSON file produced by perf-baseline.py")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Allowed regression ratio (default {DEFAULT_TOLERANCE:.2f} = 5%%)",
    )
    args = parser.parse_args(argv)

    if not args.run.is_file():
        print(f"perf-gate: {args.run} does not exist", file=sys.stderr)
        return 2

    try:
        payload = json.loads(args.run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"perf-gate: cannot parse {args.run}: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for baseline in BASELINES:
        passed, line = _check(payload, baseline, args.tolerance)
        print(line)
        if not passed:
            failures.append(line)

    if failures:
        print(f"\nperf-gate: {len(failures)} metric(s) regressed beyond ±{args.tolerance * 100:.0f}%", file=sys.stderr)
        return 1
    print(f"\nperf-gate: all {len(BASELINES)} metric(s) within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
