# Benchmarks

The Benchmarks tab runs the loaded model against a structured workload and
records the result in a persistent history with side-by-side diffs across
prior runs.

## Benchmark modes

| Mode | What it measures | Typical use |
|---|---|---|
| **Throughput** | Tokens-per-second generation rate + TTFT against a fixed prompt or preset. | Comparing engines, quants, cache strategies. |
| **Perplexity** | Standard LM perplexity over a held-out dataset. | Checking that a new quant didn't tank quality. |
| **Task accuracy** | MMLU / HellaSwag accuracy on a sampled subset. | Multi-choice eval for instruction-tuned variants. |

## Running a benchmark

The configure pane collects:

- **Model** — any model in the library.
- **Mode** — throughput / perplexity / task accuracy.
- **Workload** — preset prompt list (short / medium / long), an uploaded
  custom prompt set, or a built-in eval dataset (MMLU / HellaSwag) for
  task-accuracy mode.
- **Token budget** — caps the longest single generation; mostly relevant
  for throughput mode.
- **Sampling presets** — same `temperature` / `top_p` / `seed` / etc. as
  the chat composer.
- **Runs** — repeat count; the report aggregates min / median / max / std-dev.

Hit Run and the runner streams progress:

- Per-prompt tok/s, TTFT, and memory pressure.
- For perplexity / task-accuracy: per-batch loss / accuracy.
- Phase log: load, warm, eval, write.

## Report card

When the run finishes:

- **Throughput mode** — bar chart of tok/s per prompt, a TTFT histogram, a
  peak-memory readout, and the full per-prompt sample.
- **Perplexity mode** — final PPL + standard error + the slice of dataset
  used.
- **Task accuracy** — per-subject accuracy table + total + standard error.

Every run lands in the persistent history. Pick two runs from the History
view and the page diffs them across throughput, latency, and quality
metrics. Red / green delta colors make regressions obvious.

## Cross-strategy matrix

For a structured sweep across every supported (cache strategy × spec-dec
method × representative model) combination, use the standalone runner:

```bash
# Quick smoke (~5 min on M-series; CI-friendly)
.venv/bin/python scripts/cache-strategy-matrix.py --quick

# Full sweep (~20 min; gates a release)
.venv/bin/python scripts/cache-strategy-matrix.py
```

It writes a CSV + Markdown report to `~/.chaosengine/test-results/` and
asserts the legacy `chaosengine` / `rotorquant` strategy ids coerce to
`turboquant` (the FU-030 contract — exits with code 2 if either
regresses).

## Programmatic alternative

```bash
# Quick benchmark — load + N-prompt sweep + tok/s
./scripts/chaosengine-cli bench "Qwen/Qwen3-4B" --runs 3

# Full-payload benchmark (perplexity, task-accuracy)
./scripts/chaosengine-cli benchmark-run --body '{
  "modelRef": "Qwen/Qwen3.5-14B",
  "mode": "perplexity",
  "dataset": "wikitext-2",
  "runs": 1
}'
```

## What benchmarks don't tell you

- **Cold-start time.** The runner warms the model first; load latency is
  not part of the headline number. Cold-start matters if you reload often;
  see the warm-pool counters in the Server tab for that signal.
- **Quality of long-form generation.** PPL on a dataset is a proxy for
  language modeling, not a vibe check. Compare mode in Chat is the right
  tool for "does this response feel right."
- **Memory under concurrency.** The runner is single-threaded. Concurrent
  request shape is a separate question; the OpenAI-compatible server's
  active-connection counter is the right place to look.
