# CLI recipes

Practical end-to-end workflows you can copy + paste. Each recipe assumes the
backend is running on `127.0.0.1:8876` (the default).

## Smoke test a fresh install

```bash
./scripts/chaosengine-cli health
./scripts/chaosengine-cli status | jq '.runtime.state'
./scripts/chaosengine-cli routes | jq '.count'
./scripts/chaosengine-cli call GET /api/setup/mtplx-status | jq '.'
```

If every command returns 200 / non-error and you see 125 routes, the
install is healthy.

## Discover + download a model

```bash
# Catalog search (curated models)
./scripts/chaosengine-cli search "qwen3" | jq '.results[].repo'

# Hugging Face Hub search (everything)
./scripts/chaosengine-cli hub-search "qwen3.5 14b" --limit 10

# Look at what files a repo carries
./scripts/chaosengine-cli hub-files "Qwen/Qwen3.5-14B"

# Trigger a download
./scripts/chaosengine-cli download "Qwen/Qwen3.5-14B"

# Watch progress
watch -n2 './scripts/chaosengine-cli download-status | jq ".active[] | {repo, percent}"'
```

## Load + chat

```bash
./scripts/chaosengine-cli load "Qwen/Qwen3.5-14B" \
    --context 32768 --spec --fused-attention

./scripts/chaosengine-cli prompt "Write a haiku about a chaotic engine" \
    --max-tokens 128 --stream --metrics
```

## Batch prompts from a file

```bash
# prompts.txt: one prompt per line
while IFS= read -r line; do
    ./scripts/chaosengine-cli prompt "$line" --max-tokens 256 --json \
        > "out/$(date +%s%N).json"
done < prompts.txt
```

## Benchmark a model

```bash
./scripts/chaosengine-cli load "Qwen/Qwen3-4B" --context 8192
./scripts/chaosengine-cli bench "Qwen/Qwen3-4B" --runs 3 --prompts short

./scripts/chaosengine-cli benchmark-run --body '{
    "modelRef": "Qwen/Qwen3-4B",
    "mode": "perplexity",
    "dataset": "wikitext-2",
    "runs": 1
}'
```

## Compare two models on the same prompt

```bash
./scripts/chaosengine-cli compare --body '{
    "prompt": "Explain the difference between MTPLX and DFlash in one paragraph.",
    "slots": [
        {"modelRef": "Qwen/Qwen3.5-14B", "speculativeDecoding": true},
        {"modelRef": "Qwen/Qwen3.5-9B",  "speculativeDecoding": true}
    ]
}'
```

## Generate an image, watch progress, save the output

```bash
./scripts/chaosengine-cli image-generate "a desert sunset, photoreal, 35mm" \
    --model FLUX.1-schnell --steps 4 --width 1024 --height 1024 \
    --seed 42 | tee /tmp/gen.json

JOB=$(jq -r '.jobId' < /tmp/gen.json)

# Poll until done
while :; do
    state=$(./scripts/chaosengine-cli image-progress \
        | jq -r --arg j "$JOB" '.jobs[] | select(.jobId==$j) | .state')
    [[ "$state" == "complete" || "$state" == "failed" ]] && break
    sleep 1
done

# Fetch the artifact
./scripts/chaosengine-cli image-outputs | jq '.outputs[0]'
```

## Generate a Wan 2.1 video on Apple Silicon

```bash
# One-time: install + convert the model
./scripts/chaosengine-cli wan-install "Wan-AI/Wan2.1-T2V-1.3B"

# Watch the install + convert progress
./scripts/chaosengine-cli call GET /api/setup/install-mlx-video-wan/status

# Confirm conversion landed
./scripts/chaosengine-cli wan-inventory

# Generate
./scripts/chaosengine-cli video-generate \
    "a fox running through a forest at dawn" \
    --model "Wan-AI/Wan2.1-T2V-1.3B" \
    --frames 5 --fps 16 --steps 4 --seed 42
```

## Diagnose a failed load

```bash
./scripts/chaosengine-cli diagnostics-snapshot > /tmp/snap.json

# Capabilities + recent errors
jq '.capabilities, .recentErrors' < /tmp/snap.json

# Live log tail
./scripts/chaosengine-cli diagnostics-log-tail --lines 200
```

## Install MTPLX from scratch

```bash
./scripts/chaosengine-cli mtplx-install
# Phase events stream to stderr; final JSON to stdout.

./scripts/chaosengine-cli mtplx-status | jq '.installed, .version'
```

If install fails, see
[Troubleshooting → MTPLX install issues](../troubleshooting/mtplx-install-issues.md).

## Run the E2E suite

```bash
# Smoke (~60s)
./scripts/e2e_test_suite.py --smoke

# Full sweep
./scripts/e2e_test_suite.py

# Specific phases
./scripts/e2e_test_suite.py --phases 0,1,7
```

Reports land in `~/.chaosengine/test-results/`.

## See also

- [CLI overview](overview.md)
- [CLI reference](reference.md)
- [Automation](automation.md)
