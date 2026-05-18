# Compare mode

Compare mode streams the same prompt through two models side-by-side, each
with its own runtime settings, and renders both responses with independent
metrics. It's the fastest way to answer questions like:

- "Does the new 27B quant actually pay its keep over the 14B at the same
  context length?"
- "Does enabling DFlash on this model produce identical tokens to vanilla
  generation?"
- "How does FLUX.1-dev compare to FLUX.1-schnell on the same image prompt?"
  (The image equivalent lives in the Image Studio gallery, not Compare.)

## Setup

The Compare tab takes two **slots**. Each slot picks:

- A model ref (anything in your library)
- An engine (MLX / GGUF / vLLM / remote — auto-selected from the ref)
- Cache strategy + bits + fp16-layer count
- Speculative decoding toggle (DFlash / DDTree / MTPLX)
- Context length and sampling presets

Slots share **only the prompt**. Everything else is independently
configurable, so you can compare two different models, or the same model with
two different cache strategies, or the same model with DFlash on vs off.

## Streaming

Hit Compare and both slots start streaming in parallel. The shared prompt
panel sits above; each slot owns its own bubble below with its own:

- **Stream area** — tokens render as they arrive.
- **Metrics block** — tok/s, TTFT, eval-tok count, peak memory.
- **Runtime note** — the cache strategy and speculative-decoding mode the
  backend actually routed through (e.g. `"dflash on"`, `"mtplx active"`,
  `"speculative fallback to native"`).

When both streams finish, the metrics block flips to its final state and the
two responses sit side-by-side for review. The Compare tab keeps the last
several runs in memory; switch slots or refresh the prompt to start over.

## What it proves

Compare mode is **routing-aware**. The `runtimeNote` on each slot makes it
explicit which backend handled the turn — so when DFlash silently falls back
to native generation because the draft model isn't on disk, you see it
immediately rather than guessing why the speedup didn't show up.

## CLI equivalent

The compare endpoint is reachable from the CLI via:

```bash
./scripts/chaosengine-cli compare --body '{
  "prompt": "Explain MTPLX in one paragraph.",
  "slots": [
    {"modelRef": "Qwen/Qwen3.6-27B", "engine": "mlx", "speculativeDecoding": true},
    {"modelRef": "Qwen/Qwen3-Coder-Next", "engine": "mlx", "speculativeDecoding": true}
  ]
}'
```

The endpoint accepts the full payload shape from the UI — see
`POST /api/chat/compare` in the [API reference](../reference/api.md).

## Caveats

- Both slots run against the **same** backend, sharing CPU / GPU / RAM.
  Throughput numbers will not match what you'd see running each model in
  isolation; the slower slot drags on the faster one's memory pressure.
- The warm pool tries to keep both models hot; if either model gets evicted
  mid-run the metrics will reflect a cold-start. This usually only happens
  when memory is tight.
- Streaming is round-robin at the chunk level; the per-slot tok/s is computed
  from each slot's own clock, not from the shared wall time.
