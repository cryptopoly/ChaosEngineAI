# Fused attention

Fused attention is a runtime toggle that asks the engine to use a single
fused kernel for the attention block instead of the unfused
`softmax(QK^T / sqrt(d)) V` sequence. It's a transparent speed win where it's
supported, and a no-op everywhere else.

## When it helps

- **Apple Silicon, MLX worker.** mlx-lm exposes a fused-attention path on
  recent silicon (M3+). Enabling it on smaller models is usually a small win
  (~5 - 15% prompt-eval throughput, depending on context length); on larger
  models the win is larger because the kernel keeps more of the working set
  in registers.
- **llama.cpp GGUF.** The bundled / Homebrew `llama-server` accepts
  `--flash-attn` when the model architecture supports it. Generally a win
  for prompt eval; effect on token generation is smaller.
- **vLLM (Linux + CUDA).** Always uses fused attention internally. The toggle
  is effectively a no-op here.

## When it doesn't

- **CPU-only inference.** No fused kernel path; toggle is a no-op.
- **MLX worker on older silicon.** The fused kernel falls back to the
  unfused implementation; the toggle remains safe to enable.
- **Some quant types on llama.cpp.** Older quants (Q3_K, Q2_K) sometimes
  reject `--flash-attn`; the engine logs the rejection in `runtimeNote`
  and falls back to standard attention.

## Where the toggle lives

- **Launch modal.** "Fused attention" checkbox under Runtime Controls.
- **Settings → Defaults.** The default for new model loads.
- **CLI.** `--fused-attention` / `--no-fused-attention` on
  `chaosengine-cli load`.

## How it interacts with cache strategies

| Strategy | Fused-attention behaviour |
|---|---|
| Native f16 | Fully supported on both MLX and `llama.cpp`. |
| TurboQuant (MLX) | Supported — the cache and the attention kernel are orthogonal. |
| TurboQuant (`llama-server-turbo`) | Supported. The turbo fork inherits llama.cpp's flash-attention path; the turbo cache types compose with it. |
| TriAttention (vLLM) | vLLM picks the kernel; the toggle is a no-op. |
| TriAttention (MLX direct path) | Falls back to standard attention — the TriAttention compressor patches the cache, not the kernel. |

If you enable fused attention and the engine downgrades it for any reason,
`runtimeNote` reports the downgrade so you can see what the backend
actually did.

## Diagnostics

The per-turn host strip on Chat / Compare turns includes the active engine
+ binary + cache strategy. The fused-attention state isn't called out
explicitly there, but it surfaces in the runtime snapshot:

```bash
./scripts/chaosengine-cli runtime | jq '.attention, .runtimeNote'
```

If you're benchmarking and the toggle isn't showing the expected win, check
the diagnostics snapshot's `attention` field — it'll report whether the
fused path actually engaged.
