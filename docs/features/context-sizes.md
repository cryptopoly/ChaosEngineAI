# Context sizes

The context length you launch a model with directly controls how much memory
the KV cache will occupy at peak. The cache strategy you pick — native f16,
TurboQuant, TriAttention — scales the same baseline by a compression ratio.

This page is a quick reference for picking a context length without running
out of memory.

## Baseline math

The KV cache size for a single sequence is roughly:

```
baseline_bytes = 2 * num_layers * num_kv_heads * head_dim * context_tokens * 2
```

`* 2` at the front accounts for the K and V projections; the trailing `* 2`
is f16 bytes-per-element. For Grouped Query Attention (GQA) models, the
`num_kv_heads` is smaller than `num_heads` — `cache_compression.CacheStrategy.estimate_cache_bytes`
defaults to `num_heads` for plain multi-head attention but accepts a
`num_kv_heads` override.

A representative table for common context sizes (f16, GQA model with
`num_layers=64`, `num_kv_heads=8`, `head_dim=128`):

| Context | Baseline KV (f16) | TurboQuant 3-bit | TriAttention 3-bit |
|---|---|---|---|
| 8 k | ~2.0 GB | ~0.5 GB | ~0.5 GB |
| 16 k | ~4.0 GB | ~1.0 GB | ~1.0 GB |
| 32 k | ~8.0 GB | ~2.0 GB | ~2.0 GB |
| 64 k | ~16.0 GB | ~4.0 GB | ~4.0 GB |
| 128 k | ~32.0 GB | ~8.0 GB | ~8.0 GB |

The exact numbers depend on the model. The launch modal's **Cache preview**
panel shows the precise baseline + optimised numbers for the model you're
about to load.

## Practical ceilings

For total memory pressure (model weights + KV cache + framework overhead +
diffusion pipeline residency):

| Hardware | Comfortable context |
|---|---|
| 16 GB Mac | 4 - 8 k for 4 - 7B 4-bit. |
| 32 GB Mac | 16 - 32 k for 4 - 14B 4-bit. |
| 64 GB Mac | 32 - 64 k for 14 - 27B 4-bit; 128 k with TurboQuant. |
| 128 GB Mac Studio / NVIDIA 24 GB | 128 k for 14B; 64 - 128 k for 27 - 35B with TurboQuant. |
| Multi-GPU CUDA | model-dependent; vLLM splits cleanly across GPUs. |

These are guidelines — actual ceilings vary based on prompt length, the
specific model's architecture, and whether you're running other workloads.

## Setting the default

**Settings → Defaults → Context tokens** is the value the launch modal
pre-populates with. The per-launch modal can override it. The CLI accepts
`--context <N>` on `chaosengine-cli load`:

```bash
./scripts/chaosengine-cli load "Qwen/Qwen3.5-14B" --context 32768
```

## "Fit in memory" toggle

The launch modal has a **Fit in memory** safety toggle. With it enabled,
the runtime estimates the requested launch profile's working set, compares
against available RAM, and refuses to launch if it would push the system
into swap. The launch modal surfaces the predicted peak alongside the
available memory so you can manually adjust before launching.

This is a soft safety check — the runtime can't actually predict every
factor (other apps, browser tabs, the OS's adaptive memory pressure). If
you're confident, disable the toggle and let it rip.

## See also

- [Cache strategies](cache-strategies.md) — the compression ratios behind
  the table above.
- [Benchmarks](../usage/benchmarks.md) — measure the actual peak with the
  perplexity / throughput runners.
- `GET /api/cache/preview` — the same calculator the launch modal uses.
