# GPU detection

ChaosEngineAI probes for GPU support at backend startup and exposes the
result through:

- `GET /api/health` — the `nativeBackends` block.
- `GET /api/system/gpu-status` — a structured GPU detail.
- `GET /api/metrics/gpu` — live utilization metrics.

If the dashboard says "No GPU detected" on a machine you know has one,
or live GPU metrics don't show up while a generation is in flight,
walk through the steps below.

## Apple Silicon

On Apple Silicon the "GPU" is the unified-memory GPU baked into the
SoC. There's no separate driver to check; MLX talks to Metal directly.

What to verify:

- `./scripts/chaosengine-cli health | jq '.nativeBackends.mlxUsable'`
  should be `true`.
- `./scripts/chaosengine-cli gpu-status | jq '.'` should report the
  Metal device.
- During a generation, `./scripts/chaosengine-cli metrics-gpu` reports
  live MPS / Metal utilization (sampled from the Activity Monitor APIs).

If the live metrics endpoint reports zero utilization during an
active generation, the model probably isn't actually using MPS — check
the per-turn host strip's engine field. If it shows
`engine: "llama-server"` instead of `engine: "mlx"`, you're on the
CPU GGUF path.

## Linux + CUDA

CUDA detection runs through `torch.cuda.is_available()` + `nvidia-smi`.

What to verify:

- `./scripts/chaosengine-cli health | jq '.nativeBackends'` should
  include CUDA-aware backend flags.
- `nvidia-smi` works from the same shell that launched the backend.
- The CUDA + PyTorch versions match — mismatched torch / cuda installs
  silently fall back to CPU.

Common gotchas:

- **`torch` was pip-installed before CUDA tooling.** Reinstall torch
  with the right CUDA wheel:
  ```bash
  .venv/bin/pip install --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision
  ```
  Or use the **Setup → Install CUDA torch** button in the app, which
  wraps the same install.

- **The user running the backend doesn't have GPU permissions.** On
  Linux, ensure your user is in the `video` group (some distros) or
  has read access to `/dev/nvidia*`.

- **Driver / runtime mismatch.** `nvidia-smi` reports the driver
  version; `torch.version.cuda` reports the runtime. They need to be
  compatible — `nvidia-smi` will refuse to run if not.

## Windows + CUDA

Same as Linux but the path resolution is different. The backend
defers to `torch.cuda.is_available()` first, then `nvidia-smi.exe`.

Common gotchas:

- **Conda / system Python mix.** If you've installed PyTorch into a
  conda env but the backend runs from a different Python, CUDA
  detection will fail. Match the interpreter.
- **WSL.** WSL 2 with CUDA-on-WSL is supported but the GPU bundle
  installer assumes native Windows. Use the WSL Linux path instead.

## "GPU bundle" install

The GPU bundle installer fetches a pre-built set of CUDA wheels
matched to a specific CUDA version. It's a convenience wrapper for
the "first time on a new box" case.

Status endpoint:

```bash
./scripts/chaosengine-cli gpu-bundle-status
./scripts/chaosengine-cli gpu-bundle-info
```

If you've already got a working torch + CUDA install, skip the bundle
— installing it on top of an existing setup can cause version drift.

## CPU fallback

When no GPU is detected, the runtime falls back to CPU. Everything
works but it's slow. The dashboard reports `engine: "cpu"` and the
per-turn host strip carries the same. There's no toggle for "force
GPU" — the engine selection is automatic based on what's actually
available.

## Refreshing the probe

After installing drivers / torch / CUDA tooling, refresh the
capability probe so the backend re-detects:

```bash
./scripts/chaosengine-cli setup-refresh
```

Then check `health` again. If the probe still says no GPU, the issue
is in the underlying detection — file an issue with:

- `./scripts/chaosengine-cli diagnostics-snapshot > snap.json` output.
- `nvidia-smi` output (Linux / Windows).
- `system_profiler SPDisplaysDataType` output (macOS).

## See also

- [Model load failures](model-load-failures.md)
- [System requirements](../getting-started/system-requirements.md)
