# Model load failures

When a load fails, the runtime emits a structured error with a
`runtimeNote` explaining what tried and what failed. The Chat tab
surfaces this in the host strip, but the easier way to debug is the
diagnostics snapshot.

```bash
./scripts/chaosengine-cli diagnostics-snapshot | jq '.recentErrors'
```

## Common failure patterns

### `MLX is not available in this environment`

You're either on non-Apple-Silicon hardware, or `mlx` / `mlx_lm`
aren't importable in the bundled Python. See the
[FAQ](faq.md#mlx-is-not-available-in-this-environment).

### `This is a GGUF model which requires llama-server. Install with: brew install llama.cpp`

The autodetect saw a `.gguf` path but couldn't resolve a
`llama-server` binary. Fixes:

- macOS: `brew install llama.cpp`.
- Linux: `apt install llama.cpp` (Ubuntu 22.04+) or build from source.
- Or just set `CHAOSENGINE_LLAMA_SERVER` to a binary you've built.

### `Cache type 'turbo3' is not supported by this llama-server`

You picked the TurboQuant cache strategy but the standard
`llama-server` binary is being used. The TurboQuant cache types need
the `llama-server-turbo` fork. Either:

- Build the turbo binary: `scripts/build-llama-turbo.sh`. The binary
  lands at `~/.chaosengine/bin/llama-server-turbo` and the next load
  picks it up automatically.
- Or switch the cache strategy to native f16 in the launch modal.

The engine pre-validates cache types against the binary's `--help`
output, so this error fires early — before the server attempts to
start.

### `Loading 27B model needs ~30 GB; available 18 GB`

The fit-in-memory check refused the launch. Either:

- Drop to a smaller quant (4-bit instead of 8-bit).
- Drop the context length (the KV cache scales linearly).
- Pick a more aggressive cache strategy (TurboQuant 3-bit).
- Disable the fit-in-memory toggle if you accept the swap risk.

### `model_index.json missing from <path>`

The model directory you pointed the loader at doesn't have the
diffusers / transformers metadata it expects. Usually this means an
incomplete download. Check `~/AI_Models/<repo>/` — every file in
the upstream repo should be present.

```bash
./scripts/chaosengine-cli list-weights "<path>"
```

If files are missing, re-trigger the download via:

```bash
./scripts/chaosengine-cli download <repo>
```

### `MTPLX server failed to bind to port`

The MTPLX subprocess couldn't get an open port. Almost always means
a previous MTPLX process is still alive. Clean it up:

```bash
ps aux | grep mtplx
kill <pid>
```

Then re-trigger the load. The backend should pick a free port on the
retry.

### Hardened-runtime crash on macOS release builds

Symptom: model loads in dev (`npm run tauri:dev`) but the packaged
`.dmg` crashes the worker subprocess on every load attempt.

This is usually a missing entitlement on the bundled Python — Apple's
hardened runtime requires `com.apple.security.cs.allow-jit` for
mlx-lm's JIT path, and `com.apple.security.cs.disable-library-validation`
for the bundled Python loader. Check `src-tauri/entitlements.plist`
and the build's notarization log.

The macOS spam suppression in FU-038 was a side-effect of fixing the
related malloc-debug noise; the underlying entitlement plist is
already correct in shipped builds.

## Diagnosis flow

When a load fails:

1. **Check `runtimeNote` on the failed load.** Look for the exact
   stop-condition.
2. **Snapshot the diagnostics.**
   `./scripts/chaosengine-cli diagnostics-snapshot > snap.json`
   then `jq '.recentErrors, .capabilities' < snap.json`.
3. **Tail the logs.**
   `./scripts/chaosengine-cli diagnostics-log-tail --lines 500`.
4. **Try the same load via the CLI.** Removes the UI from the
   equation:
   ```bash
   ./scripts/chaosengine-cli load <repo> --context 4096
   ```
5. **Try a smaller model** (a 1.3B Qwen3 4-bit MLX). If that loads
   cleanly, the bigger model is the issue, not the runtime.

If you can reproduce against a small model, file an issue with the
diagnostics snapshot + the load command. Include the model size,
quant scheme, cache strategy, and platform.

## See also

- [FAQ](faq.md)
- [GPU detection](gpu-detection.md)
- [Orphan workers](orphan-workers.md)
