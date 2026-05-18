# FAQ

## "Backend offline" on the Dashboard

The Tauri shell spawns the Python backend on `127.0.0.1:8876` and waits for
`/api/health` to return 200. If the dashboard banner is red:

1. Try **Settings → Diagnostics → Re-extract runtime**. This forces the
   bundled Python runtime + binaries to re-extract from the app bundle.
2. Check the **Diagnostics log tail** for the failure mode. Common
   culprits: a port already bound on 8876, a corrupted runtime
   extraction, missing OS-level dependencies (Linux: `libwebkit2gtk`,
   `librsvg2-dev`).
3. From a terminal: `curl http://127.0.0.1:8876/api/health` should
   return JSON. If it doesn't, the backend never started; the Tauri
   stderr buffer (View → Developer Tools → Console in dev builds, or
   `right-click → Inspect Element` in release builds — added in FU-037)
   carries the actual stack trace.

## "MLX is not available in this environment"

The MLX runtime requires Apple Silicon. On Intel Macs / x86_64 Linux /
Windows, MLX falls back gracefully and the launch modal shows GGUF /
vLLM options instead.

If you're on Apple Silicon and still get this error:

- Check the bundled Python is native arm64:
  `file /path/to/python` should report `arm64`.
- Confirm `mlx` and `mlx_lm` are importable: `.venv/bin/python -c
  "import mlx, mlx_lm; print(mlx.__version__, mlx_lm.__version__)"`.
- Inspect the capabilities probe:
  `./scripts/chaosengine-cli health | jq '.nativeBackends'`. Look for
  `mlxUsable: true`.

## DFlash toggle is hidden in the launch modal

After FU-034, the launch modal hides options the user has no in-app
path to recover. The DFlash toggle is hidden when:

- The model has no draft in [`DRAFT_MODEL_MAP`](../features/dflash.md).
- The selected engine is GGUF (DFlash needs MLX / vLLM today).

If your model is in the registry but the toggle is still missing,
check `./scripts/chaosengine-cli status | jq '.capabilities.dflashAvailable'`.
A `false` reading means `dflash-mlx` / `dflash` isn't installed — fix
by `.venv/bin/pip install dflash-mlx` on Apple Silicon, or
`.venv/bin/pip install dflash` on CUDA.

## "DFLASH unavailable for X: no compatible draft model is registered"

The model isn't in `DRAFT_MODEL_MAP` and doesn't match any alias. Two
fixes:

- If the model is a community quant of a registry-listed canonical
  repo, add the alias to
  [`dflash/__init__.py::_ALIASES`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/dflash/__init__.py).
- If the model genuinely has no drafter (z-lab hasn't released one),
  there's nothing to fix — DFlash isn't available for this model. The
  E2E suite reports `skip` for these cases rather than `fail`.

## "speculativeDecoding=true" but `runtimeNote` shows only `mlx`

The model is in your library, you flipped the toggle, but the
runtime didn't actually route through MTPLX / DFlash. Check:

1. **MTPLX:** is the model in `MTP_MODEL_MAP`? Is MTPLX installed?
   `./scripts/chaosengine-cli mtplx-status` should report `installed: true`.
2. **DFlash:** is `dflashAvailable: true`? Is the draft on disk? The
   first DFlash load downloads the drafter; subsequent loads are fast.
   Check `~/AI_Models/z-lab/` for the drafter checkpoint.

The runtimeNote string is the source of truth — when it shows only
`mlx`, the speculative-decoding path didn't fire. The diagnostics
snapshot's `recentErrors` block usually carries the underlying reason.

## Generation crashes the Chat tab

FU-037 added a React error boundary that scopes crashes to the
current tab. If a single bad message took down only the Chat tab,
hit **Try again** in the boundary's fallback UI to recover.

The FU-039 fix landed alongside that — when a model emitted a tool
call with `arguments: null`, the old `ToolCallCard` rendered called
`Object.entries(null)` and crashed forever. Both backend and frontend
now coerce `null` / non-dict arguments to `{}`.

If your Chat tab is wedged on an older session that pre-dates the
fix, the boundary will recover it; you can also delete the offending
turn from the session JSON under
`~/Library/Application Support/com.chaosengineai.desktop/sessions/`.

## Model download stuck at 99%

Hugging Face downloads are resumable. If the spinner says 99% forever:

```bash
./scripts/chaosengine-cli download-status | jq '.active'
./scripts/chaosengine-cli download-cancel <repo>
./scripts/chaosengine-cli download <repo>  # resumes
```

The download manager uses `huggingface_hub`'s standard cache + lock
files; corrupted partials should self-heal on retry.

## "MTPLX is not installed" but I just installed it

The capability probe caches results for the lifetime of the backend
process. After installing MTPLX:

```bash
./scripts/chaosengine-cli setup-refresh
```

Then load the model again — the runtimeNote should pick up `mtplx`.

## I keep seeing `MallocStackLogging` spam in the logs

Pre-FU-038 builds hit this on macOS hardened runtime. The fix
(2026-05-10) suppresses it at the subprocess-spawn boundary in
[`src-tauri/src/backend.rs`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/backend.rs)
and adds a regex filter to the diagnostics log endpoints so older
installs also surface clean logs. If you're on a release that
predates the fix, update — there's no opt-in cleanup.

## "No orphan workers" check failed in E2E Phase 7

The diagnostics snapshot is reporting subprocess children that
didn't shut down cleanly. From the CLI:

```bash
./scripts/chaosengine-cli diagnostics-snapshot \
    | jq '.recentOrphanedWorkers'
```

Each entry includes the PID + worker type. Kill them manually:

```bash
kill -9 <pid>
```

Then file an issue — orphan tracking is a regression signal, not a
normal-state condition.

## See also

- [Model load failures](model-load-failures.md)
- [MTPLX install issues](mtplx-install-issues.md)
- [GPU detection](gpu-detection.md)
- [Orphan workers](orphan-workers.md)
