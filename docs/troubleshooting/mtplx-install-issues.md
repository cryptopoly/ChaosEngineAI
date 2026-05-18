# MTPLX install issues

The MTPLX installer (`scripts/install-mtplx.sh`, also reachable via
Setup → Install MTPLX or `./scripts/chaosengine-cli mtplx-install`)
emits structured `PHASE:` / `OK` / `FAIL:` markers for each step so
the install log panel can render a clean phase timeline. Here are the
phases and what each one can fail on.

## Phase 1 — preflight

Checks for native arm64 Python 3.10+. The installer fails with:

- `MTPLX requires native arm64 Python (got: x86_64). Make sure you are
  not running under Rosetta.`

The mtplx package ships a forked `mlx` that's Apple Silicon only.
Running under Rosetta won't work even if the Python binary thinks
it's native. Fix:

```bash
# Confirm arm64
file $(which python3)
# Should report: Mach-O 64-bit executable arm64

# If you're getting x86_64, your shell is in Rosetta. Open a new
# Terminal *without* "Open using Rosetta" checked.
arch -arm64 ./scripts/install-mtplx.sh
```

You may also need a fresh Python:

```bash
brew install python@3.12  # native arm64
PYTHON=/opt/homebrew/bin/python3.12 ./scripts/install-mtplx.sh
```

## Phase 2 — creating-venv

Creates `~/.chaosengine/mtplx-venv/`. If a previous venv exists, it's
wiped first.

Failure modes are rare here — usually permission errors on
`~/.chaosengine/`. Make sure the directory is writable.

## Phase 3 — installing

Runs `pip install --upgrade mtplx` inside the isolated venv. This is
where most failures happen because pip pulls the forked `mlx` as a
transitive dep and builds Metal kernels.

Typical failures:

- **Network errors** — flaky PyPI / Hugging Face. Re-run.
- **Build errors on the forked mlx** — usually a missing Xcode
  command-line tool. `xcode-select --install` then re-run.
- **`metal-cpp not found`** — older macOS. The mtplx fork needs
  macOS 13+; older versions can't compile the Metal kernels.

The full pip log goes to stderr; the install log panel surfaces the
last 200 lines on failure.

## Phase 4 — verify

Imports `mtplx` from the isolated venv and writes the version marker
to `~/.chaosengine/bin/mtplx.version`. If verify fails after a clean
install, the package is broken somehow — try:

```bash
rm -rf ~/.chaosengine/mtplx-venv
./scripts/install-mtplx.sh
```

## Refreshing the capability probe

The backend caches the `mtplxAvailable` flag at startup. After
installing MTPLX:

```bash
./scripts/chaosengine-cli setup-refresh
```

This re-runs the probe. The diagnostics snapshot should then show:

```json
{
  "capabilities": {
    "mtplxAvailable": true,
    "mtplxPythonPath": "/Users/<you>/.chaosengine/mtplx-venv/bin/python"
  }
}
```

## "MTPLX is installed but `runtimeNote` still says only `mlx`"

If you've installed MTPLX, refreshed capabilities, and verified the
model is in `MTP_MODEL_MAP`, but the engine still doesn't route
through MTPLX:

1. Inspect the load result's full `runtimeNote`. It usually carries
   the actual fallback reason ("mtplx fallback: port already bound",
   "mtplx fallback: subprocess crashed during load").
2. Check the diagnostics snapshot's `recentErrors` block.
3. From the CLI, try a manual MTPLX-only load:
   ```bash
   ./scripts/chaosengine-cli load <repo> --backend mlx --spec
   ```
   The `--backend mlx --spec` combination forces the routing decision
   that `_select_engine` would make. If this fails, you'll see the
   exact engine startup error in the response.

## Uninstalling MTPLX

```bash
rm -rf ~/.chaosengine/mtplx-venv
rm -f  ~/.chaosengine/bin/mtplx.version
./scripts/chaosengine-cli setup-refresh
```

The backend will report `mtplxAvailable: false` and the launch modal
will hide the MTPLX-only branch.

## See also

- [MTPLX deep dive](../features/mtplx.md)
- [Engine routing](../architecture/routing.md)
- [Adding checks](../testing/adding-checks.md) — assert MTPLX behaviour in E2E
