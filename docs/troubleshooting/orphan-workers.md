# Orphan workers

The backend tracks every subprocess it spawns — MLX worker, `llama-server`,
MTPLX server, mlx-video subprocess, sd-cli. On clean shutdown they're
killed; on a crash or a parent-side bug, they can leak.

The diagnostics snapshot's `recentOrphanedWorkers` field reports leaks
the backend itself can detect. The E2E suite's Phase 7 asserts this
field is `[]`.

## Detecting orphans

```bash
./scripts/chaosengine-cli diagnostics-snapshot \
    | jq '.recentOrphanedWorkers'
```

Each entry includes:

- `pid` — the orphaned process.
- `kind` — `mlx_worker`, `llama_server`, `mtplx`, etc.
- `reason` — "parent missed shutdown", "subprocess.wait timed out",
  etc.
- `firstSeen` — when the tracker first noticed it.

On macOS / Linux:

```bash
ps aux | grep -E "(llama-server|mlx_worker|mtplx)"
```

The PID column should match the snapshot's `pid` field.

## Killing orphans manually

```bash
# macOS / Linux
kill -TERM <pid>  # graceful
kill -KILL <pid>  # force, only if -TERM didn't work

# Windows
taskkill /F /PID <pid>
```

After killing, `setup-refresh` clears the snapshot's
`recentOrphanedWorkers` tracking:

```bash
./scripts/chaosengine-cli setup-refresh
./scripts/chaosengine-cli diagnostics-snapshot | jq '.recentOrphanedWorkers'
```

## Why they happen

Orphan tracking was added because of real bugs. Two examples from the
v0.8.0 changelog:

- **`JsonRpcProcess.close()` timeout under memory pressure.** Force-
  killing an MLX worker holding ~47 GB of weights routinely raised
  `TimeoutExpired` on the macOS vm_map teardown. The route layer's
  broad `except` swallowed the exception, `self.process` was never
  nulled, and the next load spawned a second worker alongside the
  dying one. Activity Monitor showed two ~47 GB Python processes;
  `/api/server/status` reported one model.

  Fix: capture and null `self.process` up-front, wrap the post-kill
  `wait()` in `try/except TimeoutExpired` with a 1 s ceiling.

- **`llama-server` not killed when the engine swapped out under it.**
  The cache-strategy fallback chain (`requested → native`) could swap
  the engine binary without killing the previous server first. Fixed
  in the FU-030 cleanup.

Both fixes shipped before v0.8.0; if you're on a current build the
orphan tracker should only fire on actual subprocess crashes.

## Reporting

If you see persistent orphans on a clean install, file an issue with:

- The `recentOrphanedWorkers` block from the snapshot.
- The `recentErrors` block (the underlying failure usually shows up
  here).
- The log tail (`./scripts/chaosengine-cli diagnostics-log-tail
  --lines 500`).
- A reproduction sequence (which model, which load command, which
  unload command).

## See also

- [FAQ](faq.md)
- [E2E testing](../testing/e2e-testing.md) — Phase 7 contract.
- [Adding checks](../testing/adding-checks.md).
