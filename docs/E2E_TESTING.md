# End-to-End Testing — ChaosEngineAI

This document is the standardised procedure for proving the app's feature
surface is **100% operable** from the CLI. It's the gate that runs before
every release and the safety net that catches regressions when a feature
ships.

The suite lives at [`scripts/e2e_test_suite.py`](../scripts/e2e_test_suite.py)
and drives the [`scripts/chaosengine-cli`](../scripts/chaosengine-cli) wrapper
sequentially through every major surface in the app.

## Phases

Each phase mirrors a top-level tab in the desktop app. Phases auto-skip when
prerequisites are missing — that's reported as `skip`, not `fail`, so the
suite stays green on environments where (e.g.) llama-server isn't installed
or no FLUX weights are on disk.

| Phase | Surface | What it proves |
|-------|---------|----------------|
| 0 | Environment probe | Backend reachable, OpenAPI advertises ≥100 routes, GPU detected, MTPLX/DFlash registries populated. |
| 1 | Chat — text generation | MLX + GGUF backends both produce tokens. Cache strategies (Native f16, TurboQuant). Speculative decoding paths (DFlash, MTPLX) route correctly per `runtimeNote`. `cache-preview` returns sane numbers at 32k+ context. `--fused-attention` honoured. |
| 2 | Chat Compare | `/api/chat/compare` accepts two-slot payload and returns 200. |
| 3 | HTML Challenge | List + create + delete round-trip. Skipped when no MLX text model on disk. |
| 4 | Image Studio | Catalog, library, runtime probes pass. If a model is installed, runs a 4-step 256×256 generation and confirms the artifact lands. |
| 5 | Video Studio | Same shape as Phase 4 — catalog, library, mlx-runtime probes. Generation runs against any installed video model. |
| 6 | Setup probes (read-only) | `mtplx-status`, `longlive-status`, `wan-status/inventory`, `gpu-bundle-info/status`, `turbo-update-check`. Destructive install actions are intentionally NOT run (too slow + side-effecty for a routine suite). |
| 7 | Diagnostics + cleanup | `diagnostics-snapshot` + `log-tail` return data. No orphan MLX / llama-server / MTPLX subprocesses left dangling. Runtime returns to `idle` or `loaded` (not `error`). |

## Pass criteria

Concrete, not "feels right":

- HTTP status codes assert 200 (or 2 in the suite's exit for the call-level
  check).
- For Phase 1 generation checks: `tokS > 0` AND a completion was produced.
- DFlash / MTPLX checks additionally assert the expected token appears
  in `runtimeNote` (`"dflash"` / `"speculative"` / `"mtplx"`). A pass for
  "speculativeDecoding=true" alone is **not** sufficient — backend must
  actually have routed through the speculative engine.
- Phase 7 asserts `recentOrphanedWorkers` is empty (no subprocess leak)
  and runtime state is one of `idle` / `loaded` (not `error`).

## Skip semantics

A check returns `skip` (not `fail`) when:

- A required model isn't on disk (e.g. no MTPLX-bearing model under
  `~/AI_Models/Youssofal/`).
- A backend dependency isn't present (e.g. no `llama-server` binary, no
  installed image-generation pipeline).
- A capability the user has explicitly not installed (MTPLX venv missing,
  DFlash pip package missing) and the check needs it.

`skip` is the suite's way of saying *"can't tell — environment doesn't
have what this check needs."* It's safe; only `fail` raises a regression
alarm.

## How to run

### Full sweep (every phase, every check)

```bash
# In one shell — keep the backend running for the entire suite
./scripts/chaosengine-cli serve

# In another shell
./scripts/e2e_test_suite.py
```

Wall time depends on hardware and which models are on disk. M-series with
27B MLX models on hand: 10–25 minutes. Add another 10–20 if Phase 4 / 5
actually run generation (depends on installed image/video pipelines).

### Smoke pass (skip heavy chat generation)

```bash
./scripts/e2e_test_suite.py --smoke
```

Runs phases 0, 2, 3, 4, 5, 6, 7 — skips Phase 1's model-load × prompt
loop. Typical wall time: < 60s. Use this in CI or to validate the suite
itself.

### Specific phases

```bash
./scripts/e2e_test_suite.py --phases 0,1,7
```

### Custom report location

```bash
./scripts/e2e_test_suite.py --report-dir /tmp/my-results
```

## Output

Two files per run, written to `~/.chaosengine/test-results/` by default:

- `e2e-YYYYMMDD-HHMMSS.json` — full machine-readable report
  (capabilities probe, phase results, per-check pass/fail/skip + reason +
  elapsed seconds).
- `e2e-YYYYMMDD-HHMMSS.md` — human-readable summary table.

Stderr streams a one-line phase-start / phase-end log so you know what
the suite is doing without `tail -f`-ing the report.

### Exit codes

- **0** — every phase passed (or was correctly skipped). Suite is happy.
- **1** — at least one phase had a `fail` check. **Regression**.
- **2** — backend was not reachable; suite could not run.

## Cleanup

Every check that loads a model is wrapped to `unload` afterwards. The
Phase 7 "no orphan workers" check guarantees nothing leaked. If a check
fails partway through a model-load cycle, the suite still issues the
follow-up `unload` so the next check starts from a clean slate.

## When this suite is the right tool

Use the E2E suite when:

- Cutting a release. Required gate before pushing tags.
- Landing any change that touches inference routing
  (`inference/controller.py`, engine implementations, cache strategy
  registry, setup install endpoints).
- Bumping an upstream dependency that touches model load paths
  (`mlx-lm`, `llama.cpp`, `diffusers`, `mlx-video`, `dflash-mlx`,
  `turboquant-mlx-full`, `mtplx`).
- Verifying a UX bug fix that depends on backend behaviour — the suite
  catches whether the *backend* did what the UI claims.

Use plain `pytest tests/` (unit/integration) when:

- Changing pure-function helpers, formatting, parsing, validation.
- Editing TypeScript-only changes.

The two suites are complementary — pytest covers fast, mocked-engine
correctness; E2E covers real-engine routing and feature operability.

## Adding new checks

When a new feature ships, the procedure is:

1. Identify which phase the feature belongs to (text generation → 1,
   image → 4, video → 5, install endpoint → 6, etc.).
2. Add a check function inside that phase. Convention:

   ```python
   def _your_check():
       # Skip if prerequisite missing — return ("skip", reason, {}).
       # Return ("pass", "", {detail}) on success.
       # Return ("fail", reason, {detail}) on failure.
       ...

   phase.checks.append(_check("your-check name", _your_check))
   ```

3. Run `./scripts/e2e_test_suite.py --phases <N>` against a live backend
   to confirm the new check passes.
4. Commit the suite change in the same PR as the feature.

Discoverability is the key invariant — if a feature ships without an E2E
check, future "is the app still working?" runs are silently incomplete.
The CLAUDE.md "Build Checklist" section enforces this gate before
release.
