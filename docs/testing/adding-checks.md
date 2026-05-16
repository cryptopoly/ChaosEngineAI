# Adding checks

The E2E suite's value is its **discoverability invariant**: if a feature
ships without an E2E check, future "is the app still working?" runs are
silently incomplete. The pre-build check's phase 9 enforces this gate.
Here's how to extend it when you add a feature.

## Step 1 — pick a phase

Every E2E phase mirrors a top-level tab in the desktop app. New feature
lands in:

| Phase | Surface | Add a check here when... |
|---|---|---|
| 0 | Environment probe | A new capability flag is exposed by `/api/health` or the diagnostics snapshot. |
| 1 | Chat — text generation | A new chat-time path: cache strategy, speculative-decoding mode, sampler, runtime route. |
| 2 | Chat Compare | New compare-specific shape or per-slot setting. |
| 3 | HTML Challenge | New per-slot setting, repair / retry flow, validation status. |
| 4 | Image Studio | New image runtime, sampler, distill LoRA, cache strategy, or live-preview behaviour. |
| 5 | Video Studio | Same as Phase 4 but for video. |
| 6 | Setup probes (read-only) | New install endpoint, status surface, inventory query. |
| 7 | Diagnostics + cleanup | New subprocess type that needs orphan tracking, new state the runtime can land in. |

## Step 2 — write the check function

The convention inside `scripts/e2e_test_suite.py`:

```python
def _your_check():
    # Skip when prerequisites are missing
    if not _model_on_disk("Qwen/Qwen3-4B"):
        return ("skip", "Qwen/Qwen3-4B not on disk", {})

    # Drive the CLI / backend
    result = _cli_json("prompt", "hello", "--max-tokens", "4")
    if not result.get("text"):
        return ("fail", "empty generation", {"result": result})

    # Pass with detail
    return ("pass", "", {"tokS": result["metrics"]["tokS"]})

phase.checks.append(_check("your-check name", _your_check))
```

### Pass / fail / skip contract

- **`pass`** — the feature worked. Detail dict carries useful metrics for
  the report.
- **`fail`** — regression. Detail dict carries enough info to debug.
- **`skip`** — prerequisite missing (model not on disk, dependency not
  installed). Reason string explains *what's missing*, not *what failed*.

Skips keep the suite green on environments that don't have every model.
The suite shouldn't fail on a fresh CI box that hasn't downloaded a
14B model yet.

## Step 3 — assert routing, not just success

A common bug pattern is: the feature toggle is enabled, the backend
returns 200, but the actual code path didn't fire. The E2E suite catches
this by asserting on `runtimeNote`:

```python
result = _cli_json("prompt", "hello", "--spec")
note = result.get("runtimeNote", "")

# Wrong — just checks the toggle was accepted
if "speculativeDecoding=true" not in note:
    return ("fail", "spec-dec toggle didn't surface", {})

# Right — asserts the engine actually routed through MTPLX
if "mtplx" not in note.lower():
    return ("fail", "speculative decoding enabled but MTPLX didn't route", {})
```

The Phase 1 DFlash / MTPLX checks model this pattern.

## Step 4 — run against a live backend

```bash
./scripts/chaosengine-cli serve &
./scripts/e2e_test_suite.py --phases <N>
```

The phase filter runs just your phase so the iteration loop is fast.
Once the new check is green locally, run the smoke pass to confirm you
haven't broken anything else:

```bash
./scripts/e2e_test_suite.py --smoke
```

## Step 5 — commit alongside the feature

The CLAUDE.md "Build Checklist" enforces that E2E checks ship in the same
PR as the feature. Reviewers will reject feature PRs that don't include
a corresponding E2E check.

## Step 6 — update the docs

If your check exercises a user-visible flow, update the relevant `usage/`
page (Chat, Image Studio, etc.) to mention the new flag. If it's a CLI
shortcut, update `cli/overview.md` and the [reference](../cli/reference.md).

## Anti-patterns

- **Checking for a string that's user-visible but not unique.** Use
  literal tokens the backend always emits (e.g. `"mtplx"`, `"dflash"`).
- **Hard-coding model refs.** Use the same fuzzy-resolution helper the
  backend uses, or skip when the model isn't on disk.
- **Asserting on wall time.** E2E runs on heterogeneous hardware; tok/s
  thresholds are a fool's errand. Assert `tokS > 0` instead.
- **Side-effects that aren't cleaned up.** Every check that loads a
  model must `unload` afterwards. Phase 7's orphan check is your safety
  net but you should not rely on it.

## See also

- [E2E testing](e2e-testing.md) — the suite's contract.
- [Pre-build check](pre-build-check.md) — where E2E sits in the release gate.
