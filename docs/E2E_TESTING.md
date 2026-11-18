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
| 0 | Environment probe | Backend reachable, OpenAPI advertises ≥100 routes, GPU detected, MTPLX/DFlash/TensorFold registries populated, and every TensorFold catalog row agrees with the registry (exclusive families are tagged for the TensorFold backend; checkpoints that can read images carry the vision tag). |
| 1 | Chat — text generation | MLX + GGUF backends both produce tokens. Cache strategies (Native f16, TurboQuant). Speculative decoding paths (DFlash, MTPLX, TensorFold) route correctly per `runtimeNote` / engine. `cache-preview` returns sane numbers at 32k+ context. `--fused-attention` honoured. |
| 2 | Chat Compare | `/api/chat/compare` accepts two-slot payload and returns 200. |
| 3 | HTML Challenge | List + create + delete round-trip. Skipped when no MLX text model on disk. |
| 4 | Image Studio | Catalog, library, runtime probes pass. If a model is installed, runs a 4-step 256×256 generation and confirms the artifact lands. |
| 5 | Video Studio | Same shape as Phase 4 — catalog, library, mlx-runtime probes. Generation runs against any installed video model. |
| 6 | Setup probes (read-only) | `mtplx-status`, `tensorfold-status`, `longlive-status`, `wan-status/inventory`, `gpu-bundle-info/status`, `turbo-update-check`. Destructive install actions are intentionally NOT run (too slow + side-effecty for a routine suite). |
| 7 | Diagnostics + cleanup | `diagnostics-snapshot` + `log-tail` return data. No orphan MLX / llama-server / MTPLX / TensorFold subprocesses left dangling. Runtime returns to `idle` or `loaded` (not `error`). |
| 8 | Voice I/O | Voice runtime, STT/TTS lanes, synthesize, transcribe, library. |
| 9 | Agent bridge | `/mcp`: the off answer names the Settings path, `tools/list` has the 18 tools (12 media, 5 storyboard, 1 music), `system_status` and `recommend_models` answer, the stdio relay (`chaosengine-cli mcp`) forwards, the agent guide (`AGENTS.md`) is served by `GET /api/agent/guide`, the `chaosengine://guide` resource and the `start_here` prompt with every real tool named in it, and with an installed image model `generate_image` + `wait_for_job` ends in a PNG that carries its settings. Switches Agent access on for the run and restores the previous state. A last check sends a reference image to a model that cannot take one (image and video routes) and expects a 400 that says why; another uploads an image (bad bytes refused), refuses a path outside the output folders, and fetches the image job's output with a matching checksum. Run alone: `--phases 9`. |
| 10 | Storyboard core | `/api/storyboard`: create a project, add three shots and edit one (the timeline total follows the trim), a write with an old revision answers 409 with the current revision, import two takes (stand-in bytes), pick one and undo the pick, export a `.chaosfilm` bundle, fetch it and import it as a new project, and refuse a hostile bundle (absolute path in `project.json`, newer format, a `../` entry). No model runs; the projects it made are deleted. Run alone: `--phases 10`. |
| 11 | Music | `/api/music`: the runtime route has its fields, the catalog's ACE-Step 1.5 turbo row carries the licence (MIT, commercial, not gated, summary lines, URL) and the UI fields, the estimate answers and out-of-range or unsupported requests (5 s, 3 variations, a song without lyrics, `reference_audio`) are refused with 422, and a synthetic library item is listed, renamed, tagged, fetched with a `Range` request (206) and deleted; bad ids are refused. Then a real 20 s instrumental that **skips unless the music pack and the turbo weights are installed**: 48 kHz stereo FLAC of the right length, carrying the `chaosengine:` tag, peak footprint under 26 GB; the track is deleted afterwards. Run alone: `--phases 11`. |
| 19 | Skills import (D-62) and scan (D-68) | `/api/skills`: a temp skill with a unique name imports from a folder switched off and rated Low by the security scan, with a binary file skipped and listed; switching it on works and a second import answers 409; a zip with a `../` entry and a zip bomb are refused with 400 and nothing lands outside; a crafted bad skill rates High, switching it on without `acknowledgeFindings` answers 409 and with it 200; delete removes it. No model runs. `tests/test_skills_e2e_phase.py` runs this phase against an in-process app. Run alone: `--phases 19`. |
| 20 | Storyboard 0.10 | `/api/storyboard`: plan, brand kit and Keep for final ops undo and redo (non-ASCII brief); a plan applied, changed in one step and undone, the chat kept across Undo; plan routes refuse no brief (400) and no chat model (409, only checked when none is loaded); style pictures give a palette, a logo and caption look are kept, and a bundle carries them; the final pass counts kept shots and a draft final pass is refused; redo kinds are priced and an upload shot has nothing to redo; Use this one (`pick_take`) and undo; the sample film installs finished and its timeline and clips feed the rough cut; the render plan burns captions in the brand kit's look; a project file left running is reset on open; `/api/workspace` carries `activeGenerations`; `POST /api/files/open` refuses non-media, files outside the app folders and a symlink out; the Chat plan-card pre-check refuses calls that cannot run (in process, temporary store); the Chat turn route answers for a reloaded window, a waiting plan is listed for its chat and answered there, and a refused call's reason reaches the next turn's history. Then, **skipping unless the models are installed**: a real draft with the fast still model (the image engine shows in `activeGenerations` meanwhile) and a draft clip, a cancelled Redo clip that keeps the take in use (FU-181), and a Redo still that keeps the old clip and marks the shot stale. The final pass is priced, never run. Two checks write into the backend's project folder, so set `CHAOSENGINE_STORYBOARDS_DIR` for the suite when the backend uses one. Run alone: `--phases 20`. |
| 211 | Benchmarks and Conversion to MLX | `POST /api/benchmarks/run`, `DELETE /api/benchmarks/{id}`, `POST /api/models/convert`, `POST /api/models/convert/cancel` (file `scripts/e2e_phases/phase_21_benchmarks_conversion.py`; 211 = review group 21, stream A). Guards: an unknown mode and `maxTokens` 1 are refused (422), a model that is not on disk is a plain 400; conversion with no source, a dtype outside the list, 9 bits, a missing local path and a GGUF-only base repo are each refused with their own message. Then, **with real models and skipping when there are none**: a throughput run on the smallest local MLX text model and the smallest local GGUF one (speed above zero, tokens generated, first in the history), a perplexity run (WikiText-2, 8 samples) and a task-accuracy run (MMLU, 10 questions), both skipping when the dataset cannot be fetched; the runs are in the saved history, one delete of an unknown id answers 404, and every run the phase made is deleted. Conversion: the smallest local Transformers checkpoint of 10 GB or less (never downloaded) is converted to 4-bit MLX in a temporary folder (config and safetensors present, bits per weight in the log), a second run into the same folder is refused and leaves it alone, a third run is cancelled through `POST /api/models/convert/cancel` (400 "Conversion cancelled", no folder left), and the converted model loads and answers. The temporary folder is removed. Browser side: `e2e/tests/27-benchmarks-conversion.spec.ts` (mocked runs, no model). Run alone: `--phases 211`. |
| 212 | Fine-Tuning (review stream B) | `/api/adapters` answers with rows and a matching count; two temp adapter folders (a PEFT config and an mlx-lm config) under the backend's data directory are listed with the fields the tab reads and the right base model, then deleted; the training status is `idle`; a bad start body answers 422; a valid `POST /api/finetuning/start` answers 501, starts nothing and leaves the status `idle`, because no training backend exists. When training is built that check fails on purpose and must become a real run. `scripts/e2e_phases/phase_21_finetuning.py`. Run alone: `--phases 212`. |
| 213 | Prompts and Plugins (review stream C) | `/api/prompts`: the list carries its fields, a template with variables and presets saves with bad variable rows dropped, an update keeps the id and search finds it by text, tag and category, a missing name, a prompt over 16000 characters, bad tags, a bad variable and an unknown delete are refused, delete removes it and a second delete is 404. `/api/plugins`: the answer is the type-to-rows map the page reads (no wrapper key) with the built-in cache strategies and the tools Chat offers, a switch round-trips and is put back, an unknown id is 404. Both need the API token (skips when auth is off). Then, **skipping when no MLX text model is on disk**: the smallest one is loaded, the same question is asked with and without a template as the system prompt, and the prompt the model saw must be at least 10 tokens longer with it (the reply following the template is reported, not required). Every template made is deleted. `tests/test_prompts_plugins_e2e_phase.py` runs this part against an in-process app. Other streams of the review register their own parts under the same number (`phase_21_prompts_plugins.joined`). Run alone: `--phases 213`. |
| 214 | Server Activity (D-88, review stream E) | `/api/server/events`: the route shape; the smallest local Qwen3 0.6B or Qwen2.5 0.5B MLX model is loaded, answers one prompt and is unloaded through the CLI, and the events since then hold requested, loading, loaded (seconds and memory), a generation with tok/s and an unload by the user, the load events sharing one request id; the event stream and the old log stream each send a first line. Skips the model part when no small MLX model is installed. Run alone: `--phases 214`. |
| 215 | GGUF routing from files (review stream GGUF) | GGUF or MLX is read from the model's files, never from "gguf" in a name: the smallest local MLX text model, linked into a temporary folder as `my-gguf-notes-mlx`, loads on MLX (auto and explicit `mlx`) and answers; the smallest local GGUF, linked as a file inside a folder called `mlx`, loads on llama.cpp and answers; a folder with only `config.json` is refused with a plain message naming that file. Only symlinks are made; they are removed at the end. Skips a check when no model of that kind is on disk. Run alone: `--phases 215` (scripts/e2e_phases/phase_gguf_routing.py). |

## Pass criteria

Concrete, not "feels right":

- HTTP status codes assert 200 (or 2 in the suite's exit for the call-level
  check).
- For Phase 1 generation checks: `tokS > 0` AND a completion was produced.
- The TensorFold check asserts the loaded engine really is `tensorfold`
  (a silent fallback to standard MLX still produces text, so the token check
  alone would not catch it), and that image input is on exactly when the
  `vision` extra is installed and the checkpoint has a vision tower.
- The TensorFold 1.2.7 families check (FU-168) fails when any LFM3, LFM2.5,
  Spark, Nanbeige or MiniMax checkpoint drops out of the status payload, then
  loads one checkpoint per family that is on disk with speculative decoding
  ticked and asserts the engine is `tensorfold-native`. Nanbeige skips on
  M1 to M4 (TensorFold needs an M5); nothing is downloaded.
- DFlash / MTPLX checks additionally assert the expected token appears
  in `runtimeNote` (`"dflash"` / `"speculative"` / `"mtplx"`). A pass for
  "speculativeDecoding=true" alone is **not** sufficient — backend must
  actually have routed through the speculative engine.
- Phase 7 asserts `recentOrphanedWorkers` is empty (no subprocess leak)
  and runtime state is one of `idle` / `loaded` (not `error`).
- Phase 7 also drives the Keep going run API (FU-191) on a throwaway chat:
  start is armed and idempotent, the cap and grant hours are bounded, Stop
  ends the run and drops the render grant, Resume refuses a run that never
  had a brief, and deleting the chat drops the run. No model, nothing renders
  (`scripts/e2e_phases/keep_going.py`).

## Skip semantics

A check returns `skip` (not `fail`) when:

- A required model isn't on disk (e.g. no MTPLX-bearing model under
  `~/AI_Models/Youssofal/`).
- A backend dependency isn't present (e.g. no `llama-server` binary, no
  installed image-generation pipeline).
- A capability the user has explicitly not installed (MTPLX venv missing,
  TensorFold venv missing, DFlash pip package missing) and the check needs it.

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
  `turboquant-mlx-full`, `mtplx`, `tensorfold`).
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
