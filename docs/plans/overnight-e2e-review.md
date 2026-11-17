# Overnight whole-app review (v0.10.0)

Status: **planned 2026-11-17, not started.** Owner asked for a run that works through the night and leaves a morning report. Decisions D-87 and D-88, follow-up FU-220.

## Why

Seven features have not been used for a long time and have no end-to-end check (no phase in `scripts/e2e_test_suite.py`, no Playwright spec; unit tests exist only for Plugins and Prompts): Benchmarks, Conversion to MLX, Fine Tuning, Prompts, Plugins, Logs, Server. The Server page buries its useful part under a long "Connect your app" block that is open by default, and its logs say little about a model's life (load, prefill, generate, unload). A podcast invitation makes the first minutes of the app matter more than usual.

## Challenge (once, then follow the owner)

**Costly and risky:** a redesign of Server and Logs, plus removals, late in v0.10.0, with the Windows and Linux passes still to merge. Alternatives, in order of preference:
1. **Audit overnight, decide in the morning (recommended).** Agents produce evidence and a verdict per feature. Nothing is deleted. A feature that fails is hidden behind a Settings switch ("Show experimental tabs", off by default) in a separate commit the owner can revert. The Server redesign is built, because it is the one change that is clearly worth it.
2. Audit and delete. Cleaner, but a wrong call is a revert and a rebuild.

## Start gate

The run starts only when all of these hold: `release/v0.10.0` is green (pytest, vitest, tsc, i18n, smoke E2E), the Windows and Linux fixes are merged, and the tree is frozen for the run. Reason: every stream edits shared files (`App.tsx`, locales, the sidebar), so a moving tree turns the morning into merge work. If the owner prefers to start tonight on the Mac tip, streams A to D still work; the merge back is on the integrator.

## Safety rules for every agent

- One git worktree per agent, branch `review/<stream>`, pushed early, small commits. Never stash, reset or discard in a shared tree. The main checkout and `test/rtx4090` are not touched.
- **One model-loading stream at a time** on the 64 GB Mac (two loaded models exhaust RAM, CLAUDE.md). Read-only and code-only streams run in parallel; GPU and RAM streams queue behind a lock file in the scratchpad.
- Own backend port per agent (8900 + stream index). Kill only a PID the agent started.
- No model deletes, no downloads over 2 GB without a line in the progress file, no Hugging Face tokens printed or committed, no publish, no tag, no push to `release/v0.10.0`, `staging` or `main`.
- No names of other desktop AI apps in code, comments, docs or commits (editorial rule). For the logs redesign, describe what we build, not what others do.
- Every stream keeps a progress file `docs/progress/review-<stream>.md` (PASS, FAIL or SKIP with one line of evidence, the exact command, the log tail for a FAIL) so a fresh agent can resume.
- Stall guard: a stream with no new commit or progress line for 45 minutes stops and writes why.
- Wall clock budget 9 hours. Keep the Mac awake for the run (`request_keep_awake`).
- Fix on the spot only small, clearly local bugs, each with a test. Anything wider goes in the report with a recommendation.

## Streams

Model: Opus for design-heavy or risky work (E, F), Fable 5.2 for the long audits where depth matters (A to D), Sonnet for checks and the i18n and test sweeps (G). Effort high.

| Stream | Scope | Output |
|---|---|---|
| **A: Benchmarks and Conversion** | Run a real benchmark on a small local MLX and a small GGUF model; read history, compare, export, delete. Convert one small model to MLX at two quant levels; cancel mid-way; low-disk and bad-input errors; the converted model loads and chats. | Per-feature rubric, bugs fixed with tests, verdict. |
| **B: Fine Tuning** | What the tab promises versus what runs. Try the shortest real job on the smallest supported model; read the dataset format, progress, cancel, output, loading the result. If nothing real runs on this Mac, say so and mark it hide-until-wired. | Verdict with evidence, plus the "no-code fine-tuning" idea only as a note. |
| **C: Prompts and Plugins** | Prompts: save, edit, delete, import, use from Chat and from Studio; compare with the Skills page and the chat slash menu to find overlap; decide keep, merge into Chat or hide. Plugins: install a sample plugin, run it, error path, uninstall, trust and permission model, what the safe-mode story is. Efficacy: does any user task need it that Skills or the agent bridge do not already cover? | Verdict per page, with a merge proposal if there is overlap. |
| **D: Whole-app sweep** | Every top-level tab and every sub-tab on empty state, loaded state, error state, narrow window and light and dark theme; each tab with the backend stopped mid-use; first-run path from a clean data dir; Settings; Setup; Diagnostics; every i18n locale boots with no raw keys. Add a Playwright spec per tab touched (`e2e/tests/27-...`), and a `phase_21` in `scripts/e2e_test_suite.py` for the seven features. | Defect list ranked by user impact; new specs. |
| **E: Server and Logs redesign** | Spec first, then build (see below). | Branch with the new UI, tests, screenshots in light and dark. |
| **F: Demo path** | A scripted five-minute path for the podcast: first launch, pick a model, chat, a Studio film clip, voice, the Server page. Time each step, find every stumble, fix the cheap ones. Rehearsal file `docs/plans/demo-path.md`. | A path that runs clean twice in a row. |
| **G: Verify** | Full pytest, vitest, tsc, `i18n:validate`, smoke and full `e2e_test_suite.py`, Playwright at `--workers=1`, upstream check, audits, locale parity for every new string (new keys ship with `null` placeholders per CLAUDE.md, except `en`). | Green or a named list of failures. |

Order: A to D run now in parallel where they do not load a model (read, code and UI checks) and take the model lock for their real runs. E starts with its spec and the logging backend while A to D audit. F starts after E merges. G runs last on the integrated branch `review/integrate`.

## Verdict rubric (per feature)

Each of the seven gets one table row: **Works** (end to end with a real model), **Honest** (the page promises only what runs), **Valuable** (a user would miss it; not covered elsewhere), **Clutter cost** (sidebar slot, screen size, upkeep lines), **Verdict**: keep, fix then keep, merge into another page, hide behind the experimental switch, or remove. A "remove" verdict needs a named reason and a list of the files and tests that go with it. Nothing is removed overnight.

## Server and Logs redesign (stream E)

Findings to change:
- "Connect your app" is open by default and fills the page. **Default collapsed**, one line with the address and a Copy button; the open state is remembered per user.
- The model life is not visible. Add a **structured event log** in the backend (`backend_service/state/logs.py` today holds plain lines) with one event per step: requested, resolving, loading (bytes, seconds), loaded (memory used), first token (prefill time), generation (tokens per second), idle, unloaded (reason: user, swap, idle, error), plus request id, model, engine, cache strategy and speculative method.
- UI: a **timeline per model** (a bar with phases and times), a filterable event list (model, level, request), expand a row for detail (full command line, environment, last stderr lines), Copy and Export selected, live tail with pause, and an "open the worst recent failure" shortcut.
- **Logs page:** recommendation is to fold it into Server as a tab ("Activity"), because the logs are server logs. Remove the sidebar entry only after the redirect and the saved-state migration are tested. Old log lines keep showing.
- Keep every existing endpoint working (agent bridge, CLI, Diagnostics read them); add, do not break. Golden MCP list stays as is unless a tool is added.
- Strings go through i18n; no emoji; monochrome icons only.

## Morning deliverables

1. `docs/progress/overnight-review-report.md`: one page. Headline, per-feature verdict table, defects fixed, defects left (ranked), what was not tested.
2. Branch `review/integrate` with all streams merged and green, plus per-stream branches. Nothing merged into `release/v0.10.0`; the owner or the integrator does that after reading the report.
3. Hub status entry, decisions D-87 and D-88 updated with evidence, FU rows for what is left, hub rebuilt.
4. Screenshots of the new Server page (light and dark) and the timeline for one real model load.

## Not in this run

Windows and Linux testing (the 4090 plan), tag and publish, MiniMax H3 (licence read pending; it excludes the UK, so it cannot be tested legally from here), the H3 low-VRAM claim.

## Launch

Needs the owner's yes (it spends many agent hours and runs for most of a night). Launch as a Workflow with the streams above, or one agent per stream in worktrees with this file as the brief.
