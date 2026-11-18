# Keep going and tools, version 2 (plan)

Status: **plan only, no code changed.** Stream PLAN, branch `review/plan` (from `release/v0.10.0` at `1bcb4e21`), written 2026-11-18. Proposes decisions D-89, D-90, D-91 and follow-ups FU-230 to FU-242; the integrator adds the rows (ready to paste at the end). No model was loaded and no server was started for this plan: every figure is either quoted from a run that is named, or read from the code, or marked as an estimate.

## Summary

Three owner asks share one root cause. On the MLX path the chat model is shown a tool list that is **long** (20 Studio tools, 14,708 characters, about 4,000 tokens: half of the default 8K context), **written in one generic call format** whatever the model was trained on, and **silent about the values a call may take**: the model reads `kind: string (required)` with no list of kinds, `frames: integer` with no minimum, `width` and `height` with no default. The misfires seen in rehearsal (a `list_models` loop, a clip under 8 frames) and the render that took twice as long (the model picked 1280x720) are all things the prompt never told the model. The recommendation is therefore: **measure first** with a tool-call scorecard on four models already on disk; **make the tool prompt honest and smaller, and keep its fixed part in the prompt cache**; build a **guided Keep going run** in which the app does the steps that have one right answer (make and bind the project, hold the tool list to the current stage, keep to a time budget, say what it is doing) behind a preflight check; and let **the app, not the model, own render settings the user did not state** (the plan card shows and prices the real settings, with an app default the model cannot exceed and a one-click raise). Demo model: Qwen3.8 27B MLX 4-bit, on disk, the only model that has finished a hands-off film (FU-219). Grammar-constrained decoding on MLX is parked; a per-model native call format is built only where the scorecard says it pays.

## Recommended order of work

Sizes: **S** up to half a day, **M** one to two days, **L** three to five days of agent work including tests. Estimates, not measurements.

| # | Work | FU | Size | Agent | Needs | Why here |
|---|---|---|---|---|---|---|
| 1 | Tool-call scorecard script, first run on four models | FU-230 | M, plus about 3 h of model time | Sonnet | nothing | Every later claim is checked against it; gives the demo model verdict |
| 2 | Honest, staged tool prompt (enums, ranges, defaults; Storyboard group only when needed; history budget counts the block) | FU-231 | M | Sonnet | 1 for the before figure | Cheapest fix, helps every model, halves the wait before a plan card |
| 2 | Agent loop: same-call guard, safe argument repair, corrective message names the allowed values | FU-233 | S to M | Sonnet | nothing | Ends the loop seen in R1; runs beside row above |
| 2 | Plan card always shows size, length, frames, steps and a time priced for those settings | FU-240 | S | Sonnet | nothing | Independent files; removes the wrong "at the recommended size" time |
| 3 | App default and ceiling for render settings, Raise control, editable fields, per-chat memory | FU-241 | M to L | Opus | FU-240 | The 2x render cannot happen silently again |
| 3 | Prompt cache keeps the fixed tool prefix across turns on hybrid models | FU-232 | M | Opus | 1 (to prove the miss) | Removes 15 to 20 s from every nudge with Qwen3.8 |
| 4 | Guided run: app-made project, stage tools, demo preset | FU-237 | M | Opus | FU-231 | The first-project failure goes away by construction |
| 4 | Time budget, wrap-up, step list, status line, resume after a stall | FU-238 | M | Opus | FU-237 | What the audience follows |
| 4 | Capability probe (3 calls, cached per model) and preflight API; preflight panel | FU-235, FU-239 | M, S to M | Sonnet | FU-231 | Warn before starting |
| 5 | Three rehearsals of the demo film, then live or recording | (in FU-237) | S | owner present | 2 to 4 | Go or no-go on evidence |
| 6 | Native call format on MLX for templates that take tools (LFM2.5 first) | FU-234 | M | Opus | scorecard still fails after 2 | Only if measured |
| 6 | "Tool calls" mode in Benchmarks | FU-236 | M | Sonnet | D-87 keeps Benchmarks | The scorecard, in the app |
| - | Grammar-constrained tool calls on MLX | FU-242 | L | - | parked | See section 2 |

**Demo-ready cut:** rows 1 and 2, the ceiling part of row 3 (FU-241 steps 3.2 and 3.3), row 4, then row 5. About 12 agent-days; in three parallel worktrees (tools, Keep going, plan card) about a week including the rehearsal. **Smallest useful cut** if the podcast is sooner: the loop guard (FU-233), the honest tool block (FU-231 step 2.2 only), the card's facts and price (FU-240), the app-made project with the demo preset (FU-237 steps 1.1 and 1.7) and the budget with the status line (FU-238 steps 1.3 and 1.4): about 6 agent-days, three calendar days. Everything else can follow the podcast. The podcast date is not in the files read, so the cut is the owner's call.

Parallel streams and the files each owns: **T** (tools: `mlx_worker_request.py`, `agent*.py`, `tools/studio.py`, `state/_helpers.py`, the scorecard script), **K** (Keep going: `keep_going/**`, `routes/keep_going.py`, `ChatKeepGoing.tsx`, `keepGoing.ts`, `DetachedTurnBubble.tsx`), **P** (plan card: `tools/studio_gate.py`, `StudioPlanCard.tsx`, `studioChat.ts`), **C** (cache: `mlx_worker_prompt_cache.py`, `mlx_worker_generate.py`). Shared and merged by the integrator: `state/generation.py`, `src/locales/*/chat.json`. Files already over the soft cap get a sibling module, not more lines: `generation.py` (1,189 lines), `agent.py` (662), `keep_going/controller.py` (657).

## Challenges, said once

- **Keep going live on a podcast: risky.** It has run clean once with Qwen3.8 27B (1 h 37 min, FU-219) and never in a short form. Alternative already in `demo-path.md`: talk over a recording. Rule proposed in section 1: live only after three clean rehearsals in a row.
- **"Remind the user to specify resolution": weak on its own.** It depends on the user remembering, and in a hands-off run there is nobody to remind. Alternative: an app default the model cannot exceed, shown on the card with a raise control (section 3).
- **Grammar-constrained decoding: costly now.** The MLX worker has no such path (it does not read `jsonSchema` at all), it needs a new dependency and a licence check, and it touches the same cache code as FU-232. Alternative: fix the prompt, measure, revisit with numbers.

## What the evidence and the code say

### From runs (quoted, not re-run)

| # | Fact | Source |
|---|---|---|
| E1 | LFM2.5 24B-A2B (114 tok/s) sent four calculator calls of which two were unreadable, asked for a clip under 8 frames (refused), and in Keep going called `list_models` in a loop; no project after 90 s | `review/f:docs/progress/review-f.md`, R1 |
| E2 | Qwen3.8 27B 4-bit (about 30 tok/s) made one clean calculator call and a correct Studio plan first time; plan card at 34 s (R2), 40 to 60 s (R3), 37.9 s (R4), of which 15 to 20 s is "Processing prompt" | `review/f:docs/plans/demo-path.md` |
| E3 | With no size in the prompt Qwen chose 1280x720: render 3 min 36 s at 1280x704. With "at 768x512": 1 min 38 s to 1 min 48 s (three runs). The card estimated 1 min 29 s | same |
| E4 | Keep going was not run with Qwen3.8 in stream F's rehearsals. It had been run once before: a 2-minute ad brief, stop `complete` after 1 h 37 min, film 80 s for a 120 s brief, two nudges answered with text only, rendered twice | FU-219 |
| E5 | LFM2.5 can make a project in Keep going: its first `storyboard_project create` was untagged and one brace short, the nudge told it, the next turn made the project and two shots | FU-199 |
| E6 | Qwen3.8 itself wrote every `storyboard_audio add` without the opening tag and one brace short in one rehearsal | FU-196 |
| E7 | LFM2.5 wrote `<tool_call>` JSON, not its own block, "every time (the MLX worker's tool prompt asks for it)" | FU-192 |
| E8 | LFM2.5 called `list_models` without `kind` once and corrected itself; LFM3 1.7B could not drive `storyboard_song` | status 2026-11-08; FU-190 |
| E9 | A 4-clip LTX storyboard takes about 25 minutes; a hung job once kept one turn alive for 76 minutes | `tools/studio.py` comment; FU-207 |

### From the code (read on this branch)

| # | Finding | Where |
|---|---|---|
| F1 | On MLX the tools reach the model as one generic text block that asks for `<tool_call>` JSON. It prints each parameter as `name: type (required) - description`. **Enums, minimum and maximum, defaults and the shape of nested objects are not printed.** The model's own chat template is not given the tools | `mlx_worker_request._format_tools_for_prompt`; `mlx_worker_generate.py:104-108, 351-353`; `mlx_worker_prompt._build_prompt_text` |
| F2 | Measured by running that formatter on the real schemas (no model): 20 Studio tools, **14,708 characters** (about 4,000 tokens at 3.6 characters per token; not tokenised). The seven `storyboard_*` tools are 7,814 of them (53 %); `storyboard_project` alone is 3,000. The three chat tools add 1,166, the Studio hint 780. The same list as JSON schema is 23,540 characters | scratch script, section 6 |
| F3 | What the model actually reads: `list_models` has `kind: string (required)` and nothing else; `generate_video` has `width: integer`, `height: integer`, `seconds: number`, `frames: integer`, `steps: integer` with no text at all; `storyboard_shot` has `action: string (required)` with the actions only in the prose description | same |
| F4 | The llama.cpp path is different: the server starts with `--jinja` and gets `tools` in the request, and tool calls stay structured in the history. GGUF models already use their own template | `llama_cpp_engine.py:671, 1025, 1114, 521-546` |
| F5 | Models loaded through the vision runtime (Gemma 4 and Qwen-VL prefixes) get no tool list at all: that prompt is built from the system prompt only | `helpers/chat_template.py:86-100`; `mlx_worker_multimodal.py` |
| F6 | Reading and repair already exist: three call formats, the LFM2.5 block, a call that lacks only closing braces is mended, an unreadable call gets an error with the reason and the format (at most two per turn), a broken call beside a good one is reported. A schema fault comes back as a tool error, for example "'kind' must be one of image, video, voice, stt, music" | `agent_parse.py`, `agent_lfm_calls.py`, `agent_call_repair.py`, `agent.py`, `mcp_bridge/tools.validate` |
| F7 | **Nothing stops the same call being made again and again.** A Studio turn allows 30 tool steps and Keep going stops after three nudges without progress, so a looping model can make about 120 calls before the run ends | `agent.py`; `state/generation.py:764`; `keep_going/run.py` `NO_PROGRESS_LIMIT` |
| F8 | Tool steps run at the chat's temperature (default 0.7) with the chat's reply limit (default 4,096 tokens) | `agent.py:495-503`; `models/__init__.py:133-134` |
| F9 | The history budget does not count the tool block: it reserves 512 tokens for "tool-schema overhead" against a block of about 4,000. At 8K with the default reply limit there is no room on paper (8,192 less 4,096 less about 4,300). Keep going stops a run at 85 % of the context | `state/generation.py:459-464`; `state/_helpers.py:91-108`; `run.py` `CONTEXT_FULL_SHARE` |
| F10 | The MLX prompt cache is one slot, reused by longest common prefix. For hybrid models (Qwen3.5, 3.6, 3.8) the recurrent state is saved only at the last prompt token, and reuse needs the shared prefix to reach it. Inside one turn the prompt only grows, so it hits. On a new user turn or a nudge the history is rebuilt from the saved thread, which keeps no tool results, so the prompt parts ways before that point and **the whole prompt, tool block included, is processed again**. Read from the code; not timed | `mlx_worker_prompt_cache.py:151-160, 204-223`; `state/_helpers.py` |
| F11 | In Keep going the model must create the project itself; a run has a nudge cap, a stall limit and a grant, but **no time budget**; the status carries phase, nudge count and the job waited on | `keep_going/adapters.py:54-61`; `run.py`; `controller.status` |
| F12 | After a `job_stalled` stop the job is left as it is, and Resume then waits on that same job again until the stall limit passes | `stall.py`; `controller._dispatch`, `_blocked` |
| F13 | A nudge can name `storyboard_batch`, a tool Chat does not offer | `project_state.py:140`; `tools/studio.py` `STUDIO_TOOL_NAMES` |
| F14 | A render already leaves out shots that have no clip, so a film can be closed early | `storyboard/timeline.py:90-97, 222` |
| F15 | The plan card shows kind, prompt, model and memory always; **size only when the call gives width and height** (else the recommended size from the catalogue); **frames, fps and steps only when the call gives them**; the time is the catalogue figure at the recommended size, labelled "at the recommended size" when the call differs. **No field can be edited.** "Edit by chat" declines the plan and costs another model turn | `tools/studio_gate.plan_item`; `StudioPlanCard.tsx` `ItemFacts` |
| F16 | `generate_video` takes any size from 256 to 4096; the default (the model's recommended size) applies only when the call leaves the size out. The job's own estimate is scaled to the real size (`variant_seconds`); the card's is not | `mcp_bridge/tools.py:350-439` |
| F17 | A Storyboard project defaults to 1280x720; `draft: true` makes takes at half that size | `storyboard_tools.py:60-61`; `storyboard/batch.draft_size` |
| F18 | The app already turns a brief into shots with one JSON reply | `storyboard/planner.py` (S8) |
| F19 | The Tools, Studio and Keep going controls are hidden when the catalogue row lacks a tools tag. That is a catalogue claim, not a measurement | `catalog/capabilities.py`; `ChatComposer.tsx:171, 436, 459` |
| F20 | A JSON schema is passed to every engine but the MLX worker never reads it | no `jsonSchema` in `mlx_worker*.py`; `llama_cpp_engine.py:186-191` |

## 1. Keep going, ready for a demo

### Problem

Keep going works as a long, unattended run with a strong model and has no short form. On a stage it fails in five ways: the first step (create the project) is the model's and a weak model never gets past it (E1); nothing bounds the time; the audience sees "nudge 3 of 20, the model is working" and not what is being made; a slow or stuck render looks like a hang; and there is no check before starting that the model can do the job.

### Evidence

E1, E4, E5, E9, and F7, F9, F11 to F14. Two points decide the design. First, the failing step is the one with a single right answer: a project with a name and a format. Second, the app already knows the stage of the film at every moment (`FilmState`: no shots, stills missing, clips missing, music missing, render missing), so it can show it and can hold the tool list to it.

### Options

| | Option | For | Against |
|---|---|---|---|
| A | Change nothing in the mechanism; name Qwen3.8 27B as the model and rehearse | No work | No time bound, nothing for the audience to follow, still breaks on the first step with a weaker model |
| B | **Guided run**: the app makes and binds the project, each turn gets only the tools of the current stage, a time budget with a wrap-up rule, a step list and one status line, a preflight check | Removes the observed failure by construction; reuses `FilmState`; the model still writes the shots and calls the tools | About 6 agent-days; a second way to start a run to keep tested |
| C | The app runs the whole pipeline and asks the model for words only (the planner in F18, then generate, music, render) | Most reliable; works with a small fast model | It is the Storyboard's own "film from an idea", not Keep going; shows nothing of a model using tools; a second pipeline to maintain |
| D | Recording only | No risk on air | Not a live demo |

### Recommendation

**B, with D as the fallback when the rehearsal rule is not met.** Reason: B fixes what failed (no project) without depending on the model, gives the audience a list of steps and a clock, and keeps the part worth showing, a local model driving the studios. C is kept in reserve for one step only: if the scorecard shows that models fail at writing shots as tool calls, the guided run may fill the shots through the planner after two failed nudges on that step. Not built unless measured.

### Steps

| # | Step | Size | Notes |
|---|---|---|---|
| 1.1 | **App-made project.** When a guided run takes the brief, the app creates the Storyboard project (name from the brief's first words, format from the preset), binds it to the run, and the first turn carries one app line: the project exists, its id, revision and size; add the shots. `storyboard_project create` is not offered in that run | S | New module `keep_going/guided.py`; `controller.on_user_turn` only calls it |
| 1.2 | **Stage tools.** The stage comes from `FilmState`: Plan (shots), Make (stills and clips), Sound (music, voice, captions), Finish (render). A nudge turn gets the tools of its stage plus the job and read tools: three to six tools in place of twenty. A call to a Studio tool outside the stage answers with the tools that are on. Fix F13: offer `storyboard_batch` status and resume in Make, or reword the step | M | Needs the per-turn tool subset from FU-231 |
| 1.3 | **Time budget and wrap-up.** `budgetMinutes` on a run (off by default, 10 in the demo preset, 5 to 240). At 80 % of the budget, or when the time left is under the next job's estimate, the next step becomes "render the film now from the shots that are ready" (F14) and optional lanes are skipped. At the budget the run stops with a new resumable code `time_up`. Stop keeps its meaning; the stopped chip also says what is saved and links to the project | M | The nudge cap stays; the demo preset sets it to 8 |
| 1.4 | **Step list and status line.** `status()` gains the stage, a list of steps with done, current and to do ("Clips 1 of 2"), elapsed and budget, and one sentence built from the phase, the tool in flight and the job's own progress: "Making clip 2 of 2, about 1 min 10 s left". Shown in the composer chip and in the thread in place of "This reply is still being made" | M | New `KeepGoingSteps.tsx`; strings in `en`, `null` elsewhere |
| 1.5 | **Preflight** (below) before Turn on | S to M | FU-239; uses FU-235 |
| 1.6 | **Resume after a stall.** When a run stopped on `job_stalled` and the job is still unfinished, Resume asks: cancel the stuck job and redo that step (default), or keep waiting (F12). The chip shows the job's time left while it waits | S | |
| 1.7 | **Demo preset** in the Keep going panel: two shots of 3 s, a music bed, no narration, nudge cap 8, budget 10 minutes, project size from measurement C10 (768x512 class). It fills the cap, budget and format; it never preselects how renders are approved (D-86 stands) | S | |
| 1.8 | **Three rehearsals** on the Mac with the demo model, timed, with the full path from `demo-path.md` around them | S | Results in `docs/plans/demo-path.md` |

**Preflight checks** (all cheap reads; they warn, only "no model loaded" blocks):

1. A chat model is loaded, and its tool verdict: Good, Risky, Not for tools, or "not tested: test now (about 20 s)" (FU-235).
2. Context size: under 16K warns that the run will stop early (F9); 32K recommended (R2 ran at 32K).
3. What the brief needs is installed: a video model, an image model, the music pack when the film has music, a voice when the brief asks for narration, ffmpeg for the render.
4. Memory: the chat model plus the largest job against free memory, with the "Free GPU memory while rendering" setting it implies.
5. Time: the preset's estimate from the catalogue times and the model's measured turn time, against the budget.
6. How renders are approved has been chosen.

**Model recommendations** (names from the Hugging Face cache, by name only; the owner's rule is newer and faster first, so a model is ranked by release date and measured speed on the M4 Max once it passes the tool bar):

| Model on disk | Use for the demo | Basis |
|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | **Yes, the demo model** | E2, E4; about 30 tok/s, about 15 GB |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | Not for tools today; first to promote if it passes after FU-231 or FU-234 | E1, E5, E7; four times faster |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` | Untested with Studio tools; candidate for the same quality, faster | scorecard |
| `unsloth/Qwen3.8-27B-GGUF` (with `z-lab/Qwen3.8-27B-DFlash2-GGUF`) | Untested with Studio tools; runs on the path that uses the model's own template (F4) | scorecard, claim C7 |
| `mlx-community/Qwen3-8B-4bit`, `mlx-community/Qwen3-4B-bf16` | Scorecard only (the 8B and the 4B) | FU-190: the catalogue hides Tools for Qwen3-4B |
| `LiquidAI/LFM3-1.7B-MLX-4bit` | Floor of the scorecard | E8 |
| `mlx-community/Qwen3.6-27B-4bit` | No: older than 3.8 | owner's rule |

The rest of the demo pipeline is on disk by name too: `dgrauet/ltx-2.5-mlx-q4` (clips; ran in R1 to R4), `speridlabs/iris-3b` and `black-forest-labs/flux-3-image-klein` (stills), `ACE-Step/Ace-Step1.5` (music), `prince-canuma/Kokoro-82M` (voice). Nothing needs a download. Whether each snapshot is complete was not checked.

**Time for the demo film** (estimate from measured parts; to be replaced by rehearsal figures): two clips of 3 s at 768x512, 3 min 20 s (E3); two stills, under a minute; music, 3 to 10 s (status 2026-11-07); final render, a few seconds (a 32.9 s film took 7.8 s); model turns, four to six, each 20 to 40 s today and less once the tool prefix stays cached. **About 6 to 9 minutes**, so a 10-minute budget. The size matters: at the Storyboard's default 1280x720 a 4-clip batch took about 25 minutes (E9), which is why the preset sets a smaller one, and why Storyboard clip times are measured before the preset is fixed (C10). Memory: stream F's path peaked at about 37 GB (chat model, LTX-2.5, speech; `demo-path.md`) and the music model peaks at 21 to 23 GB by itself (status 2026-11-07), with one heavy job at a time, so no model swap is expected on the 64 GB Mac; the rehearsal confirms it.

**When a render takes minutes:** the run waits inside the turn as today; the status line carries the job's step and time left; the presenter talks. The wait costs no tool step (existing rule). The stall limit in the preset is 5 minutes, the current minimum.

### Tests

- pytest, `tests/test_keep_going_guided.py`: the project is made and bound before the first model turn; a second `create` is not offered; the stage for each `FilmState`; the tool subset per stage; wrap-up at 80 % and when the next job does not fit; `time_up` is resumable; the step list for five film states; the status sentence for a turn, a wait and a plan card; Resume after a stall with and without cancelling; preflight verdicts with fakes (no model, 8K context, no video model, tight memory).
- vitest: the panel with the preset and the preflight rows; the chip with steps, clock and status; the stopped chip with "saved so far" and the stall choice; `KEEP_GOING_STOP_CODES` lists `time_up`.
- **E2E (new feature gate):** `scripts/e2e_phases/keep_going.py` gains: start with `preset: "demo"` and `budgetMinutes` (the armed run carries them; 4 is refused with 422), status has `steps` and `activity`, `GET /api/chat/keepgoing/{id}/preflight` answers with no model loaded (a verdict, not an error). In the model-loaded phase, skipped when no model is on disk: a guided start with the smallest model binds a project id before the turn ends, then Stop.
- Playwright (next free number on `review/integrate`, 37 at the time of writing), scripted stream: preflight rows, step list moving through the stages, the status line, the stall choice.

### Risks

- A stage hides a tool the model wants (a new shot during Make). The off-stage answer names what is on, and the stage follows the film: adding a shot moves it back to Plan.
- The app-made project has a dull name. The model can rename it; the name is the brief's first six words.
- The wrap-up shows a shorter film than the brief asked for. The status line says so ("closing early: 1 of 2 clips ready").
- Two ways to start a run. The guided run is the same controller with a preset; the adapter interface is unchanged.

### What would make us stop

- After the guided changes, Qwen3.8 27B fails to finish the demo film inside the budget in two of three rehearsals: Keep going stays out of the live path and the recording is used. No further guided features are built before the cause is known.
- The scorecard shows the demo model below the tool bar on the stage tasks: stop and fix section 2 first.
- The guided code needs changes inside the controller's locking: stop and review; that code holds the "a user message always wins" rule.

## 2. Small fast models and tools

### Problem

Small fast models write calls that cannot be read, pass wrong or missing arguments, and repeat the same call. The owner wants the fast model (114 tok/s against 30) to be usable. Today the app cannot say which models are good enough, and its own tool prompt causes part of the trouble.

### Evidence

E1, E5 to E8 and F1 to F10. The two argument faults seen in rehearsal map directly onto F3: the model was never told the kinds or the frame minimum. The format faults map onto F1 and E7: LFM2.5 is asked for a format it was not trained on and gets it slightly wrong. The loop has no guard (F7). The wait before a plan card is the block's size (F2) and, for Qwen3.8, a cache miss on every new turn (F10).

### Options

| | Option | For | Against |
|---|---|---|---|
| A | Native call format per model: give the model's own template the tools (LFM2.5's block, the XML form, the JSON form) | The format the model was trained on; readers for all three exist (F6) | As JSON schema the list is 23,540 characters against 14,708 (F2): slower and more context unless the list is cut first; per-family template quirks; changes what the prompt cache sees |
| B | Grammar-constrained decoding of the call | A call that always parses and always fits the schema | No path on MLX today (F20); a new dependency with a licence check; touches the worker's sampling and the cache; cannot fix a wrong but valid choice (the 720p size); the llama.cpp path may already do this (to confirm, C7) |
| C | Repair, with one corrective retry that says exactly what was wrong | Mostly exists (F6) | What is missing is small: the allowed values in the message, safe fixes of harmless faults, and a guard against the same call again |
| D | Fewer and shorter tool descriptions per turn, and the fixed tool prompt kept in the cache | Engine-independent, helps every model, cuts 15 to 20 s; fixes the cause in F3 | The tool list is no longer the same in every turn, so the staging rule must be simple and tested |
| E | Capability probe: three calls, cached per model | Honest warning before a run; feeds the preflight | Three calls are a smoke test, not a guarantee |
| F | Tool-call benchmark in Benchmarks: parse rate, argument validity, task completion | The owner sees the numbers per model; repeatable | Benchmarks is under review (D-87); the metrics should be proven as a script first |

### Recommendation

**Prompt first, measured: D and C now, E and F as the measuring tools, A only for a family the scorecard still fails, B parked.** Reason: the code shows the model is not told what a valid call is (F3) and is shown twice the text it needs (F2), so the cheapest change attacks the observed faults directly and helps Qwen as much as the small models. A and B treat a symptom that may shrink once the prompt is right; building them before measuring would be guessing. The fast model the owner prefers gets the first native-format attempt if it still fails.

### Steps

| # | Step | Size | Notes |
|---|---|---|---|
| 2.1 | **Scorecard** (FU-230): `scripts/tool-call-scorecard.py`, see the measurement plan | M | Reuses the app's own reader and validator, so it scores what the app would do |
| 2.2 | **Honest tool block** (FU-231): the MLX text block prints, per parameter, the allowed values, the range, the default and "required"; a nested object lists its keys. One line each, no prose added. The bridge's tool list and its golden test are not touched: this is the worker's formatter | S | `mlx_worker_request.py` |
| 2.3 | **Staged Studio list** (FU-231): core tools always (models, voices, image, video, speech, music, sound, job tools: about 6,400 characters); the Storyboard group (7,814 characters) only when the message asks for a film, a project is bound to the chat, or the model called one of its tools (the answer says the group is now on). Once on, it stays on for the chat, which also keeps the prompt prefix stable. Keep going passes its stage subset through the same switch | M | `tools/studio.py`; routing words tested per locale default (`en`), not translated |
| 2.4 | **History budget counts the tool block** (FU-231), and the preflight warns under 16K | S | `state/_helpers.py`, `generation.py` |
| 2.5 | **Loop guard** (FU-233): the same tool with the same arguments and the same result twice in a row gets a corrective message ("you already have this answer: ...; do the next step"); a third time ends the turn with a reason Keep going reads as a stall cause | S | New `agent_guard.py`; `wait_for_job` is exempt (it is meant to repeat) |
| 2.6 | **Argument repair** (FU-233): only fixes that cannot change meaning: a number written as text, an enum in another letter case, a value one step outside a range moved to the limit with a note in the result ("frames 6 raised to 8"). Everything else stays an error, and the error names the allowed values | S to M | The note goes to the model and the card |
| 2.7 | **Tool-step temperature** (FU-231): 0.2 for tool steps if claim C5 holds | S | One line, behind the measurement |
| 2.8 | **Prompt cache prefix** (FU-232): for hybrid models keep a second saved state at the end of the system message, and restore from it when the prompts part ways later than that point. Log, per generate call, the prompt tokens, the tokens reused and the seconds spent | M | Any doubt falls back to a fresh prefill, as the module does today |
| 2.9 | **Probe** (FU-235): three calls on the loaded model with canned tool answers (nothing real runs): one with an enum, one with a nested object and a number, one that gets an error back and must fix it. Verdict Good, Risky or Not for tools, saved per model, quantisation, engine and tool-prompt version in the app data folder | M | `GET` and `POST /api/chat/tools/probe` |
| 2.10 | **Native format on MLX** (FU-234), only if the scorecard still fails a family: pass the tools to the model's own template when it takes them, keep the text block as the fallback, send tool results in the template's own role | M | First check: does the LFM2.5 template accept tools (not read for this plan) |
| 2.11 | **Benchmarks mode "Tool calls"** (FU-236): the scorecard's tiers 1 and 2 with the three figures and a per-model history | M | After D-87 |

### Measurement plan

Every claim below is checked with real models before the change that rests on it is merged, and again after.

**Models** (all on disk): `mlx-community/Qwen3.8-27B-4bit`, `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, `mlx-community/Qwen3-8B-4bit` (the 8B), `mlx-community/Qwen3-4B-bf16` (the 4B). Extra columns when time allows: `unsloth/Qwen3.8-27B-GGUF` (the llama.cpp path), `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, `LiquidAI/LFM3-1.7B-MLX-4bit`.

**Tiers.**
- **Tier 1, one call** (no tool runs): 12 prompts, each with one right call (an enum, a required argument, a number in range, a nested object, a call that needs no arguments, two calls in one reply). Sent through the OpenAI-compatible chat endpoint with the Studio schemas, so no product code is needed. Scored with the app's own `read_calls` and `validate`: read without repair, read with repair, right tool, valid arguments, right values.
- **Tier 2, a short task** (real Studio chat, no renders): make a project at a named size and add two shots (checked in the saved project); name the recommended size of the installed video model; recover from a project name passed as its id. Scored: done or not, tool steps, unreadable calls, invalid calls, repeated calls, seconds, seconds of prompt processing.
- **Tier 3, the demo film** (only models that pass tier 2): the guided run, real renders, three runs.

**Runs:** five per prompt and task at the app's settings. Results are reported as counts ("52 of 60"), never a bare percentage; with five runs a difference of one or two is noise and is said to be. Output: a CSV and a Markdown table in `~/.chaosengine/test-results/`, and `docs/research/tool-call-scorecard-2026-11.md`.

**Conduct:** one model loaded at a time (64 GB), own backend port, a throwaway home folder with the model caches linked as stream F did, Hugging Face offline, the model lock file, unload after each model. About 3 hours of model time for four models and three conditions; tier 3 adds about 30 minutes.

| # | Claim | How it is checked | Holds when |
|---|---|---|---|
| C1 | The Studio tool block costs 15 to 20 s with Qwen3.8 | Time to first token with Studio on and off, same question, cold cache | The difference is within 5 s of that |
| C2 | A new turn re-processes the block on hybrid models (F10) | Reused tokens and seconds on step 2 inside a turn against the first step of the next turn | Step 2 reuses the block; the next turn does not |
| C3 | Printing enums, ranges and defaults raises first-try valid calls | Tier 1 before and after 2.2, per model | Up by 6 of 60 or more on at least two models, down on none |
| C4 | A staged list raises valid calls and cuts the wait | Tier 1 and time to first token with the core list against all twenty | Wait roughly halves; valid calls not lower |
| C5 | Temperature 0.2 on tool steps raises valid calls | Tier 1 at 0.7 and 0.2 | Up by 6 of 60 or more on a small model |
| C6 | The native format helps LFM2.5 | Tier 1 and 2 with the 2.10 prototype against the text block | Tier 1 at 54 of 60 or better |
| C7 | The llama.cpp path is more reliable than MLX for the same model | Qwen3.8 27B as GGUF against MLX, tiers 1 and 2 | Reported either way; decides whether B is worth building for MLX |
| C8 | The three-call probe predicts tier 2 | Probe verdict against tier 2 for every model and condition | Agrees in at least 10 of 12 cells |
| C9 | The guided run finishes the demo film in the budget | Tier 3, three runs | Three of three, each under 10 minutes |
| C10 | Clip time by size on this Mac | One 3 s LTX-2.5 clip each at 640x384, 768x512, 1024x576, 1280x704 | Sets the defaults in section 3; the card's time is within 15 % |
| C11 | The loop guard ends a repeat without hurting a good run | Tier 2 with the guard on: repeats stopped, no change for Qwen3.8 | No task lost that passed before |

**Proposed tool bar** (to tune after the first run): **Good** is tier 1 at 54 of 60 valid on the first try and all three tier 2 tasks in four of five runs; **Risky** is tier 1 at 42 of 60 or two tier 2 tasks; below that, **Not for tools**. The catalogue flag in F19 is then set from the measured result.

### Tests

- pytest: the formatter's text for three tools (golden text with enums, ranges, defaults); the staged list for six messages and for a bound project; the off-group call turns the group on; the history budget with and without tools; the loop guard (two repeats corrected, the third ends the turn, `wait_for_job` exempt); each safe repair and each refusal; the probe's verdicts and its cache key; the scorecard's scoring on recorded replies (`tests/fixtures/keep_going_qwen_unclosed_calls.json` is a first fixture).
- pytest, cache: a fake hybrid model; after the prompts part ways the restored state equals a fresh prefill token for token; a failed restore falls back to fresh.
- vitest: the verdict badge in the model picker and the preflight row.
- **E2E (new feature gate):** probe endpoint with no model ("not tested") in phase 7; in the model-loaded phase the probe runs on the smallest model on disk, returns a verdict, and the second call answers from the cache; after FU-236, the `tool_calls` benchmark mode in stream A's benchmarks phase (shape of the run and the three figures).
- `tests/test_mcp_contract.py` stays green without a change to the golden list (the bridge's tools are not edited).

### Risks

- The router leaves the Storyboard group off when the user wanted a film. The first Storyboard call turns it on, at the cost of one step.
- The staged list changes the prompt prefix once per chat. It is ordered core first, so the shared prefix survives.
- Cache work can corrupt a reply. The rule stays: equal to a fresh prefill, proven in tests, or not reused.
- The probe gives false comfort. The wording is "passed a 3-call check", and a run's own stop rules still apply.
- Lower temperature makes replies duller. It applies to tool steps only, never to the final answer.

### What would make us stop

- After 2.2 to 2.7 a small model is still under "Risky": it is marked Not for tools in v0.10.0. One native-format attempt for LFM2.5 only (the owner's fast model); if that is under 54 of 60, stop there.
- The cache change cannot be shown equal to a fresh prefill in every test: it is not merged. Correctness before speed.
- C3 and C4 do not hold (no gain from the honest, shorter prompt): stop, the diagnosis in this plan is wrong, and go back to the transcripts before building more.
- C7 shows the llama.cpp path clearly better for the same model: bring FU-242 forward and reconsider the default engine for tool chats.

## 3. Who decides the render settings

### Problem

The model picks the size and other settings, and the render can take twice as long as the user expects. The user sees it only on the card, where the time shown is for a different size.

### Evidence

E3 (3 min 36 s against 1 min 43 s), F15 and F16. Checked in the code as asked: the card is **partly there**. It always shows the model and memory; it shows size, frames and steps only when the model happened to pass them; its time is the catalogue figure at the recommended size even when the call asks for more; no field is editable, and the only way to change a setting is "Edit by chat", another 34 to 60 s turn (E2). In a run with automatic approval no card shows at all.

### Options

| | Option | For | Against |
|---|---|---|---|
| A | Remind the user to state the resolution (a hint in the composer, or the model told to ask) | Simple | Relies on memory; no help when renders are auto-approved or in Keep going |
| B | The card always shows resolution, length, frames, steps and a time priced for those settings | Small; the estimate function exists (F16) | Shows the surprise, does not prevent it; no card in auto-approved runs |
| C | App defaults per model and hardware that the model cannot silently exceed, with a clear raise control | Deterministic; also covers runs with no card | Needs a rule for "the user asked for it" |
| D | Editable fields on the card before Approve | Saves a model turn for a small change | The call that runs differs from the call the model wrote; must be told to the model |
| E | A one-line preference per chat ("always 768x512 for drafts") | No repeat work | Another setting to find and undo |
| F | The model asks one short question when a choice changes the time more than 2x | Feels attentive | Costs a full turn; depends on the model's judgement and an estimate it does not have; nobody to ask in a hands-off run |

### Recommendation

**C and B now, D and E next, A only as a small hint, F not built.** Reason: the card is already the question, so a second question from the model is a slower copy of it; and only an app-side default covers the runs where no card is shown. The owner's instinct (make the user's intent on size explicit) is met by the card showing the setting with its time and a one-click raise.

**The rule.** For Chat's `generate_video` and `generate_image` (not for external agents on the bridge, whose calls are explicit): each installed model has a **default** on this machine, the catalogue's recommended size, default length and recommended steps (768x512 for LTX-2.5 here). When the user's message of the turn, or the run's brief, names a size or resolution (a pattern like 1024x576, 720p, 4K), the call is honoured up to what the model and the machine allow. When it does not, a call above the default is brought down to the default's pixel count, keeping the shape the model asked for, and the result says so: "size set to 768x512 (app default; no size was asked for)". The card shows that line with **Raise**, which lists the next sizes with their times. With D, the card's fields can be changed before Approve, the time updates as they change (`src/utils/timeEstimate.ts` mirrors the Python estimate), and the model is told what ran. With E, a tick "use these settings for the rest of this chat" keeps them for the chat and shows a chip to turn it off, like the existing auto-approve chip.

Storyboard films set their size once, at project creation (1280x720 by default, F17). This plan leaves that default alone outside the demo preset; the plan card for `storyboard_generate` already shows time per shot and in total, and gains the size line. Whether chat-made projects should start smaller is asked in D-91 and decided with C10's figures.

### Steps

| # | Step | Size | Notes |
|---|---|---|---|
| 3.1 | **Card facts and price** (FU-240): the card always lists size, length in seconds, frames, steps (where the model uses them: LTX-2.5 distilled runs a set schedule) and the time for exactly those settings, filled from the same defaults the handler will use; the "at the recommended size" wording goes | S | `studio_gate.plan_item`, `StudioPlanCard.tsx` |
| 3.2 | **Defaults and ceiling** (FU-241): one function that returns, per installed model, the default and the largest allowed settings; the rule above applied before the gate; the note in the tool result and on the job card | M | New `tools/studio_defaults.py`; `GET /api/chat/studio/defaults` for the UI and the E2E check |
| 3.3 | **Raise control** on the card: the next two or three sizes with their times | S | |
| 3.4 | **Editable fields**: size, length and steps for image and video rows; the decision call carries the changes, validated against the tool's schema and the ceiling; the tool result names what changed | M | The decision route gains `overrides`; "Edit by chat" stays for everything else |
| 3.5 | **Per-chat memory** of the settings, with a chip to turn it off; ends with the chat | S | Stored with the chat, not in global settings |
| 3.6 | **Composer hint** (only if rehearsal still shows surprise): one line under the Studio switch, "renders use the fast default size unless you name one" | S | |

### Tests

- pytest: the card's facts for a call with no size, a named size and a larger size; the price equals `variant_seconds` for those settings; the ceiling with and without a size in the message (six phrasings), shape kept, note text; an override outside the schema or above the ceiling is refused; the model's next step sees the changed arguments; the per-chat memory applies and clears.
- vitest: the card always shows the five facts; editing a field updates the time; Raise lists sizes with times; the chip.
- Playwright (extends `15-chat-studio-tools.spec.ts`, scripted stream): the card shows size, length, frames, steps and time; change the size; Approve; the job card shows the changed size.
- **E2E (new feature gate):** a check in phase 7 that `GET /api/chat/studio/defaults` returns, for every installed image and video model, a default no larger than its ceiling and a positive time (skips with a reason when no model is installed).

### Risks

- The pattern misses a size written in words ("in HD"). The card shows the default and Raise is one click; in an auto-approved run the note is on the job card.
- A user who wants large renders by habit finds the default in the way. The per-chat tick removes it for that chat.
- The edited call and the model's memory drift apart. The tool result states the settings that ran, and the saved turn keeps them.

### What would make us stop

- C10 shows the default size is not much faster than the next one up on this Mac (under 1.3x): the ceiling adds friction for no gain; ship 3.1 only.
- Editable fields cannot be kept consistent with the model's history without touching the agent loop's message handling: ship Raise (a fixed list) and leave free editing out.
- The card passes 500 lines: split it before adding.

## 4. Not in this plan

Windows and Linux (the CUDA engines take tools their own way and are not read here); voice calls; a Keep going adapter for chats without Studio tools; changes to the bridge's tool descriptions for external agents; the Storyboard default size outside the demo preset.

## 5. Not verified

- No model was loaded. Token counts are characters divided by 3.6. The prefill speed implied by E2 and F2 is an inference.
- F10 (the cache miss on a new turn for hybrid models) is read from the code, not timed. Claim C2 checks it.
- Whether the pinned llama.cpp build constrains tool calls with a grammar was not checked. Claim C7 checks the outcome.
- Whether the LFM2.5 chat template accepts a tools argument was not read (model files were checked by name only).
- Clip times at sizes other than 768x512 and 1280x704, and Storyboard clip times against Video Studio clip times. Claim C10.
- Whether the model snapshots on disk are complete, apart from those that ran in stream F's rehearsals.
- Why "Processing prompt" was 5 s longer at 8K than at 32K (`demo-path.md`): not explained; the scorecard records both.

## 6. How the tool-list figures were made

A read-only script imported `TOOLS`, `chat_schema` and `STUDIO_TOOL_NAMES`, built the 20 schemas as Chat does, and passed them to `_format_tools_for_prompt`. Output: 14,708 characters for the 20 Studio tools; per tool, largest first: `storyboard_project` 3,000, `make_sound` 1,156, `generate_video` 1,078, `storyboard_song` 1,062, `storyboard_audio` 983, `generate_image` 972, `make_music` 888, `speak` 884, `storyboard_shot` 857, `storyboard_captions` 708, `storyboard_generate` 637, `storyboard_render` 567, `music_library` 454, `wait_for_job` 275, `cancel_job` 206, `job_status` 180, `list_jobs` 168, `list_models` 139, `system_status` 117, `list_voices` 64. The script lives in the session scratchpad and is not committed; FU-230 turns it into a test.

## Decision rows to add

For `docs/hub/decisions.md`, newest at the top. The link points at this branch; repoint it after the merge.

| ID | Decision | ELI5 | Question for Dan | Recommendation | Status | Date | Links |
|---|---|---|---|---|---|---|---|
| D-91 | Who decides render settings in Chat | The chat model picked 1280x720 for a clip and the render took 3 min 36 s, not 1 min 43 s. The plan card showed a time for a different size and nothing on it can be changed. | Remind the user to state the size, let the model ask, or let the app hold a default? | **App default with a raise control (recommended).** Each model has a default on this machine (768x512 for LTX-2.5). A call above it is brought down unless the message names a size; the card always shows size, length, frames, steps and a time for exactly those settings, with Raise, editable fields and "use for the rest of this chat". The model does not ask a question: the card is the question, and a hands-off run has nobody to ask. Open point: should a film project made from Chat start below 1280x720? Decide with measured clip times. | open | 2026-11-18 | [plan](https://github.com/cryptopoly/ChaosEngineAI/blob/review/plan/docs/plans/keep-going-and-tools-v2.md), FU-240, FU-241 |
| D-90 | Small fast models and tool calls | The fast model (114 tok/s) writes broken calls and loops; the 27B model (30 tok/s) works. On the Mac the model is shown a 4,000-token tool list that does not say which values a call may take, in one format for every model. | Build per-model formats and grammar decoding now, or fix the prompt and measure first? | **Fix the prompt, measure, then decide (recommended).** A scorecard on four models on disk (Qwen3.8 27B, LFM2.5 24B-A2B, an 8B, a 4B): calls read, arguments valid, task done. Then: print allowed values, ranges and defaults; show the Storyboard tools only when needed; keep the fixed tool text in the cache; stop repeated calls; a 3-call check per model that feeds a warning before a run. A native format is built only for a model that still fails (LFM2.5 first). Grammar decoding on MLX is parked. | open | 2026-11-18 | [plan](https://github.com/cryptopoly/ChaosEngineAI/blob/review/plan/docs/plans/keep-going-and-tools-v2.md), FU-230 to FU-236, FU-242 |
| D-89 | Keep going for a live demo | Keep going finished one long film with Qwen3.8 27B (1 h 37 min) and made no project in 90 s with the fast model. It has no short form, no time limit and shows the audience little. | Build a guided short run for the podcast, or keep it to a recording? | **Guided run, live only after three clean rehearsals (recommended).** The app makes and binds the project, each turn gets only the tools of the current stage, a 10-minute budget closes the film early if needed, the chip shows the steps and one line on what is being made, and a preflight check warns before starting (model verdict, context, models installed, memory, time). Demo model: Qwen3.8 27B MLX 4-bit, on disk. If two of three rehearsals fail, use the recording. Render approval stays an explicit choice (D-86). | open | 2026-11-18 | [plan](https://github.com/cryptopoly/ChaosEngineAI/blob/review/plan/docs/plans/keep-going-and-tools-v2.md), FU-235, FU-237 to FU-239 |

## FU rows to add

For `FOLLOW_UPS.md`, in this order.

| ID | Item | Trigger / Condition | Notes |
|----|------|---------------------|-------|
| FU-230 | Tool-call scorecard: a script that scores chat models on calls read, arguments valid and task done | Queued (2026-11-18), D-90. First: it gates FU-231 to FU-236 | `scripts/tool-call-scorecard.py`, three tiers (one call through the OpenAI-compatible endpoint with the Studio schemas; a short Studio task with no render; the demo film). Models on disk: Qwen3.8 27B MLX 4-bit, LFM2.5 24B-A2B MLX 4-bit, Qwen3 8B 4-bit, Qwen3 4B; extra: Qwen3.8 27B GGUF, the MTPLX speed build, LFM3 1.7B. Five runs each, counts not percentages, one model loaded at a time. Claims C1 to C11 and the proposed tool bar in [the plan](docs/plans/keep-going-and-tools-v2.md). Output in `docs/research/tool-call-scorecard-2026-11.md`. |
| FU-231 | Honest, staged tool prompt on the MLX path | Queued (2026-11-18), D-90. After the FU-230 baseline | The text block prints allowed values, ranges, defaults and nested keys (today `list_models` shows `kind: string (required)` and nothing else; `generate_video` shows no limits). Core Studio tools always; the seven Storyboard tools (7,814 of 14,708 characters) only when the message asks for a film, a project is bound, or the model calls one. The history budget counts the tool block (it reserves 512 tokens against about 4,000). Tool steps at temperature 0.2 if claim C5 holds. The bridge's tool list and its golden test are unchanged. |
| FU-232 | Prompt cache: keep the fixed tool prefix across turns on hybrid models | Queued (2026-11-18), D-90. After FU-230 proves the miss (claim C2) | `mlx_worker_prompt_cache.py` saves the recurrent state only at the last prompt token, so a new user turn or a Keep going nudge re-processes the whole prompt with Qwen3.5, 3.6 and 3.8 (read from the code, not timed). Add a second saved state at the end of the system message and a log line per call (prompt tokens, reused, seconds). Must equal a fresh prefill in tests or it is not merged. |
| FU-233 | Agent loop: stop repeated calls, repair harmless argument faults | Queued (2026-11-18), D-90 | No guard exists: a Studio turn allows 30 steps and a run 3 idle nudges, about 120 calls. Same tool, arguments and result twice: a corrective message; a third time ends the turn with a reason (`wait_for_job` exempt). Safe repairs only (a number as text, enum letter case, a value one step outside a range moved to the limit with a note); other faults stay errors that name the allowed values. New `agent_guard.py` (`agent.py` is over the soft cap). |
| FU-234 | Native tool-call format on MLX for templates that take tools | Only if the scorecard still fails a family after FU-231 and FU-233; LFM2.5 first | Pass the tools to the model's own template, keep the text block as the fallback, send results in the template's role. Readers for the three formats exist. As JSON schema the list is 23,540 characters, so this follows the staged list. First check: the LFM2.5 template accepts tools. Stop if tier 1 stays under 54 of 60. |
| FU-235 | Tool capability probe (3 calls, cached per model) and its API | Queued (2026-11-18), D-89 and D-90. After FU-231 | Three calls with canned answers on the loaded model: an enum, a nested object with a number, a recovery from an error. Verdict Good, Risky or Not for tools, saved per model, quantisation, engine and tool-prompt version. `GET` and `POST /api/chat/tools/probe`. Also reports a model on the vision runtime, which gets no tool list today. The catalogue's tools flag is then set from measured results. E2E: phase 7 (no model) and the model-loaded phase. |
| FU-236 | Benchmarks mode "Tool calls" | After D-87 keeps Benchmarks and FU-230 has proven the figures | Tiers 1 and 2 of the scorecard as a benchmark mode with history: calls read, arguments valid, tasks done. E2E in the benchmarks phase. |
| FU-237 | Keep going guided run: app-made project, stage tools, demo preset | Queued (2026-11-18), D-89. After FU-231 | The app creates and binds the Storyboard project when it takes the brief; `create` is not offered in that run. Each nudge turn gets the tools of the film's stage (Plan, Make, Sound, Finish) plus job and read tools. Demo preset: two 3 s shots, music, cap 8, budget 10 minutes; it never preselects render approval. Fixes a nudge that names `storyboard_batch`, which Chat does not offer. New `keep_going/guided.py`. Ends with three timed rehearsals: live only if all three finish in the budget, else the recording. |
| FU-238 | Keep going: time budget, wrap-up, step list, status line, resume after a stall | Queued (2026-11-18), D-89. After FU-237 | `budgetMinutes` (off by default), a wrap-up at 80 % that renders the shots that are ready (the render already leaves unfinished shots out), a new resumable stop `time_up`. Status gains steps, elapsed, budget and one sentence ("Making clip 2 of 2, about 1 min 10 s left"), shown in the chip and the thread. After `job_stalled`, Resume offers to cancel the stuck job and redo the step (today it waits on the same job again). Strings in `en`, `null` elsewhere. |
| FU-239 | Keep going preflight panel | Queued (2026-11-18), D-89. After FU-235 | Before Turn on: model loaded and its tool verdict, context size (warn under 16K), models and ffmpeg installed for the brief, memory against the largest job, the time estimate against the budget, render approval chosen. Warns; only "no model loaded" blocks. `GET /api/chat/keepgoing/{id}/preflight`; E2E in `scripts/e2e_phases/keep_going.py`. |
| FU-240 | Plan card: always show size, length, frames, steps, and a time for those settings | Queued (2026-11-18), D-91 | Today size, frames and steps show only when the model passed them, and the time is the catalogue figure "at the recommended size" (a 3 min 36 s render was shown against a 1 min 29 s figure). Price with `variant_seconds` for the settings that will run. `tools/studio_gate.plan_item`, `StudioPlanCard.tsx`. |
| FU-241 | Chat render defaults and ceiling, Raise, editable card fields, per-chat memory | Queued (2026-11-18), D-91. After FU-240 and claim C10 | Per installed model a default and a ceiling on this machine. A Chat call above the default is brought down unless the message names a size; the result and the job card say so. The card gets Raise (next sizes with times), editable size, length and steps (the decision carries validated overrides, the model is told), and "use for the rest of this chat" with a chip. External agents are unchanged. New `tools/studio_defaults.py`, `GET /api/chat/studio/defaults` with an E2E check. Stop at FU-240 if the default is under 1.3x faster than the next size. |
| FU-242 | Grammar-constrained tool calls on MLX | Parked. Revisit when the demo model is under 54 of 60 valid calls after FU-231 to FU-234, or claim C7 shows the llama.cpp path clearly better, or mlx-lm ships structured output | The MLX worker does not read `jsonSchema` today. Needs a new dependency (licence check and `THIRD_PARTY_NOTICES.md` per CLAUDE.md), a logits processor in the worker and care with the prompt cache. Cannot fix a valid but wrong choice, such as a large size. |
