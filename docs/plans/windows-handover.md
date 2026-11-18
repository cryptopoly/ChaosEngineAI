# Windows handover (from the Mac session of 2026-11-18)

For a fresh Claude Code session on the Windows PC (RTX 4090, 1 TB drive; Ubuntu 26 is on the same machine as a dual boot and comes after Windows). Read this file top to bottom, then `CLAUDE.md`, `docs/hub/status.md` and `docs/hub/decisions.md`. It says exactly where the Mac session stopped.

## The first three things to do

1. **Ask the owner whether `review/integrate` has been merged into `release/v0.10.0`.** The coordinator merges it on the owner's yes; this session never merges into `release/v0.10.0`, `staging` or `main`, never tags and never force-pushes. If it is merged, recut `test/rtx4090` from the new `release/v0.10.0` tip. If not, cut a branch `test/rtx4090-integrate` from `review/integrate` and work there.
2. **Set up a throwaway data folder before running anything** (the test suite once wrote 505 junk rows into a real prompt library): in PowerShell `$t = Join-Path $env:TEMP ("ce-" + [guid]::NewGuid()); New-Item -ItemType Directory $t | Out-Null; $env:HOME = $t; $env:USERPROFILE = $t` for every `pytest` and `e2e_test_suite.py` run. The Mac's model library paths (`/Users/dan/...`) do not exist here; link or point to the PC's own Hugging Face cache read-only, never a datasets folder.
3. **Run the gate in section 2 once on the cut branch**, record PASS / FAIL / SKIP with one line of evidence per item in `docs/progress/rtx4090-progress.md`, commit and push the branch, then work down section 4.

## State of play (one screen)

- **v0.10.0** (plan `docs/plans/v0.10.0-all-singing.md`) holds: the Storyboard and project workspace with draft and final passes, Chat with Studio tools and a plan card (D-60, D-85), Keep going (D-86), the eight closing improvements (D-84), Music on Apple Silicon only (D-80), the voice makeover and cloning form, stronger field borders on by default (D-82), MiniMax H3 built but hidden behind `licence_gate.H3_RELEASED = False` (FU-166), pins llama.cpp `b12441`, TensorFold 1.2.9, sd.cpp `532d2c1` (bumped 2026-11-16). A notarised Mac DMG was built and checked from `500590a6`.
- **The overnight review (2026-11-18, FU-220)** audited seven features that had no end-to-end check. Verdicts (D-87): Benchmarks and Conversion keep (both fixed; Conversion failed on every model in the Hugging Face cache before), Prompts keep for now and later merge into Chat (`docs/plans/review-c-prompts-into-chat.md`), **Fine Tuning and Plugins hidden** behind Settings > General > Appearance > "Show experimental tabs" (off by default; the hide is commit `41f5a92a`, which reverts alone), Logs folded into the redesigned Server page (D-88: Overview and Activity, per-model timeline from structured model events). Plus: API keys masked on screen until Reveal (Copy still copies the real key), Dashboard copy says "70B class at 4-bit", GGUF or MLX is read from the files and not from "gguf" in a name, a critical catalogue-match bug fixed (a small Qwen loaded as another model, D-13 in `docs/progress/review-d.md`), first-run, small-window, backend-offline and dialog-keyboard fixes. Report: [overnight-review-report.md](../progress/overnight-review-report.md); integration log with evidence: [review-integrate.md](../progress/review-integrate.md).
- **Merged:** every review stream is merged into `review/integrate` (table below). **Not merged:** `review/integrate` into `release/v0.10.0` (owner's yes first), and nothing from the review is on `test/rtx4090` yet (that branch is at `878901fa`, 2026-11-14, behind the release branch).
- **Also new and open (not built):** the Keep going and tools v2 plan, [keep-going-and-tools-v2.md](keep-going-and-tools-v2.md), decisions D-89 to D-92 (D-92's two open points answered by the owner), FU-230 to FU-246.

## Branches and worktrees

All worktrees named here live on the Mac only. On the PC only the pushed branches matter.

| Branch | Tip | What | Merged into | Delete when |
|---|---|---|---|---|
| `review/a` | `02d41b40` | Benchmarks and Conversion audit and fixes | `review/integrate` | after `review/integrate` is in `release/v0.10.0` and pushed |
| `review/a2` | `14c2590d` | Benchmarks and Conversion follow-up: real cancel (200 with result null; a second run answers 409), real progress, no invented Quality or Compression figures (runs carry no `quality`; a conversion reports `diskRatio`, `bitsPerWeight` and a measured perplexity delta), next-token accuracy scoring with `taskScoringMethod` saved per run | `review/integrate` | same |
| `review/b` | `c4982f56` | Fine Tuning audit (not built; crash and honesty fixes; start route answers 501) | `review/integrate` | same |
| `review/c` | `169be582` | Prompts and Plugins audit and fixes | `review/integrate` | same |
| `review/d` | `31e9d194` | Whole-app sweep (D-1 to D-18 fixed, L-1 to L-21 left) | `review/integrate` | same |
| `review/e` | `1410fa08` | Server page Overview and Activity, model event log | `review/integrate` | same |
| `review/f` | `072d68b7` | Podcast demo path (`docs/plans/demo-path.md`) and its fixes | `review/integrate` | same |
| `review/gguf` | `f7283367` | GGUF or MLX from the files | `review/integrate` | same |
| `review/plan` | `1ce012ab` | Keep going and tools v2 plan (docs only) | `review/integrate` | same |
| `review/routing` | `1ac81af9` | Backend takes the repo from the model's own folder and ignores a disagreeing `canonicalRepo` (L-16, FU-248); the CLI prints `cacheBits`. Behaviour change: a local copy matched to another uploader's catalogue row now loads under its own repo, so a drafter pinned (`PINNED_DRAFTERS`) to the other uploader's repo id is not used for it | `review/integrate` (`599802fe`) | same |
| `review/i18nfix` | I18NFIX_TIP | Diagnostics strings translated (L-2, FU-249), language banner wording (L-1, FU-250) | I18NFIX_MERGED | same |
| `review/integrate` | see `git log` | Everything above, plus the experimental-tabs switch, key masking, Dashboard copy | **not merged**: the owner decides | after it is merged into `release/v0.10.0`, the merge hash is confirmed there and pushed |
| `test/rtx4090` | `878901fa` | The Windows pass branch | n/a | never; recut it from `release/v0.10.0` |

Deleting a worktree or branch (local and remote) happens only once the merge hash is confirmed on the target and pushed, and never for a branch that is unmerged, in use or has uncommitted tracked changes.

## 1. Setup and build on Windows (PowerShell)

The commands in [rtx4090-test.md](rtx4090-test.md), "Start here", still hold:

```powershell
git fetch
git switch -c test/rtx4090-integrate origin/review/integrate   # or recut test/rtx4090 from release/v0.10.0 once merged
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -c release-constraints.txt -e .   # plus the CUDA extras the README names
npm install
npm run stage:runtime
npm run tauri:dev          # the desktop app; for sections 2 to 4 the backend alone is enough:
.venv\Scripts\python -m backend_service.app   # then http://127.0.0.1:8876/api/health
cd e2e; npm ci; npx playwright install chromium --only-shell
nvidia-smi                 # GPU, driver, VRAM; peak VRAM: nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

Installer: build it the way `release.yml` does, from a clean Python 3.12 venv with `release-constraints.txt` (`./build.sh` now makes `.venv-release` itself), then `npx tauri build --bundles nsis`. `scripts/stage-runtime.mjs` stops at `NSIS_ARCHIVE_LIMIT_BYTES` (1.9 GB): NSIS cannot map an archive past 2 GB, and staging the dev `.venv` once produced 3.04 GB. A good runtime archive is about 132 MB, the installer about 137 MB (last built 2026-11-11 and 2026-11-13).

**Rows of [rtx4090-test.md](rtx4090-test.md) that the review changes** (rows 1 to 37 are the v0.10.0 Windows pass, 38 to 44 the FU-178 and pin rows):

| Row | Change |
|---|---|
| 1 | Still the gate, now larger: `e2e_test_suite.py` has phases 211 (Benchmarks and Conversion), 212 (Fine Tuning), 213 (Prompts and Plugins), 214 (Server Activity), 215 (GGUF routing), 216 (routing from the model's path); Playwright has specs 27 to 39 |
| 3 | **Superseded by row 44**: the CUDA llama-server pin is now `b12441` (CUDA 12.4 and 13.5 archives), not `b12317` |
| 12 | Unchanged; the new "Show experimental tabs" sits in Settings > General > Appearance next to it |
| 44 | Unchanged and still open: new pins on CUDA (llama.cpp `b12441`, TensorFold 1.2.9, sd.cpp `532d2c1`) |

**Rows to add** (number them 45 and up in the plan's table when you start):

| New row | Test | Expected |
|---|---|---|
| 45 | Server page on Windows: Overview, Activity, a real GGUF load on CUDA and an answer | Timeline shows requested, loading, loaded (seconds, memory), first token, generation (tok/s), unloaded; the `loaded` detail carries the full `llama-server` command line with Windows paths; `Logs` in an old saved state opens Server > Activity |
| 46 | Show experimental tabs | Off: Tools shows Prompts only (Conversion is Mac-only, so it is hidden on Windows); on: Fine-Tuning and Plugins appear; the choice survives a restart of the desktop app |
| 47 | API key masking in the desktop webview | Server key field, Connect samples, Test commands, Settings > Integrations and Agent access show dots and the last 4 characters until Reveal; Copy puts the real key on the Windows clipboard |
| 48 | GGUF routing from files on NTFS | An MLX-style folder whose name contains "gguf" is not sent to llama.cpp; a `.GGUF` file in upper case loads (NTFS is case-insensitive); a folder with only `config.json` is refused naming the missing weights |
| 49 | `canonicalRepo` (`review/routing`, phase 216) | A load with a `canonicalRepo` that disagrees with the folder loads the folder's own model, also through the CLI; a pinned drafter of another uploader's repo is not used for a local copy; full `e2e_test_suite.py` passes (inference routing changed) |
| 50 | Benchmarks on CUDA (a2) | Throughput on a GGUF model; Cancel mid-run answers with no saved run and the model unloaded or kept as chosen; the progress dialog shows real phases; perplexity and accuracy say they need an MLX model |
| 51 | Smallest window and backend-offline (review D) | At 1100 x 760 no panel is cut; stop the backend from Task Manager: the offline notice shows at the top, then clears when the backend is back |

## 2. Test gates on Windows (one suite at a time)

Always with the throwaway `HOME` and `USERPROFILE` from "first three things". Record each in `docs/progress/rtx4090-progress.md`.

```powershell
.venv\Scripts\python -m pytest tests/ -q        # Mac reference on review/integrate: see review-integrate.md
npx vitest run; npx tsc --noEmit; npm run i18n:validate
.venv\Scripts\python scripts\e2e_test_suite.py --smoke
.venv\Scripts\python scripts\e2e_test_suite.py  # full, real models, phases 0 to 20 and 211 to 216
cd e2e; $env:E2E_PYTHON = (Resolve-Path ..\.venv\Scripts\python.exe); npx playwright test --workers=1
.venv\Scripts\python scripts\check-upstreams.py
.venv\Scripts\python scripts\audit-dependencies.py
cargo test --manifest-path src-tauri/Cargo.toml
```

`scripts/pre-build-check.sh` needs Git Bash; run it with the throwaway `HOME` too, and only with no backend on port 8876 (its step 10 runs the smoke suite against whatever answers there).

## 3. Windows risks the Mac could not test

- **Paths:** long paths (over 260 characters) in the Hugging Face cache and in project folders; non-ASCII project names (the sample film takes the user's language); backslashes in paths shown in the Server event detail and in copied commands.
- **NTFS:** case-insensitive names (a `.GGUF` file, two projects differing only in case); file locks when deleting a model, a project or a Clean-up item while a worker holds it open; antivirus scanning slowing the first model load.
- **Process kill:** killing the backend from Task Manager mid-render (pause and resume across a restart, rtx4090 row 19); the Dashboard's Stop on processes it did not start (review D L-11, FU-247: other instances' workers are listed with Stop buttons); a worker that calls `os._exit` (Errno 22 fix from 2026-11-11).
- **CUDA installs of the new pins:** llama.cpp `b12441` CUDA 12.4 and 13.5 archives (digests checked against release metadata only), TensorFold 1.2.9 native, sd.cpp `532d2c1`; record the archive names, sizes and the first run of each.
- **Packaging:** signing and the NSIS installer (`npx tauri build --bundles nsis`), the 1.9 GB archive guard, runtime extraction under `%TEMP%` (old extractions are never cleaned), uninstall leaving no process behind, the desktop webview (not Chromium) for keyboard, focus and the new dialogs.
- **Review changes never run on Windows:** the model event log (`backend_service/model_events.py`), conversion cancel (Mac-only feature, but its route must refuse cleanly), the process-scan guard, the composer's "loading" placeholder.

## 4. Open product work, in priority order

1. **Owner decisions waiting** (D-87): remove Plugins after v0.10.0 or plan a loader (FU-225); move Prompts into Chat or only show the applied prompt per thread (FU-224, proposal `docs/plans/review-c-prompts-into-chat.md`).
2. **Keep going and tools v2** ([plan](keep-going-and-tools-v2.md), D-89 to D-92, FU-230 to FU-246): start with the scorecard (FU-230), then the honest staged tool prompt (FU-231) and its cache (FU-232), the plan card showing the real settings (FU-240) and the quality tiers (FU-241). D-92: the "start on Draft and stop for picks" behaviour for films over 8 shots is an option, off by default; the thresholds (5 minutes, 5 GB, 10 %) are the starting values.
3. **Review leftovers:** FU-249 and FU-250 (if `review/i18nfix` has not closed them), FU-251 (copy, i18n and accessibility leftovers from review D), FU-221 and FU-222 (Fine Tuning: training worker and applying an adapter), FU-223, FU-226, FU-227 (Logs code to delete once the owner agrees), FU-228, FU-229, FU-247.
4. **Benchmarks and Conversion:** verdict keep; the a2 follow-up is merged. Conversion is Apple Silicon only.
5. **The podcast demo path:** `docs/plans/demo-path.md` (rehearsed four times on the Mac; on-air risks ranked there).

## 5. Owner-only

- Legal read of the MiniMax H3 licence, then flip `licence_gate.H3_RELEASED` (one line and a test). Until then H3 stays hidden.
- Tag and publish: ask first, every time.
- The R5 voice call in Chat with a microphone; listen to the voice clones and the music.
- **Turn LAN access off for any demo on someone else's Wi-Fi**: the owner's real settings had LAN access on and "Require API token" off.
- The 505 junk "Phase 0 test template" rows in the Mac's `~/.chaosengine/prompt_templates.json`: the clean-up command (keeps a backup) is in `docs/progress/review-c.md` and needs the owner's yes.
- The notarised Mac DMG exists and was checked; a new one is needed only after `review/integrate` is merged.

## 6. Linux (Ubuntu 26, after Windows)

The same list, adapted:

- Packages: `sudo apt-get install -y libgtk-3-dev libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf` (from `release.yml`), plus build-essential, git, Node 20 or newer, Rust stable, Python 3.11 for the dev venv and 3.12 for the release venv.
- CUDA: the native driver and toolkit on Ubuntu 26 (not the WSL setup); `nvidia-smi`, `nvcc --version`; then the app's CUDA llama-server lane at `b12441`, TensorFold 1.2.9 and sd.cpp `532d2c1`.
- Build: `npx tauri build --bundles appimage,deb` the way `release.yml` does (clean release venv, `release-constraints.txt`).
- History and gotchas from the earlier WSL2 run (Ubuntu 24.04): [WSL_CUDA_TESTING.md](../WSL_CUDA_TESTING.md) (CRLF shebangs, `ninja` on `PATH`, driver on the host). Native Ubuntu should avoid most of them; record which still apply.
- Gates: section 2 with `.venv/bin/python` and `HOME=$(mktemp -d)`; record in a `docs/progress/linux-progress.md` on a `test/linux` branch.
- Linux-only risks: Wayland versus X11 webview, AppImage sandboxing and `/tmp` noexec, file-descriptor limits with many workers, the process scan reading other users' processes.

## 7. Ground rules for that session

From [rtx4090-test.md](rtx4090-test.md), plus the review's:

- Work on the test branch only; push it after each section; never push to `staging`, `main` or `release/v0.10.0`; never force-push; never tag.
- Read `CLAUDE.md`, `docs/hub/status.md`, `docs/hub/decisions.md` first. Never name other desktop AI apps as inspiration or competitors in code, docs or commits (compatibility targets in the Connect snippets are allowed, D-93).
- One test suite at a time; Playwright with `--workers=1`.
- Throwaway `HOME` and `USERPROFILE` for every pytest and E2E run; never let a test write to the real data folder.
- The Mac's model library paths do not exist on Windows. Point the app at the PC's own data folder (Settings > Storage > Data directory) and model folders; link model folders read-only into a throwaway home, never dataset folders.
- Never print or commit a Hugging Face token; a gated model means stop and ask the owner.
- Never kill a process you did not start (other instances' workers show in the Dashboard with Stop buttons).
- No model deletes, no downloads over 2 GB without a line in the progress file.
- PASS / FAIL / SKIP with one line of evidence, the exact command and the log tail for a FAIL.
- Fix small, clearly Windows-only bugs on the test branch with a test; list anything bigger in the progress file.
- Log every question or decision on `docs/hub/decisions.md` the same turn, add FU rows in `FOLLOW_UPS.md` for deferred work, and rebuild the hub after each batch.

## 8. Rebuild the hub

```powershell
python tools/hub_tracker.py; python tools/build_hub.py   # expect "0 warnings, 0 errors"
.venv\Scripts\python -m pytest tests/test_hub_tracker.py -q
```

Then open `docs/site/index.html`.
