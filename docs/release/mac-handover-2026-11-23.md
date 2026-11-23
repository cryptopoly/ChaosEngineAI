# Handover to the Mac session, v0.10.0 (2026-11-23)

Branch `test/rtx4090` (tip after this note), pushed. **Not merged into `release/v0.10.0`**: that waits for the owner.

## Proven on Windows (RTX 4090) and WSL2

| Gate | Result |
|---|---|
| Full pytest, WSL2 Ubuntu-26.04, Python 3.14.4 | 6628 passed, 0 failed, 80 skipped (`84315da2`) |
| Smoke E2E | 7/7 |
| Full E2E phases 0-22, 211-216 (CUDA 4090, dev backend) | every phase passes or skips with a reason; 216 skips (needs MLX) |
| Phase 22 lip sync with the add-on | pass, 2-take shot keeps the better take |
| Playwright `--workers=1` | 250 pass, 14 fail: the known Apple-Silicon mock specs (FU-256: spec 27 x10, spec 06 x3, spec 21 x1) |
| `audit-dependencies.py` | no blocking advisory |
| Windows NSIS bundle with CUDA `sd.exe` (FU-399) | built, 582 MiB, `sd.exe` made a real image |
| Nunchaku INT4 FLUX.1-schnell (D-133, scratch wheel) | 1024 px, 4 steps, 4.8 s generation, 17.0 GB peak with offload, image checked |

## Not proven (needs the Mac, native Linux, or the owner)

- MLX, mflux, ltx-2-mlx, Inkling 2 Mini on the M4 Max (FU-541), TensorFold 1.3.4 rows (FU-561), FU-564, Kolibri/LFM3 MLX.
- Mac rows `M-xx` in [cross-platform-test-tracker.md](cross-platform-test-tracker.md), including **M-16 Stable Audio 3 on the Mac** (D-130, FU-600): keep SA3 on the Mac lane, run a real install and generation, fix what fails.
- `FU-514`: re-freeze `release-constraints.txt` on the Mac after the dependency changes.
- The packaged app (installer, window, tray), microphone voice call, listening to voices and music, the new lip-sync UI: owner hands-on.
- Not run in the final gate: vitest, `tsc`, `npm run i18n:validate`, `cargo test` (run them on the tip before the tag).
- GPU speech/STT add-ons, music and sounds packs, voice cloning, LTX-2.5 (gated), TensorFold 1.x on Windows (no archive).

## Open code items that could matter for the release

- **FU-610** library ignores a moved Hugging Face cache, **FU-611** drafter-only repo listed as a model. Product bugs found by the final gate; not fixed.
- FU-612 (sd.cpp missing-binary copy), FU-613 (llama.cpp b12651 out), FU-563 (re-measure sd.cpp at next master, D-131: wait), FU-605, FU-606, FU-421 (native-speaker review of translations).
- One flaky test: `test_music_routes ... unload_while_a_track_is_made` (timing; passes alone).
- `test_html_challenge_persists_manifest_and_html_files` fails on Windows and failed the same way on the base commit.

## Owner-held decisions

D-99 (H3 flag stays off until the lawyer replies), D-63, D-29, D-03 (hold), D-16 (awaiting testers). LipDub gate on Hugging Face still to accept (FU-385).

## For the Mac session to do

1. Pull `test/rtx4090`, run the Mac tracker rows, fix and record.
2. Hub removal from the repo (D-17: skip the history rewrite) and the privacy clean-up: lawyer memo, `.gitignore` patterns, the 540 binaries, path and LAN-IP tidy. The private ProjectHub copy is current to `d9fcfc65`; the repo's `docs/hub` is still live and untouched.
3. Merge into `release/v0.10.0` only when the owner says Windows and Mac are done.
