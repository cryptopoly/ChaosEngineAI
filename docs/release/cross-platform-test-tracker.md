# Cross-platform test tracker (macOS and native Linux) for v0.10.0

What is still to be tested off the Windows 4090 PC before the v0.10.0 tag. One row per check: what, the exact
command, what a pass looks like, who and when, status. Rows come from
[mac-regression review](../review/mac-regression-review-2026-11-20.md), [test coverage](../review/test-coverage-2026-11-20.md),
the UI/UX reviews (`docs/review/ui-ux-v0100-a/`, `-b/`, `voice-v0100/`), FOLLOW_UPS rows that need a Mac or Linux,
[E2E_TESTING.md](../E2E_TESTING.md), [windows-handover.md](../plans/windows-handover.md) section 6 and the Build
Checklist in `CLAUDE.md`.

**Status words:** `todo` (not run anywhere off Windows), `WSL-done` (ran under WSL2 on the 4090 PC; native Linux still
needed for anything that touches the driver, the desktop shell or packaging), `passed`, `failed` (link the FU row).
Edit the row in place when you run it: date, machine, result, FU row if it fails.

**Ground rules for every row:** throwaway `HOME` for pytest and E2E (`HOME=$(mktemp -d)`); never write to the real
`~/.chaosengine`; one GPU job at a time; a gated model or a token prompt means stop and ask; never name other desktop AI
apps in anything written here.

Commands assume the repo at `~/ChaosEngineAI`, the dev venv at `.venv`, and a backend started as
`HOME=$T CHAOSENGINE_PORT=8876 .venv/bin/python -m backend_service.app` with `T=$(mktemp -d)` in another shell
(for model runs, link the model folders read-only into `$T`, or set the data and model folders in that profile).
`B=http://127.0.0.1:8876`. The CLI-driven E2E suite needs `CHAOSENGINE_REQUIRE_AUTH=0` on that backend, or a token.

## What WSL2 already covered (2026-11-12 and 2026-11-23)

| Run | Where | Result |
|---|---|---|
| Full pytest, CPU and CUDA torch, Python 3.12 | WSL2 Ubuntu 24.04, `test/rtx4090` of 2026-11-12 | 3740 passed, 14 failed, 71 skipped; Linux bugs fixed then (ffmpeg `probe_seconds`, missing `libgomp1` message, SageAttention mask fallback, `malloc_trim`, `torch.compile` blocker). `docs/progress/rtx4090-progress.md` "Linux pass of the backend on WSL2". |
| CUDA llama.cpp, TensorFold 1.2.7 Linux archive, SageAttention, `torch.compile`, E2E smoke | same | pass, numbers in the same section (llama.cpp about 12 % below Windows native; TensorFold sm_80/86 images run on sm_89). |
| vLLM 0.33 lane, TriAttention (vLLM 0.18 venv), cache matrix | WSL2 Ubuntu 24.04, 2026-11-12/13 | pass after fixes; "W4 leftovers on Linux" and "Qwen3.8-27B + DFlash2" sections. |
| WSL vLLM bridge (Windows backend driving vLLM 0.34 in WSL) | WSL2 Ubuntu-26.04, 2026-11-19/20 | load, chat, think split, GPU share, unload pass; DFlash over the bridge not run (does not fit 24 GB). |
| Full pytest on the current tip, CPU torch, Python 3.14.4 | WSL2 Ubuntu-26.04, 2026-11-23 | see "WSL2 pass 2026-11-23" at the end of this page. |

WSL2 is not native Linux: host-to-device copies run at about half of PCIe 4 x16, anything that offloads to system RAM
is slower than on Windows, there is no desktop session, no packaging and the driver is the Windows one. Every Linux row
below that touches those still needs the dual-boot Ubuntu.

## macOS (Apple Silicon, M4 Max 64 GB)

| ID | What | Command | Expected | Who / when | Status |
|---|---|---|---|---|---|
| M-01 | Python gate on the merged tree | `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/ -q` | 0 failed. The Windows host-bound tests of FU-256 (`test_licence_gate` H3 x2, `test_music_env` mirror x3, `test_time_estimate` Mac row, `test_music_worker` absolute paths) pass here. | Claude, Mac session before the merge into `release/v0.10.0` | todo |
| M-02 | Frontend gate | `npm test && npx tsc --noEmit && npm run i18n:validate` | all pass, i18n OK | Claude, same session | todo |
| M-03 | Tauri shell tests | `cargo test --manifest-path src-tauri/Cargo.toml` | pass | Claude | todo |
| M-04 | E2E smoke, then full | backend as above with `CHAOSENGINE_REQUIRE_AUTH=0`; `./scripts/e2e_test_suite.py --smoke`; then `./scripts/e2e_test_suite.py` | smoke exit 0; full: every phase pass or skip with a reason. On a Mac phase 4 "new image families" and phase 5 "LTX-2.5 on ltx-2-mlx" pass (they fail by design on Windows and Linux). | Claude, end of the Mac session | todo |
| M-05 | Review phases 211 to 216 with real MLX models | `./scripts/e2e_test_suite.py --phases 211,212,213,214,215,216` | pass; MLX throughput, perplexity, conversion to 4-bit MLX and back, Server Activity timeline | Claude | todo |
| M-06 | MLX chat engine after the Windows pass (`mlx_worker*`, `minContextTokens`) | `./scripts/e2e_test_suite.py --phases 0,1,7` with a small Qwen3 MLX model on disk | MLX checks pass; no orphan workers in phase 7 | Claude | todo |
| M-07 | MLX tool block prints allowed values and limits (FU-231 part 1, FU-437) | `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_tool_prompt_limits.py -v`; then load a small MLX model with Studio tools on and read the prompt in Server > Activity (the live phase 1 check FU-437 asks for is not written yet) | prompt lists enum values, ranges and defaults | Claude, write the phase 1 check in the same session | todo |
| M-08 | dflash-mlx at the pinned `9ca0028` | `.venv/bin/pip install "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@9ca002898b48e14c9727dec17299f497e8467870"`; `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_dflash.py -v`; `./scripts/e2e_test_suite.py --phases 1` with a DFlash target and drafter on disk | tests pass; phase 1 DFlash check passes with `dflash` in `runtimeNote` (a plain MLX fallback is a fail) | Claude | todo |
| M-09 | Cache strategies on MLX (TurboQuant, native) | `.venv/bin/python scripts/cache-strategy-matrix.py --quick` against the backend | exit 0; legacy `chaosengine` and `rotorquant` load as `turboquant` (exit 2 if not) | Claude | todo |
| M-10 | TensorFold 1.2.9 native on macOS arm64, and the Python 0.6.6 fallback for `native=False` families | `curl -X POST $B/api/setup/install-tensorfold-native`; `curl -X POST $B/api/setup/install-tensorfold`; `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_tensorfold_native.py tests/test_tensorfold_registry.py tests/test_tensorfold_routing.py tests/test_tensorfold_setup.py tests/test_tensorfold_engine_integration.py -v`; `./scripts/e2e_test_suite.py --phases 0,1` | install jobs end "done"; phase 0 registry check and phase 1 TensorFold load pass with engine `tensorfold-native` or `tensorfold` (not a silent MLX fallback); Nanbeige and Flint skip below M5 | Claude | todo |
| M-11 | MTPLX | `curl -X POST $B/api/setup/install-mtplx`; `./scripts/e2e_test_suite.py --phases 1` | `mtplx` in `runtimeNote` | Claude | todo |
| M-12 | mflux 0.24.0: Iris-3B, FLUX 3 Klein edit with reference images | `curl -X POST $B/api/setup/install-mflux`; `./scripts/e2e_test_suite.py --phases 4,9` | phase 4 generation and "new image families" pass; phase 9 reference-image round trip passes | Claude | todo |
| M-13 | ltx-2-mlx `895201d` text and image to video | `curl -X POST $B/api/setup/install-ltx2`; `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_ltx2_mlx_runtime.py -v`; `./scripts/e2e_test_suite.py --phases 5` | LTX-2.5 / 2.3 MLX clips render with sound | Claude | todo |
| M-14 | Lip sync on the Mac: `ltx-2-mlx a2v --distilled` (FU-370, FU-493 item 2), never run | `curl -X POST $B/api/voice/synthesize -H 'content-type: application/json' -d '{"text":"Hello from the lighthouse.","voice":"af_heart"}'` for a WAV, an Iris-3B still from phase 4, then `curl -X POST $B/api/video/generate -H 'content-type: application/json' -d '{"modelId":"dgrauet/ltx-2.3-mlx-q4","prompt":"a keeper speaking to camera","startImagePath":"<still.png>","audioPath":"<line.wav>","numFrames":121,"width":768,"height":512}'` | a 5.04 s clip whose sound is the line, the face moves with it; record time and peak memory, then fill the row's `lipSync` block (now `basis: untested`) and an estimate. If `audio_vae.safetensors` lacks the encoder half, add it to the download list. | Claude, M4 Max session | todo |
| M-15 | Music: ACE-Step 1.5 on MLX after the CUDA lane changes (FU-493 item 4) | `curl -X POST $B/api/music/install`; download `ACE-Step/Ace-Step1.5`; `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_music_catalog.py tests/test_music_env.py tests/test_music_worker.py tests/test_music_client.py tests/test_music_library.py tests/test_music_routes.py -v`; `./scripts/e2e_test_suite.py --phases 11` | lane `mlx`, `ACESTEP_LM_BACKEND=mlx`; a marker without `lane` reads as `mlx`; the real 20 s instrumental passes; Cancel during diffusion keeps the worker | Claude | todo |
| M-16 | **Stable Audio 3 on the Mac (owner decision D-130: SA3 should run on Mac).** Today the `mps` lane skips it (`music_sound_env._with_sa3()` is CUDA only, FU-493) and the three SA3 rows are `platforms: ["cuda"]`. First build FU-600 (install and verify-import SA3 on `mps`, add `apple-silicon` and Mac estimates to the rows), then prove it | 1. `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_music_sound.py tests/test_music_sound_1_0.py tests/test_music_sound_steps.py -v` 2. `curl -X POST $B/api/music/sound/install`, poll `curl $B/api/music/sound/install/status` until `done` 3. `cat $T/.chaosengine/music/sound-pack.json` (or the data folder in use) 4. `curl -X POST $B/api/music/download -H 'content-type: application/json' -d '{"repo":"stabilityai/stable-audio-3-small-sfx"}'` (gated: the owner's token, stop and ask if refused) 5. `curl -X POST $B/api/music/sound/generate -H 'content-type: application/json' -d '{"modelId":"stabilityai/stable-audio-3-small-sfx","caption":"door creaks open, then slams","seconds":5}'` 6. the same with `stabilityai/stable-audio-3-small-music` and `stabilityai/stable-audio-3-medium` 7. `curl -X POST $B/api/music/sound/generate ... '{"modelId":"stabilityai/stable-audio-open-small","caption":"rain on a tin roof","seconds":5}'` | install `done`; marker `schema: 2`, `lane: "mps"`, `sa3` = the pinned commit; each SA3 row writes a 5 s file of the right length on MPS (the library loads float32 off CUDA), with time and peak memory recorded for the Mac `estimates`; Stable Audio Open Small still works after the reinstall. **If the install or the verify import fails on MPS:** fix it (record the error in FU-600) and re-run; do not fall back to hiding SA3. | Claude builds FU-600 on a branch, runs it on the Mac; owner listens | todo |
| M-17 | Sounds pack reinstall after the merge (schema 1 to 2) (FU-493 item 1) | install a v0.10.0 pack (schema 1), switch to the merged tree, open Music > Sounds, then step 2 of M-16 | the UI asks for a reinstall once; afterwards Stable Audio Open Small makes a sound | Claude | todo |
| M-18 | Paused and resumed Hub download with xet (FU-275, FU-432, FU-493 item 3) | start a download of a 2 GB+ row in the UI, pause at about 30 %, quit the backend, start it, resume; `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_download_resume.py -v` | resumes from the partial file (not from zero); sha matches | Claude | todo |
| M-19 | Nemotron 3.5 Lightning GGUF with the 16K minimum context (FU-265, FU-493 item 5) | load `lmstudio-community/...Nemotron...Lightning... Q4_K_M` from Chat Models, ask one question | loads with at least 16,384 tokens and says so in the load note; memory fits | Claude | todo |
| M-20 | Voice > Models on a Mac (FU-493 item 6, review voice-v0100) | open Voice > Models and Dictation; `cd e2e && npx playwright test tests/14-voice-speak.spec.ts tests/21-kokoro-optional.spec.ts --workers=1` | collapsed family rows show the recommended row, Dictation default Parakeet, no "Mac-only" fold line; spec 14 passes (it fails on Windows by design) | Claude | todo |
| M-21 | Image catalog on a Mac: no Nunchaku rows (FU-490, FU-493 item 7) | `curl -s $B/api/images/catalog \| grep -c nunchaku` | 0; FLUX.1 Schnell and Dev listed and installed as before | Claude | todo |
| M-22 | Browser specs that only pass on a Mac (FU-256) | `cd e2e && npx playwright test tests/04-music-sounds-licence.spec.ts tests/06-busy-advice.spec.ts tests/27-benchmarks-conversion.spec.ts tests/37-chat-local-model.spec.ts --workers=1` | pass (they fail on Windows because they mock an MLX model or Mac-only text) | Claude | todo |
| M-23 | Spec 21 slow-row labels on a real Mac catalogue | `cd e2e && npx playwright test tests/21-slow-row-labels.spec.ts --workers=1` | pass: the darwin fake now has a Mac row to label (fails on Windows and Linux) | Claude | todo |
| M-24 | Voice lanes on MLX: Kokoro, Parakeet, Qwen3-TTS clone with the 1.7B | `curl -X POST $B/api/voice/install`; `./scripts/e2e_test_suite.py --phases 8` | phase 8 speak, transcribe and clone checks pass; watermark on the clone | Claude | todo |
| M-25 | Mac-only catalogue rows still run (LTX-2.5 on ltx-2-mlx, mflux rows, Kolibri 1 MLX, MLX vision) | phases 1, 4, 5 above, plus `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_mlx_models.py -v` | pass | Claude | todo |
| M-26 | Packaged app (DMG) from the merged tree | clean release venv `python3.12 -m venv .venv-release && .venv-release/bin/pip install -c release-constraints.txt mlx mlx-lm gguf fastapi psutil uvicorn "pypdf>=6.21.4" python-multipart huggingface_hub babel`, then `./build.sh` (or `npx tauri build`); `cat src-tauri/resources/embedded/runtime-*.manifest.json`; `tar -tzf src-tauri/resources/embedded/runtime-*.tar.gz \| grep backend/scripts` | manifest has `llamaServer` and `llamaServerTurbo`, `warnings` empty; the four installers (`install-mflux.sh`, `install-ltx2-mlx.sh`, `install-mtplx.sh`, `install-tensorfold.sh`) and `chaosengine-cli` are staged; the packaged backend answers `/api/health` and `--smoke` passes against it | Claude builds; owner opens the window | todo |
| M-27 | Codesign and notarisation of the DMG | `codesign --verify --deep --strict --verbose=2 /Applications/ChaosEngineAI.app`; `spctl -a -vvv -t exec /Applications/ChaosEngineAI.app`; `xcrun stapler validate ChaosEngineAI_0.10.0_aarch64.dmg` | valid on disk, "accepted, source=Notarized Developer ID", stapled; nested binaries (llama-server, sd, the runtime's Python) signed | owner (signing identity), before the tag | todo |
| M-28 | Microphone in the packaged app: voice call in Chat (R5), dictation, clone recording | `plutil -p /Applications/ChaosEngineAI.app/Contents/Info.plist \| grep -i microphone`; then in the app: Chat voice call, Voice > Dictation record, Voice > Clone record | usage string present; macOS asks once; each records and transcribes | owner (needs a person at the Mac) | todo |
| M-29 | Packaged window: Tauri dialogs (Choose file, Reveal), tray, menus, compact drawer in light theme, update prompt | open the DMG build; Image Studio > Reference image > Choose file; Library > Reveal; resize under 1180 px in light theme | dialogs open, Reveal opens Finder, drawer readable | owner | todo |
| M-30 | Weight-copy advice off Windows (FU-494) | data folder on an exFAT disk, Music load | the notice must not tell a Mac user about Windows Developer Mode (known wrong, low) | Claude when music copy is touched | todo |
| M-31 | Catalogue and upstream checks | `.venv/bin/python scripts/audit-catalog-repos.py`; `.venv/bin/python scripts/check-upstreams.py`; `.venv/bin/python scripts/audit-dependencies.py` | every repo exists; no high or critical advisory in a shipped package | Claude, before the tag | todo |
| M-32 | Python 3.14 on the Mac dev venv (FU-049, found in the WSL run) | `python3.14 -m venv /tmp/ce314 && /tmp/ce314/bin/pip install -e ".[desktop,dev]" pytest && HOME=$(mktemp -d) /tmp/ce314/bin/python -m pytest tests/ -q` | optional; the release runtime stays 3.12 | anyone, optional | todo |

## Native Linux (Ubuntu 26.04 dual boot, RTX 4090)

Record results in `docs/progress/linux-progress.md` on a `test/linux` branch (windows-handover section 6).

| ID | What | Command | Expected | Who / when | Status |
|---|---|---|---|---|---|
| L-01 | System packages and toolchains | `sudo apt-get install -y build-essential git curl libgtk-3-dev libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf libgomp1 python3-venv python3-dev`; Node 20+ (`node -v`), Rust stable (`rustc -V`), Python 3.12 for the release venv (Ubuntu 26.04 ships 3.14: use `uv python install 3.12` or a standalone build) | all present | owner (sudo), first Linux session | todo |
| L-02 | Native NVIDIA driver and toolkit | `nvidia-smi`; `nvcc --version` | the 4090 listed, driver loaded natively (not `/usr/lib/wsl`), CUDA 13.x | owner | todo |
| L-03 | Python gate | `python3.12 -m venv .venv && .venv/bin/pip install -e ".[desktop,dev,images]" && HOME=$(mktemp -d) .venv/bin/python -m pytest tests/ -q` | 0 failed except rows already known on Linux; compare with the WSL list at the end of this page | Claude | WSL-done (CPU torch, 3.14) |
| L-04 | Frontend gate | `npm ci && npm test && npx tsc --noEmit && npm run i18n:validate` | pass. Not run in WSL (the distro has no Linux Node; only the Windows `npm` was on `PATH`). | Claude | todo |
| L-05 | Tauri shell tests | `cargo test --manifest-path src-tauri/Cargo.toml` | pass (sidecar lifecycle, port wait) | Claude | todo |
| L-06 | E2E smoke and full against a native backend | backend with `CHAOSENGINE_REQUIRE_AUTH=0`; `./scripts/e2e_test_suite.py --smoke`; `./scripts/e2e_test_suite.py` | smoke exit 0; full: Mac-only rows fail as on Windows (phase 4 "new image families", phase 5 LTX-2.5 MLX), everything else pass or skip | Claude | WSL-done (smoke, 2026-11-12) |
| L-07 | CUDA llama.cpp `b12441` through the app | `curl -X POST $B/api/setup/install-llama-cuda`; load Qwen3-0.6B Q8_0, 3 x 300 tokens; `./scripts/e2e_test_suite.py --phases 1` | archive and `cudart-` sha256 verified; GGUF on the GPU; tok/s near Windows native (549 to 563), WSL gave 480; without `libgomp1` the message names it | Claude | WSL-done (b12292) |
| L-08 | TensorFold 1.2.9 Linux x86_64 archive | `curl -X POST $B/api/setup/install-tensorfold-native`; `~/.chaosengine/bin/tensorfold capabilities --json`; load Qwen3.5-2B 4-bit with `backend: tensorfold`; `./scripts/e2e_test_suite.py --phases 0,1` | `nvidia-sm89`, back end cuda; about 450 tok/s; engine `tensorfold-native` | Claude | WSL-done (1.2.7) |
| L-09 | vLLM native lane (not the WSL bridge) | Setup > Boost Pack > DFlash (CUDA), or `.venv/bin/pip install "vllm>=0.28.0"` into the extras folder as the app does; load `Qwen/Qwen3-0.6B` with backend `vllm`; chat; unload | loads from the library folder (no second download), think split, VRAM back after unload. Check FU-532 here too: `--trust-remote-code` only for rows that need it. Check the native lane passes a GPU share like the bridge does | Claude | WSL-done (inside WSL, 0.33) |
| L-10 | TriAttention on its own vLLM 0.18 venv (FU-171) | `HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_cache_strategies.py -v`; `.venv/bin/python scripts/cache-strategy-matrix.py` | TriAttention cell compacts on 0.18 | Claude | WSL-done |
| L-11 | stable-diffusion.cpp `532d2c1` CUDA build | `./scripts/build-sdcpp.sh` (CUDA); `./scripts/e2e_test_suite.py --phases 4` with the `leejet/Qwen-Image-2.1-GGUF-q4k` row | `sd` builds, device label `cuda`, one image | Claude | todo |
| L-12 | GPU bundle and CUDA torch | `curl -X POST $B/api/setup/install-gpu-bundle`; `curl $B/api/setup/gpu-bundle-status` | CUDA torch from the app's index order (cu134 first); `bitsandbytes` and `ftfy` installed; Wan TI2V and LTX-2.3 4-bit render (`--phases 5`) | Claude | todo |
| L-13 | Music and sounds on CUDA on Linux (FU-139 says Linux untested) | `curl -X POST $B/api/music/install`; `curl -X POST $B/api/music/sound/install`; `./scripts/e2e_test_suite.py --phases 11` | lane `cuda`; real 20 s ACE-Step track; "a real 5 s Stable Audio 3 sound on CUDA" passes | Claude | todo |
| L-14 | Voice add-ons: speech GPU and stt-gpu | `curl -X POST $B/api/voice/install` (with the GPU add-ons from Voice > Models); `./scripts/e2e_test_suite.py --phases 8` | "stt on the NVIDIA GPU", Chatterbox / VoxCPM2 / Orpheus checks pass; word timings in English (FU-311 lists Linux as not run) | Claude | todo |
| L-15 | Nunchaku INT4 FLUX.1-schnell | install the Linux wheel matching the torch build; `./scripts/e2e_test_suite.py --phases 4` with the pack on disk | Nunchaku runs (5.9 s at 1024 px on Windows with offload) | Claude | todo |
| L-16 | SageAttention 2 and `torch.compile` natively | `CHAOSENGINE_SPEEDUPS=compile` on the backend; Wan TI2V 832x480 49 frames twice | warm speed-up as in WSL (1.49x sage, 1.67x compile); without gcc or `Python.h` the blocker names `build-essential` / `python3-dev` | Claude | WSL-done |
| L-17 | AppImage and deb from a clean release venv | `.venv-release` from Python 3.12 with `-c release-constraints.txt` and the `release.yml` package list; `npx tauri build --bundles appimage,deb`; `cat src-tauri/resources/embedded/runtime-*.manifest.json`; `tar -tzf src-tauri/resources/embedded/runtime-*.tar.gz \| grep backend/scripts` | both bundles built; manifest `llamaServer` set, `warnings` empty; installers and `chaosengine-cli` staged | Claude builds | todo |
| L-18 | Install and start: deb and AppImage | `sudo apt install ./src-tauri/target/release/bundle/deb/*.deb`; `chmod +x *.AppImage && ./ChaosEngineAI_*.AppImage` | window opens; the sidecar answers on its port; GPU detected; tray icon | owner (sudo, desktop session) | todo |
| L-19 | Sidecar lifecycle: quit and kill | quit from the tray; separately `kill -9 <app pid>` while a model is loaded; `pgrep -af "llama-server\|mlx_worker\|backend_service"` | no backend or worker left after quit; after a kill, workers end too (Windows uses a Job Object; check what Linux does) | Claude | todo |
| L-20 | Wayland and X11 webview | log in once with "Ubuntu" (Wayland) and once with "Ubuntu on Xorg"; open the app | renders, no blank window; if blank on Wayland, try `WEBKIT_DISABLE_DMABUF_RENDERER=1` and record it | owner | todo |
| L-21 | AppImage sandboxing, `/tmp` noexec, exec bits | `mount \| grep ' /tmp '`; run the AppImage with `TMPDIR` on a noexec mount; check `ls -l` of the extracted `llama-server`, `sd`, `tensorfold` | runtime extracts and runs; staged binaries keep `+x` | Claude | todo |
| L-22 | Paths and permissions | `stat -c '%a %n' ~/.chaosengine/agent_access.json ~/.chaosengine/settings.json`; a model folder on the NTFS partition (dual boot, `ntfs3` mount) linked read-only; symlink and hard-link tests (`HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_model_import.py tests/test_music_weight_copy.py -v`) | token files mode 600; reading models from the NTFS mount works; link fallbacks behave (Linux on NTFS may refuse hard links) | Claude | todo |
| L-23 | File-descriptor limit and process scan | `ulimit -n`; run phases 4, 5, 8 back to back; Dashboard running-process list with a second user's Python running | no "too many open files"; other users' processes not listed or stoppable | Claude | todo |
| L-24 | WSL against native numbers | rerun L-07, L-08, L-16 and a Klein / Wan image and video timing natively | write the native numbers beside the WSL ones; the catalogue estimates use Windows-native figures, so check Linux is within 10 % | Claude | todo |
| L-25 | RAM given back after unload (glibc, `malloc_trim`) | load Klein then Wan, unload, `ps -o rss -p <backend pid>` | under 2 GB left with nothing loaded | Claude | WSL-done |
| L-26 | Microphone in the Linux webview | Chat voice call, Voice > Dictation record | PipeWire / PulseAudio input works in the webview | owner | todo |
| L-27 | Linux ARM64 (FU-137, DGX Spark class) | release CI job on `ubuntu-22.04-arm` | n/a on the owner's hardware; stays "experimental, untested" in the release notes | nobody planned | todo |

## WSL2 pass 2026-11-23 (current tip, Ubuntu-26.04)

Clone of `test/rtx4090` at `c28b0994` on the distro's ext4 disk (`~/ce-wsl`), a venv on Python 3.14.4 (the distro's
own), CPU torch 2.15, the other test packages unpinned at their newest, throwaway `HOME` per run.

| Run | Result |
|---|---|
| Full `pytest tests/`, newest packages | 6591 tests: 6494 passed, **18 failed**, 79 skipped (307 s) |
| Same after the fixes below, FastAPI pinned to the release's 0.135.3 | 6592 tests: 6503 passed, **9 failed**, 80 skipped (382 s) |
| `CHAOSENGINE_TEST_WSL=1` bridge tests from Windows (`test_vllm_wsl_engine/_prereqs/_install`) | 55 passed, live injection and PATH checks included |
| vLLM 0.34 smoke in the distro (Qwen3-0.6B read from `E:\AI Models` over `/mnt/e`, `--gpu-memory-utilization 0.30`, GPU lock held) | ready in 50 s, 8.6 GB on the card, "2 + 2 = 4.", server gone and VRAM back to 1.2 GB after stop |
| `npm test`, `tsc` | **not run**: the distro has no Linux Node (only the Windows `npm` through interop). L-04. |

The 18 failures, classified:

| Tests | Cause | Linux-specific? | Done |
|---|---|---|---|
| `test_agent_guide` 1, `test_mcp_contract` 7 | `make_lipsync_shot` and `make_sound` drifted from the contract tests | no, fail on Windows too | FU-601 |
| `test_size_limits::test_every_family_has_a_row_in_the_table` | no `moss-image` size row | no, fails on Windows too | FU-602 |
| `test_size_limits::test_a_big_machine_runs_it` | test capped by the host's real 31 GB total | no, any host under about 46 GB | test fixed (pins the total) |
| `test_music_env` InstallJobTests 6 | tests ran into the real "ACE-Step needs Python 3.11 or 3.12" check on 3.14 | no, any 3.13+ interpreter | tests fixed (patch `_python_supported`, as `test_music_sound` does) |
| `test_model_events` stream 2 | FastAPI 0.165 keeps included routers as `_IncludedRouter`; the test looks the route up in `app.routes` | no, dependency version; pass on 0.135.3 | FU-603 |

**Linux-specific code bugs found by pytest: none.** One WSL bug fixed from a live probe: the bridge's PATH entry
was a literal `~/.chaosengine/vllm-venv/bin`, which bash's own lookup finds but a child process's `execvp` does not
(FU-533, now `$HOME`, live test added). The earlier Linux-only bugs of 2026-11-12 stay fixed (their tests pass).

Left in the distro: `~/ce-wsl` (clone), `~/ce-venv` (about 3 GB), `~/ce-results`. Delete with
`rm -rf ~/ce-wsl ~/ce-venv ~/ce-results` when no longer needed.
