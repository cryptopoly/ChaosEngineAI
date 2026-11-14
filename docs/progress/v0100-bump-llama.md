# v0.10.0: bump llama.cpp b12387 and Playwright 1.65.1

Branch `v0100/bump-llama`, commits a9139306 (llama.cpp) and ba6aa7bc (Playwright).

## Done
- llama.cpp b12317 -> b12387 (newest release with macos-arm64 and win-cuda assets). Asset names unchanged. New Windows CUDA digests in `_PINNED_SHA256` taken from the release asset `digest` fields (12.4 x64, 13.5 x64, 13.5 arm64); cudart zips are tagless and unchanged. Also bumped `linux-gpu-check.sh` and three tests that name the tag. CLAUDE.md row updated.
- Playwright 1.65.0 -> 1.65.1 exact in `e2e/package.json` and its lockfile; chromium headless shell installed. CLAUDE.md row updated. `docs/research/playwright-security.md` left alone (dated research record naming 1.65.0 as latest on 2026-11-07).
- pytest tests/test_llama_cuda.py tests/test_llama_cuda_windows.py tests/test_linux_arm64_build.py: pass (some skips).

## Mac checks (after the gate was DONE)
- tests/test_inference.py: pass (70 tests). Run on the code, not on a new binary.
- Playwright 1.65.1, `npx playwright test 01-smoke --workers=1`: 3 passed.

## llama-server b12387 installed on the Mac (owner approved the download)
- Prebuilt `llama-b12387-bin-macos-arm64.tar.gz` (official release), sha256 matched the release digest (a654e602...).
- The new build is dylib-linked (`@loader_path`), the old one was a single file. Installed `llama-server` plus the `libggml*`/`libllama*`/`libmtmd*` dylibs into `~/.chaosengine/bin`. Old binary kept as `llama-server.bak-20261114`; no dylibs existed there before. To roll back: copy the .bak over `llama-server` and delete the `*.dylib` files in that folder. `llama-server-turbo` still runs.
- Direct run, Llama-3.2-1B Q4_K_M (Metal, -ngl 99): `/v1/chat/completions` returned a reply, version `build 12387 (a956a852c)`. Not run through the app's engine path.

## Not done
- Windows CUDA install on b12387 (4090 item).
- Chat through the app's llama.cpp engine path (direct server only).
