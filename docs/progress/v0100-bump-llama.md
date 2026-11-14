# v0.10.0: bump llama.cpp b12387 and Playwright 1.65.1

Branch `v0100/bump-llama`, commits a9139306 (llama.cpp) and ba6aa7bc (Playwright).

## Done
- llama.cpp b12317 -> b12387 (newest release with macos-arm64 and win-cuda assets). Asset names unchanged. New Windows CUDA digests in `_PINNED_SHA256` taken from the release asset `digest` fields (12.4 x64, 13.5 x64, 13.5 arm64); cudart zips are tagless and unchanged. Also bumped `linux-gpu-check.sh` and three tests that name the tag. CLAUDE.md row updated.
- Playwright 1.65.0 -> 1.65.1 exact in `e2e/package.json` and its lockfile; chromium headless shell installed. CLAUDE.md row updated. `docs/research/playwright-security.md` left alone (dated research record naming 1.65.0 as latest on 2026-11-07).
- pytest tests/test_llama_cuda.py tests/test_llama_cuda_windows.py tests/test_linux_arm64_build.py: pass (some skips).

## Mac checks (after the gate was DONE)
- tests/test_inference.py: pass (70 tests). Run on the code, not on a new binary.
- Playwright 1.65.1, `npx playwright test 01-smoke --workers=1`: 3 passed.

## Not done
- llama-server b12387 install and GGUF chat. `scripts/update-llama-cpp.sh` only builds in a sibling `../llama.cpp` checkout (none exists on this Mac) and never copies into `~/.chaosengine/bin`, so it needs a git clone of ggml-org/llama.cpp (or the prebuilt macos-arm64 archive, sha256 a654e6020f85d71274f1cf29460caec6537db31decc95a1a9e2f91ce05e648d7) plus a manual copy. Either is a download that needs the owner's yes. Installed binary is untouched (May build 6049906), so there was nothing to back up or restore.
- Windows CUDA install on b12387 (4090 item).
