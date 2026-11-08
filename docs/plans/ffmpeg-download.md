# ffmpeg on demand: one-click download, SHA-256 checked (D-35, D-49)

Status: plan only, 2026-11-07, branch `research/ffmpeg-lgpl` (from `feat/home-studio`). No code yet.
Update 2026-11-08 (D-49): the lighter path was chosen, so there is no own macOS build and no hosting; the macOS rows and the CI job (step 5) below are not planned. The shipped scope is in `docs/hub/progress/ffmpeg-light.md`.

## Goal

Storyboard and Music export use an ffmpeg build that the user fetches with one click. The file is pinned per
platform, checked by SHA-256, installed atomically, and works offline afterwards. The app never needs libx264,
except inside the ltx-2-mlx venv, which keeps its own PyPI copy.

## Pin table (one Python dict, `backend_service/ffmpeg_fetch/pins.py`)

| Key | Source | Asset | Member to extract |
|---|---|---|---|
| darwin-arm64 | our release `ffmpeg-lgpl-n9.0.2-r1` | `ffmpeg-n9.0.2-macos-arm64-lgpl.tar.xz` | `bin/ffmpeg`, `bin/ffprobe` |
| windows-x86_64 | BtbN month-end tag | `ffmpeg-n9.0.2-…-win64-lgpl-9.0.zip` | `*/bin/ffmpeg.exe`, `ffprobe.exe` |
| windows-arm64 | BtbN month-end tag | `…-winarm64-lgpl-9.0.zip` | same |
| linux-x86_64 | BtbN month-end tag | `…-linux64-lgpl-9.0.tar.xz` | `*/bin/ffmpeg`, `ffprobe` |
| linux-aarch64 | BtbN month-end tag | `…-linuxarm64-lgpl-9.0.tar.xz` | same |

Each row: `url`, `sha256` (copied once from `checksums.sha256` / our `SHA256SUMS` when pinning, never fetched at
run time), `size`, `version`. All rows on the same ffmpeg minor (9.0) so filters
and encoder names match. A test fails if a row lacks any field. `scripts/audit-catalog-repos.py` gets a check
that each URL answers HEAD 200 (network, release-time only).

## Steps

### 1. Resolution order in `ffmpeg_bin.py` (0.5 day)

`ffmpeg_exe()` returns the first that exists and passes a cached `-version` check:
1. `CHAOSENGINE_FFMPEG` env (dev and power users);
2. `~/.chaosengine/bin/ffmpeg/<version>/ffmpeg[.exe]` (the download, via `current` pointer file);
3. imageio-ffmpeg (today's path, kept during rollout, removed in step 9);
4. else raise `FfmpegMissing` (a RuntimeError subclass) carrying `{"action": "install-ffmpeg"}` so routes
   answer 409 with a code the UI turns into the first-use prompt.
`media.py` line 166 stops doing its own imageio lookup and calls `ffmpeg_bin`. Never `PATH` (section 5.4 kept).
Add `ffmpeg_info()`: path, source (env/download/imageio), version, build type (`-L` first line), encoders.

### 2. Download manager `backend_service/ffmpeg_fetch/` (1.5 days)

- `install(job)`: free-space check (3 x size), stream to `~/.chaosengine/bin/ffmpeg/.partial-<rand>/` with
  `urllib`, hashing while writing; progress via the existing Setup job pattern (`gpu_bundle.py`: job worker +
  `/status` polling); cancel supported; resume not needed (files are 10 to 160 MB).
- Verify SHA-256 against the pin **before** extracting. Mismatch: delete, error "download corrupted or
  changed upstream", no retry loop.
- Extract only the listed members with a path check (resolve + parent prefix; reject links and `..`), into
  `.partial-…/<version>/`; `chmod 0755`; run `ffmpeg -version` and `-L` from there and require "Lesser
  General Public License" in `-L` and no `--enable-gpl` in `-buildconf` (the Storyboard filter set); check `missing_filters(REQUIRED)` is empty.
- macOS: if `com.apple.quarantine` is present (it should not be), remove it on our verified file only; run
  `codesign --verify --strict` and check the Team ID matches ours.
- Atomic install: `os.replace` the version dir into place, then write `current` (temp file + replace). Old
  versions are removed after a successful switch. A crash leaves only `.partial-*`, cleaned at next start.
- Settings: `ffmpegSource` (`auto` | `custom path`), `ffmpegAutoUpdate` (off by default: a new pin ships with
  an app update and the Setup row shows "Update available").
- Routes: `GET /api/setup/ffmpeg` (info + pin + installed), `POST /api/setup/ffmpeg/install`,
  `GET /api/setup/ffmpeg/install/status`, `DELETE /api/setup/ffmpeg` (remove download).
- Offline: if the download fails with no network, the message names the URL and the SHA-256 and offers
  "Choose a file…" (a zip or tar the user fetched elsewhere); the same hash check applies, so a manual copy is
  as safe as ours. Custom path in settings is the escape hatch for a system ffmpeg (shown as "not verified").

### 3. UI (1 day)

- First-use prompt in Storyboard render and Music export: "Exporting needs FFmpeg (about N MB). Download
  now?" with Download / Choose file / Cancel and a progress bar. The render
  resumes on success.
- Setup page row "FFmpeg": status (not installed / version + source / update available), Install, Remove,
  encoder found (from step 4).
- New strings in `en` with `null` placeholders in other locales (i18n rule).

### 4. Encoder chooser `ffmpeg_bin.h264_encoder(kind)` (1.5 days incl. matrix)

Order, first that is listed in `ffmpeg -encoders` **and** passes a 1-frame trial encode (cached per binary;
hardware can be listed but absent):

| Platform | Order |
|---|---|
| macOS | `h264_videotoolbox` |
| Windows | `h264_nvenc`, `h264_amf`, `h264_qsv`, `h264_mf`, `libopenh264` |
| Linux | `h264_nvenc`, `h264_vaapi`, `h264_qsv`, `libopenh264` |
| any | `libx264` if present (custom or env binary) |

Settings: `final` and `preview` presets per encoder (starting values, to be tuned by the matrix):
- libx264: today's `-preset medium -crf 18` / `-preset ultrafast -crf 28`.
- videotoolbox: `-q:v 65` / `-q:v 45 -realtime 1` (falls back to `-b:v` from resolution if `-q:v` is refused).
- nvenc: `-preset p5 -rc vbr -cq 19 -b:v 0` / `-preset p1 -cq 28`.
- amf: `-quality quality -rc cqp -qp_i 18 -qp_p 20` / `-quality speed -qp_i 28 -qp_p 30`.
- qsv: `-global_quality 20` / `-global_quality 30`; vaapi: `-qp 20` / `-qp 30` with `hwupload`.
- openh264: `-b:v` from a bits-per-pixel table, `-allow_skip_frames 0`.
All add `-pix_fmt yuv420p` (or `nv12` + upload for vaapi), `-movflags +faststart`. Audio stays `-c:a aac`.
A user setting "Video encoder: Auto / <list>" overrides. `_codec_args` in `assemble.py` and the imageio writer
in `pipeline_helpers.py` (switch to a pipe into `ffmpeg_bin` with the chooser) use it.

Test matrix (manual, recorded in `docs/research/`): 3 clips (talking head, fast motion, flat colour title) x
each encoder present x final/preview; record size, encode time, SSIM and PSNR against a lossless master
(`ssim`/`psnr` filters are built in), and eyeball banding. Bar: SSIM within 0.01 of libx264 `-crf 18` at no
more than 1.5x its size. Machines: M-series Mac (VT), Windows NVIDIA (NVENC), Windows AMD or Intel if available,
Linux CPU-only (openh264).

### 5. CI job `.github/workflows/ffmpeg-macos.yml` (2 days, mostly first green run)

- Trigger: `workflow_dispatch` with `ffmpeg_version` and `revision`; never on every push.
- `runs-on: macos-14`, pinned Xcode. Script `scripts/build-ffmpeg-macos.sh` (in repo, runnable locally):
  fetch pinned tarballs (ffmpeg + `.asc` GPG check, freetype, harfbuzz, fribidi, libass) by SHA-256; build
  deps static into `$DEPS` with meson/autotools; configure with the line in the research doc; `make -j`.
- Gates (fail the job): `-buildconf` has no `enable-gpl|enable-nonfree|enable-version3`; `-L` says Lesser;
  `otool -L` lists only `/usr/lib` and `/System/Library`; `missing_filters(REQUIRED)` empty; a 2 s test render
  with xfade + loudnorm + subtitles + `h264_videotoolbox` succeeds; `lipo -archs` is `arm64`.
- Sign with the existing Developer ID keychain steps from `release.yml` (`--options runtime --timestamp`), zip,
  `xcrun notarytool submit --wait`; no staple (not possible for a bare binary).
- Publish to release `ffmpeg-lgpl-n<ver>-r<rev>`: `.tar.xz`, `SHA256SUMS`, `ffmpeg-<ver>-lgpl-source.tar.xz`
  (all source tarballs + build script + `configure` line + `changes.diff`, even if empty),
  `LICENSE.LGPL-2.1.txt`, `NOTICES.txt` (ISC, FTL credit, Old MIT, FriBidi LGPL), `sbom.spdx.json`
  (generated from the pin list in the script). The job prints the SHA-256 to paste into `pins.py`.
- macOS x86_64: not built (app ships arm64 only). If ever needed: same script with `--arch=x86_64` on a
  `macos-13` runner.

### 6. Keep libx264 where a model needs it (0.5 day)

ltx-2-mlx calls `ffmpeg -c:v libx264` for output and for image preprocessing (`video_vae.py` line 407,
`media_io.py` line 169). Install `imageio-ffmpeg` into `~/.chaosengine/ltx2-venv` from PyPI as part of the
LTX installer (`install-ltx2-mlx.sh`) and point the PATH shim at that copy, not at our LGPL binary. The user's
machine fetches it from PyPI, as with PyAV in the voice venv.

### 7. Notices (0.5 day)

`THIRD_PARTY_NOTICES.md` FFmpeg entry rewritten: FFmpeg is not bundled; on request the app downloads a pinned
BtbN build on Windows and Linux (the exact tag and its `-buildconf` are listed there); imageio-ffmpeg is installed
from PyPI on demand and used by the LTX venv and as the macOS path.

### 8. Tests and E2E (1 day)

- `tests/test_ffmpeg_fetch.py`: pin table shape; SHA mismatch deletes and errors; path traversal member
  rejected; atomic switch survives an exception mid-way (old binary still current); `.partial-*` cleanup;
  non-matching build refused (fake binary printing `--enable-gpl`); offline "choose file" path; routes contract
  (TestClient, fake downloader serving a local file).
- `tests/test_ffmpeg_bin.py`: resolution order with env / download / imageio / none; `FfmpegMissing` maps to
  409 + action code; `h264_encoder` picks by parsed `-encoders` text and trial result (subprocess mocked);
  preset args per encoder.
- Frontend: Setup row and first-use prompt render states (vitest).
- E2E (feature gate): `scripts/e2e_test_suite.py` setup phase `ffmpeg-download`: if a pin exists for the host,
  install into a temp `CHAOSENGINE_HOME`, check version + build type + filters, render a 2 s storyboard with the
  chosen encoder, assert MP4 has H.264 + AAC; skip with reason when offline. Smoke mode checks only the
  pin table and `GET /api/setup/ffmpeg` shape.
- `tests/test_hub_tracker.py` unaffected; add the CLAUDE.md upstream table row for ffmpeg (both sources).

### 9. Rollout order

1. Land CI job (5) and publish `r1` for macOS; pin BtbN month-end tag. (Nothing user-visible.)
2. Land resolution order (1), download manager (2), routes, tests (8). imageio stays as fallback.
3. Land encoder chooser (4) with libx264 still preferred when present, then run the matrix and set the order
   above as default.
4. UI (3) + notices (7) + LTX venv change (6). Ship in a release; Setup offers the download.
5. Next release: remove imageio-ffmpeg from the Setup install list, `gpu_bundle.py`, the `ffmpeg` extra and
   `release-constraints.txt`; drop fallback 3; e2e uses the downloaded binary.

## Effort

About 9.5 working days: CI 2, manager 1.5, chooser + matrix 1.5, UI 1, tests/E2E 1, resolution 0.5,
LTX venv 0.5, notices 0.5, rollout and pin refresh docs 1. First CI green run is the biggest unknown.

## Open questions for the owner

- Windows/Linux: allow openh264 as the last H.264 fallback, or use AV1 via SVT-AV1 instead? Recommended: openh264
  last, with AV1 as a later option.
- Pin refresh cadence: yearly, or with each ffmpeg minor? Recommended: with each app release that needs a fix,
  at least yearly (BtbN month-end builds live two years).
