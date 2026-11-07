# ffmpeg: building and hosting our own LGPL build, and what GPL really asks of us (D-35)

Date: 2026-11-07. Follows `docs/research/licences-wavelets-ffmpeg.md`. Plain-English research, **not legal
advice**. Anything marked UNVERIFIED was not checked against a primary source or a real build.
Plan: `docs/plans/ffmpeg-download.md`.

## Facts found in our own tree first

- The release build does **not** ship imageio-ffmpeg. `release.yml` installs only fastapi, psutil, uvicorn,
  pypdf, python-multipart, huggingface_hub, babel (plus mlx, mlx-lm, gguf on macOS). imageio-ffmpeg is installed
  on demand from PyPI by the Setup page (`routes/setup/__init__.py` line 65, `gpu_bundle.py` line 76) or the
  `ffmpeg` extra. So today the user's machine fetches the GPL binary from PyPI and we are not its distributor.
  `THIRD_PARTY_NOTICES.md` says "the binary it downloads or ships"; "downloads" is the accurate word. UNVERIFIED
  against a built runtime archive (`tar -tzf ... | grep imageio_ffmpeg` should be empty).
- Consequence: building our own macOS binary and hosting it makes us a distributor for the first time. That is
  fine (LGPL duties are small, list below), but it is a new duty, not a lighter one.
- Callers of ffmpeg: `storyboard/ffmpeg_bin.py` (renders, probe, filters), `storyboard/media.py` line 166 (its own
  imageio lookup, bypasses `ffmpeg_bin`), `storyboard/peaks.py`, `ltx2_mlx_runtime.py` (PATH shim for ltx-2-mlx),
  `video_runtime/pipeline_helpers.py` line 412 (imageio writer, `codec="libx264"`), `scripts/e2e_test_suite.py`.
- **ltx-2-mlx hard-codes libx264**: `ltx_core_mlx/model/video_vae/video_vae.py` line 407
  (`-c:v libx264 -crf 18`) and `ltx_pipelines_mlx/utils/media_io.py` line 169, which round-trips the input
  image through libx264 to match the training distribution. An LGPL ffmpeg on that PATH shim breaks LTX-2.3
  output and changes image-to-video conditioning. Keep a libx264 ffmpeg for that venv only (see plan step 6).
- Filters the Storyboard needs (`assemble.py`, `assemble_graphs.py`): xfade, acrossfade, sidechaincompress,
  asplit, loudnorm, amix, afade, adelay, apad, atrim, aresample, aformat, tpad, trim, setpts, volume, anullsrc,
  and **`subtitles`** for captions (not drawtext). Checked in ffmpeg 8.0 `configure`: none of these is in the
  GPL-only filter list. They need no external library except `subtitles` (`avformat avcodec libass`).
  `drawtext` (not used today) needs `libfreetype libharfbuzz`.
- App licence: Apache-2.0 (`LICENSE`). The app is not sandboxed and has no `LSFileQuarantineEnabled`
  (`src-tauri/macos/ChaosEngineAI.entitlements`, `Info.plist`); hardened runtime is on.
- Release targets: macOS **arm64 only** (`release.yml` matrix: `macos-latest`, `aarch64`), Linux x86_64 and
  arm64, Windows x86_64. So macOS x86_64 is not needed.

## Q1. Building and hosting our own LGPL ffmpeg for macOS arm64

### configure line (ffmpeg n9.0.x, static executables, LGPL-2.1-or-later)

```
PKG_CONFIG_LIBDIR="$DEPS/lib/pkgconfig" ./configure \
  --prefix="$OUT" --arch=arm64 --target-os=darwin --cc=clang \
  --extra-cflags="-mmacosx-version-min=12.0 -I$DEPS/include" \
  --extra-ldflags="-mmacosx-version-min=12.0 -L$DEPS/lib" \
  --pkg-config-flags=--static --disable-shared --enable-static \
  --disable-autodetect --enable-pthreads --enable-zlib --enable-iconv \
  --enable-videotoolbox --enable-audiotoolbox \
  --enable-libass --enable-libfreetype --enable-libharfbuzz --enable-libfribidi \
  --disable-network --disable-ffplay --disable-doc --disable-debug
```

- Never: `--enable-gpl`, `--enable-nonfree`, `--enable-version3` (the last turns it into LGPLv3, not needed).
  No libx264, libx265, libfdk_aac, frei0r, rubberband, vidstab, xvid (ffmpeg's GPL/nonfree library lists).
- `--disable-autodetect` is the important one: the macOS runner has Homebrew, and autodetect would link
  `/opt/homebrew` dylibs that users do not have. Build freetype, harfbuzz, fribidi, libass from pinned source
  tarballs as static libraries into `$DEPS`, and point `PKG_CONFIG_LIBDIR` only there.
- libass on macOS: `-Dcoretext=enabled -Dfontconfig=disabled` so system fonts come through CoreText and
  fontconfig is not needed (meson option names UNVERIFIED for the pinned libass version).
- Keep `ffprobe` (small, handy). `--disable-network` matches our `-protocol_whitelist file,pipe`.
- Not needed on macOS: libopenh264 (VideoToolbox is on every Apple Silicon Mac, and a from-source openh264
  is not covered by Cisco's patent licence, see below) and libvpx.
- macOS cannot link libSystem statically, so "static" means every non-Apple library is inside the binary and
  only `/usr/lib` and `/System/Library/Frameworks` are dynamic. CI must assert this with `otool -L`.

### Licences of what goes into the binary

| Library | Licence | LGPL-compatible | Duty |
|---|---|---|---|
| FFmpeg (libav*, fftools) | LGPL-2.1-or-later (no gpl flags) | yes | source + licence text |
| libass | ISC | yes | notice |
| FreeType | FTL or GPL-2.0 (we take FTL) | yes (FTL) | credit line in docs: "Portions of this software are copyright The FreeType Project" |
| HarfBuzz | "Old MIT" | yes | notice |
| FriBidi | LGPL-2.1-or-later | yes | source + licence text |
| zlib, iconv, VideoToolbox, AudioToolbox | system (Apple) | n/a | none |
| (Win/Linux BtbN only) openh264 | BSD-2-Clause + patent caveat | yes | notice |
| (Win/Linux BtbN only) libvpx | BSD-3-Clause + Google patent grant | yes | notice |

Licences read from each GitHub repo's licence field on 2026-11-07 (FreeType and HarfBuzz from memory of their
own licence files; UNVERIFIED today).

### Encoders without x264

- macOS: `h264_videotoolbox` and `hevc_videotoolbox` (hardware; quality via `-q:v` on Apple Silicon, or
  `-b:v`; no CRF), `aac` (built-in, LGPL) or `aac_at` (AudioToolbox). Apple's licence covers the platform codec.
- openh264: Cisco pays the MPEG LA royalty **only for Cisco's own prebuilt binary, downloaded from Cisco at
  install time**; "the team is responsible for paying all applicable license fees" when you compile it
  yourself (<https://www.openh264.org/faq.html>). BtbN builds openh264 from source into every variant
  (`scripts.d/50-openh264.sh`, `ffbuild_enabled` returns 0), so that coverage does not apply to it.
- libvpx (VP9) and SVT-AV1 / libaom (AV1) are royalty-free alternatives if a software fallback is wanted.

### Code signing, hardened runtime, notarisation, quarantine

- Every arm64 Mach-O must carry at least an ad-hoc signature to run; the linker adds one. That is enough for
  a file our own process downloads and launches.
- Quarantine: macOS adds `com.apple.quarantine` when the downloading app is sandboxed or sets
  `LSFileQuarantineEnabled`, or calls the quarantine API. Our Python sidecar using `urllib` does none of those,
  so the file arrives without the flag and Gatekeeper never checks it (same as our existing downloaded
  `llama-server-turbo` and prebuilt llama). UNVERIFIED by a live test; the plan adds one. If a user downloads
  the zip by hand in a browser, the flag is set, so notarise anyway.
- Recommended anyway: sign with our existing Developer ID (`codesign --options runtime --timestamp`), notarise
  by zipping the binary and sending it to `notarytool`. A bare Mach-O **cannot be stapled** (stapling works
  only for bundles, pkg and dmg; Apple forum thread 736973), so Gatekeeper checks online on first launch of a
  quarantined copy. ffmpeg needs no entitlements under hardened runtime (VideoToolbox works; UNVERIFIED).
- The app's own hardened runtime and library validation do not apply to a child process with its own signature.

### Hosting

- GitHub Releases of our repo under a separate tag, e.g. `ffmpeg-lgpl-n9.0.2-r1`, so URLs never move with
  app releases. Asset limit 2 GiB per file; expected size about 25 to 40 MB raw, 10 to 15 MB as `.tar.xz`
  (estimate; imageio's GPL arm64 binary with x264/x265 and more is 49 MB).
- Same release must carry: the binary archive, `SHA256SUMS`, the **source tarball** (ffmpeg plus freetype,
  harfbuzz, fribidi, libass exact tarballs, our build script, `configure` line, any patches as `changes.diff`),
  `LICENSE.LGPL-2.1.txt`, an SBOM (SPDX or CycloneDX JSON). Same server as the binary is what ffmpeg's
  checklist asks (<https://ffmpeg.org/legal.html>) and what LGPL-2.1 section 4 accepts.

### LGPL obligations when we distribute (ffmpeg checklist, adapted to a separate executable)

1. No `--enable-gpl` / `--enable-nonfree`; no GPL libraries in the binary.
2. Give the exact corresponding source (tarball, same place), our patches, and the build instructions.
3. Licence text, and an attribution line: "This software uses code of FFmpeg licensed under the LGPLv2.1"
   with links, in the About box, Setup row and notices.
4. Do not claim ownership; no anti-reverse-engineering terms (our terms of use, D-41, must not forbid it).
5. Repeat for every LGPL library compiled in (FriBidi).
6. Static vs dynamic: the checklist's "use dynamic linking" is for an *application that links* the libraries,
   so users can swap them (LGPL-2.1 section 6). Our app links nothing; ffmpeg runs as its own program. Inside
   the binary everything is LGPL or permissive, so the whole executable is a work based on the library and
   section 4 (source) is the duty; anyone can rebuild it from our tarball, which is what relinking protects.
   A static CLI is therefore fine. UNVERIFIED legally, but this matches how BtbN and others ship static LGPL.

### Reproducibility and pinning

- Pin the ffmpeg release tarball by SHA-256 and check its GPG signature (ffmpeg.org publishes `.asc`); pin every
  dependency tarball by SHA-256 in the build script; pin the runner image (`macos-14`) and Xcode version.
- Set `SOURCE_DATE_EPOCH`, record `ffmpeg -buildconf` in the release notes. Compare the **unsigned** binary
  hash between two runs; the signed hash differs because of the signing timestamp. Full bit-for-bit
  reproducibility on macOS is UNVERIFIED.

### Windows and Linux (BtbN)

- Asset names (release `autobuild-2026-11-06-12-50`, checked with `gh api`):
  `ffmpeg-n9.0.2-47-gb204e94e90-{win64,winarm64}-lgpl-9.0.zip` (161 / 112 MB),
  `ffmpeg-n9.0.2-47-gb204e94e90-{linux64,linuxarm64}-lgpl-9.0.tar.xz` (129 / 111 MB); `-lgpl-shared` variants
  are about half the size but carry DLLs. A `checksums.sha256` asset (sha256sum format) is in every release.
- What `lgpl` contains: BtbN's scripts enable openh264, libass (with fontconfig, fribidi, freetype, harfbuzz),
  nv-codec-headers (NVENC), AMF, and oneVPL/QSV (not on arm64) in every variant; x264/x265 are disabled for
  `lgpl*`. So `subtitles`, `drawtext` and openh264 are present. UNVERIFIED on a downloaded binary.
- Pinning: BtbN keeps "the last build of each month for two years" and the last 14 daily builds (README).
  Pin a **month-end** tag and store our own SHA-256 for each asset in code, never trust `checksums.sha256` at
  run time. Refresh the pin at least once a year. Downloading from BtbN keeps BtbN as the distributor; we owe
  a notice and a link. Mirroring the zip into our releases would make us the distributor (then source duty).

## Q2. Is GPL "just a notice" for us?

**ELI5.** GPL is a deal: you may hand out the program, but whoever you hand it to must also be able to get
its recipe (source) and the same freedoms. It only reaches programs that are *part of* the GPL program. A
separate program that merely *runs* ffmpeg, like you run a calculator, stays under its own licence.

**Precisely.**

- Running ffmpeg as a separate process over command-line arguments and pipes is "mere aggregation" in the
  FSF's view: pipes, sockets and command-line arguments are the mechanisms "normally used between two separate
  programs", unless the exchange is so intimate (shared complex internal data structures) that they form one
  program (<https://www.gnu.org/licenses/gpl-faq.html#MereAggregation>). Our use (build a filter graph, run,
  read stderr) is the normal case. So GPL does **not** make our app GPL. PyAV is different: it loads GPL
  libraries into our voice worker's process (linking), but users install it from PyPI, so we distribute nothing.
- Even in a worst case, Apache-2.0 code can be combined into a GPLv3 work (FSF licence list,
  <https://www.gnu.org/licenses/license-list.html#apache2>), and ffmpeg's GPL is "v2 or later", so an open-source
  Apache app is never stuck. It is not a threat to us either way.
- If **we distribute** a GPL ffmpeg (inside the installer or on our release page), it is more than a notice:
  1. the GPL text, and a notice that this binary is GPL;
  2. the complete corresponding source of **that exact binary**: ffmpeg, x264, x265 and every library
     compiled in, plus the scripts used to build it (GPLv2 section 3, GPLv3 section 1 and 6);
  3. delivered either alongside the binary (same release page counts; GPLv3 also allows another server with
     clear directions, <https://www.gnu.org/licenses/gpl-faq.html#SourceAndBinaryOnDifferentSites>) or as a
     written offer valid for at least three years (GPLv2 3(b)). A bare link to upstream ffmpeg.org is not
     enough for a binary we built, because the upstream tarball is not the exact source of our build.
- If the **user's machine downloads** it from a third party (PyPI, BtbN), the host distributes. Our part is
  a correct notice and a link. That is the situation today with imageio-ffmpeg.
- Patents are a separate question from copyright licences: H.264 in x264 or a from-source openh264 can carry
  pool royalties for a distributor; the GPL/LGPL choice does not change that. VideoToolbox, NVENC/AMF/QSV and
  OS codecs move that question to the platform vendor (for GPU encoders, UNVERIFIED).

## Sources

- <https://ffmpeg.org/legal.html> (LGPL checklist, patents), FFmpeg `configure` on branch `release/8.0`
  (GPL filter and library lists, `subtitles` and `drawtext` deps)
- <https://www.gnu.org/licenses/gpl-faq.html#MereAggregation>,
  <https://www.gnu.org/licenses/gpl-faq.html#SourceAndBinaryOnDifferentSites>,
  <https://www.gnu.org/licenses/license-list.html#apache2> (gnu.org did not load during this research; wording
  above is from quotations of the FAQ, UNVERIFIED verbatim)
- <https://www.openh264.org/faq.html>
- <https://github.com/BtbN/FFmpeg-Builds> (README, `variants/`, `scripts.d/50-openh264.sh`, `50-libass.sh`,
  `50-x264.sh`, `50-ffnvcodec.sh`, `50-amf.sh`, `50-onevpl.sh`), release `autobuild-2026-11-06-12-50`
- Apple Developer Forums thread 736973 (stapling a bare binary), <https://tuist.dev/blog/2024/12/31/signing-macos-clis>
