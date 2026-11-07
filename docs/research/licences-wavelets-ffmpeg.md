# Licence research: pytorch-wavelets (D-37) and ffmpeg (D-35)

Date: 2026-11-07. Plain-English research, **not legal advice**. Anything marked UNVERIFIED was not checked against a primary source.

## Summary in plain words

**Job 1, pytorch-wavelets.** The package's licence file is an MIT licence (Fergal Cotter, 2020) for the parts he wrote, which includes the plain wavelet transform (DWT) that ACE-Step uses. The same file says the "dual-tree" (DTCWT) parts come from a MATLAB toolbox whose author says "supplied free of charge for research purposes only". PyPI's label "Free To Use But Restricted" is a catch-all for that. ACE-Step only needs the package for an optional quality tweak (DCW), and its own code already does the default Haar version without it. So we can **leave the package out** and lose nothing on Apple Silicon.

**Job 2, ffmpeg.** If we put ffmpeg inside the installer we are the one handing it to people, so we owe the licence paperwork (a notice, and the source code for the exact build, including x264 and x265). If the app downloads it when the user clicks a button, the download comes from someone else's server and we mostly just point at it. A build without x264 (an "LGPL build") has lighter rules than a GPL one. ffmpeg is the right tool for what we do. The best plan is a download button with a checksum, using an LGPL build.

## Job 1: pytorch-wavelets

### The licence, from the sources

- LICENSE (master, fetched 2026-11-07, <https://github.com/fbcotter/pytorch_wavelets/blob/master/LICENSE>): it says it covers "any parts of this library which are novel in comparison to the original DTCWT MATLAB toolbox" by Nick Kingsbury and Cian Shaffrey, "The DWT sections come under this license", and "If you wish to use the DTCWT, you should read that license as well". The licence itself is the standard **MIT**, copyright 2020 Fergal Cotter.
- README.rst "Provenance" section (<https://github.com/fbcotter/pytorch_wavelets/blob/master/README.rst>): the DTCWT is "Based on the Dual-Tree Complex Wavelet Transform Pack for MATLAB by Nick Kingsbury, Cambridge University", and: "This code is copyright and is supplied free of charge for research purposes only", plus a request to cite the work in papers. It does not use the words "commercial" or "non-commercial".
- PyPI 1.3.0 (<https://pypi.org/project/pytorch-wavelets/1.3.0/>; uploaded 2023-04-13): `License: Free To Use`, classifier `License :: Free To Use But Restricted` (checked also in the METADATA of the spike venv). Requires `numpy`, `six`, `torch`. It does **not** list PyWavelets as a requirement, but the code does `import pywt` at the top of the DWT modules, so PyWavelets is needed in practice (ACE-Step lists it separately).
- PyWavelets 1.9.0 / 1.10.0: `MIT AND BSD-3-Clause` (PyPI metadata).

### What the restriction means

- Commercial use: the **DWT** code is plain MIT, so commercial use, redistribution and modification are allowed if the copyright and permission notice are kept. The **DTCWT** code (folder `dtcwt/`, which ships `.npz` filter files such as `near_sym_*` and `qshift_*`) is derived from the Cambridge MATLAB toolbox, whose stated condition is research use. That wording is ambiguous for a product, and nobody wrote a clear carve-out for commercial use.
- The repo gives no line-by-line split, so the licence of the DTCWT files is unclear. We use none of them.
- The package imports the DTCWT module at `import pytorch_wavelets` time (its `__init__.py` imports `dtcwt.transform2d`), so installing it puts that code on the user's disk even though nothing calls it.

### How ACE-Step 1.5 (commit `ca1e85f`) uses it

Read from `~/.chaosengine/spikes/acestep/src` (read-only):

- Only place that imports it: `acestep/models/common/dcw_loader.py` line 36: `from pytorch_wavelets import DWT1DForward, DWT1DInverse`, inside a lazy `try/except ImportError`. Its docstring says the dependency is "optional" and DCW is opt-in. If missing, it logs one warning and returns `None`, and the caller skips the correction (a no-op).
- DCW = "Differential Correction in Wavelet domain" (CVPR 2026, arXiv 2604.16044), a small sampler correction on the DiT latents. It is not part of the VAE, the text encoder or the weights. Defaults: `dcw_wavelet="haar"`, scalers 0.05/0.02. It is on by default for Turbo models, off for others (`inference.py`, around line 159).
- Only the 1-D DWT classes are used. The 2-D DWT and the DTCWT are never called.
- MLX path (what we run on Apple Silicon): `acestep/models/mlx/dcw_correction_mlx.py` has its own `_haar_dwt_1d` in MLX. Haar needs no pytorch-wavelets. Other wavelets (db4, sym8) go through a torch bridge, and if the import fails that code falls back to native Haar with one warning.
- ACE-Step's own `pyproject.toml` lists `pytorch-wavelets>=1.3.0` and `pywavelets>=1.9.0` as hard dependencies, but we install ACE-Step with `pip install --no-deps` (`music_env.py` line 419), so only our exported requirements file decides.

### Can we stub or drop it?

Static evidence says yes. I did **not** run a generation with it removed (no throwaway venv was built; the 35 GB spike dir was only read). A short test is still needed before we trust it, see "Next step".

- Dropping it: MLX + Haar (the default) is unchanged. MLX + a non-Haar wavelet quietly becomes Haar. CPU/CUDA (not our Mac target) skips DCW, so output differs slightly from reference.
- A stub is not needed, because ACE-Step already handles the missing import.

### Alternatives (licences checked)

| Option | Licence | Verdict |
|---|---|---|
| Leave pytorch-wavelets out | n/a | Best for Apple Silicon. Zero code. |
| Keep PyWavelets only | MIT AND BSD-3-Clause (PyPI) | Already installed. Pure NumPy `pywt.dwt` works on CPU arrays, no torch. Fine as a base. |
| ptwt (PyTorch Wavelet Toolbox) | **EUPL-1.2** (PyPI classifier; not MIT) | Avoid. A copyleft licence, and it needs PyWavelets as well. |
| Own small Haar DWT | ours | ACE-Step already has one for MLX (`_haar_dwt_1d`). A 20-line torch twin would cover CPU/CUDA if ever needed. |
| Vendor only the 1-D DWT files from pytorch-wavelets | MIT for those files, with the notice kept | Possible but pointless, and we would have to keep the notice. |

### Recommendation (D-37)

1. **Replace/avoid:** remove the `pytorch-wavelets==1.3.0` line (and its `# pytorch-wavelets` "via" comments) from `backend_service/music_requirements.txt`. Keep `pywavelets` (MIT). Run the one-time smoke test below. If it passes, close FU-138 as "dropped".
2. **If a test fails and we must ship it:** it is acceptable to install on demand into the isolated music venv, never bundled, under this wording: the DWT part is MIT; the DTCWT part is unused. Notice text is already in `THIRD_PARTY_NOTICES.md` (updated in this commit). Residual risk is the ambiguous DTCWT wording; asking Fergal Cotter by GitHub issue for a clear MIT statement would settle it.
3. Not legal advice. If the music pack becomes a paid or bundled feature, have a lawyer read the two files above.

**Done 2026-11-07 (D-37):** removed from `music_requirements.txt`; fresh venv from the new file, 20 s instrumental on MLX with default DCW passed. The worker log shows `DCW enabled ... wavelet=haar, backend=MLX-native Haar` and no wavelet warning (the skip warning only appears on the torch path). FU-138 closed.

## Job 2: ffmpeg

### (a) Bundle vs download button, explained simply

What we have today: `imageio-ffmpeg` (BSD-2-Clause wrapper) pinned at 0.6.0 (`release-constraints.txt`). Its wheel carries an ffmpeg 7.1 binary (`ffmpeg-macos-aarch64-v7.1`, 49 MB in our `.venv`). I could not find a licence statement in its README (<https://github.com/imageio/imageio-ffmpeg>). Our existing notice records that `ffmpeg -version` shows `--enable-gpl` with libx264 and libx265, i.e. **GPL v2 or later**. imageio-ffmpeg's current master pins binaries by SHA-256 from BtbN (`-gpl` builds) for Linux and Windows, and for macOS arm64 from osxexperts.net with "nothing published upstream" to verify against (`tools/ffmpeg-sources.json`). Its latest release is 0.8.1 (ffmpeg 9).

| | Bundle in installer | Button downloads it |
|---|---|---|
| Who is the "distributor" | **We are.** | The host (BtbN, GitHub, etc.). We start a download on the user's machine. A lawyer might still call this facilitation, so keep the notice and a link anyway. |
| GPL build | Must give the licence text and a written offer (or a link) for the exact source of ffmpeg, x264 and x265 for three years or as long as we ship it (GPL v2 clause 3, v3 section 6). | Host carries the source duty. We link the source and the licence in the app and notices. |
| LGPL build | Notice, licence text, source link, and the user must be able to swap the library (easy when ffmpeg is a separate executable). See <https://ffmpeg.org/legal.html> checklist: no `--enable-gpl`, no `--enable-nonfree`, no libx264. | Same as above but lighter, because we never hold the file. |
| Our own code's licence | Unaffected either way: we run ffmpeg as a **separate program**, never linked. | Same. |
| Patents (H.264, AAC) | ffmpeg's FAQ says it does not know what is patented and that private users have little to worry about; commercial products should "read on" (<https://ffmpeg.org/legal.html>). Shipping an H.264 encoder to the public is where licensing bodies (MPEG LA / Via LA) can matter. UNVERIFIED for our volume. | The user's own install, so less direct exposure. Not zero. |
| Size | +47 to 49 MB (our binary). Full BtbN LGPL archives are 115 to 175 MB. | 0 in the installer. |
| Offline | Works first run. | Needs one download; works offline afterwards. |
| Updates / trust | We control the exact build and sign it with the app. | We must verify a checksum or signature ourselves, and a host can vanish (Zeranoe did in 2020). |

Platform notes: macOS needs the binary signed and notarised if shipped inside the app bundle (UNVERIFIED for third-party static binaries; a downloaded file also gets a quarantine flag unless fetched by our own code, which does not set it). Windows and Linux have no equivalent step. The Microsoft/Apple/Linux native codecs are covered by their own licences, so using the operating system encoder moves the patent question onto the platform.

Important side finding: **PyAV's wheels are also GPL.** `THIRD_PARTY_NOTICES.md` said PyAV bundles FFmpeg under LGPL-2.1. I downloaded the macOS arm64 wheel (av 14.1.0, scratch dir only) and its libavcodec's configure line includes `--enable-gpl --enable-version3 --enable-libx264 --enable-libx265`, and libavutil reports "GPL version 3 or later". The pyav-ffmpeg README (<https://github.com/PyAV-Org/pyav-ffmpeg>) lists x264 and x265 for current builds too. Unlike imageio-ffmpeg, PyAV loads these libraries **into our voice worker's process**, which is a stronger GPL link than a separate program. The voice venv installs PyAV from PyPI on the user's machine (`voice_env.py` "av>=13"), so we do not distribute it, but the notice was wrong and is fixed in this commit.

### (b) Is ffmpeg the best open-source choice?

For concat/xfade/acrossfade, sidechaincompress, loudnorm, drawtext, probing, waveform peaks and an H.264/AAC mux: **yes**, with no close open-source rival for the filter set.

| Candidate | Licence | Fit |
|---|---|---|
| ffmpeg CLI | LGPL-2.1+ (GPL with some options) | Has every filter we use. Our code already builds filter graphs for it. Best. |
| PyAV | BSD-3-Clause wrapper, but wheels are GPL (above) | Python API over the same libraries. Filter graphs are possible but clumsier, and no ffprobe-free probe benefit. Keep only for audio decode. |
| MoviePy 2.2.1 | MIT | A wrapper that shells out to ffmpeg via imageio-ffmpeg (<https://pypi.org/project/moviepy/>). Adds no filters and no legal change. |
| GStreamer | LGPL-2.1 | Powerful, but a large dependency, plugins split into good/bad/ugly/libav, and no equal one-line filters (no `loudnorm`, `xfade`). Reject. |
| libav (the fork) | LGPL/GPL | Dormant, behind ffmpeg. Reject. |
| MLT / melt | LGPL-2.1 (GPL modules) | Built for editors; wraps ffmpeg anyway; heavy for our needs. Reject. |
| OpenCV VideoWriter | Apache-2.0 | Video only, no audio filters. Reject. |
| AVFoundation / Media Foundation | OS APIs | Native, no licence exposure for H.264/AAC (OS holds it), but three separate code paths, no filter graph. Possible later for the final mux only. |
| Rust crates (ffmpeg-next, etc.) | binding crates link ffmpeg | Same libraries with more build pain. Reject. |

**Verdict:** keep ffmpeg. The only real question is which build and how we deliver it.

### (c) Options, ranked

Facts about trusted builds (checked 2026-11-07):

- **BtbN/FFmpeg-Builds** (<https://github.com/BtbN/FFmpeg-Builds>, repo MIT): variants `gpl`, `lgpl` ("lacking libraries that are GPL-only, most prominently libx264 and libx265") and `nonfree`. Targets: win64, winarm64, linux64, linuxarm64. **No macOS.** LGPL archives for n9.0: linux64 136 MB, linuxarm64 116 MB, win64 168 MB, winarm64 117 MB. It builds openh264 (BSD, Cisco) into its variants, and `lgpl` should use it; I did not confirm the exact configure line of each release (UNVERIFIED, run `ffmpeg -version` and check for `libopenh264` and the absence of `--enable-gpl`). The AMD AMF, and Vulkan hooks exist too; NVENC is available through the headers (UNVERIFIED which encoders are in the LGPL build).
- **evermeet.cx** (macOS x86_64 only on our reading): its configure line shows `--enable-libx264 --enable-libx265 --enable-version3`, so GPL. Not a fit for arm64 anyway.
- **Martin Riedl** (<https://ffmpeg.martin-riedl.de>): macOS builds include x264 and x265, so GPL; I found no LGPL variant (UNVERIFIED beyond the page text).
- **Gyan.dev** (<https://www.gyan.dev/ffmpeg/builds/>): "All builds are 64-bit, static and licensed as GPLv3". Not LGPL.
- **osxexperts.net** (what imageio-ffmpeg uses for macOS arm64): no upstream checksum, so lowest trust.
- **No prebuilt LGPL macOS arm64 build from a trusted source found.** To get one we compile ffmpeg ourselves without `--enable-gpl` (with `--enable-videotoolbox`; `--enable-libopenh264` optional) and host it in our GitHub release with a checksum. That is a small CI job (UNVERIFIED effort; ffmpeg's configure is stable).

Encoders without x264: ffmpeg's built-in **AAC** is LGPL, so audio is unchanged. For H.264: **VideoToolbox** on macOS (hardware, good speed, quality per bit lower than x264 `-crf 18`, and no CRF flag: use `-q:v` or bitrate), **NVENC/AMF/QSV** on Windows and Linux (similar tradeoff, and needs the right GPU), **openh264** (software, baseline-quality, rate-control only, no CRF). Our storyboard renders use `libx264 -crf 18 -preset medium` (final) and `-crf 28 -preset ultrafast` (preview) in `backend_service/storyboard/assemble.py` lines 158 and 159, so this needs a small encoder switch plus a test of file size and look. imageio-based export in `video_runtime/pipeline_helpers.py` line 412 also asks for libx264.

Ranking:

1. **Download on demand, LGPL build, with SHA-256 (recommended).** Zero size in the installer, lighter licence duties, no x264. Windows and Linux: pull BtbN `lgpl` assets pinned by tag. macOS arm64: our own LGPL build uploaded to our releases. About a 130 MB download on Windows/Linux (the ffmpeg exe is a fraction once extracted; extract only `bin/ffmpeg`).
2. **Bundle an LGPL build.** Same legal posture as 1 but installer grows by the exe size and we carry the source offer. Good if offline first-run matters.
3. **Keep the GPL build with notice and source offer** (current D-35 decision). Cheapest today; keeps best H.264 quality (libx264) but we remain the GPL distributor when bundled via pip, and the PyAV finding shows the notice must say so.

### Implementation sketch (option 1)

- Add `backend_service/storyboard/ffmpeg_fetch.py`: a table `{platform: (url, sha256, member, version)}`; download to `~/.chaosengine/bin/ffmpeg-<ver>/` with progress through the existing Setup install route; verify SHA-256 before extracting; `chmod +x`.
- `ffmpeg_bin.ffmpeg_exe()` order: `CHAOSENGINE_FFMPEG` env, then the downloaded file, then imageio-ffmpeg as today (so nothing breaks).
- Add an `h264_encoder()` chooser (videotoolbox / nvenc / openh264, else libx264 when the binary has it, read from `ffmpeg -encoders`), used by `_codec_args`. Add a Setup row and a check in `scripts/e2e_test_suite.py` (new feature gate in CLAUDE.md).
- The Setup row links the licence text and source; add the FFmpeg paragraph to the app About page; update `THIRD_PARTY_NOTICES.md`.
- Check the filters we need exist in the downloaded build (`ffmpeg_filters()` already lists them; `drawtext` needs libfreetype, UNVERIFIED in `lgpl` builds).

## Changes made with this research

- `THIRD_PARTY_NOTICES.md`: pytorch-wavelets entry rewritten with the verified licence text and the recommendation; PyAV entry corrected (wheels are GPL builds).
- `docs/hub/decisions.md` was **not** edited. Suggested hub text for you: D-37 "Recommend dropping `pytorch-wavelets` from the music requirements (ACE-Step skips DCW cleanly; Haar is native on MLX); confirm with one smoke generation." D-35 "Recommend a download-on-demand LGPL ffmpeg with SHA-256; note PyAV wheels are also GPL."

## Sources

- <https://github.com/fbcotter/pytorch_wavelets> (LICENSE, README.rst), <https://pypi.org/project/pytorch-wavelets/1.3.0/>
- <https://pypi.org/project/PyWavelets/>, <https://pypi.org/project/ptwt/>
- ACE-Step pinned commit files read locally: `acestep/models/common/dcw_loader.py`, `acestep/models/mlx/dcw_correction_mlx.py`, `acestep/inference.py`
- <https://ffmpeg.org/legal.html>, <https://github.com/BtbN/FFmpeg-Builds>, <https://github.com/imageio/imageio-ffmpeg>, <https://github.com/PyAV-Org/pyav-ffmpeg>
- <https://evermeet.cx/ffmpeg/>, <https://ffmpeg.martin-riedl.de>, <https://www.gyan.dev/ffmpeg/builds/>
