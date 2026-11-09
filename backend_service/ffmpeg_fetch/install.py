"""Download, verify, extract and atomically install one pinned ffmpeg.

Order is fixed and nothing is run until the archive's SHA-256 matches the pin:
download (hashing as it streams) -> compare hash -> extract only ``bin/ffmpeg[.exe]`` and ``bin/ffprobe[.exe]``
(no links, no ``..``) -> only then run the extracted binary for ``-version``, ``-L``, ``-buildconf`` and
``-filters`` -> rename the version folder into place -> rewrite the ``current`` pointer (temp file + replace).
A crash leaves only ``.partial-*`` folders, which the next install removes.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import stat
import subprocess
import tarfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, IO

from backend_service.ffmpeg_fetch.pins import Pin

Progress = Callable[[str, int, int], None]  # (phase, done, total)

REQUIRED_FILTERS = frozenset({"xfade", "acrossfade", "loudnorm", "amix", "sidechaincompress", "subtitles"})
_MEMBER = re.compile(r"^[^/]+/bin/(ffmpeg|ffprobe)(\.exe)?$")
_CHUNK = 1 << 20


class FetchError(Exception):
    """A readable failure: shown to the user as is."""


class Cancelled(FetchError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def cleanup_partials(root: Path) -> None:
    if root.is_dir():
        for entry in root.glob(".partial-*"):
            shutil.rmtree(entry, ignore_errors=True)


def download(pin: Pin, target: Path, progress: Progress, cancel: threading.Event | None,
             opener: Callable[..., IO[bytes]] | None = None) -> str:
    """Stream the pinned URL into ``target``; returns the SHA-256 of what was written."""
    if not pin.url.startswith("https://"):
        raise FetchError("Download refused: the pinned address is not https.")
    if opener is None:
        import urllib.request

        opener = urllib.request.urlopen
    digest = hashlib.sha256()
    done = 0
    try:
        with opener(pin.url, timeout=60) as response, target.open("wb") as out:  # type: ignore[call-arg]
            total = int(getattr(response, "length", None) or pin.size)
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("Install cancelled.")
                block = response.read(_CHUNK)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                done += len(block)
                progress("download", done, total)
    except Cancelled:
        raise
    except OSError as exc:  # urllib's URLError and socket errors are OSErrors
        raise FetchError(
            f"Could not download {pin.url} ({exc}). You can fetch that file in a browser and choose it instead; "
            f"its SHA-256 must be {pin.sha256}."
        ) from exc
    return digest.hexdigest()


def _check_hash(pin: Pin, actual: str) -> None:
    if actual != pin.sha256:
        raise FetchError("The download is corrupted or changed upstream (its SHA-256 does not match the pin). Nothing was installed.")


def _is_link_zip(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def extract_members(archive: Path, pin: Pin, out_dir: Path) -> list[Path]:
    """Copy only ``*/bin/ffmpeg`` and ``*/bin/ffprobe`` out of the archive, by file name, into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def accept(name: str) -> str | None:
        # Checked on the name as stored in the archive: on Windows the zip reader turns "\\" into "/" before
        # we see ``filename``, so zip members are checked by ``orig_filename``. Drive letters ("C:") are refused too.
        if (
            "\\" in name
            or name.startswith("/")
            or re.match(r"^[A-Za-z]:", name)
            or ".." in PurePosixPath(name).parts
        ):
            raise FetchError(f"The archive holds an unsafe path ({name}). Nothing was installed.")
        match = _MEMBER.match(name)
        return PurePosixPath(name).name if match else None

    def put(base: str, source: IO[bytes]) -> None:
        dest = (out_dir / base).resolve()
        if dest.parent != out_dir.resolve():
            raise FetchError("The archive holds an unsafe path. Nothing was installed.")
        with dest.open("wb") as handle:
            shutil.copyfileobj(source, handle, _CHUNK)
        dest.chmod(0o755)
        written.append(dest)

    try:
        if pin.archive == "zip":
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    base = accept(info.orig_filename)
                    if base is None or info.is_dir():
                        continue
                    if _is_link_zip(info):
                        raise FetchError("The archive holds a link where a program should be. Nothing was installed.")
                    with zf.open(info) as source:
                        put(base, source)
        else:
            with tarfile.open(archive, "r:*") as tf:
                for member in tf:
                    base = accept(member.name)
                    if base is None:
                        continue
                    if not member.isreg():
                        raise FetchError("The archive holds a link where a program should be. Nothing was installed.")
                    source = tf.extractfile(member)
                    if source is None:
                        continue
                    with source:
                        put(base, source)
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        raise FetchError(f"The archive could not be read ({exc}). Nothing was installed.") from exc
    names = {path.stem if path.suffix == ".exe" else path.name for path in written}
    if "ffmpeg" not in names:
        raise FetchError("The archive holds no ffmpeg program. Nothing was installed.")
    return written


def _run(exe: Path, *args: str) -> str:
    result = subprocess.run([str(exe), "-hide_banner", "-nostdin", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    return result.stdout + result.stderr


def validate_binary(exe: Path, runner: Callable[..., str] = _run) -> str:
    """Run the already hash-verified binary: LGPL only, no ``--enable-gpl``, the Storyboard filters present.

    Returns the version text. Raises ``FetchError`` otherwise.
    """
    try:
        version = runner(exe, "-version")
        licence = runner(exe, "-L")
        buildconf = runner(exe, "-buildconf")
        filters = runner(exe, "-filters")
    except (OSError, subprocess.SubprocessError) as exc:
        raise FetchError(f"The downloaded ffmpeg does not run on this machine ({exc}). Nothing was installed.") from exc
    match = re.search(r"ffmpeg version (\S+)", version)
    if not match:
        raise FetchError("The downloaded file does not look like ffmpeg. Nothing was installed.")
    if "Lesser General Public" not in licence[:600] or "--enable-gpl" in version + buildconf or "--enable-nonfree" in version + buildconf:
        raise FetchError("The downloaded ffmpeg is not an LGPL build, so it was refused. Nothing was installed.")
    present = set(re.findall(r"^\s*[T.][S.][C.]\s+(\w+)\s", filters, re.MULTILINE))
    missing = sorted(REQUIRED_FILTERS - present)
    if missing:
        raise FetchError(f"The downloaded ffmpeg lacks filters Storyboard needs ({', '.join(missing)}). Nothing was installed.")
    return match.group(1)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def switch_to(root: Path, version: str, staged: Path) -> Path:
    """Move ``staged`` (a folder holding the binaries) to ``root/version`` and point ``current`` at it."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / version
    previous = None
    if target.exists():
        previous = root / f".partial-old-{secrets.token_hex(4)}"
        os.replace(target, previous)
    try:
        os.replace(staged, target)
        _write_atomic(root / "current", version)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        if previous is not None:
            os.replace(previous, target)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)
    for entry in root.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != version:
            shutil.rmtree(entry, ignore_errors=True)
    return target


def _finish(pin: Pin, archive: Path, work: Path, root: Path, progress: Progress,
            runner: Callable[..., str], actual_hash: str | None = None) -> Path:
    progress("verify", 0, 1)
    _check_hash(pin, actual_hash or sha256_of(archive))
    progress("extract", 0, 1)
    staged = work / "staged"
    written = extract_members(archive, pin, staged)
    exe = next(path for path in written if path.name.lower() in ("ffmpeg", "ffmpeg.exe"))
    validate_binary(exe, runner)
    progress("install", 0, 1)
    switch_to(root, pin.version, staged)
    progress("install", 1, 1)
    return root / pin.version / exe.name


def install_pin(pin: Pin, root: Path, progress: Progress, cancel: threading.Event | None = None, *,
                opener: Callable[..., IO[bytes]] | None = None,
                runner: Callable[..., str] = _run,
                free_bytes: Callable[[Path], int] | None = None) -> Path:
    """Download ``pin`` and install it under ``root``. Returns the installed ffmpeg path."""
    root.mkdir(parents=True, exist_ok=True)
    cleanup_partials(root)
    free = (free_bytes or (lambda p: shutil.disk_usage(p).free))(root)
    if free < pin.size * 3:
        raise FetchError(f"Not enough free disk space: about {pin.size * 3 // 1_000_000} MB is needed.")
    work = root / f".partial-{secrets.token_hex(4)}"
    work.mkdir()
    try:
        archive = work / pin.asset
        actual = download(pin, archive, progress, cancel, opener)
        return _finish(pin, archive, work, root, progress, runner, actual)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install_from_file(pin: Pin, source: Path, root: Path, progress: Progress, *,
                      runner: Callable[..., str] = _run) -> Path:
    """Offline path: the user fetched the pinned archive elsewhere. The same hash and checks apply."""
    if not source.is_file():
        raise FetchError("That file was not found.")
    root.mkdir(parents=True, exist_ok=True)
    cleanup_partials(root)
    work = root / f".partial-{secrets.token_hex(4)}"
    work.mkdir()
    try:
        archive = work / pin.asset
        shutil.copyfile(source, archive)
        return _finish(pin, archive, work, root, progress, runner)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def validate_custom(path: Path, runner: Callable[..., str] = _run) -> str:
    """A user-chosen ffmpeg: must run and say ``ffmpeg version`` (it is not licence-checked: shown as not verified)."""
    if not path.is_file():
        raise FetchError("That file was not found.")
    try:
        text = runner(path, "-version")
    except (OSError, subprocess.SubprocessError) as exc:
        raise FetchError(f"That file does not run ({exc}).") from exc
    match = re.search(r"ffmpeg version (\S+)", text)
    if not match:
        raise FetchError("That file does not look like ffmpeg.")
    return match.group(1)
