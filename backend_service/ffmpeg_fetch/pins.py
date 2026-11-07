"""Pinned ffmpeg downloads for Windows and Linux (D-49: we build and host nothing).

The files are BtbN's LGPL builds (``lgpl`` variant: no ``--enable-gpl``, no x264/x265) from one month-end
release tag, so the files never move. Each row's ``sha256`` was copied once from the ``checksums.sha256``
asset of that release (2026-11-07); the installer checks it before it extracts anything and never trusts a
checksum fetched at run time. macOS has no row: Setup installs ``imageio-ffmpeg`` there (hardware H.264 via
VideoToolbox is chosen at encode time).

``PINS_VERIFIED`` is the switch for the whole download path. Re-verify and flip it when refreshing the pins:
every URL must answer HEAD 200 with the listed size (``scripts/audit-catalog-repos.py`` checks this at release
time) and every hash must match the release's ``checksums.sha256``.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

TAG = "autobuild-2026-10-31-12-59"
_BASE = f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{TAG}"
LICENCE_URL = "https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html"
SOURCE_URL = f"https://github.com/BtbN/FFmpeg-Builds/releases/tag/{TAG}"
VERSION = "9.0.2-39-g7adfbe40dc"

# Asset names, sizes and hashes checked against the live release on 2026-11-07 (GitHub API plus the
# release's checksums.sha256). The archive content itself was not re-hashed on a second machine.
PINS_VERIFIED = True


@dataclass(frozen=True)
class Pin:
    key: str
    asset: str
    sha256: str
    size: int
    archive: str  # "zip" or "tar.xz"
    version: str = VERSION
    licence_url: str = LICENCE_URL
    source_url: str = SOURCE_URL

    @property
    def url(self) -> str:
        return f"{_BASE}/{self.asset}"


PINS: dict[str, Pin] = {
    "windows-x86_64": Pin(
        "windows-x86_64", "ffmpeg-n9.0.2-39-g7adfbe40dc-win64-lgpl-9.0.zip",
        "9d37c564e0d01c18cde184a079fa6ef04d7d2783ada41da81d04165cc8f1a650", 168715328, "zip"),
    "windows-arm64": Pin(
        "windows-arm64", "ffmpeg-n9.0.2-39-g7adfbe40dc-winarm64-lgpl-9.0.zip",
        "d1981518b99a1c05d6b864200b1ac9459b09e2ee174feffe84b6e6bb0bfa79e2", 117554579, "zip"),
    "linux-x86_64": Pin(
        "linux-x86_64", "ffmpeg-n9.0.2-39-g7adfbe40dc-linux64-lgpl-9.0.tar.xz",
        "6046d6731b26f58887e953561551d254a6eec9fbca4100dfeab56e19af8328d5", 136027168, "tar.xz"),
    "linux-aarch64": Pin(
        "linux-aarch64", "ffmpeg-n9.0.2-39-g7adfbe40dc-linuxarm64-lgpl-9.0.tar.xz",
        "3d1f66827a73ce5a446312bc4924234dc1515726d3743e2df35c0177e1f82128", 116688828, "tar.xz"),
}


def host_key(system: str | None = None, machine: str | None = None) -> str | None:
    """``windows-x86_64`` style key for this machine, or None (macOS and anything unlisted)."""
    system = (system or sys.platform).lower()
    machine = (machine or platform.machine()).lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64" if machine in ("amd64", "x86_64") else None
    if arch is None:
        return None
    if system.startswith("win"):
        return f"windows-{arch}"
    if system.startswith("linux"):
        return "linux-aarch64" if arch == "arm64" else "linux-x86_64"
    return None


def pin_for_host(system: str | None = None, machine: str | None = None) -> Pin | None:
    """The pin to download on this machine; None when the path is off, unverified, or has no row."""
    if not PINS_VERIFIED:
        return None
    key = host_key(system, machine)
    return PINS.get(key) if key else None
