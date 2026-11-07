"""Long renders: keep the machine awake while one runs, and tell the user when a long one ends.

Both are best effort and never raise: a missing helper program just means no keep-awake or no banner.
The app shell has no notification plugin, so the backend uses what each system already has: ``caffeinate``
(bound to this process, so it ends if the backend is killed) on macOS, ``SetThreadExecutionState`` on
Windows, ``systemd-inhibit`` on Linux; ``osascript`` or ``notify-send`` for the banner. Every program is
started as an argument list; the text of the banner is passed as arguments, never put into a script.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import subprocess
import sys
from typing import Iterator

LONG_RENDER_SECONDS = 30.0  # a render shorter than this ends without a banner: you are still looking at it

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

_MAC_SCRIPT = ["on run argv", "display notification (item 2 of argv) with title (item 1 of argv)", "end run"]


def _spawn(argv: list[str]) -> subprocess.Popen[bytes] | None:
    try:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


@contextlib.contextmanager
def keep_awake(reason: str) -> Iterator[bool]:
    """Stop the system from sleeping while the block runs. Yields whether anything is holding it awake."""
    helper: subprocess.Popen[bytes] | None = None
    windows = False
    try:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            helper = _spawn(["caffeinate", "-i", "-w", str(os.getpid())])
        elif sys.platform == "win32":
            # The flag belongs to the calling thread, which is the render thread that also clears it.
            windows = bool(ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))  # type: ignore[attr-defined]
        elif sys.platform.startswith("linux") and shutil.which("systemd-inhibit") and shutil.which("tail"):
            helper = _spawn(["systemd-inhibit", "--what=idle:sleep", "--who=ChaosEngineAI", f"--why={reason}",
                             "tail", f"--pid={os.getpid()}", "-f", "/dev/null"])
    except Exception:  # noqa: BLE001 - best effort
        helper, windows = None, False
    try:
        yield helper is not None or windows
    finally:
        if windows:
            with contextlib.suppress(Exception):
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)  # type: ignore[attr-defined]
        if helper is not None:
            with contextlib.suppress(Exception):
                helper.terminate()
                helper.wait(timeout=5)


def notify(title: str, body: str) -> bool:
    """Show a system banner. Returns whether a helper accepted it (it may still be hidden by Focus modes)."""
    title, body = title[:100], body[:200]
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            argv = ["osascript"]
            for line in _MAC_SCRIPT:
                argv += ["-e", line]
            return subprocess.run([*argv, title, body], stdin=subprocess.DEVNULL, capture_output=True, timeout=10).returncode == 0
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            return subprocess.run(["notify-send", "--app-name=ChaosEngineAI", "--", title, body],
                                  stdin=subprocess.DEVNULL, capture_output=True, timeout=10).returncode == 0
    except Exception:  # noqa: BLE001 - best effort
        return False
    return False
