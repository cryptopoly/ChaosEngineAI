"""Windows: a child of the backend ends when the backend is killed from outside (rtx4090, 2026-11-19)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
import unittest


@unittest.skipUnless(sys.platform == "win32", "Job Objects are Windows only")
class KillOnCloseJobTests(unittest.TestCase):
    def test_a_grandchild_ends_when_the_parent_is_killed_hard(self) -> None:
        script = textwrap.dedent("""
            import subprocess, sys, time
            from backend_service.windows_job import attach_kill_on_close_job
            assert attach_kill_on_close_job()
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            print(child.pid, flush=True)
            time.sleep(120)
        """)
        parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(parent.stdout.readline())
            self.assertTrue(self._alive(child_pid))
            subprocess.run(["taskkill", "/F", "/PID", str(parent.pid)], capture_output=True, check=True)
            for _ in range(50):
                if not self._alive(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(self._alive(child_pid), "the child outlived its killed parent")
        finally:
            parent.kill()
            parent.wait()

    @staticmethod
    def _alive(pid: int) -> bool:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out

    def test_other_platforms_do_nothing(self) -> None:
        from backend_service import windows_job

        if sys.platform != "win32":
            self.assertFalse(windows_job.attach_kill_on_close_job())


if __name__ == "__main__":
    unittest.main()
