"""The backend turns off the Hugging Face libraries' usage reporting before anything imports them."""
from __future__ import annotations

import subprocess
import sys
import unittest


class HfTelemetryOffTests(unittest.TestCase):
    def _run(self, env_value: str | None) -> str:
        code = (
            "import os, backend_service.app; "
            "print(os.environ.get('HF_HUB_DISABLE_TELEMETRY'))"
        )
        import os

        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_TELEMETRY"}
        if env_value is not None:
            env["HF_HUB_DISABLE_TELEMETRY"] = env_value
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        return out.stdout.strip().splitlines()[-1]

    def test_importing_the_app_turns_it_off(self) -> None:
        self.assertEqual(self._run(None), "1")

    def test_a_user_value_is_kept(self) -> None:
        self.assertEqual(self._run("0"), "0")


if __name__ == "__main__":
    unittest.main()
