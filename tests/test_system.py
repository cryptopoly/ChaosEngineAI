import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service.helpers.system import _get_top_memory_for_pid, _parse_top_mem_value


class TopMemoryParsingTests(unittest.TestCase):
    def test_parse_top_mem_value_handles_gigabytes_with_suffix(self):
        self.assertEqual(_parse_top_mem_value("85G+"), 85.0)

    def test_parse_top_mem_value_handles_megabytes(self):
        self.assertAlmostEqual(_parse_top_mem_value("1172M"), 1172 / 1024, places=4)

    def test_get_top_memory_for_pid_parses_single_pid_output(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="PID  MEM\n404  1377M\n",
        )
        with mock.patch("backend_service.helpers.system.platform.system", return_value="Darwin"):
            with mock.patch("backend_service.helpers.system.subprocess.run", return_value=completed):
                self.assertAlmostEqual(_get_top_memory_for_pid(404), 1377 / 1024, places=4)


if __name__ == "__main__":
    unittest.main()
