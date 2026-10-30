import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service.helpers.system import _get_top_memory_for_pid, _parse_top_mem_value
from backend_service.helpers.system_processes import _list_llm_processes


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
        with mock.patch("backend_service.helpers.system_processes.platform.system", return_value="Darwin"):
            with mock.patch("backend_service.helpers.system_processes.subprocess.run", return_value=completed):
                self.assertAlmostEqual(_get_top_memory_for_pid(404), 1377 / 1024, places=4)


class RunningProcessListTests(unittest.TestCase):
    """The Dashboard process table lists model workers, not whatever mentions one."""

    @staticmethod
    def _process(pid: int, name: str, cmdline: list[str]):
        return SimpleNamespace(info={"pid": pid, "name": name, "cmdline": cmdline})

    def _listed(self, processes):
        described = lambda pid, **_: {"pid": pid, "name": "p", "owner": "System", "memoryGb": 1.0, "cpuPercent": 0.0, "kind": "other"}
        with mock.patch("backend_service.helpers.system_processes.platform.system", return_value="Linux"), \
                mock.patch("backend_service.helpers.system_processes.psutil.process_iter", return_value=processes), \
                mock.patch("backend_service.helpers.system_processes._describe_process", side_effect=described):
            return [item["pid"] for item in _list_llm_processes()]

    def test_model_workers_are_listed(self):
        self.assertEqual(
            self._listed([
                self._process(1, "Python", ["/x/Python", "-m", "backend_service.mlx_worker"]),
                self._process(2, "llama-server", ["/x/llama-server", "--port", "9000"]),
            ]),
            [1, 2],
        )

    def test_a_program_that_only_mentions_a_worker_or_an_other_agent_app_is_not_listed(self):
        self.assertEqual(
            self._listed([
                self._process(10, "claude.exe", ["/Users/me/.openclaw/tools/node/bin/claude.exe", "--chrome-native-host"]),
                self._process(11, "claude", ["claude", "--append-system-prompt", "restart the mlx_worker and llama-server"]),
                self._process(12, "node", ["node", "script.js", "--note", "llama-cli"]),
            ]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
