import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend_service.services import ServiceCoordinator
from backend_service.services.log_service import LogService
from backend_service.services.download_service import DownloadService


class LogServiceTests(unittest.TestCase):
    def setUp(self):
        self.log_svc = LogService(maxlen=100)

    def test_add_log_returns_entry(self):
        entry = self.log_svc.add_log("test", "info", "Hello log")
        self.assertEqual(entry["source"], "test")
        self.assertEqual(entry["level"], "info")
        self.assertEqual(entry["message"], "Hello log")
        self.assertIn("timestamp", entry)

    def test_add_log_with_extra(self):
        entry = self.log_svc.add_log("test", "debug", "Details", extra={"key": "value"})
        self.assertEqual(entry["extra"]["key"], "value")

    def test_get_logs_returns_all(self):
        self.log_svc.add_log("src1", "info", "msg1")
        self.log_svc.add_log("src2", "error", "msg2")
        self.log_svc.add_log("src1", "warn", "msg3")
        logs = self.log_svc.get_logs()
        self.assertEqual(len(logs), 3)

    def test_get_logs_filter_by_level(self):
        self.log_svc.add_log("src", "info", "msg1")
        self.log_svc.add_log("src", "error", "msg2")
        self.log_svc.add_log("src", "info", "msg3")
        logs = self.log_svc.get_logs(level="error")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["message"], "msg2")

    def test_get_logs_filter_by_source(self):
        self.log_svc.add_log("api", "info", "msg1")
        self.log_svc.add_log("runtime", "info", "msg2")
        self.log_svc.add_log("api", "info", "msg3")
        logs = self.log_svc.get_logs(source="api")
        self.assertEqual(len(logs), 2)

    def test_get_logs_with_limit(self):
        for i in range(10):
            self.log_svc.add_log("src", "info", f"msg{i}")
        logs = self.log_svc.get_logs(limit=3)
        self.assertEqual(len(logs), 3)
        # Should return the last 3
        self.assertEqual(logs[0]["message"], "msg7")

    def test_clear_removes_all_logs(self):
        self.log_svc.add_log("src", "info", "msg")
        self.log_svc.clear()
        self.assertEqual(len(self.log_svc.get_logs()), 0)

    def test_maxlen_enforced(self):
        svc = LogService(maxlen=5)
        for i in range(10):
            svc.add_log("src", "info", f"msg{i}")
        logs = svc.get_logs()
        self.assertEqual(len(logs), 5)
        # Oldest entries should have been evicted
        self.assertEqual(logs[0]["message"], "msg5")


class DownloadServiceTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(
            _downloads={},
            _download_cancel={},
            _download_processes={},
            _download_tokens={},
        )
        self.dl_svc = DownloadService(self.state)

    def test_start_download(self):
        entry = self.dl_svc.start_download("org/model-v1")
        self.assertEqual(entry["repo"], "org/model-v1")
        self.assertEqual(entry["status"], "downloading")
        self.assertEqual(entry["progress"], 0.0)

    def test_get_status(self):
        self.dl_svc.start_download("org/model-v1")
        status = self.dl_svc.get_status("org/model-v1")
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "downloading")

    def test_get_status_nonexistent(self):
        self.assertIsNone(self.dl_svc.get_status("nonexistent"))

    def test_get_all_downloads(self):
        self.dl_svc.start_download("org/model-a")
        self.dl_svc.start_download("org/model-b")
        all_dl = self.dl_svc.get_all_downloads()
        self.assertEqual(len(all_dl), 2)
        self.assertIn("org/model-a", all_dl)
        self.assertIn("org/model-b", all_dl)

    def test_active_repos(self):
        self.dl_svc.start_download("org/model-a")
        self.dl_svc.start_download("org/model-b")
        self.dl_svc.mark_complete("org/model-b")
        active = self.dl_svc.active_repos
        self.assertEqual(active, ["org/model-a"])

    def test_cancel_download(self):
        self.dl_svc.start_download("org/model-a")
        result = self.dl_svc.cancel_download("org/model-a")
        self.assertTrue(result)
        status = self.dl_svc.get_status("org/model-a")
        self.assertEqual(status["status"], "cancelled")

    def test_cancel_nonexistent(self):
        self.assertFalse(self.dl_svc.cancel_download("nonexistent"))

    def test_mark_complete(self):
        self.dl_svc.start_download("org/model-a")
        self.dl_svc.mark_complete("org/model-a")
        status = self.dl_svc.get_status("org/model-a")
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["progress"], 100.0)

    def test_mark_complete_with_error(self):
        self.dl_svc.start_download("org/model-a")
        self.dl_svc.mark_complete("org/model-a", error="Network failure")
        status = self.dl_svc.get_status("org/model-a")
        self.assertEqual(status["status"], "error")
        self.assertEqual(status["error"], "Network failure")

    def test_remove(self):
        self.dl_svc.start_download("org/model-a")
        self.dl_svc.remove("org/model-a")
        self.assertIsNone(self.dl_svc.get_status("org/model-a"))
        self.assertEqual(len(self.dl_svc.get_all_downloads()), 0)


class ServiceCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.mock_state = SimpleNamespace(
            runtime=MagicMock(),
            settings={"preferredServerPort": 8080},
            chat_sessions=[],
            benchmark_runs=[],
            server_port=8080,
            active_requests=0,
            requests_served=42,
            _downloads={},
            _download_cancel={},
            _download_processes={},
            _download_tokens={},
            custom_attr="custom_value",
        )
        self.coordinator = ServiceCoordinator(self.mock_state)

    def test_runtime_delegation(self):
        self.assertIs(self.coordinator.runtime, self.mock_state.runtime)

    def test_settings_delegation(self):
        self.assertEqual(self.coordinator.settings["preferredServerPort"], 8080)

    def test_settings_setter(self):
        self.coordinator.settings = {"new": True}
        self.assertEqual(self.mock_state.settings, {"new": True})

    def test_chat_sessions_delegation(self):
        self.assertEqual(self.coordinator.chat_sessions, [])

    def test_requests_served_delegation(self):
        self.assertEqual(self.coordinator.requests_served, 42)

    def test_getattr_fallback(self):
        # custom_attr is not explicitly defined on ServiceCoordinator
        self.assertEqual(self.coordinator.custom_attr, "custom_value")

    def test_log_service_available(self):
        self.assertIsInstance(self.coordinator.log, LogService)

    def test_downloads_service_available(self):
        self.assertIsInstance(self.coordinator.downloads, DownloadService)

    def test_log_service_functional(self):
        entry = self.coordinator.log.add_log("test", "info", "msg")
        logs = self.coordinator.log.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["message"], "msg")

    def test_downloads_service_functional(self):
        self.coordinator.downloads.start_download("org/model")
        all_dl = self.coordinator.downloads.get_all_downloads()
        self.assertIn("org/model", all_dl)


if __name__ == "__main__":
    unittest.main()
