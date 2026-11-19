"""A Windows path over 260 characters answers with a plain sentence, not a 500 (rtx4090 handover, section 3)."""

from __future__ import annotations

import unittest

from backend_service.routes.storyboard import _http


class PathTooLongTests(unittest.TestCase):
    def test_winerror_206_and_3_become_a_readable_400(self) -> None:
        for code in (206, 3):
            exc = OSError(2, "The filename or extension is too long")
            exc.winerror = code  # only set by Windows' own OSError; set here so the test runs anywhere
            found = _http(exc)
            self.assertEqual(found.status_code, 400)
            self.assertIn("too long for Windows", found.detail)
            self.assertIn("Settings > Storage", found.detail)

    def test_other_os_errors_are_still_raised(self) -> None:
        with self.assertRaises(OSError):
            _http(OSError(13, "Permission denied"))


if __name__ == "__main__":
    unittest.main()
