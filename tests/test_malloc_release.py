"""Freed system memory goes back to the OS after Unload on Linux."""
import sys
import unittest
from unittest import mock

from backend_service.helpers import malloc_release


class ReleaseFreedMemoryTests(unittest.TestCase):
    def test_off_linux_it_does_nothing(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertFalse(malloc_release.release_freed_memory())

    def test_a_libc_without_malloc_trim_is_not_an_error(self):
        with mock.patch.object(sys, "platform", "linux"), mock.patch("ctypes.CDLL", side_effect=OSError("no libc")):
            self.assertFalse(malloc_release.release_freed_memory())

    def test_it_calls_malloc_trim_with_zero(self):
        libc = mock.Mock()
        libc.malloc_trim.return_value = 1
        with mock.patch.object(sys, "platform", "linux"), mock.patch("ctypes.CDLL", return_value=libc):
            self.assertTrue(malloc_release.release_freed_memory())
        libc.malloc_trim.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
