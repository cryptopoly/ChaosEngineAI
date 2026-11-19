"""FU-255: a Windows path with single backslashes in a tool call gets advice that fixes it."""

from __future__ import annotations

import unittest

from backend_service.agent_call_repair import _lone_backslash, unreadable_call


class WindowsPathTests(unittest.TestCase):
    def test_lone_backslashes_are_found_and_valid_escapes_are_not(self) -> None:
        self.assertTrue(_lone_backslash(r'{"path": "C:\Users\me"}'))
        self.assertFalse(_lone_backslash('{"path": "C:' + "\\\\" + 'Users", "t": "a\\nb \\u00e9 \\"q\\" a\\/b"}'))

    def test_the_advice_names_the_backslash_not_a_missing_brace(self) -> None:
        text = r'<tool_call>{"name": "speak", "arguments": {"text": "hi", "path": "C:\Users\me\a.wav"}}</tool_call>'
        found = unreadable_call(text, {"speak"})
        self.assertIsNotNone(found)
        self.assertIn("backslash", found.message)
        self.assertNotIn("closing brace", found.message)


if __name__ == "__main__":
    unittest.main()
