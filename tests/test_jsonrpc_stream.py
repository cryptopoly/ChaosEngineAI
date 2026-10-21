"""The MLX worker pipe stays in step when a stream is abandoned early."""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from backend_service.inference.jsonrpc import JsonRpcProcess

# Streams five chunks then ``done``; any other request gets a result payload.
_FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "stream_generate":
            for i in range(5):
                print(json.dumps({"ok": True, "chunk": {"text": str(i)}}), flush=True)
            print(json.dumps({"ok": True, "done": True, "result": {"tokS": 1.0}}), flush=True)
        else:
            print(json.dumps({"ok": True, "result": {"echo": request.get("op")}}), flush=True)
    """
)


class AbandonedStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        script = Path(self._tmp.name) / "fake_worker.py"
        script.write_text(_FAKE_WORKER, encoding="utf-8")
        self.worker = JsonRpcProcess([sys.executable, str(script)], timeout=10)

    def tearDown(self) -> None:
        self.worker.close(force=True)
        self._tmp.cleanup()

    def test_request_after_an_abandoned_stream_gets_its_own_answer(self) -> None:
        stream = self.worker.stream_request({"op": "stream_generate"})
        self.assertEqual(next(stream)["chunk"]["text"], "0")
        stream.close()  # what a runaway guard or a dropped client does

        self.assertEqual(self.worker.request({"op": "update_profile"}), {"echo": "update_profile"})

    def test_a_finished_stream_leaves_nothing_behind(self) -> None:
        chunks = list(self.worker.stream_request({"op": "stream_generate"}))
        self.assertTrue(chunks[-1]["done"])
        self.assertEqual(self.worker.request({"op": "unload_model"}), {"echo": "unload_model"})


if __name__ == "__main__":
    unittest.main()
