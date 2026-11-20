"""Dashboard first-screen copy (overnight review, stream F): the recommendation names the real format.

On an MLX Mac the title said "70B class @ Native f16" (the KV cache label read as the weights' precision; a 70B at
f16 does not fit in 64 GB), and the activity feed said "Recommended target: 70B GGUF" because it was written from the
boot snapshot, before the MLX probe had finished.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend_service.helpers.system import _best_fit_recommendation


def _system(**overrides):
    from tests.test_backend_service import fake_system_snapshot

    snapshot = fake_system_snapshot()
    snapshot.update(overrides)
    return snapshot


class RecommendationTitleTests(unittest.TestCase):
    def test_mlx_mac_title_names_4bit_not_the_cache(self):
        rec = _best_fit_recommendation(_system(mlxUsable=True, totalMemoryGb=64.0))
        self.assertEqual(rec["title"], "Recommended target: 70B class at 4-bit")
        self.assertNotIn("f16", rec["title"])
        self.assertEqual(rec["titleKey"], "recommendation.mlxTitle")
        self.assertEqual(rec["cacheLabel"], "Native f16")  # still reported, just not in the title

    def test_machine_without_mlx_keeps_the_gguf_title(self):
        rec = _best_fit_recommendation(_system(platform="Linux", arch="x86_64", mlxUsable=False, totalMemoryGb=32.0))
        self.assertEqual(rec["title"], "Recommended target: 14B GGUF")

    def test_a_pc_with_its_own_graphics_card_is_judged_by_the_card(self):
        # FU-452: 64 GB of RAM said "70B GGUF" on a PC whose 24 GB card holds a 27B at 4-bit at most.
        pc = dict(platform="Windows", arch="AMD64", mlxUsable=False, totalMemoryGb=63.8, gpuUnifiedMemory=False)
        self.assertEqual(_best_fit_recommendation(_system(gpuVramTotalGb=23.99, **pc))["targetModel"], "14B")
        self.assertEqual(_best_fit_recommendation(_system(gpuVramTotalGb=None, **pc))["targetModel"], "70B")


class HardwareProfileActivityTests(unittest.TestCase):
    def _state(self, snapshots: list[dict]):
        from backend_service.state import ChaosEngineState
        from tests.test_backend_service import fake_library

        calls = iter(snapshots)
        last: dict = {}

        def provider(**_kwargs):
            nonlocal last
            last = next(calls, last)
            return last

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return ChaosEngineState(
            system_snapshot_provider=provider,
            library_provider=fake_library,
            settings_path=Path(tmp.name) / "settings.json",
            benchmarks_path=Path(tmp.name) / "benchmarks.json",
            chat_sessions_path=Path(tmp.name) / "chat.json",
        )

    def test_activity_uses_the_snapshot_after_the_probe(self):
        from tests.test_backend_service import FakeRuntime

        boot = _system(mlxUsable=False, totalMemoryGb=64.0)  # probe not finished yet
        ready = _system(mlxAvailable=True, mlxLmAvailable=True, mlxUsable=True, totalMemoryGb=64.0)
        state = self._state([boot, ready])
        state.runtime = FakeRuntime()
        state.runtime.capabilities = SimpleNamespace(probing=False)
        state.activity.clear()
        state._announce_backend_readiness_when_known(ready)
        details = {row["title"]: row["detail"] for row in state.activity}
        # FU-456: the line names the hardware the fit verdicts are judged on, not a size-class target that
        # could disagree with the Dashboard's best-fit row.
        from backend_service.helpers.hardware_class import hardware_class

        self.assertEqual(details["Hardware profile refreshed"], hardware_class(ready).label)
        self.assertNotIn("GGUF", details["Hardware profile refreshed"])
        self.assertNotIn("Recommended target", details["Hardware profile refreshed"])
        self.assertIn("MLX usable: yes", details["Backend readiness"])

    def test_a_pc_is_not_told_three_times_that_mlx_is_missing(self):
        from tests.test_backend_service import FakeRuntime

        pc = _system(platform="Windows", arch="AMD64", mlxUsable=False, ggufAvailable=True, totalMemoryGb=64.0)
        state = self._state([pc])
        state.runtime = FakeRuntime()
        state.runtime.capabilities = SimpleNamespace(probing=False)
        state.activity.clear()
        state._announce_backend_readiness_when_known(pc)
        row = next(row for row in state.activity if row["title"] == "Backend readiness")
        self.assertEqual(row["detail"], "GGUF runtime: yes")
        self.assertEqual(row["detailKey"], "activity.backendReadiness.detailGguf")
        self.assertNotIn("MLX", row["detail"])


if __name__ == "__main__":
    unittest.main()
