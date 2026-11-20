"""The Dashboard's best-fit chat row and Chat Models' short list (FU-456, FU-457, D-105 B and D).

Both come from ``helpers.chat_picks`` and read the ``fit`` the Models page shows, so the three cannot disagree.
"""

from __future__ import annotations

import unittest

from backend_service.helpers.chat_picks import apply_chat_picks, chat_picks
from backend_service.helpers.hardware_class import HardwareClass
from backend_service.helpers.system import _best_fit_recommendation

CARD_24 = HardwareClass(kind="cuda", bucket="cuda-24", label="NVIDIA GPU, 24 GB VRAM", total_gb=64.0, usable_gb=24.0, unified=False)


def _variant(vid: str, *, status: str = "fits", release: str = "2026-06", params: float = 9.0, need: float = 8.0,
             size: float = 6.0, local: bool = False) -> dict:
    return {
        "id": vid, "name": vid.upper(), "repo": f"org/{vid}", "paramsB": params, "sizeGb": size,
        "format": "GGUF", "quantization": "Q4_K_M", "releaseDate": release, "availableLocally": local,
        "fit": {"status": status, "needGb": need, "usableGb": 24.0, "headroomPct": round((24 - need) / 24 * 100),
                "reason": "x", "basis": "estimated"},
    }


def _family(fid: str, *variants: dict, best: str | None = None) -> dict:
    return {"id": fid, "name": fid.title(), "variants": list(variants), "bestFitVariantId": best}


def _picks(families, **kwargs):
    return chat_picks(families, CARD_24, bandwidth_gbps=lambda: 1000.0, **kwargs)


class ChatPickTests(unittest.TestCase):
    def test_only_each_familys_best_fit_row_that_fits(self):
        families = [
            _family("a", _variant("a1"), _variant("a2", release="2026-10"), best="a1"),
            _family("b", _variant("b1", status="tight"), best="b1"),  # tight is never recommended
            _family("c", _variant("c1"), best=None),  # no best fit in this family
        ]
        self.assertEqual([p["variantId"] for p in _picks(families)], ["a1"])

    def test_order_is_newest_then_larger_then_faster(self):
        families = [
            _family("old", _variant("old", release="2025-01", params=30), best="old"),
            _family("new-small", _variant("new-small", release="2026-10", params=4), best="new-small"),
            _family("new-big", _variant("new-big", release="2026-10", params=27, need=12), best="new-big"),
        ]
        self.assertEqual([p["variantId"] for p in _picks(families)], ["new-big", "new-small", "old"])

    def test_measured_speed_wins_over_the_estimate(self):
        families = [_family("a", _variant("a"), best="a")]
        measured = _picks(families, measured_speed_for=lambda v: 233.4)[0]
        self.assertEqual((measured["tokS"], measured["tokSBasis"]), (233.4, "measured"))
        estimated = _picks(families, measured_speed_for=lambda v: None)[0]
        self.assertEqual(estimated["tokSBasis"], "estimated")
        self.assertGreater(estimated["tokS"], 0)

    def test_limit(self):
        families = [_family(f"f{i}", _variant(f"v{i}"), best=f"v{i}") for i in range(9)]
        self.assertEqual(len(_picks(families)), 5)


class DashboardRecommendationTests(unittest.TestCase):
    def _system(self, **overrides):
        from tests.test_backend_service import fake_system_snapshot

        snapshot = fake_system_snapshot()
        snapshot.update(dict(platform="Windows", arch="AMD64", mlxUsable=False, totalMemoryGb=64.0,
                             gpuVramTotalGb=24.0, gpuUnifiedMemory=False))
        snapshot.update(overrides)
        return snapshot

    def test_the_headroom_is_the_picked_rows_own_not_a_constant(self):
        system = self._system()
        rec = _best_fit_recommendation(system)
        self.assertIsNone(rec["headroomPercent"])  # no constant per size class any more
        families = [_family("a", _variant("a", need=10.4), best="a")]
        rec = apply_chat_picks(rec, families, system)
        self.assertEqual(rec["bestFit"]["variantId"], "a")
        self.assertEqual(rec["headroomPercent"], families[0]["variants"][0]["fit"]["headroomPct"])
        self.assertEqual(rec["targetModel"], "A")
        self.assertEqual(rec["chatPicks"][0]["fit"], families[0]["variants"][0]["fit"])  # the Models page's fit

    def test_nothing_fits_keeps_the_size_class_and_no_headroom(self):
        system = self._system()
        rec = apply_chat_picks(_best_fit_recommendation(system), [_family("a", _variant("a", status="wont-fit"), best=None)], system)
        self.assertIsNone(rec["bestFit"])
        self.assertEqual(rec["chatPicks"], [])
        self.assertIsNone(rec["headroomPercent"])
        self.assertEqual(rec["targetModel"], "14B")

    def test_the_real_catalogue_gives_a_fitting_pick_on_a_24_gb_card(self):
        from backend_service.helpers.model_family_payload import _model_family_payloads

        system = self._system()
        families = _model_family_payloads(system, [])
        picks = chat_picks(families, CARD_24, bandwidth_gbps=lambda: 1008.0)
        self.assertTrue(picks)
        for pick in picks:
            self.assertEqual(pick["fit"]["status"], "fits")
            family = next(f for f in families if f["id"] == pick["familyId"])
            self.assertEqual(family["bestFitVariantId"], pick["variantId"])


if __name__ == "__main__":
    unittest.main()
