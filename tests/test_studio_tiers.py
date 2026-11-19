"""FU-241: the tier definition for a single image or clip in Chat."""

from __future__ import annotations

import unittest
from unittest import mock

from backend_service.tools.studio_tiers import apply_tier, tier_from_text, tier_size

CLIP = {"id": "v", "repo": "x/y", "recommendedResolution": "768x512", "maxSide": 1280}


class TierTests(unittest.TestCase):
    def test_words_that_name_a_tier(self) -> None:
        self.assertEqual(tier_from_text("make a quick draft of a fox"), "draft")
        self.assertEqual(tier_from_text("final quality please"), "final")
        self.assertEqual(tier_from_text("a quick draft, then the final"), "final")
        self.assertIsNone(tier_from_text("a fox in the snow"))
        self.assertIsNone(tier_from_text("the draftsman"))

    def test_sizes_per_tier(self) -> None:
        self.assertEqual(tier_size("standard", CLIP), (768, 512))
        self.assertEqual(tier_size("draft", CLIP), (384, 256))
        self.assertEqual(tier_size("final", CLIP), (1152, 768))
        capped = tier_size("final", {**CLIP, "maxSide": 1024})
        self.assertLessEqual(max(capped), 1024)
        self.assertIsNone(tier_size("draft", {"id": "x"}))
        draft, standard, final = (tier_size(t, CLIP) for t in ("draft", "standard", "final"))
        self.assertTrue(draft[0] * draft[1] <= standard[0] * standard[1] <= final[0] * final[1])

    def _apply(self, tier, name, args):
        with mock.patch("backend_service.mcp_bridge.tools._recommended", return_value={"steps": 30}):
            return apply_tier(tier, name, args, CLIP)

    def test_a_call_above_the_tier_runs_at_the_tier_with_a_note(self) -> None:
        args, notes = self._apply("standard", "generate_video",
                                  {"prompt": "x", "width": 1280, "height": 720, "fps": 48, "steps": 60, "seconds": 3})
        self.assertEqual((args["width"], args["height"], args["fps"], args["steps"]), (768, 512, 24, 30))
        self.assertEqual(args["seconds"], 3)  # length is content
        self.assertEqual(len(notes), 3)

    def test_a_smaller_call_is_left_alone_and_standard_without_a_size_adds_none(self) -> None:
        args, notes = self._apply("standard", "generate_video", {"prompt": "x", "width": 512, "height": 320})
        self.assertEqual((args["width"], args["height"], notes), (512, 320, []))
        args, notes = self._apply("standard", "generate_video", {"prompt": "x"})
        self.assertNotIn("width", args)

    def test_draft_and_final_name_their_size_when_the_call_gave_none(self) -> None:
        args, notes = self._apply("draft", "generate_video", {"prompt": "x"})
        self.assertEqual((args["width"], args["height"]), (384, 256))
        self.assertEqual(len(notes), 1)

    def test_other_tools_are_untouched(self) -> None:
        args, notes = self._apply("draft", "speak", {"text": "hi"})
        self.assertEqual((args, notes), ({"text": "hi"}, []))


if __name__ == "__main__":
    unittest.main()


class ExpertTests(unittest.TestCase):
    def _plan(self):
        from backend_service.tools import studio_gate

        return studio_gate.PendingPlan(
            id="p", session_id="s", memory=None,
            items=[{"callId": "c1", "tool": "generate_video"}, {"callId": "c2", "tool": "speak"}],
            call_args={"c1": {"prompt": "waves"}, "c2": {"text": "hi"}})

    def test_valid_expert_fields_pass_and_empty_ones_are_dropped(self) -> None:
        from backend_service.tools.studio_gate import clean_overrides

        got = clean_overrides(self._plan(), {"c1": {"width": 1280, "height": 704, "fps": 30, "steps": "", "seed": None}}, None)
        self.assertEqual(got, {"c1": {"width": 1280, "height": 704, "fps": 30}})

    def test_bad_expert_fields_are_refused_with_a_sentence(self) -> None:
        from backend_service.tools.studio_gate import clean_overrides

        plan = self._plan()
        for bad, text in [({"c1": {"width": 1280}}, "both the width and the height"),
                          ({"c1": {"width": 99999, "height": 704}}, "Invalid Expert setting"),
                          ({"c1": {"prompt": "x"}}, "no Expert setting 'prompt'"),
                          ({"c2": {"seed": 1}}, "only apply to an image or a clip"),
                          ({"c1": {"fps": 500}}, "Invalid Expert setting")]:
            with self.assertRaises(ValueError) as caught:
                clean_overrides(plan, bad, None)
            self.assertIn(text, str(caught.exception))

    def test_typed_settings_run_above_the_tier_and_a_model_call_does_not(self) -> None:
        from backend_service.tools.studio_tiers import TierHolder

        holder = TierHolder("standard")
        typed = {"prompt": "x", "width": 1280, "height": 720}
        holder.exempt.add(__import__("json").dumps(typed, sort_keys=True))
        self.assertTrue(holder.exempts(typed))
        self.assertFalse(holder.exempts({"prompt": "x", "width": 1920, "height": 1080}))
