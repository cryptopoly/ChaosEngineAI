"""Keep going phase 2 (FU-191): free GPU memory while rendering. The swap rule and the unload, render, reload
sequence with fakes; the real CUDA path is not exercised here."""

import unittest
from types import SimpleNamespace

from backend_service.keep_going.gpu_swap import launch_fields, same_model, swap_decision
from tests.test_keep_going_controller import SID, Base, job


class SwapRuleTests(unittest.TestCase):
    def test_never_and_always(self):
        self.assertFalse(swap_decision("never", job_gb=99, free_gb=1, chat_gb=20).swap)
        self.assertTrue(swap_decision("always", job_gb=1, free_gb=99, chat_gb=1).swap)

    def test_auto_on_a_24_gb_card(self):
        # Qwen3.8-27B 4-bit holds about 16 GB; an LTX clip peaks about 18 GB; 6 GB free beside the model.
        self.assertEqual(swap_decision("auto", job_gb=18, free_gb=6, chat_gb=16).why, "swap")
        # A small image fits next to it: no swap.
        self.assertEqual(swap_decision("auto", job_gb=3, free_gb=6, chat_gb=16).why, "fits")
        # Too big even with the chat model gone: leave it to the job's own memory gate.
        self.assertEqual(swap_decision("auto", job_gb=40, free_gb=6, chat_gb=16).why, "too_big")

    def test_auto_on_a_64_gb_mac_rarely_swaps(self):
        self.assertFalse(swap_decision("auto", job_gb=20, free_gb=38, chat_gb=16).swap)

    def test_unknown_memory_never_swaps_on_auto(self):
        self.assertEqual(swap_decision("auto", job_gb=None, free_gb=6, chat_gb=16).why, "unknown")
        self.assertEqual(swap_decision("auto", job_gb=10, free_gb=None, chat_gb=16).why, "unknown")
        self.assertTrue(swap_decision("auto", job_gb=18, free_gb=6, chat_gb=None).swap)

    def test_launch_fields_restore_the_same_settings(self):
        loaded = SimpleNamespace(ref="q", name="Q", canonicalRepo="org/q", source="library", backend="llama.cpp",
                                 path="/m/q.gguf", cacheStrategy="turboquant", cacheBits=4, fp16Layers=2,
                                 fusedAttention=True, fitModelInMemory=False, contextTokens=32768,
                                 requestedContextTokens=65536, speculativeDecoding=True, treeBudget=8)
        fields = launch_fields(loaded)
        self.assertEqual((fields["cacheStrategy"], fields["cacheBits"], fields["contextTokens"], fields["treeBudget"]),
                         ("turboquant", 4, 65536, 8))
        self.assertTrue(same_model(loaded, fields))
        self.assertFalse(same_model(SimpleNamespace(ref="other", path=None), fields))
        self.assertIsNone(launch_fields(None))


class FakeSwap:
    def __init__(self, hooks):
        self.hooks = hooks
        self.free, self.chat, self.peak = 6.0, 16.0, 18.0
        self.calls = []
        self.reload_error = None
        self.on_render = None

    def free_gb(self):
        return self.free

    def chat_gb(self):
        return self.chat

    def job_gb(self, name, args):
        return self.peak

    def unload(self):
        self.calls.append("unload")
        return {"modelRef": "qwen", "modelName": "Qwen3.8-27B", "cacheBits": 4}

    def reload(self, fields):
        self.calls.append(("reload", fields["modelRef"], fields["cacheBits"]))
        if self.reload_error:
            raise RuntimeError(self.reload_error)

    def start_render(self, app_request, name, args):
        self.calls.append(("render", name, args.get("what")))
        if self.on_render:
            return self.on_render(name, args)
        self.hooks.jobs.append(job(f"j{len(self.hooks.jobs)}", "done", "video"))
        return self.hooks.jobs[-1].id, None


class SwapSequenceTests(Base):
    def make(self):
        from backend_service.keep_going.adapters import StoryboardAdapter
        from backend_service.keep_going.controller import KeepGoing

        self.swap = getattr(self, "swap", None) or FakeSwap(self.hooks)
        ctrl = KeepGoing(self.path, self.hooks, clock=self.clock, threaded=False, poll_seconds=0.01,
                         adapters=lambda name: StoryboardAdapter(reader=lambda pid: self.film), swap=self.swap)
        return ctrl

    def begin_swap(self, mode="auto"):
        self.ctrl.start(SID, app_request=object(), gpu_mode=mode)
        self.ctrl.on_user_turn(SID, {"studioTools": True}, "Make the ad.", object())

    def test_a_render_that_does_not_fit_runs_after_the_turn_with_the_model_unloaded(self):
        self.begin_swap()
        self.assertTrue(self.ctrl.defer_render(SID, "storyboard_generate", {"project_id": "p1", "what": "clip"}))
        self.assertTrue(self.ctrl.defer_render(SID, "make_music", {"prompt": "warm"}))  # joins the first
        self.assertEqual(self.swap.calls, [])  # nothing happens during the turn
        self.finish(text="Queued the clips and the music.")
        self.assertEqual(self.swap.calls, ["unload", ("render", "storyboard_generate", "clip"), ("render", "make_music", None),
                                           ("reload", "qwen", 4)])
        self.assertEqual(len(self.hooks.turns), 1)
        prompt = self.hooks.turns[0].prompt
        self.assertIn("ran while the chat model was unloaded", prompt)
        self.assertIn("video job j0 done", prompt)
        self.assertEqual(self.state().swap_phase, None)

    def test_never_and_fits_do_not_defer(self):
        self.begin_swap("never")
        self.assertFalse(self.ctrl.defer_render(SID, "generate_video", {}))
        self.ctrl.forget(SID)
        self.begin_swap("auto")
        self.swap.peak = 2.0
        self.assertFalse(self.ctrl.defer_render(SID, "generate_image", {}))
        self.ctrl.forget(SID)
        self.begin_swap("always")
        self.assertTrue(self.ctrl.defer_render(SID, "generate_image", {}))

    def test_outside_a_running_turn_nothing_is_deferred(self):
        self.assertFalse(self.ctrl.defer_render(SID, "generate_video", {}))
        self.ctrl.start(SID, app_request=object(), gpu_mode="always")
        self.assertFalse(self.ctrl.defer_render(SID, "generate_video", {}))  # armed, no turn yet

    def test_a_model_that_will_not_load_again_stops_the_run_once(self):
        self.begin_swap()
        self.swap.reload_error = "out of memory"
        self.ctrl.defer_render(SID, "generate_video", {})
        self.finish()
        run = self.state()
        self.assertEqual((run.state, run.stop_code), ("stopped", "gpu_reload_failed"))
        self.assertIn("out of memory", run.stop_reason)
        self.assertEqual([c for c in self.swap.calls if c == "unload"], ["unload"])
        self.assertEqual(len([c for c in self.swap.calls if isinstance(c, tuple) and c[0] == "reload"]), 1)
        self.assertEqual(self.hooks.turns, [])

    def test_stop_while_rendering_still_reloads_and_never_nudges(self):
        self.begin_swap()

        def render(name, args):
            self.ctrl.stop(SID)  # the user pressed Stop while the clip renders
            self.hooks.jobs.append(job("j9", "running"))
            return "j9", None

        self.swap.on_render = render
        self.ctrl.defer_render(SID, "generate_video", {})
        self.finish()
        self.assertEqual(self.swap.calls[-1], ("reload", "qwen", 4))
        self.assertEqual(self.state().stop_code, "user_stopped")
        self.assertEqual(self.hooks.turns, [])

    def test_a_render_that_cannot_start_is_reported_and_the_model_comes_back(self):
        self.begin_swap()
        self.swap.on_render = lambda name, args: (None, "No video model installed.")
        self.ctrl.defer_render(SID, "generate_video", {})
        self.finish()
        self.assertEqual(self.swap.calls[-1][0], "reload")
        self.assertIn("Last error (render): generate_video: No video model installed.", self.hooks.turns[0].prompt)

    def test_waits_for_the_render_jobs_before_reloading(self):
        self.begin_swap()
        polls = {"n": 0}
        real_jobs = self.hooks.chat_jobs

        def jobs():
            polls["n"] += 1
            if polls["n"] == 3:
                self.hooks.jobs = [job("j5", "done", "video")]
            return real_jobs()

        self.hooks.chat_jobs = jobs

        def render(name, args):
            self.hooks.jobs = [job("j5", "running", "video")]
            return "j5", None

        self.swap.on_render = render
        self.ctrl.defer_render(SID, "generate_video", {})
        self.finish()
        self.assertGreaterEqual(polls["n"], 3)
        self.assertEqual(self.swap.calls[-1][0], "reload")
        self.assertEqual(len(self.hooks.turns), 1)

    def test_a_stopped_run_drops_kept_renders(self):
        self.begin_swap()
        self.ctrl.defer_render(SID, "generate_video", {})
        self.ctrl.on_user_turn(SID, {"studioTools": True}, "wait, change it", object())
        self.assertEqual(self.state().deferred, [])
        self.assertEqual(self.swap.calls, [])

    def test_status_shows_the_swap(self):
        self.begin_swap()
        self.state().swap_phase = "rendering"
        self.assertEqual(self.ctrl.status(SID)["phase"], "gpu_swap")
        self.assertEqual(self.ctrl.status(SID)["swapPhase"], "rendering")


class StudioToolDeferTests(unittest.TestCase):
    def test_a_kept_render_never_reaches_the_handler(self):
        from backend_service.tools import StructuredToolOutput
        from backend_service.tools.studio import StudioTool

        seen = []
        tool = {"name": "generate_image", "description": "x",
                "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}
        studio = StudioTool(tool, lambda ctx, args: seen.append(args) or {"jobId": "j1"}, SimpleNamespace(app=None),
                            defer=lambda name, args: StructuredToolOutput(text="Queued", render_as="studio",
                                                                         data={"deferred": True}))
        out = studio.execute_structured(prompt="rose")
        self.assertEqual((out.text, seen), ("Queued", []))


if __name__ == "__main__":
    unittest.main()
