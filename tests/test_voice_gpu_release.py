"""Idle speech workers give the NVIDIA GPU back to image and video jobs, and the idle-unload setting reaches both
speech workers (FU-174). The live measurement is in docs/progress/rtx4090-progress.md; nothing here needs a GPU."""

from __future__ import annotations

import unittest
from unittest import mock

from backend_service import voice_client


class _Proc:
    """Stands in for a running worker process."""

    pid = 4242

    def poll(self):
        return None


def _worker(loaded: dict[str, str]) -> voice_client.VoiceWorker:
    worker = voice_client.VoiceWorker()
    worker._process = _Proc()  # noqa: SLF001
    worker.loaded = dict(loaded)
    worker._kill = mock.Mock(side_effect=lambda: setattr(worker, "_process", None))  # noqa: SLF001
    return worker


class ReleaseForMediaTests(unittest.TestCase):
    def _run(self, main, qwen, cuda=True):
        with mock.patch.object(voice_client, "_WORKER", main), mock.patch.object(voice_client, "_QWEN_WORKER", qwen), \
                mock.patch.object(voice_client.voice_env, "voice_uses_cuda", return_value=cuda):
            return voice_client.release_gpu_for_media()

    def test_an_idle_gpu_speech_model_is_released(self) -> None:
        main = _worker({"tts": "C:/voice/kokoro", "stt": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"})
        qwen = _worker({"tts": "C:/hf/Qwen3-TTS-12Hz-1.7B-CustomVoice"})
        released = self._run(main, qwen)
        self.assertEqual(released, ["mobiuslabsgmbh/faster-whisper-large-v3-turbo", "C:/hf/Qwen3-TTS-12Hz-1.7B-CustomVoice"])
        self.assertIsNone(main.pid)
        self.assertIsNone(qwen.pid)

    def test_kokoro_alone_stays(self) -> None:
        main = _worker({"tts": "C:/voice/kokoro"})
        self.assertEqual(self._run(main, None), [])
        main._kill.assert_not_called()  # noqa: SLF001

    def test_a_worker_answering_a_request_stays(self) -> None:
        qwen = _worker({"tts": "C:/hf/Qwen3-TTS"})
        qwen._lock.acquire()  # noqa: SLF001 - a request is running
        try:
            self.assertEqual(self._run(None, qwen), [])
        finally:
            qwen._lock.release()  # noqa: SLF001
        qwen._kill.assert_not_called()  # noqa: SLF001

    def test_nothing_happens_without_an_nvidia_gpu(self) -> None:
        qwen = _worker({"tts": "C:/hf/Qwen3-TTS"})
        self.assertEqual(self._run(None, qwen, cuda=False), [])
        qwen._kill.assert_not_called()  # noqa: SLF001

    def test_the_video_and_image_managers_release_before_loading(self) -> None:
        from backend_service.image_runtime import ImageRuntimeManager
        from backend_service.video_runtime import VideoRuntimeManager

        video = VideoRuntimeManager()
        video._engine = mock.Mock()  # noqa: SLF001
        video._engine.probe.return_value.realGenerationAvailable = True  # noqa: SLF001
        image = ImageRuntimeManager()
        image._diffusers = mock.Mock()  # noqa: SLF001
        image._diffusers.probe.return_value.realGenerationAvailable = True  # noqa: SLF001
        with mock.patch.object(voice_client, "release_gpu_for_media") as release:
            video.preload("Lightricks/LTX-2.5-Diffusers")
            image.preload("black-forest-labs/FLUX.2-klein-4B")
        self.assertEqual(release.call_count, 2)


class IdleSettingTests(unittest.TestCase):
    def setUp(self) -> None:
        patches = [mock.patch.object(voice_client, name, None) for name in ("_WORKER", "_QWEN_WORKER", "_SPEECH_IDLE_MINUTES")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(voice_client.atexit, "register").start()

    def test_a_new_worker_reads_the_stored_setting(self) -> None:
        with mock.patch("backend_service.voice_runtime.load_settings", return_value={"idleUnloadMinutes": 3}):
            self.assertEqual(voice_client.get_qwen_worker().idle_minutes, 3.0)
            self.assertEqual(voice_client.get_worker().idle_minutes, 3.0)

    def test_saving_the_setting_reaches_both_speech_workers(self) -> None:
        with mock.patch("backend_service.voice_runtime.load_settings", return_value={"idleUnloadMinutes": 10}):
            main, qwen = voice_client.get_worker(), voice_client.get_qwen_worker()
        voice_client.set_speech_idle_minutes(1)
        self.assertEqual((main.idle_minutes, qwen.idle_minutes), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
