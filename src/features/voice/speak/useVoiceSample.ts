import { useCallback, useEffect, useRef, useState } from "react";
import { speakVoice } from "../../../api";
import type { VoiceEntry } from "../../../types";
import { speechPlayer, useSpeechPlayer } from "../speechPlayer";

const OWNER_PREFIX = "voice-sample:";

/** A short line in the voice's own language, for the sample button. */
const SAMPLE_LINES: Record<string, string> = {
  en: "Hello! This is how I sound.",
  "en-US": "Hello! This is how I sound.",
  "en-GB": "Hello! This is how I sound.",
  es: "¡Hola! Así suena mi voz.",
  fr: "Bonjour ! Voici ma voix.",
  hi: "नमस्ते! मेरी आवाज़ ऐसी है।",
  it: "Ciao! Questa è la mia voce.",
  ja: "こんにちは。これが私の声です。",
  ko: "안녕하세요! 제 목소리는 이렇습니다.",
  "pt-BR": "Olá! Esta é a minha voz.",
  zh: "你好！这是我的声音。",
};

/** Whether a sample can be played: built-in voices and presets. Designed and cloned voices have none. */
export function canPlaySample(voice: VoiceEntry): boolean {
  return (voice.kind === "builtin" || voice.kind === "preset") && voice.available;
}

/**
 * Plays a sample line in a voice. Built-in voices stream through the shared read-aloud player; presets make one
 * short speak request (not saved anywhere). One sample plays at a time.
 */
export function useVoiceSample(speed = 1) {
  const player = useSpeechPlayer();
  const [presetId, setPresetId] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const stopPreset = useCallback(() => {
    audioRef.current?.pause();
    audioRef.current = null;
    setPresetId(null);
    setLoadingId(null);
  }, []);

  useEffect(() => stopPreset, [stopPreset]);

  const isPlaying = (id: string) =>
    presetId === id || loadingId === id || (player.ownerId === `${OWNER_PREFIX}${id}` && player.status !== "idle");

  const isLoading = (id: string) => loadingId === id;

  const stop = useCallback(() => {
    stopPreset();
    speechPlayer.stop();
  }, [stopPreset]);

  const play = async (voice: VoiceEntry) => {
    if (isPlaying(voice.id)) {
      stop();
      return;
    }
    stop();
    const line = SAMPLE_LINES[voice.language] ?? SAMPLE_LINES.en;
    if (voice.kind === "builtin") {
      await speechPlayer.play(`${OWNER_PREFIX}${voice.id}`, line, voice.id, speed);
      return;
    }
    setLoadingId(voice.id);
    try {
      const result = await speakVoice({ text: line, voice: voice.id, speed });
      const audio = new Audio(`data:audio/wav;base64,${result.audio}`);
      audio.onended = () => setPresetId(null);
      audioRef.current = audio;
      setLoadingId(null);
      setPresetId(voice.id);
      await audio.play();
    } catch {
      stopPreset();
    }
  };

  return { isPlaying, isLoading, play, stop };
}
