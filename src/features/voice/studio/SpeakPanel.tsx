import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { encodeMp3, saveSpeech, synthesizeSpeech } from "../../../api";
import type { TtsVoice, VoiceRuntime } from "../../../types";
import { downloadBlob, formatClock } from "../audio";
import { speechPlayer, useSpeechPlayer } from "../speechPlayer";
import { useVoice } from "../VoiceContext";
import { joinWavs, splitSpeech } from "./speech";

const MAX_CHARS = 200_000;
const PREVIEW_OWNER = "voice-preview";

/** A short line in the voice's own language, for the preview button. */
const PREVIEW_LINES: Record<string, string> = {
  "en-US": "Hello! This is how I sound.",
  "en-GB": "Hello! This is how I sound.",
  es: "¡Hola! Así suena mi voz.",
  fr: "Bonjour ! Voici ma voix.",
  hi: "नमस्ते! मेरी आवाज़ ऐसी है।",
  it: "Ciao! Questa è la mia voce.",
  ja: "こんにちは。これが私の声です。",
  "pt-BR": "Olá! Esta é a minha voz.",
  zh: "你好！这是我的声音。",
};

export function SpeakPanel({ runtime }: { runtime: VoiceRuntime }) {
  const { t } = useTranslation("studio");
  const { settings } = useVoice();
  const player = useSpeechPlayer();
  const [text, setText] = useState("");
  const [voice, setVoice] = useState(settings.ttsVoice);
  const [speed, setSpeed] = useState(settings.speed);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ url: string; blob: Blob; durationS: number; text: string } | null>(null);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [busyExport, setBusyExport] = useState(false);
  const cancelRef = useRef(false);

  useEffect(
    () => () => {
      if (result) URL.revokeObjectURL(result.url);
    },
    [result],
  );

  const voices = runtime.ttsVoices.filter((v: TtsVoice) => v.available !== false);
  const languages = [...new Set(voices.map((v) => v.languageLabel ?? v.language))];
  const selected = voices.find((v) => v.id === voice);
  const parts = splitSpeech(text);
  const previewing = player.ownerId === PREVIEW_OWNER && player.status !== "idle";

  const generate = async () => {
    setError(null);
    setSavedId(null);
    cancelRef.current = false;
    setProgress({ done: 0, total: parts.length });
    try {
      const wavs: ArrayBuffer[] = [];
      for (const [index, part] of parts.entries()) {
        if (cancelRef.current) return;
        wavs.push(await (await synthesizeSpeech(part.text, voice, speed)).arrayBuffer());
        setProgress({ done: index + 1, total: parts.length });
      }
      if (cancelRef.current) return;
      const { blob, durationS } = joinWavs(wavs, parts.map((p) => p.gapAfter));
      setResult({ url: URL.createObjectURL(blob), blob, durationS, text });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setProgress(null);
    }
  };

  const preview = () => {
    if (previewing) {
      speechPlayer.stop();
      return;
    }
    const line = PREVIEW_LINES[selected?.language ?? "en-US"] ?? PREVIEW_LINES["en-US"];
    void speechPlayer.play(PREVIEW_OWNER, line, voice, speed);
  };

  const exportMp3 = async () => {
    if (!result) return;
    setBusyExport(true);
    try {
      downloadBlob(await encodeMp3(result.blob), "speech.mp3");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyExport(false);
    }
  };

  const save = async () => {
    if (!result) return;
    try {
      const item = await saveSpeech(result.blob, { text: result.text, voice, speed });
      setSavedId(item.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Panel
      title={t("voice.speak.title", { defaultValue: "Speak" })}
      subtitle={t("voice.speak.subtitleLong", { defaultValue: "Turn text into speech. Blank lines start a new paragraph; [pause] adds a pause." })}
      className="span-2"
    >
      <div className="voice-studio">
        <textarea
          className="text-input voice-textarea voice-textarea--long"
          value={text}
          maxLength={MAX_CHARS}
          onChange={(e) => setText(e.target.value)}
          placeholder={t("voice.speak.placeholder", { defaultValue: "Type or paste text. Markdown and code are tidied up before speaking." })}
        />
        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>{t("voice.speak.voice", { defaultValue: "Voice" })}</span>
            <span className="voice-studio__voice">
              <select className="text-input" value={voice} onChange={(e) => setVoice(e.target.value)}>
                {languages.map((language) => (
                  <optgroup key={language} label={language}>
                    {voices
                      .filter((v) => (v.languageLabel ?? v.language) === language)
                      .map((v) => (
                        <option key={v.id} value={v.id}>
                          {v.name} ({v.gender === "male" ? "M" : "F"})
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
              <button
                type="button"
                className="secondary-button"
                onClick={preview}
                aria-label={t("voice.speak.preview", { defaultValue: "Preview this voice" })}
                title={t("voice.speak.preview", { defaultValue: "Preview this voice" })}
              >
                {previewing ? t("voice.stop", { defaultValue: "Stop" }) : t("voice.speak.listen", { defaultValue: "Listen" })}
              </button>
            </span>
          </label>
          <label className="voice-settings__field">
            <span>
              {t("voice.speak.speed", { defaultValue: "Speed" })} ({speed.toFixed(1)}×)
            </span>
            <input type="range" min={0.5} max={2} step={0.1} value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
          </label>
        </div>
        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="button-row">
          {progress ? (
            <>
              <span className="voice-studio__progress" aria-live="polite">
                {t("voice.speak.progress", { defaultValue: "Part {done} of {total}", done: Math.min(progress.done + 1, progress.total), total: progress.total })}
              </span>
              <progress max={progress.total} value={progress.done} />
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  cancelRef.current = true;
                }}
              >
                {t("voice.cancel", { defaultValue: "Cancel" })}
              </button>
            </>
          ) : (
            <button type="button" className="primary-button" disabled={parts.length === 0} onClick={() => void generate()}>
              {parts.length > 1
                ? t("voice.speak.generateParts", { defaultValue: "Generate audio ({n} parts)", n: parts.length })
                : t("voice.speak.generate", { defaultValue: "Generate audio" })}
            </button>
          )}
        </div>
        {result ? (
          <div className="voice-studio__result">
            <audio controls src={result.url} />
            <div className="button-row">
              <span className="muted-text">{formatClock(result.durationS)}</span>
              <button type="button" className="secondary-button" onClick={() => downloadBlob(result.blob, "speech.wav")}>
                WAV
              </button>
              <button type="button" className="secondary-button" disabled={busyExport} onClick={() => void exportMp3()}>
                {busyExport ? t("voice.speak.encoding", { defaultValue: "Encoding…" }) : "MP3"}
              </button>
              <button type="button" className="secondary-button" disabled={savedId != null} onClick={() => void save()}>
                {savedId ? t("voice.savedToLibrary", { defaultValue: "Saved to Library" }) : t("voice.saveToLibrary", { defaultValue: "Save to Library" })}
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}
