import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { saveGalleryAudio, saveGalleryTranscript, synthesizeSpeech, transcribeBlob, type TranscribeResult } from "../../api";
import type { SttModel, TabId, TtsVoice, VoiceRuntime } from "../../types";
import {
  downloadBlob,
  formatClock,
  formatTimestamp,
  micErrorMessage,
  startMicCapture,
  toSrt,
  toVtt,
  type MicCapture,
} from "./audio";
import { speechPlayer, useSpeechPlayer } from "./speechPlayer";
import { useVoice } from "./VoiceContext";
import { VOICE_LANGUAGES } from "./VoiceModelsTab";

export interface VoiceStudioTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onSendToChat: (text: string) => void;
  onTabChange: (tab: TabId) => void;
}

const ACCEPTED_FILES = "audio/*,video/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.webm,.mp4,.mov,.mkv,.m4v,.aiff,.caf";

/** Prefer a downloaded model, then the recommended one. */
export function pickDefaultSttModel(models: SttModel[], preferred?: string): SttModel | undefined {
  const runnable = models.filter((m) => m.backendInstalled !== false);
  const pool = runnable.length > 0 ? runnable : models;
  const chosen = preferred ? pool.find((m) => m.id === preferred || m.key === preferred) : undefined;
  if (chosen) return chosen;
  const recommended = pool.find((m) => m.default);
  if (recommended?.installed) return recommended;
  return pool.find((m) => m.installed) ?? recommended ?? pool[0];
}

function useFlash(initial: string): [string, (label: string) => void] {
  const [label, setLabel] = useState(initial);
  const flash = useCallback(
    (next: string) => {
      setLabel(next);
      window.setTimeout(() => setLabel(initial), 2000);
    },
    [initial],
  );
  return [label, flash];
}

function TranscribePanel({ runtime, onSendToChat }: { runtime: VoiceRuntime; onSendToChat: (text: string) => void }) {
  const { t } = useTranslation("studio");
  const { settings, platform } = useVoice();
  const [model, setModel] = useState("");
  const [language, setLanguage] = useState(settings.language);
  const [state, setState] = useState<"idle" | "recording" | "transcribing">("idle");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [source, setSource] = useState<string | null>(null);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [copyLabel, flashCopy] = useFlash(t("voice.copy", { defaultValue: "Copy" }));
  const [saveLabel, flashSave] = useFlash(t("voice.save", { defaultValue: "Save to gallery" }));
  const captureRef = useRef<MicCapture | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!model) setModel(pickDefaultSttModel(runtime.sttModels, settings.sttModel)?.id ?? "");
  }, [model, runtime.sttModels, settings.sttModel]);

  useEffect(() => {
    if (state !== "recording") return;
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds((Date.now() - started) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [state]);

  useEffect(
    () => () => {
      captureRef.current?.cancel();
      abortRef.current?.abort();
    },
    [],
  );

  const run = useCallback(
    async (blob: Blob, filename: string) => {
      setState("transcribing");
      setError(null);
      setSource(filename);
      abortRef.current = new AbortController();
      try {
        const next = await transcribeBlob(blob, filename, {
          model: model || undefined,
          language: language || undefined,
          signal: abortRef.current.signal,
        });
        setResult(next);
        setText(next.text);
      } catch (err) {
        if (!abortRef.current?.signal.aborted) setError(err instanceof Error ? err.message : String(err));
      } finally {
        setState("idle");
        setSeconds(0);
        abortRef.current = null;
      }
    },
    [language, model],
  );

  const stopRecording = useCallback(async () => {
    const capture = captureRef.current;
    captureRef.current = null;
    if (!capture) return;
    setLevel(0);
    const wav = await capture.stop();
    await run(wav, "recording.wav");
  }, [run]);

  const record = async () => {
    if (state === "recording") {
      await stopRecording();
      return;
    }
    setError(null);
    try {
      captureRef.current = await startMicCapture({ onLevel: setLevel, onLimit: () => void stopRecording() });
      setState("recording");
    } catch (err) {
      setError(micErrorMessage(err, platform));
    }
  };

  const onFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) void run(file, file.name);
  };

  const segments = result?.segments ?? [];
  const baseName = (source ?? "transcript").replace(/\.[^.]+$/, "");

  return (
    <Panel
      title={t("voice.transcribe.title", { defaultValue: "Transcribe" })}
      subtitle={t("voice.transcribe.subtitle", { defaultValue: "Record, or drop an audio or video file" })}
    >
      <div className="voice-studio">
        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>{t("voice.transcribe.model", { defaultValue: "Model" })}</span>
            <select className="text-input" value={model} onChange={(e) => setModel(e.target.value)} disabled={state !== "idle"}>
              {runtime.sttModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                  {m.installed ? "" : ` (${m.sizeGb} GB download)`}
                </option>
              ))}
            </select>
          </label>
          <label className="voice-settings__field">
            <span>{t("voice.transcribe.language", { defaultValue: "Language" })}</span>
            <select className="text-input" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={state !== "idle"}>
              {VOICE_LANGUAGES.map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div
          className={`voice-drop${dragging ? " voice-drop--active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (state === "idle") onFiles(e.dataTransfer.files);
          }}
        >
          <button
            type="button"
            className={`voice-record${state === "recording" ? " voice-record--active" : ""}`}
            onClick={() => void record()}
            disabled={state === "transcribing"}
            aria-label={state === "recording" ? t("voice.stopRecording", { defaultValue: "Stop recording" }) : t("voice.record", { defaultValue: "Record" })}
          >
            <span className="voice-record__dot" />
          </button>
          <div className="voice-drop__text">
            {state === "recording" ? (
              <>
                <strong>{formatClock(seconds)}</strong>
                <span className="voice-level">
                  <span className="voice-level__fill" style={{ transform: `scaleX(${0.05 + level * 0.95})` }} />
                </span>
              </>
            ) : state === "transcribing" ? (
              <span>
                {t("voice.transcribing", { defaultValue: "Transcribing" })} {source}…{" "}
                <button type="button" className="link-button" onClick={() => abortRef.current?.abort()}>
                  {t("voice.cancel", { defaultValue: "Cancel" })}
                </button>
              </span>
            ) : (
              <span>
                {t("voice.transcribe.hint", { defaultValue: "Click the button to record, or drop a file here." })}{" "}
                <button type="button" className="link-button" onClick={() => fileInput.current?.click()}>
                  {t("voice.transcribe.choose", { defaultValue: "Choose a file" })}
                </button>
              </span>
            )}
          </div>
          <input ref={fileInput} type="file" accept={ACCEPTED_FILES} hidden onChange={(e) => onFiles(e.target.files)} />
        </div>

        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}

        {result ? (
          <p className="muted-text voice-studio__meta">
            {result.durationS != null ? `${formatClock(result.durationS)} · ` : ""}
            {result.language ? `${result.language} · ` : ""}
            {result.seconds != null ? t("voice.transcribe.took", { defaultValue: "transcribed in {s} s", s: result.seconds.toFixed(1) }) : ""}
          </p>
        ) : null}
        <textarea
          className="text-input voice-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={t("voice.transcribe.placeholder", { defaultValue: "The transcript appears here. You can edit it." })}
        />
        {segments.length > 1 ? (
          <details className="voice-segments">
            <summary>{t("voice.transcribe.segments", { defaultValue: "Timestamps ({n})", n: segments.length })}</summary>
            <ol>
              {segments.map((segment, index) => (
                <li key={index}>
                  <code>{formatTimestamp(segment.start).slice(0, 8)}</code> {segment.text}
                </li>
              ))}
            </ol>
          </details>
        ) : null}
        <div className="button-row">
          <button type="button" className="primary-button" disabled={!text.trim()} onClick={() => onSendToChat(text)}>
            {t("voice.sendToChat", { defaultValue: "Send to chat" })}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!text.trim()}
            onClick={() => void navigator.clipboard.writeText(text).then(() => flashCopy(t("voice.copied", { defaultValue: "Copied" })))}
          >
            {copyLabel}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!text.trim()}
            onClick={() => void saveGalleryTranscript(text).then(() => flashSave(t("voice.saved", { defaultValue: "Saved" })))}
          >
            {saveLabel}
          </button>
          <button type="button" className="secondary-button" disabled={!text.trim()} onClick={() => downloadBlob(new Blob([text], { type: "text/plain" }), `${baseName}.txt`)}>
            TXT
          </button>
          <button type="button" className="secondary-button" disabled={segments.length === 0} onClick={() => downloadBlob(new Blob([toSrt(segments)], { type: "application/x-subrip" }), `${baseName}.srt`)}>
            SRT
          </button>
          <button type="button" className="secondary-button" disabled={segments.length === 0} onClick={() => downloadBlob(new Blob([toVtt(segments)], { type: "text/vtt" }), `${baseName}.vtt`)}>
            VTT
          </button>
        </div>
      </div>
    </Panel>
  );
}

function SpeakPanel({ runtime }: { runtime: VoiceRuntime }) {
  const { t } = useTranslation("studio");
  const { settings } = useVoice();
  const player = useSpeechPlayer();
  const [text, setText] = useState("");
  const [voice, setVoice] = useState(settings.ttsVoice);
  const [speed, setSpeed] = useState(settings.speed);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audio, setAudio] = useState<{ url: string; blob: Blob } | null>(null);
  const [saveLabel, flashSave] = useFlash(t("voice.save", { defaultValue: "Save to gallery" }));

  useEffect(() => () => {
    if (audio) URL.revokeObjectURL(audio.url);
  }, [audio]);

  const voices = runtime.ttsVoices.filter((v: TtsVoice) => v.available !== false);
  const languages = [...new Set(voices.map((v) => v.languageLabel ?? v.language))];
  const reading = player.ownerId === "studio" && player.status !== "idle";

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      const blob = await synthesizeSpeech(text, voice, speed);
      setAudio({ url: URL.createObjectURL(blob), blob });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title={t("voice.speak.title", { defaultValue: "Speak" })} subtitle={t("voice.speak.subtitle", { defaultValue: "Turn text into speech" })}>
      <div className="voice-studio">
        <textarea
          className="text-input voice-textarea"
          value={text}
          maxLength={20000}
          onChange={(e) => setText(e.target.value)}
          placeholder={t("voice.speak.placeholder", { defaultValue: "Type or paste text. Markdown and code are tidied up before speaking." })}
        />
        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>{t("voice.speak.voice", { defaultValue: "Voice" })}</span>
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
          <button type="button" className="primary-button" disabled={!text.trim() || busy} onClick={() => void generate()}>
            {busy ? t("voice.speak.generating", { defaultValue: "Generating…" }) : t("voice.speak.generate", { defaultValue: "Generate audio" })}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!text.trim()}
            onClick={() => (reading ? speechPlayer.stop() : void speechPlayer.play("studio", text, voice, speed))}
          >
            {reading ? t("voice.stop", { defaultValue: "Stop" }) : t("voice.speak.readNow", { defaultValue: "Read aloud now" })}
          </button>
        </div>
        {audio ? (
          <div className="voice-studio__result">
            <audio controls src={audio.url} />
            <div className="button-row">
              <button type="button" className="secondary-button" onClick={() => downloadBlob(audio.blob, "speech.wav")}>
                {t("voice.speak.download", { defaultValue: "Download WAV" })}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => void saveGalleryAudio(text, voice, audio.blob).then(() => flashSave(t("voice.saved", { defaultValue: "Saved" })))}
              >
                {saveLabel}
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export function VoiceStudioTab({ voiceRuntime, backendOnline, onSendToChat, onTabChange }: VoiceStudioTabProps) {
  const { t } = useTranslation("studio");

  if (!backendOnline || !voiceRuntime) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={t("voice.studio.title", { defaultValue: "Voice Studio" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">{t("voice.offline", { defaultValue: "Waiting for the backend…" })}</p>
          </div>
        </Panel>
      </div>
    );
  }

  if (!voiceRuntime.installed) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={t("voice.studio.title", { defaultValue: "Voice Studio" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">
              {t("voice.studio.needsPack", {
                defaultValue: "Voice runs on this computer and needs the voice pack (about 2 GB). Install it once from Voice > Models.",
              })}
            </p>
            <button type="button" className="primary-button" onClick={() => onTabChange("voice-models")}>
              {t("voice.studio.openModels", { defaultValue: "Open Voice Models" })}
            </button>
          </div>
        </Panel>
      </div>
    );
  }

  return (
    <div className="content-grid discover-page voice-page">
      <TranscribePanel runtime={voiceRuntime} onSendToChat={onSendToChat} />
      <SpeakPanel runtime={voiceRuntime} />
    </div>
  );
}
