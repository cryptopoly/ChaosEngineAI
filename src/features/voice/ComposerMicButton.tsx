import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { transcribeBlob } from "../../api";
import { formatClock, micErrorMessage, startMicCapture, type MicCapture } from "./audio";
import { useVoice } from "./VoiceContext";

type MicState = "idle" | "recording" | "transcribing";

/** Hold longer than this and releasing the button stops (push-to-talk). */
const HOLD_MS = 350;

export interface ComposerMicButtonProps {
  /** Text to add to the composer. */
  onTranscript: (text: string) => void;
  /** Called after onTranscript when the user enabled auto-send. */
  onAutoSend?: () => void;
  disabled?: boolean;
}

/**
 * Dictation into a composer. Click to start and click to stop, or hold and
 * release (push-to-talk). Ctrl/Cmd+Shift+M toggles from the keyboard.
 */
export function ComposerMicButton({ onTranscript, onAutoSend, disabled = false }: ComposerMicButtonProps) {
  const { t } = useTranslation("chat");
  const { runtime, settings, platform, openVoiceModels } = useVoice();
  const [state, setState] = useState<MicState>("idle");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const captureRef = useRef<MicCapture | null>(null);
  const pressedAtRef = useRef(0);
  const stateRef = useRef<MicState>("idle");
  stateRef.current = state;

  const installed = Boolean(runtime?.sttAvailable);

  const finish = useCallback(async () => {
    const capture = captureRef.current;
    captureRef.current = null;
    if (!capture) return;
    setState("transcribing");
    setLevel(0);
    try {
      const wav = await capture.stop();
      const result = await transcribeBlob(wav, "dictation.wav", {
        model: settings.sttModel || undefined,
        language: settings.language || undefined,
      });
      const text = result.text.trim();
      if (text) {
        onTranscript(text);
        if (settings.autoSend) onAutoSend?.();
      } else {
        setError(t("voice.heardNothing", { defaultValue: "No speech was heard. Try again closer to the microphone." }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setState("idle");
      setSeconds(0);
    }
  }, [onAutoSend, onTranscript, settings.autoSend, settings.language, settings.sttModel, t]);

  const start = useCallback(async () => {
    setError(null);
    try {
      captureRef.current = await startMicCapture({
        onLevel: setLevel,
        onLimit: () => void finish(),
      });
      setState("recording");
    } catch (err) {
      setError(micErrorMessage(err, platform));
    }
  }, [finish, platform]);

  const toggle = useCallback(() => {
    if (!installed) {
      openVoiceModels();
      return;
    }
    if (stateRef.current === "idle") void start();
    else if (stateRef.current === "recording") void finish();
  }, [finish, installed, openVoiceModels, start]);

  // Recording timer.
  useEffect(() => {
    if (state !== "recording") return;
    const startedAt = Date.now();
    const timer = window.setInterval(() => setSeconds((Date.now() - startedAt) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [state]);

  // Keyboard toggle.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "m") {
        event.preventDefault();
        if (!disabled) toggle();
      }
      if (event.key === "Escape" && stateRef.current === "recording") {
        captureRef.current?.cancel();
        captureRef.current = null;
        setState("idle");
        setSeconds(0);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [disabled, toggle]);

  // Never leave the microphone open when the composer unmounts.
  useEffect(() => () => captureRef.current?.cancel(), []);

  const title = !installed
    ? t("voice.installToDictate", { defaultValue: "Voice input needs the voice pack. Click to install it." })
    : state === "recording"
      ? t("voice.stopDictation", { defaultValue: "Stop and transcribe (Esc cancels)" })
      : state === "transcribing"
        ? t("voice.transcribing", { defaultValue: "Transcribing…" })
        : t("voice.startDictation", { defaultValue: "Dictate (click, or hold to talk; Ctrl/Cmd+Shift+M)" });

  return (
    <div className="composer-mic">
      {state === "recording" ? (
        <span className="composer-mic__meter" aria-live="polite">
          <span className="composer-mic__bar" style={{ transform: `scaleX(${0.1 + level * 0.9})` }} />
          <span className="composer-mic__time">{formatClock(seconds)}</span>
        </span>
      ) : null}
      <button
        type="button"
        className={`composer-mic__button composer-mic__button--${state}${installed ? "" : " composer-mic__button--off"}`}
        title={title}
        aria-label={title}
        aria-pressed={state === "recording"}
        disabled={disabled || state === "transcribing"}
        onPointerDown={() => {
          pressedAtRef.current = Date.now();
          if (installed && stateRef.current === "idle") void start();
          else toggle();
        }}
        onPointerUp={() => {
          const held = Date.now() - pressedAtRef.current;
          if (held >= HOLD_MS && stateRef.current === "recording") void finish();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggle();
          }
        }}
      >
        {state === "transcribing" ? (
          <span className="composer-mic__spinner" aria-hidden="true" />
        ) : (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="17" x2="12" y2="22" />
          </svg>
        )}
      </button>
      {error ? (
        <span className="composer-mic__error" role="alert" title={error} onClick={() => setError(null)}>
          {error}
        </span>
      ) : null}
    </div>
  );
}
