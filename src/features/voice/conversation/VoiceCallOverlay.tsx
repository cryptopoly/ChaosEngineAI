import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import type { ChatMessage } from "../../../types";
import { useVoice } from "../VoiceContext";
import { useVoiceCall, type CallPhase } from "./useVoiceCall";

export interface VoiceCallOverlayProps {
  messages: ChatMessage[];
  modelName: string | null;
  send: (text: string) => Promise<void> | void;
  cancel: () => void;
  onEnd: () => void;
}

/** Hands-free voice call over the chat thread. Space mutes, Esc ends. */
export function VoiceCallOverlay({ messages, modelName, send, cancel, onEnd }: VoiceCallOverlayProps) {
  const { t } = useTranslation("chat");
  const { runtime, settings, platform } = useVoice();
  const call = useVoiceCall({ active: true, messages, platform, settings, send, cancel });
  const voiceName = runtime?.ttsVoices.find((v) => v.id === settings.ttsVoice)?.name ?? settings.ttsVoice;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (event.key === "Escape") {
        event.preventDefault();
        onEnd();
      } else if (event.key === " " && !typing && target?.tagName !== "BUTTON") {
        event.preventDefault();
        call.toggleMute();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [call.toggleMute, onEnd]);

  const labels: Record<CallPhase, string> = {
    starting: t("voiceCall.starting", { defaultValue: "Starting the microphone…" }),
    listening: t("voiceCall.listening", { defaultValue: "Listening…" }),
    hearing: t("voiceCall.hearing", { defaultValue: "Hearing you…" }),
    transcribing: t("voiceCall.transcribing", { defaultValue: "Got it…" }),
    thinking: t("voiceCall.thinking", { defaultValue: "Thinking…" }),
    speaking: t("voiceCall.speaking", { defaultValue: "Speaking…" }),
    error: t("voiceCall.error", { defaultValue: "The call stopped" }),
  };
  const label = call.muted && call.phase === "listening" ? t("voiceCall.muted", { defaultValue: "Muted" }) : labels[call.phase];
  const orbScale = call.phase === "hearing" || call.phase === "listening" ? 1 + call.level * 0.35 : 1;

  return (
    <div className="voice-call" role="dialog" aria-modal="true" aria-label={t("voiceCall.title", { defaultValue: "Voice call" })}>
      <div className="voice-call__stage">
        <div className={`voice-call__orb voice-call__orb--${call.phase}${call.muted ? " voice-call__orb--muted" : ""}`} style={{ transform: `scale(${orbScale})` }} aria-hidden="true" />
        <div className="voice-call__status" aria-live="polite">
          {label}
        </div>
        {call.error ? (
          <div className="voice-call__error" role="alert">
            {call.error}
          </div>
        ) : null}
        <div className="voice-call__captions">
          {call.userText ? (
            <p className="voice-call__line">
              <span className="voice-call__who">{t("voiceCall.you", { defaultValue: "You" })}</span>
              {call.userText}
            </p>
          ) : null}
          {call.replyText ? (
            <p className="voice-call__line voice-call__line--model">
              <span className="voice-call__who">{modelName ?? t("voiceCall.model", { defaultValue: "Model" })}</span>
              {call.replyText}
              {call.interrupted ? (
                <em className="voice-call__interrupted"> {t("voiceCall.interrupted", { defaultValue: "(interrupted)" })}</em>
              ) : null}
            </p>
          ) : null}
        </div>
      </div>
      <div className="voice-call__bar">
        <button type="button" className={`secondary-button voice-call__button${call.muted ? " voice-call__button--on" : ""}`} aria-pressed={call.muted} onClick={call.toggleMute} title={t("voiceCall.muteHint", { defaultValue: "Mute the microphone (Space)" })}>
          {call.muted ? t("voiceCall.unmute", { defaultValue: "Unmute" }) : t("voiceCall.mute", { defaultValue: "Mute" })}
        </button>
        <button type="button" className="voice-call__button voice-call__button--end" onClick={onEnd} title={t("voiceCall.endHint", { defaultValue: "End the call (Esc)" })}>
          {t("voiceCall.end", { defaultValue: "End call" })}
        </button>
        <span className="voice-call__meta">
          {t("voiceCall.meta", { defaultValue: "Voice: {voice}", voice: voiceName })}
          {settings.bargeIn ? ` · ${t("voiceCall.headphones", { defaultValue: "Headphones work best for interrupting" })}` : ""}
        </span>
      </div>
    </div>
  );
}
