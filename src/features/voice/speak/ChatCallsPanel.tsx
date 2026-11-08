import { useTranslation } from "react-i18next";
import type { VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";

/**
 * How chat speaks: the chat voice (set with Use in Chat on the list above), reading speed, auto-read, auto-send,
 * the call pause and interrupt. Same settings keys as before (``ttsVoice``, ``speed``, ``autoRead``, ``autoSend``,
 * ``vadSilenceMs``, ``bargeIn``); dictation language and voice memory stay on Voice Models.
 */
export function ChatCallsPanel({ voices }: { voices: VoiceEntry[] }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const chatVoice = voices.find((v) => v.id === settings.ttsVoice);
  return (
    <section className="voice-chat-calls" aria-labelledby="voice-chat-calls-title">
      <h4 id="voice-chat-calls-title" className="voice-group__title">
        {t("voice.chatCalls.title", { defaultValue: "Chat and calls" })}
      </h4>
      <p className="muted-text">
        {t("voice.chatCalls.voice", {
          defaultValue: "Chat voice: {name}. Change it with Use in Chat above.",
          name: chatVoice?.name ?? settings.ttsVoice,
        })}
      </p>
      <div className="voice-settings">
        <label className="voice-settings__field">
          <span>
            {t("voice.settings.speed", { defaultValue: "Reading speed" })} ({settings.speed.toFixed(1)}×)
          </span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.1}
            value={settings.speed}
            onChange={(e) => void saveSettings({ speed: parseFloat(e.target.value) })}
          />
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.autoRead} onChange={(e) => void saveSettings({ autoRead: e.target.checked })} />
          {t("voice.settings.autoRead", { defaultValue: "Read every chat reply aloud when it finishes" })}
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.autoSend} onChange={(e) => void saveSettings({ autoSend: e.target.checked })} />
          {t("voice.settings.autoSend", { defaultValue: "Send the message as soon as dictation finishes" })}
        </label>
        <label className="voice-settings__field">
          <span>
            {t("voice.settings.silence", { defaultValue: "Voice call: pause that ends your turn" })} (
            {(settings.vadSilenceMs / 1000).toFixed(1)} s)
          </span>
          <input
            type="range"
            min={300}
            max={2000}
            step={100}
            value={settings.vadSilenceMs}
            onChange={(e) => void saveSettings({ vadSilenceMs: Number(e.target.value) })}
          />
          <small className="muted-text">
            {t("voice.settings.silenceHint", { defaultValue: "Shorter replies faster; longer lets you pause to think." })}
          </small>
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.bargeIn} onChange={(e) => void saveSettings({ bargeIn: e.target.checked })} />
          {t("voice.settings.bargeIn", { defaultValue: "Voice call: interrupt the model by speaking" })}
        </label>
      </div>
    </section>
  );
}
