import { useTranslation } from "react-i18next";
import type { VoiceChatCost, VoiceEntry } from "../../../types";
import { callVoiceOf } from "../conversation/useVoiceCall";
import { useVoice } from "../VoiceContext";
import { chatVoiceChoices } from "./speakLogic";

/** Measured cost of a voice in Chat: time to the first sentence and memory, and a label when it is slow for calls. */
export function ChatCost({ cost }: { cost?: VoiceChatCost | null }) {
  const { t } = useTranslation("studio");
  if (!cost) return null;
  return (
    <span className="muted-text voice-chat-cost" title={cost.measuredOn ?? undefined}>
      {t("voice.chatCalls.cost", {
        defaultValue: "first audio {warm} s ({cold} s cold), {memory} GB",
        warm: cost.firstAudioS,
        cold: cost.firstAudioColdS ?? "?",
        memory: cost.peakGb ?? "?",
      })}
      {cost.slowForCalls ? <span className="badge warning">{t("voice.chatCalls.slow", { defaultValue: "Slow for calls" })}</span> : null}
    </span>
  );
}

const KIND_ORDER: VoiceEntry["kind"][] = ["preset", "clone", "builtin"];

/** One select of the voices allowed for ``use``, grouped by kind, each option naming its measured first-audio time. */
function VoiceSelect({
  voices,
  use,
  value,
  label,
  extra,
  onChange,
}: {
  voices: VoiceEntry[];
  use: "read" | "call";
  value: string;
  label: string;
  extra?: { value: string; label: string };
  onChange: (id: string) => void;
}) {
  const { t } = useTranslation("studio");
  const choices = chatVoiceChoices(voices, use);
  const groupLabel: Record<string, string> = {
    preset: t("voice.voices.kindPreset", { defaultValue: "Natural" }),
    clone: t("voice.voices.kindClone", { defaultValue: "Cloned" }),
    builtin: t("voice.voices.kindBuiltin", { defaultValue: "Quick" }),
  };
  const known = choices.some((v) => v.id === value) || extra?.value === value;
  const optionText = (v: VoiceEntry) =>
    v.chatCost
      ? t("voice.chatCalls.option", {
          defaultValue: "{name}, {seconds} s",
          name: v.name,
          seconds: v.chatCost.firstAudioS,
        }) + (use === "call" && v.chatCost.slowForCalls ? ` (${t("voice.chatCalls.slow", { defaultValue: "Slow for calls" })})` : "")
      : v.name;
  return (
    <label className="voice-settings__field">
      <span>{label}</span>
      <select className="text-input" value={value} aria-label={label} onChange={(e) => onChange(e.target.value)}>
        {extra ? <option value={extra.value}>{extra.label}</option> : null}
        {known ? null : <option value={value}>{value}</option>}
        {KIND_ORDER.map((kind) => {
          const group = choices.filter((v) => v.kind === kind);
          return group.length > 0 ? (
            <optgroup key={kind} label={groupLabel[kind]}>
              {group.map((v) => (
                <option key={v.id} value={v.id}>
                  {optionText(v)}
                </option>
              ))}
            </optgroup>
          ) : null;
        })}
      </select>
    </label>
  );
}

/**
 * How chat speaks: the chat voice and the call voice (D-72, any voice that runs here; cloned voices read aloud only),
 * each with its measured cost, then reading speed, auto-read, auto-send, the call pause and interrupt. Same settings
 * keys as before (``ttsVoice``, ``speed``, ``autoRead``, ``autoSend``, ``vadSilenceMs``, ``bargeIn``) plus
 * ``callVoice``; dictation language and voice memory stay on Voice Models.
 */
export function ChatCallsPanel({ voices }: { voices: VoiceEntry[] }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const chatVoice = voices.find((v) => v.id === settings.ttsVoice);
  const callId = callVoiceOf(settings);
  const callVoice = voices.find((v) => v.id === callId);
  return (
    <section className="voice-chat-calls" aria-labelledby="voice-chat-calls-title">
      <h4 id="voice-chat-calls-title" className="voice-group__title">
        {t("voice.chatCalls.title", { defaultValue: "Chat and calls" })}
      </h4>
      <div className="voice-settings">
        <VoiceSelect
          voices={voices}
          use="read"
          value={settings.ttsVoice}
          label={t("voice.chatCalls.chatVoice", { defaultValue: "Chat voice (read-aloud)" })}
          onChange={(id) => void saveSettings({ ttsVoice: id })}
        />
        <ChatCost cost={chatVoice?.chatCost} />
        <VoiceSelect
          voices={voices}
          use="call"
          value={settings.callVoice ?? ""}
          label={t("voice.chatCalls.callVoice", { defaultValue: "Call voice" })}
          extra={{ value: "", label: t("voice.chatCalls.sameAsChat", { defaultValue: "Same as the chat voice" }) }}
          onChange={(id) => void saveSettings({ callVoice: id })}
        />
        <ChatCost cost={callVoice?.chatCost} />
        <small className="muted-text">
          {t("voice.chatCalls.memory", {
            defaultValue:
              "Natural voices load a speech model of up to 3.6 GB next to the chat model. One speech model stays loaded and is freed after the idle time set on Voice Models. If a voice cannot speak, the Kokoro voice reads instead.",
          })}
        </small>
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
