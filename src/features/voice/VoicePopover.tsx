/**
 * Chat's Voice button and popover (D-73 S3): which voice reads replies, which answers calls and which model takes
 * dictation, each with Change; how Chat reads and calls (auto-read, auto-send, reading speed, the call pause,
 * interrupt); and what a voice call still needs, with one button to fix it. Same settings keys as before.
 */
import { useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { getVoices } from "../../api";
import type { VoiceEntry } from "../../types";
import { useVoice } from "./VoiceContext";
import { callVoiceOf } from "./conversation/useVoiceCall";
import { chatVoiceName, effectiveFallback, fallbackSetting } from "./speak/speakLogic";
import { voiceKindLabel } from "./speak/VoiceRow";
import { FallbackNote } from "./speak/FallbackNote";
import { VOICE_LANGUAGES, voiceLanguageName } from "./voiceLanguages";

export interface VoiceRequirement {
  label: string;
  ready: boolean;
  /** Fixed on Voice > Models (dictation, a voice, turn detection); the chat model is loaded in Chat. */
  fixOnModels?: boolean;
}

function VoiceIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" />
      <path d="M19 11a7 7 0 0 1-14 0M12 18v3" />
    </svg>
  );
}

export function VoicePopover({ requirements, initialOpen = false }: { requirements: VoiceRequirement[]; initialOpen?: boolean }) {
  const { t } = useTranslation("chat");
  const { t: ts, i18n } = useTranslation("studio");
  const tf = ts as unknown as (key: string, options?: Record<string, unknown>) => string;
  const { runtime, settings, saveSettings, openVoiceModels, openVoices } = useVoice();
  const [open, setOpen] = useState(initialOpen);
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const root = useRef<HTMLDivElement | null>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open || !runtime?.installed) return;
    let live = true;
    void getVoices()
      .then((list) => {
        if (live) setVoices(list);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [open, runtime?.installed]);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const quick = runtime?.ttsVoices ?? [];
  const nameOf = (id: string) => voices.find((v) => v.id === id)?.name ?? chatVoiceName(id, quick, ts("voice.voices.kindClone", { defaultValue: "Cloned voice" }));
  const kindOf = (id: string) => voices.find((v) => v.id === id);
  // A read-aloud speed outside the range (an old settings file) shows as 1.0x instead of breaking the slider.
  const speed = Number.isFinite(settings.speed) ? settings.speed : 1;
  const silence = Number.isFinite(settings.vadSilenceMs) ? settings.vadSilenceMs : 700;
  const callId = callVoiceOf(settings);
  // The fallback in effect (D-75): until the voice list has loaded, the setting itself.
  const fallback = effectiveFallback(voices, settings);
  const fallbackId = voices.length > 0 ? (fallback?.id ?? "") : fallbackSetting(settings);
  const sttModel =
    runtime?.sttModels.find((m) => m.id === settings.sttModel || m.key === settings.sttModel) ?? runtime?.sttModels.find((m) => m.default) ?? null;
  const languageLabel = VOICE_LANGUAGES.find(([code]) => code === settings.language)?.[1] ?? settings.language;
  const missing = requirements.filter((req) => !req.ready);
  const fixable = missing.some((req) => req.fixOnModels);
  const changeVoices = () => {
    setOpen(false);
    (openVoices ?? openVoiceModels)();
  };

  const voiceLine = (label: string, id: string, note?: string) => {
    const entry = kindOf(id);
    return (
      <div className="voice-popover__line">
        <span className="voice-popover__what">{label}</span>
        <span className="voice-popover__who">
          <strong>{id ? nameOf(id) : t("voicePopover.noFallback", { defaultValue: "None set" })}</strong>
          {entry ? <span className="badge">{voiceKindLabel(entry, tf)}</span> : null}
          {note ? <span className="muted-text">{note}</span> : null}
        </span>
        <button type="button" className="link-button" onClick={changeVoices}>
          {t("voicePopover.change", { defaultValue: "Change" })}
        </button>
      </div>
    );
  };

  return (
    <div className="voice-popover" ref={root}>
      <button
        type="button"
        className="secondary-button voice-popover__button"
        aria-expanded={open}
        aria-controls={panelId}
        title={t("voicePopover.title", { defaultValue: "Voice in Chat: which voice reads, calls and dictation" })}
        onClick={() => setOpen((on) => !on)}
      >
        <VoiceIcon /> {t("voicePopover.button", { defaultValue: "Voice" })}
      </button>
      {open ? (
        <div id={panelId} className="voice-popover__panel" role="dialog" aria-label={t("voicePopover.button", { defaultValue: "Voice" })}>
          <div className="voice-popover__head">
            <strong>{t("voicePopover.button", { defaultValue: "Voice" })}</strong>
            <button type="button" className="link-button" onClick={changeVoices}>
              {t("voicePopover.openVoices", { defaultValue: "Open Voices" })}
            </button>
          </div>
          {runtime?.installed ? (
            <>
              {voiceLine(t("voicePopover.read", { defaultValue: "Chat read-aloud" }), settings.ttsVoice)}
              {voiceLine(
                t("voicePopover.call", { defaultValue: "Voice calls" }),
                callId,
                settings.callVoice ? undefined : t("voicePopover.sameAsChat", { defaultValue: "(same as Chat)" }),
              )}
              <div className="voice-popover__line">
                <span className="voice-popover__what">{t("voicePopover.dictation", { defaultValue: "Dictation" })}</span>
                <span className="voice-popover__who">
                  <strong>{sttModel?.name ?? "—"}</strong>
                  <span className="muted-text">{voiceLanguageName(settings.language, languageLabel, i18n.language)}</span>
                </span>
                <button
                  type="button"
                  className="link-button"
                  onClick={() => {
                    setOpen(false);
                    openVoiceModels();
                  }}
                >
                  {t("voicePopover.change", { defaultValue: "Change" })}
                </button>
              </div>
              {voiceLine(t("voicePopover.fallback", { defaultValue: "Fallback" }), fallbackId)}
              {voices.length > 0 ? (
                <p className="muted-text voice-popover__fallback">
                  <FallbackNote voice={fallback} />
                </p>
              ) : null}
              <label className="checkbox-label">
                <input type="checkbox" checked={settings.autoRead} onChange={(e) => void saveSettings({ autoRead: e.target.checked })} />
                {ts("voice.settings.autoRead", { defaultValue: "Read every chat reply aloud when it finishes" })}
              </label>
              <label className="checkbox-label">
                <input type="checkbox" checked={settings.autoSend} onChange={(e) => void saveSettings({ autoSend: e.target.checked })} />
                {ts("voice.settings.autoSend", { defaultValue: "Send the message as soon as dictation finishes" })}
              </label>
              <label className="voice-settings__field">
                <span>
                  {ts("voice.settings.speed", { defaultValue: "Reading speed" })} ({speed.toFixed(1)}×)
                </span>
                <input type="range" min={0.5} max={2} step={0.1} value={speed} onChange={(e) => void saveSettings({ speed: parseFloat(e.target.value) })} />
              </label>
              <label className="voice-settings__field">
                <span>
                  {ts("voice.settings.silence", { defaultValue: "Voice call: pause that ends your turn" })} ({(silence / 1000).toFixed(1)} s)
                </span>
                <input
                  type="range"
                  min={300}
                  max={2000}
                  step={100}
                  value={silence}
                  onChange={(e) => void saveSettings({ vadSilenceMs: Number(e.target.value) })}
                />
              </label>
              <label className="checkbox-label">
                <input type="checkbox" checked={settings.bargeIn} onChange={(e) => void saveSettings({ bargeIn: e.target.checked })} />
                {ts("voice.settings.bargeIn", { defaultValue: "Voice call: interrupt the model by speaking" })}
              </label>
            </>
          ) : (
            <p className="muted-text">
              {t("voicePopover.noRuntime", { defaultValue: "Voice in Chat needs the voice runtime. Install it once from Voice > Models." })}
            </p>
          )}
          <div className="voice-popover__needs">
            <span className="voice-popover__what">{t("header.callNeeds", { defaultValue: "Needs:" })}</span>
            <ul>
              {requirements.map((req) => (
                <li key={req.label}>
                  <span className={req.ready ? "badge success" : "badge warning"}>
                    {req.ready ? t("header.callReady", { defaultValue: "Ready" }) : t("header.callMissing", { defaultValue: "Missing" })}
                  </span>{" "}
                  {req.label}
                </li>
              ))}
            </ul>
            {fixable ? (
              <button
                type="button"
                className="primary-button"
                onClick={() => {
                  setOpen(false);
                  openVoiceModels();
                }}
              >
                {t("voicePopover.fix", { defaultValue: "Install what is missing" })}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
