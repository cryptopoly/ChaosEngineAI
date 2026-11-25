/**
 * Chat's Voice button and popover (D-73 S3, D0142): "Voices in Chat" (Read-aloud, Calls, Backup, each with Change,
 * which opens the voice picker here), the model that takes dictation; how Chat reads and calls (auto-read,
 * auto-send, reading speed, the call pause, interrupt); and what a voice call still needs, with one button to fix
 * it. Same settings keys as before. The Voice Studio no longer holds these choices.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import { useVoice } from "./VoiceContext";
import { callVoiceOf } from "./conversation/useVoiceCall";
import { ChatVoiceDrawer, type ChatVoiceUse } from "./ChatVoiceDrawer";
import { VOICE_USE_SETTING, chatVoiceName, effectiveFallback, fallbackSetting } from "./speak/speakLogic";
import { useVoiceCatalog } from "./speak/useVoiceCatalog";
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

/** Gap kept under the button and above the window's bottom edge. */
const POPOVER_GAP = 6;
const POPOVER_MARGIN = 12;
/** Below this width the stylesheet pins the panel to the window's sides (phone layout). */
const POPOVER_NARROW_PX = 760;

/**
 * Where the open popover goes: fixed under its button, as tall as the window leaves room for (it scrolls inside), so
 * the chat panel's clipping cannot cut it off. ``right`` is left to the stylesheet in the narrow layout.
 */
export function popoverPlacement(
  button: { bottom: number; right: number },
  viewport: { width: number; height: number },
): CSSProperties {
  const top = Math.round(button.bottom + POPOVER_GAP);
  const placement: CSSProperties = { top, maxHeight: Math.max(160, viewport.height - top - POPOVER_MARGIN) };
  if (viewport.width > POPOVER_NARROW_PX) placement.right = Math.max(POPOVER_MARGIN, Math.round(viewport.width - button.right));
  return placement;
}

export function VoicePopover({ requirements, initialOpen = false }: { requirements: VoiceRequirement[]; initialOpen?: boolean }) {
  const { t } = useTranslation("chat");
  const { t: ts, i18n } = useTranslation("studio");
  const tf = ts as unknown as (key: string, options?: Record<string, unknown>) => string;
  const { runtime, settings, saveSettings, openVoiceModels, openVoices } = useVoice();
  const [open, setOpen] = useState(initialOpen);
  // The voice picker that Change opens (read-aloud, calls or backup); null while closed.
  const [choosing, setChoosing] = useState<ChatVoiceUse | null>(null);
  const root = useRef<HTMLDivElement | null>(null);
  const button = useRef<HTMLButtonElement | null>(null);
  const [placement, setPlacement] = useState<CSSProperties | undefined>(undefined);
  const panelId = useId();
  // The voices (names, kinds) and, for the picker, the speech models; loaded while the popover is open.
  const catalog = useVoiceCatalog(open && Boolean(runtime?.installed), runtime?.lane);
  const voices = catalog.voices;

  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const rect = button.current?.getBoundingClientRect();
      if (rect) setPlacement(popoverPlacement(rect, { width: window.innerWidth, height: window.innerHeight }));
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      // Escape closes the picker first (its drawer handles the key itself), the popover next.
      if (event.key === "Escape" && !choosing) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, choosing]);

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

  const voiceLine = (label: string, use: ChatVoiceUse, id: string, note?: string) => {
    const entry = kindOf(id);
    return (
      <div className="voice-popover__line">
        <span className="voice-popover__what">{label}</span>
        <span className="voice-popover__who">
          <strong>{id ? nameOf(id) : t("voicePopover.noFallback", { defaultValue: "None set" })}</strong>
          {entry ? <span className="badge">{voiceKindLabel(entry, tf)}</span> : null}
          {note ? <span className="muted-text">{note}</span> : null}
        </span>
        <button type="button" className="link-button" aria-label={t("voicePopover.changeFor", { defaultValue: "Change {use}", use: label })} onClick={() => setChoosing(use)}>
          {t("voicePopover.change", { defaultValue: "Change" })}
        </button>
      </div>
    );
  };

  return (
    <div className="voice-popover" ref={root}>
      <button
        type="button"
        ref={button}
        className="secondary-button voice-popover__button"
        aria-expanded={open}
        aria-controls={panelId}
        title={t("voicePopover.title", { defaultValue: "Voice in Chat: which voice reads, calls and dictation" })}
        onClick={() => setOpen((on) => !on)}
      >
        <VoiceIcon /> {t("voicePopover.button", { defaultValue: "Voice" })}
      </button>
      {open ? (
        <div id={panelId} className="voice-popover__panel" style={placement} role="dialog" aria-label={t("voicePopover.button", { defaultValue: "Voice" })}>
          <div className="voice-popover__head">
            <strong>{t("voicePopover.button", { defaultValue: "Voice" })}</strong>
            <button type="button" className="link-button" onClick={changeVoices}>
              {t("voicePopover.openStudio", { defaultValue: "Open Voice Studio" })}
            </button>
          </div>
          {runtime?.installed ? (
            <>
              <strong className="voice-popover__section">{t("voicePopover.voicesInChat", { defaultValue: "Voices in Chat" })}</strong>
              {voiceLine(t("voicePopover.readAloud", { defaultValue: "Read-aloud" }), "read", settings.ttsVoice)}
              {voiceLine(
                t("voicePopover.calls", { defaultValue: "Calls" }),
                "call",
                callId,
                settings.callVoice ? undefined : t("voicePopover.sameAsChat", { defaultValue: "(same as Chat)" }),
              )}
              {voiceLine(t("voicePopover.backup", { defaultValue: "Backup" }), "fallback", fallbackId)}
              {voices.length > 0 ? (
                <p className="muted-text voice-popover__fallback">
                  <FallbackNote voice={fallback} />
                </p>
              ) : null}
              <div className="voice-popover__line">
                <span className="voice-popover__what">{t("voicePopover.dictate", { defaultValue: "Dictate" })}</span>
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
      {choosing && runtime?.installed ? (
        <ChatVoiceDrawer
          use={choosing}
          catalog={catalog}
          currentId={choosing === "read" ? settings.ttsVoice : choosing === "call" ? callId : fallbackId}
          speed={speed}
          onAssign={(id) => saveSettings({ [VOICE_USE_SETTING[choosing]]: id })}
          onClose={() => setChoosing(null)}
          onGetMore={() => {
            setChoosing(null);
            setOpen(false);
            openVoiceModels();
          }}
        />
      ) : null}
    </div>
  );
}
