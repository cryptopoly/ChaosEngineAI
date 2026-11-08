import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { StudioShell } from "../../components/StudioShell";
import type { TabId, VoiceRuntime } from "../../types";
import { SpeakScreen } from "./speak/SpeakScreen";
import { VoicesScreen } from "./speak/VoicesScreen";
import { DESIGN_VOICE_ID, defaultVoice } from "./speak/speakLogic";
import { useVoiceCatalog } from "./speak/useVoiceCatalog";
import { useVoice } from "./VoiceContext";
import { TranscribePanel, type OpenRequest as TranscriptRequest } from "./studio/TranscribePanel";
import { STUDIO_MODES, readStudioMode, requestedMode, writeStudioMode, type StudioMode, type VoiceStudioRequest } from "./voiceStudioMode";

export { pickDefaultSttModel } from "./studio/TranscribePanel";
/** What another tab asks the Studio to open: a transcript, a mode, a voice on Speak. */
export type OpenRequest = VoiceStudioRequest;

export interface VoiceStudioTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  /** A Library transcript to open, or a mode and voice to show (Models > Use in Speak). */
  openRequest: OpenRequest | null;
  onSendToChat: (text: string) => void;
  onTabChange: (tab: TabId) => void;
}

// A request is acted on once, not again each time the Studio is opened while it is still the latest one.
let handledNonce = 0;

function ModelsIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2.5" y="2.5" width="11" height="4" rx="1" />
      <rect x="2.5" y="9.5" width="11" height="4" rx="1" />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 2.5v11M6.5 2.5v11M10 3l3 10.2" />
    </svg>
  );
}

export function VoiceStudioTab({ voiceRuntime, backendOnline, openRequest, onSendToChat, onTabChange }: VoiceStudioTabProps) {
  const { t, i18n } = useTranslation("studio");
  const [mode, setModeState] = useState<StudioMode>(readStudioMode);
  const { settings, saveSettings } = useVoice();
  const catalog = useVoiceCatalog(backendOnline && Boolean(voiceRuntime?.installed));
  const [voiceId, setVoiceId] = useState("");
  const [speakText, setSpeakText] = useState<{ text: string; nonce: number } | null>(null);
  // Until the person picks a voice, Speak follows the default rule, so a natural voice that finishes downloading
  // becomes the starting voice; a chosen voice that disappears (a deleted clone) falls back to the rule too.
  const picked = useRef(false);
  const speakVoice = settings.speakVoice ?? "";

  const setMode = (next: StudioMode) => {
    setModeState(next);
    writeStudioMode(next);
  };

  useEffect(() => {
    if (!catalog.loaded) return;
    if (picked.current && catalog.voices.some((v) => v.id === voiceId)) return;
    const next = defaultVoice(catalog.voices, speakVoice, i18n.language);
    if (next && next !== voiceId) setVoiceId(next);
  }, [catalog.loaded, catalog.voices, voiceId, speakVoice, i18n.language]);

  /** The person chose a voice: it becomes the Speak voice next time too (the chat voice is not touched). */
  const pickVoice = (id: string) => {
    picked.current = true;
    setVoiceId(id);
    if (id !== speakVoice) void saveSettings({ speakVoice: id });
  };

  const useVoiceInSpeak = (id: string) => {
    pickVoice(id);
    setMode("speak");
  };

  // The transcript part of a request goes to the Transcribe editor; the mode and voice are handled here.
  const transcriptRequest = useMemo<TranscriptRequest | null>(
    () => (openRequest?.id ? { id: openRequest.id, nonce: openRequest.nonce } : null),
    [openRequest],
  );

  useEffect(() => {
    if (!openRequest || openRequest.nonce === handledNonce) return;
    handledNonce = openRequest.nonce;
    const wanted = requestedMode(openRequest);
    if (wanted) setMode(wanted);
    if (openRequest.voice) pickVoice(openRequest.voice);
    if (openRequest.text != null) setSpeakText({ text: openRequest.text, nonce: openRequest.nonce });
    // pickVoice and setMode only set state; the request is what matters here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openRequest]);

  const title = t("voice.studio.title", { defaultValue: "Voice Studio" });

  if (!backendOnline || !voiceRuntime) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={title} className="span-2">
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
        <Panel title={title} className="span-2">
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

  const header = (
    <>
      <div className="voice-studio__title">
        <h2>{title}</h2>
        <div className="segmented voice-studio__modes" role="tablist" aria-label={title}>
          {STUDIO_MODES.map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={mode === value}
              className={mode === value ? "segment active" : "segment"}
              onClick={() => setMode(value)}
            >
              {value === "transcribe"
                ? t("voice.transcribe.title", { defaultValue: "Transcribe" })
                : value === "speak"
                  ? t("voice.speak.title", { defaultValue: "Speak" })
                  : t("voice.voices.tab", { defaultValue: "Voices" })}
            </button>
          ))}
        </div>
      </div>
      <div className="button-row voice-studio__links">
        <button type="button" className="secondary-button" onClick={() => onTabChange("voice-models")}>
          <ModelsIcon /> {t("studioKit.models", { defaultValue: "Models" })}
        </button>
        <button type="button" className="secondary-button" onClick={() => onTabChange("voice-gallery")}>
          <LibraryIcon /> {t("voice.studio.library", { defaultValue: "Library" })}
        </button>
      </div>
    </>
  );

  // All three stay mounted so a running queue or generation survives a switch.
  const panes = (
    <>
      <div className={`voice-studio__pane${mode === "transcribe" ? "" : " voice-studio__pane--hidden"}`}>
        <TranscribePanel
          runtime={voiceRuntime}
          openRequest={transcriptRequest}
          onSendToChat={onSendToChat}
          onOpenModels={() => onTabChange("voice-models")}
        />
      </div>
      <div className={`voice-studio__pane${mode === "speak" ? "" : " voice-studio__pane--hidden"}`}>
        <SpeakScreen
          catalog={catalog}
          voiceId={voiceId}
          onVoiceChange={pickVoice}
          onTabChange={onTabChange}
          onNewClone={() => setMode("voices")}
          textRequest={speakText}
        />
      </div>
      <div className={`voice-studio__pane${mode === "voices" ? "" : " voice-studio__pane--hidden"}`}>
        <VoicesScreen
          catalog={catalog}
          onUse={useVoiceInSpeak}
          onDesign={() => useVoiceInSpeak(DESIGN_VOICE_ID)}
          onGetMore={() => onTabChange("voice-models")}
        />
      </div>
    </>
  );

  return (
    <div className="content-grid discover-page voice-page voice-studio-page">
      <div className="span-2 voice-studio__shell">
        <StudioShell studioId="voice-studio" header={header} inputs={panes} />
      </div>
    </div>
  );
}
