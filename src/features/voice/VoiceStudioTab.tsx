import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import type { TabId, VoiceRuntime } from "../../types";
import { SpeakScreen } from "./speak/SpeakScreen";
import { VoicesScreen } from "./speak/VoicesScreen";
import { DESIGN_VOICE_ID, defaultVoice } from "./speak/speakLogic";
import { useVoiceCatalog } from "./speak/useVoiceCatalog";
import { useVoice } from "./VoiceContext";
import { TranscribePanel, type OpenRequest } from "./studio/TranscribePanel";

export { pickDefaultSttModel } from "./studio/TranscribePanel";
export type { OpenRequest } from "./studio/TranscribePanel";

export interface VoiceStudioTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  /** A Library transcript to open in the editor. */
  openRequest: OpenRequest | null;
  onSendToChat: (text: string) => void;
  onTabChange: (tab: TabId) => void;
}

type StudioMode = "transcribe" | "speak" | "voices";

export function VoiceStudioTab({ voiceRuntime, backendOnline, openRequest, onSendToChat, onTabChange }: VoiceStudioTabProps) {
  const { t, i18n } = useTranslation("studio");
  const [mode, setMode] = useState<StudioMode>("transcribe");
  const { settings, saveSettings } = useVoice();
  const catalog = useVoiceCatalog(backendOnline && Boolean(voiceRuntime?.installed));
  const [voiceId, setVoiceId] = useState("");
  // Until the person picks a voice, Speak follows the default rule, so a natural voice that finishes downloading
  // becomes the starting voice; a chosen voice that disappears (a deleted clone) falls back to the rule too.
  const picked = useRef(false);
  const speakVoice = settings.speakVoice ?? "";

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

  useEffect(() => {
    if (openRequest) setMode("transcribe");
  }, [openRequest]);

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
    <div className="content-grid discover-page voice-page voice-studio-page">
      <div className="segmented voice-studio__modes span-2" role="tablist" aria-label={t("voice.studio.title", { defaultValue: "Voice Studio" })}>
        {(["transcribe", "speak", "voices"] as const).map((value) => (
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
      {/* All three stay mounted so a running queue or generation survives a switch. */}
      <div className={`voice-studio__pane span-2${mode === "transcribe" ? "" : " voice-studio__pane--hidden"}`}>
        <TranscribePanel
          runtime={voiceRuntime}
          openRequest={openRequest}
          onSendToChat={onSendToChat}
          onOpenModels={() => onTabChange("voice-models")}
        />
      </div>
      <div className={`voice-studio__pane span-2${mode === "speak" ? "" : " voice-studio__pane--hidden"}`}>
        <SpeakScreen catalog={catalog} voiceId={voiceId} onVoiceChange={pickVoice} onTabChange={onTabChange} />
      </div>
      <div className={`voice-studio__pane span-2${mode === "voices" ? "" : " voice-studio__pane--hidden"}`}>
        <VoicesScreen
          catalog={catalog}
          onUse={useVoiceInSpeak}
          onDesign={() => useVoiceInSpeak(DESIGN_VOICE_ID)}
          onGetMore={() => onTabChange("voice-models")}
        />
      </div>
    </div>
  );
}
