import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import type { TabId, VoiceRuntime } from "../../types";
import { SpeakPanel } from "./studio/SpeakPanel";
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

type StudioMode = "transcribe" | "speak";

export function VoiceStudioTab({ voiceRuntime, backendOnline, openRequest, onSendToChat, onTabChange }: VoiceStudioTabProps) {
  const { t } = useTranslation("studio");
  const [mode, setMode] = useState<StudioMode>("transcribe");

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
                defaultValue: "Voice runs on this computer and needs the voice pack (about 2 GB). Install it once from Voice > My Models.",
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
        {(["transcribe", "speak"] as const).map((value) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={mode === value}
            className={mode === value ? "segment active" : "segment"}
            onClick={() => setMode(value)}
          >
            {value === "transcribe" ? t("voice.transcribe.title", { defaultValue: "Transcribe" }) : t("voice.speak.title", { defaultValue: "Speak" })}
          </button>
        ))}
      </div>
      {/* Both stay mounted so a running queue or generation survives a switch. */}
      <div className={`voice-studio__pane span-2${mode === "transcribe" ? "" : " voice-studio__pane--hidden"}`}>
        <TranscribePanel
          runtime={voiceRuntime}
          openRequest={openRequest}
          onSendToChat={onSendToChat}
          onOpenModels={() => onTabChange("voice-models")}
        />
      </div>
      <div className={`voice-studio__pane span-2${mode === "speak" ? "" : " voice-studio__pane--hidden"}`}>
        <SpeakPanel runtime={voiceRuntime} />
      </div>
    </div>
  );
}
