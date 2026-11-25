/**
 * Voice > Models (D-73): Voice setup, the speech-to-text models and the speech models on one page, each with Delete.
 * A model is on Models, a voice is in the Studio, a setting sits next to the thing it changes: the speech-to-text
 * language in its panel, memory in Voice setup.
 */
import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import type { ModelAvailabilityFilter } from "../../components/models";
import type { VoiceRuntime } from "../../types";
import { DictationModelsPanel } from "./models/DictationModelsPanel";
import { SpeechModelsPanel } from "./models/SpeechModelsPanel";
import { useVoiceCatalog } from "./speak/useVoiceCatalog";
import { VoiceRuntimePanel, useVoicePackJob } from "./VoicePackPanel";
import type { StudioMode } from "./voiceStudioMode";

export { VOICE_LANGUAGES, voiceLanguageName } from "./voiceLanguages";

export interface VoiceModelsTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onRefreshVoiceRuntime: () => void;
  /** Open Voice Studio, optionally on a mode with a voice selected (Use in Speak, Browse voices). */
  onOpenStudio?: (target?: StudioTarget) => void;
  /** The speech models table's availability filter (kept by the app like the other Models pages). */
  availability?: ModelAvailabilityFilter;
  onAvailabilityChange?: (value: ModelAvailabilityFilter) => void;
  onOpenExternalUrl?: (url: string) => void;
}

export type StudioTarget = { mode: StudioMode; voice?: string };

export function VoiceModelsTab(props: VoiceModelsTabProps) {
  const { t } = useTranslation("studio");
  const { voiceRuntime, backendOnline } = props;

  if (!backendOnline || !voiceRuntime) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={t("voice.models.title", { defaultValue: "Voice" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">{t("voice.offline", { defaultValue: "Waiting for the backend…" })}</p>
          </div>
        </Panel>
      </div>
    );
  }

  return <VoiceModelsBody {...props} voiceRuntime={voiceRuntime} />;
}

function VoiceModelsBody({
  voiceRuntime,
  onRefreshVoiceRuntime,
  onOpenStudio,
  availability = "all",
  onAvailabilityChange = () => {},
  onOpenExternalUrl,
}: VoiceModelsTabProps & { voiceRuntime: VoiceRuntime }) {
  const { t } = useTranslation("studio");
  // The speech model list also before the install: Voice setup sizes the natural voice model from it.
  const catalog = useVoiceCatalog(true, voiceRuntime.lane);
  const refreshModels = catalog.refresh;
  const onInstallDone = useCallback(() => {
    onRefreshVoiceRuntime();
    void refreshModels();
  }, [onRefreshVoiceRuntime, refreshModels]);
  const pack = useVoicePackJob(voiceRuntime, onInstallDone);

  return (
    <div className="content-grid discover-page voice-page">
      <VoiceRuntimePanel
        runtime={voiceRuntime}
        pack={pack}
        catalog={catalog}
        onOpenClone={onOpenStudio ? () => onOpenStudio({ mode: "clone" }) : undefined}
        actions={
          voiceRuntime.installed && onOpenStudio ? (
            <button type="button" className="primary-button" onClick={() => onOpenStudio()}>
              {t("voice.models.openStudio", { defaultValue: "Open Studio" })}
            </button>
          ) : null
        }
      />
      {voiceRuntime.installed ? (
        <>
          <DictationModelsPanel runtime={voiceRuntime} onChanged={onRefreshVoiceRuntime} onOpenExternalUrl={onOpenExternalUrl} />
          <SpeechModelsPanel
            catalog={catalog}
            runtime={voiceRuntime}
            availability={availability}
            onAvailabilityChange={onAvailabilityChange}
            onOpenStudio={onOpenStudio}
            onOpenExternalUrl={onOpenExternalUrl}
          />
        </>
      ) : null}
    </div>
  );
}
