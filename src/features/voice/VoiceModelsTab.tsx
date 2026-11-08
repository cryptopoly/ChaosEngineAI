import { useEffect, useState } from "react";
import i18next from "i18next";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { downloadModel, getDownloadStatus, stopVoiceWorker } from "../../api";
import type { DownloadStatus } from "../../api";
import type { SttModel, VoiceRuntime } from "../../types";
import { TtsModelsPanel } from "./speak/TtsModelsPanel";
import { useVoice } from "./VoiceContext";
import { InstallLog, InstallProgress, VoicePackPanel, useVoicePackJob, type VoicePackJob } from "./VoicePackPanel";
import { chineseAddonPending, quickVoiceSummary } from "./voicePackState";
import type { StudioMode } from "./voiceStudioMode";

export interface VoiceModelsTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onRefreshVoiceRuntime: () => void;
  /** Open Voice Studio, optionally on a mode with a voice selected (Use in Speak, Browse voices). */
  onOpenStudio?: (target?: StudioTarget) => void;
}

export type StudioTarget = { mode: StudioMode; voice?: string };

const LANGUAGES: Array<[string, string]> = [
  ["", "Detect automatically"],
  ["en", "English"],
  ["de", "German"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["pl", "Polish"],
  ["ru", "Russian"],
  ["uk", "Ukrainian"],
  ["ja", "Japanese"],
  ["zh", "Chinese"],
  ["ko", "Korean"],
  ["hi", "Hindi"],
  ["ar", "Arabic"],
  ["tr", "Turkish"],
];

export const VOICE_LANGUAGES = LANGUAGES;

/** A language's name in the UI language (the list above is the English fallback). */
export function voiceLanguageName(code: string, fallback: string, uiLanguage: string): string {
  if (!code) return i18next.t("studio:voice.languages.auto", { defaultValue: fallback });
  try {
    const name = new Intl.DisplayNames([uiLanguage], { type: "language" }).of(code);
    return name ? name.charAt(0).toLocaleUpperCase(uiLanguage) + name.slice(1) : fallback;
  } catch {
    return fallback;
  }
}

function SttModelCard({
  model,
  selected,
  onSelect,
  onInstalled,
}: {
  model: SttModel;
  selected: boolean;
  onSelect: () => void;
  onInstalled: () => void;
}) {
  const { t } = useTranslation("studio");
  const [status, setStatus] = useState<DownloadStatus | null>(null);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const timer = window.setInterval(() => {
      void getDownloadStatus().then((all) => {
        const mine = all.find((d) => d.repo === model.id);
        if (!mine) return;
        setStatus(mine);
        if (mine.state === "completed") onInstalled();
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [status, model.id, onInstalled]);

  const download = async () => {
    try {
      setStatus(await downloadModel(model.id));
    } catch (err) {
      setStatus({ repo: model.id, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: String(err) });
    }
  };

  return (
    <article className={`image-library-card voice-model-card${selected ? " voice-model-card--selected" : ""}`}>
      <div className="image-library-card-head">
        <div>
          <h3 className="voice-model-card__name">{model.name}</h3>
          <p className="muted-text voice-model-card__meta">
            {model.languages} · {model.sizeGb} GB · {model.backend}
          </p>
        </div>
        <div className="voice-model-card__badges">
          {model.default ? <span className="badge subtle">{t("voice.models.recommended", { defaultValue: "Recommended" })}</span> : null}
          {model.installed ? (
            <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span>
          ) : null}
        </div>
      </div>
      <div className="voice-model-card__actions">
        <label className="checkbox-label">
          <input type="radio" name="voice-stt-model" checked={selected} onChange={onSelect} />
          {t("voice.models.use", { defaultValue: "Use for dictation" })}
        </label>
        {!model.installed ? (
          status?.state === "downloading" ? (
            <span className="muted-text">{Math.round((status.progress ?? 0) * 100)}%</span>
          ) : (
            <button type="button" className="secondary-button" onClick={() => void download()}>
              {t("voice.models.download", { defaultValue: "Download" })}
            </button>
          )
        ) : null}
      </div>
      {!model.installed && status?.state !== "downloading" ? (
        <p className="muted-text voice-model-card__meta">
          {t("voice.models.firstUse", { defaultValue: "Downloads on first use if you skip this." })}
        </p>
      ) : null}
      {status?.state === "failed" ? <p className="voice-error" role="alert">{status.error}</p> : null}
    </article>
  );
}

/** One line for the quick voices (they are chosen on Voice Studio > Voices now), the Mandarin add-on, Browse voices. */
function QuickVoicesPanel({ runtime, pack, onBrowse }: { runtime: VoiceRuntime; pack: VoicePackJob; onBrowse?: () => void }) {
  const { t } = useTranslation("studio");
  const { count, languages } = quickVoiceSummary(runtime.ttsVoices);
  const addonPending = chineseAddonPending(runtime.extras ?? []);
  return (
    <Panel title={t("voice.quickVoices.title", { defaultValue: "Quick voices" })} className="span-2">
      <div className="voice-quick-line">
        <span className="muted-text">
          {t("voice.quickVoices.line", {
            defaultValue: "Kokoro: {count} voices in {languages} languages, part of the voice pack, under 1 GB.",
            count,
            languages,
          })}
        </span>
        {addonPending ? (
          pack.job.running ? (
            <InstallProgress pack={pack} />
          ) : (
            <button type="button" className="secondary-button" onClick={() => void pack.start(["chinese"])}>
              {t("voice.picker.addMandarin", { defaultValue: "Add Mandarin voices" })}
            </button>
          )
        ) : null}
        {onBrowse ? (
          <button type="button" className="secondary-button" onClick={onBrowse}>
            {t("voice.quickVoices.browse", { defaultValue: "Browse voices" })}
          </button>
        ) : null}
      </div>
      {pack.job.extras.includes("chinese") || pack.error ? <InstallLog pack={pack} /> : null}
    </Panel>
  );
}

/** Dictation language and voice memory. How chat speaks lives on Voice Studio > Voices ("Chat and calls"). */
function SettingsPanel({ runtime }: { runtime: VoiceRuntime }) {
  const { t, i18n } = useTranslation("studio");
  const { settings, saveSettings, refresh } = useVoice();
  return (
    <Panel
      title={t("voice.settings.runtimeTitle", { defaultValue: "Dictation and memory" })}
      subtitle={t("voice.settings.chatMoved", { defaultValue: "How chat reads aloud and voice calls: Voice Studio > Voices." })}
      className="span-2"
    >
      <div className="voice-settings">
        <label className="voice-settings__field">
          <span>{t("voice.settings.language", { defaultValue: "Dictation language" })}</span>
          <select className="text-input" value={settings.language} onChange={(e) => void saveSettings({ language: e.target.value })}>
            {LANGUAGES.map(([code, label]) => (
              <option key={code} value={code}>
                {voiceLanguageName(code, label, i18n.language)}
              </option>
            ))}
          </select>
          <small className="muted-text">
            {t("voice.settings.languageHint", { defaultValue: "Whisper uses it; Parakeet detects the language itself." })}
          </small>
        </label>
        <label className="voice-settings__field">
          <span>{t("voice.settings.idle", { defaultValue: "Free voice memory after (minutes idle, 0 = never)" })}</span>
          <input
            type="number"
            className="text-input"
            min={0}
            max={240}
            value={settings.idleUnloadMinutes}
            onChange={(e) => void saveSettings({ idleUnloadMinutes: Math.max(0, Math.min(240, Number(e.target.value) || 0)) })}
          />
        </label>
        <div className="voice-settings__worker">
          <span className="muted-text">
            {runtime.workerPid
              ? t("voice.settings.workerRunning", { defaultValue: "Voice models are loaded (process {pid}).", pid: runtime.workerPid })
              : t("voice.settings.workerStopped", { defaultValue: "Voice models are not loaded." })}
          </span>
          {runtime.workerPid ? (
            <button type="button" className="secondary-button" onClick={() => void stopVoiceWorker().then(refresh)}>
              {t("voice.settings.unload", { defaultValue: "Unload now" })}
            </button>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

export function VoiceModelsTab({ voiceRuntime, backendOnline, onRefreshVoiceRuntime, onOpenStudio }: VoiceModelsTabProps) {
  const { t } = useTranslation("studio");

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

  return <VoiceModelsBody voiceRuntime={voiceRuntime} onRefreshVoiceRuntime={onRefreshVoiceRuntime} onOpenStudio={onOpenStudio} />;
}

function VoiceModelsBody({
  voiceRuntime,
  onRefreshVoiceRuntime,
  onOpenStudio,
}: {
  voiceRuntime: VoiceRuntime;
  onRefreshVoiceRuntime: () => void;
  onOpenStudio?: (target?: StudioTarget) => void;
}) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const pack = useVoicePackJob(voiceRuntime, onRefreshVoiceRuntime);
  const selectedModel =
    voiceRuntime.sttModels.find((m) => m.id === settings.sttModel || m.key === settings.sttModel)?.id ??
    voiceRuntime.sttModels.find((m) => m.default)?.id;

  return (
    <div className="content-grid discover-page voice-page">
      <VoicePackPanel
        runtime={voiceRuntime}
        pack={pack}
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
          <Panel
            title={t("voice.models.sttTitle", { defaultValue: "Speech-to-text models" })}
            subtitle={t("voice.models.sttSubtitle", { defaultValue: "Used by the microphone in chat and by Voice Studio." })}
            className="span-2"
          >
            <div className="image-library-grid">
              {voiceRuntime.sttModels.map((model) => (
                <SttModelCard
                  key={model.id}
                  model={model}
                  selected={model.id === selectedModel}
                  onSelect={() => void saveSettings({ sttModel: model.key ?? model.id })}
                  onInstalled={onRefreshVoiceRuntime}
                />
              ))}
            </div>
          </Panel>
          <TtsModelsPanel onOpenStudio={onOpenStudio} />
          <QuickVoicesPanel runtime={voiceRuntime} pack={pack} onBrowse={onOpenStudio ? () => onOpenStudio({ mode: "voices" }) : undefined} />
          <SettingsPanel runtime={voiceRuntime} />
        </>
      ) : null}
    </div>
  );
}
