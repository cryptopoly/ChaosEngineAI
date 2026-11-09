/**
 * Voice > Models > Dictation models (D-73 M1): the speech-to-text models with "Use for dictation", Download and a row
 * menu with Delete (confirm first; refused while voice is busy). The dictation language sits in the panel header.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { ModelActionMenu } from "../../../components/models";
import { deleteSttModel, downloadModel, getDownloadStatus } from "../../../api";
import type { DownloadStatus } from "../../../api";
import type { SttModel, VoiceRuntime } from "../../../types";
import { useVoice } from "../VoiceContext";
import { VOICE_LANGUAGES, voiceLanguageName } from "../voiceLanguages";
import { sttDeleteConfirmText } from "./voiceModelsLogic";

function SttModelCard({
  model,
  selected,
  onSelect,
  onChanged,
}: {
  model: SttModel;
  selected: boolean;
  onSelect: () => void;
  onChanged: () => void;
}) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;
  const [status, setStatus] = useState<DownloadStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const timer = window.setInterval(() => {
      void getDownloadStatus().then((all) => {
        const mine = all.find((d) => d.repo === model.id);
        if (!mine) return;
        setStatus(mine);
        if (mine.state === "completed") onChanged();
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [status, model.id, onChanged]);

  const download = async () => {
    setError(null);
    try {
      setStatus(await downloadModel(model.id));
    } catch (err) {
      setStatus({ repo: model.id, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: String(err) });
    }
  };

  const remove = async () => {
    if (!window.confirm(sttDeleteConfirmText(model.name, model.sizeGb, tf))) return;
    setError(null);
    setDeleting(true);
    try {
      await deleteSttModel(model.key ?? model.id);
      setStatus(null);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <article className={`image-library-card voice-model-card${selected ? " voice-model-card--selected" : ""}`} data-stt-model={model.key ?? model.id}>
      <div className="image-library-card-head">
        <div>
          <h3 className="voice-model-card__name">{model.name}</h3>
          <p className="muted-text voice-model-card__meta">
            {model.languages} · {model.sizeGb} GB
          </p>
        </div>
        <div className="voice-model-card__badges">
          {model.installed ? (
            <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span>
          ) : model.default ? (
            <span className="badge subtle">{t("voice.models.recommended", { defaultValue: "Recommended" })}</span>
          ) : null}
        </div>
      </div>
      <div className="voice-model-card__actions">
        <label className="checkbox-label">
          <input type="radio" name="voice-stt-model" checked={selected} onChange={onSelect} />
          {t("voice.models.use", { defaultValue: "Use for dictation" })}
        </label>
        {model.installed ? (
          deleting ? (
            <span className="muted-text" aria-live="polite">{t("voice.models.deleting", { defaultValue: "Deleting…" })}</span>
          ) : (
            <ModelActionMenu
              label={t("voice.models.moreFor", { defaultValue: "More actions for {name}", name: model.name })}
              actions={[
                {
                  id: "delete",
                  icon: "delete",
                  label: t("voice.models.deleteModel", { defaultValue: "Delete model ({gb} GB)", gb: model.sizeGb }),
                  onClick: () => void remove(),
                },
              ]}
            />
          )
        ) : status?.state === "downloading" ? (
          <span className="muted-text">{Math.round((status.progress ?? 0) * 100)}%</span>
        ) : (
          <button type="button" className="secondary-button" onClick={() => void download()}>
            {t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb })}
          </button>
        )}
      </div>
      {status?.state === "failed" ? <p className="voice-error" role="alert">{status.error}</p> : null}
      {error ? <p className="voice-error" role="alert">{error}</p> : null}
    </article>
  );
}

export function DictationModelsPanel({ runtime, onChanged }: { runtime: VoiceRuntime; onChanged: () => void }) {
  const { t, i18n } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const selectedModel =
    runtime.sttModels.find((m) => m.id === settings.sttModel || m.key === settings.sttModel)?.id ??
    runtime.sttModels.find((m) => m.default)?.id;

  return (
    <Panel
      title={t("voice.models.sttTitle", { defaultValue: "Dictation models" })}
      subtitle={t("voice.models.sttSubtitle", { defaultValue: "Turn speech into text: the microphone in Chat, voice calls and Transcribe." })}
      className="span-2 voice-dictation"
      actions={
        <label className="voice-dictation__language">
          <span>{t("voice.settings.language", { defaultValue: "Language" })}</span>
          <select className="text-input" value={settings.language} onChange={(e) => void saveSettings({ language: e.target.value })}>
            {VOICE_LANGUAGES.map(([code, label]) => (
              <option key={code} value={code}>
                {voiceLanguageName(code, label, i18n.language)}
              </option>
            ))}
          </select>
        </label>
      }
    >
      <div className="image-library-grid">
        {runtime.sttModels.map((model) => (
          <SttModelCard
            key={model.id}
            model={model}
            selected={model.id === selectedModel}
            onSelect={() => void saveSettings({ sttModel: model.key ?? model.id })}
            onChanged={onChanged}
          />
        ))}
      </div>
      <p className="muted-text voice-model-card__meta">
        {t("voice.models.firstUseLine", {
          defaultValue: "A model you pick but have not downloaded is fetched the first time you dictate. Whisper uses the language; Parakeet and Nemotron find it themselves.",
        })}
      </p>
    </Panel>
  );
}
