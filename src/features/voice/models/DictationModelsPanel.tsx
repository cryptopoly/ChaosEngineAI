/**
 * Voice > Models > Dictation models (D-73 M1, D-102 D): the speech-to-text models with "Use for dictation", a line
 * on what each is for, its licence by name with a link, the GPU memory it needs (measured or estimated), Download
 * with a bar, size and Cancel, and a row menu with Delete (confirm first; refused while voice is busy). The model that
 * is or becomes the default is badged. With the GPU dictation models comes the "Word timings" download (the forced
 * aligner, D-101). The dictation language sits in the panel header.
 */
import { useEffect, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { ModelActionMenu } from "../../../components/models";
import { cancelDownload, deleteSttModel, downloadModel, getDownloadStatus } from "../../../api";
import type { DownloadStatus } from "../../../api";
import type { SttAligner, SttModel, VoiceRuntime } from "../../../types";
import { useVoice } from "../VoiceContext";
import { VOICE_LANGUAGES, voiceLanguageName } from "../voiceLanguages";
import { gpuDefaultPending, licenceLabel, licenceUrl, sttDeleteConfirmText, sttFitText, sttGuideText } from "./voiceModelsLogic";

type TF = (key: string, options?: Record<string, unknown>) => string;

/** Why this GPU cannot run a GPU dictation row; the same words as the speech models' rows. */
function gpuFactsText(facts: NonNullable<SttModel["gpuFacts"]>, t: TF): string {
  if (facts.kind === "compute") {
    return t("voice.speak.gpuTooOld", {
      need: facts.need,
      have: facts.have,
      defaultValue: "Needs an NVIDIA GPU of compute capability {need} or newer; this one is {have}.",
    });
  }
  return t("voice.speak.gpuTooSmall", {
    need: facts.need,
    have: facts.have,
    defaultValue: "Needs about {need} GB of GPU memory; this GPU has {have} GB.",
  });
}

/** A Hugging Face download of one repo: start, follow, cancel. Picks up a download already running (after a reload). */
function useRepoDownload(repo: string, onChanged: () => void) {
  const [status, setStatus] = useState<DownloadStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getDownloadStatus()
      .then((all) => {
        const mine = all.find((d) => d.repo === repo);
        if (!cancelled && mine?.state === "downloading") setStatus(mine);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [repo]);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const timer = window.setInterval(() => {
      void getDownloadStatus().then((all) => {
        const mine = all.find((d) => d.repo === repo);
        if (!mine) return;
        setStatus(mine);
        if (mine.state === "completed") onChanged();
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [status, repo, onChanged]);

  const start = async () => {
    try {
      setStatus(await downloadModel(repo));
    } catch (err) {
      setStatus({ repo, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: String(err) });
    }
  };
  const cancel = async () => {
    try {
      setStatus(await cancelDownload(repo));
    } catch (err) {
      setStatus({ repo, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: String(err) });
    }
  };
  return { status, setStatus, start, cancel };
}

/** The running download: a bar, the GB so far against the size, Cancel; announced as a progress bar. */
export function SttDownloadProgress({ name, status, sizeGb, onCancel }: { name: string; status: DownloadStatus; sizeGb: number; onCancel: () => void }) {
  const { t } = useTranslation("studio");
  const percent = Math.round((status.progress ?? 0) * 100);
  const total = status.totalGb ?? sizeGb;
  return (
    <div className="voice-model-card__progress">
      <div className="progress-meta">
        <span aria-live="polite">
          {t("voice.models.sttDownloading", {
            defaultValue: "{percent}% · {done} of {total} GB",
            percent,
            done: (status.downloadedGb ?? 0).toFixed(2),
            total: total.toFixed(2),
          })}
        </span>
        <button type="button" className="secondary-button" onClick={onCancel}>
          {t("voice.models.cancelDownload", { defaultValue: "Cancel" })}
        </button>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label={t("voice.models.sttDownloadingName", { defaultValue: "Downloading {name}", name })}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <div className="progress-fill" style={{ width: `${Math.max(2, percent)}%` }} />
      </div>
    </div>
  );
}

/** Size and licence by name, the licence linking to the model's page. */
function MetaLine({ repo, sizeGb, licence, lead, onOpenUrl }: { repo: string; sizeGb: number; licence?: string | null; lead?: string; onOpenUrl: (url: string) => void }) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TF;
  const name = licenceLabel(licence, tf);
  return (
    <p className="muted-text voice-model-card__meta">
      {lead ? `${lead} · ` : ""}
      {sizeGb} GB
      {name ? (
        <>
          {" · "}
          <button
            type="button"
            className="link-button"
            title={t("voice.models.licenceOpen", { defaultValue: "Read the licence on the model page" })}
            onClick={() => onOpenUrl(licenceUrl(repo))}
          >
            {name}
          </button>
        </>
      ) : null}
    </p>
  );
}

function DownloadOrMenu({
  installed,
  deleting,
  download,
  cannotRun,
  name,
  sizeGb,
  onDelete,
}: {
  installed: boolean;
  deleting: boolean;
  download: ReturnType<typeof useRepoDownload>;
  cannotRun: boolean;
  name: string;
  sizeGb: number;
  onDelete: () => void;
}): ReactNode {
  const { t } = useTranslation("studio");
  if (installed) {
    return deleting ? (
      <span className="muted-text" aria-live="polite">{t("voice.models.deleting", { defaultValue: "Deleting…" })}</span>
    ) : (
      <ModelActionMenu
        label={t("voice.models.moreFor", { defaultValue: "More actions for {name}", name })}
        actions={[
          { id: "delete", icon: "delete", label: t("voice.models.deleteModel", { defaultValue: "Delete model ({gb} GB)", gb: sizeGb }), onClick: onDelete },
        ]}
      />
    );
  }
  if (download.status?.state === "downloading") {
    return <SttDownloadProgress name={name} status={download.status} sizeGb={sizeGb} onCancel={() => void download.cancel()} />;
  }
  if (cannotRun) return null;
  return (
    <button type="button" className="secondary-button" onClick={() => void download.start()}>
      {t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: sizeGb })}
    </button>
  );
}

function useDelete(key: string, name: string, sizeGb: number, onChanged: () => void, onDeleted: () => void) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TF;
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const remove = async () => {
    if (!window.confirm(sttDeleteConfirmText(name, sizeGb, tf))) return;
    setError(null);
    setDeleting(true);
    try {
      await deleteSttModel(key);
      onDeleted();
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(false);
    }
  };
  return { deleting, error, setError, remove };
}

function SttModelCard({
  model,
  selected,
  recommended,
  onSelect,
  onChanged,
  onOpenUrl,
}: {
  model: SttModel;
  selected: boolean;
  /** The GPU row that becomes the default once it is downloaded (D-102 C). */
  recommended: boolean;
  onSelect: () => void;
  onChanged: () => void;
  onOpenUrl: (url: string) => void;
}) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TF;
  const download = useRepoDownload(model.id, onChanged);
  const removal = useDelete(model.key ?? model.id, model.name, model.sizeGb, onChanged, () => download.setStatus(null));
  // A GPU row this GPU cannot run (memory, compute capability): not offered for download or dictation, as on the
  // speech models' rows. One that is already on disk keeps its Delete.
  const cannotRun = Boolean(model.gpuFacts) && !model.installed;
  const guide = sttGuideText(model.key, tf);
  const fit = sttFitText(model, tf);

  return (
    <article className={`image-library-card voice-model-card${selected ? " voice-model-card--selected" : ""}`} data-stt-model={model.key ?? model.id}>
      <div className="image-library-card-head">
        <div>
          <h3 className="voice-model-card__name">{model.name}</h3>
          {guide ? <p className="voice-model-card__guide">{guide}</p> : null}
          <MetaLine repo={model.id} sizeGb={model.sizeGb} licence={model.licence} lead={model.languages} onOpenUrl={onOpenUrl} />
          {model.note ? <p className="muted-text voice-model-card__meta">{model.note}</p> : null}
          {model.gpuFacts ? <p className="voice-error">{gpuFactsText(model.gpuFacts, tf)}</p> : null}
        </div>
        <div className="voice-model-card__badges">
          {model.default ? (
            <span className="badge accent" title={t("voice.models.defaultTitle", { defaultValue: "Dictation uses this model unless you pick another." })}>
              {t("voice.models.defaultBadge", { defaultValue: "Default" })}
            </span>
          ) : recommended ? (
            <span className="badge subtle" title={t("voice.models.recommendedGpuTitle", { defaultValue: "Becomes the default once it is downloaded." })}>
              {t("voice.models.recommendedGpu", { defaultValue: "Recommended for this GPU" })}
            </span>
          ) : null}
          {model.installed ? <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span> : null}
          {fit ? (
            <span className="badge muted" title={fit.title}>
              {fit.text}
            </span>
          ) : null}
        </div>
      </div>
      <div className="voice-model-card__actions">
        <label className="checkbox-label">
          <input type="radio" name="voice-stt-model" checked={selected} disabled={cannotRun && !selected} onChange={onSelect} />
          {t("voice.models.use", { defaultValue: "Use for dictation" })}
        </label>
        <DownloadOrMenu
          installed={model.installed}
          deleting={removal.deleting}
          download={download}
          cannotRun={cannotRun}
          name={model.name}
          sizeGb={model.sizeGb}
          onDelete={() => void removal.remove()}
        />
      </div>
      {download.status?.state === "failed" ? <p className="voice-error" role="alert">{download.status.error}</p> : null}
      {removal.error ? <p className="voice-error" role="alert">{removal.error}</p> : null}
    </article>
  );
}

/** The forced aligner as its own small download (D-101): word timings for the dictation models that have none. */
function WordTimingsCard({ aligner, onChanged, onOpenUrl }: { aligner: SttAligner; onChanged: () => void; onOpenUrl: (url: string) => void }) {
  const { t } = useTranslation("studio");
  const download = useRepoDownload(aligner.id, onChanged);
  const removal = useDelete(aligner.key, aligner.name, aligner.sizeGb, onChanged, () => download.setStatus(null));
  return (
    <article className="image-library-card voice-model-card voice-word-timings" data-stt-aligner={aligner.key}>
      <div className="image-library-card-head">
        <div>
          <h3 className="voice-model-card__name">{t("voice.models.wordTimings", { defaultValue: "Word timings" })}</h3>
          <p className="voice-model-card__guide">
            {t("voice.models.wordTimingsLine", {
              defaultValue:
                "Times each word (captions, the transcript editor) for Qwen3-ASR, Granite Speech and Nemotron Transcribe, which have no word times of their own; Nemotron ASR uses it too. Adds about {vram} GB of GPU memory while in use.",
              vram: aligner.vramGb,
            })}
          </p>
          <MetaLine repo={aligner.id} sizeGb={aligner.sizeGb} licence={aligner.licence} lead={aligner.name} onOpenUrl={onOpenUrl} />
        </div>
        <div className="voice-model-card__badges">
          {aligner.installed ? <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span> : null}
        </div>
      </div>
      <div className="voice-model-card__actions">
        <DownloadOrMenu
          installed={aligner.installed}
          deleting={removal.deleting}
          download={download}
          cannotRun={false}
          name={aligner.name}
          sizeGb={aligner.sizeGb}
          onDelete={() => void removal.remove()}
        />
      </div>
      {download.status?.state === "failed" ? <p className="voice-error" role="alert">{download.status.error}</p> : null}
      {removal.error ? <p className="voice-error" role="alert">{removal.error}</p> : null}
    </article>
  );
}

export function DictationModelsPanel({
  runtime,
  onChanged,
  onOpenExternalUrl,
}: {
  runtime: VoiceRuntime;
  onChanged: () => void;
  onOpenExternalUrl?: (url: string) => void;
}) {
  const { t, i18n } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const selectedModel =
    runtime.sttModels.find((m) => m.id === settings.sttModel || m.key === settings.sttModel)?.id ??
    runtime.sttModels.find((m) => m.default)?.id;
  // The first GPU row is the default once it is on disk (voice_runtime.list_stt_models); until then Whisper is. Said
  // here and badged while no model was picked by hand, so the switch after the download is not a surprise.
  const gpuDefault = gpuDefaultPending(runtime.sttModels);
  const openUrl = (url: string) => (onOpenExternalUrl ? onOpenExternalUrl(url) : void window.open(url, "_blank", "noopener"));

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
            recommended={model.id === gpuDefault?.id}
            onSelect={() => void saveSettings({ sttModel: model.key ?? model.id })}
            onChanged={onChanged}
            onOpenUrl={openUrl}
          />
        ))}
        {runtime.aligner ? <WordTimingsCard aligner={runtime.aligner} onChanged={onChanged} onOpenUrl={openUrl} /> : null}
      </div>
      <p className="muted-text voice-model-card__meta">
        {t("voice.models.firstUseLanguageLine", {
          defaultValue:
            "A model you pick but have not downloaded is fetched the first time you dictate. Whisper, Qwen3-ASR and Nemotron 3.5 ASR use the Language above (or find it when it is set to detect); Parakeet, Granite and Nemotron Transcribe always find it themselves.",
        })}
        {gpuDefault && !settings.sttModel
          ? ` ${t("voice.models.gpuDefaultLine", {
              defaultValue: "{name} becomes the dictation model once it is downloaded, unless you pick another.",
              name: gpuDefault.name,
            })}`
          : ""}
      </p>
    </Panel>
  );
}
