import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import type { VoiceCapabilities, VoiceModelRow } from "../../../types";
import { costLabel } from "./speakLogic";
import { useVoiceCatalog } from "./useVoiceCatalog";

const CAPS: Array<keyof VoiceCapabilities> = ["speak", "style", "clone", "design"];
const CAP_LABELS: Record<keyof VoiceCapabilities, string> = { speak: "Speak", style: "Style", clone: "Clone", design: "Design" };

/** Download, memory, speed, cold start and where it was measured; an unmeasured model says so instead of guessing. */
function CostCells({ model }: { model: VoiceModelRow }) {
  const { t } = useTranslation("studio");
  const cost = costLabel(model);
  const cells: Array<[string, string]> = [[t("voice.models.cost.download", { defaultValue: "Download size" }), `${model.sizeGb} GB`]];
  if (cost.kind === "measured") {
    cells.push(
      [t("voice.models.cost.memory", { defaultValue: "Memory" }), t("voice.models.cost.peak", { defaultValue: "{gb} GB peak", gb: cost.peakGb })],
      [t("voice.models.cost.speed", { defaultValue: "Speed" }), t("voice.models.cost.per10s", { defaultValue: "{seconds} s per 10 s", seconds: cost.secondsPer10s })],
    );
    if (cost.coldStartS != null) {
      cells.push([t("voice.models.cost.coldStart", { defaultValue: "Cold start" }), t("voice.models.cost.seconds", { defaultValue: "{seconds} s", seconds: cost.coldStartS })]);
    }
    cells.push([t("voice.models.cost.measuredOn", { defaultValue: "Measured on" }), cost.measuredOn]);
  } else {
    cells.push([t("voice.models.cost.speed", { defaultValue: "Speed" }), t("voice.models.cost.notMeasured", { defaultValue: "Not measured yet" })]);
  }
  return (
    <dl className="voice-model-row__cost">
      {cells.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function TtsModelRow({ model, onDownload }: { model: VoiceModelRow; onDownload: () => void }) {
  const { t } = useTranslation("studio");
  return (
    <article className="image-library-card voice-model-card voice-tts-row">
      <div className="image-library-card-head">
        <strong className="voice-model-card__name">{model.name}</strong>
        <div className="voice-model-card__badges">
          {model.installed ? <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span> : null}
        </div>
      </div>
      <p className="muted-text voice-model-card__meta">
        {model.sizeGb} GB · {model.languages}. {model.summary}
      </p>
      <CostCells model={model} />
      <div className="voice-model-row__caps" aria-label={t("voice.models.capabilities", { defaultValue: "What it can do" })}>
        {CAPS.filter((cap) => model.capabilities[cap]).map((cap) => (
          <span key={cap} className="badge accent">
            {t(`voice.models.caps.${cap}`, { defaultValue: CAP_LABELS[cap] })}
          </span>
        ))}
      </div>
      <div className="voice-model-card__actions">
        {model.installed ? null : !model.supported ? (
          <span className="muted-text">{t("voice.speak.appleOnly", { defaultValue: "Apple Silicon Macs only for now." })}</span>
        ) : model.download.state === "downloading" ? (
          <span className="muted-text" aria-live="polite">
            {model.download.phase === "watermark"
              ? t("voice.clone.installingWatermark", { defaultValue: "Installing the watermark component…" })
              : t("voice.clone.downloading", { defaultValue: "Downloading…" })}
          </span>
        ) : (
          <button type="button" className="secondary-button" onClick={onDownload}>
            {t("voice.models.download", { defaultValue: "Download" })}
          </button>
        )}
      </div>
      {model.download.error ? (
        <p className="voice-error" role="alert">
          {model.download.error}
        </p>
      ) : null}
    </article>
  );
}

/** The Qwen3-TTS models: what each one can do, its size and whether it is downloaded. */
export function TtsModelsPanel() {
  const { t } = useTranslation("studio");
  const catalog = useVoiceCatalog(true);
  return (
    <Panel
      title={t("voice.models.ttsTitle", { defaultValue: "Speech models" })}
      subtitle={t("voice.models.ttsSubtitle", {
        defaultValue: "Natural voices, style, voice design and cloning. The quick voices (Kokoro) come with the voice pack.",
      })}
      className="span-2"
    >
      {catalog.error ? (
        <p className="voice-error" role="alert">
          {catalog.error}
        </p>
      ) : null}
      {catalog.loaded && catalog.models.length === 0 && !catalog.error ? (
        <p className="muted-text">{t("voice.models.ttsNone", { defaultValue: "No speech models are listed." })}</p>
      ) : null}
      <div className="image-library-grid">
        {catalog.models.map((model) => (
          <TtsModelRow key={model.key} model={model} onDownload={() => void catalog.download(model.key)} />
        ))}
      </div>
    </Panel>
  );
}
