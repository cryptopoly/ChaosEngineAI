import { useTranslation } from "react-i18next";
import type { VoiceModelRow } from "../../../types";
import { costLabel } from "../speak/speakLogic";
import { useCostText } from "../speak/costText";
import { offeredCloneModels, modelToDownload } from "./cloneLogic";
import type { CloneModel } from "./cloneApi";

/**
 * The "Cloning model" choice in the new-voice form: one radio per downloaded cloning model (name, download size and
 * the measured cost), the default marked with a plain tag. With none downloaded it offers the download instead.
 */
export function CloneModelPicker({
  models,
  selected,
  onSelect,
  onDownload,
  title,
}: {
  models: CloneModel[];
  selected: string;
  onSelect: (key: string) => void;
  onDownload: (key: string) => void;
  /** The legend, e.g. "4. Model" when the form numbers its steps. */
  title?: string;
}) {
  const { t } = useTranslation("studio");
  const cost = useCostText();
  const legend = title ?? t("voice.clone.modelTitle", { defaultValue: "Cloning model" });
  const offered = offeredCloneModels(models);
  if (offered.length === 0) {
    const fetching = models.find((m) => m.download.state === "downloading");
    const target = fetching ?? modelToDownload(models);
    return (
      <fieldset className="voice-clone__model">
        <legend>{legend}</legend>
        <p className="muted-text">{t("voice.clone.modelNone", { defaultValue: "No cloning model is downloaded yet." })}</p>
        {target ? (
          <button type="button" className="secondary-button" disabled={fetching !== undefined} onClick={() => onDownload(target.key)}>
            {fetching
              ? t("voice.clone.downloading", { defaultValue: "Downloading…" })
              : t("voice.clone.modelDownload", { defaultValue: "Download {name} ({gb} GB)", name: target.name, gb: target.downloadGb ?? target.sizeGb })}
          </button>
        ) : models.length > 0 ? (
          <p className="muted-text">{t("voice.clone.modelNoneHere", { defaultValue: "No cloning model runs on this computer." })}</p>
        ) : null}
      </fieldset>
    );
  }
  return (
    <fieldset className="voice-clone__model">
      <legend>{legend}</legend>
      {offered.map((m) => (
        <label key={m.key} className="voice-clone__model-row">
          <input type="radio" name="clone-model" checked={selected === m.key} onChange={() => onSelect(m.key)} /> <strong>{m.name}</strong>
          {m.default ? <span className="badge muted">{t("voice.clone.modelDefault", { defaultValue: "Default" })}</span> : null}
          <span className="muted-text">
            {" "}
            {t("voice.clone.modelSize", { defaultValue: "{gb} GB download", gb: m.downloadGb ?? m.sizeGb })}
            {", "}
            {/* The row carries the same size and estimates the voice models table does. */}
            {cost.chip(costLabel(m as unknown as VoiceModelRow))}
          </span>
        </label>
      ))}
    </fieldset>
  );
}
