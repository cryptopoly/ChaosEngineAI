/**
 * "Choose parts": the voice runtime's optional parts as tick boxes, named by what they let you do. Before the first
 * install they choose what the install adds; after it, ticking installs the part and unticking removes it. Before
 * install a speech model can be ticked too (the Full preset ticks it).
 */
import { useTranslation } from "react-i18next";
import type { VoiceExtra } from "../../../types";
import { addonSizeText } from "../runtimeSize";
import { gbText } from "./voiceSetupLogic";

// i18n-extract: voice.setup.part.kokoro
// i18n-extract: voice.setup.part.chinese
// i18n-extract: voice.setup.part.speech-gpu
// i18n-extract: voice.setup.part.stt-gpu
const PART_LABELS: Record<string, string> = {
  kokoro: "Quick voices (Kokoro, about {size})",
  chinese: "Mandarin Chinese quick voices (8 voices, about {size})",
  "speech-gpu": "Natural voices, cloning and design on this GPU (about {size})",
  "stt-gpu": "More accurate speech to text on this GPU (about {size})",
};

/** The speech model the pre-install list can tick. */
export interface ModelChoice {
  name: string;
  gb: number | null;
}

export function ChooseParts({
  extras,
  chosen,
  onToggle,
  disabled,
  installed,
  model,
  modelChosen,
  onToggleModel,
}: {
  extras: VoiceExtra[];
  chosen: (extra: VoiceExtra) => boolean;
  onToggle: (extra: VoiceExtra, on: boolean) => void;
  disabled: boolean;
  installed: boolean;
  model?: ModelChoice | null;
  modelChosen?: boolean;
  onToggleModel?: (on: boolean) => void;
}) {
  const { t } = useTranslation("studio");
  if (extras.length === 0 && !model) return null;
  return (
    <details className="voice-setup__parts">
      <summary>{t("voice.setup.chooseParts", { defaultValue: "Choose parts" })}</summary>
      <fieldset className="voice-runtime__addons" disabled={disabled}>
        <legend>{t("voice.setup.optionalParts", { defaultValue: "Optional parts" })}</legend>
        {extras.map((extra) => (
          <label key={extra.id} className="checkbox-label">
            <input type="checkbox" checked={chosen(extra)} onChange={(e) => onToggle(extra, e.target.checked)} />
            {t(`voice.setup.part.${extra.id}`, { defaultValue: PART_LABELS[extra.id] ?? extra.label, size: addonSizeText(extra.id) })}
          </label>
        ))}
        {model && onToggleModel ? (
          <label className="checkbox-label">
            <input type="checkbox" checked={Boolean(modelChosen)} onChange={(e) => onToggleModel(e.target.checked)} />
            {model.gb != null
              ? t("voice.setup.part.model", { defaultValue: "Natural voice model: {name} ({gb} GB, downloads after the install)", name: model.name, gb: gbText(model.gb) })
              : t("voice.setup.part.modelNoSize", { defaultValue: "Natural voice model: {name} (downloads after the install)", name: model.name })}
          </label>
        ) : null}
        <small className="muted-text">
          {installed
            ? t("voice.setup.partsHint", { defaultValue: "A part installs when ticked; untick it to remove it." })
            : t("voice.setup.partsHintInstall", { defaultValue: "Ticked parts install with Voice setup; you can change them later." })}
        </small>
      </fieldset>
    </details>
  );
}
