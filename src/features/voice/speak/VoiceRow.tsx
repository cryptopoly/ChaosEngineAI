/**
 * The voice in use on Speak (D-73 S1): its name, kind, one line about it, the cost chip, a sample button and Change
 * voice, which opens the picker in a drawer. Enter or Space on Change voice opens it; focus comes back here on close.
 */
import { useTranslation } from "react-i18next";
import type { VoiceEntry, VoiceModelRow } from "../../../types";
import { useCostText } from "./costText";
import { costLabel } from "./speakLogic";
import { canPlaySample } from "./useVoiceSample";
import { SampleIcon } from "./VoiceCard";

export function voiceKindLabel(voice: Pick<VoiceEntry, "kind">, t: (key: string, options?: Record<string, unknown>) => string): string {
  return voice.kind === "clone"
    ? t("voice.voices.kindClone", { defaultValue: "Cloned" })
    : voice.kind === "preset"
      ? t("voice.voices.kindPreset", { defaultValue: "Natural" })
      : voice.kind === "design"
        ? t("voice.voices.kindDesign", { defaultValue: "Designed" })
        : t("voice.voices.kindBuiltin", { defaultValue: "Quick" });
}

/** "English, male. Dynamic, strong rhythm." for a voice; the consent record for a cloned one. */
export function voiceAbout(voice: VoiceEntry, t: (key: string, options?: Record<string, unknown>) => string): string {
  if (voice.kind === "design") {
    return t("voice.voices.designRowDetail", { defaultValue: "Any language. A new voice from your description each run." });
  }
  if (voice.consent) {
    return voice.consent.basis === "own"
      ? t("voice.clone.consentOwn", { defaultValue: "own voice, confirmed {date}", date: voice.consent.consentedAt.slice(0, 10) })
      : t("voice.clone.consentPermission", {
          defaultValue: "{name} gave permission, confirmed {date}",
          name: voice.consent.speakerName,
          date: voice.consent.consentedAt.slice(0, 10),
        });
  }
  const gender =
    voice.gender === "male"
      ? t("voice.voices.male", { defaultValue: "male" })
      : voice.gender === "female"
        ? t("voice.voices.female", { defaultValue: "female" })
        : "";
  return [voice.languageLabel ?? voice.language, gender, voice.about].filter(Boolean).join(", ");
}

export interface VoiceRowProps {
  voice: VoiceEntry | null;
  /** The model that will run (null for a quick voice). */
  model: VoiceModelRow | null;
  playing: boolean;
  loading: boolean;
  onSample: () => void;
  onChange: () => void;
}

export function VoiceRow({ voice, model, playing, loading, onSample, onChange }: VoiceRowProps) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;
  const costs = useCostText();
  const name = voice ? (voice.kind === "design" ? t("voice.speak.designCard", { defaultValue: "Design a voice from a description" }) : voice.name) : "";
  const cost = voice ? (voice.kind === "builtin" ? costLabel(null) : model ? costLabel(model) : null) : null;
  return (
    <div className="voice-row" aria-label={t("voice.speak.voice", { defaultValue: "Voice" })} role="group">
      {voice && canPlaySample(voice) ? (
        <button
          type="button"
          className="voice-row__play"
          aria-label={t("voice.speak.sampleOf", { defaultValue: "Play a sample of {name}", name })}
          aria-busy={loading}
          onClick={onSample}
        >
          <SampleIcon playing={playing} loading={loading} />
        </button>
      ) : null}
      <div className="voice-row__who">
        <span className="voice-row__name">
          <strong>{voice ? name : t("voice.speak.noVoice", { defaultValue: "No voice chosen" })}</strong>
          {voice ? <span className="badge">{voiceKindLabel(voice, tf)}</span> : null}
        </span>
        {voice ? <span className="muted-text voice-row__about">{voiceAbout(voice, tf)}</span> : null}
      </div>
      {cost ? (
        <span className={`voice-card__cost voice-card__cost--${cost.kind === "kokoro" ? "plain" : model?.installed ? "ok" : "need"}`}>
          {model && !model.installed ? t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb }) : costs.chip(cost)}
        </span>
      ) : null}
      <button type="button" className="secondary-button voice-row__change" onClick={onChange}>
        {t("voice.speak.changeVoice", { defaultValue: "Change voice" })}
      </button>
    </div>
  );
}
