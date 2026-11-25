/**
 * The voice in use on Speak (D-73 S1): its name, kind, one line about it, the cost chip, a sample button and Change
 * voice, which opens the picker in a drawer. Enter or Space on Change voice opens it; focus comes back here on close.
 */
import { useTranslation } from "react-i18next";
import type { VoiceEntry, VoiceModelRow } from "../../../types";
import { useCostText } from "./costText";
import { languageName } from "./SpeakControls";
import { costLabel, type VoiceAbilities } from "./speakLogic";
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

/**
 * One line under the voice row: what this voice can do (speed, style, sound tags) and its languages. Controls it cannot
 * use are not drawn; Speak says why in the place they would be.
 */
export function VoiceCanLine({ abilities }: { abilities: VoiceAbilities | null }) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;
  if (!abilities) return null;
  const can = [
    abilities.speed ? t("voice.speak.can.speed", { defaultValue: "speed" }) : null,
    abilities.style ? t("voice.speak.can.style", { defaultValue: "style" }) : null,
    abilities.tags ? t("voice.speak.can.tags", { defaultValue: "sound tags" }) : null,
  ].filter((item): item is string => Boolean(item));
  const languages = abilities.languages;
  const languageText = !languages
    ? ""
    : languages.kind === "any"
      ? t("voice.speak.can.anyLanguage", { defaultValue: "any language" })
      : languages.kind === "own"
        ? languages.label
        : languages.codes.length > 3
          ? t("voice.speak.can.nLanguages", { defaultValue: "{n} languages", n: languages.codes.length })
          : languages.codes.map((code) => languageName(tf, code)).join(", ");
  return (
    <p className="muted-text voice-speak__hint voice-speak__can">
      {can.length > 0
        ? t("voice.speak.can.line", { defaultValue: "This voice can: {list}.", list: can.join(", ") })
        : t("voice.speak.can.none", { defaultValue: "This voice reads the text as it is." })}
      {languageText ? ` ${t("voice.speak.can.languages", { defaultValue: "Languages: {list}.", list: languageText })}` : ""}
    </p>
  );
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
