import { useTranslation } from "react-i18next";
import type { VoiceSliderRange } from "../../../types";
import type { VoiceControls } from "./speakLogic";

const TAG_LABELS: Record<string, string> = {
  laugh: "Laugh",
  chuckle: "Chuckle",
  sigh: "Sigh",
  gasp: "Gasp",
  cough: "Cough",
  "clear throat": "Clear throat",
  sniff: "Sniff",
  sniffle: "Sniffle",
  groan: "Groan",
  shush: "Shush",
  yawn: "Yawn",
  angry: "Angry",
  fear: "Fear",
  surprised: "Surprised",
  whispering: "Whispering",
  advertisement: "Advertisement",
  dramatic: "Dramatic",
  narration: "Narration",
  crying: "Crying",
  happy: "Happy",
  sarcastic: "Sarcastic",
};

const tagKey = (tag: string) => tag.replace(/\s+/g, "_");

/** Sound tag chips under the text box: each inserts [tag] at the cursor. Turbo's style tokens come with a warning. */
export function TagChips({ controls, onInsert }: { controls: VoiceControls; onInsert: (tag: string) => void }) {
  const { t } = useTranslation("studio");
  if (controls.tags.length === 0 && controls.styleTags.length === 0) return null;
  const chip = (tag: string) => (
    <button key={tag} type="button" className="voice-speak__chip" onClick={() => onInsert(tag)}>
      [{t(`voice.speak.tags.${tagKey(tag)}`, { defaultValue: TAG_LABELS[tag] ?? tag })}]
    </button>
  );
  return (
    <div className="voice-speak__tags">
      {controls.tags.length > 0 ? (
        <div className="voice-speak__chips" role="group" aria-label={t("voice.speak.tagChips", { defaultValue: "Sound tags" })}>
          {controls.tags.map(chip)}
        </div>
      ) : null}
      {controls.styleTags.length > 0 ? (
        <>
          <div className="voice-speak__chips" role="group" aria-label={t("voice.speak.styleTagChips", { defaultValue: "Style tags (undocumented)" })}>
            {controls.styleTags.map(chip)}
          </div>
          <p className="muted-text voice-speak__hint">
            {t("voice.speak.styleTagsNote", {
              defaultValue:
                "Style tags are in the model but its authors do not document them, so results vary. Put one at the start of a sentence.",
            })}
          </p>
        </>
      ) : null}
      <p className="muted-text voice-speak__hint">
        {t("voice.speak.tagsHint", { defaultValue: "Tags such as [laugh] are performed, not read aloud. Other voices skip them." })}
      </p>
    </div>
  );
}

function Slider({ label, range, value, onChange }: { label: string; range: VoiceSliderRange; value: number; onChange: (v: number) => void }) {
  return (
    <label className="voice-settings__field">
      <span>
        {label} ({value.toFixed(2)})
      </span>
      <input
        type="range"
        min={range.min}
        max={range.max}
        step={range.step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

const LANGUAGE_NAMES: Record<string, string> = {
  ar: "Arabic", da: "Danish", de: "German", el: "Greek", en: "English", es: "Spanish", fi: "Finnish", fr: "French",
  he: "Hebrew", hi: "Hindi", it: "Italian", ja: "Japanese", ko: "Korean", ms: "Malay", nl: "Dutch", no: "Norwegian",
  pl: "Polish", pt: "Portuguese", ru: "Russian", sv: "Swedish", sw: "Swahili", tr: "Turkish", zh: "Chinese",
};

/** Chatterbox V3: the language it speaks and the Expression slider (calm to dramatic). */
export function LanguageAndExpression({
  controls,
  language,
  onLanguage,
  expression,
  onExpression,
}: {
  controls: VoiceControls;
  language: string;
  onLanguage: (code: string) => void;
  expression: number;
  onExpression: (value: number) => void;
}) {
  const { t } = useTranslation("studio");
  if (controls.languages.length === 0 && !controls.expression) return null;
  return (
    <div className="voice-studio__row">
      {controls.languages.length > 0 ? (
        <label className="voice-settings__field">
          <span>{t("voice.speak.speechLanguage", { defaultValue: "Speech language" })}</span>
          <select className="text-input" value={language} onChange={(e) => onLanguage(e.target.value)}>
            {controls.languages.map((code) => (
              <option key={code} value={code}>
                {t(`voice.speak.languages.${code}`, { defaultValue: LANGUAGE_NAMES[code] ?? code })}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {controls.expression ? (
        <Slider
          label={t("voice.speak.expression", { defaultValue: "Expression (calm to dramatic)" })}
          range={controls.expression}
          value={expression}
          onChange={onExpression}
        />
      ) : null}
    </div>
  );
}

/** Chatterbox V3, under Advanced: how closely a voice sample is followed, with the model card's tips. */
export function GuidanceSlider({ range, value, onChange }: { range: VoiceSliderRange | null; value: number; onChange: (v: number) => void }) {
  const { t } = useTranslation("studio");
  if (!range) return null;
  return (
    <>
      <Slider label={t("voice.speak.guidance", { defaultValue: "Follow the voice sample closely" })} range={range} value={value} onChange={onChange} />
      <p className="muted-text voice-speak__hint">
        {t("voice.speak.guidanceHint", {
          defaultValue: "Try 0.3 for a fast speaker, and 0 when the sample is in another language (drops its accent).",
        })}
      </p>
    </>
  );
}
