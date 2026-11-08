import { useMemo, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import type { VoiceEntry, VoiceModelRow } from "../../../types";
import { useCostText } from "./costText";
import { DESIGN_VOICE_ID, chooseModel, costLabel, pickerVoices } from "./speakLogic";
import { canPlaySample } from "./useVoiceSample";

export interface VoicePickerProps {
  voices: VoiceEntry[];
  /** The speech models, for each card's cost chip. */
  models: VoiceModelRow[];
  selectedId: string;
  onSelect: (id: string) => void;
  /** Preset and designed voices run on Apple Silicon Macs for now. */
  modelsAvailable: boolean;
  isPlaying: (id: string) => boolean;
  /** A sample is being made (a preset loads a model first). */
  isLoading?: (id: string) => boolean;
  onSample: (voice: VoiceEntry) => void;
}

function Card({
  voice,
  title,
  detail,
  selected,
  disabled,
  onSelect,
  playing,
  loading,
  tabStop,
  onSample,
  cost,
}: {
  voice: VoiceEntry;
  title: string;
  detail: string;
  /** Cost chip: memory and measured speed, or the download still needed. */
  cost?: { text: string; tone: "ok" | "need" | "plain" } | null;
  selected: boolean;
  disabled?: boolean;
  onSelect: () => void;
  playing: boolean;
  loading: boolean;
  /** The one card in the group that is a Tab stop (arrow keys move between cards). */
  tabStop: boolean;
  onSample: () => void;
}) {
  const { t } = useTranslation("studio");
  return (
    <div className={`voice-card${selected ? " voice-card--selected" : ""}${disabled ? " voice-card--disabled" : ""}`}>
      <button
        type="button"
        role="radio"
        aria-checked={selected}
        tabIndex={tabStop ? 0 : -1}
        data-voice-id={voice.id}
        className="voice-card__main"
        disabled={disabled}
        onClick={onSelect}
      >
        <span className="voice-card__title">{title}</span>
        <span className="voice-card__detail muted-text">{detail}</span>
        {cost ? <span className={`voice-card__cost voice-card__cost--${cost.tone}`}>{cost.text}</span> : null}
      </button>
      {canPlaySample(voice) ? (
        <button
          type="button"
          className="voice-card__play"
          aria-label={
            loading
              ? t("voice.speak.sampleLoading", { defaultValue: "Making a sample of {name}. Click to stop.", name: title })
              : t("voice.speak.sampleOf", { defaultValue: "Play a sample of {name}", name: title })
          }
          aria-busy={loading}
          onClick={onSample}
        >
          {loading ? (
            <svg className="voice-card__spinner" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="8" cy="8" r="5.5" strokeOpacity="0.3" />
              <path d="M8 2.5a5.5 5.5 0 0 1 5.5 5.5" strokeLinecap="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="currentColor">
              {playing ? <rect x="3" y="3" width="10" height="10" rx="1" /> : <path d="M4 2.5v11l9-5.5z" />}
            </svg>
          )}
        </button>
      ) : null}
    </div>
  );
}

function genderLetter(voice: VoiceEntry): string {
  return voice.gender === "male" ? "M" : voice.gender === "female" ? "F" : "";
}

/** Voice cards: your clones, presets, the design card and built-in voices by language. */
export function VoicePicker({ voices, models, selectedId, onSelect, modelsAvailable, isPlaying, isLoading, onSample }: VoicePickerProps) {
  const { t } = useTranslation("studio");
  const costs = useCostText();
  const costOf = (v: VoiceEntry): { text: string; tone: "ok" | "need" | "plain" } | null => {
    if (v.kind === "builtin") return { text: costs.chip(costLabel(null)), tone: "plain" };
    const model = chooseModel(v, models, false);
    if (!model) return null;
    if (!model.installed) return { text: t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb }), tone: "need" };
    return { text: costs.chip(costLabel(model)), tone: "ok" };
  };
  const list = useMemo(() => pickerVoices(voices), [voices]);
  const clones = list.filter((v) => v.kind === "clone");
  const presets = list.filter((v) => v.kind === "preset");
  const design = list.find((v) => v.kind === "design") ?? null;
  const builtin = list.filter((v) => v.kind === "builtin");
  const languages = useMemo(() => [...new Set(builtin.map((v) => v.languageLabel ?? v.language))], [builtin]);
  const selected = list.find((v) => v.id === selectedId);
  const selectedLanguage = selected?.kind === "builtin" ? (selected.languageLabel ?? selected.language) : null;
  const [language, setLanguage] = useState<string | null>(null);
  const shownLanguage = language ?? selectedLanguage ?? languages[0] ?? "";
  const builtinShown = builtin.filter((v) => (v.languageLabel ?? v.language) === shownLanguage);

  // Arrow keys move through the cards that can be picked, in the order shown; only one card is a Tab stop.
  const order = [
    ...clones,
    ...(modelsAvailable ? presets : []),
    ...(design && modelsAvailable ? [design] : []),
    ...builtinShown,
  ].map((v) => v.id);
  const tabStopId = order.includes(selectedId) ? selectedId : (order[0] ?? "");
  const loadingOf = (id: string) => isLoading?.(id) ?? false;

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key) || order.length === 0) return;
    const current = order.indexOf(tabStopId);
    const from = order.indexOf(selectedId) >= 0 ? order.indexOf(selectedId) : current;
    let next = from;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = order.length - 1;
    else if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (from + 1) % order.length;
    else next = (from - 1 + order.length) % order.length;
    event.preventDefault();
    const id = order[next];
    const group = event.currentTarget;
    onSelect(id);
    requestAnimationFrame(() => {
      group.querySelector<HTMLButtonElement>(`[data-voice-id="${CSS.escape(id)}"]`)?.focus();
    });
  };

  const consent = (voice: VoiceEntry): string => {
    if (!voice.consent) return t("voice.speak.cloneKind", { defaultValue: "Your cloned voice" });
    return voice.consent.basis === "own"
      ? t("voice.clone.consentOwn", { defaultValue: "own voice, confirmed {date}", date: voice.consent.consentedAt.slice(0, 10) })
      : t("voice.clone.consentPermission", {
          defaultValue: "{name} gave permission, confirmed {date}",
          name: voice.consent.speakerName,
          date: voice.consent.consentedAt.slice(0, 10),
        });
  };

  return (
    <div className="voice-picker" role="radiogroup" aria-label={t("voice.speak.voice", { defaultValue: "Voice" })} onKeyDown={onKeyDown}>
      {clones.length > 0 ? (
        <section className="voice-picker__group">
          <h4 className="voice-group__title">{t("voice.speak.yourVoices", { defaultValue: "Your voices" })}</h4>
          <div className="voice-cards">
            {clones.map((v) => (
              <Card key={v.id} voice={v} title={v.name} detail={consent(v)} cost={costOf(v)} selected={v.id === selectedId} onSelect={() => onSelect(v.id)} playing={false} loading={false} tabStop={v.id === tabStopId} onSample={() => {}} />
            ))}
          </div>
        </section>
      ) : null}
      {presets.length > 0 ? (
        <section className="voice-picker__group">
          <h4 className="voice-group__title">
            {t("voice.speak.presets", { defaultValue: "Natural voices" })}
            {!modelsAvailable ? (
              <span className="muted-text voice-group__note">{t("voice.speak.appleOnly", { defaultValue: "Apple Silicon Macs only for now." })}</span>
            ) : null}
          </h4>
          <div className="voice-cards">
            {presets.map((v) => (
              <Card
                key={v.id}
                voice={v}
                cost={costOf(v)}
                title={v.name}
                detail={`${v.languageLabel ?? v.language}${genderLetter(v) ? `, ${genderLetter(v)}` : ""}${v.about ? `: ${v.about}` : ""}`}
                selected={v.id === selectedId}
                disabled={!modelsAvailable}
                onSelect={() => onSelect(v.id)}
                playing={isPlaying(v.id)}
                loading={loadingOf(v.id)}
                tabStop={v.id === tabStopId}
                onSample={() => onSample(v)}
              />
            ))}
          </div>
        </section>
      ) : null}
      {design ? (
        <section className="voice-picker__group">
          <div className="voice-cards">
            <Card
              voice={design}
              cost={costOf(design)}
              title={t("voice.speak.designCard", { defaultValue: "Design a voice from a description" })}
              detail={t("voice.speak.designDetail", { defaultValue: "Describe age, tone and pace in the style box." })}
              selected={design.id === selectedId || selectedId === DESIGN_VOICE_ID}
              disabled={!modelsAvailable}
              onSelect={() => onSelect(design.id)}
              playing={false}
              loading={false}
              tabStop={design.id === tabStopId}
              onSample={() => {}}
            />
          </div>
        </section>
      ) : null}
      {builtin.length > 0 ? (
        <section className="voice-picker__group">
          <h4 className="voice-group__title">
            {t("voice.speak.quickVoices", { defaultValue: "Quick voices" })}
            <select
              className="text-input voice-picker__language"
              value={shownLanguage}
              aria-label={t("voice.speak.language", { defaultValue: "Language" })}
              onChange={(e) => setLanguage(e.target.value)}
            >
              {languages.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </h4>
          <div className="voice-cards">
            {builtinShown.map((v) => (
              <Card
                key={v.id}
                voice={v}
                cost={costOf(v)}
                title={v.name}
                detail={genderLetter(v) || (v.languageLabel ?? v.language)}
                selected={v.id === selectedId}
                onSelect={() => onSelect(v.id)}
                playing={isPlaying(v.id)}
                loading={loadingOf(v.id)}
                tabStop={v.id === tabStopId}
                onSample={() => onSample(v)}
              />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
