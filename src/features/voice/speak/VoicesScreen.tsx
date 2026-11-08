import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import type { VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { deleteClonedVoice } from "../clone/cloneApi";
import { CloneStatement } from "../clone/CloneStatement";
import { NewVoiceForm } from "../clone/NewVoiceForm";
import { matchesFilter, type VoiceFilter } from "./speakLogic";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { canPlaySample, useVoiceSample } from "./useVoiceSample";

const FILTERS: VoiceFilter[] = ["all", "mine", "builtin"];
const FILTER_LABELS: Record<VoiceFilter, string> = { all: "All", mine: "Mine", builtin: "Built-in" };

/** Every voice in one list (built-in and your clones), with ways to make or find more. */
export function VoicesScreen({
  catalog,
  onUse,
  onDesign,
  onGetMore,
}: {
  catalog: VoiceCatalog;
  /** Jump to Speak with this voice selected. */
  onUse: (id: string) => void;
  /** Jump to Speak with the design voice. */
  onDesign: () => void;
  /** Open Voice Models, where the voice pack and its add-ons live. */
  onGetMore: () => void;
}) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const sample = useVoiceSample(settings.speed);
  const [filter, setFilter] = useState<VoiceFilter>("all");
  const [language, setLanguage] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const listed = useMemo(() => catalog.voices.filter((v) => v.kind !== "design" && (v.kind !== "builtin" || v.available)), [catalog.voices]);
  const languages = useMemo(() => [...new Set(listed.filter((v) => v.kind !== "clone").map((v) => v.languageLabel ?? v.language))], [listed]);
  const shown = listed.filter((v) => matchesFilter(v, filter) && (!language || v.kind === "clone" || (v.languageLabel ?? v.language) === language));

  const remove = async (voice: VoiceEntry) => {
    const ask = t("voice.clone.deleteConfirm", {
      defaultValue: "Delete the voice \"{name}\"? Its sample and permission record are deleted too. This cannot be undone.",
      name: voice.name,
    });
    if (!window.confirm(ask)) return;
    try {
      await deleteClonedVoice(voice.id.replace(/^clone:/, ""));
      await catalog.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const detail = (voice: VoiceEntry): string => {
    if (voice.consent) {
      return voice.consent.basis === "own"
        ? t("voice.clone.consentOwn", { defaultValue: "own voice, confirmed {date}", date: voice.consent.consentedAt.slice(0, 10) })
        : t("voice.clone.consentPermission", {
            defaultValue: "{name} gave permission, confirmed {date}",
            name: voice.consent.speakerName,
            date: voice.consent.consentedAt.slice(0, 10),
          });
    }
    const gender = voice.gender === "male" ? "M" : voice.gender === "female" ? "F" : "";
    return [voice.languageLabel ?? voice.language, gender, voice.about].filter(Boolean).join(", ");
  };

  const cloneNote = !catalog.modelsAvailable
    ? t("voice.clone.macOnly", { defaultValue: "Voice cloning runs on Apple Silicon Macs for now." })
    : t("voice.clone.notHere", { defaultValue: "Cloned voices can only be used from the ChaosEngineAI window on this computer." });

  return (
    <Panel
      title={t("voice.voices.listTitle", { defaultValue: "Voices" })}
      subtitle={t("voice.voices.listSubtitle", { defaultValue: "Built-in voices and the voices you cloned, in one list." })}
      className="span-2"
    >
      <div className="voice-studio voice-voices">
        <div className="voice-cards voice-newcards">
          <div className="voice-card voice-card--action">
            <span className="voice-card__title">{t("voice.voices.cloneCard", { defaultValue: "Clone a voice" })}</span>
            <span className="voice-card__detail muted-text">
              {catalog.cloneAllowed ? t("voice.voices.cloneDetail", { defaultValue: "From a short recording of your own voice, or of someone who gave permission." }) : cloneNote}
            </span>
            <button type="button" className="secondary-button" disabled={!catalog.cloneAllowed || adding} onClick={() => setAdding(true)}>
              {t("voice.clone.new", { defaultValue: "New cloned voice" })}
            </button>
          </div>
          <div className="voice-card voice-card--action">
            <span className="voice-card__title">{t("voice.voices.designCard", { defaultValue: "Design a voice" })}</span>
            <span className="voice-card__detail muted-text">{t("voice.voices.designDetail", { defaultValue: "Describe the voice you want in words." })}</span>
            <button type="button" className="secondary-button" disabled={!catalog.modelsAvailable} onClick={onDesign}>
              {t("voice.voices.designAction", { defaultValue: "Describe a voice" })}
            </button>
          </div>
          <div className="voice-card voice-card--action">
            <span className="voice-card__title">{t("voice.voices.moreCard", { defaultValue: "Get more voices" })}</span>
            <span className="voice-card__detail muted-text">{t("voice.voices.moreDetail", { defaultValue: "Add Mandarin Chinese voices and more from the voice pack." })}</span>
            <button type="button" className="secondary-button" onClick={onGetMore}>
              {t("voice.voices.moreAction", { defaultValue: "Open Voice Models" })}
            </button>
          </div>
        </div>

        {catalog.cloneAllowed ? <CloneStatement /> : null}
        {adding ? (
          <NewVoiceForm
            onCancel={() => setAdding(false)}
            onCreated={() => {
              setAdding(false);
              void catalog.refresh();
            }}
          />
        ) : null}

        <div className="voice-studio__row voice-voices__filters">
          <div className="segmented" role="tablist" aria-label={t("voice.voices.filter", { defaultValue: "Show" })}>
            {FILTERS.map((value) => (
              <button key={value} type="button" role="tab" aria-selected={filter === value} className={filter === value ? "segment active" : "segment"} onClick={() => setFilter(value)}>
                {t(`voice.voices.filters.${value}`, { defaultValue: FILTER_LABELS[value] })}
              </button>
            ))}
          </div>
          <select
            className="text-input"
            value={language}
            aria-label={t("voice.speak.language", { defaultValue: "Language" })}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="">{t("voice.voices.allLanguages", { defaultValue: "All languages" })}</option>
            {languages.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>

        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}
        {catalog.error ? <p className="voice-error" role="alert">{catalog.error}</p> : null}
        {shown.length === 0 && catalog.loaded ? (
          <p className="muted-text">
            {filter === "mine"
              ? t("voice.voices.noneMine", { defaultValue: "No cloned voices yet. Use Clone a voice above." })
              : t("voice.voices.none", { defaultValue: "No voices to show." })}
          </p>
        ) : null}
        <ul className="voice-list">
          {shown.map((voice) => (
            <li key={voice.id} className="voice-list__row">
              <span className="voice-list__name">
                <strong>{voice.name}</strong>
                <span className="badge">
                  {voice.kind === "clone"
                    ? t("voice.voices.kindClone", { defaultValue: "Cloned" })
                    : voice.kind === "preset"
                      ? t("voice.voices.kindPreset", { defaultValue: "Natural" })
                      : t("voice.voices.kindBuiltin", { defaultValue: "Quick" })}
                </span>
                <span className="muted-text">{detail(voice)}</span>
              </span>
              <span className="voice-list__actions">
                {canPlaySample(voice) ? (
                  <button type="button" className="secondary-button" onClick={() => void sample.play(voice)}>
                    {sample.isPlaying(voice.id) ? t("voice.stop", { defaultValue: "Stop" }) : t("voice.voices.playSample", { defaultValue: "Play sample" })}
                  </button>
                ) : null}
                <button type="button" className="primary-button" onClick={() => onUse(voice.id)}>
                  {t("voice.voices.use", { defaultValue: "Use" })}
                </button>
                {voice.kind === "clone" ? (
                  <button
                    type="button"
                    className="secondary-button"
                    aria-label={t("voice.clone.deleteVoice", { defaultValue: "Delete the voice {name}", name: voice.name })}
                    onClick={() => void remove(voice)}
                  >
                    {t("voice.clone.delete", { defaultValue: "Delete" })}
                  </button>
                ) : null}
              </span>
            </li>
          ))}
        </ul>

        {catalog.cloneAllowed ? (
          <label className="voice-clone__confirm">
            <input type="checkbox" checked={Boolean(settings.allowAgentCloning)} onChange={(e) => void saveSettings({ allowAgentCloning: e.target.checked })} />{" "}
            {t("voice.clone.allowAgents", { defaultValue: "Allow agents and other computers to use cloned voices (off by default)" })}
          </label>
        ) : null}
      </div>
    </Panel>
  );
}
