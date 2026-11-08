import { useMemo, useState, type KeyboardEvent, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { VoiceEntry, VoiceModelRow, VoiceRuntime } from "../../../types";
import { useVoice } from "../VoiceContext";
import { InstallProgress, useVoicePackJob } from "../VoicePackPanel";
import { useCostText } from "./costText";
import { DESIGN_VOICE_ID, chooseModel, costLabel, groupVoices, quickLanguage, quickShortlist } from "./speakLogic";
import { ActionCard, GroupHead, VoiceCard, type CardCost } from "./VoiceCard";

export interface VoicePickerProps {
  voices: VoiceEntry[];
  /** The speech models, for each card's cost chip. */
  models: VoiceModelRow[];
  /** Shown first in Recommended (the "Install a better voice" card). */
  lead?: ReactNode;
  selectedId: string;
  onSelect: (id: string) => void;
  /** Preset and designed voices run on Apple Silicon Macs for now. */
  modelsAvailable: boolean;
  isPlaying: (id: string) => boolean;
  /** A sample is being made (a preset loads a model first). */
  isLoading?: (id: string) => boolean;
  onSample: (voice: VoiceEntry) => void;
  /** Open the new cloned voice form (Voices tab). */
  onNewClone?: () => void;
  /** Open Models at the speech models. */
  onGetMore?: () => void;
  /** The Mandarin add-on finished installing: list the voices again. */
  onAddonInstalled?: () => void;
}

function genderLetter(voice: VoiceEntry): string {
  return voice.gender === "male" ? "M" : voice.gender === "female" ? "F" : "";
}

/** The Mandarin add-on, installed from the Quick group when Chinese is chosen and the add-on is missing. */
function MandarinCard({ onInstalled }: { onInstalled?: () => void }) {
  const { runtime, refresh } = useVoice();
  if (!runtime) return null;
  return (
    <MandarinCardBody
      runtime={runtime}
      onChanged={() => {
        refresh();
        onInstalled?.();
      }}
    />
  );
}

function MandarinCardBody({ runtime, onChanged }: { runtime: VoiceRuntime; onChanged: () => void }) {
  const { t } = useTranslation("studio");
  const pack = useVoicePackJob(runtime, onChanged);
  return (
    <ActionCard
      title={t("voice.picker.addMandarin", { defaultValue: "Add Mandarin voices" })}
      detail={t("voice.picker.addMandarinDetail", { defaultValue: "Eight quick voices, a small add-on to the voice pack." })}
    >
      {pack.job.running ? (
        <InstallProgress pack={pack} />
      ) : (
        <button type="button" className="secondary-button" onClick={() => void pack.start(["chinese"])}>
          {t("voice.voices.addChinese", { defaultValue: "Add Mandarin Chinese voices" })}
        </button>
      )}
      {pack.error ? (
        <p className="voice-error" role="alert">
          {pack.error}
        </p>
      ) : null}
    </ActionCard>
  );
}

/** Voice cards in groups: Recommended, Quick (graded shortlist and an expander), Cloned, Designed, More. */
export function VoicePicker({
  voices,
  models,
  lead,
  selectedId,
  onSelect,
  modelsAvailable,
  isPlaying,
  isLoading,
  onSample,
  onNewClone,
  onGetMore,
  onAddonInstalled,
}: VoicePickerProps) {
  const { t, i18n } = useTranslation("studio");
  const costs = useCostText();
  const groups = useMemo(() => groupVoices(voices, i18n.language, modelsAvailable), [voices, i18n.language, modelsAvailable]);
  const languages = useMemo(() => [...new Set(groups.quick.map(quickLanguage))], [groups.quick]);
  const selected = voices.find((v) => v.id === selectedId);
  const selectedLanguage = selected?.kind === "builtin" ? quickLanguage(selected) : null;
  const uiQuick = groups.recommended.find((v) => v.kind === "builtin");
  const [language, setLanguage] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const shownLanguage = language ?? selectedLanguage ?? (uiQuick ? quickLanguage(uiQuick) : languages[0]) ?? "";
  const inLanguage = groups.quick.filter((v) => quickLanguage(v) === shownLanguage);
  const needsAddon = inLanguage.length > 0 && inLanguage.every((v) => !v.available && v.addon);
  const shortlist = quickShortlist(inLanguage);
  // The selected quick voice always shows, even when it is not on the shortlist.
  const quickShown = needsAddon
    ? []
    : showAll
      ? inLanguage
      : [...shortlist, ...(selected && inLanguage.includes(selected) && !shortlist.includes(selected) ? [selected] : [])];
  const hidden = inLanguage.length - quickShown.length;
  const quickCount = groups.quick.length;

  const costOf = (v: VoiceEntry): CardCost | null => {
    if (v.kind === "builtin") return { text: costs.chip(costLabel(null)), tone: "plain" };
    const model = chooseModel(v, models);
    if (!model) return null;
    if (!model.installed) return { text: t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb }), tone: "need" };
    return { text: costs.chip(costLabel(model)), tone: "ok" };
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

  const naturalDetail = (v: VoiceEntry) =>
    t("voice.picker.naturalDetail", {
      defaultValue: "Natural, {language}{gender}. {about}",
      language: v.languageLabel ?? v.language,
      gender: genderLetter(v) ? `, ${genderLetter(v)}` : "",
      about: v.about ?? "",
    }).trim();
  const designTitle = t("voice.speak.designCard", { defaultValue: "Design a voice from a description" });
  const designDetail = t("voice.picker.designDetail", {
    defaultValue: "Describe age, gender, tone, accent and pace. A new voice each run unless you set a seed.",
  });

  const titleOf = (v: VoiceEntry) => (v.kind === "design" ? designTitle : v.name);
  const detailOf = (v: VoiceEntry, where: "recommended" | "group"): string => {
    if (v.kind === "clone") return consent(v);
    if (v.kind === "design") return designDetail;
    if (v.kind === "preset") return naturalDetail(v);
    if (where === "recommended") {
      return t("voice.picker.quickDetail", {
        defaultValue: "Quick, {language}{gender}. Best-graded quick voice.",
        language: quickLanguage(v),
        gender: genderLetter(v) ? `, ${genderLetter(v)}` : "",
      });
    }
    return genderLetter(v) || quickLanguage(v);
  };
  const disabledOf = (v: VoiceEntry) => (v.kind === "preset" || v.kind === "design") && !modelsAvailable;

  // Arrow keys move through the cards that can be picked, in the order shown (a voice in two groups counts once);
  // only one card is a Tab stop.
  const sections: Array<[string, VoiceEntry[]]> = [
    ["recommended", groups.recommended],
    ["quick", quickShown],
    ["clones", groups.clones],
    ["design", groups.design && modelsAvailable ? [groups.design] : []],
    ["more", modelsAvailable ? groups.more : []],
  ];
  const order = [...new Set(sections.flatMap(([, list]) => list.map((v) => v.id)))];
  const tabStopId = order.includes(selectedId) ? selectedId : (order[0] ?? "");
  const tabStopSection = sections.find(([, list]) => list.some((v) => v.id === tabStopId))?.[0];
  const loadingOf = (id: string) => isLoading?.(id) ?? false;

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key) || order.length === 0) return;
    const from = order.indexOf(selectedId) >= 0 ? order.indexOf(selectedId) : order.indexOf(tabStopId);
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

  const card = (v: VoiceEntry, section: string, where: "recommended" | "group" = "group") => (
    <VoiceCard
      key={`${section}-${v.id}`}
      voice={v}
      cost={costOf(v)}
      title={titleOf(v)}
      detail={detailOf(v, where)}
      selected={v.id === selectedId || (v.kind === "design" && selectedId === DESIGN_VOICE_ID)}
      disabled={disabledOf(v)}
      onSelect={() => onSelect(v.id)}
      playing={isPlaying(v.id)}
      loading={loadingOf(v.id)}
      tabStop={v.id === tabStopId && section === tabStopSection}
      onSample={() => onSample(v)}
    />
  );

  const appleOnly = !modelsAvailable ? t("voice.speak.appleOnly", { defaultValue: "Apple Silicon Macs only for now." }) : undefined;
  const uiLanguageName = (() => {
    try {
      return new Intl.DisplayNames([i18n.language], { type: "language" }).of(i18n.language.split("-")[0]) ?? i18n.language;
    } catch {
      return i18n.language;
    }
  })();

  return (
    <div className="voice-picker" role="radiogroup" aria-label={t("voice.speak.voice", { defaultValue: "Voice" })} onKeyDown={onKeyDown}>
      <section className="voice-picker__group voice-picker__group--recommended">
        <GroupHead
          title={t("voice.picker.recommended", { defaultValue: "Recommended" })}
          note={t("voice.picker.recommendedNote", { defaultValue: "For this computer and {language}", language: uiLanguageName })}
        />
        {lead}
        <div className="voice-cards">{groups.recommended.map((v) => card(v, "recommended", "recommended"))}</div>
      </section>

      {quickCount > 0 ? (
        <section className="voice-picker__group voice-picker__group--quick">
          <GroupHead
            title={t("voice.picker.quick", { defaultValue: "Quick" })}
            note={t("voice.picker.quickNote", {
              defaultValue: "Kokoro, part of the voice pack. Fast, under 1 GB, speed only. {count} voices in {languages} languages.",
              count: quickCount,
              languages: languages.length,
            })}
          >
            <select
              className="text-input voice-picker__language"
              value={shownLanguage}
              aria-label={t("voice.speak.language", { defaultValue: "Language" })}
              onChange={(e) => {
                setLanguage(e.target.value);
                setShowAll(false);
              }}
            >
              {languages.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </GroupHead>
          <div className="voice-cards">
            {quickShown.map((v) => card(v, "quick"))}
            {needsAddon ? <MandarinCard onInstalled={onAddonInstalled} /> : null}
          </div>
          {hidden > 0 || (showAll && inLanguage.length > shortlist.length) ? (
            <button type="button" className="link-button voice-picker__expander" aria-expanded={showAll} onClick={() => setShowAll((on) => !on)}>
              {showAll
                ? t("voice.picker.fewer", { defaultValue: "Show the shortlist" })
                : t("voice.picker.allVoices", { defaultValue: "All {count} {language} voices", count: inLanguage.length, language: shownLanguage })}
            </button>
          ) : null}
        </section>
      ) : null}

      {groups.clones.length > 0 || (modelsAvailable && onNewClone) ? (
        <section className="voice-picker__group voice-picker__group--clones">
          <GroupHead
            title={t("voice.picker.cloned", { defaultValue: "Cloned" })}
            note={t("voice.picker.clonedNote", {
              defaultValue: "Only your own voice or someone who gave permission. Every clip is labelled as AI-made and watermarked.",
            })}
          />
          <div className="voice-cards">
            {groups.clones.map((v) => card(v, "clones"))}
            {modelsAvailable && onNewClone ? (
              <ActionCard
                title={t("voice.picker.newClone", { defaultValue: "New cloned voice" })}
                detail={t("voice.picker.newCloneDetail", { defaultValue: "From a 10 to 20 s recording. Opens the consent form." })}
              >
                <button type="button" className="secondary-button" onClick={onNewClone}>
                  {t("voice.picker.newCloneOpen", { defaultValue: "Open the form" })}
                </button>
              </ActionCard>
            ) : null}
          </div>
        </section>
      ) : null}

      {groups.design ? (
        <section className="voice-picker__group voice-picker__group--designed">
          <GroupHead title={t("voice.picker.designed", { defaultValue: "Designed" })} note={appleOnly ?? t("voice.picker.designedNote", { defaultValue: "A voice made from a description." })} />
          <div className="voice-cards">{card(groups.design, "design")}</div>
        </section>
      ) : null}

      {groups.more.length > 0 || onGetMore ? (
        <section className="voice-picker__group voice-picker__group--more">
          <GroupHead
            title={t("voice.picker.more", { defaultValue: "More" })}
            note={appleOnly ?? t("voice.picker.moreNote", { defaultValue: "Natural voices in other languages, and more to install." })}
          />
          <div className="voice-cards">
            {groups.more.map((v) => card(v, "more"))}
            {onGetMore ? (
              <ActionCard
                title={t("voice.picker.getMore", { defaultValue: "Get more voices" })}
                detail={t("voice.picker.getMoreDetail", { defaultValue: "The cloning models and the rest of the speech models." })}
              >
                <button type="button" className="secondary-button" onClick={onGetMore}>
                  {t("voice.picker.openModels", { defaultValue: "Open Models" })}
                </button>
              </ActionCard>
            ) : null}
          </div>
        </section>
      ) : null}
    </div>
  );
}
