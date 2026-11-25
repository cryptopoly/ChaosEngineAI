/**
 * The voice catalogue (voice studio UX audit, change 3; D0142): every voice on this computer grouped like the picker
 * (Natural, Quick shortlist with an expander, Cloned, Designed), each with a sample, "Use in Speak" and Delete for
 * cloned voices. Chat's voice choices (read-aloud, calls, backup) are no longer here: they live in Chat's Voice
 * popover. "New cloned voice" and "Design a voice" say in words why they are greyed, with the button that fixes it.
 * Agent cloning stays in Settings > Agent access.
 */
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import type { VoiceChatCost, VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { unsupportedModelText } from "../models/voiceModelsLogic";
import { deleteClonedVoice } from "../clone/cloneApi";
import { CloneVoiceScreen } from "../clone/CloneVoiceScreen";
import { cloneActionBlock, type ActionBlock } from "../clone/cloneLogic";
import { VoiceErrorNotice } from "../clone/VoiceErrorNotice";
import { chineseAddonPending, kokoroInstalled } from "../voicePackState";
import { callVoiceOf } from "../conversation/useVoiceCall";
import { describeVoiceError, type VoiceErrorView } from "../voiceErrorText";
import { useCostText } from "./costText";
import { MandarinCard } from "./MandarinCard";
import { chooseModel, costLabel, fallbackSetting, managerVoices, matchesFilter, quickLanguage, quickShortlist, topQuickVoice, usesOf, type VoiceFilter, type VoiceUse } from "./speakLogic";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { canPlaySample, useVoiceSample } from "./useVoiceSample";
import { SampleIcon } from "./VoiceCard";
import { voiceAbout, voiceKindLabel } from "./VoiceRow";

const FILTERS: VoiceFilter[] = ["all", "natural", "quick", "cloned", "designed"];
const FILTER_LABELS: Record<VoiceFilter, string> = {
  all: "All",
  recommended: "Recommended",
  quick: "Quick",
  natural: "Natural",
  cloned: "Cloned",
  designed: "Designed",
};
// i18n-extract: voice.voices.badge.speak
// i18n-extract: voice.voices.badge.read
// i18n-extract: voice.voices.badge.call
// i18n-extract: voice.voices.badge.backup
const USE_BADGES: Record<VoiceUse, string> = { speak: "Speak", read: "Chat", call: "Calls", fallback: "Backup" };
const USE_BADGE_KEYS: Record<VoiceUse, string> = { speak: "speak", read: "read", call: "call", fallback: "backup" };

type T = (key: string, options?: Record<string, unknown>) => string;

/** Measured first-audio time of a voice in Chat, and a label when it is slow for calls. */
export function ChatCost({ cost }: { cost?: VoiceChatCost | null }) {
  const { t } = useTranslation("studio");
  if (!cost) return null;
  return (
    <span className="muted-text voice-chat-cost" title={cost.measuredOn ?? undefined}>
      {t("voice.chatCalls.cost", {
        defaultValue: "first audio {warm} s ({cold} s cold), {memory} GB",
        warm: cost.firstAudioS,
        cold: cost.firstAudioColdS ?? "?",
        memory: cost.peakGb ?? "?",
      })}
      {cost.slowForCalls ? <span className="badge warning">{t("voice.chatCalls.slow", { defaultValue: "Slow for calls" })}</span> : null}
    </span>
  );
}

export interface VoicesScreenProps {
  catalog: VoiceCatalog;
  /** The voice Speak starts on (the Studio's current Speak voice). */
  speakVoiceId: string;
  /** Make this the Speak voice (stays on Voices). */
  onUseInSpeak: (id: string) => void;
  /** Jump to Speak with the design voice. */
  onDesign: () => void;
  /** Open Voice Models at the speech models (the fix for a missing speech part). */
  onGetMore: () => void;
  /** After a cloned voice is saved, "Speak with it"; ``onUseInSpeak`` when not given. */
  onSpeakWith?: (voiceId: string) => void;
}

/** One greyed action's reason as text, with the button that fixes it. */
function BlockedNote({ label, block, onFix }: { label: string; block: ActionBlock; onFix: () => void }) {
  const { t } = useTranslation("studio");
  return (
    <p className="muted-text voice-voices__reason">
      <strong>{label}</strong> {block.reason}
      {block.fix === "setup" ? (
        <>
          {" "}
          <button type="button" className="secondary-button" onClick={onFix}>
            {t("voice.clone.openSetup", { defaultValue: "Open Voice setup" })}
          </button>
        </>
      ) : null}
    </p>
  );
}

export function VoicesScreen({ catalog, speakVoiceId, onUseInSpeak, onDesign, onGetMore, onSpeakWith }: VoicesScreenProps) {
  const { t, i18n } = useTranslation("studio");
  const tf = t as unknown as T;
  const costs = useCostText();
  const { settings, runtime } = useVoice();
  const sample = useVoiceSample(settings.speed);
  const [filter, setFilter] = useState<VoiceFilter>("all");
  const [language, setLanguage] = useState("");
  const [adding, setAdding] = useState(false);
  const [showAllQuick, setShowAllQuick] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<VoiceErrorView | null>(null);

  const callId = callVoiceOf(settings);
  const listed = useMemo(() => managerVoices(catalog.voices), [catalog.voices]);
  const anyLanguage = (v: VoiceEntry) => v.kind === "clone" || v.kind === "design";
  const languages = useMemo(() => [...new Set(listed.filter((v) => !anyLanguage(v)).map((v) => v.languageLabel ?? v.language))], [listed]);
  const shown = listed.filter((v) => matchesFilter(v, filter) && (!language || anyLanguage(v) || (v.languageLabel ?? v.language) === language));
  // Mandarin voices are Kokoro voices: offered only while Kokoro is installed.
  const addonPending = kokoroInstalled(runtime?.extras ?? [], runtime?.ttsModelInstalled ?? true) && chineseAddonPending(runtime?.extras ?? []);
  const kokoroReady = kokoroInstalled(runtime?.extras ?? [], runtime?.ttsModelInstalled ?? true);

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
      setError(describeVoiceError(err));
    }
  };

  const state = { modelsAvailable: catalog.modelsAvailable, cloneAllowed: catalog.cloneAllowed, runtime };
  // Until the catalogue has loaded nothing is known: the buttons wait, no reason is shown yet.
  const cloneBlock = catalog.loaded ? cloneActionBlock("clone", state, tf) : null;
  const designBlock = catalog.loaded ? cloneActionBlock("design", state, tf) : null;
  const waiting = !catalog.loaded;

  // --- the grouped list -----------------------------------------------------------------------------------------
  const natural = shown.filter((v) => v.kind === "preset");
  const quick = shown.filter((v) => v.kind === "builtin");
  const clones = shown.filter((v) => v.kind === "clone");
  const designed = shown.filter((v) => v.kind === "design");
  const quickLanguages = [...new Set(quick.map(quickLanguage))];
  const uiQuick = topQuickVoice(quick, i18n.language);
  const orderedQuickLanguages = uiQuick ? [quickLanguage(uiQuick), ...quickLanguages.filter((l) => l !== quickLanguage(uiQuick))] : quickLanguages;
  // A voice in use anywhere stays on the quick shortlist, even when it is not one of the best graded.
  const inUse = new Set<string>([speakVoiceId, settings.ttsVoice, callId, fallbackSetting(settings)]);

  const row = (voice: VoiceEntry) => {
    const uses = usesOf(voice.id, settings, speakVoiceId, callId);
    const model = chooseModel(voice, catalog.models);
    const needsDownload = Boolean(model && !model.installed && voice.kind !== "builtin");
    const name = voice.kind === "design" ? t("voice.voices.designCard", { defaultValue: "Design a voice" }) : voice.name;
    return (
      <li key={voice.id} className="voice-list__row" data-voice-row={voice.id}>
        {canPlaySample(voice) ? (
          <button
            type="button"
            className="secondary-button voice-list__play"
            aria-label={
              sample.isPlaying(voice.id)
                ? t("voice.voices.stopSample", { defaultValue: "Stop the sample of {name}", name })
                : t("voice.speak.sampleOf", { defaultValue: "Play a sample of {name}", name })
            }
            aria-busy={sample.isLoading(voice.id)}
            onClick={() => void sample.play(voice)}
          >
            <SampleIcon playing={sample.isPlaying(voice.id)} loading={sample.isLoading(voice.id)} />
          </button>
        ) : null}
        <span className="voice-list__name">
          <strong>{name}</strong>
          <span className="badge">{voiceKindLabel(voice, tf)}</span>
          {uses.map((use) => (
            <span key={use} className="badge success">
              {t(`voice.voices.badge.${USE_BADGE_KEYS[use]}`, { defaultValue: USE_BADGES[use] })}
            </span>
          ))}
          <span className="muted-text voice-list__about">
            {voiceAbout(voice, tf)}
            {model ? ` · ${model.name}` : ""}
          </span>
          {model && model.installed ? <span className="voice-card__cost voice-card__cost--ok">{costs.chip(costLabel(model))}</span> : null}
          {voice.kind !== "design" && voice.chatUse?.read ? <ChatCost cost={voice.chatCost} /> : null}
        </span>
        <span className="voice-list__actions">
          {needsDownload && model && !model.supported ? (
            <span className="muted-text voice-list__unsupported">{unsupportedModelText(model, runtime, tf)}</span>
          ) : needsDownload && model ? (
            model.download.state === "downloading" ? (
              <span className="muted-text">{t("voice.clone.downloading", { defaultValue: "Downloading…" })}</span>
            ) : (
              <button type="button" className="secondary-button" onClick={() => void catalog.download(model.key)}>
                {t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb })}
              </button>
            )
          ) : null}
          {voice.kind === "design" ? (
            <button type="button" className="secondary-button" disabled={waiting || designBlock !== null} onClick={onDesign}>
              {t("voice.voices.useInSpeak", { defaultValue: "Use in Speak" })}
            </button>
          ) : (
            <button
              type="button"
              className="secondary-button"
              disabled={voice.id === speakVoiceId}
              aria-label={t("voice.voices.useNameInSpeak", { defaultValue: "Use {name} in Speak", name })}
              onClick={() => onUseInSpeak(voice.id)}
            >
              {voice.id === speakVoiceId
                ? t("voice.voices.inSpeak", { defaultValue: "In use in Speak" })
                : t("voice.voices.useInSpeak", { defaultValue: "Use in Speak" })}
            </button>
          )}
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
    );
  };

  const group = (id: string, title: string, note: string, children: ReactNode) => (
    <section key={id} className="voice-voices__group" aria-labelledby={`voice-group-${id}`}>
      <div className="voice-group__head">
        <h4 id={`voice-group-${id}`} className="voice-group__title">
          {title}
        </h4>
        <span className="muted-text voice-group__note">{note}</span>
      </div>
      {children}
    </section>
  );

  const kokoroMissingGroup =
    !kokoroReady && (filter === "all" || filter === "quick")
      ? group(
          "quick",
          t("voice.picker.quick", { defaultValue: "Quick" }),
          t("voice.voices.kokoroMissing", {
            defaultValue: "The Kokoro quick voices are not installed. Tick them under Voice > Models > Voice runtime > Add-ons (about 330 MB).",
          }),
          <div className="button-row">
            <button type="button" className="secondary-button" onClick={onGetMore}>
              {t("voice.voices.openModels", { defaultValue: "Open Voice Models" })}
            </button>
          </div>,
        )
      : null;

  const quickGroup =
    quick.length > 0 || (addonPending && (filter === "all" || filter === "quick"))
      ? group(
          "quick",
          t("voice.picker.quick", { defaultValue: "Quick" }),
          t("voice.voices.quickNoteAddon", { defaultValue: "Kokoro, an add-on of the voice runtime. Fast, under 1 GB, speed only. The best-graded shown first." }),
          <>
            {orderedQuickLanguages.map((lang) => {
              const inLanguage = quick.filter((v) => quickLanguage(v) === lang);
              const short = quickShortlist(inLanguage);
              const open = showAllQuick.has(lang);
              const visible = open ? inLanguage : inLanguage.filter((v) => short.includes(v) || inUse.has(v.id));
              return (
                <div key={lang} className="voice-voices__quick-language">
                  {orderedQuickLanguages.length > 1 ? <h5 className="voice-voices__language">{lang}</h5> : null}
                  <ul className="voice-list">{visible.map(row)}</ul>
                  {inLanguage.length > visible.length || open ? (
                    <button
                      type="button"
                      className="link-button voice-picker__expander"
                      aria-expanded={open}
                      onClick={() =>
                        setShowAllQuick((previous) => {
                          const next = new Set(previous);
                          if (next.has(lang)) next.delete(lang);
                          else next.add(lang);
                          return next;
                        })
                      }
                    >
                      {open
                        ? t("voice.picker.fewer", { defaultValue: "Show the shortlist" })
                        : t("voice.picker.allVoices", { defaultValue: "All {count} {language} voices", count: inLanguage.length, language: lang })}
                    </button>
                  ) : null}
                </div>
              );
            })}
            {addonPending ? (
              <div className="voice-cards">
                <MandarinCard onInstalled={() => void catalog.refresh()} />
              </div>
            ) : null}
          </>,
        )
      : null;

  return (
    <Panel
      title={t("voice.voices.listTitle", { defaultValue: "Voices" })}
      subtitle={t("voice.voices.subtitleCatalogue", { defaultValue: "Every voice on this computer. Chat's voices are chosen in Chat, under Voice." })}
      className="span-2"
      actions={
        <div className="button-row voice-voices__actions">
          <button type="button" className="secondary-button" disabled={waiting || cloneBlock !== null || adding} onClick={() => setAdding(true)}>
            {t("voice.clone.new", { defaultValue: "New cloned voice" })}
          </button>
          <button type="button" className="secondary-button" disabled={waiting || designBlock !== null} onClick={onDesign}>
            {t("voice.voices.designCard", { defaultValue: "Design a voice" })}
          </button>
          <button type="button" className="secondary-button" onClick={onGetMore}>
            {t("voice.voices.moreCard", { defaultValue: "Get more voices" })}
          </button>
        </div>
      }
    >
      <div className="voice-studio voice-voices">
        {cloneBlock ? <BlockedNote label={t("voice.clone.new", { defaultValue: "New cloned voice" })} block={cloneBlock} onFix={onGetMore} /> : null}
        {designBlock ? <BlockedNote label={t("voice.voices.designCard", { defaultValue: "Design a voice" })} block={designBlock} onFix={onGetMore} /> : null}
        {adding ? (
          <>
            <CloneVoiceScreen catalog={catalog} onSpeakWith={(id) => (onSpeakWith ?? onUseInSpeak)(id)} onOpenSetup={onGetMore} />
            <div className="button-row">
              <button type="button" className="secondary-button" onClick={() => setAdding(false)}>
                {t("voice.voices.closeClone", { defaultValue: "Close" })}
              </button>
            </div>
          </>
        ) : null}

        <div className="voice-studio__row voice-voices__filters">
          <div className="segmented" role="tablist" aria-label={t("voice.voices.filter", { defaultValue: "Show" })}>
            {FILTERS.map((value) => (
              <button key={value} type="button" role="tab" aria-selected={filter === value} className={filter === value ? "segment active" : "segment"} onClick={() => setFilter(value)}>
                {t(`voice.voices.filters.${value}`, { defaultValue: FILTER_LABELS[value] })}
              </button>
            ))}
          </div>
          <select className="text-input" value={language} aria-label={t("voice.speak.language", { defaultValue: "Language" })} onChange={(e) => setLanguage(e.target.value)}>
            <option value="">{t("voice.voices.allLanguages", { defaultValue: "All languages" })}</option>
            {languages.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>

        {error ? <VoiceErrorNotice view={error} /> : null}
        {catalog.error ? <p className="voice-error" role="alert">{catalog.error}</p> : null}
        {shown.length === 0 && catalog.loaded && !((addonPending || !kokoroReady) && filter === "quick") ? (
          <p className="muted-text">
            {filter === "cloned"
              ? t("voice.voices.noneCloned", { defaultValue: "No cloned voices yet. Use New cloned voice above." })
              : t("voice.voices.none", { defaultValue: "No voices to show." })}
          </p>
        ) : null}

        {natural.length > 0
          ? group(
              "natural",
              t("voice.voices.groupNatural", { defaultValue: "Natural" }),
              t("voice.voices.naturalNote", { defaultValue: "From the speech models you downloaded. Style, tags or expression depend on the model." }),
              <ul className="voice-list">{natural.map(row)}</ul>,
            )
          : null}
        {quickGroup ?? kokoroMissingGroup}
        {clones.length > 0
          ? group(
              "cloned",
              t("voice.picker.cloned", { defaultValue: "Cloned" }),
              t("voice.voices.clonedNote", { defaultValue: "Your voice, or someone who gave permission. Labelled AI-made and watermarked." }),
              <ul className="voice-list">{clones.map(row)}</ul>,
            )
          : null}
        {designed.length > 0
          ? group(
              "designed",
              t("voice.picker.designed", { defaultValue: "Designed" }),
              t("voice.picker.designedNote", { defaultValue: "A voice made from a description." }),
              <ul className="voice-list">{designed.map(row)}</ul>,
            )
          : null}
        {catalog.hiddenVoices ? (
          <p className="muted-text voice-voices__mac-only">
            {t("voice.voices.macOnlyHidden", {
              defaultValue: "{count} voices of the speech models for Apple Silicon Macs are not listed: they do not run on this PC.",
              count: catalog.hiddenVoices,
            })}
          </p>
        ) : null}
      </div>
    </Panel>
  );
}
