/**
 * Voice Studio > Voices (D-73 S2): the main voice for Speak, Chat read-aloud and voice calls, and the Fallback voice
 * (D-75), at the top (each with Change, which opens the picker in a drawer for that use; Fallback also offers none), then every voice grouped like the picker (Natural, Quick
 * shortlist with an expander, Cloned, Designed), each with a sample, one "Use for" menu and Delete for cloned voices.
 * The chat behaviour settings live in Chat's Voice popover; agent cloning stays in Settings > Agent access.
 */
import { useMemo, useState, type MouseEvent, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Drawer } from "../../../components/Drawer";
import { Panel } from "../../../components/Panel";
import type { VoiceChatCost, VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { deleteClonedVoice } from "../clone/cloneApi";
import { NewVoiceForm } from "../clone/NewVoiceForm";
import { chineseAddonPending, kokoroInstalled } from "../voicePackState";
import { callVoiceOf } from "../conversation/useVoiceCall";
import { useCostText } from "./costText";
import { FallbackNote } from "./FallbackNote";
import { MandarinCard } from "./MandarinCard";
import {
  VOICE_USE_SETTING,
  chooseModel,
  costLabel,
  effectiveFallback,
  managerVoices,
  matchesFilter,
  quickLanguage,
  quickShortlist,
  topQuickVoice,
  useBlockReason,
  usesOf,
  voicesForUse,
  type VoiceFilter,
  type VoiceUse,
} from "./speakLogic";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { canPlaySample, useVoiceSample } from "./useVoiceSample";
import { SampleIcon } from "./VoiceCard";
import { VoicePicker } from "./VoicePicker";
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
// i18n-extract: voice.voices.use.speak
// i18n-extract: voice.voices.use.read
// i18n-extract: voice.voices.use.call
// i18n-extract: voice.voices.badge.speak
// i18n-extract: voice.voices.badge.read
// i18n-extract: voice.voices.badge.call
// i18n-extract: voice.voices.use.fallback
// i18n-extract: voice.voices.badge.fallback
const USES: VoiceUse[] = ["speak", "read", "call", "fallback"];
const USE_LABELS: Record<VoiceUse, string> = { speak: "Speak", read: "Chat read-aloud", call: "Voice calls", fallback: "Fallback" };
const USE_BADGES: Record<VoiceUse, string> = { speak: "Speak", read: "Chat", call: "Calls", fallback: "Fallback" };

type T = (key: string, options?: Record<string, unknown>) => string;

function useLabel(use: VoiceUse, t: T): string {
  return t(`voice.voices.use.${use}`, { defaultValue: USE_LABELS[use] });
}

function blockText(voice: VoiceEntry, use: VoiceUse, t: T): string | null {
  switch (useBlockReason(voice, use)) {
    case "design":
      return t("voice.voices.designNotInChat", { defaultValue: "Describe a voice needs a description each time, so it cannot be the chat voice." });
    case "clone":
      return t("voice.voices.cloneNotInCalls", { defaultValue: "Cloned voices are not used in voice calls." });
    case "unavailable":
      return voice.needs ?? t("voice.voices.notReady", { defaultValue: "Its model is not downloaded yet." });
    case "notAllowed":
      return t("voice.voices.notForChat", { defaultValue: "This voice cannot speak in Chat here." });
    default:
      return null;
  }
}

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

function closeMenu(event: MouseEvent<HTMLElement>) {
  event.currentTarget.closest("details")?.removeAttribute("open");
}

/** One menu per row: make this the main voice for Speak, Chat read-aloud or voice calls; greyed with W6's reason. */
function UseForMenu({ voice, current, onUse }: { voice: VoiceEntry; current: VoiceUse[]; onUse: (use: VoiceUse) => void }) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as T;
  const name = voice.kind === "design" ? t("voice.voices.designCard", { defaultValue: "Design a voice" }) : voice.name;
  return (
    <details className="model-action-menu voice-use-menu">
      <summary className="secondary-button" aria-label={t("voice.voices.useForLabel", { defaultValue: "Use {name} for", name })}>
        {t("voice.voices.useFor", { defaultValue: "Use for…" })}
      </summary>
      <div className="model-action-menu__list" role="menu">
        {USES.map((use) => {
          const reason = blockText(voice, use, tf);
          const slow = use === "call" && !reason && voice.chatCost?.slowForCalls;
          const on = current.includes(use);
          return (
            <button
              key={use}
              type="button"
              role="menuitemradio"
              aria-checked={on}
              className="model-action-menu__item"
              disabled={Boolean(reason) || on}
              title={reason ?? undefined}
              onClick={(event) => {
                closeMenu(event);
                onUse(use);
              }}
            >
              {useLabel(use, tf)}
              {on ? ` (${t("voice.voices.inUse", { defaultValue: "in use" })})` : ""}
              {slow ? ` (${t("voice.voices.slowFor", { defaultValue: "slow: {s} s", s: voice.chatCost?.firstAudioS })})` : ""}
              {reason ? <small className="voice-use-menu__reason">{reason}</small> : null}
            </button>
          );
        })}
      </div>
    </details>
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
  /** Open Voice Models at the speech models. */
  onGetMore: () => void;
}

export function VoicesScreen({ catalog, speakVoiceId, onUseInSpeak, onDesign, onGetMore }: VoicesScreenProps) {
  const { t, i18n } = useTranslation("studio");
  const tf = t as unknown as T;
  const costs = useCostText();
  const { settings, saveSettings, runtime } = useVoice();
  const sample = useVoiceSample(settings.speed);
  const [filter, setFilter] = useState<VoiceFilter>("all");
  const [language, setLanguage] = useState("");
  const [adding, setAdding] = useState(false);
  const [choosingFor, setChoosingFor] = useState<VoiceUse | null>(null);
  const [showAllQuick, setShowAllQuick] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  const callId = callVoiceOf(settings);
  const listed = useMemo(() => managerVoices(catalog.voices), [catalog.voices]);
  const anyLanguage = (v: VoiceEntry) => v.kind === "clone" || v.kind === "design";
  const languages = useMemo(() => [...new Set(listed.filter((v) => !anyLanguage(v)).map((v) => v.languageLabel ?? v.language))], [listed]);
  const shown = listed.filter((v) => matchesFilter(v, filter) && (!language || anyLanguage(v) || (v.languageLabel ?? v.language) === language));
  // Mandarin voices are Kokoro voices: offered only while Kokoro is installed.
  const addonPending = kokoroInstalled(runtime?.extras ?? [], runtime?.ttsModelInstalled ?? true) && chineseAddonPending(runtime?.extras ?? []);
  const byId = (id: string) => catalog.voices.find((v) => v.id === id) ?? null;

  const fallback = effectiveFallback(catalog.voices, settings);
  const kokoroReady = kokoroInstalled(runtime?.extras ?? [], runtime?.ttsModelInstalled ?? true);

  const assign = (use: VoiceUse, id: string) => {
    setError(null);
    if (use === "speak") onUseInSpeak(id);
    else void saveSettings({ [VOICE_USE_SETTING[use]]: id }).catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  };

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

  const cloneNote = !catalog.modelsAvailable
    ? t("voice.clone.macOnly", { defaultValue: "Voice cloning runs on Apple Silicon Macs for now." })
    : t("voice.clone.notHere", { defaultValue: "Cloned voices can only be used from the ChaosEngineAI window on this computer." });

  // --- the main-voices strip -------------------------------------------------------------------------------------
  const slotNote = (use: VoiceUse, voice: VoiceEntry | null): string => {
    if (use === "speak") return t("voice.voices.slotSpeak", { defaultValue: "Starts selected on the Speak screen." });
    const first = voice?.chatCost?.firstAudioS;
    const timing = first != null ? t("voice.voices.firstWords", { defaultValue: "First words after {s} s.", s: first }) : "";
    if (use === "read") return `${t("voice.voices.slotRead", { defaultValue: "Replies and the speaker button." })} ${timing}`.trim();
    if (use === "fallback") {
      return voice?.kind === "clone"
        ? t("voice.voices.slotFallbackClone", { defaultValue: "Reads when the chat voice cannot. Not used in calls." })
        : t("voice.voices.slotFallback", { defaultValue: "Reads when the chat or call voice cannot." });
    }
    return `${timing} ${t("voice.voices.slotCall", { defaultValue: "Cloned voices are not used in calls." })}`.trim();
  };
  // The fallback slot shows the fallback in effect: a set voice that is not installed (Kokoro removed) counts as none.
  const slotVoice: Record<VoiceUse, string> = { speak: speakVoiceId, read: settings.ttsVoice, call: callId, fallback: fallback?.id ?? "" };

  // --- the grouped list -----------------------------------------------------------------------------------------
  const natural = shown.filter((v) => v.kind === "preset");
  const quick = shown.filter((v) => v.kind === "builtin");
  const clones = shown.filter((v) => v.kind === "clone");
  const designed = shown.filter((v) => v.kind === "design");
  const quickLanguages = [...new Set(quick.map(quickLanguage))];
  const uiQuick = topQuickVoice(quick, i18n.language);
  const orderedQuickLanguages = uiQuick ? [quickLanguage(uiQuick), ...quickLanguages.filter((l) => l !== quickLanguage(uiQuick))] : quickLanguages;
  const assigned = new Set(Object.values(slotVoice));

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
              {t(`voice.voices.badge.${use}`, { defaultValue: USE_BADGES[use] })}
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
          {needsDownload && model ? (
            model.download.state === "downloading" ? (
              <span className="muted-text">{t("voice.clone.downloading", { defaultValue: "Downloading…" })}</span>
            ) : (
              <button type="button" className="secondary-button" onClick={() => void catalog.download(model.key)}>
                {t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb })}
              </button>
            )
          ) : null}
          {voice.kind === "design" ? (
            <button type="button" className="secondary-button" disabled={!catalog.modelsAvailable} onClick={onDesign}>
              {t("voice.voices.useInSpeak", { defaultValue: "Use in Speak" })}
            </button>
          ) : (
            <UseForMenu voice={voice} current={uses} onUse={(use) => assign(use, voice.id)} />
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
              const visible = open ? inLanguage : inLanguage.filter((v) => short.includes(v) || assigned.has(v.id));
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

  const drawerUse = choosingFor;
  const drawerVoices = drawerUse ? voicesForUse(catalog.voices, drawerUse) : [];

  return (
    <Panel
      title={t("voice.voices.listTitle", { defaultValue: "Voices" })}
      subtitle={t("voice.voices.subtitleUses", { defaultValue: "Every voice on this computer, and which one each part of the app uses." })}
      className="span-2"
      actions={
        <div className="button-row voice-voices__actions">
          <button
            type="button"
            className="secondary-button"
            disabled={!catalog.cloneAllowed || adding}
            title={catalog.cloneAllowed ? undefined : cloneNote}
            onClick={() => setAdding(true)}
          >
            {t("voice.clone.new", { defaultValue: "New cloned voice" })}
          </button>
          <button type="button" className="secondary-button" disabled={!catalog.modelsAvailable} onClick={onDesign}>
            {t("voice.voices.designCard", { defaultValue: "Design a voice" })}
          </button>
          <button type="button" className="secondary-button" onClick={onGetMore}>
            {t("voice.voices.moreCard", { defaultValue: "Get more voices" })}
          </button>
        </div>
      }
    >
      <div className="voice-studio voice-voices">
        {adding ? (
          <NewVoiceForm
            onCancel={() => setAdding(false)}
            onCreated={() => {
              setAdding(false);
              void catalog.refresh();
            }}
          />
        ) : null}

        <section className="voice-main" aria-label={t("voice.voices.mainTitle", { defaultValue: "Main voices" })}>
          <div className="voice-main__slots">
            {USES.map((use) => {
              const voice = byId(slotVoice[use]);
              return (
                <div key={use} className="voice-main__slot" data-use={use}>
                  <span className="voice-main__label">{useLabel(use, tf)}</span>
                  <span className="voice-main__voice">
                    <strong>
                      {voice
                        ? voice.kind === "design"
                          ? t("voice.voices.designCard", { defaultValue: "Design a voice" })
                          : voice.name
                        : use === "fallback"
                          ? t("voice.voices.fallbackNone", { defaultValue: "None set" })
                          : slotVoice[use] || "—"}
                    </strong>
                    {voice ? <span className="badge">{voiceKindLabel(voice, tf)}</span> : null}
                    {use === "call" && !settings.callVoice ? (
                      <span className="muted-text">{t("voice.voices.sameAsChat", { defaultValue: "(same as Chat)" })}</span>
                    ) : null}
                  </span>
                  <span className="muted-text voice-main__note">{slotNote(use, voice)}</span>
                  <button
                    type="button"
                    className="secondary-button voice-main__change"
                    aria-label={t("voice.voices.changeFor", { defaultValue: "Change the voice for {use}", use: useLabel(use, tf) })}
                    onClick={() => setChoosingFor(use)}
                  >
                    {t("voice.voices.change", { defaultValue: "Change" })}
                  </button>
                </div>
              );
            })}
          </div>
          <p className="muted-text voice-main__fallback">
            <FallbackNote voice={fallback} />
          </p>
        </section>

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

        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}
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
      </div>

      {drawerUse ? (
        <Drawer
          title={t("voice.voices.chooseFor", { defaultValue: "Choose the voice for {use}", use: useLabel(drawerUse, tf) })}
          subtitle={
            drawerUse === "call"
              ? t("voice.voices.chooseCallHint", { defaultValue: "Voices that can answer calls. Cloned voices are not used in calls." })
              : drawerUse === "read"
                ? t("voice.voices.chooseReadHint", { defaultValue: "Voices that can read chat replies aloud." })
                : drawerUse === "fallback"
                  ? t("voice.voices.chooseFallbackHint", {
                      defaultValue: "Installed voices that read when the chosen chat or call voice cannot. A cloned voice is used for read-aloud only.",
                    })
                  : t("voice.picker.drawerHint", { defaultValue: "Click a voice to use it in Speak. The sample button plays a short line." })
          }
          onClose={() => setChoosingFor(null)}
          className="voice-drawer"
        >
          {drawerUse === "fallback" ? (
            <div className="button-row voice-drawer__none">
              <button
                type="button"
                className="secondary-button"
                aria-pressed={!fallback}
                onClick={() => {
                  assign("fallback", "");
                  setChoosingFor(null);
                }}
              >
                {t("voice.voices.noFallback", { defaultValue: "No fallback voice" })}
              </button>
            </div>
          ) : null}
          <VoicePicker
            voices={drawerVoices}
            models={catalog.models}
            selectedId={slotVoice[drawerUse]}
            onSelect={(id) => assign(drawerUse, id)}
            onChoose={() => setChoosingFor(null)}
            modelsAvailable={catalog.modelsAvailable}
            isPlaying={sample.isPlaying}
            isLoading={sample.isLoading}
            onSample={(entry) => void sample.play(entry)}
            onGetMore={onGetMore}
            onAddonInstalled={() => void catalog.refresh()}
          />
        </Drawer>
      ) : null}
    </Panel>
  );
}
