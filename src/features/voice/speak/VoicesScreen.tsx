import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import type { VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { deleteClonedVoice } from "../clone/cloneApi";
import { CloneStatement } from "../clone/CloneStatement";
import { NewVoiceForm } from "../clone/NewVoiceForm";
import { chineseAddonPending } from "../voicePackState";
import { callVoiceOf } from "../conversation/useVoiceCall";
import { ChatCallsPanel, ChatCost } from "./ChatCallsPanel";
import { MandarinCard } from "./MandarinCard";
import { groupVoices, managerVoices, matchesFilter, usableInChat, type VoiceFilter } from "./speakLogic";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { canPlaySample, useVoiceSample } from "./useVoiceSample";
import { ActionCard, SampleIcon } from "./VoiceCard";

const FILTERS: VoiceFilter[] = ["all", "recommended", "quick", "natural", "cloned", "designed"];
const FILTER_LABELS: Record<VoiceFilter, string> = {
  all: "All",
  recommended: "Recommended",
  quick: "Quick",
  natural: "Natural",
  cloned: "Cloned",
  designed: "Designed",
};

/** The voice manager: every voice with a sample, Use in Speak and Use in Chat, ways to get more, and how chat speaks. */
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
  /** Open Voice Models at the speech models. */
  onGetMore: () => void;
}) {
  const { t, i18n } = useTranslation("studio");
  const { settings, saveSettings, runtime } = useVoice();
  const sample = useVoiceSample(settings.speed);
  const [filter, setFilter] = useState<VoiceFilter>("all");
  const [language, setLanguage] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const listed = useMemo(() => managerVoices(catalog.voices), [catalog.voices]);
  const recommended = useMemo(
    () => groupVoices(catalog.voices, i18n.language, catalog.modelsAvailable).recommended.map((v) => v.id),
    [catalog.voices, i18n.language, catalog.modelsAvailable],
  );
  const anyLanguage = (v: VoiceEntry) => v.kind === "clone" || v.kind === "design";
  const languages = useMemo(() => [...new Set(listed.filter((v) => !anyLanguage(v)).map((v) => v.languageLabel ?? v.language))], [listed]);
  const shown = listed.filter(
    (v) => matchesFilter(v, filter, recommended) && (!language || anyLanguage(v) || (v.languageLabel ?? v.language) === language),
  );
  const addonPending = chineseAddonPending(runtime?.extras ?? []);

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
    const gender = voice.gender === "male" ? "M" : voice.gender === "female" ? "F" : "";
    return [voice.languageLabel ?? voice.language, gender, voice.about].filter(Boolean).join(", ");
  };

  const kindLabel = (voice: VoiceEntry): string =>
    voice.kind === "clone"
      ? t("voice.voices.kindClone", { defaultValue: "Cloned" })
      : voice.kind === "preset"
        ? t("voice.voices.kindPreset", { defaultValue: "Natural" })
        : voice.kind === "design"
          ? t("voice.voices.kindDesign", { defaultValue: "Designed" })
          : t("voice.voices.kindBuiltin", { defaultValue: "Quick" });

  const cloneNote = !catalog.modelsAvailable
    ? t("voice.clone.macOnly", { defaultValue: "Voice cloning runs on Apple Silicon Macs for now." })
    : t("voice.clone.notHere", { defaultValue: "Cloned voices can only be used from the ChaosEngineAI window on this computer." });

  return (
    <Panel
      title={t("voice.voices.listTitle", { defaultValue: "Voices" })}
      subtitle={t("voice.voices.managerSubtitle", { defaultValue: "Every voice in one list: try it, use it in Speak, or make it the chat voice." })}
      className="span-2"
    >
      <div className="voice-studio voice-voices">
        <div className="voice-cards voice-newcards">
          <ActionCard
            title={t("voice.voices.cloneCard", { defaultValue: "Clone a voice" })}
            detail={catalog.cloneAllowed ? t("voice.voices.cloneDetail", { defaultValue: "From a short recording of your own voice, or of someone who gave permission." }) : cloneNote}
          >
            <button type="button" className="secondary-button" disabled={!catalog.cloneAllowed || adding} onClick={() => setAdding(true)}>
              {t("voice.clone.new", { defaultValue: "New cloned voice" })}
            </button>
          </ActionCard>
          <ActionCard
            title={t("voice.voices.designCard", { defaultValue: "Design a voice" })}
            detail={t("voice.voices.designDetail", { defaultValue: "Describe the voice you want in words." })}
          >
            <button type="button" className="secondary-button" disabled={!catalog.modelsAvailable} onClick={onDesign}>
              {t("voice.voices.designAction", { defaultValue: "Describe a voice" })}
            </button>
          </ActionCard>
          {addonPending ? <MandarinCard onInstalled={() => void catalog.refresh()} /> : null}
          <ActionCard
            title={t("voice.voices.moreCard", { defaultValue: "Get more voices" })}
            detail={t("voice.picker.getMoreDetail", { defaultValue: "The cloning models and the rest of the speech models." })}
          >
            <button type="button" className="secondary-button" onClick={onGetMore}>
              {t("voice.picker.openModels", { defaultValue: "Open Models" })}
            </button>
          </ActionCard>
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
        <p className="muted-text voice-voices__chat-note">
          {t("voice.voices.chatAnyVoice", {
            defaultValue: "Chat and calls can use any voice that runs here. Cloned voices read chat replies aloud but are not used in calls. The time is how long the first sentence takes.",
          })}
        </p>

        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}
        {catalog.error ? <p className="voice-error" role="alert">{catalog.error}</p> : null}
        {shown.length === 0 && catalog.loaded ? (
          <p className="muted-text">
            {filter === "cloned"
              ? t("voice.voices.noneCloned", { defaultValue: "No cloned voices yet. Use New cloned voice above." })
              : t("voice.voices.none", { defaultValue: "No voices to show." })}
          </p>
        ) : null}
        <ul className="voice-list">
          {shown.map((voice) => {
            const isChatVoice = settings.ttsVoice === voice.id;
            const isCallVoice = callVoiceOf(settings) === voice.id;
            const chatOk = usableInChat(voice, "read");
            const callOk = usableInChat(voice, "call");
            const notInChat =
              voice.kind === "design"
                ? t("voice.voices.designNotInChat", { defaultValue: "Describe a voice needs a description each time, so it cannot be the chat voice." })
                : (voice.needs ?? undefined);
            return (
              <li key={voice.id} className="voice-list__row">
                <span className="voice-list__name">
                  <strong>{voice.kind === "design" ? t("voice.voices.designCard", { defaultValue: "Design a voice" }) : voice.name}</strong>
                  <span className="badge">{kindLabel(voice)}</span>
                  {isChatVoice ? <span className="badge success">{t("voice.voices.chatVoice", { defaultValue: "Chat voice" })}</span> : null}
                  {isCallVoice ? <span className="badge success">{t("voice.voices.callVoice", { defaultValue: "Call voice" })}</span> : null}
                  <span className="muted-text">{detail(voice)}</span>
                  {chatOk ? <ChatCost cost={voice.chatCost} /> : null}
                </span>
                <span className="voice-list__actions">
                  {canPlaySample(voice) ? (
                    <button
                      type="button"
                      className="secondary-button voice-list__play"
                      aria-label={
                        sample.isPlaying(voice.id)
                          ? t("voice.voices.stopSample", { defaultValue: "Stop the sample of {name}", name: voice.name })
                          : t("voice.speak.sampleOf", { defaultValue: "Play a sample of {name}", name: voice.name })
                      }
                      aria-busy={sample.isLoading(voice.id)}
                      onClick={() => void sample.play(voice)}
                    >
                      <SampleIcon playing={sample.isPlaying(voice.id)} loading={sample.isLoading(voice.id)} />
                    </button>
                  ) : null}
                  <button type="button" className="primary-button" onClick={() => onUse(voice.id)}>
                    {t("voice.voices.useInSpeak", { defaultValue: "Use in Speak" })}
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={!chatOk || isChatVoice}
                    title={chatOk ? undefined : notInChat}
                    onClick={() => void saveSettings({ ttsVoice: voice.id })}
                  >
                    {isChatVoice ? t("voice.voices.inChat", { defaultValue: "In use in Chat" }) : t("voice.voices.useInChat", { defaultValue: "Use in Chat" })}
                  </button>
                  {voice.kind !== "design" ? (
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={!callOk || isCallVoice}
                      title={
                        callOk
                          ? undefined
                          : voice.kind === "clone"
                            ? t("voice.voices.cloneNotInCalls", { defaultValue: "Cloned voices are not used in voice calls." })
                            : notInChat
                      }
                      onClick={() => void saveSettings({ callVoice: voice.id })}
                    >
                      {isCallVoice ? t("voice.voices.inCalls", { defaultValue: "In use for calls" }) : t("voice.voices.useForCalls", { defaultValue: "Use for calls" })}
                    </button>
                  ) : null}
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
          })}
        </ul>

        {catalog.cloneAllowed ? (
          <label className="voice-clone__confirm">
            <input type="checkbox" checked={Boolean(settings.allowAgentCloning)} onChange={(e) => void saveSettings({ allowAgentCloning: e.target.checked })} />{" "}
            {t("voice.clone.allowAgents", { defaultValue: "Allow agents and other computers to use cloned voices (off by default)" })}
          </label>
        ) : null}

        <ChatCallsPanel voices={catalog.voices} />
      </div>
    </Panel>
  );
}
