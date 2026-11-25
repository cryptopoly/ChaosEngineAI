import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { StudioShell } from "../../components/StudioShell";
import type { TabId, VoiceRuntime } from "../../types";
import { SpeakScreen } from "./speak/SpeakScreen";
import { VoicesScreen } from "./speak/VoicesScreen";
import { DESIGN_VOICE_ID, defaultVoice } from "./speak/speakLogic";
import { useVoiceCatalog, type VoiceCatalog } from "./speak/useVoiceCatalog";
import { useVoice } from "./VoiceContext";
import { VoiceSetupCard, laneDescription, setupHasMissing } from "./setup/VoiceSetupCard";
import { useVoicePackJob } from "./setup/voicePackJob";
import { TranscribePanel, type OpenRequest as TranscriptRequest } from "./studio/TranscribePanel";
import { STUDIO_MODES, readStudioMode, requestedMode, writeStudioMode, type StudioTask, type VoiceStudioRequest } from "./voiceStudioMode";

export { pickDefaultSttModel } from "./studio/TranscribePanel";
/** What another tab asks the Studio to open: a transcript, a mode, a voice on Speak. */
export type OpenRequest = VoiceStudioRequest;

export interface VoiceStudioTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  /** A Library transcript to open, or a mode and voice to show (Models > Use in Speak). */
  openRequest: OpenRequest | null;
  onSendToChat: (text: string) => void;
  onTabChange: (tab: TabId) => void;
}

// A request is acted on once, not again each time the Studio is opened while it is still the latest one.
let handledNonce = 0;

/**
 * Voice setup in place on the Studio (voice studio UX audit, changes 1 and 10): the whole page before the first
 * install; above the tasks while something is missing; folded to one line once everything this computer can do is in.
 */
function StudioSetup({
  runtime,
  catalog,
  onOpenChat,
  onOpenClone,
}: {
  runtime: VoiceRuntime;
  catalog: VoiceCatalog;
  onOpenChat: () => void;
  onOpenClone: () => void;
}) {
  const { t } = useTranslation("studio");
  const { refresh } = useVoice();
  const refreshModels = catalog.refresh;
  const onInstallDone = useCallback(() => {
    refresh();
    void refreshModels();
  }, [refresh, refreshModels]);
  const pack = useVoicePackJob(runtime, onInstallDone);
  const missing = setupHasMissing(runtime, catalog);
  const title = t("voice.setup.title", { defaultValue: "Voice setup" });
  const card = (
    <VoiceSetupCard
      runtime={runtime}
      pack={pack}
      catalog={catalog}
      onRuntimeChanged={refresh}
      onOpenChat={onOpenChat}
      onOpenClone={runtime.installed ? onOpenClone : undefined}
    />
  );
  if (!runtime.installed || missing || pack.job.running) {
    return (
      <Panel
        title={title}
        subtitle={t("voice.setup.studioSubtitle", {
          defaultValue: "{lane}. Everything runs on this computer; nothing is sent online.",
          lane: laneDescription(runtime),
        })}
        className="span-2 voice-setup-panel"
      >
        {card}
      </Panel>
    );
  }
  return (
    <details className="span-2 voice-setup-panel voice-setup-panel--ready">
      <summary>
        {t("voice.setup.allReady", { defaultValue: "Voice setup: everything this computer can do is ready" })}
      </summary>
      {card}
    </details>
  );
}

export function VoiceStudioTab({ voiceRuntime, backendOnline, openRequest, onSendToChat, onTabChange }: VoiceStudioTabProps) {
  const { t, i18n } = useTranslation("studio");
  const [mode, setModeState] = useState<StudioTask>(readStudioMode);
  const { settings, saveSettings } = useVoice();
  const catalog = useVoiceCatalog(backendOnline, voiceRuntime?.lane);
  const [voiceId, setVoiceId] = useState("");
  const [speakText, setSpeakText] = useState<{ text: string; nonce: number } | null>(null);
  // Until the person picks a voice, Speak follows the default rule, so a natural voice that finishes downloading
  // becomes the starting voice; a chosen voice that disappears (a deleted clone) falls back to the rule too.
  const picked = useRef(false);
  const speakVoice = settings.speakVoice ?? "";

  const setMode = (next: StudioTask) => {
    setModeState(next);
    writeStudioMode(next);
  };

  useEffect(() => {
    if (!catalog.loaded) return;
    if (picked.current && catalog.voices.some((v) => v.id === voiceId)) return;
    const next = defaultVoice(catalog.voices, speakVoice, i18n.language);
    if (next && next !== voiceId) setVoiceId(next);
  }, [catalog.loaded, catalog.voices, voiceId, speakVoice, i18n.language]);

  /** The person chose a voice: it becomes the Speak voice next time too (the chat voice is not touched). */
  const pickVoice = (id: string) => {
    picked.current = true;
    setVoiceId(id);
    if (id !== speakVoice) void saveSettings({ speakVoice: id });
  };

  const useVoiceInSpeak = (id: string) => {
    pickVoice(id);
    setMode("speak");
  };

  // The transcript part of a request goes to the Transcribe editor; the mode and voice are handled here.
  const transcriptRequest = useMemo<TranscriptRequest | null>(
    () => (openRequest?.id ? { id: openRequest.id, nonce: openRequest.nonce } : null),
    [openRequest],
  );

  useEffect(() => {
    if (!openRequest || openRequest.nonce === handledNonce) return;
    handledNonce = openRequest.nonce;
    const wanted = requestedMode(openRequest);
    if (wanted) setMode(wanted);
    if (openRequest.voice) pickVoice(openRequest.voice);
    if (openRequest.text != null) setSpeakText({ text: openRequest.text, nonce: openRequest.nonce });
    // pickVoice and setMode only set state; the request is what matters here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openRequest]);

  const title = t("voice.studio.title", { defaultValue: "Voice Studio" });

  if (!backendOnline || !voiceRuntime) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={title} className="span-2">
          <div className="empty-state">
            <p className="muted-text">{t("voice.offline", { defaultValue: "Waiting for the backend…" })}</p>
          </div>
        </Panel>
      </div>
    );
  }

  const setup = (
    <StudioSetup runtime={voiceRuntime} catalog={catalog} onOpenChat={() => onTabChange("chat")} onOpenClone={() => setMode("clone")} />
  );

  // Voice always lands here (change 10): before the first install the setup card is the page. The same root and
  // position keep the card (and a model download queued behind the install) mounted when the install finishes.
  if (!voiceRuntime.installed) {
    return <div className="content-grid discover-page voice-page voice-studio-page">{setup}</div>;
  }

  const header = (
    <div className="voice-studio__title">
        <h2>{title}</h2>
        <div className="segmented voice-studio__modes" role="tablist" aria-label={title}>
          {STUDIO_MODES.map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={mode === value}
              className={mode === value ? "segment active" : "segment"}
              onClick={() => setMode(value)}
            >
              {value === "transcribe"
                ? t("voice.transcribe.title", { defaultValue: "Transcribe" })
                : value === "speak"
                  ? t("voice.speak.title", { defaultValue: "Speak" })
                  : t("voice.studio.cloneTask", { defaultValue: "Clone a voice" })}
            </button>
          ))}
        </div>
    </div>
  );

  // All three stay mounted so a running queue or generation survives a switch. Clone a voice shows the Voices
  // screen until its own screen replaces it.
  const panes = (
    <>
      <div className={`voice-studio__pane${mode === "transcribe" ? "" : " voice-studio__pane--hidden"}`}>
        <TranscribePanel
          runtime={voiceRuntime}
          openRequest={transcriptRequest}
          onSendToChat={onSendToChat}
          onOpenModels={() => onTabChange("voice-models")}
        />
      </div>
      <div className={`voice-studio__pane${mode === "speak" ? "" : " voice-studio__pane--hidden"}`}>
        <SpeakScreen
          catalog={catalog}
          voiceId={voiceId}
          onVoiceChange={pickVoice}
          onTabChange={onTabChange}
          onNewClone={() => setMode("clone")}
          textRequest={speakText}
        />
      </div>
      <div className={`voice-studio__pane${mode === "clone" ? "" : " voice-studio__pane--hidden"}`}>
        <VoicesScreen
          catalog={catalog}
          speakVoiceId={voiceId}
          onUseInSpeak={pickVoice}
          onDesign={() => useVoiceInSpeak(DESIGN_VOICE_ID)}
          onGetMore={() => onTabChange("voice-models")}
        />
      </div>
    </>
  );

  return (
    <div className="content-grid discover-page voice-page voice-studio-page">
      {setup}
      <div className="span-2 voice-studio__shell">
        <StudioShell studioId="voice-studio" header={header} inputs={panes} />
      </div>
    </div>
  );
}
