import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SendToStoryboardMenu } from "../../../components/SendToStoryboardMenu";
import { getLibraryAudio, getLibraryItem } from "../../../api";
import type { TabId, VoiceLibraryItem } from "../../../types";
import { shotTitleFrom, type SendItem } from "../../storyboard/sendToStoryboardLogic";
import { useSendToStoryboard } from "../../storyboard/useSendToStoryboard";
import { downloadBlob, formatClock } from "../audio";
import { useVoice } from "../VoiceContext";
import {
  deleteClonedVoice,
  downloadCloneModel,
  getCloneModels,
  listClonedVoices,
  speakCloned,
  type ClonedVoice,
  type CloneModels,
} from "./cloneApi";
import { pickCloneModel } from "./cloneLogic";
import { NewVoiceForm } from "./NewVoiceForm";

const MAX_CHARS = 5000;

/** The plain statement shown at the top of the panel, always. */
export function CloneStatement() {
  const { t } = useTranslation("studio");
  return (
    <p className="voice-clone__statement">
      {t("voice.clone.statement", {
        defaultValue:
          "Only copy your own voice, or the voice of someone who has given you permission and reads the consent sentence themselves. Not public figures, children or people who have died. Every clip made with a cloned voice is labelled as AI-made inside the file, and that label cannot be turned off.",
      })}
    </p>
  );
}

/** Clone a voice: collapsed by default inside Speak, so the Studio stays simple for everyone else. */
export function ClonePanel({ onTabChange, defaultOpen = false }: { onTabChange?: (tab: TabId) => void; defaultOpen?: boolean }) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details className="voice-clone" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>{t("voice.clone.title", { defaultValue: "Clone a voice" })}</summary>
      <CloneStatement />
      {open ? <CloneBody onTabChange={onTabChange} /> : null}
    </details>
  );
}

function CloneBody({ onTabChange }: { onTabChange?: (tab: TabId) => void }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();
  const storyboard = useSendToStoryboard((tab) => onTabChange?.(tab));
  const [models, setModels] = useState<CloneModels | null>(null);
  const [voices, setVoices] = useState<ClonedVoice[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [voiceId, setVoiceId] = useState("");
  const [adding, setAdding] = useState(false);
  const [text, setText] = useState("");
  const [disclaimer, setDisclaimer] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ item: VoiceLibraryItem; url: string; blob: Blob } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await getCloneModels();
      setModels(next);
      setModelKey((current) => current || pickCloneModel(next.models)?.key || "");
      if (next.allowedHere) {
        const list = await listClonedVoices();
        setVoices(list);
        setVoiceId((current) => (list.some((v) => v.id === current) ? current : list[0]?.id ?? ""));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll while a model downloads.
  const downloading = models?.models.some((m) => m.download.state === "downloading") ?? false;
  useEffect(() => {
    if (!downloading) return;
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [downloading, refresh]);

  useEffect(
    () => () => {
      if (result) URL.revokeObjectURL(result.url);
    },
    [result],
  );

  if (!models) return <p className="muted-text">{error ?? t("voice.clone.loading", { defaultValue: "Loading…" })}</p>;
  if (!models.available) {
    return <p className="muted-text">{t("voice.clone.macOnly", { defaultValue: "Voice cloning runs on Apple Silicon Macs for now." })}</p>;
  }
  const model = models.models.find((m) => m.key === modelKey) ?? null;

  const speak = async () => {
    setBusy(true);
    setError(null);
    try {
      const item = await speakCloned(voiceId, text, modelKey, disclaimer);
      const blob = await getLibraryAudio(item.id);
      setResult({ item, blob, url: URL.createObjectURL(blob) });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (voice: ClonedVoice) => {
    const ask = t("voice.clone.deleteConfirm", {
      defaultValue: "Delete the voice \"{name}\"? Its sample and consent record are deleted too. This cannot be undone.",
      name: voice.name,
    });
    if (!window.confirm(ask)) return;
    try {
      await deleteClonedVoice(voice.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const sendItem: SendItem | null = result
    ? {
        kind: "voice",
        label: shotTitleFrom(result.item.text),
        clonedVoice: true,
        resolvePath: async () => {
          const path = (await getLibraryItem(result.item.id)).audioPath;
          if (!path) throw new Error(t("sendToStoryboard.noFile", { defaultValue: "This item has no file to send." }));
          return path;
        },
      }
    : null;

  return (
    <div className="voice-studio voice-clone__body">
      <div className="voice-studio__row">
        <label className="voice-settings__field">
          <span>{t("voice.clone.model", { defaultValue: "Cloning model" })}</span>
          <select className="text-input" value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
            {models.models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.name} ({m.sizeGb} GB)
              </option>
            ))}
          </select>
        </label>
        {model ? (
          <span className="voice-clone__model-state">
            {model.installed ? (
              <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span>
            ) : model.download.state === "downloading" ? (
              <span className="muted-text">{t("voice.clone.downloading", { defaultValue: "Downloading…" })}</span>
            ) : (
              <button type="button" className="secondary-button" onClick={() => void downloadCloneModel(model.key).then(refresh)}>
                {t("voice.models.download", { defaultValue: "Download" })}
              </button>
            )}
            <span className="muted-text"> {model.summary} {model.licence.name}.</span>
          </span>
        ) : null}
      </div>
      {model?.download.error ? <p className="voice-error">{model.download.error}</p> : null}

      {!models.allowedHere ? (
        <p className="muted-text">{t("voice.clone.notHere", { defaultValue: "Cloned voices can only be used from the ChaosEngineAI window on this computer." })}</p>
      ) : adding ? (
        <NewVoiceForm
          onCancel={() => setAdding(false)}
          onCreated={(voice) => {
            setAdding(false);
            setVoiceId(voice.id);
            void refresh();
          }}
        />
      ) : (
        <>
          <ul className="voice-clone__voices">
            {voices.map((voice) => (
              <li key={voice.id}>
                <label>
                  <input type="radio" name="clone-voice" checked={voice.id === voiceId} onChange={() => setVoiceId(voice.id)} /> {voice.name}
                </label>
                <span className="muted-text">
                  {" "}
                  {voice.consent.basis === "own"
                    ? t("voice.clone.consentOwn", { defaultValue: "own voice, consent recorded {date}", date: voice.consent.consentedAt.slice(0, 10) })
                    : t("voice.clone.consentPermission", {
                        defaultValue: "{name} gave consent on {date}",
                        name: voice.consent.speakerName,
                        date: voice.consent.consentedAt.slice(0, 10),
                      })}
                </span>{" "}
                <button type="button" className="secondary-button" onClick={() => void remove(voice)}>
                  {t("voice.clone.delete", { defaultValue: "Delete" })}
                </button>
              </li>
            ))}
          </ul>
          <button type="button" className="secondary-button" onClick={() => setAdding(true)}>
            {t("voice.clone.new", { defaultValue: "New cloned voice" })}
          </button>
          {voices.length > 0 ? (
            <>
              <textarea
                className="text-input voice-textarea"
                value={text}
                maxLength={MAX_CHARS}
                onChange={(e) => setText(e.target.value)}
                placeholder={t("voice.clone.placeholder", { defaultValue: "Text to say in the cloned voice." })}
              />
              <label className="voice-clone__confirm">
                <input type="checkbox" checked={disclaimer} onChange={(e) => setDisclaimer(e.target.checked)} />{" "}
                {t("voice.clone.disclaimer", { defaultValue: "Start with a spoken line: \"This voice was generated by AI.\"" })}
              </label>
              <div className="button-row">
                <button type="button" className="primary-button" disabled={busy || !text.trim() || !voiceId || !model?.installed} onClick={() => void speak()}>
                  {busy ? t("voice.speak.generating", { defaultValue: "Generating…" }) : t("voice.speak.generate", { defaultValue: "Generate audio" })}
                </button>
              </div>
            </>
          ) : null}
        </>
      )}

      {error ? (
        <p className="voice-error" role="alert">
          {error}
        </p>
      ) : null}
      {result ? (
        <div className="voice-studio__result">
          <audio controls src={result.url} />
          <div className="button-row">
            <span className="muted-text">{formatClock(result.item.durationS ?? 0)}</span>
            <span className="badge">{t("voice.clone.tagged", { defaultValue: "Labelled as AI-made" })}</span>
            <button type="button" className="secondary-button" onClick={() => downloadBlob(result.blob, "cloned-speech.wav")}>
              WAV
            </button>
            <button type="button" className="secondary-button" onClick={() => void getLibraryAudio(result.item.id, "mp3").then((b) => downloadBlob(b, "cloned-speech.mp3"))}>
              MP3
            </button>
            <span className="muted-text">{t("voice.savedToLibrary", { defaultValue: "Saved to Library" })}</span>
            <SendToStoryboardMenu
              item={sendItem}
              projects={storyboard.projects}
              loaded={storyboard.loaded}
              busy={storyboard.busy}
              notice={storyboard.notice}
              enabled={storyboard.enabled}
              onOpen={() => void storyboard.loadProjects()}
              onDismissNotice={storyboard.dismissNotice}
              onSend={(projectId, target) => storyboard.send(sendItem as SendItem, projectId, target)}
            />
          </div>
        </div>
      ) : null}

      <label className="voice-clone__confirm">
        <input
          type="checkbox"
          checked={Boolean(settings.allowAgentCloning)}
          onChange={(e) => void saveSettings({ allowAgentCloning: e.target.checked })}
        />{" "}
        {t("voice.clone.allowAgents", { defaultValue: "Allow agents and other computers to use cloned voices (off by default)" })}
      </label>
    </div>
  );
}
