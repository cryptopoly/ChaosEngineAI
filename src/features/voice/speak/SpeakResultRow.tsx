import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { SendToStoryboardMenu } from "../../../components/SendToStoryboardMenu";
import { encodeMp3, getLibraryAudio, getLibraryItem, revealModelPath, saveSpeech } from "../../../api";
import type { TabId, VoiceEntryKind } from "../../../types";
import { shotTitleFrom, type SendItem } from "../../storyboard/sendToStoryboardLogic";
import { useSendToStoryboard } from "../../storyboard/useSendToStoryboard";
import { clipFileName, downloadBlob, firstWords, formatClock } from "../audio";

export interface SpeakOutput {
  url: string;
  blob: Blob;
  durationS: number;
  text: string;
  voice: string;
  /** Display name and kind, stored with the Library item. */
  voiceName: string;
  voiceKind: VoiceEntryKind;
  speed: number;
  /** Made with a cloned voice: labelled as AI-made, watermarked and already in the Library. */
  cloned: boolean;
  /** Set once the clip is in the Library: every Speak result is saved there automatically. */
  libraryId: string | null;
}

/**
 * A finished clip: its first words and voice, audio, length, WAV, MP3, the Library link, Show file and Send to
 * Storyboard. Speak saves every result to the Library, so this row shows "Saved to Library" as a link to it with Show
 * file beside it; Save to Library is only offered again when that automatic save failed. Downloads are named from the
 * first words and the voice.
 */
export function SpeakResultRow({
  output,
  onTabChange,
  onError,
  onOpenLibrary,
}: {
  output: SpeakOutput;
  onTabChange?: (tab: TabId) => void;
  onError: (error: unknown) => void;
  /** Open the Voice Library on Speech. */
  onOpenLibrary?: () => void;
}) {
  const { t } = useTranslation("studio");
  const storyboard = useSendToStoryboard((tab) => onTabChange?.(tab));
  const [savedId, setSavedId] = useState<string | null>(output.libraryId);
  const [busyExport, setBusyExport] = useState(false);
  const fail = (err: unknown) => onError(err);
  const rowRef = useRef<HTMLDivElement>(null);
  // The player lands under the Generate button, below the fold in a short window: bring each new clip into view.
  useEffect(() => {
    rowRef.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [output.url]);

  const exportMp3 = async () => {
    setBusyExport(true);
    try {
      const blob = output.cloned && output.libraryId ? await getLibraryAudio(output.libraryId, "mp3") : await encodeMp3(output.blob);
      downloadBlob(blob, clipFileName(output.text, output.voiceName, "mp3"));
    } catch (err) {
      fail(err);
    } finally {
      setBusyExport(false);
    }
  };

  const ensureSaved = async (): Promise<string> => {
    if (savedId) return savedId;
    const item = await saveSpeech(output.blob, {
      text: output.text,
      voice: output.voice,
      speed: output.speed,
      voiceName: output.voiceName,
      voiceKind: output.voiceKind,
    });
    setSavedId(item.id);
    return item.id;
  };

  const showFile = async () => {
    try {
      const path = savedId ? (await getLibraryItem(savedId)).audioPath : null;
      if (!path) throw new Error(t("voice.speak.noFile", { defaultValue: "This clip has no file to show." }));
      await revealModelPath(path);
    } catch (err) {
      fail(err);
    }
  };

  const heading = firstWords(output.text, 6);
  // The clip may only exist in the browser, so sending it saves it to the Library first and reads the file's place.
  const sendItem: SendItem = {
    kind: "voice",
    label: shotTitleFrom(output.text),
    clonedVoice: output.cloned,
    resolvePath: async () => {
      const path = (await getLibraryItem(await ensureSaved())).audioPath;
      if (!path) throw new Error(t("sendToStoryboard.noFile", { defaultValue: "This item has no file to send." }));
      return path;
    },
  };

  return (
    <div className="voice-studio__result" ref={rowRef}>
      <p className="muted-text voice-speak__hint">
        {heading.words ? `"${heading.words}${heading.truncated ? "…" : ""}"` : ""}
        {output.voiceName ? ` · ${output.voiceName}` : ""}
      </p>
      <audio controls src={output.url} />
      <div className="button-row">
        <span className="muted-text">{formatClock(output.durationS)}</span>
        {output.cloned ? <span className="badge">{t("voice.clone.tagged", { defaultValue: "Labelled as AI-made" })}</span> : null}
        <button type="button" className="secondary-button" onClick={() => downloadBlob(output.blob, clipFileName(output.text, output.voiceName, "wav"))}>
          WAV
        </button>
        <button type="button" className="secondary-button" disabled={busyExport} onClick={() => void exportMp3()}>
          {busyExport ? t("voice.speak.encoding", { defaultValue: "Encoding…" }) : "MP3"}
        </button>
        {savedId ? (
          onOpenLibrary ? (
            <button type="button" className="link-button" onClick={onOpenLibrary}>
              {t("voice.savedToLibrary", { defaultValue: "Saved to Library" })}
            </button>
          ) : (
            <span className="muted-text">{t("voice.savedToLibrary", { defaultValue: "Saved to Library" })}</span>
          )
        ) : (
          <button type="button" className="secondary-button" onClick={() => void ensureSaved().catch(fail)}>
            {t("voice.saveToLibrary", { defaultValue: "Save to Library" })}
          </button>
        )}
        {savedId ? (
          <button type="button" className="link-button" onClick={() => void showFile()}>
            {t("voice.speak.showFile", { defaultValue: "Show file" })}
          </button>
        ) : null}
        <SendToStoryboardMenu
          item={sendItem}
          projects={storyboard.projects}
          loaded={storyboard.loaded}
          busy={storyboard.busy}
          notice={storyboard.notice}
          enabled={storyboard.enabled}
          onOpen={() => void storyboard.loadProjects()}
          onDismissNotice={storyboard.dismissNotice}
          onSend={(projectId, target) => storyboard.send(sendItem, projectId, target)}
        />
      </div>
    </div>
  );
}
