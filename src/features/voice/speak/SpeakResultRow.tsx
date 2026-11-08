import { useState } from "react";
import { useTranslation } from "react-i18next";
import { SendToStoryboardMenu } from "../../../components/SendToStoryboardMenu";
import { encodeMp3, getLibraryAudio, getLibraryItem, saveSpeech } from "../../../api";
import type { TabId } from "../../../types";
import { shotTitleFrom, type SendItem } from "../../storyboard/sendToStoryboardLogic";
import { useSendToStoryboard } from "../../storyboard/useSendToStoryboard";
import { downloadBlob, formatClock } from "../audio";

export interface SpeakOutput {
  url: string;
  blob: Blob;
  durationS: number;
  text: string;
  voice: string;
  speed: number;
  /** Made with a cloned voice: labelled as AI-made, watermarked and already in the Library. */
  cloned: boolean;
  libraryId: string | null;
}

/** The finished clip: audio, length, WAV, MP3, Save to Library and Send to Storyboard. */
export function SpeakResultRow({ output, onTabChange, onError }: { output: SpeakOutput; onTabChange?: (tab: TabId) => void; onError: (message: string) => void }) {
  const { t } = useTranslation("studio");
  const storyboard = useSendToStoryboard((tab) => onTabChange?.(tab));
  const [savedId, setSavedId] = useState<string | null>(output.libraryId);
  const [busyExport, setBusyExport] = useState(false);
  const fail = (err: unknown) => onError(err instanceof Error ? err.message : String(err));

  const exportMp3 = async () => {
    setBusyExport(true);
    try {
      const blob = output.cloned && output.libraryId ? await getLibraryAudio(output.libraryId, "mp3") : await encodeMp3(output.blob);
      downloadBlob(blob, "speech.mp3");
    } catch (err) {
      fail(err);
    } finally {
      setBusyExport(false);
    }
  };

  const ensureSaved = async (): Promise<string> => {
    if (savedId) return savedId;
    const item = await saveSpeech(output.blob, { text: output.text, voice: output.voice, speed: output.speed });
    setSavedId(item.id);
    return item.id;
  };

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
    <div className="voice-studio__result">
      <audio controls src={output.url} />
      <div className="button-row">
        <span className="muted-text">{formatClock(output.durationS)}</span>
        {output.cloned ? <span className="badge">{t("voice.clone.tagged", { defaultValue: "Labelled as AI-made" })}</span> : null}
        <button type="button" className="secondary-button" onClick={() => downloadBlob(output.blob, "speech.wav")}>
          WAV
        </button>
        <button type="button" className="secondary-button" disabled={busyExport} onClick={() => void exportMp3()}>
          {busyExport ? t("voice.speak.encoding", { defaultValue: "Encoding…" }) : "MP3"}
        </button>
        <button type="button" className="secondary-button" disabled={savedId != null} onClick={() => void ensureSaved().catch(fail)}>
          {savedId ? t("voice.savedToLibrary", { defaultValue: "Saved to Library" }) : t("voice.saveToLibrary", { defaultValue: "Save to Library" })}
        </button>
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
