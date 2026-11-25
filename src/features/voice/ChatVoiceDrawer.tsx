import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Drawer } from "../../components/Drawer";
import { VoiceErrorNotice } from "./clone/VoiceErrorNotice";
import { describeVoiceError, type VoiceErrorView } from "./voiceErrorText";
import { voicesForUse, type VoiceUse } from "./speak/speakLogic";
import type { VoiceCatalog } from "./speak/useVoiceCatalog";
import { useVoiceSample } from "./speak/useVoiceSample";
import { VoicePicker } from "./speak/VoicePicker";

/** The three voice choices Chat owns (D0142); the settings keys are the same as before. */
export type ChatVoiceUse = Extract<VoiceUse, "read" | "call" | "fallback">;

/**
 * The voice picker that opens from Chat's Voice popover: the same cards as the Speak picker, limited to the voices
 * that can serve the chosen use (read-aloud, calls or backup), with a way to set no backup voice. A click saves the
 * choice at once and closes the drawer; a failed save shows one sentence and the raw text under Details.
 */
export function ChatVoiceDrawer({
  use,
  catalog,
  currentId,
  speed,
  onAssign,
  onClose,
  onGetMore,
}: {
  use: ChatVoiceUse;
  catalog: VoiceCatalog;
  /** The voice set for this use now ("" for no backup voice). */
  currentId: string;
  /** Reading speed, for the sample line. */
  speed: number;
  /** Save the choice; rejects when it cannot be saved. */
  onAssign: (id: string) => Promise<void>;
  onClose: () => void;
  /** Open Voice Models (the speech models, the add-ons). */
  onGetMore: () => void;
}) {
  const { t } = useTranslation("studio");
  const sample = useVoiceSample(speed);
  const [error, setError] = useState<VoiceErrorView | null>(null);

  const title =
    use === "read"
      ? t("voice.chatVoices.chooseRead", { defaultValue: "Choose the voice that reads Chat replies" })
      : use === "call"
        ? t("voice.chatVoices.chooseCall", { defaultValue: "Choose the voice for calls" })
        : t("voice.chatVoices.chooseBackup", { defaultValue: "Choose the backup voice" });
  const hint =
    use === "call"
      ? t("voice.voices.chooseCallHint", { defaultValue: "Voices that can answer calls. Cloned voices are not used in calls." })
      : use === "read"
        ? t("voice.voices.chooseReadHint", { defaultValue: "Voices that can read chat replies aloud." })
        : t("voice.chatVoices.chooseBackupHint", {
            defaultValue: "Installed voices that read when the chosen chat or call voice cannot. A cloned voice is used for read-aloud only.",
          });

  // The save in flight: a click selects (saves) and then chooses (closes), and it closes only when the save worked.
  const lastSave = useRef<Promise<boolean>>(Promise.resolve(false));
  const assign = (id: string, close: boolean) => {
    setError(null);
    const saving = onAssign(id).then(
      () => true,
      (err: unknown) => {
        setError(describeVoiceError(err));
        return false;
      },
    );
    lastSave.current = saving;
    if (close) void saving.then((ok) => ok && onClose());
  };

  return (
    <Drawer title={title} subtitle={hint} onClose={onClose} className="voice-drawer">
      {use === "fallback" ? (
        <div className="button-row voice-drawer__none">
          <button type="button" className="secondary-button" aria-pressed={currentId === ""} onClick={() => assign("", true)}>
            {t("voice.chatVoices.noBackup", { defaultValue: "No backup voice" })}
          </button>
        </div>
      ) : null}
      {error ? <VoiceErrorNotice view={error} /> : null}
      <VoicePicker
        voices={voicesForUse(catalog.voices, use)}
        models={catalog.models}
        selectedId={currentId}
        onSelect={(id) => assign(id, false)}
        onChoose={() => void lastSave.current.then((ok) => ok && onClose())}
        modelsAvailable={catalog.modelsAvailable}
        isPlaying={sample.isPlaying}
        isLoading={sample.isLoading}
        onSample={(entry) => void sample.play(entry)}
        onGetMore={onGetMore}
        onAddonInstalled={() => void catalog.refresh()}
      />
    </Drawer>
  );
}
