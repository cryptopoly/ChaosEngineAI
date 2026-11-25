import { useTranslation } from "react-i18next";
import type { VoiceEntry } from "../../../types";

/** The line under the main voices and in Chat's Voice popover: which voice reads when the chosen one cannot (D-75). */
export function FallbackNote({ voice }: { voice: VoiceEntry | null }) {
  const { t } = useTranslation("studio");
  return (
    <>
      {voice
        ? t("voice.voices.backupNoteVoice", {
            defaultValue: "If a chosen voice cannot speak (model not downloaded, or it fails to load), the backup voice {name} reads instead and Chat says so.",
            name: voice.name,
          })
        : t("voice.voices.backupNoteNone", {
            defaultValue: "No backup voice is set: if a chosen voice cannot speak, read-aloud and calls show an error instead of switching voice.",
          })}
    </>
  );
}
