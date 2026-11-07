import { isTauri } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { micSettingsUrlForError } from "./micSettings";

/** "Open microphone settings" next to a blocked-microphone error. Renders nothing in a plain browser. */
export function MicSettingsButton({ error, platform }: { error: string | null | undefined; platform: string }) {
  const { t } = useTranslation("studio");
  const url = micSettingsUrlForError(error, platform);
  if (!url || !isTauri()) return null;
  return (
    <button
      type="button"
      className="secondary-button"
      onClick={() => {
        void import("@tauri-apps/plugin-opener").then(({ openUrl }) => openUrl(url)).catch(() => undefined);
      }}
    >
      {t("voice.micError.openSettings", { defaultValue: "Open microphone settings" })}
    </button>
  );
}
