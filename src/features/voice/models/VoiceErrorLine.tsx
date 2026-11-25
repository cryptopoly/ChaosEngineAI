/** A failed voice request on Voice > Models: one plain sentence, the raw text behind a Details disclosure. */
import { useTranslation } from "react-i18next";
import type { VoiceErrorView } from "../voiceErrorText";

export function VoiceErrorLine({ view }: { view: VoiceErrorView }) {
  const { t } = useTranslation("studio");
  return (
    <div role="alert">
      <p className="voice-error">{view.sentence}</p>
      {view.details ? (
        <details className="voice-error-details">
          <summary>{t("voice.errors.detailsToggle", { defaultValue: "Details" })}</summary>
          <p className="muted-text">{view.details}</p>
        </details>
      ) : null}
    </div>
  );
}
