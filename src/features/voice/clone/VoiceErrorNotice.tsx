import { useTranslation } from "react-i18next";
import type { VoiceErrorView } from "../voiceErrorText";

/**
 * A failed voice request in plain words: one sentence, at most one button, and the raw text under Details (voice
 * studio UX audit, change 6). ``onAction`` is given only when this screen can do something for the error's action;
 * without it the sentence stands alone.
 */
export function VoiceErrorNotice({ view, onAction }: { view: VoiceErrorView; onAction?: (() => void) | null }) {
  const { t } = useTranslation("studio");
  return (
    <div className="voice-error" role="alert">
      <p>{view.sentence}</p>
      {onAction && view.actionLabel ? (
        <button type="button" className="secondary-button" onClick={onAction}>
          {view.actionLabel}
        </button>
      ) : null}
      {view.details ? (
        <details className="voice-error__details">
          <summary>{t("voice.errors.details", { defaultValue: "Details" })}</summary>
          <p className="muted-text">{view.details}</p>
        </details>
      ) : null}
    </div>
  );
}
