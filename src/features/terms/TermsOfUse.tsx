import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { useDialogFocus } from "../../hooks/useDialogFocus";
import { Panel } from "../../components/Panel";
import { TERMS_POINT_IDS, formatAcceptedDate, termsNeedAcceptance, type TermsSettings } from "./termsLogic";

const POINT_DEFAULTS: Record<(typeof TERMS_POINT_IDS)[number], string> = {
  responsibility: "You are responsible for what you generate and how you use it.",
  disclosure:
    "Say when content is AI-generated wherever the law or a platform requires it (for example Article 50 of the EU AI Act and platform labelling rules).",
  impersonation: "Do not use ChaosEngineAI to impersonate a real person without their consent.",
  voices: "Only clone a voice with the speaker's consent.",
  licences: "Each model comes with its own licence. Check it before you use the output.",
};

export interface TermsDialogProps {
  /** First run: I agree and Quit. Read-only: a single Close. */
  mode: "accept" | "read";
  onAgree?: () => void;
  onQuit?: () => void;
  onClose?: () => void;
  busy?: boolean;
  error?: string | null;
}

export function TermsDialog({ mode, onAgree, onQuit, onClose, busy, error }: TermsDialogProps) {
  const { t } = useTranslation("common");
  const dialog = useRef<HTMLDivElement>(null);
  // First run must be answered, so Escape only closes the read-only view. Focus starts on the dialog so the title and points are read first.
  useDialogFocus(dialog, { initialFocus: "container", onEscape: mode === "read" ? onClose : undefined });
  return (
    <div className="modal-overlay terms-overlay">
      <div
        ref={dialog}
        className="modal-content terms-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="terms-title"
        aria-describedby="terms-points"
        tabIndex={-1}
      >
        <div className="modal-header">
          <h3 id="terms-title">{t("terms.title", { defaultValue: "Terms of use" })}</h3>
          {mode === "accept" ? (
            <p className="muted-text">{t("terms.intro", { defaultValue: "Please read this before you start." })}</p>
          ) : null}
        </div>
        <ul id="terms-points" className="terms-points">
          {TERMS_POINT_IDS.map((id) => (
            <li key={id}>{t(`terms.points.${id}`, { defaultValue: POINT_DEFAULTS[id] })}</li>
          ))}
        </ul>
        {error ? <p className="error-text" role="alert">{error}</p> : null}
        <div className="terms-actions">
          {mode === "accept" ? (
            <>
              <button className="secondary-button" type="button" onClick={onQuit} disabled={busy}>
                {t("terms.quit", { defaultValue: "Quit" })}
              </button>
              <button className="primary-button" type="button" onClick={onAgree} disabled={busy}>
                {t("terms.agree", { defaultValue: "I agree" })}
              </button>
            </>
          ) : (
            <button className="primary-button" type="button" onClick={onClose}>
              {t("terms.close", { defaultValue: "Close" })}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/** Renders the first-run dialog only while the loaded settings lack acceptance of the current version. */
export function TermsGate({
  settings,
  ...rest
}: { settings: TermsSettings | null | undefined } & Omit<TermsDialogProps, "mode" | "onClose">) {
  if (!termsNeedAcceptance(settings)) return null;
  return <TermsDialog mode="accept" {...rest} />;
}

/** Settings > General row: when the terms were accepted, and a button to read them again. */
export function TermsOfUsePanel({ settings, onRead }: { settings: TermsSettings | null | undefined; onRead: () => void }) {
  const { t, i18n } = useTranslation("common");
  const date = formatAcceptedDate(settings?.termsAcceptedAt, i18n?.language);
  return (
    <Panel title={t("terms.title", { defaultValue: "Terms of use" })}>
      <div className="control-stack">
        <p className="muted-text">
          {date
            ? t("terms.acceptedOn", { defaultValue: "Accepted on {date}.", date })
            : t("terms.notAccepted", { defaultValue: "Not accepted yet." })}
        </p>
        <button className="secondary-button" type="button" onClick={onRead}>
          {t("terms.read", { defaultValue: "Read the terms" })}
        </button>
      </div>
    </Panel>
  );
}
