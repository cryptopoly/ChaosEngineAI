/**
 * The licence modal for gated or restricted models. The bullet lines and
 * the note come from the catalog entry (props), never from this file, so a
 * changed licence is a catalog change.
 *
 * Gated and not yet accepted: Read / Open on Hugging Face, Not now,
 * I have accepted. Anything else (already accepted, or not gated): a plain
 * notice with OK only. Escape = Not now; focus stays inside the dialog and returns to the opener.
 */
import { useId, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useDialogFocus } from "../hooks/useDialogFocus";
import type { ModelLicence } from "./LicenceChip";

export interface LicenceScreenProps {
  entry: { name: string };
  licence: ModelLicence;
  accepted: boolean;
  onAccept: () => void;
  onDismiss: () => void;
  onOpenHub: () => void;
  onOpenFull: () => void;
}

export function LicenceScreen({ entry, licence, accepted, onAccept, onDismiss, onOpenHub, onOpenFull }: LicenceScreenProps) {
  const { t } = useTranslation("studio");
  const titleId = useId();
  const descId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const needsAccept = licence.gated && !accepted;
  const lines = licence.summaryLines ?? [];
  useDialogFocus(dialogRef, { onEscape: onDismiss });

  return (
    <div className="modal-overlay" role="presentation" onClick={onDismiss}>
      <div
        ref={dialogRef}
        className="modal-content licence-screen"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id={titleId}>{entry.name}</h3>
          <p id={descId}>{licence.name}</p>
        </div>
        <div className="modal-body">
          {lines.length > 0 ? (
            <div className="licence-screen__terms">
              <strong>{t("licence.screen.asks", { defaultValue: "What this licence asks of you" })}</strong>
              <ul>
                {lines.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {licence.note ? <p className="muted-text">{licence.note}</p> : null}
          {needsAccept ? (
            <p className="muted-text">
              {t("licence.screen.gatedNote", {
                defaultValue:
                  "This model is gated: accept the terms on Hugging Face, then download here with your token (Settings > Hugging Face).",
              })}
            </p>
          ) : null}
          {needsAccept ? (
            <div className="button-row">
              <button type="button" className="secondary-button" onClick={onOpenFull}>
                {t("licence.screen.readFull", { defaultValue: "Read the full licence" })}
              </button>
              <button type="button" className="secondary-button" onClick={onOpenHub}>
                {t("licence.screen.openHub", { defaultValue: "Open on Hugging Face" })}
              </button>
            </div>
          ) : null}
        </div>
        <div className="modal-footer licence-screen__footer">
          {needsAccept ? (
            <>
              <button type="button" className="secondary-button" onClick={onDismiss}>
                {t("licence.screen.notNow", { defaultValue: "Not now" })}
              </button>
              <button type="button" className="primary-button" onClick={onAccept}>
                {t("licence.screen.accepted", { defaultValue: "I have accepted" })}
              </button>
            </>
          ) : (
            <button type="button" className="primary-button" onClick={onDismiss}>
              {t("licence.screen.ok", { defaultValue: "OK" })}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
