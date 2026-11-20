/**
 * The licence modal for gated or restricted models. The bullet lines and
 * the note come from the catalog entry (props), never from this file, so a
 * changed licence is a catalog change.
 *
 * Gated and not yet accepted: Read / Open on Hugging Face, Not now,
 * I have accepted. Anything else (already accepted, or not gated): a plain
 * notice with OK only. Escape = Not now; focus stays inside the dialog and returns to the opener.
 *
 * ``territory`` (D-76): a model licensed only outside some territories. The screen lists the excluded
 * territories, links the full licence and needs a tick on the exact statement before Confirm is enabled.
 * The app does not check where the user is; the screen says so.
 */
import { useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useDialogFocus } from "../hooks/useDialogFocus";
import type { ModelLicence } from "./LicenceChip";

export interface TerritoryTerms {
  /** Where the licence does not apply, in one list ("European Union", "United Kingdom", ...). */
  excludedTerritories: string[];
  /** The exact statement the user ticks; sent back to the backend unchanged. */
  confirmSentence: string;
  /** The current licence page, shown beside the pinned one (the acceptable use policy can change). */
  liveLicenceUrl?: string;
}

export interface LicenceScreenProps {
  entry: { name: string };
  licence: ModelLicence;
  accepted: boolean;
  onAccept: () => void;
  onDismiss: () => void;
  onOpenHub: () => void;
  onOpenFull: () => void;
  /** Opens ``territory.liveLicenceUrl``. */
  onOpenLive?: () => void;
  territory?: TerritoryTerms;
  /** Shown under the tick when saving the statement failed. */
  error?: string | null;
}

export function LicenceScreen({
  entry, licence, accepted, onAccept, onDismiss, onOpenHub, onOpenFull, onOpenLive, territory, error,
}: LicenceScreenProps) {
  const { t } = useTranslation("studio");
  const titleId = useId();
  const descId = useId();
  const tickId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const [ticked, setTicked] = useState(false);
  const needsTerritory = Boolean(territory) && !accepted;
  const needsAccept = licence.gated && !accepted && !territory;
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
          {needsTerritory && territory ? (
            <div className="licence-screen__territory">
              <p>
                {t("territoryGate.screen.excluded", {
                  defaultValue: "Not licensed for use in: {territories}.",
                  territories: territory.excludedTerritories.join(", "),
                })}
              </p>
              <p className="muted-text">
                {t("territoryGate.screen.noLookup", {
                  defaultValue: "ChaosEngineAI does not check where you are. Your tick is your own statement.",
                })}
              </p>
              <div className="button-row">
                <button type="button" className="secondary-button" onClick={onOpenFull}>
                  {t("licence.screen.readFull", { defaultValue: "Read the full licence" })}
                </button>
                {territory.liveLicenceUrl && onOpenLive ? (
                  <button type="button" className="secondary-button" onClick={onOpenLive}>
                    {t("licence.screen.readLive", { defaultValue: "Read the current licence on Hugging Face" })}
                  </button>
                ) : null}
              </div>
              <label className="licence-screen__tick" htmlFor={tickId}>
                <input id={tickId} type="checkbox" checked={ticked} onChange={(event) => setTicked(event.target.checked)} />
                <span>{territory.confirmSentence}</span>
              </label>
              {error ? (
                <p className="voice-error" role="alert">
                  {error}
                </p>
              ) : null}
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
          {needsTerritory ? (
            <>
              <button type="button" className="secondary-button" onClick={onDismiss}>
                {t("licence.screen.notNow", { defaultValue: "Not now" })}
              </button>
              <button type="button" className="primary-button" disabled={!ticked} onClick={onAccept}>
                {t("territoryGate.screen.confirm", { defaultValue: "Confirm" })}
              </button>
            </>
          ) : needsAccept ? (
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
