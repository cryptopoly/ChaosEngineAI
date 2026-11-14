/**
 * "Models with territory limits" (D-76): one collapsed notice on a Models page when a territory-limited model
 * exists for that kind. Review opens the licence screen with the territory tick; Confirm stores the user's
 * statement through the backend (the only place it counts) and the page reloads its catalog so the rows appear.
 * Withdraw hides them again. Renders nothing while no such model is released.
 */
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  confirmLicenceGate,
  getLicenceGate,
  withdrawLicenceGate,
  type LicenceGateStatus,
  type TerritoryModel,
} from "../api/licenceGate";
import { LicenceScreen } from "./LicenceScreen";
import type { ModelLicence } from "./LicenceChip";

export interface TerritoryGateNoticeProps {
  kind: "image" | "video";
  /** Reload the page's catalog after a confirm or withdraw. */
  onChanged: () => void;
  onOpenExternalUrl: (url: string) => void;
  /** Status to show before the first fetch answers (tests and server rendering). */
  initial?: LicenceGateStatus;
  /** Injectable for tests. */
  api?: {
    get: () => Promise<LicenceGateStatus>;
    confirm: (key: string, sentence: string) => Promise<LicenceGateStatus>;
    withdraw: (key: string) => Promise<LicenceGateStatus>;
  };
}

const DEFAULT_API = { get: getLicenceGate, confirm: confirmLicenceGate, withdraw: withdrawLicenceGate };

export function territoryLicence(model: TerritoryModel): ModelLicence {
  return {
    id: model.key,
    name: model.licenceName,
    commercial: "limited",
    gated: false,
    territoryRestricted: true,
    summary: model.summaryLines[0] ?? model.licenceName,
    summaryLines: model.summaryLines,
    url: model.licenceUrl,
    note: model.note,
  };
}

export function TerritoryGateNotice({
  kind, onChanged, onOpenExternalUrl, initial, api = DEFAULT_API,
}: TerritoryGateNoticeProps) {
  const { t } = useTranslation("studio");
  const [models, setModels] = useState<TerritoryModel[]>(
    () => (initial?.models ?? []).filter((model) => model.kinds.includes(kind)),
  );
  const [reviewing, setReviewing] = useState<TerritoryModel | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apply = useCallback(
    (status: LicenceGateStatus) => setModels(status.models.filter((model) => model.kinds.includes(kind))),
    [kind],
  );

  useEffect(() => {
    let alive = true;
    api
      .get()
      .then((status) => {
        if (alive) apply(status);
      })
      .catch(() => {
        // No notice when the backend cannot answer; the rows stay hidden either way.
      });
    return () => {
      alive = false;
    };
  }, [api, apply]);

  if (models.length === 0) return null;

  async function confirm(model: TerritoryModel) {
    setError(null);
    try {
      apply(await api.confirm(model.key, model.confirmSentence));
      setReviewing(null);
      onChanged();
    } catch (err) {
      setError(
        t("territoryGate.screen.error", {
          defaultValue: "Could not save your confirmation: {message}",
          message: err instanceof Error ? err.message : String(err),
        }),
      );
    }
  }

  async function withdraw(model: TerritoryModel) {
    try {
      apply(await api.withdraw(model.key));
      onChanged();
    } catch {
      // The record stays as it was; the next page load shows the true state.
    }
  }

  return (
    <details className="territory-gate-notice">
      <summary>{t("territoryGate.notice.title", { defaultValue: "Models with territory limits" })}</summary>
      <p className="muted-text">
        {t("territoryGate.notice.body", {
          defaultValue:
            "Some models may only be used outside certain countries. They stay hidden until you review the licence and confirm.",
        })}
      </p>
      <ul className="territory-gate-notice__list">
        {models.map((model) => (
          <li key={model.key}>
            <strong>{model.name}</strong>{" "}
            <span className="muted-text">
              {model.confirmed
                ? t("territoryGate.notice.confirmed", { defaultValue: "Confirmed: listed below" })
                : t("territoryGate.notice.hidden", { defaultValue: "Hidden until you confirm" })}
            </span>{" "}
            {model.confirmed ? (
              <button type="button" className="secondary-button" onClick={() => void withdraw(model)}>
                {t("territoryGate.notice.withdraw", { defaultValue: "Withdraw" })}
              </button>
            ) : (
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  setError(null);
                  setReviewing(model);
                }}
              >
                {t("territoryGate.notice.review", { defaultValue: "Review" })}
              </button>
            )}
          </li>
        ))}
      </ul>
      {reviewing ? (
        <LicenceScreen
          entry={{ name: reviewing.name }}
          licence={territoryLicence(reviewing)}
          accepted={false}
          territory={{ excludedTerritories: reviewing.excludedTerritories, confirmSentence: reviewing.confirmSentence }}
          error={error}
          onAccept={() => void confirm(reviewing)}
          onDismiss={() => setReviewing(null)}
          onOpenHub={() => onOpenExternalUrl(reviewing.licenceUrl)}
          onOpenFull={() => onOpenExternalUrl(reviewing.licenceUrl)}
        />
      ) : null}
    </details>
  );
}
