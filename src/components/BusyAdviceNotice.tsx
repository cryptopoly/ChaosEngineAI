import { useTranslation } from "react-i18next";
import type { BusyAdvice } from "../types";

export interface BusyAdviceNoticeProps {
  advice: BusyAdvice;
  /** Chat shows it dismissible; the load dialog and Studio strip keep it in view. */
  onDismiss?: () => void;
}

/** One inline notice for "the machine is busy": slower (amber) or risky (red). Reuses the chat banner styles. */
export function BusyAdviceNotice({ advice, onDismiss }: BusyAdviceNoticeProps) {
  const { t } = useTranslation("chat");
  const risky = advice.level === "risky";
  return (
    <div
      className={risky ? "panic-banner" : "panic-banner panic-banner--thermal panic-banner--moderate"}
      role={risky ? "alert" : "status"}
    >
      <div className="panic-banner__body">
        <strong className="panic-banner__title">
          {risky
            ? t("busy.riskyTitle", { defaultValue: "Not enough memory for both" })
            : t("busy.slowerTitle", { defaultValue: "Machine is busy" })}
        </strong>
        <p className="panic-banner__message">{advice.message}</p>
      </div>
      {onDismiss ? (
        <button type="button" className="secondary-button" onClick={onDismiss}>
          {t("busy.dismiss", { defaultValue: "Dismiss" })}
        </button>
      ) : null}
    </div>
  );
}
