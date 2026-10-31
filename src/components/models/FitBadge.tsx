import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { Fit } from "./fit";

function gb(value: number): string {
  return value >= 10 ? value.toFixed(0) : value.toFixed(1);
}

export function fitLabel(fit: Fit, t: TFunction<"library">): string {
  switch (fit.status) {
    case "fits":
      return t("models.fit.fits", { defaultValue: "Fits" });
    case "tight":
      return t("models.fit.tight", { defaultValue: "Tight" });
    case "wont-fit":
      return t("models.fit.wontFit", { defaultValue: "Won't fit" });
    default:
      return t("models.fit.unknown", { defaultValue: "Not for this machine" });
  }
}

/** Tooltip: the backend's reason, the numbers, and where they come from. */
export function fitTooltip(fit: Fit, t: TFunction<"library">): string {
  const parts: string[] = [];
  if (fit.reason) parts.push(fit.reason);
  if (fit.status !== "unknown" && fit.needGb !== null && fit.usableGb > 0) {
    parts.push(
      t("models.fit.numbers", {
        defaultValue: "Needs about {need} GB of {usable} GB usable on this machine.",
        need: gb(fit.needGb),
        usable: gb(fit.usableGb),
      }),
    );
  }
  if (fit.status !== "unknown") {
    parts.push(
      fit.basis === "measured"
        ? t("models.fit.measuredNote", { defaultValue: "Measured on this machine." })
        : t("models.fit.estimatedNote", { defaultValue: "Estimated from the model size; not measured on this machine." }),
    );
  }
  return parts.join("\n");
}

/** Fits / Tight / Won't fit chip with the reason as its tooltip. An
 * estimated verdict carries a visible "est." mark. ``unknown`` (the row
 * cannot run on this kind of hardware) renders a neutral chip. Nothing
 * renders without a verdict. */
export function FitBadge({ fit, className }: { fit: Fit | null | undefined; className?: string }) {
  const { t } = useTranslation("library");
  if (!fit) return null;
  const label = fitLabel(fit, t);
  const title = fitTooltip(fit, t);
  const estimated = fit.status !== "unknown" && fit.basis === "estimated";
  return (
    <span
      className={`fit-badge fit-badge--${fit.status}${className ? ` ${className}` : ""}`}
      title={title || undefined}
      aria-label={title ? `${label}. ${title}` : label}
      data-fit-status={fit.status}
      data-fit-basis={fit.basis}
    >
      {label}
      {estimated ? (
        <small className="fit-badge__basis">
          {t("models.fit.estimatedShort", { defaultValue: "est." })}
        </small>
      ) : null}
    </span>
  );
}

/** Marks the family's best fit for this machine. Chat picks differently
 * (the larger model among rows released together), so an area can pass
 * its own explanation. */
export function BestFitRibbon({ title: titleOverride }: { title?: string } = {}) {
  const { t } = useTranslation("library");
  const title =
    titleOverride ??
    t("models.fit.bestFitTitle", {
      defaultValue: "The pick in this family for this machine: the newest release that fits comfortably.",
    });
  return (
    <span className="best-fit-ribbon" title={title} aria-label={title}>
      {t("models.fit.bestFit", { defaultValue: "Best fit" })}
    </span>
  );
}
