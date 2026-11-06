/**
 * The licence at a glance: short name plus what it means for commercial
 * use, with the one-sentence summary as the tooltip. Click opens the
 * licence screen. The words come from the catalog entry, not from here.
 */
import { useTranslation } from "react-i18next";

export type CommercialUse = "yes" | "limited" | "no";

export interface ModelLicence {
  id: string;
  name: string;
  commercial: CommercialUse;
  /** A Hub accept is needed before the weights can be downloaded. */
  gated: boolean;
  /** One sentence, shown as the tooltip. */
  summary: string;
  /** Bullet lines for the licence screen. */
  summaryLines?: string[];
  /** Extra note shown on the licence screen (for example a provenance note). */
  note?: string;
}

export interface LicenceChipProps {
  licence: ModelLicence;
  onOpen?: () => void;
}

export function LicenceChip({ licence, onOpen }: LicenceChipProps) {
  const { t } = useTranslation("studio");
  const text =
    licence.commercial === "yes"
      ? t("licence.chip.commercial", { defaultValue: "{name} · commercial", name: licence.name })
      : licence.commercial === "limited"
        ? t("licence.chip.limited", { defaultValue: "{name} · limited commercial", name: licence.name })
        : t("licence.chip.noCommercial", { defaultValue: "{name} · no commercial use", name: licence.name });
  const className = `licence-chip licence-chip--${licence.commercial}${licence.gated ? " licence-chip--gated" : ""}`;
  const content = (
    <>
      {licence.gated ? <span className="licence-chip__gated">{t("licence.chip.gated", { defaultValue: "Gated" })}</span> : null}
      <span className="licence-chip__text">{text}</span>
    </>
  );

  if (!onOpen) {
    return (
      <span className={className} title={licence.summary}>
        {content}
      </span>
    );
  }
  return (
    <button type="button" className={className} title={licence.summary} onClick={onOpen}>
      {content}
    </button>
  );
}
