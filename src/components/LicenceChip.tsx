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
  /** The full licence text on the web. */
  url?: string;
  /** Extra note shown on the licence screen (for example a provenance note). */
  note?: string;
  /** Licensed only outside some territories (D-76): the user confirmed a statement before the row appeared. */
  territoryRestricted?: boolean;
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
  const className = `licence-chip licence-chip--${licence.commercial}${licence.gated || licence.territoryRestricted ? " licence-chip--gated" : ""}`;
  const content = (
    <>
      {licence.territoryRestricted ? (
        <span className="licence-chip__gated">{t("territoryGate.chip", { defaultValue: "Territory-limited" })}</span>
      ) : null}
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

export interface LicenceNoteProps {
  licence: ModelLicence;
  /** Opens the licence page (the app's external-link opener). Without it the name is a plain link. */
  onOpenLink?: (url: string) => void;
}

/**
 * The licence on a model card (D0135): the licence's own name as a link to its text, then what it means for
 * commercial use in plain words, and "Gated" when the repo needs a Hugging Face accept. The name stays English;
 * the words around it are translated. Same pill as ``LicenceChip``, which opens the licence screen instead.
 */
export function LicenceNote({ licence, onOpenLink }: LicenceNoteProps) {
  const { t } = useTranslation("studio");
  const words =
    licence.commercial === "yes"
      ? t("licence.note.commercial", { defaultValue: "Commercial use allowed" })
      : licence.commercial === "limited"
        ? t("licence.note.limited", { defaultValue: "Limited commercial use" })
        : t("licence.note.noCommercial", { defaultValue: "Not for commercial use" });
  const url = licence.url;
  const name = url ? (
    <a
      className="licence-chip__link"
      href={url}
      target="_blank"
      rel="noreferrer"
      onClick={(event) => {
        event.stopPropagation();
        if (onOpenLink) {
          event.preventDefault();
          onOpenLink(url);
        }
      }}
    >
      {licence.name}
    </a>
  ) : (
    licence.name
  );
  const className = `licence-chip licence-chip--${licence.commercial}${licence.gated || licence.territoryRestricted ? " licence-chip--gated" : ""}`;
  return (
    <span className={className} title={licence.summary}>
      {licence.territoryRestricted ? (
        <span className="licence-chip__gated">{t("territoryGate.chip", { defaultValue: "Territory-limited" })}</span>
      ) : null}
      {licence.gated ? <span className="licence-chip__gated">{t("licence.chip.gated", { defaultValue: "Gated" })}</span> : null}
      <span className="licence-chip__text">
        {name} · {words}
      </span>
    </span>
  );
}
