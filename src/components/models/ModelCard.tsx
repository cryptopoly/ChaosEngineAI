import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { ModelEntry, ModelMemoryEstimate } from "./modelEntry";
import type { ResolvedModelStatus } from "./modelStatus";
import { ModelStatusChip } from "./ModelStatusChip";
import { BestFitRibbon, FitBadge } from "./FitBadge";
import type { Fit } from "./fit";

export interface ModelCardProps {
  entry: ModelEntry;
  status: ResolvedModelStatus;
  /** Default: "provider · family name". */
  subtitle?: ReactNode;
  /** Chips under the name (style tags, access, accelerator pills...). */
  badges?: ReactNode;
  /** The area's middle columns, in header order. */
  cells: ReactNode[];
  /** Default: ``<ModelStatusChip>``. */
  statusCell?: ReactNode;
  /** Usually ``<ModelDownloadControls>``. */
  actions: ReactNode;
  /** Expandable detail (note, path, licence, model card link...). Pass a
   * function to build it only while the row is expanded. */
  detail?: ReactNode | (() => ReactNode);
  expanded?: boolean;
  onToggleExpanded?: () => void;
  /** Full-width notes under the row (download error, gated-access help). */
  callouts?: ReactNode;
  /** Tooltip of the "Best fit" ribbon when the area picks differently. */
  bestFitTitle?: string;
}

/** "provider · family" without repeating a family named like the provider. */
export function defaultModelSubtitle(entry: Pick<ModelEntry, "provider" | "familyName">): string {
  const parts = [entry.provider, entry.familyName].filter((part): part is string => Boolean(part && part.trim()));
  const unique = parts.filter((part, index) => parts.findIndex((other) => other.toLowerCase() === part.toLowerCase()) === index);
  return unique.join(" · ");
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      className={`models-card__chevron${open ? " is-open" : ""}`}
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="m9 6 6 6-6 6" />
    </svg>
  );
}

export function ModelCard({
  entry,
  status,
  subtitle,
  badges,
  cells,
  statusCell,
  actions,
  detail,
  expanded = false,
  onToggleExpanded,
  callouts,
  bestFitTitle,
}: ModelCardProps) {
  const { t } = useTranslation("library");
  const classes = ["media-model-row-wrap", "models-card"];
  if (status.state === "installed") classes.push("downloaded");
  if (status.bucket === "downloading") classes.push("download-active");
  if (status.bucket === "attention") classes.push("download-warning");
  if (!entry.hostCompatible) classes.push("models-card--incompatible");
  if (expanded) classes.push("is-expanded");
  const subtitleNode = subtitle ?? defaultModelSubtitle(entry);
  const expandable = Boolean(detail && onToggleExpanded);
  return (
    <div className={classes.join(" ")} data-model-id={entry.id} data-status={status.state}>
      <div className="media-model-row models-card__row">
        <div className="media-model-name models-card__name">
          <div className="models-card__title">
            {expandable ? (
              <button
                type="button"
                className="models-card__toggle"
                aria-expanded={expanded}
                title={
                  expanded
                    ? t("models.card.collapse", { defaultValue: "Hide details" })
                    : t("models.card.expand", { defaultValue: "Show details" })
                }
                onClick={onToggleExpanded}
              >
                <Chevron open={expanded} />
                <strong>{entry.name}</strong>
              </button>
            ) : (
              <strong>{entry.name}</strong>
            )}
            {entry.bestFit ? <BestFitRibbon title={bestFitTitle} /> : null}
          </div>
          {subtitleNode ? <small>{subtitleNode}</small> : null}
          <div className="media-model-chip-row">
            {!entry.hostCompatible ? (
              <span
                className="badge models-incompatible-chip"
                title={t("models.card.incompatibleTitle", {
                  defaultValue: "Its engine does not run on this machine. You can still delete its files.",
                })}
              >
                {t("models.card.incompatible", { defaultValue: "Cannot run on this machine" })}
              </span>
            ) : null}
            {badges}
          </div>
        </div>
        {cells.map((cell, index) => (
          <span key={index} className="models-card__cell">
            {cell}
          </span>
        ))}
        <span className="models-card__status">{statusCell ?? <ModelStatusChip status={status} reason={entry.broken?.reason} />}</span>
        {actions}
      </div>
      {expanded && detail ? (
        <div className="models-card__detail">{typeof detail === "function" ? detail() : detail}</div>
      ) : null}
      {callouts}
    </div>
  );
}

function memoryParts(label: string | null | undefined): { primary: string; secondary: string | null } | null {
  if (!label) return null;
  const [primary, secondary] = label.split(" @ ");
  if (!secondary) return { primary, secondary: null };
  return { primary: `${primary} @`, secondary };
}

/** RAM / VRAM estimate with the fit verdict under it. */
export function ModelMemoryCell({
  memory,
  fit,
  pendingLabel,
  pendingTitle,
}: {
  memory: ModelMemoryEstimate | null | undefined;
  fit?: Fit | null;
  pendingLabel: string;
  pendingTitle: string;
}) {
  const parts = memoryParts(memory?.label);
  return (
    <span className="media-model-memory models-memory-cell" title={memory?.title ?? pendingTitle}>
      <span className="models-memory-cell__text">
        <span>{parts?.primary ?? pendingLabel}</span>
        {parts?.secondary ? <small>{parts.secondary}</small> : null}
      </span>
      <FitBadge fit={fit} />
    </span>
  );
}

/** A label / value list for the expandable detail. */
export function ModelDetailList({ items }: { items: Array<{ label: string; value: ReactNode } | null | false | undefined> }) {
  const rows = items.filter((item): item is { label: string; value: ReactNode } => Boolean(item));
  if (rows.length === 0) return null;
  return (
    <dl className="models-detail-list">
      {rows.map((row) => (
        <div key={row.label} className="models-detail-list__row">
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}
