import type { MouseEvent, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { IconActionButton, type ActionIconName } from "../ModelActionIcons";
import { downloadActionsFor, type DownloadActionKind } from "./downloadActions";
import type { ResolvedModelStatus } from "./modelStatus";

/** A launch action of an installed row: Generate (image / video), Chat or
 * Load for server (chat). */
export interface ModelLaunchAction {
  id: string;
  icon: ActionIconName;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}

export interface ModelDownloadControlsProps {
  status: ResolvedModelStatus;
  /** Nothing can run the model yet (FU-061): one disabled button. */
  trackedOnly?: boolean;
  trackedOnlyTitle?: string;
  /** Per-area label overrides ("Delete shared text encoder" on video...). */
  labels?: Partial<Record<DownloadActionKind, string>>;
  /** Download, Resume and Retry. */
  onDownload: () => void;
  onPause: () => void;
  /** Cancel an in-flight download, or delete a paused / failed one. */
  onCancel: () => void;
  /** Delete the files of an installed or partial model. The caller asks
   * for confirmation (``confirmAndDelete``). Omit to hide the button. */
  onDeleteModel?: () => void;
  /** Primary launch action, shown on installed rows only. */
  launch?: ModelLaunchAction | null;
  /** Secondary launch actions, in a "More actions" menu (installed rows). */
  overflow?: ModelLaunchAction[];
  /** Trailing buttons for every state (Reveal, Open on Hugging Face). */
  trailing?: ReactNode;
}

function MoreIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <circle cx="5" cy="12" r="1.8" />
      <circle cx="12" cy="12" r="1.8" />
      <circle cx="19" cy="12" r="1.8" />
    </svg>
  );
}

function closeMenu(event: MouseEvent<HTMLElement>) {
  event.currentTarget.closest("details")?.removeAttribute("open");
}

/** A small menu of secondary actions. Uses <details> so it needs no state
 * and closes when an item is chosen. */
export function ModelActionMenu({ actions, label }: { actions: ModelLaunchAction[]; label?: string }) {
  const { t } = useTranslation("library");
  if (actions.length === 0) return null;
  const menuLabel = label ?? t("models.action.more", { defaultValue: "More actions" });
  return (
    <details className="model-action-menu">
      <summary className="secondary-button icon-button action-icon-button" title={menuLabel} aria-label={menuLabel}>
        <MoreIcon />
        <span className="sr-only">{menuLabel}</span>
      </summary>
      <div className="model-action-menu__list" role="menu">
        {actions.map((action) => (
          <button
            key={action.id}
            type="button"
            role="menuitem"
            className="model-action-menu__item"
            disabled={action.disabled}
            title={action.title}
            onClick={(event) => {
              closeMenu(event);
              action.onClick();
            }}
          >
            {action.label}
          </button>
        ))}
      </div>
    </details>
  );
}

export function ModelDownloadControls({
  status,
  trackedOnly,
  trackedOnlyTitle,
  labels,
  onDownload,
  onPause,
  onCancel,
  onDeleteModel,
  launch,
  overflow,
  trailing,
}: ModelDownloadControlsProps) {
  const { t } = useTranslation("library");
  const actions = downloadActionsFor(status, { trackedOnly, canDeleteModel: Boolean(onDeleteModel) });
  const installed = status.state === "installed" && !trackedOnly;

  function handlerFor(kind: DownloadActionKind): (() => void) | undefined {
    switch (kind) {
      case "download":
      case "resume":
      case "retry":
        return onDownload;
      case "pause":
        return onPause;
      case "cancel":
      case "deleteDownload":
        return onCancel;
      case "deleteModel":
        return onDeleteModel;
      default:
        return undefined;
    }
  }

  return (
    <div className="media-model-actions model-download-controls">
      {installed && launch ? (
        <IconActionButton
          icon={launch.icon}
          label={launch.label}
          title={launch.title}
          buttonStyle="primary"
          disabled={launch.disabled}
          onClick={launch.onClick}
        />
      ) : null}
      {installed && overflow && overflow.length > 0 ? <ModelActionMenu actions={overflow} /> : null}
      {actions.map((action) => {
        const label = labels?.[action.kind] ?? t(action.labelKey, { defaultValue: action.defaultLabel });
        return (
          <IconActionButton
            key={action.kind}
            icon={action.icon}
            label={label}
            title={action.kind === "trackedOnly" ? trackedOnlyTitle ?? label : undefined}
            danger={action.danger}
            disabled={action.disabled}
            onClick={handlerFor(action.kind)}
          />
        );
      })}
      {trailing}
    </div>
  );
}
