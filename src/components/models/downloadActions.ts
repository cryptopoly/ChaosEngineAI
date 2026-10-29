/**
 * Which download / delete buttons a row shows, as data.
 *
 * Kept pure so the rules are testable without a DOM; ``ModelDownloadControls``
 * renders the list. Behaviour follows the old Discover tabs, which were
 * right where the My Models tabs were not: a paused row offers Resume (not
 * "Download model") and "Delete download".
 */

import type { ActionIconName } from "../ModelActionIcons";
import type { ResolvedModelStatus } from "./modelStatus";

export type DownloadActionKind =
  | "download"
  | "resume"
  | "retry"
  | "pause"
  | "cancel"
  | "deleteDownload"
  | "deleteModel"
  | "trackedOnly";

export interface DownloadActionSpec {
  kind: DownloadActionKind;
  icon: ActionIconName;
  /** i18n key under ``library:models.action``. */
  labelKey: string;
  defaultLabel: string;
  danger?: boolean;
  disabled?: boolean;
}

const SPECS: Record<DownloadActionKind, DownloadActionSpec> = {
  download: { kind: "download", icon: "download", labelKey: "models.action.download", defaultLabel: "Download model" },
  resume: { kind: "resume", icon: "resume", labelKey: "models.action.resume", defaultLabel: "Resume download" },
  retry: { kind: "retry", icon: "retry", labelKey: "models.action.retry", defaultLabel: "Retry download" },
  pause: { kind: "pause", icon: "pause", labelKey: "models.action.pause", defaultLabel: "Pause download" },
  cancel: { kind: "cancel", icon: "cancel", labelKey: "models.action.cancel", defaultLabel: "Cancel download", danger: true },
  deleteDownload: {
    kind: "deleteDownload",
    icon: "delete",
    labelKey: "models.action.deleteDownload",
    defaultLabel: "Delete download",
    danger: true,
  },
  deleteModel: {
    kind: "deleteModel",
    icon: "delete",
    labelKey: "models.action.deleteModel",
    defaultLabel: "Delete model",
    danger: true,
  },
  trackedOnly: {
    kind: "trackedOnly",
    icon: "download",
    labelKey: "models.action.trackedOnly",
    defaultLabel: "Tracked only — not yet launchable",
    disabled: true,
  },
};

export interface DownloadActionOptions {
  /** Nothing in the app can run the model yet (FU-061): one disabled
   * download button, whatever the state. */
  trackedOnly?: boolean;
  /** The area can delete the files of an installed model. */
  canDeleteModel?: boolean;
}

/** The ordered download / delete actions for a row. The launch action
 * (Generate, Chat...) of an installed row is not part of this list. */
export function downloadActionsFor(
  status: Pick<ResolvedModelStatus, "state">,
  options: DownloadActionOptions = {},
): DownloadActionSpec[] {
  if (options.trackedOnly) return [SPECS.trackedOnly];
  const canDelete = options.canDeleteModel !== false;
  switch (status.state) {
    case "installed":
      return canDelete ? [SPECS.deleteModel] : [];
    case "downloading":
      return [SPECS.pause, SPECS.cancel];
    case "paused":
      return [SPECS.resume, SPECS.deleteDownload];
    case "failed":
      return [SPECS.retry, SPECS.deleteDownload];
    case "broken":
      return [SPECS.retry, SPECS.deleteDownload];
    case "partial":
      return canDelete ? [SPECS.resume, SPECS.deleteModel] : [SPECS.resume];
    default:
      return [SPECS.download];
  }
}

export function downloadActionSpec(kind: DownloadActionKind): DownloadActionSpec {
  return SPECS[kind];
}
