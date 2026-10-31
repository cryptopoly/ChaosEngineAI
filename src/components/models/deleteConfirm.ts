/**
 * Confirmation before a Models page deletes model files.
 *
 * Chat always asked; image and video deleted at once, shared downloads
 * included. Every area now asks, and when several catalog rows share one
 * download the message names the rows that lose their files too.
 */

import i18next from "i18next";
import type { ModelEntry } from "./modelEntry";

const MAX_NAMED_SIBLINGS = 6;

export function buildDeleteConfirmMessage(
  entry: Pick<ModelEntry, "name" | "repo" | "localPath" | "deleteTarget" | "shared">,
): string {
  const target = entry.localPath || entry.deleteTarget || entry.repo;
  const lines = [
    i18next.t("library:models.delete.confirmTitle", {
      defaultValue: "Delete the files of \"{name}\"?",
      name: entry.name,
    }),
    i18next.t("library:models.delete.confirmTarget", {
      defaultValue: "This removes {target} from disk. It cannot be undone.",
      target,
    }),
  ];
  const shared = entry.shared;
  if (shared && shared.count > 0) {
    const named = shared.siblingNames.slice(0, MAX_NAMED_SIBLINGS);
    const more = Math.max(0, shared.count - named.length);
    const list = named.length > 0
      ? named.map((name) => `• ${name}`).join("\n")
      : "";
    lines.push(
      i18next.t("library:models.delete.confirmShared", {
        defaultValue:
          "{count, plural, one {# other model uses} other {# other models use}} the same download and will be removed too:",
        count: shared.count,
      }),
    );
    if (list) lines.push(list);
    if (more > 0) {
      lines.push(
        i18next.t("library:models.delete.confirmSharedMore", {
          defaultValue: "and {count} more",
          count: more,
        }),
      );
    }
  }
  return lines.join("\n\n");
}

export type ConfirmFn = (message: string) => boolean;

function defaultConfirm(message: string): boolean {
  if (typeof window === "undefined" || typeof window.confirm !== "function") return false;
  return window.confirm(message);
}

/** Ask, then run ``onDelete`` only on a yes. Returns whether it ran. */
export function confirmAndDelete(
  entry: Pick<ModelEntry, "name" | "repo" | "localPath" | "deleteTarget" | "shared">,
  onDelete: (target: string) => void,
  confirm: ConfirmFn = defaultConfirm,
): boolean {
  if (!confirm(buildDeleteConfirmMessage(entry))) return false;
  onDelete(entry.deleteTarget);
  return true;
}
