/**
 * Settings > Disk > Clean up: lists storyboard projects, chat threads and gallery pictures and clips with size and
 * date, the user ticks what to delete, a confirm names the count and the total size and says it cannot be undone.
 *
 * Nothing is ticked by default and nothing is guessed to be test data. The backend lists the items and deletes only
 * ids from its own list; this panel sends the ticked ids and never a path. A project with a running batch comes back
 * refused with its reason, which the result note shows.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { deleteCleanupItems, getCleanupItems } from "../../api/cleanup";
import type { CleanupDeleteResult, CleanupItems } from "../../api/cleanup";
import { Panel } from "../../components/Panel";
import { formatBytes } from "../../utils/format";
import { ConfirmDialog } from "../storyboard/ConfirmDialog";
import {
  GROUPS,
  dateFromSeconds,
  deleteBody,
  emptySelection,
  groupState,
  pruneSelection,
  selectionTotals,
  setGroup,
  summarizeResult,
  toggleId,
} from "./cleanupLogic";
import type { GroupKey, Selection } from "./cleanupLogic";

function formatDate(date: Date | null, locale: string): string {
  if (!date) return "";
  return new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function formatIsoDate(value: string, locale: string): string {
  const date = value ? new Date(value) : null;
  return formatDate(date && !Number.isNaN(date.getTime()) ? date : null, locale);
}

function SelectAll({ state, label, onChange }: { state: "none" | "some" | "all"; label: string; onChange: (on: boolean) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = state === "some";
  }, [state]);
  return (
    <label className="check-row cleanup-select-all">
      <input ref={ref} type="checkbox" checked={state === "all"} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

export interface CleanupViewProps {
  items: CleanupItems | null;
  loading: boolean;
  error: string | null;
  selection: Selection;
  busy: boolean;
  result: CleanupDeleteResult | null;
  deleteError: string | null;
  locale: string;
  onToggle: (group: GroupKey, id: string) => void;
  onToggleGroup: (group: GroupKey, on: boolean) => void;
  onRefresh: () => void;
  onAskDelete: () => void;
}

/** The list, the ticks and the result note. All state lives in `CleanupPanel`; this only draws it. */
export function CleanupView(props: CleanupViewProps) {
  const { items, loading, error, selection, busy, result, deleteError, locale } = props;
  const { t } = useTranslation("common");
  const totals = items ? selectionTotals(items, selection) : null;
  const summary = result ? summarizeResult(result) : null;

  const groupTitle: Record<GroupKey, string> = {
    projects: t("settingsTab.cleanup.groupProjects", { defaultValue: "Storyboard projects" }),
    threads: t("settingsTab.cleanup.groupThreads", { defaultValue: "Chat threads" }),
    outputs: t("settingsTab.cleanup.groupOutputs", { defaultValue: "Gallery pictures and clips" }),
  };

  const rows = (group: GroupKey) => {
    if (!items) return [];
    if (group === "projects") {
      return items.projects.map((item) => ({
        id: item.id,
        name: item.name,
        detail: t("settingsTab.cleanup.projectDetail", {
          shots: item.shots,
          date: formatDate(dateFromSeconds(item.lastEdited), locale) || t("settingsTab.cleanup.unknownDate", { defaultValue: "unknown" }),
          defaultValue: "{shots, plural, one {# shot} other {# shots}}, edited {date}",
        }),
        bytes: item.bytes,
      }));
    }
    if (group === "threads") {
      return items.threads.map((item) => ({
        id: item.id,
        name: item.title,
        detail: t("settingsTab.cleanup.threadDetail", {
          count: item.messages,
          date: formatDate(dateFromSeconds(item.lastUsed), locale) || t("settingsTab.cleanup.unknownDate", { defaultValue: "unknown" }),
          defaultValue: "{count, plural, one {# message} other {# messages}}, last used {date}",
        }),
        bytes: item.bytes,
      }));
    }
    return items.outputs.map((item) => ({
      id: item.id,
      name: item.name,
      detail: t("settingsTab.cleanup.outputDetail", {
        kind: item.kind === "image" ? t("settingsTab.cleanup.kindImage", { defaultValue: "Picture" }) : t("settingsTab.cleanup.kindVideo", { defaultValue: "Clip" }),
        date: formatIsoDate(item.createdAt, locale) || t("settingsTab.cleanup.unknownDate", { defaultValue: "unknown" }),
        defaultValue: "{kind}, {date}",
      }),
      bytes: item.bytes,
    }));
  };

  return (
    <div className="control-stack cleanup-panel">
      <p className="help-text">
        {t("settingsTab.cleanup.note", {
          defaultValue: "Nothing is ticked. Tick what you want to delete, then confirm. Deleted items cannot be brought back.",
        })}
      </p>
      <div className="skills-card__actions">
        <button type="button" className="secondary-button" disabled={loading || busy} onClick={props.onRefresh}>
          {t("settingsTab.cleanup.refresh", { defaultValue: "Refresh the list" })}
        </button>
      </div>
      {loading && !items ? <p className="help-text" role="status">{t("settingsTab.cleanup.loading", { defaultValue: "Loading the list..." })}</p> : null}
      {error ? (
        <p className="error-text" role="alert">
          {t("settingsTab.cleanup.loadError", { message: error, defaultValue: "Could not load the list: {message}" })}
        </p>
      ) : null}
      {items
        ? GROUPS.map((group) => {
            const list = rows(group);
            return (
              <fieldset key={group} className="cleanup-group" disabled={busy}>
                <legend>
                  {groupTitle[group]} ({list.length})
                </legend>
                {list.length === 0 ? (
                  <p className="help-text">{t("settingsTab.cleanup.empty", { defaultValue: "Nothing here." })}</p>
                ) : (
                  <>
                    <SelectAll
                      state={groupState(items, selection, group)}
                      label={t("settingsTab.cleanup.selectAll", { group: groupTitle[group], defaultValue: "Select all: {group}" })}
                      onChange={(on) => props.onToggleGroup(group, on)}
                    />
                    <ul className="skills-list cleanup-list">
                      {list.map((row) => (
                        <li key={row.id}>
                          <label className="check-row cleanup-row">
                            <input type="checkbox" checked={selection[group].has(row.id)} onChange={() => props.onToggle(group, row.id)} />
                            <span className="cleanup-row__text">
                              <strong>{row.name}</strong>
                              <span className="help-text">{row.detail}</span>
                            </span>
                            <span className="cleanup-row__size">{formatBytes(row.bytes, locale)}</span>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </fieldset>
            );
          })
        : null}
      {totals ? (
        <div className="skills-card__actions">
          <button
            type="button"
            className="secondary-button danger-button"
            disabled={busy || totals.count === 0}
            aria-describedby="cleanup-selected"
            onClick={props.onAskDelete}
          >
            {busy ? t("settingsTab.cleanup.deleting", { defaultValue: "Deleting..." }) : t("settingsTab.cleanup.delete", { defaultValue: "Delete the ticked items" })}
          </button>
          <span id="cleanup-selected" className="help-text" aria-live="polite">
            {t("settingsTab.cleanup.selected", {
              count: totals.count,
              size: formatBytes(totals.bytes, locale),
              defaultValue: "{count, plural, =0 {Nothing ticked} one {# item ticked, {size}} other {# items ticked, {size}}}",
            })}
          </span>
        </div>
      ) : null}
      {deleteError ? (
        <p className="error-text" role="alert">
          {t("settingsTab.cleanup.deleteError", { message: deleteError, defaultValue: "Could not delete: {message}" })}
        </p>
      ) : null}
      <div role="status" aria-live="polite" className="cleanup-result">
        {summary ? (
          <>
            <p>
              {summary.deleted === 0
                ? t("settingsTab.cleanup.resultNone", { defaultValue: "Nothing was deleted." })
                : t("settingsTab.cleanup.resultDone", {
                    count: summary.deleted,
                    size: formatBytes(summary.freedBytes, locale),
                    defaultValue: "Deleted {count, plural, one {# item} other {# items}} and freed {size}.",
                  })}
            </p>
            {summary.refused.length > 0 ? (
              <>
                <p>
                  {t("settingsTab.cleanup.resultRefused", {
                    count: summary.refused.length,
                    defaultValue: "{count, plural, one {# item was} other {# items were}} not deleted:",
                  })}
                </p>
                <ul className="skills-list">
                  {summary.refused.map((row) => (
                    <li key={`${row.kind}:${row.id}`}>
                      <strong>{row.name ?? row.id}</strong>: {row.reason}
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

export function CleanupPanel({ backendOnline }: { backendOnline: boolean }) {
  const { t, i18n } = useTranslation("common");
  const [items, setItems] = useState<CleanupItems | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(emptySelection);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CleanupDeleteResult | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const locale = i18n.language || "en";

  const load = useCallback(async () => {
    if (!backendOnline) return;
    setLoading(true);
    setError(null);
    try {
      const next = await getCleanupItems();
      setItems(next);
      setSelection((current) => pruneSelection(current, next));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoading(false);
    }
  }, [backendOnline]);

  useEffect(() => {
    void load();
  }, [load]);

  const totals = items ? selectionTotals(items, selection) : null;

  const confirmDelete = async () => {
    if (!items) return;
    const body = deleteBody(items, selection);
    setConfirming(false);
    setBusy(true);
    setDeleteError(null);
    setResult(null);
    try {
      setResult(await deleteCleanupItems(body));
    } catch (exc) {
      setDeleteError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
      await load();
    }
  };

  return (
    <Panel
      title={t("settingsTab.cleanup.title", { defaultValue: "Clean up" })}
      subtitle={t("settingsTab.cleanup.subtitle", { defaultValue: "Delete storyboard projects, chat threads and gallery pictures and clips you no longer want." })}
    >
      <CleanupView
        items={items}
        loading={loading}
        error={error}
        selection={selection}
        busy={busy}
        result={result}
        deleteError={deleteError}
        locale={locale}
        onToggle={(group, id) => {
          setResult(null);
          setSelection((current) => toggleId(current, group, id));
        }}
        onToggleGroup={(group, on) => {
          setResult(null);
          if (items) setSelection((current) => setGroup(current, items, group, on));
        }}
        onRefresh={() => void load()}
        onAskDelete={() => setConfirming(true)}
      />
      {confirming && totals && totals.count > 0 ? (
        <ConfirmDialog
          title={t("settingsTab.cleanup.confirmTitle", {
            count: totals.count,
            defaultValue: "Delete {count, plural, one {# item} other {# items}}?",
          })}
          body={
            <>
              <p>
                {t("settingsTab.cleanup.confirmBody", {
                  count: totals.count,
                  size: formatBytes(totals.bytes, locale),
                  defaultValue: "This deletes {count, plural, one {# item} other {# items}} ({size}) from this computer. It cannot be undone.",
                })}
              </p>
              <p className="help-text">
                {t("settingsTab.cleanup.confirmBreakdown", {
                  projects: totals.byGroup.projects,
                  threads: totals.byGroup.threads,
                  outputs: totals.byGroup.outputs,
                  defaultValue: "Projects: {projects}. Chat threads: {threads}. Gallery files: {outputs}.",
                })}
              </p>
            </>
          }
          confirmLabel={t("settingsTab.cleanup.confirmButton", { defaultValue: "Delete for good" })}
          danger
          busy={busy}
          onConfirm={() => void confirmDelete()}
          onCancel={() => setConfirming(false)}
        />
      ) : null}
    </Panel>
  );
}
