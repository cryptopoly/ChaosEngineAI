import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { LogEntry } from "../../types";

export interface LogLinesProps {
  /** Newest first, as the workspace payload sends them. */
  entries: LogEntry[];
}

/** The plain log lines (sidecar, runtime, server), searchable; what the old Logs page showed. */
export function LogLines({ entries }: LogLinesProps) {
  const { t } = useTranslation("common");
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return entries;
    return entries.filter((entry) => `${entry.ts} ${entry.source} ${entry.level} ${entry.message}`.toLowerCase().includes(needle));
  }, [entries, query]);
  return (
    <section className="activity-log-lines" aria-label={t("serverActivity.lines.title", { defaultValue: "Log lines" })}>
      <div className="activity-section-head">
        <h3>{t("serverActivity.lines.title", { defaultValue: "Log lines" })}</h3>
        <input
          className="text-input"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={t("serverActivity.lines.filter", { defaultValue: "Filter log lines" })}
          aria-label={t("serverActivity.lines.filter", { defaultValue: "Filter log lines" })}
        />
      </div>
      <div className="server-log-scroll activity-lines-scroll">
        {filtered.length ? (
          filtered.map((entry, index) => (
            <div className="server-log-line" key={`${entry.ts}-${entry.source}-${index}`}>
              <small className="server-log-ts">{entry.ts}</small>
              <small className="mono-text muted-text">/{entry.source}</small>
              <span className={`log-level ${entry.level}`}>{entry.level}</span>
              <span>{entry.message}</span>
            </div>
          ))
        ) : (
          <div className="server-log-line">
            <span className="server-log-placeholder">
              {entries.length
                ? t("serverActivity.lines.noMatch", { defaultValue: "No log lines match the filter." })
                : t("serverTab.noLogLines", { defaultValue: "No log lines yet." })}
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
