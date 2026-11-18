import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { PauseIcon, PlayIcon } from "../../components/icons/ActionIcons";
import type { LogEntry } from "../../types";
import type { ModelEventsState } from "../../hooks/useModelEvents";
import { EventList } from "./EventList";
import { LogLines } from "./LogLines";
import { ModelTimeline } from "./ModelTimeline";
import {
  EMPTY_FILTER,
  downloadJson,
  eventsToJson,
  exportFilename,
  filterEvents,
  latestFailure,
  modelsOf,
  type EventFilter,
} from "./modelEvents";
import "./serverActivity.css";

export interface ServerActivityProps {
  feed: ModelEventsState;
  logEntries: LogEntry[];
  onCopy: (text: string) => void;
}

/** The Activity view: a timeline per model, the filterable event list and the plain log lines (D-88). */
export function ServerActivity({ feed, logEntries, onCopy }: ServerActivityProps) {
  const { t } = useTranslation("common");
  const [filter, setFilter] = useState<EventFilter>(EMPTY_FILTER);
  const [selected, setSelected] = useState<Set<number>>(() => new Set());
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    if (feed.paused) return;
    setNow(Date.now() / 1000);
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 5000);
    return () => window.clearInterval(timer);
  }, [feed.paused, feed.events]);

  const models = useMemo(() => modelsOf(feed.events), [feed.events]);
  const filtered = useMemo(() => filterEvents(feed.events, filter), [feed.events, filter]);
  const newestFirst = useMemo(() => [...filtered].reverse(), [filtered]);
  const failure = useMemo(() => latestFailure(feed.events), [feed.events]);
  const selectedEvents = useMemo(() => feed.events.filter((event) => selected.has(event.id)), [feed.events, selected]);

  const update = (patch: Partial<EventFilter>) => setFilter((current) => ({ ...current, ...patch }));
  const toggleSelected = (id: number) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const openFailure = () => {
    if (!failure) return;
    setFilter(EMPTY_FILTER);
    setExpandedId(failure.id);
    window.setTimeout(() => {
      document.getElementById(`activity-event-${failure.id}`)?.scrollIntoView({ block: "center" });
    }, 0);
  };
  const allVisibleSelected = newestFirst.length > 0 && newestFirst.every((event) => selected.has(event.id));

  return (
    <div className="server-activity" data-testid="server-activity">
      <ModelTimeline events={feed.events} now={now} onSelectModel={(model) => update({ model })} />

      <section className="activity-list-section">
        <div className="activity-section-head">
          <h3>{t("serverActivity.list.title", { defaultValue: "Events" })}</h3>
          <span className={`activity-live${feed.connected && !feed.paused ? " is-live" : ""}`} role="status">
            {feed.paused
              ? t("serverActivity.paused", { count: feed.pending, defaultValue: "{count, plural, =0 {Paused} one {Paused, # new} other {Paused, # new}}" })
              : feed.connected
                ? t("serverActivity.live", { defaultValue: "Live" })
                : t("serverActivity.offline", { defaultValue: "Reconnecting" })}
          </span>
        </div>

        <div className="activity-toolbar">
          <select
            className="text-input"
            value={filter.model}
            onChange={(event) => update({ model: event.target.value })}
            aria-label={t("serverActivity.filter.model", { defaultValue: "Model" })}
          >
            <option value="">{t("serverActivity.filter.allModels", { defaultValue: "All models" })}</option>
            {models.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
          <select
            className="text-input"
            value={filter.level}
            onChange={(event) => update({ level: event.target.value })}
            aria-label={t("serverActivity.filter.level", { defaultValue: "Level" })}
          >
            <option value="">{t("serverActivity.filter.allLevels", { defaultValue: "All levels" })}</option>
            <option value="info">{t("serverActivity.level.info", { defaultValue: "Info" })}</option>
            <option value="warning">{t("serverActivity.level.warning", { defaultValue: "Warning" })}</option>
            <option value="error">{t("serverActivity.level.error", { defaultValue: "Error" })}</option>
          </select>
          <input
            className="text-input"
            type="search"
            value={filter.request}
            onChange={(event) => update({ request: event.target.value })}
            placeholder={t("serverActivity.filter.request", { defaultValue: "Request id" })}
            aria-label={t("serverActivity.filter.request", { defaultValue: "Request id" })}
          />
          <input
            className="text-input"
            type="search"
            value={filter.query}
            onChange={(event) => update({ query: event.target.value })}
            placeholder={t("serverActivity.filter.search", { defaultValue: "Search events" })}
            aria-label={t("serverActivity.filter.search", { defaultValue: "Search events" })}
          />
        </div>

        <div className="button-row activity-actions">
          <button type="button" className="secondary-button" onClick={feed.paused ? feed.resume : feed.pause}>
            {feed.paused ? <PlayIcon aria-hidden="true" /> : <PauseIcon aria-hidden="true" />}
            {feed.paused
              ? t("serverActivity.resume", { defaultValue: "Resume" })
              : t("serverActivity.pause", { defaultValue: "Pause" })}
          </button>
          <button type="button" className="secondary-button" disabled={!failure} onClick={openFailure}>
            {t("serverActivity.latestFailure", { defaultValue: "Latest failure" })}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={newestFirst.length === 0}
            onClick={() => setSelected(allVisibleSelected ? new Set() : new Set(newestFirst.map((event) => event.id)))}
          >
            {allVisibleSelected
              ? t("serverActivity.selectNone", { defaultValue: "Clear selection" })
              : t("serverActivity.selectAll", { defaultValue: "Select shown" })}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={selectedEvents.length === 0}
            onClick={() => onCopy(eventsToJson(selectedEvents))}
          >
            {t("serverActivity.copySelected", { count: selectedEvents.length, defaultValue: "Copy selected ({count})" })}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={selectedEvents.length === 0}
            onClick={() => downloadJson(eventsToJson(selectedEvents), exportFilename())}
          >
            {t("serverActivity.exportSelected", { count: selectedEvents.length, defaultValue: "Export selected ({count})" })}
          </button>
        </div>

        <EventList
          events={newestFirst}
          selected={selected}
          expandedId={expandedId}
          onToggleSelected={toggleSelected}
          onToggleExpanded={(id) => setExpandedId((current) => (current === id ? null : id))}
          onCopy={onCopy}
        />
      </section>

      <LogLines entries={logEntries} />
    </div>
  );
}
