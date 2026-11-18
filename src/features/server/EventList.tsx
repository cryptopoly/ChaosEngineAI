import { useTranslation } from "react-i18next";
import { ChevronDownIcon } from "../../components/icons/ActionIcons";
import { eventSummary, eventsToJson, formatBytes, formatSeconds, type ModelEvent } from "./modelEvents";

export interface EventListProps {
  /** Newest first. */
  events: ModelEvent[];
  selected: ReadonlySet<number>;
  expandedId: number | null;
  onToggleSelected: (id: number) => void;
  onToggleExpanded: (id: number) => void;
  onCopy: (text: string) => void;
}

const BYTE_FIELDS = new Set(["bytes", "memoryBytes"]);
const SECOND_FIELDS = new Set(["seconds", "totalSeconds", "prefillSeconds"]);

function fieldValue(key: string, value: unknown): string {
  if (typeof value === "number" && BYTE_FIELDS.has(key)) return formatBytes(value);
  if (typeof value === "number" && SECOND_FIELDS.has(key)) return formatSeconds(value);
  if (value == null) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function EventDetail({ event, onCopy }: { event: ModelEvent; onCopy: (text: string) => void }) {
  const { t } = useTranslation("common");
  const columns: Array<[string, string | null]> = [
    [t("serverActivity.detail.request", { defaultValue: "Request" }), event.requestId],
    [t("serverActivity.detail.modelRef", { defaultValue: "Model id" }), event.modelRef],
    [t("serverActivity.detail.engine", { defaultValue: "Engine" }), event.engine],
    [t("serverActivity.detail.cache", { defaultValue: "Cache strategy" }), event.cacheStrategy],
    [t("serverActivity.detail.speculative", { defaultValue: "Speculative decoding" }), event.speculative],
    [t("serverActivity.detail.message", { defaultValue: "Message" }), event.message],
  ];
  const data = Object.entries(event.data ?? {}).filter(([, value]) => value !== null && value !== undefined);
  const { command, env, stderr, error } = event.detail ?? {};
  return (
    <div className="activity-detail" data-testid="activity-detail">
      <dl className="activity-detail-grid">
        {columns.filter(([, value]) => value).map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd className="mono-text">{value}</dd>
          </div>
        ))}
        {data.map(([key, value]) => (
          <div key={key}>
            <dt>{t(`serverActivity.field.${key}`, { defaultValue: key })}</dt>
            <dd className="mono-text">{fieldValue(key, value)}</dd>
          </div>
        ))}
      </dl>
      {error ? (
        <div className="activity-detail-block">
          <strong>{t("serverActivity.detail.error", { defaultValue: "Error" })}</strong>
          <pre className="mono-text server-code-block">{error}</pre>
        </div>
      ) : null}
      {command ? (
        <div className="activity-detail-block">
          <div className="server-command-header">
            <strong>{t("serverActivity.detail.command", { defaultValue: "Command line" })}</strong>
            <button type="button" className="secondary-button" onClick={() => onCopy(command)}>
              {t("serverTab.copy", { defaultValue: "Copy" })}
            </button>
          </div>
          <pre className="mono-text server-code-block">{command}</pre>
        </div>
      ) : null}
      {stderr && stderr.length ? (
        <div className="activity-detail-block">
          <strong>{t("serverActivity.detail.stderr", { defaultValue: "Last engine output" })}</strong>
          <pre className="mono-text server-code-block">{stderr.join("\n")}</pre>
        </div>
      ) : null}
      {env && env.length ? (
        <div className="activity-detail-block">
          <strong>{t("serverActivity.detail.env", { defaultValue: "Environment" })}</strong>
          <pre className="mono-text server-code-block">{env.join("\n")}</pre>
        </div>
      ) : null}
      <div className="button-row">
        <button type="button" className="secondary-button" onClick={() => onCopy(eventsToJson([event]))}>
          {t("serverActivity.copyEvent", { defaultValue: "Copy event" })}
        </button>
      </div>
    </div>
  );
}

/** The filtered events, newest first; a row opens to its detail. */
export function EventList({ events, selected, expandedId, onToggleSelected, onToggleExpanded, onCopy }: EventListProps) {
  const { t } = useTranslation("common");
  if (events.length === 0) {
    return (
      <p className="muted-text activity-empty">
        {t("serverActivity.list.empty", { defaultValue: "No events match. Model loads, answers and unloads show here." })}
      </p>
    );
  }
  return (
    <ol className="activity-events" aria-label={t("serverActivity.list.title", { defaultValue: "Events" })}>
      {events.map((event) => {
        const expanded = expandedId === event.id;
        const kindLabel = t(`serverActivity.kind.${event.kind}`, { defaultValue: event.kind });
        return (
          <li
            key={event.id}
            id={`activity-event-${event.id}`}
            className={`activity-event level-${event.level}${expanded ? " is-expanded" : ""}`}
            data-kind={event.kind}
          >
            <div className="activity-event-row">
              <input
                type="checkbox"
                checked={selected.has(event.id)}
                onChange={() => onToggleSelected(event.id)}
                aria-label={t("serverActivity.list.select", { kind: kindLabel, time: event.time, defaultValue: "Select {kind} at {time}" })}
              />
              <button
                type="button"
                className="activity-event-main"
                aria-expanded={expanded}
                onClick={() => onToggleExpanded(event.id)}
              >
                <small className="activity-event-time mono-text">{event.time.slice(11)}</small>
                <span className={`activity-kind kind-${event.kind}`}>{kindLabel}</span>
                <span className="activity-event-model">{event.model ?? ""}</span>
                <span className="activity-event-summary">{eventSummary(event, t)}</span>
                <ChevronDownIcon className={`activity-chevron${expanded ? " is-open" : ""}`} aria-hidden="true" />
              </button>
            </div>
            {expanded ? <EventDetail event={event} onCopy={onCopy} /> : null}
          </li>
        );
      })}
    </ol>
  );
}
