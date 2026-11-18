import { useTranslation } from "react-i18next";
import { eventSummary, type ModelEvent } from "./modelEvents";

export interface RecentActivityProps {
  /** Oldest first, as the feed holds them. */
  events: ModelEvent[];
  onOpenActivity: () => void;
}

const SHOWN = 4;

/** The last few model events on the Overview, with a way into the Activity view. */
export function RecentActivity({ events, onOpenActivity }: RecentActivityProps) {
  const { t } = useTranslation("common");
  const recent = events.slice(-SHOWN).reverse();
  return (
    <section className="server-recent-activity" aria-label={t("serverActivity.recent.title", { defaultValue: "Recent activity" })}>
      <div className="server-command-header">
        <span className="eyebrow">{t("serverActivity.recent.title", { defaultValue: "Recent activity" })}</span>
        <button type="button" className="secondary-button" onClick={onOpenActivity}>
          {t("serverActivity.recent.open", { defaultValue: "Open Activity" })}
        </button>
      </div>
      {recent.length ? (
        <ul className="server-recent-list">
          {recent.map((event) => (
            <li key={event.id} className={`level-${event.level}`}>
              <small className="mono-text muted-text">{event.time.slice(11)}</small>
              <span className="server-recent-model">{event.model ?? ""}</span>
              <span>{eventSummary(event, t)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted-text">
          {t("serverActivity.recent.empty", { defaultValue: "Nothing yet. Model loads, answers and unloads show here." })}
        </p>
      )}
    </section>
  );
}
