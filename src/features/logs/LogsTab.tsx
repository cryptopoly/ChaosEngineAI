import { Panel } from "../../components/Panel";

interface LogEntry {
  ts: string;
  source: string;
  level: string;
  message: string;
}

export interface LogsTabProps {
  filteredLogs: LogEntry[];
  logQuery: string;
  onLogQueryChange: (query: string) => void;
}

export function LogsTab({ filteredLogs, logQuery, onLogQueryChange }: LogsTabProps) {
  return (
    <div className="content-grid">
      <Panel
        title="Logs"
        subtitle="Searchable view over the sidecar, runtime, and server channels."
        className="span-2"
        actions={
          <input
            className="text-input"
            type="search"
            placeholder="Filter logs"
            value={logQuery}
            onChange={(event) => onLogQueryChange(event.target.value)}
          />
        }
      >
        <div className="log-list">
          {filteredLogs.length ? (
            filteredLogs.map((entry) => (
              <div className="log-line" key={`${entry.ts}-${entry.source}-${entry.message}`}>
                <span className={`log-level ${entry.level}`}>{entry.level}</span>
                <div>
                  <strong>
                    {entry.ts} / {entry.source}
                  </strong>
                  <p>{entry.message}</p>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state">
              <p>No log lines match the current filter.</p>
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
