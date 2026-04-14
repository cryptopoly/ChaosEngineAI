import type { ModelLoadingState } from "../types";

export function ModelLoadingProgress({ loading }: { loading: ModelLoadingState }) {
  const rawPct =
    typeof loading.progressPercent === "number" && !Number.isNaN(loading.progressPercent)
      ? loading.progressPercent
      : null;
  const pct = rawPct !== null ? Math.max(0, Math.min(100, rawPct)) : null;
  const phase = loading.progressPhase ?? null;
  const message = loading.progressMessage ?? null;
  const recentLogs = loading.recentLogLines ?? [];
  const hasProgress = pct !== null;
  return (
    <div className="model-loading-progress">
      <div className="loading-progress-bar">
        {hasProgress ? (
          <div
            className="loading-progress-bar-fill"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="loading-progress-bar-fill loading-progress-bar-indeterminate" />
        )}
      </div>
      {hasProgress ? (
        <p className="loading-progress-label">
          {Math.round(pct as number)}%{phase ? ` - ${phase}` : ""}
          {message ? ` - ${message}` : ""}
          {" "}
          <span className="loading-progress-elapsed">({loading.elapsedSeconds}s)</span>
        </p>
      ) : (
        <p className="loading-progress-label">
          Loading {loading.modelName}... {loading.elapsedSeconds}s elapsed
        </p>
      )}
      {recentLogs.length > 0 ? (
        <div className="loading-recent-logs">
          {recentLogs.slice(-5).map((line, idx) => (
            <div key={idx} className="loading-recent-log-line">
              {line}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
