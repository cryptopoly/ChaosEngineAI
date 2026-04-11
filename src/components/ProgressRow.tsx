interface ProgressRowProps {
  label: string;
  value: number;
  max?: number;
  valueLabel: string;
  baseline?: number;
  baselineLabel?: string;
  delta?: string;
  deltaPositive?: boolean;
}

export function ProgressRow({ label, value, max = 100, valueLabel, baseline, delta, deltaPositive }: ProgressRowProps) {
  const width = Math.max(0, Math.min(100, (value / max) * 100));
  const baselineWidth = baseline != null ? Math.max(0, Math.min(100, (baseline / max) * 100)) : null;

  return (
    <div className="progress-row">
      <div className="progress-meta">
        <span>{label}</span>
        <span className="progress-meta-right">
          <strong>{valueLabel}</strong>
          {delta ? (
            <span className={`delta-badge ${deltaPositive ? "delta-badge--positive" : "delta-badge--negative"}`}>
              {delta}
            </span>
          ) : null}
        </span>
      </div>
      <div className="progress-track" aria-hidden="true">
        {baselineWidth != null ? (
          <div className="progress-fill progress-fill--baseline" style={{ width: `${baselineWidth}%` }} />
        ) : null}
        <div className="progress-fill" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}
