export function BenchmarkGauge({ value, max, label, subtitle }: {
  value: number;
  max: number;
  label: string;
  subtitle?: string;
}) {
  const size = 180;
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.min(1, Math.max(0, value / max));
  const arcLength = circumference * 0.75; // 270° arc
  const dashOffset = arcLength * (1 - pct);
  const color = pct > 0.5 ? "#8fcf9f" : pct > 0.25 ? "var(--accent)" : "#e4be75";

  return (
    <div className="benchmark-gauge">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
          {/* Track */}
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
          />
          {/* Fill */}
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke={color} strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.6s ease" }}
          />
        </g>
        <text x={size / 2} y={size / 2 - 4} textAnchor="middle" className="benchmark-gauge-value-text">
          {value.toFixed(1)}
        </text>
        <text x={size / 2} y={size / 2 + 22} textAnchor="middle" className="benchmark-gauge-label-text">
          {label}
        </text>
        {subtitle ? (
          <text x={size / 2} y={size / 2 + 44} textAnchor="middle" className="benchmark-gauge-subtitle-text">
            {subtitle}
          </text>
        ) : null}
      </svg>
    </div>
  );
}
