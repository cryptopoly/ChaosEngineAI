import type { BenchmarkResult } from "../types";

export function BenchmarkScatter({ runs, selectedId, compareId, onSelect }: {
  runs: BenchmarkResult[];
  selectedId: string | null;
  compareId: string | null;
  onSelect: (id: string) => void;
}) {
  if (runs.length === 0) {
    return <div className="empty-state"><p>Run some benchmarks to populate the scatter plot.</p></div>;
  }
  const padding = { top: 20, right: 20, bottom: 44, left: 50 };
  const width = 560;
  const height = 320;
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const maxCache = Math.max(1, ...runs.map((r) => r.cacheGb));
  const maxTokS = Math.max(1, ...runs.map((r) => r.tokS));
  const maxQuality = 100;

  const xScale = (v: number) => (v / maxCache) * innerW;
  const yScale = (v: number) => innerH - (v / maxTokS) * innerH;
  const rScale = (q: number) => 4 + (q / maxQuality) * 6;

  const bitColor = (bits: number, strategy: string) => {
    if (strategy === "native" || bits >= 16) return "#8fb4ff";
    if (bits === 1) return "#f87171";
    if (bits === 2) return "#fb923c";
    if (bits === 3) return "#facc15";
    return "#8fcf9f";
  };

  const gridYTicks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div className="benchmark-scatter-wrap">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="benchmark-scatter">
        {/* Grid lines */}
        {gridYTicks.map((t) => (
          <g key={t}>
            <line
              x1={padding.left} x2={padding.left + innerW}
              y1={padding.top + innerH * (1 - t)} y2={padding.top + innerH * (1 - t)}
              stroke="var(--border)" strokeWidth="1" strokeDasharray="2 4"
            />
            <text
              x={padding.left - 8} y={padding.top + innerH * (1 - t) + 4}
              textAnchor="end" className="scatter-axis-label"
            >
              {(maxTokS * t).toFixed(0)}
            </text>
          </g>
        ))}
        {gridYTicks.map((t) => (
          <text
            key={`x-${t}`}
            x={padding.left + innerW * t} y={padding.top + innerH + 18}
            textAnchor="middle" className="scatter-axis-label"
          >
            {(maxCache * t).toFixed(1)}
          </text>
        ))}
        {/* Axis labels */}
        <text x={padding.left + innerW / 2} y={height - 6} textAnchor="middle" className="scatter-axis-title">
          Cache (GB) →
        </text>
        <text
          x={14} y={padding.top + innerH / 2}
          textAnchor="middle" className="scatter-axis-title"
          transform={`rotate(-90 14 ${padding.top + innerH / 2})`}
        >
          ↑ Tokens/sec
        </text>
        {/* Points */}
        {runs.map((run) => {
          const cx = padding.left + xScale(run.cacheGb);
          const cy = padding.top + yScale(run.tokS);
          const r = rScale(run.quality);
          const isSelected = run.id === selectedId;
          const isCompare = run.id === compareId;
          return (
            <g key={run.id}>
              {isSelected ? (
                <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="var(--accent)" strokeWidth="2" />
              ) : null}
              {isCompare ? (
                <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#8fcf9f" strokeWidth="2" strokeDasharray="3 3" />
              ) : null}
              <circle
                cx={cx} cy={cy} r={r}
                fill={bitColor(run.bits, run.cacheStrategy)}
                opacity="0.8"
                onClick={() => onSelect(run.id)}
                style={{ cursor: "pointer" }}
              >
                <title>{`${run.label}\n${run.tokS.toFixed(1)} tok/s · ${run.cacheGb.toFixed(1)} GB · ${run.quality}% quality`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <div className="scatter-legend">
        <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#f87171" }} />1-bit</span>
        <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#fb923c" }} />2-bit</span>
        <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#facc15" }} />3-bit</span>
        <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#8fcf9f" }} />4-bit</span>
        <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#8fb4ff" }} />Native f16</span>
        <span className="scatter-legend-item scatter-legend-spacer">Dot size = quality %</span>
      </div>
    </div>
  );
}
