interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  ticks?: Array<{ value: number; label: string }>;
  formatValue?: (v: number) => string;
  onChange: (value: number) => void;
}

export function SliderField(props: SliderFieldProps) {
  const displayValue = props.formatValue ? props.formatValue(props.value) : String(props.value);
  return (
    <div className="slider-field">
      <div className="slider-field__header">
        <span className="slider-field__label">{props.label}</span>
        <span className="slider-field__value">{displayValue}</span>
      </div>
      <input
        className="slider-input"
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        style={{ ["--slider-pct" as any]: `${((props.value - props.min) / (props.max - props.min)) * 100}%` }}
      />
      {props.ticks ? (
        <div className="slider-ticks">
          {props.ticks.map((tick) => (
            <span
              key={tick.value}
              className={`slider-tick${tick.value === props.value ? " slider-tick--active" : ""}`}
              style={{ left: `${((tick.value - props.min) / (props.max - props.min)) * 100}%` }}
            >
              {tick.label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
