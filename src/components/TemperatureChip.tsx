import { useEffect, useRef, useState } from "react";

interface TemperatureChipProps {
  /** Default value pulled from launch settings (used when no override is set) */
  defaultValue: number;
  /** Current per-thread override; null/undefined means "use default" */
  override: number | null;
  onChange: (override: number | null) => void;
  disabled?: boolean;
}

const MIN_TEMP = 0;
const MAX_TEMP = 2;
const STEP = 0.05;

export function TemperatureChip({ defaultValue, override, onChange, disabled }: TemperatureChipProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const effective = override ?? defaultValue;
  const isOverridden = override !== null && override !== undefined;

  useEffect(() => {
    if (!open) return;
    const handler = (event: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="temp-chip" ref={wrapRef}>
      <button
        type="button"
        className={`secondary-button temp-chip__trigger${isOverridden ? " temp-chip__trigger--overridden" : ""}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title={isOverridden ? `Temperature override: ${effective.toFixed(2)} (default ${defaultValue.toFixed(2)})` : `Temperature: ${effective.toFixed(2)} (from launch settings)`}
      >
        Temp {effective.toFixed(2)}
        {isOverridden ? <span className="temp-chip__dot" aria-hidden="true" /> : null}
      </button>
      {open ? (
        <div className="temp-chip__popover" role="dialog" aria-label="Temperature override">
          <label className="temp-chip__label">
            <span>Override temperature</span>
            <input
              type="range"
              min={MIN_TEMP}
              max={MAX_TEMP}
              step={STEP}
              value={effective}
              onChange={(event) => onChange(parseFloat(event.target.value))}
            />
          </label>
          <div className="temp-chip__row">
            <input
              type="number"
              className="text-input temp-chip__number"
              min={MIN_TEMP}
              max={MAX_TEMP}
              step={STEP}
              value={effective}
              onChange={(event) => {
                const n = parseFloat(event.target.value);
                if (Number.isFinite(n)) {
                  onChange(Math.min(MAX_TEMP, Math.max(MIN_TEMP, n)));
                }
              }}
            />
            <button
              type="button"
              className="secondary-button temp-chip__reset"
              onClick={() => onChange(null)}
              disabled={!isOverridden}
            >
              Reset
            </button>
          </div>
          <p className="temp-chip__hint">
            Lower = focused. Higher = creative. Default {defaultValue.toFixed(2)} from launch settings.
          </p>
        </div>
      ) : null}
    </div>
  );
}
