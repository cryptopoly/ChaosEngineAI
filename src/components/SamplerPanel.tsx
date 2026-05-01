import { useEffect, useRef, useState } from "react";
import type { SamplerOverrides } from "../types";

/**
 * Phase 2.2: advanced sampler panel for per-thread overrides.
 *
 * Renders behind the "Samplers" composer button. Each control accepts
 * `null` (= use backend default) and returns `null` again on Reset.
 * The panel does NOT own state — it's a controlled component so the
 * parent (ChatTab) can persist to localStorage on every change.
 */
export interface SamplerPanelProps {
  overrides: SamplerOverrides;
  onChange: (overrides: SamplerOverrides) => void;
  disabled?: boolean;
}

interface NumericInputProps {
  label: string;
  hint: string;
  value: number | null | undefined;
  min: number;
  max: number;
  step: number;
  defaultLabel: string;
  onChange: (value: number | null) => void;
  disabled?: boolean;
}

function NumericInput({ label, hint, value, min, max, step, defaultLabel, onChange, disabled }: NumericInputProps) {
  const isOverridden = value != null;
  return (
    <div className="sampler-row">
      <div className="sampler-row__label">
        <strong>{label}</strong>
        <small>{hint}</small>
      </div>
      <div className="sampler-row__input">
        <input
          type="number"
          className="text-input sampler-row__number"
          min={min}
          max={max}
          step={step}
          value={value ?? ""}
          placeholder={defaultLabel}
          disabled={disabled}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") {
              onChange(null);
              return;
            }
            const parsed = parseFloat(raw);
            if (Number.isFinite(parsed)) onChange(parsed);
          }}
        />
        {isOverridden ? (
          <button
            type="button"
            className="sampler-row__reset"
            onClick={() => onChange(null)}
            disabled={disabled}
            title="Use backend default"
          >
            Reset
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function SamplerPanel({ overrides, onChange, disabled }: SamplerPanelProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

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

  const overrideCount = Object.values(overrides).filter((v) => v != null).length;
  const hasOverrides = overrideCount > 0;

  function patch<K extends keyof SamplerOverrides>(key: K, value: SamplerOverrides[K]) {
    const next = { ...overrides };
    if (value == null) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  }

  return (
    <div className="sampler-panel" ref={wrapRef}>
      <button
        type="button"
        className={`secondary-button sampler-panel__trigger${hasOverrides ? " sampler-panel__trigger--overridden" : ""}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title={hasOverrides ? `${overrideCount} sampler override${overrideCount === 1 ? "" : "s"} active` : "Open sampler panel"}
      >
        Samplers
        {hasOverrides ? <span className="sampler-panel__badge" aria-hidden="true">{overrideCount}</span> : null}
      </button>
      {open ? (
        <div className="sampler-panel__popover" role="dialog" aria-label="Sampler overrides">
          <div className="sampler-panel__header">
            <strong>Sampler overrides</strong>
            <button
              type="button"
              className="sampler-panel__clear"
              onClick={() => onChange({})}
              disabled={disabled || !hasOverrides}
              title="Reset all to backend defaults"
            >
              Reset all
            </button>
          </div>
          <NumericInput
            label="top_p"
            hint="Nucleus cutoff (lower = focused)"
            value={overrides.topP}
            min={0}
            max={1}
            step={0.01}
            defaultLabel="default"
            disabled={disabled}
            onChange={(v) => patch("topP", v)}
          />
          <NumericInput
            label="top_k"
            hint="Keep N most-likely tokens (0 = disabled)"
            value={overrides.topK}
            min={0}
            max={200}
            step={1}
            defaultLabel="default"
            disabled={disabled}
            onChange={(v) => patch("topK", v == null ? null : Math.round(v))}
          />
          <NumericInput
            label="min_p"
            hint="Minimum probability cutoff"
            value={overrides.minP}
            min={0}
            max={1}
            step={0.01}
            defaultLabel="default"
            disabled={disabled}
            onChange={(v) => patch("minP", v)}
          />
          <NumericInput
            label="repeat_penalty"
            hint="1.0 = none; >1 = penalise repeats"
            value={overrides.repeatPenalty}
            min={0}
            max={2}
            step={0.01}
            defaultLabel="default"
            disabled={disabled}
            onChange={(v) => patch("repeatPenalty", v)}
          />
          <NumericInput
            label="seed"
            hint="Deterministic decode (any non-negative int)"
            value={overrides.seed}
            min={0}
            max={2 ** 31 - 1}
            step={1}
            defaultLabel="random"
            disabled={disabled}
            onChange={(v) => patch("seed", v == null ? null : Math.round(v))}
          />
          <div className="sampler-row sampler-row--mirostat">
            <div className="sampler-row__label">
              <strong>mirostat</strong>
              <small>Adaptive sampling target entropy</small>
            </div>
            <div className="sampler-row__input">
              <select
                className="text-input sampler-row__select"
                value={overrides.mirostatMode ?? ""}
                disabled={disabled}
                onChange={(event) => {
                  const raw = event.target.value;
                  if (raw === "") {
                    patch("mirostatMode", null);
                    return;
                  }
                  const mode = parseInt(raw, 10);
                  if (mode === 0 || mode === 1 || mode === 2) {
                    patch("mirostatMode", mode);
                  }
                }}
              >
                <option value="">default</option>
                <option value="0">off</option>
                <option value="1">v1</option>
                <option value="2">v2</option>
              </select>
            </div>
          </div>
          {overrides.mirostatMode === 1 || overrides.mirostatMode === 2 ? (
            <>
              <NumericInput
                label="mirostat_tau"
                hint="Target entropy"
                value={overrides.mirostatTau}
                min={0}
                max={10}
                step={0.1}
                defaultLabel="5.0"
                disabled={disabled}
                onChange={(v) => patch("mirostatTau", v)}
              />
              <NumericInput
                label="mirostat_eta"
                hint="Learning rate"
                value={overrides.mirostatEta}
                min={0}
                max={1}
                step={0.01}
                defaultLabel="0.1"
                disabled={disabled}
                onChange={(v) => patch("mirostatEta", v)}
              />
            </>
          ) : null}
          <p className="sampler-panel__hint">
            Per-thread overrides. llama.cpp applies all; mlx-lm uses what it
            supports (top_p / top_k / min_p) and ignores the rest. Empty
            field = use the backend default.
          </p>
        </div>
      ) : null}
    </div>
  );
}
