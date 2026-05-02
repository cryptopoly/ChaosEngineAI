import { useEffect, useRef, useState } from "react";
import type { SystemStats } from "../types";
import type { KvStrategyOverride } from "../features/chat/kvStrategyOverride";

/**
 * Phase 3.2: per-turn KV strategy chip for the composer.
 *
 * Lets the user change cache strategy (TurboQuant / ChaosEngine /
 * Native f16, etc.) and bit width without touching launch settings.
 * The chip shows the *effective* strategy — either the override or
 * the session default — and clicking it opens a popover with the
 * available strategies plus a clear-override action.
 *
 * The backend reloads the runtime transparently when the requested
 * cacheStrategy / cacheBits don't match the currently-loaded profile.
 * Strategies marked `available: false` are still rendered (greyed)
 * with a tooltip explaining the gap so users know the option exists.
 */
export interface KvStrategyChipProps {
  override: KvStrategyOverride | null;
  defaultStrategy: string;
  defaultBits: number;
  availableStrategies: SystemStats["availableCacheStrategies"];
  onChange: (override: KvStrategyOverride | null) => void;
  disabled?: boolean;
}

function formatBits(bits: number): string {
  if (bits <= 0) return "f16";
  return `${bits}-bit`;
}

function formatLabel(strategy: string, bits: number): string {
  return `${strategy} ${formatBits(bits)}`;
}

export function KvStrategyChip({
  override,
  defaultStrategy,
  defaultBits,
  availableStrategies,
  onChange,
  disabled,
}: KvStrategyChipProps) {
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

  const effectiveStrategy = override?.strategy ?? defaultStrategy;
  const effectiveBits = override?.bits ?? defaultBits;
  const isOverridden = override != null;

  // Bit-options come from the strategy's bitRange. When none is set
  // (e.g. native f16), default to a single 0-bits ("f16") option.
  const selectedEntry = availableStrategies?.find((s) => s.id === effectiveStrategy);
  const bitOptions = selectedEntry?.bitRange?.length ? selectedEntry.bitRange : [0];

  return (
    <div className="kv-chip" ref={wrapRef}>
      <button
        type="button"
        className={`secondary-button kv-chip__trigger${isOverridden ? " kv-chip__trigger--active" : ""}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title={
          isOverridden
            ? `KV cache override: ${formatLabel(effectiveStrategy, effectiveBits)} (next turn will reload runtime if needed)`
            : `Default KV cache: ${formatLabel(effectiveStrategy, effectiveBits)} — click to override for next turn`
        }
      >
        <span className="kv-chip__label">KV: {formatLabel(effectiveStrategy, effectiveBits)}</span>
        {isOverridden ? (
          <span
            className="kv-chip__clear"
            role="button"
            tabIndex={0}
            aria-label="Clear KV override"
            title="Revert to session default"
            onClick={(e) => {
              e.stopPropagation();
              onChange(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.stopPropagation();
                onChange(null);
              }
            }}
          >
            ×
          </span>
        ) : null}
      </button>
      {open ? (
        <div className="kv-chip__popover" role="dialog" aria-label="KV cache strategy">
          <div className="kv-chip__heading">
            <strong>KV cache for next turn</strong>
            <small>Switching reloads the runtime if needed.</small>
          </div>
          {(availableStrategies ?? []).map((strategy) => {
            const isActive = strategy.id === effectiveStrategy;
            const range = strategy.bitRange?.length ? strategy.bitRange : [0];
            return (
              <div key={strategy.id} className={`kv-chip__strategy${isActive ? " kv-chip__strategy--active" : ""}`}>
                <div className="kv-chip__strategy-row">
                  <span className="kv-chip__strategy-name">
                    {strategy.name}
                    {!strategy.available ? (
                      <small
                        className="kv-chip__strategy-flag"
                        title={strategy.availabilityReason ?? "Strategy not currently installable"}
                      >
                        unavailable
                      </small>
                    ) : null}
                  </span>
                </div>
                <div className="kv-chip__strategy-bits">
                  {range.map((bits) => {
                    const label = formatBits(bits);
                    const isSelected = isActive && bits === effectiveBits;
                    return (
                      <button
                        key={`${strategy.id}-${bits}`}
                        type="button"
                        className={`kv-chip__bits-button${isSelected ? " kv-chip__bits-button--active" : ""}`}
                        disabled={!strategy.available}
                        onClick={() => {
                          onChange({ strategy: strategy.id, bits });
                          setOpen(false);
                        }}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
          {isOverridden ? (
            <button
              type="button"
              className="kv-chip__reset"
              onClick={() => {
                onChange(null);
                setOpen(false);
              }}
            >
              Clear override (use session default)
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
