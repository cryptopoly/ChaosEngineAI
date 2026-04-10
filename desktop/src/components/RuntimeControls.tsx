import type { LaunchPreferences, PreviewMetrics } from "../types";
import { SliderField } from "./SliderField";
import { PerformancePreview } from "./PerformancePreview";

const LAUNCH_PRESETS: Array<{
  id: string;
  label: string;
  hint: string;
  values: Partial<LaunchPreferences>;
}> = [
  { id: "quality", label: "Max Quality", hint: "Coding & reasoning", values: { cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, contextTokens: 8192, fusedAttention: false, fitModelInMemory: true } },
  { id: "balanced", label: "Balanced", hint: "General chat", values: { cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, contextTokens: 8192, fusedAttention: false, fitModelInMemory: true } },
  { id: "speed", label: "Max Speed", hint: "Fast iteration", values: { cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, contextTokens: 4096, fusedAttention: true, fitModelInMemory: true } },
  { id: "memory", label: "Min Memory", hint: "Tight RAM", values: { cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, contextTokens: 2048, fusedAttention: true, fitModelInMemory: true } },
];

function activePresetId(settings: LaunchPreferences): string | null {
  for (const preset of LAUNCH_PRESETS) {
    const match = Object.entries(preset.values).every(
      ([key, value]) => settings[key as keyof LaunchPreferences] === value,
    );
    if (match) return preset.id;
  }
  return null;
}

function contextTicksFor(max: number): Array<{ value: number; label: string }> {
  const fmt = (v: number) =>
    v >= 1_000_000 ? `${(v / 1_048_576).toFixed(0)}M` : v >= 1024 ? `${Math.round(v / 1024)}K` : String(v);
  let raw: number[];
  if (max >= 524288) raw = [8192, 32768, 131072, 524288, max];
  else if (max >= 131072) raw = [2048, 8192, 32768, 131072, max];
  else if (max >= 32768) raw = [2048, 8192, 16384, max];
  else if (max >= 8192) raw = [1024, 2048, 4096, max];
  else if (max >= 2048) raw = [512, 1024, 2048, max];
  else raw = [256, max];
  const seen = new Set<number>();
  const out: Array<{ value: number; label: string }> = [];
  for (const v of raw) {
    if (v > max || seen.has(v)) continue;
    seen.add(v);
    out.push({ value: v, label: fmt(v) });
  }
  return out;
}

interface CacheStrategyOption {
  id: string;
  name: string;
  available: boolean;
  bitRange: number[] | null;
  defaultBits: number | null;
  supportsFp16Layers: boolean;
}

interface RuntimeControlsProps {
  settings: LaunchPreferences;
  onChange: <K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) => void;
  maxContext?: number | null;
  diskSizeGb?: number;
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  compact?: boolean;
  showTemperature?: boolean;
  showPreview?: boolean;
  availableCacheStrategies?: CacheStrategyOption[];
}

export function RuntimeControls({
  settings,
  onChange,
  maxContext,
  diskSizeGb,
  preview,
  availableMemoryGb,
  totalMemoryGb,
  compact,
  showTemperature = true,
  showPreview = true,
  availableCacheStrategies,
}: RuntimeControlsProps) {
  const effectiveMaxContext = Math.max(2048, maxContext ?? 262144);
  const contextMin = Math.min(2048, Math.max(256, Math.floor(effectiveMaxContext / 4)));
  const clampedContext = Math.max(contextMin, Math.min(settings.contextTokens, effectiveMaxContext));
  const contextStep = effectiveMaxContext >= 131072 ? 2048 : effectiveMaxContext >= 16384 ? 1024 : 256;
  const currentPreset = activePresetId(settings);
  const strategies = availableCacheStrategies ?? [{id: "native", name: "Native f16", available: true, bitRange: null, defaultBits: null, supportsFp16Layers: false}];
  const selectedStrategy = strategies.find(s => s.id === settings.cacheStrategy) ?? strategies[0];

  function applyPreset(presetId: string) {
    const preset = LAUNCH_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    for (const [key, value] of Object.entries(preset.values)) {
      onChange(key as keyof LaunchPreferences, value as any);
    }
  }

  return (
    <>
      <div className="preset-row">
        {LAUNCH_PRESETS.map((preset) => (
          <button
            key={preset.id}
            className={`preset-button${currentPreset === preset.id ? " preset-button--active" : ""}`}
            type="button"
            onClick={() => applyPreset(preset.id)}
          >
            <strong>{preset.label}</strong>
            <small>{preset.hint}</small>
          </button>
        ))}
      </div>

      <div className="slider-grid">
        {selectedStrategy?.bitRange != null ? (
          <SliderField
            label="Cache bits"
            value={settings.cacheBits}
            min={selectedStrategy.bitRange[0]} max={selectedStrategy.bitRange[selectedStrategy.bitRange.length - 1]} step={1}
            ticks={selectedStrategy.bitRange.map((v) => ({ value: v, label: String(v) }))}
            formatValue={(v) => `${v}-bit`}
            onChange={(v) => onChange("cacheBits", v)}
          />
        ) : null}
        {selectedStrategy?.supportsFp16Layers ? (
          <SliderField
            label="FP16 layers"
            value={settings.fp16Layers}
            min={0} max={16} step={1}
            ticks={[{ value: 0, label: "0" }, { value: 4, label: "4" }, { value: 8, label: "8" }, { value: 12, label: "12" }, { value: 16, label: "16" }]}
            onChange={(v) => onChange("fp16Layers", v)}
          />
        ) : null}
        <SliderField
          label="Context"
          value={clampedContext}
          min={contextMin} max={effectiveMaxContext} step={contextStep}
          ticks={contextTicksFor(effectiveMaxContext)}
          formatValue={(v) => v >= 1_000_000 ? `${(v / 1_048_576).toFixed(1)}M` : v >= 1024 ? `${(v / 1024).toFixed(0)}K` : String(v)}
          onChange={(v) => onChange("contextTokens", Math.min(v, effectiveMaxContext))}
        />
        <SliderField
          label="Max tokens"
          value={settings.maxTokens}
          min={256} max={32768} step={256}
          ticks={[
            { value: 256, label: "256" },
            { value: 2048, label: "2K" },
            { value: 4096, label: "4K" },
            { value: 8192, label: "8K" },
            { value: 16384, label: "16K" },
            { value: 32768, label: "32K" },
          ]}
          formatValue={(v) => v >= 1024 ? `${(v / 1024).toFixed(0)}K` : String(v)}
          onChange={(v) => onChange("maxTokens", v)}
        />
        {showTemperature ? (
          <SliderField
            label="Temperature"
            value={settings.temperature}
            min={0} max={2} step={0.1}
            ticks={[{ value: 0, label: "0" }, { value: 0.5, label: "0.5" }, { value: 1, label: "1.0" }, { value: 1.5, label: "1.5" }, { value: 2, label: "2.0" }]}
            formatValue={(v) => v.toFixed(1)}
            onChange={(v) => onChange("temperature", v)}
          />
        ) : null}
      </div>

      <div className="field-grid compact-field-grid">
        <label>
          Cache
          <select
            className="text-input"
            value={settings.cacheStrategy}
            onChange={(event) => onChange("cacheStrategy", event.target.value)}
          >
            {(availableCacheStrategies ?? [{id: "native", name: "Native f16", available: true}]).map(s => (
              <option key={s.id} value={s.id} disabled={!s.available}>{s.name}{!s.available ? " (not installed)" : ""}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="toggle-row">
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.fitModelInMemory}
            onChange={(event) => onChange("fitModelInMemory", event.target.checked)}
          />
          Fit in memory
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.fusedAttention}
            onChange={(event) => onChange("fusedAttention", event.target.checked)}
          />
          Fused attention
        </label>
      </div>
      {showPreview ? (
        <PerformancePreview
          preview={preview}
          availableMemoryGb={availableMemoryGb}
          totalMemoryGb={totalMemoryGb}
          actualDiskSizeGb={diskSizeGb}
          compact={compact}
        />
      ) : null}
    </>
  );
}
