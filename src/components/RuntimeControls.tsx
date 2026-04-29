import { useEffect, useState } from "react";
import type { LaunchPreferences, PreviewMetrics } from "../types";
import { SliderField } from "./SliderField";
import { PerformancePreview } from "./PerformancePreview";
import {
  isStrategyCompatible,
  resolveDflashSupport,
  strategyIncompatReason,
} from "./runtimeSupport";

const LAUNCH_PRESETS: Array<{
  id: string;
  label: string;
  hint: string;
  values: Partial<LaunchPreferences>;
}> = [
  { id: "quality", label: "Max Quality", hint: "Coding & reasoning", values: { contextTokens: 32768, fusedAttention: false, fitModelInMemory: true, maxTokens: 8192 } },
  { id: "balanced", label: "Balanced", hint: "General chat", values: { contextTokens: 16384, fusedAttention: false, fitModelInMemory: true, maxTokens: 4096 } },
  { id: "speed", label: "Max Speed", hint: "Fast iteration", values: { contextTokens: 8192, fusedAttention: true, fitModelInMemory: true, maxTokens: 2048 } },
  { id: "memory", label: "Min Memory", hint: "Tight RAM", values: { contextTokens: 4096, fusedAttention: true, fitModelInMemory: false, maxTokens: 2048 } },
];

const STRATEGY_INFO: Record<string, { description: string; install: string; requires: string; installHint?: string; autoInstallPackage?: string }> = {
  native: {
    description: "Full-precision FP16 KV cache. No compression, maximum quality. Works with all backends.",
    install: "",
    requires: "Built-in",
  },
  rotorquant: {
    description: "IsoQuant/PlanarQuant KV cache compression using 4D quaternion or 2D Givens rotations. 3-4 bit quantisation with minimal quality loss.",
    install: "./scripts/build-llama-turbo.sh && ./.venv/bin/python3 -m pip install turboquant",
    requires: "llama-server-turbo (TurboQuant fork) + CUDA/Metal GPU",
    installHint: "Run scripts/build-llama-turbo.sh to build the forked llama-server, then install the Python marker package and restart ChaosEngineAI.",
    autoInstallPackage: "turboquant",
  },
  triattention: {
    description: "Transparent KV cache compression integrated via vLLM. Linux + CUDA only; ChaosEngineAI does not support TriAttention on macOS.",
    install: "./.venv/bin/python3 -m pip install triattention vllm",
    requires: "Linux + CUDA GPU + vLLM backend. Not supported on macOS.",
    installHint: "Use this only in a Linux/CUDA environment used by ChaosEngineAI, then restart the app.",
    autoInstallPackage: "triattention",
  },
  turboquant: {
    description: "TurboQuant KV cache compression (~4.6x at 3-bit). On MLX, uses Hadamard rotation + Lloyd-Max codebooks. On GGUF, uses llama-server-turbo. Works with all model architectures.",
    install: "./.venv/bin/python3 -m pip install turboquant-mlx-full && ./scripts/build-llama-turbo.sh",
    requires: "Apple Silicon + MLX for MLX path; llama-server-turbo for GGUF path.",
    installHint: "For MLX: install turboquant-mlx-full. For GGUF: run scripts/build-llama-turbo.sh. Then restart ChaosEngineAI.",
    autoInstallPackage: "turboquant-mlx",
  },
  chaosengine: {
    description: "PCA-based decorrelation + channel truncation + hybrid quantization. Achieves ~3.7x KV cache compression with minimal quality loss (0.034 avg attention error). Supports 2-8 bit compression tiers.",
    install: "Bundled automatically in desktop builds when vendor/ChaosEngine is present; otherwise ./.venv/bin/python3 -m pip install -e /path/to/ChaosEngine",
    requires: "PyTorch 2.2+ (CUDA recommended, MPS/CPU supported). llama.cpp or vLLM backend.",
    installHint: "Desktop builds can bundle a vendored ChaosEngine checkout automatically. For source/dev installs, clone the repo, pip install it into ChaosEngineAI's backend Python, then restart.",
  },
};

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
  availabilityBadge?: string | null;
  availabilityTone?: string | null;
  availabilityReason?: string | null;
  requiredLlamaBinary?: string | null;
  appliesTo?: string[];
}

interface DFlashInfo {
  available: boolean;
  mlxAvailable: boolean;
  vllmAvailable: boolean;
  ddtreeAvailable?: boolean;
  supportedModels: string[];
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
  onInstallPackage?: (strategyId: string) => void;
  installingPackage?: string | null;
  dflashInfo?: DFlashInfo;
  /** Backend of the selected model (e.g. "mlx", "gguf", "vllm", "auto"). Used for compatibility validation. */
  selectedBackend?: string | null;
  selectedModelRef?: string | null;
  selectedCanonicalRepo?: string | null;
  selectedModelName?: string | null;
  /** Whether llama-server-turbo is installed (for RotorQuant/TurboQuant GGUF support). */
  turboInstalled?: boolean;
  /** Whether an update is available for llama-server-turbo. */
  turboUpdateAvailable?: boolean;
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
  onInstallPackage,
  installingPackage,
  dflashInfo,
  selectedBackend,
  selectedModelRef,
  selectedCanonicalRepo,
  selectedModelName,
  turboInstalled,
  turboUpdateAvailable,
}: RuntimeControlsProps) {
  const effectiveMaxContext = Math.max(2048, maxContext ?? 262144);
  const contextMin = Math.min(2048, Math.max(256, Math.floor(effectiveMaxContext / 4)));
  const clampedContext = Math.max(contextMin, Math.min(settings.contextTokens, effectiveMaxContext));
  const contextStep = effectiveMaxContext >= 131072 ? 2048 : effectiveMaxContext >= 16384 ? 1024 : 256;
  const currentPreset = activePresetId(settings);
  const isGgufBackend = selectedBackend ? (selectedBackend.includes("gguf") || selectedBackend.includes("llama")) : false;
  const dflashInstalled = dflashInfo?.available ?? false;
  const dflashSupport = resolveDflashSupport({
    dflashInfo,
    selectedBackend,
    modelRef: selectedModelRef,
    canonicalRepo: selectedCanonicalRepo,
    modelName: selectedModelName,
  });
  const dflashAvailable = dflashSupport.enabled;
  const dflashUnavailableReason = dflashSupport.reason;
  const ddtreeAvailable = dflashSupport.ddtreeAvailable;
  const canInstallDflashForModel = dflashSupport.modelSupported === true;
  const specActive = settings.speculativeDecoding && dflashAvailable;
  const strategies = (availableCacheStrategies ?? [{id: "native", name: "Native f16", available: true, bitRange: null, defaultBits: null, supportsFp16Layers: false}])
    .filter((s) => !s.appliesTo || s.appliesTo.length === 0 || s.appliesTo.includes("text"));
  const hasSelectedStrategy = strategies.some((strategy) => strategy.id === settings.cacheStrategy);
  const selectedStrategy = strategies.find(s => s.id === settings.cacheStrategy) ?? strategies[0];
  const fp16LayersSupported = Boolean(selectedStrategy?.supportsFp16Layers) && !isGgufBackend;
  const [expandedInfo, setExpandedInfo] = useState<string | null>(null);

  useEffect(() => {
    if (isGgufBackend && settings.fp16Layers !== 0) {
      onChange("fp16Layers", 0);
    }
  }, [isGgufBackend, onChange, settings.fp16Layers]);

  useEffect(() => {
    if (settings.cacheStrategy === "native") return;
    if (hasSelectedStrategy && selectedStrategy?.available && isStrategyCompatible(settings.cacheStrategy, selectedBackend)) return;
    onChange("cacheStrategy", "native");
    if (settings.cacheBits !== 0) onChange("cacheBits", 0);
    if (settings.fp16Layers !== 0) onChange("fp16Layers", 0);
  }, [
    hasSelectedStrategy,
    onChange,
    selectedBackend,
    selectedStrategy?.available,
    settings.cacheBits,
    settings.cacheStrategy,
    settings.fp16Layers,
  ]);

  useEffect(() => {
    if (!settings.speculativeDecoding) return;
    if (!dflashAvailable) {
      onChange("speculativeDecoding", false);
    }
  }, [dflashAvailable, onChange, settings.speculativeDecoding]);

  useEffect(() => {
    if (!ddtreeAvailable && (settings.treeBudget ?? 0) !== 0) {
      onChange("treeBudget", 0);
    }
  }, [ddtreeAvailable, onChange, settings.treeBudget]);

  function applyPreset(presetId: string) {
    const preset = LAUNCH_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    for (const [key, value] of Object.entries(preset.values)) {
      onChange(key as keyof LaunchPreferences, value as any);
    }
    // Set strategy-appropriate cache parameters based on preset intent.
    // The currently selected strategy is preserved — only bits/fp16 are tuned.
    if (selectedStrategy.bitRange != null && selectedStrategy.defaultBits != null) {
      const minBits = selectedStrategy.bitRange[0];
      const maxBits = selectedStrategy.bitRange[selectedStrategy.bitRange.length - 1];
      const midBits = selectedStrategy.defaultBits;
      // Gradual steps: quality → balanced → speed → memory
      const bits =
        presetId === "quality" ? maxBits :
        presetId === "balanced" ? Math.min(maxBits, Math.max(minBits, midBits + 1)) :
        presetId === "speed" ? midBits :
        minBits;
      onChange("cacheBits", bits);
      if (selectedStrategy.supportsFp16Layers && !isGgufBackend) {
        const fp16 =
          presetId === "quality" ? 8 :
          presetId === "balanced" ? 6 :
          presetId === "speed" ? 2 :
          0;
        onChange("fp16Layers", fp16);
      }
    } else {
      // Native strategy: no compression
      onChange("cacheBits", 0);
    }
    if (isGgufBackend || !selectedStrategy.supportsFp16Layers) {
      onChange("fp16Layers", 0);
    }
  }

  function selectStrategy(strategy: CacheStrategyOption) {
    if (!strategy.available || !isStrategyCompatible(strategy.id, selectedBackend)) return;
    onChange("cacheStrategy", strategy.id);
    if (strategy.defaultBits != null) {
      onChange("cacheBits", strategy.defaultBits);
    } else {
      onChange("cacheBits", 0);
    }
    if (strategy.supportsFp16Layers && !isGgufBackend) {
      if (settings.fp16Layers === 0) onChange("fp16Layers", 4);
    } else {
      onChange("fp16Layers", 0);
    }
    // Re-apply the active preset with this strategy's parameters so the
    // preview updates immediately (fixes stale preview after strategy switch).
    const active = currentPreset;
    if (active) {
      // Defer so the strategy change propagates first
      setTimeout(() => applyPreset(active), 0);
    }
  }

  return (
    <>
      <span className="eyebrow">Cache strategy {specActive ? <small style={{ fontWeight: "normal", opacity: 0.6 }}> (locked to Native during speculative decoding)</small> : null}</span>
      <div className="cache-strategy-cards">
        {strategies.map((strategy) => {
          const info = STRATEGY_INFO[strategy.id];
          const isSelected = specActive ? strategy.id === "native" : settings.cacheStrategy === strategy.id;
          const isExpanded = expandedInfo === strategy.id;
          const incompatReason = strategyIncompatReason(strategy.id, selectedBackend);
          const isIncompat = incompatReason != null;
          const needsTurbo = strategy.requiredLlamaBinary === "turbo";
          const turboMissing = needsTurbo && isGgufBackend && turboInstalled === false;
          const isDisabled = !strategy.available || (specActive && strategy.id !== "native") || isIncompat || turboMissing;

          return (
            <div key={strategy.id} className={`cache-strategy-card${isSelected ? " cache-strategy-card--active" : ""}${isDisabled ? " cache-strategy-card--disabled" : ""}`} title={incompatReason ?? (turboMissing ? "Requires llama-server-turbo binary. Run scripts/build-llama-turbo.sh to install." : undefined)}>
              <div className="cache-strategy-card-header">
                <button
                  type="button"
                  className="cache-strategy-card-select"
                  disabled={isDisabled}
                  onClick={() => selectStrategy(strategy)}
                >
                  <span className={`cache-strategy-radio${isSelected ? " cache-strategy-radio--checked" : ""}`} />
                  <span className="cache-strategy-card-name">{strategy.name}</span>
                  <span
                    className={`cache-strategy-badge cache-strategy-badge--${
                      isIncompat ? "warning" : turboMissing ? "warning" : strategy.available ? "ready" : strategy.availabilityTone ?? "install"
                    }`}
                  >
                    {isIncompat ? "N/A" : turboMissing ? "No turbo binary" : strategy.available ? "Ready" : strategy.availabilityBadge ?? "Install"}
                  </span>
                </button>
                {info ? (
                  <button
                    type="button"
                    className="cache-strategy-info-btn"
                    onClick={() => setExpandedInfo(isExpanded ? null : strategy.id)}
                    title="More info"
                  >
                    i
                  </button>
                ) : null}
              </div>
              {isExpanded && info ? (
                <div className="cache-strategy-info-panel">
                  <p>{info.description}</p>
                  {!strategy.available && strategy.availabilityReason ? (
                    <p className="cache-strategy-status-note">{strategy.availabilityReason}</p>
                  ) : null}
                  <div className="cache-strategy-meta">
                    <span className="cache-strategy-meta-label">Requires:</span>
                    <span>{info.requires}</span>
                  </div>
                  {info.install ? (
                    <div className="cache-strategy-install">
                      <span className="cache-strategy-meta-label">Install:</span>
                      <code>{info.install}</code>
                      {info.autoInstallPackage && onInstallPackage && !strategy.available ? (
                        <button
                          type="button"
                          className="cache-strategy-install-btn"
                          disabled={installingPackage != null}
                          onClick={() => onInstallPackage(strategy.id)}
                        >
                          {installingPackage === strategy.id
                            ? strategy.requiredLlamaBinary === "turbo" && isGgufBackend && !turboInstalled
                              ? "Building llama-server-turbo..."
                              : "Installing..."
                            : "Install now"}
                        </button>
                      ) : (
                        <p className="cache-strategy-install-hint">{info.installHint ?? "Run in your terminal, then restart ChaosEngineAI."}</p>
                      )}
                      {strategy.available && strategy.requiredLlamaBinary === "turbo" && isGgufBackend && turboUpdateAvailable && onInstallPackage ? (
                        <button
                          type="button"
                          className="cache-strategy-install-btn"
                          disabled={installingPackage != null}
                          onClick={() => onInstallPackage(strategy.id)}
                        >
                          {installingPackage === strategy.id ? "Updating..." : "Update available"}
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

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
        {fp16LayersSupported ? (
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

      <div className="toggle-column">
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.fitModelInMemory}
            onChange={(event) => onChange("fitModelInMemory", event.target.checked)}
          />
          <span>Fit in memory</span>
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.fusedAttention}
            onChange={(event) => onChange("fusedAttention", event.target.checked)}
          />
          <span>Fused attention</span>
        </label>
        <div className="check-row">
          <label className="check-row" style={{ margin: 0 }} title={dflashUnavailableReason ?? "DFlash speculative decoding: 3-5x faster generation with zero quality loss."}>
            <input
              type="checkbox"
              checked={settings.speculativeDecoding && dflashAvailable}
              disabled={!dflashAvailable}
              onChange={(event) => {
                const enabled = event.target.checked;
                onChange("speculativeDecoding", enabled);
                if (enabled) {
                  onChange("cacheStrategy", "native");
                  onChange("cacheBits", 0);
                  onChange("fp16Layers", 0);
                }
              }}
            />
            <span>DFlash</span>
          </label>
          {!dflashInstalled && !isGgufBackend && canInstallDflashForModel && onInstallPackage ? (
            <button
              type="button"
              className="cache-strategy-install-btn"
              disabled={installingPackage != null}
              onClick={() => onInstallPackage("dflash-mlx")}
            >
              {installingPackage === "dflash-mlx" ? "Installing..." : "Install DFlash"}
            </button>
          ) : null}
          {dflashUnavailableReason ? (
            <span
              className="cache-strategy-badge cache-strategy-badge--warning"
              style={{ marginLeft: 4, fontSize: "0.7em" }}
              title={dflashUnavailableReason}
            >N/A</span>
          ) : null}
          <button
            type="button"
            className="cache-strategy-info-btn"
            onClick={() => setExpandedInfo(expandedInfo === "dflash" ? null : "dflash")}
            title="About DFlash speculative decoding"
          >
            i
          </button>
        </div>
        {expandedInfo === "dflash" ? (
          <div className="cache-strategy-info-panel" style={{ marginTop: 4 }}>
            <p>DFlash uses a small draft model to propose multiple tokens in parallel, then verifies them in a single forward pass. This gives 3-5x faster generation with zero quality loss.</p>
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">Requires:</span>
              <span>Apple Silicon + dflash-mlx, or Linux/CUDA + dflash. Compatible draft model for the target.</span>
            </div>
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">Current model:</span>
              <span>{dflashSupport.matchedModel ? `Supported via ${dflashSupport.matchedModel}` : dflashUnavailableReason ?? "Compatibility not resolved yet."}</span>
            </div>
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">Registered targets:</span>
              <span>{dflashInfo?.supportedModels?.length ?? 0} target models</span>
            </div>
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">DDTree:</span>
              <span>{ddtreeAvailable ? "Available" : "Not available in the current dflash runtime."}</span>
            </div>
            {!dflashInstalled && canInstallDflashForModel ? (
              <div className="cache-strategy-install">
                <span className="cache-strategy-meta-label">Install:</span>
                <code>./.venv/bin/python3 -m pip install "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@f825ffb268e50d531e8b6524413b0847334a14dd"</code>
              </div>
            ) : null}
          </div>
        ) : null}
        {settings.speculativeDecoding && dflashAvailable ? (
          <div className="slider-row" style={{ marginTop: 6 }}>
            <label className="slider-label" title="DDTree: tree-based speculative decoding. 0 = linear DFlash, >0 = explore multiple draft paths in parallel for higher acceptance rates.">
              Tree budget (DDTree)
            </label>
            <input
              type="range"
              min={0}
              max={64}
              step={4}
              value={settings.treeBudget ?? 0}
              disabled={!ddtreeAvailable}
              onChange={(event) => onChange("treeBudget", parseInt(event.target.value, 10))}
            />
            <span className="slider-value">{settings.treeBudget ?? 0}</span>
          </div>
        ) : null}
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
