import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { RuntimeControls } from "./RuntimeControls";
import { number, sizeLabel } from "../utils";
import type { LaunchPreferences, ModelCapabilities, PreviewMetrics, StrategyInstallLog, SystemStats } from "../types";
import type { ChatModelOption } from "../types/chat";
import type { MtplxJobState } from "../api";
import { candidateKeys } from "./runtimeSupport";

/**
 * Phase 2.11: typed capability badges for the picker. Mirrors the
 * map in ChatHeader so the same flag surfaces with the same label
 * across the loaded-model header and the picker.
 */
const CAPABILITY_BADGES: Array<{
  flag: keyof Omit<ModelCapabilities, "tags">;
  labelKey: string;
  labelDefault: string;
  titleKey: string;
  titleDefault: string;
}> = [
  { flag: "supportsVision", labelKey: "capability.vision.label", labelDefault: "Vision", titleKey: "capability.vision.title", titleDefault: "Model accepts image input" },
  { flag: "supportsTools", labelKey: "capability.tools.label", labelDefault: "Tools", titleKey: "capability.tools.title", titleDefault: "Model supports tool / function calling" },
  { flag: "supportsReasoning", labelKey: "capability.reasoning.label", labelDefault: "Reasoning", titleKey: "capability.reasoning.title", titleDefault: "Model emits a reasoning trace" },
  { flag: "supportsCoding", labelKey: "capability.code.label", labelDefault: "Code", titleKey: "capability.code.title", titleDefault: "Model is tuned for code generation" },
  { flag: "supportsAgents", labelKey: "capability.agents.label", labelDefault: "Agents", titleKey: "capability.agents.title", titleDefault: "Model is tuned for multi-step agentic flows" },
  { flag: "supportsAudio", labelKey: "capability.audio.label", labelDefault: "Audio", titleKey: "capability.audio.title", titleDefault: "Model accepts audio input" },
  { flag: "supportsVideo", labelKey: "capability.video.label", labelDefault: "Video", titleKey: "capability.video.title", titleDefault: "Model accepts video input" },
];

function CapabilityBadges({ capabilities }: { capabilities: ModelCapabilities | null | undefined }) {
  const { t } = useTranslation("common");
  if (!capabilities) return null;
  const active = CAPABILITY_BADGES.filter((entry) => capabilities[entry.flag]);
  if (active.length === 0) return null;
  return (
    <span className="capability-badges" aria-label={t("modelLaunchModal.capabilityBadgesAriaLabel", { defaultValue: "Model capabilities" })}>
      {active.map((entry) => (
        <span
          key={entry.flag}
          className="capability-badge"
          title={t(`modelLaunchModal.${entry.titleKey}`, { defaultValue: entry.titleDefault })}
        >
          {t(`modelLaunchModal.${entry.labelKey}`, { defaultValue: entry.labelDefault })}
        </span>
      ))}
    </span>
  );
}

export interface ModelLaunchModalProps {
  open: boolean;
  title?: string;
  confirmLabel: string;
  selectedKey?: string;
  collapseOnOpen?: boolean;
  search: string;
  options: ChatModelOption[];
  settings: LaunchPreferences;
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  gpuVramTotalGb?: number | null;
  availableCacheStrategies: SystemStats["availableCacheStrategies"] | undefined;
  dflashInfo?: SystemStats["dflash"];
  installingPackage: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  turboInstalled?: boolean;
  mtplxSystemInfo?: SystemStats["mtplx"];
  onInstallMtplx?: () => void;
  installingMtplx?: boolean;
  mtplxJob?: MtplxJobState | null;
  /** FU-056 follow-up: forwarded to ``RuntimeControls`` so the MTPLX
   * block hides on non-Apple-Silicon hosts where MTPLX can't run. */
  isAppleSilicon?: boolean;
  onSelectedKeyChange: (key: string) => void;
  onSearchChange: (value: string) => void;
  onSettingChange: <K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) => void;
  onConfirm: (selectedKey: string) => void;
  onClose: () => void;
  onInstallPackage: (strategyId: string) => void;
}

export function ModelLaunchModal({
  open,
  title,
  confirmLabel,
  selectedKey,
  collapseOnOpen = false,
  search,
  options,
  settings,
  preview,
  availableMemoryGb,
  totalMemoryGb,
  gpuVramTotalGb,
  availableCacheStrategies,
  dflashInfo,
  installingPackage,
  installLogs,
  turboInstalled,
  mtplxSystemInfo,
  onInstallMtplx,
  installingMtplx,
  mtplxJob,
  isAppleSilicon = false,
  onSelectedKeyChange,
  onSearchChange,
  onSettingChange,
  onConfirm,
  onClose,
  onInstallPackage,
}: ModelLaunchModalProps) {
  const { t } = useTranslation("common");
  const resolvedTitle = title ?? t("modelLaunchModal.defaultTitle", { defaultValue: "Select Model" });
  const [showList, setShowList] = useState(true);

  useEffect(() => {
    if (!open) {
      setShowList(true);
      return;
    }
    setShowList(!collapseOnOpen);
  }, [open, collapseOnOpen]);

  if (!open) return null;

  const searchLower = search.toLowerCase();
  const filteredOptions = options.filter(
    (option) =>
      !searchLower
      || option.label.toLowerCase().includes(searchLower)
      || option.detail.toLowerCase().includes(searchLower),
  );
  const selectedOption = options.find((option) => option.key === selectedKey) ?? options[0] ?? null;
  const resolvedSelectedKey = selectedOption?.key ?? "";
  const listVisible = showList || !selectedOption || search.length > 0;

  const mtplxModelSupported = (() => {
    if (!mtplxSystemInfo?.supportedModels?.length) return false;
    const modelKeys = candidateKeys([selectedOption?.canonicalRepo, selectedOption?.modelRef]);
    return mtplxSystemInfo.supportedModels.some((ref) =>
      candidateKeys([ref]).some((k) => modelKeys.includes(k))
    );
  })();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-wide" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>{resolvedTitle}</h3>
        </div>
        <div className="modal-body">
          {selectedOption ? (
            <div className="model-selected-card">
              <div className="model-selected-info">
                <strong>{selectedOption.label}</strong>
                <div className="model-selected-meta">
                  {selectedOption.paramsB ? <span className="badge muted">{number(selectedOption.paramsB)}B</span> : null}
                  <span className="badge muted">{selectedOption.format ?? selectedOption.detail}</span>
                  {selectedOption.quantization ? <span className="badge muted">{selectedOption.quantization}</span> : null}
                  {selectedOption.sizeGb ? <span className="badge muted">{sizeLabel(selectedOption.sizeGb)}</span> : null}
                  {selectedOption.contextWindow ? <span className="badge muted">{selectedOption.contextWindow}</span> : null}
                  <span className={`badge ${selectedOption.source === "library" ? "success" : "accent"}`}>{selectedOption.group}</span>
                </div>
                <CapabilityBadges capabilities={selectedOption.capabilities} />
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  onSearchChange("");
                  setShowList(true);
                }}
              >
                {t("modelLaunchModal.changeButton", { defaultValue: "Change" })}
              </button>
            </div>
          ) : null}

          {listVisible ? (
            <>
              <input
                className="text-input"
                type="search"
                placeholder={t("modelLaunchModal.searchPlaceholder", { defaultValue: "Search models..." })}
                value={search}
                onChange={(event) => onSearchChange(event.target.value)}
                autoFocus
              />
              <div className="model-select-list">
                {filteredOptions.map((option) => (
                  <button
                    key={option.key}
                    className={`model-select-item${option.key === resolvedSelectedKey ? " active" : ""}`}
                    type="button"
                    onClick={() => {
                      onSelectedKeyChange(option.key);
                      onSearchChange("");
                      setShowList(false);
                    }}
                  >
                    <div className="model-select-item-info">
                      <strong>{option.label}</strong>
                      <div className="model-select-item-meta">
                        {option.paramsB ? <span>{number(option.paramsB)}B</span> : null}
                        <span>{option.format ?? option.detail}</span>
                        {option.quantization ? <span>{option.quantization}</span> : null}
                        {option.sizeGb ? <span>{sizeLabel(option.sizeGb)}</span> : null}
                        {option.contextWindow ? <span>{option.contextWindow}</span> : null}
                        {option.maxContext ? (
                          <span>
                            {t("modelLaunchModal.contextDetected", {
                              defaultValue: "{contextLabel} detected",
                              contextLabel: option.maxContext >= 1_000_000
                                ? (option.maxContext / 1_048_576).toFixed(1) + "M"
                                : Math.round(option.maxContext / 1024) + "K",
                            })}
                          </span>
                        ) : null}
                      </div>
                      <CapabilityBadges capabilities={option.capabilities} />
                    </div>
                    <span className={`badge ${option.source === "library" ? "success" : "accent"}`}>{option.group}</span>
                  </button>
                ))}
                {filteredOptions.length === 0 ? (
                  <p className="model-select-empty">
                    {t("modelLaunchModal.emptyState", { defaultValue: "No models match your search." })}
                  </p>
                ) : null}
              </div>
            </>
          ) : null}

          <div className="model-select-settings">
            <span className="eyebrow">{t("modelLaunchModal.launchSettings", { defaultValue: "Launch settings" })}</span>
            <RuntimeControls
              settings={settings}
              onChange={onSettingChange}
              maxContext={selectedOption?.maxContext}
              diskSizeGb={selectedOption?.sizeGb}
              preview={preview}
              availableMemoryGb={availableMemoryGb}
              totalMemoryGb={totalMemoryGb}
              gpuVramTotalGb={gpuVramTotalGb}
              availableCacheStrategies={availableCacheStrategies}
              onInstallPackage={onInstallPackage}
              installingPackage={installingPackage}
              installLogs={installLogs}
              dflashInfo={dflashInfo}
              selectedBackend={selectedOption?.backend}
              selectedModelRef={selectedOption?.modelRef}
              selectedCanonicalRepo={selectedOption?.canonicalRepo}
              selectedModelName={selectedOption?.model}
              turboInstalled={turboInstalled}
              mtplxInfo={mtplxSystemInfo ? { available: mtplxSystemInfo.available, modelSupported: mtplxModelSupported } : undefined}
              onInstallMtplx={onInstallMtplx}
              installingMtplx={installingMtplx}
              mtplxJob={mtplxJob}
              isAppleSilicon={isAppleSilicon}
              compact
            />
          </div>
        </div>
        <div className="modal-footer">
          <button
            className="primary-button"
            type="button"
            disabled={!resolvedSelectedKey}
            onClick={() => onConfirm(resolvedSelectedKey)}
          >
            {confirmLabel}
          </button>
          <button className="secondary-button" type="button" onClick={onClose}>
            {t("modelLaunchModal.cancelButton", { defaultValue: "Cancel" })}
          </button>
        </div>
      </div>
    </div>
  );
}
