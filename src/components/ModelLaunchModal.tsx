import { useEffect, useState } from "react";
import { RuntimeControls } from "./RuntimeControls";
import { number, sizeLabel } from "../utils";
import type { LaunchPreferences, PreviewMetrics, SystemStats } from "../types";
import type { ChatModelOption } from "../types/chat";

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
  availableCacheStrategies: SystemStats["availableCacheStrategies"] | undefined;
  dflashInfo?: SystemStats["dflash"];
  installingPackage: string | null;
  onSelectedKeyChange: (key: string) => void;
  onSearchChange: (value: string) => void;
  onSettingChange: <K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) => void;
  onConfirm: (selectedKey: string) => void;
  onClose: () => void;
  onInstallPackage: (strategyId: string) => void;
}

export function ModelLaunchModal({
  open,
  title = "Select Model",
  confirmLabel,
  selectedKey,
  collapseOnOpen = false,
  search,
  options,
  settings,
  preview,
  availableMemoryGb,
  totalMemoryGb,
  availableCacheStrategies,
  dflashInfo,
  installingPackage,
  onSelectedKeyChange,
  onSearchChange,
  onSettingChange,
  onConfirm,
  onClose,
  onInstallPackage,
}: ModelLaunchModalProps) {
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

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-wide" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
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
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  onSearchChange("");
                  setShowList(true);
                }}
              >
                Change
              </button>
            </div>
          ) : null}

          {listVisible ? (
            <>
              <input
                className="text-input"
                type="search"
                placeholder="Search models..."
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
                        {option.maxContext ? <span>{`${option.maxContext >= 1_000_000 ? (option.maxContext / 1_048_576).toFixed(1) + "M" : Math.round(option.maxContext / 1024) + "K"} detected`}</span> : null}
                      </div>
                    </div>
                    <span className={`badge ${option.source === "library" ? "success" : "accent"}`}>{option.group}</span>
                  </button>
                ))}
                {filteredOptions.length === 0 ? <p className="model-select-empty">No models match your search.</p> : null}
              </div>
            </>
          ) : null}

          <div className="model-select-settings">
            <span className="eyebrow">Launch settings</span>
            <RuntimeControls
              settings={settings}
              onChange={onSettingChange}
              maxContext={selectedOption?.maxContext}
              diskSizeGb={selectedOption?.sizeGb}
              preview={preview}
              availableMemoryGb={availableMemoryGb}
              totalMemoryGb={totalMemoryGb}
              availableCacheStrategies={availableCacheStrategies}
              onInstallPackage={onInstallPackage}
              installingPackage={installingPackage}
              dflashInfo={dflashInfo}
              selectedBackend={selectedOption?.backend}
              selectedModelRef={selectedOption?.modelRef}
              selectedCanonicalRepo={selectedOption?.canonicalRepo}
              selectedModelName={selectedOption?.model}
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
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
