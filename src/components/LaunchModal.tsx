import { RuntimeControls } from "./RuntimeControls";
import { number, sizeLabel } from "../utils";
import type { LaunchPreferences, PreviewMetrics, SystemStats } from "../types";
import type { ChatModelOption } from "../types/chat";

export interface PendingLaunch {
  action: "chat" | "server" | "thread";
  preselectedKey?: string;
}

export interface LaunchModalProps {
  pendingLaunch: PendingLaunch | null;
  launchModelSearch: string;
  libraryChatOptions: ChatModelOption[];
  threadModelOptions: ChatModelOption[];
  launchSettings: LaunchPreferences;
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  availableCacheStrategies: SystemStats["availableCacheStrategies"] | undefined;
  installingPackage: string | null;
  onPendingLaunchChange: (value: PendingLaunch | null | ((prev: PendingLaunch | null) => PendingLaunch | null)) => void;
  onLaunchModelSearchChange: (value: string) => void;
  onLaunchSettingChange: <K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) => void;
  onConfirmLaunch: (selectedKey: string) => void;
  onInstallPackage: (strategyId: string) => void;
}

export function LaunchModal({
  pendingLaunch,
  launchModelSearch,
  libraryChatOptions,
  threadModelOptions,
  launchSettings,
  preview,
  availableMemoryGb,
  totalMemoryGb,
  availableCacheStrategies,
  installingPackage,
  onPendingLaunchChange,
  onLaunchModelSearchChange,
  onLaunchSettingChange,
  onConfirmLaunch,
  onInstallPackage,
}: LaunchModalProps) {
  if (!pendingLaunch) return null;
  const actionLabel = pendingLaunch.action === "thread" ? "Start Chat" : pendingLaunch.action === "chat" ? "Load for Chat" : "Load for Server";
  // Prefer local library, but if the caller preselected a catalog key
  // (e.g. the Discover Chat/Server buttons pass `catalog:${variant.id}`)
  // or the library is empty, fall back to the combined set so the
  // preselection actually resolves.
  const preselectedIsCatalog = pendingLaunch.preselectedKey?.startsWith("catalog:") ?? false;
  const localOptions =
    preselectedIsCatalog || libraryChatOptions.length === 0
      ? threadModelOptions
      : libraryChatOptions;
  const searchLower = launchModelSearch.toLowerCase();
  const filteredOptions = localOptions.filter(
    (o) => !searchLower || o.label.toLowerCase().includes(searchLower) || o.detail.toLowerCase().includes(searchLower),
  );
  const selectedLaunchKey = pendingLaunch.preselectedKey ?? localOptions[0]?.key ?? "";
  const setSelectedLaunchKey = (key: string) => onPendingLaunchChange((prev) => prev ? { ...prev, preselectedKey: key } : null);
  const selectedOption = localOptions.find((o) => o.key === selectedLaunchKey);
  const hasPreselection = Boolean(pendingLaunch.preselectedKey);
  const showList = !hasPreselection || launchModelSearch.length > 0;

  return (
    <div className="modal-overlay" onClick={() => onPendingLaunchChange(null)}>
      <div className="modal-content modal-wide" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>Select Model</h3>
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
              <button className="secondary-button" type="button" onClick={() => { onLaunchModelSearchChange(""); onPendingLaunchChange((prev) => prev ? { ...prev, preselectedKey: undefined } : null); }}>
                Change
              </button>
            </div>
          ) : null}

          {showList || !selectedOption ? (
            <>
              <input
                className="text-input"
                type="search"
                placeholder="Search models..."
                value={launchModelSearch}
                onChange={(e) => onLaunchModelSearchChange(e.target.value)}
                autoFocus={!hasPreselection}
              />
              <div className="model-select-list">
                {filteredOptions.map((option) => (
                  <button
                    key={option.key}
                    className={`model-select-item${option.key === selectedLaunchKey ? " active" : ""}`}
                    type="button"
                    onClick={() => { setSelectedLaunchKey(option.key); onLaunchModelSearchChange(""); }}
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
              settings={launchSettings}
              onChange={onLaunchSettingChange}
              maxContext={selectedOption?.maxContext}
              diskSizeGb={selectedOption?.sizeGb}
              preview={preview}
              availableMemoryGb={availableMemoryGb}
              totalMemoryGb={totalMemoryGb}
              availableCacheStrategies={availableCacheStrategies}
              onInstallPackage={onInstallPackage}
              installingPackage={installingPackage}
              compact
            />
          </div>
        </div>
        <div className="modal-footer">
          <button
            className="primary-button"
            type="button"
            disabled={!selectedLaunchKey}
            onClick={() => onConfirmLaunch(selectedLaunchKey)}
          >
            {actionLabel}
          </button>
          <button className="secondary-button" type="button" onClick={() => onPendingLaunchChange(null)}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
