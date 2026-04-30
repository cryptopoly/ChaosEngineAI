import { ModelLaunchModal } from "./ModelLaunchModal";
import type { LaunchPreferences, PreviewMetrics, StrategyInstallLog, SystemStats } from "../types";
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
  dflashInfo?: SystemStats["dflash"];
  installingPackage: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  turboInstalled?: boolean;
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
  dflashInfo,
  installingPackage,
  installLogs,
  turboInstalled,
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
  const selectedLaunchKey = pendingLaunch.preselectedKey ?? localOptions[0]?.key ?? "";
  const setSelectedLaunchKey = (key: string) => onPendingLaunchChange((prev) => prev ? { ...prev, preselectedKey: key } : null);

  return (
    <ModelLaunchModal
      open
      title="Select Model"
      confirmLabel={actionLabel}
      selectedKey={selectedLaunchKey}
      collapseOnOpen={Boolean(pendingLaunch.preselectedKey)}
      search={launchModelSearch}
      options={localOptions}
      settings={launchSettings}
      preview={preview}
      availableMemoryGb={availableMemoryGb}
      totalMemoryGb={totalMemoryGb}
      availableCacheStrategies={availableCacheStrategies}
      dflashInfo={dflashInfo}
      installingPackage={installingPackage}
      installLogs={installLogs}
      turboInstalled={turboInstalled}
      onSelectedKeyChange={setSelectedLaunchKey}
      onSearchChange={onLaunchModelSearchChange}
      onSettingChange={onLaunchSettingChange}
      onConfirm={onConfirmLaunch}
      onClose={() => onPendingLaunchChange(null)}
      onInstallPackage={onInstallPackage}
    />
  );
}
