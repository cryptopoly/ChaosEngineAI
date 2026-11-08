import { stopIdleModelProcesses, stopModelProcess } from "../api";
import { serverOriginFromBase } from "../utils";
import { DashboardTab } from "../features/dashboard/DashboardTab";
import { ServerTab } from "../features/server/ServerTab";
import { LogsTab } from "../features/logs/LogsTab";
import { SettingsTab } from "../features/settings/SettingsTab";
import { FineTuningTab } from "../features/finetuning/FineTuningTab";
import { PromptLibraryTab } from "../features/prompts/PromptLibraryTab";
import { PluginsTab } from "../features/plugins/PluginsTab";
import type { AppTabContext } from "./context";

export function DashboardView({ ctx }: { ctx: AppTabContext }) {
  const { workspace, backendOnline, setError, refreshWorkspace } = ctx;
  return <DashboardTab
    system={workspace.system}
    recommendation={workspace.recommendation}
    runtime={workspace.runtime}
    activity={workspace.activity}
    backendOnline={backendOnline}
    onStopProcess={async (pid) => {
      try {
        await stopModelProcess(pid);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not stop that process.");
      }
      await refreshWorkspace();
    }}
    onStopIdleProcesses={async () => {
      try {
        await stopIdleModelProcesses();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not stop idle processes.");
      }
      await refreshWorkspace();
    }}
  />;
}

export function ServerView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace, busy, busyAction, backendOnline, apiToken, selectedServerOption,
    showRemoteTest, setShowRemoteTest, testModelId, setTestModelId,
    openModelSelector, handleLoadModel, handleUnloadWarmModel,
  } = ctx;
  const { settingsDraft, setSettingsDraft, handleRestartServer, handleStopServer, handleSaveSettings } = ctx.settings;
  const serverLogEntries = workspace.logs.slice(0, 120).reverse();
  const remoteAccessRequested = settingsDraft.allowRemoteConnections;
  const remoteAccessActive = workspace.server.remoteAccessActive ?? false;
  const serverLanUrls = workspace.server.lanUrls ?? [];
  const localServerUrl = workspace.server.localhostUrl ?? workspace.server.baseUrl;
  const localServerOrigin = serverOriginFromBase(localServerUrl);
  const primaryLanUrl = serverLanUrls[0] ?? null;
  const primaryLanOrigin = primaryLanUrl ? serverOriginFromBase(primaryLanUrl) : null;
  const preferredPortUnavailable = workspace.server.port !== (workspace.settings?.preferredServerPort ?? 8876);
  const authHeaderFlag = apiToken
    ? ` -H 'Authorization: Bearer ${apiToken}'`
    : " -H 'Authorization: Bearer <chaosengine-api-token>'";
  const localHealthCurl = `curl -sS ${localServerOrigin}/api/health`;
  const localModelsCurl = `curl -sS ${localServerUrl}/models${authHeaderFlag}`;
  const remoteHealthCurl = primaryLanOrigin ? `curl -sS ${primaryLanOrigin}/api/health` : null;
  const remoteModelsCurl = primaryLanUrl ? `curl -sS ${primaryLanUrl}/models${authHeaderFlag}` : null;
  return (
    <ServerTab
      serverStatus={workspace.server.status}
      serverPort={workspace.server.port}
      localServerUrl={localServerUrl}
      primaryLanUrl={primaryLanUrl}
      primaryLanOrigin={primaryLanOrigin}
      remoteAccessActive={remoteAccessActive}
      remoteAccessRequested={remoteAccessRequested}
      preferredPortUnavailable={preferredPortUnavailable}
      busyAction={busyAction}
      busy={busy}
      backendOnline={backendOnline}
      warmModels={workspace.runtime.warmModels ?? []}
      serverLoading={workspace.server.loading}
      requestsServed={workspace.server.requestsServed}
      activeConnections={workspace.server.activeConnections}
      engineLabel={workspace.runtime.engineLabel}
      settingsDraft={settingsDraft}
      serverLogEntries={serverLogEntries}
      showRemoteTest={showRemoteTest}
      testModelId={testModelId}
      apiToken={apiToken}
      localHealthCurl={localHealthCurl}
      localModelsCurl={localModelsCurl}
      remoteHealthCurl={remoteHealthCurl}
      remoteModelsCurl={remoteModelsCurl}
      recentOrphanedWorkers={workspace.server.recentOrphanedWorkers ?? []}
      selectedServerOptionKey={selectedServerOption?.key}
      onOpenModelSelector={openModelSelector}
      onRestartServer={handleRestartServer}
      onStopServer={handleStopServer}
      onLoadModel={handleLoadModel}
      onUnloadWarmModel={handleUnloadWarmModel}
      onSaveSettings={handleSaveSettings}
      onSettingsDraftChange={setSettingsDraft}
      onShowRemoteTestChange={setShowRemoteTest}
      onTestModelIdChange={setTestModelId}
    />
  );
}

export function LogsView({ ctx }: { ctx: AppTabContext }) {
  const { workspace, logQuery, setLogQuery } = ctx;
  const filteredLogs = workspace.logs.filter((entry) => {
    const haystack = `${entry.ts} ${entry.source} ${entry.level} ${entry.message}`.toLowerCase();
    return haystack.includes(logQuery.toLowerCase());
  });
  return <LogsTab filteredLogs={filteredLogs} logQuery={logQuery} onLogQueryChange={setLogQuery} />;
}

export function SettingsView({ ctx }: { ctx: AppTabContext }) {
  const { workspace, apiToken, backendOnline, busyAction, sidebarPrefs, uiScalePrefs, copyText } = ctx;
  const {
    settingsDraft, setSettingsDraft,
    newDirectoryLabel, setNewDirectoryLabel,
    newDirectoryPath, setNewDirectoryPath,
    handleAddDirectory,
    handleToggleDirectory,
    handleRemoveDirectory,
    handleUpdateDirectoryPath,
    pickDirectory,
    handlePickDataDirectory,
    handlePickImageOutputsDirectory,
    handlePickVideoOutputsDirectory,
    handleSaveSettings,
    handleRestartServer,
  } = ctx.settings;
  return <SettingsTab
    terms={workspace.settings}
    hostSystem={workspace.system}
    settingsDraft={settingsDraft}
    onSettingsDraftChange={setSettingsDraft}
    newDirectoryLabel={newDirectoryLabel}
    onNewDirectoryLabelChange={setNewDirectoryLabel}
    newDirectoryPath={newDirectoryPath}
    onNewDirectoryPathChange={setNewDirectoryPath}
    onPickDataDirectory={handlePickDataDirectory}
    onPickImageOutputsDirectory={handlePickImageOutputsDirectory}
    onPickVideoOutputsDirectory={handlePickVideoOutputsDirectory}
    onSaveSettings={handleSaveSettings}
    onPickDirectory={pickDirectory}
    onAddDirectory={handleAddDirectory}
    onUpdateDirectoryPath={handleUpdateDirectoryPath}
    onToggleDirectory={handleToggleDirectory}
    onRemoveDirectory={handleRemoveDirectory}
    onCopyText={copyText}
    serverLocalhostUrl={workspace.server.localhostUrl}
    serverPort={workspace.server.port}
    loadedModelName={workspace.runtime.loadedModel?.name}
    apiToken={apiToken}
    sidebarMode={sidebarPrefs.mode}
    onSidebarModeChange={sidebarPrefs.setMode}
    sidebarKeepAllOpen={sidebarPrefs.keepAllOpen}
    onSidebarKeepAllOpenChange={sidebarPrefs.setKeepAllOpen}
    uiScale={uiScalePrefs.uiScale}
    onUiScaleChange={uiScalePrefs.setUiScale}
    backendOnline={backendOnline}
    onRestartServer={() => void handleRestartServer()}
    busyAction={busyAction}
  />;
}

export function FineTuningView({ ctx }: { ctx: AppTabContext }) {
  return <FineTuningTab backendOnline={ctx.backendOnline} />;
}

export function PromptLibraryView({ ctx }: { ctx: AppTabContext }) {
  const { backendOnline, setActiveTab } = ctx;
  const { setSystemPrompt } = ctx.settings;
  return <PromptLibraryTab backendOnline={backendOnline} onApplyTemplate={(prompt) => { setSystemPrompt(prompt); setActiveTab("chat"); }} />;
}

export function PluginsView({ ctx }: { ctx: AppTabContext }) {
  return <PluginsTab backendOnline={ctx.backendOnline} />;
}
