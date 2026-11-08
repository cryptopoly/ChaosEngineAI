import { useEffect } from "react";
import { deleteSessionDocument } from "../api";
import { ChatTab } from "../features/chat/ChatTab";
import { OPEN_TAB_EVENT } from "../features/chat/studioChat";
import type { TabId } from "../types";
import { CompareView } from "../features/chat/CompareView";
import { HtmlChallengeTab } from "../features/chat/HtmlChallengeTab";
import { modelsPageRequest } from "../constants";
import type { AppTabContext } from "./context";

export function ChatView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    chat,
    settings,
    busy,
    busyAction,
    activeThreadOption,
    libraryChatOptions,
    refreshWorkspace,
    setError,
    setActiveTab,
    openModelSelector,
    handleUnloadModel,
    handleLoadModel,
    handleDetailsToggle,
  } = ctx;
  const {
    activeChatId,
    setActiveChatId,
    threadTitleDraft,
    setThreadTitleDraft,
    draftMessage,
    setDraftMessage,
    chatBusySessionId,
    pendingImages,
    setPendingImages,
    chatScrollRef,
    sortedChatSessions,
    activeChat,
    activeThinkingMode,
    activeRuntimeProfileMatchesLaunchSettings,
    handleCreateSession,
    handleRenameActiveThread,
    handleToggleThreadPin,
    handleThinkingModeChange,
    handleDeleteSession,
    sendMessage,
  } = chat;
  const {
    launchSettings,
    installingPackage,
    handleInstallPackage,
  } = settings;
  // D-60: a studio job card's "Open in Library" asks for a tab.
  useEffect(() => {
    const onOpen = (event: Event) => {
      const tab = (event as CustomEvent<TabId>).detail;
      if (tab) setActiveTab(tab);
    };
    window.addEventListener(OPEN_TAB_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_TAB_EVENT, onOpen);
  }, [setActiveTab]);
  return (
    <ChatTab
      sortedChatSessions={sortedChatSessions}
      activeChat={activeChat}
      activeChatId={activeChatId}
      threadTitleDraft={threadTitleDraft}
      draftMessage={draftMessage}
      pendingImages={pendingImages}
      chatBusySessionId={chatBusySessionId}
      busy={busy}
      busyAction={busyAction}
      chatScrollRef={chatScrollRef}
      serverLoading={workspace.server.loading}
      loadedModelRef={workspace.runtime.loadedModel?.ref}
      loadedModelCapabilities={workspace.runtime.loadedModel?.capabilities ?? null}
      loadedModelEngine={workspace.runtime.loadedModel?.engine ?? null}
      engineLabel={workspace.runtime.engineLabel}
      launchSettings={launchSettings}
      warmModels={workspace.runtime.warmModels ?? []}
      activeThreadOptionKey={activeThreadOption?.key}
      onSetActiveChatId={setActiveChatId}
      onThreadTitleDraftChange={setThreadTitleDraft}
      thinkingMode={activeThinkingMode}
      runtimeProfileReady={activeRuntimeProfileMatchesLaunchSettings}
      onThinkingModeChange={handleThinkingModeChange}
      onDraftMessageChange={setDraftMessage}
      onPendingImagesChange={setPendingImages}
      onCreateSession={handleCreateSession}
      onToggleThreadPin={handleToggleThreadPin}
      onDeleteSession={handleDeleteSession}
      onRenameActiveThread={handleRenameActiveThread}
      onOpenModelSelector={openModelSelector}
      onUnloadModel={() => void handleUnloadModel()}
      onLoadModel={handleLoadModel}
      onChatFileDrop={chat.handleChatFileDrop}
      onDeleteSessionDocument={deleteSessionDocument}
      onRefreshWorkspace={async (id?: string) => { await refreshWorkspace(id); }}
      onCopyMessage={chat.handleCopyMessage}
      onRetryMessage={chat.handleRetryMessage}
      onDeleteMessage={chat.handleDeleteMessage}
      onForkAtMessage={chat.handleForkAtMessage}
      onAddVariant={chat.handleAddVariant}
      onDelveMessage={chat.handleDelveMessage}
      onDetailsToggle={handleDetailsToggle}
      onSendMessage={sendMessage}
      onSetError={setError}
      enableTools={chat.enableTools}
      onToggleTools={chat.setEnableTools}
      studioTools={chat.studioTools}
      onToggleStudioTools={chat.setStudioTools}
      onCancelGeneration={chat.cancelGeneration}
      oneTurnOverride={chat.oneTurnOverride}
      onOneTurnOverrideChange={chat.setOneTurnOverride}
      availableCacheStrategies={workspace.system.availableCacheStrategies}
      dflashInfo={workspace.system.dflash}
      loadedModelCanonicalRepo={workspace.runtime.loadedModel?.canonicalRepo ?? null}
      loadedModelName={workspace.runtime.loadedModel?.name ?? null}
      onInstallPackage={handleInstallPackage}
      installingPackage={installingPackage}
      noChatModelsInstalled={libraryChatOptions.length === 0}
      onBrowseDiscover={() => setActiveTab(modelsPageRequest("chat", "available"))}
      onOpenModels={() => setActiveTab(modelsPageRequest("chat", "installed"))}
    />
  );
}

export function CompareChatView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    settings,
    libraryChatOptions,
    setActiveTab,
    handleInstallMtplx,
    installingMtplx,
    mtplxJob,
    tensorfoldControls,
    isAppleSilicon,
  } = ctx;
  const {
    launchSettings,
    handleInstallPackage,
    installingPackage,
    installLogs,
  } = settings;
  return (
    <CompareView
      modelOptions={libraryChatOptions}
      onBack={() => setActiveTab("chat")}
      launchSettings={launchSettings}
      availableMemoryGb={workspace.system.availableMemoryGb}
      totalMemoryGb={workspace.system.totalMemoryGb}
      gpuVramTotalGb={workspace.system.gpuVramTotalGb}
      availableCacheStrategies={workspace.system.availableCacheStrategies}
      dflashInfo={workspace.system.dflash}
      turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
      mtplxSystemInfo={workspace.system.mtplx}
      onInstallMtplx={() => void handleInstallMtplx()}
      installingMtplx={installingMtplx}
      mtplxJob={mtplxJob}
      tensorfold={tensorfoldControls}
      isAppleSilicon={isAppleSilicon}
      onInstallPackage={handleInstallPackage}
      installingPackage={installingPackage}
      installLogs={installLogs}
    />
  );
}

export function HtmlChallengeView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    settings,
    libraryChatOptions,
    handleInstallMtplx,
    installingMtplx,
    mtplxJob,
    tensorfoldControls,
    isAppleSilicon,
    fileRevealLabel,
    handleRevealPath,
    handleOpenFilePath,
  } = ctx;
  const {
    launchSettings,
    handleInstallPackage,
    installingPackage,
    installLogs,
  } = settings;
  return (
    <HtmlChallengeTab
      modelOptions={libraryChatOptions}
      launchSettings={launchSettings}
      availableMemoryGb={workspace.system.availableMemoryGb}
      totalMemoryGb={workspace.system.totalMemoryGb}
      gpuVramTotalGb={workspace.system.gpuVramTotalGb}
      availableCacheStrategies={workspace.system.availableCacheStrategies}
      dflashInfo={workspace.system.dflash}
      turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
      mtplxSystemInfo={workspace.system.mtplx}
      onInstallMtplx={() => void handleInstallMtplx()}
      installingMtplx={installingMtplx}
      mtplxJob={mtplxJob}
      tensorfold={tensorfoldControls}
      isAppleSilicon={isAppleSilicon}
      onInstallPackage={handleInstallPackage}
      installingPackage={installingPackage}
      installLogs={installLogs}
      fileRevealLabel={fileRevealLabel}
      onRevealPath={(path) => void handleRevealPath(path)}
      onOpenFilePath={(path) => void handleOpenFilePath(path)}
    />
  );
}
