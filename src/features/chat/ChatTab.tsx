import type { Ref } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import type { ChatSession, ChatThinkingMode, ModelCapabilities, ModelLoadingState, LaunchPreferences, SamplerOverrides, WarmModel } from "../../types";

/**
 * Phase 2.12: imported here so the type appears in the ChatTab module
 * surface and is forwarded as a prop to consumers like the ChatComposer.
 */
type WarmModelType = WarmModel;
import type { ChatModelOption } from "../../types/chat";
import { ChatSidebar } from "./ChatSidebar";
import { ChatHeader } from "./ChatHeader";
import { ChatThread } from "./ChatThread";
import { ChatComposer } from "./ChatComposer";
import { readSamplerOverrides, writeSamplerOverrides } from "./samplerOverrides";
import { matchSlashCommands, type SlashCommand, type SlashCommandContext } from "./slashCommands";

/**
 * Phase 2.1: ChatTab is now a thin composition root that owns the
 * cross-cutting state (sidebar collapse, slash-menu index, per-thread
 * temperature override, per-thread reasoning effort) and threads it
 * through the four extracted subcomponents:
 *   - ChatSidebar   — session list + search + actions
 *   - ChatHeader    — title editor + model picker + export + runtime
 *   - ChatThread    — message list + reasoning + banners + metrics
 *   - ChatComposer  — textarea + slash menu + thinking + temp + send
 *
 * State that any future Phase 2 feature will need (branching, multi-
 * model compare in-thread, @mentions, etc.) lives here; the children
 * receive narrow prop slices.
 */

export interface ChatTabProps {
  sortedChatSessions: ChatSession[];
  activeChat: ChatSession | undefined;
  activeChatId: string;
  threadTitleDraft: string;
  draftMessage: string;
  pendingImages: string[];
  chatBusySessionId: string | null;
  busy: boolean;
  busyAction: string | null;
  chatScrollRef: Ref<HTMLDivElement>;
  serverLoading: ModelLoadingState | null;
  loadedModelRef: string | undefined;
  loadedModelCapabilities?: ModelCapabilities | null;
  engineLabel: string;
  launchSettings: LaunchPreferences;
  warmModels: WarmModel[];
  activeThreadOptionKey: string | undefined;
  thinkingMode: ChatThinkingMode;
  runtimeProfileReady: boolean;
  onSetActiveChatId: (id: string) => void;
  onThreadTitleDraftChange: (title: string) => void;
  onThinkingModeChange: (mode: ChatThinkingMode) => void;
  onDraftMessageChange: (message: string) => void;
  onPendingImagesChange: React.Dispatch<React.SetStateAction<string[]>>;
  onCreateSession: () => void;
  onToggleThreadPin: (session: ChatSession) => void;
  onDeleteSession: (sessionId: string) => void;
  onRenameActiveThread: () => void;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onLoadModel: (payload: {
    modelRef: string;
    modelName?: string;
    canonicalRepo?: string | null;
    source?: string;
    backend?: string;
    path?: string;
    busyLabel?: string;
    cacheStrategy?: string;
    cacheBits?: number;
    fp16Layers?: number;
    fusedAttention?: boolean;
    fitModelInMemory?: boolean;
    contextTokens?: number;
    speculativeDecoding?: boolean;
    treeBudget?: number;
  }) => void;
  onChatFileDrop: (files: FileList) => void;
  onDeleteSessionDocument: (sessionId: string, docId: string) => Promise<void>;
  onRefreshWorkspace: (preferredChatId?: string) => Promise<void>;
  onCopyMessage: (text: string) => void;
  onRetryMessage: (index: number) => void;
  onDeleteMessage: (index: number) => void;
  /** Phase 2.4: fork the thread at this assistant message index. */
  onForkAtMessage: (index: number) => void;
  onDetailsToggle: (opened: boolean) => void;
  onSendMessage: () => void;
  onSetError: (msg: string | null) => void;
  enableTools: boolean;
  onToggleTools: (enabled: boolean) => void;
  onCompareMode: () => void;
  onCancelGeneration: () => void;
  /**
   * Phase 2.12: lifted to the parent so it survives across re-renders
   * and so useChat can read it without prop drilling. The "warm model
   * to send the next turn through" — null means use the session
   * default. Cleared after a successful onDone in useChat.
   */
  oneTurnOverride: WarmModelType | null;
  onOneTurnOverrideChange: (warm: WarmModelType | null) => void;
}

// Avoid an unused-import diagnostic — ChatModelOption is still part of
// the wider chat type vocabulary; keeping the import registered here
// preserves the module's surface for downstream Phase 2 work.
type _ChatModelOptionRef = ChatModelOption;

export function ChatTab({
  sortedChatSessions,
  activeChat,
  threadTitleDraft,
  draftMessage,
  pendingImages,
  chatBusySessionId,
  busy,
  busyAction,
  chatScrollRef,
  serverLoading,
  loadedModelRef,
  loadedModelCapabilities,
  engineLabel,
  launchSettings,
  warmModels,
  activeThreadOptionKey,
  thinkingMode,
  onSetActiveChatId,
  onThreadTitleDraftChange,
  onThinkingModeChange,
  onDraftMessageChange,
  onPendingImagesChange,
  onCreateSession,
  onToggleThreadPin,
  onDeleteSession,
  onRenameActiveThread,
  onOpenModelSelector,
  onLoadModel,
  onChatFileDrop,
  onDeleteSessionDocument,
  onRefreshWorkspace,
  onCopyMessage,
  onRetryMessage,
  onDeleteMessage,
  onForkAtMessage,
  onDetailsToggle,
  onSendMessage,
  onSetError,
  enableTools,
  onToggleTools,
  onCompareMode,
  onCancelGeneration,
  oneTurnOverride,
  onOneTurnOverrideChange,
}: ChatTabProps) {
  const modelBusyLabel =
    busyAction === "Loading model..." || busyAction === "Reloading model for updated launch settings..."
      ? busyAction
      : null;

  // Sidebar collapse — persisted in localStorage so the choice survives
  // navigation between tabs and app restarts.
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem("chat.sidebarCollapsed") === "1";
    } catch {
      return false;
    }
  });
  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem("chat.sidebarCollapsed", next ? "1" : "0");
      } catch {
        // localStorage may be unavailable; collapse still works in-memory
      }
      return next;
    });
  }, []);

  // Sidebar session-search query — local-only, resets on remount.
  const [sessionSearchQuery, setSessionSearchQuery] = useState("");

  const onClearDraft = useCallback(() => {
    onDraftMessageChange("");
    onPendingImagesChange([]);
  }, [onDraftMessageChange, onPendingImagesChange]);

  // Slash-command menu wiring lives at this level so the textarea
  // (inside ChatComposer) and the menu can share the same matches +
  // selection cursor.
  const slashContext = useMemo<SlashCommandContext>(() => ({
    args: "",
    activeChat,
    loadedModelRef,
    enableTools,
    chatBusySessionId,
    onClearDraft,
    onThinkingModeChange,
    onToggleTools,
    onOpenModelSelector,
    onCancelGeneration,
    activeThreadOptionKey,
  }), [
    activeChat,
    loadedModelRef,
    enableTools,
    chatBusySessionId,
    onClearDraft,
    onThinkingModeChange,
    onToggleTools,
    onOpenModelSelector,
    onCancelGeneration,
    activeThreadOptionKey,
  ]);
  const slashMatches = useMemo(
    () => matchSlashCommands(draftMessage, slashContext),
    [draftMessage, slashContext],
  );
  const showSlashMenu = slashMatches.length > 0;
  const [slashIndex, setSlashIndex] = useState(0);
  useEffect(() => {
    setSlashIndex((current) => (current >= slashMatches.length ? 0 : current));
  }, [slashMatches]);
  const runSlashCommand = useCallback((cmd: SlashCommand) => {
    const keepDraft = cmd.run(slashContext);
    if (!keepDraft) {
      onDraftMessageChange("");
    }
  }, [slashContext, onDraftMessageChange]);

  // Per-thread temperature override (Phase 1.10). Persisted in
  // localStorage keyed by session id so the chip survives navigation
  // between threads. useChat reads the same key when assembling the
  // stream payload — see readTemperatureOverride() in useChat.ts.
  const tempOverrideKey = activeChat ? `chat.tempOverride.${activeChat.id}` : null;
  const [temperatureOverride, setTemperatureOverride] = useState<number | null>(() => {
    if (!tempOverrideKey || typeof window === "undefined") return null;
    try {
      const raw = window.localStorage.getItem(tempOverrideKey);
      if (raw == null) return null;
      const parsed = parseFloat(raw);
      return Number.isFinite(parsed) ? parsed : null;
    } catch {
      return null;
    }
  });
  useEffect(() => {
    if (!tempOverrideKey) {
      setTemperatureOverride(null);
      return;
    }
    try {
      const raw = window.localStorage.getItem(tempOverrideKey);
      if (raw == null) { setTemperatureOverride(null); return; }
      const parsed = parseFloat(raw);
      setTemperatureOverride(Number.isFinite(parsed) ? parsed : null);
    } catch {
      setTemperatureOverride(null);
    }
  }, [tempOverrideKey]);
  const handleTemperatureOverrideChange = useCallback((value: number | null) => {
    setTemperatureOverride(value);
    if (!tempOverrideKey) return;
    try {
      if (value == null) {
        window.localStorage.removeItem(tempOverrideKey);
      } else {
        window.localStorage.setItem(tempOverrideKey, String(value));
      }
    } catch {
      // localStorage may be unavailable; override still applies to current render
    }
  }, [tempOverrideKey]);

  // Phase 1.12: reasoning effort level (Off | Low | Med | High). Stored
  // alongside thinkingMode so a session can independently track "Off"
  // vs Low/Medium/High. useChat reads the same key when assembling
  // stream payloads.
  const effortKey = activeChat ? `chat.reasoningEffort.${activeChat.id}` : null;
  type EffortLevel = "low" | "medium" | "high";
  const [reasoningEffort, setReasoningEffort] = useState<EffortLevel>(() => {
    if (!effortKey || typeof window === "undefined") return "medium";
    try {
      const raw = window.localStorage.getItem(effortKey);
      if (raw === "low" || raw === "medium" || raw === "high") return raw;
    } catch {
      // ignore
    }
    return "medium";
  });
  useEffect(() => {
    if (!effortKey) {
      setReasoningEffort("medium");
      return;
    }
    try {
      const raw = window.localStorage.getItem(effortKey);
      if (raw === "low" || raw === "medium" || raw === "high") setReasoningEffort(raw);
      else setReasoningEffort("medium");
    } catch {
      setReasoningEffort("medium");
    }
  }, [effortKey]);
  const handleEffortChange = useCallback((level: EffortLevel) => {
    setReasoningEffort(level);
    if (effortKey) {
      try {
        window.localStorage.setItem(effortKey, level);
      } catch {
        // ignore
      }
    }
    if (thinkingMode !== "auto") {
      onThinkingModeChange("auto");
    }
  }, [effortKey, thinkingMode, onThinkingModeChange]);
  const handleEffortOff = useCallback(() => {
    if (thinkingMode !== "off") {
      onThinkingModeChange("off");
    }
  }, [thinkingMode, onThinkingModeChange]);

  // Phase 2.2: per-thread sampler overrides (top_p, top_k, min_p,
  // repeat_penalty, seed, mirostat). Persisted to localStorage; read
  // back when the thread changes. useChat reads the same key when
  // assembling stream payloads — single source of truth.
  const [samplerOverrides, setSamplerOverridesState] = useState<SamplerOverrides>(() =>
    readSamplerOverrides(activeChat?.id),
  );
  useEffect(() => {
    setSamplerOverridesState(readSamplerOverrides(activeChat?.id));
  }, [activeChat?.id]);
  const handleSamplerOverridesChange = useCallback((overrides: SamplerOverrides) => {
    setSamplerOverridesState(overrides);
    writeSamplerOverrides(activeChat?.id, overrides);
  }, [activeChat?.id]);

  return (
    <div className={`chat-layout-2col${sidebarCollapsed ? " chat-layout-2col--sidebar-collapsed" : ""}`}>
      {!sidebarCollapsed ? (
        <ChatSidebar
          sortedChatSessions={sortedChatSessions}
          activeChat={activeChat}
          warmModels={warmModels}
          searchQuery={sessionSearchQuery}
          onSearchQueryChange={setSessionSearchQuery}
          onSetActiveChatId={onSetActiveChatId}
          onCreateSession={onCreateSession}
          onToggleThreadPin={onToggleThreadPin}
          onDeleteSession={onDeleteSession}
          onCompareMode={onCompareMode}
          onToggleCollapsed={toggleSidebar}
        />
      ) : null}
      <Panel title="Active Thread" subtitle="Response metadata is collapsed by default, but available per agent turn." className="chat-thread">
        <ChatHeader
          activeChat={activeChat}
          threadTitleDraft={threadTitleDraft}
          activeThreadOptionKey={activeThreadOptionKey}
          loadedModelRef={loadedModelRef}
          loadedModelCapabilities={loadedModelCapabilities ?? null}
          serverLoading={serverLoading}
          modelBusyLabel={modelBusyLabel}
          busy={busy}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={toggleSidebar}
          onThreadTitleDraftChange={onThreadTitleDraftChange}
          onRenameActiveThread={onRenameActiveThread}
          onOpenModelSelector={onOpenModelSelector}
          onLoadModel={onLoadModel}
          onDeleteSessionDocument={onDeleteSessionDocument}
          onRefreshWorkspace={onRefreshWorkspace}
          onSetError={onSetError}
        />
        <ChatThread
          activeChat={activeChat}
          chatBusySessionId={chatBusySessionId}
          chatScrollRef={chatScrollRef}
          serverLoading={serverLoading}
          engineLabel={engineLabel}
          launchSettings={launchSettings}
          busy={busy}
          onChatFileDrop={onChatFileDrop}
          onCopyMessage={onCopyMessage}
          onRetryMessage={onRetryMessage}
          onDeleteMessage={onDeleteMessage}
          onForkAtMessage={onForkAtMessage}
          onDetailsToggle={onDetailsToggle}
          onCancelGeneration={onCancelGeneration}
          onLoadModel={onLoadModel}
        />
        <ChatComposer
          draftMessage={draftMessage}
          pendingImages={pendingImages}
          loadedModelRef={loadedModelRef}
          loadedModelCapabilities={loadedModelCapabilities ?? null}
          thinkingMode={thinkingMode}
          reasoningEffort={reasoningEffort}
          enableTools={enableTools}
          chatBusySessionId={chatBusySessionId}
          activeChat={activeChat}
          launchSettings={launchSettings}
          temperatureOverride={temperatureOverride}
          samplerOverrides={samplerOverrides}
          warmModels={warmModels}
          oneTurnOverride={oneTurnOverride}
          onOneTurnOverrideChange={onOneTurnOverrideChange}
          showSlashMenu={showSlashMenu}
          slashMatches={slashMatches}
          slashIndex={slashIndex}
          setSlashIndex={setSlashIndex}
          onDraftMessageChange={onDraftMessageChange}
          onPendingImagesChange={onPendingImagesChange}
          onSendMessage={onSendMessage}
          onCancelGeneration={onCancelGeneration}
          onClearDraft={onClearDraft}
          onChatFileDrop={onChatFileDrop}
          onToggleTools={onToggleTools}
          onSetError={onSetError}
          onTemperatureOverrideChange={handleTemperatureOverrideChange}
          onSamplerOverridesChange={handleSamplerOverridesChange}
          runSlashCommand={runSlashCommand}
          handleEffortOff={handleEffortOff}
          handleEffortChange={handleEffortChange}
        />
      </Panel>
    </div>
  );
}
