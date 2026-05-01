import type { Ref } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { RichMarkdown } from "../../components/RichMarkdown";
import { PromptPhaseIndicator } from "../../components/PromptPhaseIndicator";
import { downloadExport, type ExportFormat } from "./exportThread";
import { filterSessions } from "./sessionSearch";
import { matchSlashCommands, type SlashCommand, type SlashCommandContext } from "./slashCommands";
import { TemperatureChip } from "../../components/TemperatureChip";
import { Panel } from "../../components/Panel";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import { ToolCallCard } from "../../components/ToolCallCard";
import { CitationBadge } from "../../components/CitationBadge";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import type { ChatSession, ChatThinkingMode, ModelLoadingState, LaunchPreferences, WarmModel } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import { number } from "../../utils";
import {
  requestedCacheLabel,
  requestedSpeculativeMode,
  resolvedCacheBits,
  resolvedCacheLabel,
  resolvedCacheStrategy,
  resolvedDraftModel,
  resolvedFp16Layers,
  resolvedSpeculativeMode,
  resolvedTreeBudget,
  runtimeOutcomeWarning,
} from "./runtimeDetails";

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
  onDetailsToggle: (opened: boolean) => void;
  onSendMessage: () => void;
  onSetError: (msg: string | null) => void;
  enableTools: boolean;
  onToggleTools: (enabled: boolean) => void;
  onCompareMode: () => void;
  onCancelGeneration: () => void;
}

export function ChatTab({
  sortedChatSessions,
  activeChat,
  activeChatId,
  threadTitleDraft,
  draftMessage,
  pendingImages,
  chatBusySessionId,
  busy,
  busyAction,
  chatScrollRef,
  serverLoading,
  loadedModelRef,
  engineLabel,
  launchSettings,
  warmModels,
  activeThreadOptionKey,
  thinkingMode,
  runtimeProfileReady,
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
  onDetailsToggle,
  onSendMessage,
  onSetError,
  enableTools,
  onToggleTools,
  onCompareMode,
  onCancelGeneration,
}: ChatTabProps) {
  const modelBusyLabel =
    busyAction === "Loading model..." || busyAction === "Reloading model for updated launch settings..."
      ? busyAction
      : null;

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

  const [sessionSearchQuery, setSessionSearchQuery] = useState("");
  const filteredChatSessions = useMemo(
    () => filterSessions(sortedChatSessions, sessionSearchQuery),
    [sortedChatSessions, sessionSearchQuery],
  );

  const onClearDraft = useCallback(() => {
    onDraftMessageChange("");
    onPendingImagesChange([]);
  }, [onDraftMessageChange, onPendingImagesChange]);

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

  // Per-thread temperature override (Phase 1.10). Persisted in localStorage
  // keyed by session id so the chip survives navigation between threads.
  // useChat reads the same key when assembling the stream payload — see
  // readTemperatureOverride() in useChat.ts.
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

  // Re-read when the active thread changes
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

  // Phase 1.12: reasoning effort levels. Stored alongside thinkingMode but
  // separate so a session can be Off (no thinking) OR Low/Medium/High effort.
  // useChat reads the same localStorage key when assembling stream payloads.
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
    // Selecting any effort level implies thinking is on
    if (thinkingMode !== "auto") {
      onThinkingModeChange("auto");
    }
  }, [effortKey, thinkingMode, onThinkingModeChange]);

  const handleEffortOff = useCallback(() => {
    if (thinkingMode !== "off") {
      onThinkingModeChange("off");
    }
  }, [thinkingMode, onThinkingModeChange]);

  return (
    <div className={`chat-layout-2col${sidebarCollapsed ? " chat-layout-2col--sidebar-collapsed" : ""}`}>
      {!sidebarCollapsed ? (
      <Panel
        title="Chats"
        subtitle=""
        className="chat-column"
        actions={
          <>
            <button className="secondary-button" type="button" onClick={() => void onCreateSession()}>
              New thread
            </button>
            <button className="secondary-button" type="button" onClick={onCompareMode} title="Compare two models side-by-side" style={{ fontSize: 11 }}>
              Compare
            </button>
            <button
              className="secondary-button sidebar-collapse-toggle"
              type="button"
              onClick={toggleSidebar}
              title="Collapse chat list"
              aria-label="Collapse chat list"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="15 18 9 12 15 6" />
              </svg>
            </button>
          </>
        }
      >
        <div className="thread-list-panel">
          <div className="session-search">
            <input
              type="search"
              className="text-input session-search__input"
              placeholder="Search threads..."
              value={sessionSearchQuery}
              onChange={(event) => setSessionSearchQuery(event.target.value)}
              aria-label="Search threads"
            />
            {sessionSearchQuery ? (
              <button
                type="button"
                className="session-search__clear"
                onClick={() => setSessionSearchQuery("")}
                aria-label="Clear search"
                title="Clear search"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            ) : null}
          </div>
          {sessionSearchQuery && filteredChatSessions.length === 0 ? (
            <p className="muted-text" style={{ fontSize: 12, padding: "8px 4px", margin: 0 }}>
              No threads match "{sessionSearchQuery}".
            </p>
          ) : null}
          <div className="session-list">
            {filteredChatSessions.map((session) => (
              <div className="session-row" key={session.id}>
                <button
                  className={session.id === activeChat?.id ? "session-button active" : "session-button"}
                  type="button"
                  onClick={() => onSetActiveChatId(session.id)}
                >
                  <div className="session-title-row">
                    <strong>{session.title}</strong>
                    <span className="session-actions">
                      <span
                        className={`pin-icon${session.pinned ? " pinned" : ""}`}
                        role="button"
                        tabIndex={0}
                        title={session.pinned ? "Unpin" : "Pin"}
                        onClick={(e) => { e.stopPropagation(); void onToggleThreadPin(session); }}
                        onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); void onToggleThreadPin(session); } }}
                      >
                        {"\uD83D\uDCCC"}
                      </span>
                      <span
                        className="session-delete-icon"
                        role="button"
                        tabIndex={0}
                        title="Delete chat"
                        onClick={(e) => { e.stopPropagation(); void onDeleteSession(session.id); }}
                        onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); void onDeleteSession(session.id); } }}
                      >
                        {"\u2715"}
                      </span>
                    </span>
                  </div>
                  <div className="session-meta-row">
                    <small>{session.updatedAt}</small>
                    {session.modelRef && warmModels.some((w) => w.ref === session.modelRef) ? (
                      <span
                        className="badge success session-warm-badge"
                        title="Model is already loaded — this chat will respond instantly with no reload time."
                      >
                        {"\u26A1"} ready
                      </span>
                    ) : null}
                  </div>
                </button>
              </div>
            ))}
          </div>
        </div>
      </Panel>
      ) : null}

      <Panel title="Active Thread" subtitle="Response metadata is collapsed by default, but available per agent turn." className="chat-thread">
        {sidebarCollapsed ? (
          <button
            type="button"
            className="secondary-button sidebar-expand-toggle"
            onClick={toggleSidebar}
            title="Expand chat list"
            aria-label="Expand chat list"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="9 18 15 12 9 6" />
            </svg>
            <span style={{ fontSize: 11 }}>Chats</span>
          </button>
        ) : null}
        <div className="thread-toolbar">
          <label className="thread-title-field">
            Thread name
            <input
              className="text-input"
              type="text"
              value={threadTitleDraft}
              onChange={(event) => onThreadTitleDraftChange(event.target.value)}
              onBlur={() => void onRenameActiveThread()}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void onRenameActiveThread();
                }
              }}
            />
          </label>
          <div className="thread-toolbar-actions">
            <button className="secondary-button" type="button" onClick={() => onOpenModelSelector("chat", activeThreadOptionKey)}>
              {activeChat?.model ?? "Select Model"}
            </button>
            {activeChat && activeChat.messages.length > 0 ? (
              <details className="thread-export-menu">
                <summary
                  className="secondary-button thread-export-menu__summary"
                  title="Export this thread"
                  aria-label="Export this thread"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Export</span>
                </summary>
                <div className="thread-export-menu__content">
                  {(["md", "json", "txt"] as ExportFormat[]).map((fmt) => (
                    <button
                      key={fmt}
                      type="button"
                      className="thread-export-menu__item"
                      onClick={(event) => {
                        event.preventDefault();
                        downloadExport(activeChat, fmt);
                        const details = (event.currentTarget.closest("details")) as HTMLDetailsElement | null;
                        if (details) details.open = false;
                      }}
                    >
                      {fmt === "md" ? "Markdown (.md)" : fmt === "json" ? "JSON (.json)" : "Plain text (.txt)"}
                    </button>
                  ))}
                </div>
              </details>
            ) : null}
            {activeChat?.modelRef === loadedModelRef ? (
              <span className="badge success">Ready</span>
            ) : serverLoading ? (
              <div className="badge accent chat-loading-pill">
                <span className="busy-dot" />
                Loading {serverLoading.modelName}... {serverLoading.elapsedSeconds}s
                {serverLoading.progressPhase ? ` (${serverLoading.progressPhase})` : ""}
              </div>
            ) : modelBusyLabel ? (
              <div className="badge accent chat-loading-pill">
                <span className="busy-dot" />
                {modelBusyLabel}
              </div>
            ) : activeChat?.modelRef ? (
              <button
                className="primary-button action-convert"
                type="button"
                disabled={busy}
                title="Load this chat's model"
                onClick={() => {
                  if (!activeChat?.modelRef) return;
                  void onLoadModel({
                    modelRef: activeChat.modelRef,
                    modelName: activeChat.model,
                    canonicalRepo: activeChat.canonicalRepo,
                    source: activeChat.modelSource ?? "library",
                    backend: activeChat.modelBackend ?? "auto",
                    path: activeChat.modelPath ?? undefined,
                    cacheStrategy: activeChat.cacheStrategy ?? undefined,
                    cacheBits: activeChat.cacheBits ?? undefined,
                    fp16Layers: activeChat.fp16Layers ?? undefined,
                    fusedAttention: activeChat.fusedAttention ?? undefined,
                    fitModelInMemory: activeChat.fitModelInMemory ?? undefined,
                    contextTokens: activeChat.contextTokens ?? undefined,
                    speculativeDecoding: activeChat.speculativeDecoding ?? undefined,
                    treeBudget: activeChat.treeBudget ?? undefined,
                  });
                }}
              >
                {busy ? "Loading..." : "Load model"}
              </button>
            ) : null}
          </div>
          {activeChat?.cacheStrategy ? (
            <div className="thread-runtime-summary">
              <small>
                {activeChat.cacheStrategy}
                {activeChat.cacheBits != null && activeChat.cacheBits > 0
                  ? ` ${activeChat.cacheBits}-bit`
                  : " f16"}
                {activeChat.contextTokens
                  ? ` \u00b7 ${activeChat.contextTokens >= 1024 ? `${Math.round(activeChat.contextTokens / 1024)}K` : activeChat.contextTokens} ctx`
                  : ""}
                {activeChat.speculativeDecoding
                  ? activeChat.treeBudget ? ` \u00b7 DDTree(${activeChat.treeBudget})` : " \u00b7 DFlash"
                  : ""}
                {activeChat.speculativeDecoding && activeChat.dflashDraftModel
                  ? ` (${activeChat.dflashDraftModel.split("/").pop()})`
                  : ""}
              </small>
            </div>
          ) : null}
          {activeChat?.documents && activeChat.documents.length > 0 ? (
            <div className="session-documents">
              {activeChat.documents.map((doc) => (
                <span key={doc.id} className="session-document-chip" title={`${doc.chunkCount} chunks · ${(doc.sizeBytes / 1024).toFixed(0)} KB`}>
                  {"\uD83D\uDCC4"} {doc.originalName}
                  <button
                    type="button"
                    className="session-document-remove"
                    onClick={async () => {
                      if (!activeChat) return;
                      try {
                        await onDeleteSessionDocument(activeChat.id, doc.id);
                        await onRefreshWorkspace(activeChat.id);
                      } catch (err) {
                        onSetError(err instanceof Error ? err.message : "Delete failed");
                      }
                    }}
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
          ) : null}
        </div>

        <div
          className="message-list message-scroll"
          ref={chatScrollRef}
          onDragOver={(event) => {
            event.preventDefault();
            event.currentTarget.classList.add("drag-over");
          }}
          onDragLeave={(event) => {
            event.currentTarget.classList.remove("drag-over");
          }}
          onDrop={(event) => {
            event.preventDefault();
            event.currentTarget.classList.remove("drag-over");
            if (event.dataTransfer?.files) {
              void onChatFileDrop(event.dataTransfer.files);
            }
          }}
        >
          {activeChat?.messages.length ? (
            activeChat.messages.map((message, index) => {
              const isStreamingMessage = chatBusySessionId === activeChat?.id && index === activeChat.messages.length - 1 && !message.metrics;
              const messageSpeculativeMode = message.metrics ? resolvedSpeculativeMode(message.metrics) : null;
              const messageDraftModel = message.metrics ? resolvedDraftModel(message.metrics) : null;
              const messageRequestedCache = message.metrics ? requestedCacheLabel(message.metrics) : null;
              const messageRequestedSpeculativeMode = message.metrics ? requestedSpeculativeMode(message.metrics) : null;
              const messageRuntimeWarning = message.metrics ? runtimeOutcomeWarning(message.metrics) : null;
              const actualFitInMemory = message.metrics?.fitModelInMemory;
              const requestedFitInMemory = message.metrics?.requestedFitModelInMemory;
              const fitInMemoryLabel = actualFitInMemory == null ? "Unknown" : actualFitInMemory ? "On" : "Off";
              const requestedFitInMemoryLabel = requestedFitInMemory == null ? null : requestedFitInMemory ? "On" : "Off";
              return (
              <div className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
                <div className="message-header">
                  <span className="eyebrow">{message.role === "assistant" ? "Agent" : "User"}</span>
                  {!isStreamingMessage ? (
                    <div className="message-actions">
                      <button
                        type="button"
                        className="message-action-btn"
                        title="Copy message"
                        onClick={() => onCopyMessage(message.text)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                        </svg>
                      </button>
                      {message.role === "assistant" ? (
                        <button
                          type="button"
                          className="message-action-btn"
                          title="Retry response"
                          onClick={() => void onRetryMessage(index)}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                          </svg>
                        </button>
                      ) : null}
                      <button
                        type="button"
                        className="message-action-btn message-action-delete"
                        title="Delete message"
                        onClick={() => onDeleteMessage(index)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          <line x1="10" y1="11" x2="10" y2="17" />
                          <line x1="14" y1="11" x2="14" y2="17" />
                        </svg>
                      </button>
                    </div>
                  ) : null}
                </div>
                {message.role === "assistant" ? (
                  <ReasoningPanel
                    text={message.reasoning}
                    streaming={isStreamingMessage && message.reasoningDone !== true}
                  />
                ) : null}
                {message.role === "assistant" && isStreamingMessage && message.streamPhase ? (
                  <PromptPhaseIndicator phase={message.streamPhase} />
                ) : null}
                {message.role === "assistant" ? (
                  <div className={`markdown-content${isStreamingMessage && !message.streamPhase ? " streaming-cursor" : ""}`}>
                    <RichMarkdown>{message.text || "\u200B"}</RichMarkdown>
                  </div>
                ) : (
                  <p>{message.text}</p>
                )}
                {message.toolCalls?.length ? (
                  <div style={{ margin: "4px 0" }}>
                    {message.toolCalls.map((tc) => (
                      <ToolCallCard key={tc.id} toolCall={tc} />
                    ))}
                  </div>
                ) : null}
                {message.citations?.length ? (
                  <CitationBadge citations={message.citations} />
                ) : null}
                {message.metrics ? (
                  <details className="message-details" onToggle={(event) => void onDetailsToggle(event.currentTarget.open)}>
                    <summary>
                      <span>Model details</span>
                      <small className="message-meta">
                        {(message.metrics.model ?? activeChat.model) || "Unknown"} | {number(message.metrics.tokS)} tok/s
                        {message.metrics.dflashAcceptanceRate != null ? ` | DFLASH ${number(message.metrics.dflashAcceptanceRate)} avg accepted` : ""}
                        {messageSpeculativeMode && messageSpeculativeMode !== "Off" ? ` | ${messageSpeculativeMode}` : ""}
                        {messageRuntimeWarning ? ` | ${messageRuntimeWarning}` : ""}
                        {" | "}{number(message.metrics.responseSeconds ?? 0)} s
                      </small>
                    </summary>
                    <div className="message-detail-grid">
                      <div>
                        <span className="eyebrow">Model</span>
                        <p>{message.metrics.model ?? activeChat.model}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Runtime</span>
                        <p>{message.metrics.engineLabel ?? engineLabel}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Cache</span>
                        <p>{resolvedCacheLabel(message.metrics)}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Strategy</span>
                        <p>{resolvedCacheStrategy(message.metrics)}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Cache bits</span>
                        <p>{resolvedCacheBits(message.metrics)}</p>
                      </div>
                      <div>
                        <span className="eyebrow">FP16 layers</span>
                        <p>{resolvedFp16Layers(message.metrics)}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Backend</span>
                        <p>{message.metrics.backend ?? activeChat.modelBackend ?? "Auto"}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Context</span>
                        <p>{message.metrics.contextTokens?.toLocaleString() ?? launchSettings.contextTokens.toLocaleString()}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Fit in memory</span>
                        <p>{fitInMemoryLabel}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Tokens</span>
                        <p>{message.metrics.totalTokens} total</p>
                      </div>
                      <div>
                        <span className="eyebrow">Response time</span>
                        <p>{number(message.metrics.responseSeconds ?? 0)} s</p>
                      </div>
                      <div>
                        <span className="eyebrow">Decode speed</span>
                        <p>{number(message.metrics.tokS)} tok/s</p>
                      </div>
                      <div>
                        <span className="eyebrow">DFlash / DDTree</span>
                        <p>{messageSpeculativeMode}</p>
                      </div>
                      {messageRequestedCache && messageRequestedCache !== resolvedCacheLabel(message.metrics) ? (
                        <div>
                          <span className="eyebrow">Requested cache</span>
                          <p>{messageRequestedCache}</p>
                        </div>
                      ) : null}
                      {requestedFitInMemoryLabel && requestedFitInMemory !== actualFitInMemory ? (
                        <div>
                          <span className="eyebrow">Requested fit</span>
                          <p>{requestedFitInMemoryLabel}</p>
                        </div>
                      ) : null}
                      {messageRequestedSpeculativeMode && messageRequestedSpeculativeMode !== "Off" ? (
                        <div>
                          <span className="eyebrow">Requested DFlash / DDTree</span>
                          <p>{messageRequestedSpeculativeMode}</p>
                        </div>
                      ) : null}
                      {messageRuntimeWarning ? (
                        <div>
                          <span className="eyebrow">Runtime status</span>
                          <p>{messageRuntimeWarning}</p>
                        </div>
                      ) : null}
                      <div>
                        <span className="eyebrow">Tree budget</span>
                        <p>{resolvedTreeBudget(message.metrics)}</p>
                      </div>
                      {message.metrics.dflashAcceptanceRate != null ? (
                        <div>
                          <span className="eyebrow">DFLASH acceptance</span>
                          <p>{number(message.metrics.dflashAcceptanceRate)} avg tokens</p>
                        </div>
                      ) : null}
                      {messageDraftModel ? (
                        <div>
                          <span className="eyebrow">Draft model</span>
                          <p>{messageDraftModel}</p>
                        </div>
                      ) : null}
                    </div>
                    <button
                      className="secondary-button message-reload-settings"
                      type="button"
                      disabled={busy}
                      title="Load the exact model and runtime settings used for this response"
                      onClick={() => {
                        const ref = message.metrics!.modelRef ?? activeChat?.modelRef;
                        if (!ref) return;
                        void onLoadModel({
                          modelRef: ref,
                          modelName: message.metrics!.model ?? activeChat?.model,
                          canonicalRepo: message.metrics!.canonicalRepo ?? activeChat?.canonicalRepo ?? null,
                          source: message.metrics!.modelSource ?? activeChat?.modelSource ?? "library",
                          backend: message.metrics!.backend ?? activeChat?.modelBackend ?? "auto",
                          path: message.metrics!.modelPath ?? activeChat?.modelPath ?? undefined,
                          cacheStrategy: message.metrics!.cacheStrategy ?? activeChat?.cacheStrategy ?? undefined,
                          cacheBits: message.metrics!.cacheBits ?? activeChat?.cacheBits ?? undefined,
                          fp16Layers: message.metrics!.fp16Layers ?? activeChat?.fp16Layers ?? undefined,
                          fusedAttention: message.metrics!.fusedAttention ?? activeChat?.fusedAttention ?? undefined,
                          fitModelInMemory: message.metrics!.fitModelInMemory ?? activeChat?.fitModelInMemory ?? undefined,
                          contextTokens: message.metrics!.contextTokens ?? activeChat?.contextTokens ?? undefined,
                          speculativeDecoding: message.metrics!.speculativeDecoding ?? activeChat?.speculativeDecoding ?? undefined,
                          treeBudget: message.metrics!.treeBudget ?? activeChat?.treeBudget ?? undefined,
                        });
                      }}
                    >
                      Reload these settings
                    </button>
                  </details>
                ) : null}
              </div>
              );
            })
          ) : (
            <div className="empty-state">
              <p>Send a message to start the conversation.</p>
            </div>
          )}
          {serverLoading ? (
            <div className="message-bubble assistant">
              <span className="eyebrow">Agent</span>
              <div className="model-loading-chat">
                <ModelLoadingProgress loading={serverLoading} />
              </div>
            </div>
          ) : null}
        </div>
        <div className="composer">
          {pendingImages.length > 0 ? (
            <div className="composer-image-previews">
              {pendingImages.map((img, i) => (
                <div key={i} className="composer-image-thumb">
                  <img src={`data:image/png;base64,${img}`} alt={`Attachment ${i + 1}`} />
                  <button
                    className="composer-image-remove"
                    type="button"
                    onClick={() => onPendingImagesChange((prev) => prev.filter((_, j) => j !== i))}
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <div className="composer-input-wrap">
          {showSlashMenu ? (
            <div className="slash-command-menu" role="listbox" aria-label="Slash commands">
              {slashMatches.map((cmd, idx) => (
                <button
                  key={cmd.command}
                  type="button"
                  role="option"
                  aria-selected={idx === slashIndex}
                  className={`slash-command-menu__item${idx === slashIndex ? " slash-command-menu__item--active" : ""}`}
                  onMouseEnter={() => setSlashIndex(idx)}
                  onClick={() => runSlashCommand(cmd)}
                >
                  <span className="slash-command-menu__command">{cmd.command}</span>
                  <span className="slash-command-menu__desc">{cmd.description}</span>
                </button>
              ))}
            </div>
          ) : null}
          <textarea
            className="text-area"
            placeholder={
              loadedModelRef
                ? "Type a message... (Enter to send, Shift+Enter for new line, / for commands)"
                : "Load a model first — pick one from My Models or Discover, then hit CHAT."
            }
            rows={3}
            value={draftMessage}
            onChange={(event) => onDraftMessageChange(event.target.value)}
            onKeyDown={(event) => {
              if (showSlashMenu) {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setSlashIndex((current) => (current + 1) % slashMatches.length);
                  return;
                }
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  setSlashIndex((current) => (current - 1 + slashMatches.length) % slashMatches.length);
                  return;
                }
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  const target = slashMatches[slashIndex];
                  if (target) runSlashCommand(target);
                  return;
                }
                if (event.key === "Escape") {
                  event.preventDefault();
                  onDraftMessageChange("");
                  return;
                }
                if (event.key === "Tab") {
                  event.preventDefault();
                  const target = slashMatches[slashIndex];
                  if (target) onDraftMessageChange(`${target.command} `);
                  return;
                }
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                // Mirror the Send button's disabled state — if no model is
                // loaded, Enter is a no-op so users don't hit a confusing
                // backend 500 / "no model loaded" error mid-draft.
                if (!loadedModelRef) return;
                void onSendMessage();
              }
            }}
            onDrop={(event) => {
              const files = event.dataTransfer?.files;
              if (!files?.length) return;
              event.preventDefault();
              void onChatFileDrop(files);
            }}
            onDragOver={(event) => event.preventDefault()}
          />
          </div>
          <div className="button-row composer-button-row">
            <div className="composer-button-group composer-button-group--left">
              <label className="secondary-button composer-attach-btn" title="Attach image">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(event) => {
                    const files = event.target.files;
                    if (!files) return;
                    for (const file of Array.from(files)) {
                      if (file.size > 10 * 1024 * 1024) { onSetError("Image must be under 10MB"); continue; }
                      const reader = new FileReader();
                      reader.onload = () => {
                        const b64 = (reader.result as string).split(",")[1];
                        if (b64) onPendingImagesChange((prev) => [...prev, b64]);
                      };
                      reader.readAsDataURL(file);
                    }
                    event.target.value = "";
                  }}
                />
                {"\uD83D\uDCCE"}
              </label>
              <div
                className="composer-mode-control"
                title="Choose how much reasoning the model performs before answering. Off = direct answers; Low / Medium / High = increasing reasoning depth for capable models."
              >
                <span className="composer-mode-label">Thinking</span>
                <div className="thread-mode-toggle composer-thinking-toggle" role="group" aria-label="Thinking mode">
                  <button
                    type="button"
                    className={`thread-mode-button${thinkingMode === "off" ? " thread-mode-button--active" : ""}`}
                    disabled={chatBusySessionId === activeChat?.id}
                    onClick={handleEffortOff}
                    title="No reasoning — model answers directly"
                  >
                    Off
                  </button>
                  <button
                    type="button"
                    className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "low" ? " thread-mode-button--active" : ""}`}
                    disabled={chatBusySessionId === activeChat?.id}
                    onClick={() => handleEffortChange("low")}
                    title="Brief reasoning"
                  >
                    Low
                  </button>
                  <button
                    type="button"
                    className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "medium" ? " thread-mode-button--active" : ""}`}
                    disabled={chatBusySessionId === activeChat?.id}
                    onClick={() => handleEffortChange("medium")}
                    title="Default reasoning depth"
                  >
                    Med
                  </button>
                  <button
                    type="button"
                    className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "high" ? " thread-mode-button--active" : ""}`}
                    disabled={chatBusySessionId === activeChat?.id}
                    onClick={() => handleEffortChange("high")}
                    title="Extended reasoning"
                  >
                    High
                  </button>
                </div>
              </div>
              <TemperatureChip
                defaultValue={launchSettings.temperature}
                override={temperatureOverride}
                onChange={handleTemperatureOverrideChange}
                disabled={chatBusySessionId === activeChat?.id}
              />
              <button
                className={`secondary-button${enableTools ? " active-toggle" : ""}`}
                type="button"
                onClick={() => onToggleTools(!enableTools)}
                title={enableTools ? "Tools enabled (web search, code, calculator, file reader)" : "Enable agent tools"}
                style={{
                  background: enableTools ? "#1e3a5f" : undefined,
                  borderColor: enableTools ? "#3b82f6" : undefined,
                  color: enableTools ? "#8fb4ff" : undefined,
                  fontSize: 12,
                  padding: "4px 10px",
                }}
              >
                {enableTools ? "Tools ON" : "Tools"}
              </button>
            </div>
            <div className="composer-button-group composer-button-group--right">
              <button className="secondary-button" type="button" onClick={onClearDraft}>
                Clear
              </button>
              {chatBusySessionId !== null ? (
                <button className="secondary-button" type="button" onClick={onCancelGeneration} style={{ background: "#7f1d1d", borderColor: "#dc2626", color: "#fca5a5" }}>
                  Stop
                </button>
              ) : (
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => void onSendMessage()}
                  disabled={!loadedModelRef}
                  title={!loadedModelRef ? "Load a model first to send messages" : undefined}
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </Panel>

    </div>
  );
}
