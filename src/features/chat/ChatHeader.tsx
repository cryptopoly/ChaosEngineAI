import type { ChatSession, ModelCapabilities, ModelLoadingState } from "../../types";
import { downloadExport, type ExportFormat } from "./exportThread";

const CAPABILITY_BADGES: Array<{
  flag: keyof ModelCapabilities;
  label: string;
  title: string;
}> = [
  { flag: "supportsVision", label: "Vision", title: "Model accepts image input" },
  { flag: "supportsTools", label: "Tools", title: "Model supports tool / function calling" },
  { flag: "supportsReasoning", label: "Reasoning", title: "Model emits a reasoning trace" },
  { flag: "supportsCoding", label: "Code", title: "Model is tuned for code generation" },
  { flag: "supportsAgents", label: "Agents", title: "Model is tuned for multi-step agentic flows" },
  { flag: "supportsAudio", label: "Audio", title: "Model accepts audio input" },
  { flag: "supportsVideo", label: "Video", title: "Model accepts video input" },
];

/**
 * Phase 2.1: extracted from ChatTab.tsx. The thread header — title
 * editor, model selector, export menu, runtime summary, document
 * chips, and the optional sidebar-expand toggle (rendered when the
 * sidebar is collapsed). Pure presentation; all mutating actions go
 * through the parent's handlers.
 */
export interface ChatHeaderProps {
  activeChat: ChatSession | undefined;
  threadTitleDraft: string;
  activeThreadOptionKey: string | undefined;
  loadedModelRef: string | undefined;
  loadedModelCapabilities?: ModelCapabilities | null;
  serverLoading: ModelLoadingState | null;
  modelBusyLabel: string | null;
  busy: boolean;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  onThreadTitleDraftChange: (title: string) => void;
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
  onDeleteSessionDocument: (sessionId: string, docId: string) => Promise<void>;
  onRefreshWorkspace: (preferredChatId?: string) => Promise<void>;
  onSetError: (msg: string | null) => void;
}

export function ChatHeader({
  activeChat,
  threadTitleDraft,
  activeThreadOptionKey,
  loadedModelRef,
  loadedModelCapabilities,
  serverLoading,
  modelBusyLabel,
  busy,
  sidebarCollapsed,
  onToggleSidebar,
  onThreadTitleDraftChange,
  onRenameActiveThread,
  onOpenModelSelector,
  onLoadModel,
  onDeleteSessionDocument,
  onRefreshWorkspace,
  onSetError,
}: ChatHeaderProps) {
  return (
    <>
      {sidebarCollapsed ? (
        <button
          type="button"
          className="secondary-button sidebar-expand-toggle"
          onClick={onToggleSidebar}
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
          {activeChat?.modelRef === loadedModelRef && loadedModelCapabilities ? (
            <span className="capability-badges" aria-label="Model capabilities">
              {CAPABILITY_BADGES.filter((entry) => loadedModelCapabilities[entry.flag]).map((entry) => (
                <span key={entry.flag} className="capability-badge" title={entry.title}>
                  {entry.label}
                </span>
              ))}
            </span>
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
                ? ` · ${activeChat.contextTokens >= 1024 ? `${Math.round(activeChat.contextTokens / 1024)}K` : activeChat.contextTokens} ctx`
                : ""}
              {activeChat.speculativeDecoding
                ? activeChat.treeBudget ? ` · DDTree(${activeChat.treeBudget})` : " · DFlash"
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
                {"📄"} {doc.originalName}
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
    </>
  );
}
