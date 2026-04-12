import type { Ref } from "react";
import Markdown from "react-markdown";
import { Panel } from "../../components/Panel";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import type { ChatSession, ModelLoadingState, LaunchPreferences, WarmModel } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import { number } from "../../utils";

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
  onSetActiveChatId: (id: string) => void;
  onThreadTitleDraftChange: (title: string) => void;
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
    source?: string;
    backend?: string;
    path?: string;
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
  onSetActiveChatId,
  onThreadTitleDraftChange,
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
}: ChatTabProps) {
  return (
    <div className="chat-layout-2col">
      <Panel
        title="Chats"
        subtitle=""
        className="chat-column"
        actions={
          <button className="secondary-button" type="button" onClick={() => void onCreateSession()}>
            New thread
          </button>
        }
      >
        <div className="thread-list-panel">
          <div className="session-list">
            {sortedChatSessions.map((session) => (
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

      <Panel title="Active Thread" subtitle="Response metadata is collapsed by default, but available per agent turn." className="chat-thread">
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
            {activeChat?.modelRef === loadedModelRef ? (
              <span className="badge success">Ready</span>
            ) : serverLoading ? (
              <div className="badge accent chat-loading-pill">
                <span className="busy-dot" />
                Loading {serverLoading.modelName}... {serverLoading.elapsedSeconds}s
              </div>
            ) : busyAction === "Loading model..." ? (
              <div className="badge accent chat-loading-pill">
                <span className="busy-dot" />
                Loading model...
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
                    source: activeChat.modelSource ?? "library",
                    backend: activeChat.modelBackend ?? "auto",
                    path: activeChat.modelPath ?? undefined,
                  });
                }}
              >
                {busy ? "Loading..." : "Load model"}
              </button>
            ) : null}
          </div>
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
                  <div className={`markdown-content${isStreamingMessage ? " streaming-cursor" : ""}`}>
                    <Markdown>{message.text || "\u200B"}</Markdown>
                  </div>
                ) : (
                  <p>{message.text}</p>
                )}
                {message.metrics ? (
                  <details className="message-details" onToggle={(event) => void onDetailsToggle(event.currentTarget.open)}>
                    <summary>
                      <span>Model details</span>
                      <small className="message-meta">
                        {(message.metrics.model ?? activeChat.model) || "Unknown"} | {number(message.metrics.tokS)} tok/s | {number(message.metrics.responseSeconds ?? 0)} s
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
                        <p>{message.metrics.cacheLabel ?? activeChat.cacheLabel}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Backend</span>
                        <p>{message.metrics.backend ?? activeChat.modelBackend ?? "Auto"}</p>
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
                        <span className="eyebrow">Context</span>
                        <p>{message.metrics.contextTokens?.toLocaleString() ?? launchSettings.contextTokens.toLocaleString()}</p>
                      </div>
                    </div>
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
          {chatBusySessionId === activeChat?.id && serverLoading ? (
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
          <textarea
            className="text-area"
            placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
            rows={3}
            value={draftMessage}
            onChange={(event) => onDraftMessageChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
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
          <div className="button-row">
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
            <button className="primary-button" type="button" onClick={() => void onSendMessage()} disabled={chatBusySessionId !== null}>
              Send
            </button>
            <button className="secondary-button" type="button" onClick={() => { onDraftMessageChange(""); onPendingImagesChange([]); }}>
              Clear
            </button>
          </div>
        </div>
      </Panel>

    </div>
  );
}
