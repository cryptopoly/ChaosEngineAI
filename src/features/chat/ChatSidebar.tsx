import { Panel } from "../../components/Panel";
import type { ChatSession, WarmModel } from "../../types";
import { filterSessions } from "./sessionSearch";

/**
 * Phase 2.1: extracted from ChatTab.tsx. Sidebar listing chat sessions
 * with title/body search, pin / delete affordances, warm-model badges,
 * and the collapsible toggle. Renders nothing when collapsed (parent
 * removes it from the layout).
 */
export interface ChatSidebarProps {
  sortedChatSessions: ChatSession[];
  activeChat: ChatSession | undefined;
  warmModels: WarmModel[];
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSetActiveChatId: (id: string) => void;
  onCreateSession: () => void;
  onToggleThreadPin: (session: ChatSession) => void;
  onDeleteSession: (sessionId: string) => void;
  onCompareMode: () => void;
  onToggleCollapsed: () => void;
}

export function ChatSidebar({
  sortedChatSessions,
  activeChat,
  warmModels,
  searchQuery,
  onSearchQueryChange,
  onSetActiveChatId,
  onCreateSession,
  onToggleThreadPin,
  onDeleteSession,
  onCompareMode,
  onToggleCollapsed,
}: ChatSidebarProps) {
  const filteredChatSessions = filterSessions(sortedChatSessions, searchQuery);

  return (
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
            onClick={onToggleCollapsed}
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
            value={searchQuery}
            onChange={(event) => onSearchQueryChange(event.target.value)}
            aria-label="Search threads"
          />
          {searchQuery ? (
            <button
              type="button"
              className="session-search__clear"
              onClick={() => onSearchQueryChange("")}
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
        {searchQuery && filteredChatSessions.length === 0 ? (
          <p className="muted-text" style={{ fontSize: 12, padding: "8px 4px", margin: 0 }}>
            No threads match "{searchQuery}".
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
                      {"📌"}
                    </span>
                    <span
                      className="session-delete-icon"
                      role="button"
                      tabIndex={0}
                      title="Delete chat"
                      onClick={(e) => { e.stopPropagation(); void onDeleteSession(session.id); }}
                      onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); void onDeleteSession(session.id); } }}
                    >
                      {"✕"}
                    </span>
                  </span>
                </div>
                <div className="session-meta-row">
                  <small>{session.updatedAt}</small>
                  {session.parentSessionId ? (
                    <span
                      className="badge session-fork-badge"
                      title={`Forked from another thread at message #${(session.forkedAtMessageIndex ?? 0) + 1}`}
                    >
                      ⑂ fork
                    </span>
                  ) : null}
                  {session.modelRef && warmModels.some((w) => w.ref === session.modelRef) ? (
                    <span
                      className="badge success session-warm-badge"
                      title="Model is already loaded — this chat will respond instantly with no reload time."
                    >
                      {"⚡"} ready
                    </span>
                  ) : null}
                </div>
              </button>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}
