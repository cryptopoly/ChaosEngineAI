import type { ChatSession } from "../types";

export function upsertSession(sessions: ChatSession[], nextSession: ChatSession): ChatSession[] {
  return [nextSession, ...sessions.filter((session) => session.id !== nextSession.id)];
}

export function sessionPreview(session: ChatSession) {
  return session.messages[session.messages.length - 1]?.text ?? "No messages yet";
}

export function sortSessions(sessions: ChatSession[]) {
  return [...sessions].sort((left, right) => {
    if (Boolean(left.pinned) !== Boolean(right.pinned)) {
      return left.pinned ? -1 : 1;
    }
    return 0;
  });
}

export function titleFromPrompt(prompt: string) {
  return prompt.trim().split(/\s+/).slice(0, 4).join(" ") || "New chat";
}
