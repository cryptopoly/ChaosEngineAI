import type { ChatSession } from "../../types";

/**
 * Case-insensitive substring search across session title and all message bodies
 * (including reasoning traces). Returns sessions whose title OR any message
 * matches the query. An empty query passes everything through unchanged.
 */
export function filterSessions(sessions: ChatSession[], query: string): ChatSession[] {
  const trimmed = query.trim().toLowerCase();
  if (!trimmed) return sessions;
  return sessions.filter((session) => sessionMatchesQuery(session, trimmed));
}

function sessionMatchesQuery(session: ChatSession, lowerQuery: string): boolean {
  if (session.title && session.title.toLowerCase().includes(lowerQuery)) return true;
  for (const msg of session.messages) {
    if (msg.text && msg.text.toLowerCase().includes(lowerQuery)) return true;
    if (msg.reasoning && msg.reasoning.toLowerCase().includes(lowerQuery)) return true;
  }
  return false;
}
