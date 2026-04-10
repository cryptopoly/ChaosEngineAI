import type { ChatSession } from "../types";

export function upsertSession(sessions: ChatSession[], nextSession: ChatSession): ChatSession[] {
  return [nextSession, ...sessions.filter((session) => session.id !== nextSession.id)];
}

function titleVariantPattern(baseTitle: string) {
  return new RegExp(`^${baseTitle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?: \\((\\d+)\\))?$`);
}

export function uniqueSessionTitle(
  sessions: ChatSession[],
  baseTitle: string,
  excludeId?: string,
) {
  const base = baseTitle.trim() || "New chat";
  if (base === "New chat") return base;

  const pattern = titleVariantPattern(base);
  let highestSuffix = 0;
  for (const session of sessions) {
    if (excludeId && session.id === excludeId) continue;
    const match = pattern.exec((session.title ?? "").trim());
    if (!match) continue;
    highestSuffix = Math.max(highestSuffix, match[1] ? Number.parseInt(match[1], 10) : 1);
  }

  if (highestSuffix === 0) return base;
  return `${base} (${highestSuffix + 1})`;
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

export function titleFromPrompt(prompt: string, sessions?: ChatSession[], excludeId?: string) {
  const baseTitle = prompt.trim().split(/\s+/).slice(0, 4).join(" ") || "New chat";
  if (!sessions) return baseTitle;
  return uniqueSessionTitle(sessions, baseTitle, excludeId);
}
