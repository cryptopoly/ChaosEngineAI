import type { ChatSession } from "../types";

export function upsertSession(sessions: ChatSession[], nextSession: ChatSession): ChatSession[] {
  return [nextSession, ...sessions.filter((session) => session.id !== nextSession.id)];
}

function titleVariantPattern(baseTitle: string) {
  return new RegExp(`^${baseTitle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?: \\((\\d+)\\))?$`);
}

const TITLE_LEADING_PATTERNS = [
  /^(?:please\s+)+/i,
  /^(?:can|could|would|will)\s+you\s+/i,
  /^(?:can|could|would|will)\s+we\s+/i,
  /^i\s+(?:need|want|would\s+like)\s+(?:you\s+to\s+)?/i,
  /^help\s+me\s+/i,
  /^make\s+it\s+so\s+that\s+/i,
  /^tell\s+me\s+(?:about\s+)?(?:the\s+)?/i,
  /^show\s+me\s+(?:how\s+to\s+)?/i,
  /^give\s+me\s+/i,
];

function cleanPromptForTitle(prompt: string) {
  let text = prompt
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .trim()
    .replace(/^[\s#>*\-\d.)]+/, "")
    .replace(/\s+/g, " ");
  if (!text) return "";

  text = text.split(/(?<=[.!?])\s+/, 1)[0] ?? text;
  let candidate = text.trim().replace(/^[\s"'`*_~:;,.!?()[\]{}]+|[\s"'`*_~:;,.!?()[\]{}]+$/g, "");
  for (let i = 0; i < 4; i += 1) {
    const previous = candidate;
    for (const pattern of TITLE_LEADING_PATTERNS) {
      candidate = candidate.replace(pattern, "").trim();
    }
    candidate = candidate.replace(/\s+please$/i, "").trim();
    if (candidate === previous) break;
  }
  return candidate.replace(/^[\s"'`*_~:;,.!?()[\]{}]+|[\s"'`*_~:;,.!?()[\]{}]+$/g, "");
}

export function isUnsavedEmptySession(session: Pick<ChatSession, "id" | "messages" | "documents">) {
  return (
    session.id.startsWith("draft-") &&
    session.messages.length === 0 &&
    (!session.documents || session.documents.length === 0)
  );
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
  const words = cleanPromptForTitle(prompt).match(/[A-Za-z0-9][A-Za-z0-9+'’._/#-]*/g) ?? [];
  let baseTitle = words.slice(0, 6).join(" ").trim();
  if (baseTitle.length > 64) {
    baseTitle = baseTitle.slice(0, 64).replace(/\s+\S*$/, "") || baseTitle.slice(0, 64);
  }
  baseTitle = baseTitle.replace(/^[\s"'`*_~:;,.!?()[\]{}]+|[\s"'`*_~:;,.!?()[\]{}]+$/g, "") || "New chat";
  if (baseTitle === baseTitle.toLowerCase()) {
    baseTitle = baseTitle.charAt(0).toUpperCase() + baseTitle.slice(1);
  }
  if (!sessions) return baseTitle;
  return uniqueSessionTitle(sessions, baseTitle, excludeId);
}
