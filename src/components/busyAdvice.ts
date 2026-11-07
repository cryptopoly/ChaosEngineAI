/** Pure helpers for the busy-machine notice (chat composer, model load dialog, Studio estimate strip). */
import type { BusyAdvice, ChatMessage } from "../types";

/** Advice worth showing: anything but "ok" with a message. */
export function isVisibleAdvice(advice: BusyAdvice | null | undefined): advice is BusyAdvice {
  return advice != null && advice.level !== "ok" && advice.message.length > 0;
}

/** The notice for the open turn: the last assistant message's advice, while its reply is being generated.
 *  The key changes per message so a dismissal only hides that turn's notice. */
export function pickChatAdvice(
  chatId: string,
  messages: ChatMessage[],
  generating: boolean,
): { key: string; advice: BusyAdvice } | null {
  if (!generating || messages.length === 0) return null;
  const index = messages.length - 1;
  const advice = messages[index]?.role === "assistant" ? messages[index].busyAdvice : null;
  return isVisibleAdvice(advice) ? { key: `${chatId}:${index}`, advice } : null;
}

/** A risky load is offered a confirmation rather than blocked. */
export function needsConfirmation(advice: BusyAdvice | null | undefined): boolean {
  return advice?.level === "risky";
}
