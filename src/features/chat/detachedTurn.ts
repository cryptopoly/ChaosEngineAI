/**
 * S-J: a studio turn goes on in the backend when the window that started it reloads. The reloaded window sees
 * the thread's last message without a reply, asks the backend whether that turn still runs, shows the plan
 * cards it waits on (the click answers the same turn) and refreshes the thread when the reply is saved.
 */
import { useEffect, useRef, useState } from "react";
import { fetchStudioTurn } from "../../api";
import type { ChatSession, StudioPlan, ToolCallInfo } from "../../types";

export const DETACHED_POLL_MS = 2000;
const RETRY_MS = 5000;

/** FU-208: the reply so far of a turn this window does not stream (text, reasoning, tool calls). */
export interface DetachedPartial {
  text: string;
  reasoning: string;
  toolCalls: ToolCallInfo[];
}

export interface DetachedTurn {
  sessionId: string;
  plans: StudioPlan[];
  partial?: DetachedPartial;
}

/** The thread's last message is the user's, with no reply yet: a turn may be running elsewhere. */
export function awaitingReply(session: Pick<ChatSession, "messages"> | undefined): boolean {
  const messages = session?.messages ?? [];
  return messages.length > 0 && messages[messages.length - 1].role === "user";
}

/**
 * Poll one thread's turn until it ends. `onTurn` gets the running turn (or null once it ended); `onFinished` runs
 * once when a turn seen running has ended. Returns the stop function.
 */
export function watchDetachedTurn(
  sessionId: string,
  onTurn: (turn: DetachedTurn | null) => void,
  onFinished: (sessionId: string) => void,
  fetchTurn: typeof fetchStudioTurn = fetchStudioTurn,
): () => void {
  let stopped = false;
  let seenRunning = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const poll = async () => {
    try {
      const next = await fetchTurn(sessionId);
      if (stopped) return;
      if (next.running) {
        seenRunning = true;
        onTurn({ sessionId, plans: next.plans ?? [], ...(next.partial ? { partial: next.partial } : {}) });
        timer = setTimeout(() => void poll(), DETACHED_POLL_MS);
        return;
      }
      onTurn(null);
      if (seenRunning) onFinished(sessionId);
    } catch {
      if (!stopped) timer = setTimeout(() => void poll(), RETRY_MS);  // backend restarting: ask again
    }
  };
  void poll();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}

/**
 * The running turn of the open thread that this window does not stream itself, or null. `ownStreamSessionId` is
 * the thread this window streams (its own cards arrive on the stream). `onFinished` runs once when a turn this
 * hook saw running has ended, so the saved reply can be loaded.
 */
export function useDetachedTurn(
  session: ChatSession | undefined,
  ownStreamSessionId: string | null,
  onFinished: (sessionId: string) => void,
): DetachedTurn | null {
  const [turn, setTurn] = useState<DetachedTurn | null>(null);
  const finishedRef = useRef(onFinished);
  useEffect(() => {
    finishedRef.current = onFinished;
  }, [onFinished]);
  const sessionId = session?.id ?? null;
  const watch = Boolean(sessionId) && awaitingReply(session) && ownStreamSessionId !== sessionId;

  useEffect(() => {
    if (!watch || !sessionId) {
      setTurn(null);
      return;
    }
    return watchDetachedTurn(sessionId, setTurn, (id) => finishedRef.current(id));
  }, [watch, sessionId]);

  return turn && turn.sessionId === sessionId ? turn : null;
}
