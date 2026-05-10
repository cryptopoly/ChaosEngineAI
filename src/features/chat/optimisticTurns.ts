/**
 * Optimistic chat-turn state mutations.
 *
 * Three helpers pulled out of ``useChat`` so the streaming hook can stay
 * focused on the network/abort/SSE machinery. Each takes a workspace
 * setter and produces an updater that:
 *
 * * ``appendOptimisticTurn`` — push a user message + an empty assistant
 *   message in ``prompt_eval`` phase. Renders before the first SSE event
 *   so the typing indicator shows immediately on send.
 * * ``replaceOptimisticAssistant`` — fill the empty assistant turn with
 *   the final text once the stream completes. Falls back to appending a
 *   fresh user/assistant pair if the optimistic turn was already
 *   replaced (e.g. a parallel render swept it).
 * * ``rollbackOptimisticTurn`` — drop the empty optimistic pair when
 *   the stream errors before producing any text.
 *
 * Extracted from ``src/hooks/useChat.ts`` as part of the v0.8.0 Phase
 * 2c-4 refactor. The hook now thin-wraps each one.
 */

import type { Dispatch, SetStateAction } from "react";
import type { ChatSession, WorkspaceData } from "../../types";


type WorkspaceSetter = Dispatch<SetStateAction<WorkspaceData>>;


export function appendOptimisticTurn(
  setWorkspace: WorkspaceSetter,
  sessionId: string,
  prompt: string,
): void {
  const updatedAt = new Date().toLocaleString();
  setWorkspace((current) => ({
    ...current,
    chatSessions: current.chatSessions.map((session) =>
      session.id === sessionId
        ? {
            ...session,
            updatedAt,
            messages: [
              ...session.messages,
              { role: "user" as const, text: prompt, metrics: null },
              {
                role: "assistant" as const,
                text: "",
                reasoning: "",
                reasoningDone: true,
                metrics: null,
                // Phase 2.0: start in prompt_eval so the indicator shows
                // immediately on send, before backend's first SSE phase
                // event arrives. Cleared by onDone via the session refresh.
                streamPhase: "prompt_eval",
              },
            ],
          }
        : session,
    ),
  }));
}


export function replaceOptimisticAssistant(
  setWorkspace: WorkspaceSetter,
  sessionId: string,
  prompt: string,
  text: string,
): void {
  const updatedAt = new Date().toLocaleString();
  setWorkspace((current) => ({
    ...current,
    chatSessions: current.chatSessions.map((session) => {
      if (session.id !== sessionId) return session;
      const messages = [...session.messages];
      const last = messages[messages.length - 1];
      const previous = messages[messages.length - 2];
      if (
        last?.role === "assistant" &&
        !last.text &&
        !last.metrics &&
        previous?.role === "user" &&
        previous.text === prompt
      ) {
        messages[messages.length - 1] = { ...last, text };
      } else {
        messages.push({ role: "user", text: prompt, metrics: null });
        messages.push({ role: "assistant", text, metrics: null });
      }
      return { ...session, updatedAt, messages };
    }),
  }));
}


export function rollbackOptimisticTurn(
  setWorkspace: WorkspaceSetter,
  sessionId: string,
  prompt: string,
): void {
  setWorkspace((current) => ({
    ...current,
    chatSessions: current.chatSessions.map((session) => {
      if (session.id !== sessionId) return session;
      const messages = [...session.messages];
      const last = messages[messages.length - 1];
      const previous = messages[messages.length - 2];
      if (
        last?.role === "assistant" &&
        !last.text &&
        !last.metrics &&
        previous?.role === "user" &&
        previous.text === prompt
      ) {
        return { ...session, messages: messages.slice(0, -2) };
      }
      return session;
    }),
  }));
}


export function mergeSessionMetadata(
  session: ChatSession,
  patch: Partial<ChatSession>,
): ChatSession {
  return { ...session, ...patch };
}
