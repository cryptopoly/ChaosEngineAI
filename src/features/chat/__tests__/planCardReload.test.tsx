import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatSession, StudioPlan, ToolCallInfo, WorkspaceData } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { ChatThread, type ChatThreadProps } from "../ChatThread";
import { StudioJobCard } from "../StudioJobCard";
import { DETACHED_POLL_MS, awaitingReply, watchDetachedTurn } from "../detachedTurn";
import { appendOptimisticTurn } from "../optimisticTurns";
import { refusalReason } from "../studioChat";

const REASON = "Invalid input for storyboard_generate: 'what' must be one of still, clip, both, final, redo-clip, redo-still.";

function plan(over: Partial<StudioPlan> = {}): StudioPlan {
  return {
    planId: "plan-1",
    sessionId: "s1",
    items: [{ callId: "c1", tool: "generate_image", kind: "image", prompt: "Rose poster", model: "FLUX.1-schnell" }],
    memory: null,
    status: "pending",
    ...over,
  };
}

function session(over: Partial<ChatSession> = {}): ChatSession {
  return {
    id: "s1",
    title: "New chat",
    updatedAt: "now",
    model: "Qwen",
    messages: [],
    ...over,
  } as ChatSession;
}

function refusedCall(over: Partial<ToolCallInfo> = {}): ToolCallInfo {
  return {
    id: "c1",
    name: "storyboard_generate",
    arguments: { project_id: "p1", what: "stills" },
    result: `Error: ${REASON}`,
    elapsed: 0,
    renderAs: "studio",
    data: { tool: "storyboard_generate", error: REASON },
    ...over,
  };
}

function threadProps(over: Partial<ChatThreadProps> = {}): ChatThreadProps {
  const noop = () => undefined;
  return {
    activeChat: session(),
    chatBusySessionId: null,
    chatScrollRef: { current: null },
    serverLoading: null,
    engineLabel: "MLX",
    launchSettings: { contextTokens: 8192 } as ChatThreadProps["launchSettings"],
    busy: false,
    loadedModelRef: "qwen",
    onChatFileDrop: noop,
    onCopyMessage: noop,
    onRetryMessage: noop,
    onDeleteMessage: noop,
    onForkAtMessage: noop,
    warmModels: [],
    onAddVariant: noop,
    onDelveMessage: noop,
    onDetailsToggle: noop,
    onCancelGeneration: noop,
    onLoadModel: noop,
    ...over,
  };
}

describe("refused studio call", () => {
  it("is read from the call's error", () => {
    expect(refusalReason(refusedCall())).toBe(REASON);
    expect(refusalReason(refusedCall({ renderAs: "json" }))).toBeNull();
    expect(refusalReason(refusedCall({ data: { jobId: "job-1" } }))).toBeNull();
    expect(refusalReason(refusedCall({ running: true }))).toBeNull();
  });

  it("shows a notice card with the reason in full", () => {
    const html = renderToStaticMarkup(<StudioJobCard toolCall={refusedCall()} />);
    expect(html).toContain("studio-job-card--refused");
    expect(html).toContain("Not started");
    expect(html).toContain("&#x27;what&#x27; must be one of still");
    expect(html).toContain("The model was told why");
  });

  it("is a notice card in the thread, not a folded tool line", () => {
    const chat = session({
      title: "Make the stills",
      messages: [
        { role: "user", text: "make the stills", metrics: null },
        { role: "assistant", text: "", metrics: null, toolCalls: [refusedCall()] },
      ],
    });
    const html = renderToStaticMarkup(<ChatThread {...threadProps({ activeChat: chat })} />);
    expect(html).toContain("studio-job-card--refused");
    expect(html).toContain("must be one of still");
  });
});

describe("running thread title", () => {
  function apply(workspace: WorkspaceData, run: (set: (u: (w: WorkspaceData) => WorkspaceData) => void) => void) {
    let current = workspace;
    run((updater) => {
      current = updater(current);
    });
    return current;
  }

  it("takes the first message as soon as it is sent", () => {
    const ws = { chatSessions: [session(), session({ id: "s2", title: "Other" })] } as unknown as WorkspaceData;
    const next = apply(ws, (set) => appendOptimisticTurn(set as never, "s1", "Paint a red rose poster for the shop"));
    expect(next.chatSessions[0].title).toBe("Paint a red rose poster for");
    expect(next.chatSessions[1].title).toBe("Other");
  });

  it("names two running threads apart and keeps a title already given", () => {
    let ws = { chatSessions: [session(), session({ id: "s2" })] } as unknown as WorkspaceData;
    ws = apply(ws, (set) => appendOptimisticTurn(set as never, "s1", "Rose poster"));
    ws = apply(ws, (set) => appendOptimisticTurn(set as never, "s2", "Rose poster"));
    expect(ws.chatSessions.map((s) => s.title)).toEqual(["Rose poster", "Rose poster (2)"]);
    const named = { chatSessions: [session({ title: "My film" })] } as unknown as WorkspaceData;
    expect(apply(named, (set) => appendOptimisticTurn(set as never, "s1", "Rose poster")).chatSessions[0].title).toBe("My film");
    const later = {
      chatSessions: [session({ messages: [{ role: "user", text: "hi", metrics: null }, { role: "assistant", text: "hello", metrics: null }] })],
    } as unknown as WorkspaceData;
    expect(apply(later, (set) => appendOptimisticTurn(set as never, "s1", "Rose poster")).chatSessions[0].title).toBe("New chat");
  });
});

describe("plan card after a reload", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("waits only for a thread whose last message has no reply", () => {
    expect(awaitingReply(session({ messages: [{ role: "user", text: "go", metrics: null }] }))).toBe(true);
    expect(awaitingReply(session({ messages: [{ role: "user", text: "go", metrics: null }, { role: "assistant", text: "ok", metrics: null }] }))).toBe(false);
    expect(awaitingReply(session())).toBe(false);
    expect(awaitingReply(undefined)).toBe(false);
  });

  it("shows the waiting card again, then loads the reply once the turn ends", async () => {
    const answers = [
      { sessionId: "s1", running: true, plans: [plan()] },
      { sessionId: "s1", running: true, plans: [] },
      { sessionId: "s1", running: false, plans: [] },
    ];
    const fetchTurn = vi.fn(async () => answers.shift()!);
    const turns: unknown[] = [];
    const finished = vi.fn();
    const stop = watchDetachedTurn("s1", (t) => turns.push(t), finished, fetchTurn);
    await vi.advanceTimersByTimeAsync(0);
    expect(turns[0]).toEqual({ sessionId: "s1", plans: [plan()] });
    await vi.advanceTimersByTimeAsync(DETACHED_POLL_MS);
    expect(turns[1]).toEqual({ sessionId: "s1", plans: [] });
    expect(finished).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(DETACHED_POLL_MS);
    expect(turns[2]).toBeNull();
    expect(finished).toHaveBeenCalledWith("s1");
    await vi.advanceTimersByTimeAsync(DETACHED_POLL_MS * 3);
    expect(fetchTurn).toHaveBeenCalledTimes(3);
    stop();
  });

  it("does nothing for a thread with no running turn", async () => {
    const fetchTurn = vi.fn(async () => ({ sessionId: "s1", running: false, plans: [] }));
    const finished = vi.fn();
    watchDetachedTurn("s1", () => undefined, finished, fetchTurn);
    await vi.advanceTimersByTimeAsync(DETACHED_POLL_MS * 2);
    expect(fetchTurn).toHaveBeenCalledTimes(1);
    expect(finished).not.toHaveBeenCalled();
  });

  it("renders the waiting card under the user's message", () => {
    const chat = session({ title: "Paint a rose", messages: [{ role: "user", text: "Paint a rose", metrics: null }] });
    const html = renderToStaticMarkup(
      <ChatThread {...threadProps({ activeChat: chat, chatBusySessionId: "s1", detachedTurn: { sessionId: "s1", plans: [plan()] } })} />,
    );
    expect(html).toContain('data-testid="detached-turn"');
    expect(html).toContain("waiting for your answer");
    expect(html).toContain("Rose poster");
    const running = renderToStaticMarkup(
      <ChatThread {...threadProps({ activeChat: chat, detachedTurn: { sessionId: "s1", plans: [] } })} />,
    );
    expect(running).toContain("still being made");
    const other = renderToStaticMarkup(
      <ChatThread {...threadProps({ activeChat: chat, detachedTurn: { sessionId: "s2", plans: [plan()] } })} />,
    );
    expect(other).not.toContain("detached-turn");
  });
});
