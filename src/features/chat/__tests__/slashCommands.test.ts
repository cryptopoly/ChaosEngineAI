import { describe, it, expect, vi } from "vitest";
import { matchSlashCommands, findExactCommand, type SlashCommandContext } from "../slashCommands";
import type { ChatSession } from "../../../types";

function makeContext(overrides: Partial<SlashCommandContext> = {}): SlashCommandContext {
  return {
    args: "",
    activeChat: undefined,
    loadedModelRef: undefined,
    enableTools: false,
    chatBusySessionId: null,
    onClearDraft: vi.fn(),
    onThinkingModeChange: vi.fn(),
    onToggleTools: vi.fn(),
    onOpenModelSelector: vi.fn(),
    onCancelGeneration: vi.fn(),
    activeThreadOptionKey: undefined,
    ...overrides,
  };
}

function makeChat(messages = [{ role: "user" as const, text: "hi" }]): ChatSession {
  return {
    id: "s1",
    title: "Test",
    updatedAt: "2026-05-01",
    cacheLabel: "f16",
    model: "Test/Model",
    messages,
  };
}

describe("matchSlashCommands", () => {
  it("returns empty when draft does not start with slash", () => {
    expect(matchSlashCommands("hello", makeContext())).toEqual([]);
  });

  it("returns empty when draft contains a newline", () => {
    expect(matchSlashCommands("/think\non", makeContext())).toEqual([]);
  });

  it("filters by command prefix", () => {
    const result = matchSlashCommands("/think", makeContext());
    const commands = result.map((c) => c.command);
    expect(commands).toContain("/think on");
    expect(commands).toContain("/think off");
  });

  it("hides /cancel when not generating", () => {
    const result = matchSlashCommands("/", makeContext({ chatBusySessionId: null }));
    expect(result.map((c) => c.command)).not.toContain("/cancel");
  });

  it("shows /cancel when generating", () => {
    const result = matchSlashCommands("/cancel", makeContext({ chatBusySessionId: "s1" }));
    expect(result.map((c) => c.command)).toContain("/cancel");
  });

  it("hides /export when no messages", () => {
    const result = matchSlashCommands("/export", makeContext({ activeChat: undefined }));
    expect(result.map((c) => c.command)).not.toContain("/export md");
  });

  it("shows /export when messages exist", () => {
    const result = matchSlashCommands("/export", makeContext({ activeChat: makeChat() }));
    expect(result.map((c) => c.command)).toContain("/export md");
    expect(result.map((c) => c.command)).toContain("/export json");
    expect(result.map((c) => c.command)).toContain("/export txt");
  });
});

describe("findExactCommand", () => {
  it("finds /clear exactly", () => {
    expect(findExactCommand("/clear")?.command).toBe("/clear");
  });

  it("finds /think on with trailing whitespace", () => {
    expect(findExactCommand("/think on   ")?.command).toBe("/think on");
  });

  it("returns undefined for partial match", () => {
    expect(findExactCommand("/thi")).toBeUndefined();
  });
});

describe("slash command run", () => {
  it("/clear calls onClearDraft and keeps draft", () => {
    const ctx = makeContext();
    const cmd = findExactCommand("/clear")!;
    const shouldClear = cmd.run(ctx);
    expect(ctx.onClearDraft).toHaveBeenCalledTimes(1);
    expect(shouldClear).toBe(false);
  });

  it("/think on calls onThinkingModeChange with 'auto'", () => {
    const ctx = makeContext();
    findExactCommand("/think on")!.run(ctx);
    expect(ctx.onThinkingModeChange).toHaveBeenCalledWith("auto");
  });

  it("/tools off disables tools", () => {
    const ctx = makeContext();
    findExactCommand("/tools off")!.run(ctx);
    expect(ctx.onToggleTools).toHaveBeenCalledWith(false);
  });

  it("/cancel calls onCancelGeneration", () => {
    const ctx = makeContext({ chatBusySessionId: "s1" });
    findExactCommand("/cancel")!.run(ctx);
    expect(ctx.onCancelGeneration).toHaveBeenCalled();
  });
});
