import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));
vi.mock("../../voice/ComposerMicButton", () => ({ ComposerMicButton: () => null }));

import { ChatComposer, type ChatComposerProps } from "../../chat/ChatComposer";
import { ChatPromptsPanel, type ChatPromptsPanelProps } from "../ChatPromptsPanel";
import {
  applyVariables,
  insertIntoDraft,
  nameFromDraft,
  parseVariablesBox,
  seedVariableValues,
} from "../promptLogic";

// D-87: the prompt library moved from its own page into Chat. The panel's interactions run in
// e2e/tests/29-prompts-in-chat.spec.ts; these check the pure helpers and what the closed panel renders.

function memoryStorage() {
  const store = new Map<string, string>();
  return {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    key: (i: number) => [...store.keys()][i] ?? null,
    get length() {
      return store.size;
    },
  };
}

beforeEach(() => {
  vi.stubGlobal("window", { localStorage: memoryStorage(), sessionStorage: memoryStorage(), dispatchEvent: () => true });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function panel(over: Partial<ChatPromptsPanelProps> = {}) {
  const props: ChatPromptsPanelProps = {
    backendOnline: true,
    draftMessage: "",
    onDraftMessageChange: () => undefined,
    systemPrompt: "",
    onSystemPromptChange: () => undefined,
    ...over,
  };
  return renderToStaticMarkup(<ChatPromptsPanel {...props} />);
}

describe("ChatPromptsPanel (closed)", () => {
  it("is one Prompts button that announces a dialog and is collapsed", () => {
    const html = panel();
    expect(html).toContain(">Prompts<");
    expect(html).toContain('aria-haspopup="dialog"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain('role="dialog"');
    expect(html).not.toContain("chat-prompts__dot");
  });

  it("marks the button when a system prompt is in use", () => {
    const html = panel({ systemPrompt: "Answer in Welsh." });
    expect(html).toContain("chat-prompts__trigger--active");
    expect(html).toContain("chat-prompts__dot");
    expect(html).toContain("A system prompt is in use.");
  });
});

describe("the composer hosts the Prompts panel", () => {
  function composer(over: Partial<ChatComposerProps> = {}) {
    const noop = () => undefined;
    const props = {
      draftMessage: "", pendingImages: [], loadedModelRef: "m", loadedModelCapabilities: null, thinkingMode: "off",
      reasoningEffort: "medium", enableTools: false, chatBusySessionId: null, activeChat: undefined, warmModels: [],
      oneTurnOverride: null, onOneTurnOverrideChange: noop, launchSettings: { temperature: 0.7, cacheStrategy: "native", cacheBits: 0 },
      temperatureOverride: null, samplerOverrides: {}, kvStrategyOverride: null, onKvStrategyOverrideChange: noop,
      availableCacheStrategies: [], showSlashMenu: false, slashMatches: [], slashIndex: 0, setSlashIndex: noop,
      onDraftMessageChange: noop, onPendingImagesChange: noop, onSendMessage: noop, onCancelGeneration: noop, onClearDraft: noop,
      onChatFileDrop: noop, onToggleTools: noop, onSetError: noop, onTemperatureOverrideChange: noop, onSamplerOverridesChange: noop,
      runSlashCommand: noop, handleEffortOff: noop, handleEffortChange: noop,
      ...over,
    } as unknown as ChatComposerProps;
    return renderToStaticMarkup(<ChatComposer {...props} />);
  }

  it("puts the Prompts button first in the right-hand group, before Clear and Send", () => {
    const html = composer({
      promptsControl: (
        <ChatPromptsPanel backendOnline draftMessage="" onDraftMessageChange={() => undefined} systemPrompt="" onSystemPromptChange={() => undefined} />
      ),
    });
    const right = html.slice(html.indexOf("composer-button-group--right"));
    expect(right.indexOf("chat-prompts__trigger")).toBeGreaterThan(-1);
    expect(right.indexOf("chat-prompts__trigger")).toBeLessThan(right.indexOf(">Clear<"));
    expect(right.indexOf(">Clear<")).toBeLessThan(right.indexOf(">Send<"));
  });

  it("renders no Prompts button when none is given", () => {
    expect(composer()).not.toContain("chat-prompts");
  });
});

describe("prompt helpers", () => {
  it("fills placeholders and leaves an unknown one visible", () => {
    expect(applyVariables("Answer in {{ language }} about {{topic}}.", { language: "Welsh" })).toBe("Answer in Welsh about {{topic}}.");
    expect(applyVariables("{{a}} {{b}} {{c}}", { a: true, b: 3, c: null })).toBe("true 3 ");
    expect(applyVariables("", { a: "x" })).toBe("");
  });

  it("seeds each variable with its default or an empty value of its type", () => {
    expect(
      seedVariableValues([
        { name: "language", type: "string", default: "French" },
        { name: "count", type: "number" },
        { name: "formal", type: "boolean" },
        { name: "note", type: "string" },
      ]),
    ).toEqual({ language: "French", count: 0, formal: false, note: "" });
    expect(seedVariableValues(undefined)).toEqual({});
  });

  it("inserts into an empty draft as is and after typed text on a new paragraph", () => {
    expect(insertIntoDraft("", "Summarise this.")).toBe("Summarise this.");
    expect(insertIntoDraft("   ", "Summarise this.")).toBe("Summarise this.");
    expect(insertIntoDraft("Hello  \n", "Summarise this.")).toBe("Hello\n\nSummarise this.");
  });

  it("names a template saved from the message after its first line, cut to 48 characters", () => {
    expect(nameFromDraft("  Review this code\nfor bugs")).toBe("Review this code");
    const long = nameFromDraft("x".repeat(60));
    expect(long.length).toBe(48);
    expect(long.endsWith("…")).toBe(true);
  });

  it("reads the variables box back, dropping rows without a name and anything that is not a list", () => {
    expect(parseVariablesBox('[{"name": "topic", "type": "string"}, {"type": "number"}, 3]')).toEqual([{ name: "topic", type: "string" }]);
    expect(parseVariablesBox('{"name": "topic"}')).toEqual([]);
    expect(parseVariablesBox("[{")).toEqual([]);
    expect(parseVariablesBox("  ")).toEqual([]);
  });
});
