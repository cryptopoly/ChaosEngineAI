import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { ChatHeader, type ChatHeaderProps } from "../ChatHeader";
import type { ChatSession } from "../../../types";

// FU-450: with no thread and no model loaded, both refs are undefined and used to compare equal, so a first
// launch showed a green "Ready" badge beside "Select Model".
const noop = () => undefined;
const props = (over: Partial<ChatHeaderProps>): ChatHeaderProps => ({
  activeChat: undefined,
  threadTitleDraft: "",
  activeThreadOptionKey: undefined,
  loadedModelRef: undefined,
  serverLoading: null,
  modelBusyLabel: null,
  busy: false,
  sidebarCollapsed: false,
  onToggleSidebar: noop,
  onThreadTitleDraftChange: noop,
  onRenameActiveThread: noop,
  onOpenModelSelector: noop,
  onLoadModel: noop,
  onDeleteSessionDocument: async () => undefined,
  onRefreshWorkspace: async () => undefined,
  onSetError: noop,
  onUnloadModel: noop,
  ...over,
});
const chat = { id: "s1", title: "A thread", modelRef: "org/model", model: "Model", messages: [] } as unknown as ChatSession;

describe("ChatHeader Ready badge", () => {
  it("is not shown when nothing is loaded and there is no thread", () => {
    expect(renderToStaticMarkup(<ChatHeader {...props({})} />)).not.toContain("badge success");
  });

  it("is shown when the thread's model is the loaded one", () => {
    const html = renderToStaticMarkup(<ChatHeader {...props({ activeChat: chat, loadedModelRef: "org/model" })} />);
    expect(html).toContain("badge success");
  });
});
