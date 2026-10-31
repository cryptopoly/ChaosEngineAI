import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LEGACY_TAB_REDIRECTS,
  isLegacyTabId,
  migrateLastChildByGroup,
  modelsAreaForTab,
  modelsPageRequest,
  normalizeStoredTabId,
  resolveTabRequest,
  type TabRequest,
} from "../tabRedirects";
import { tabs } from "../tabs";
import { readStoredSidebarPrefs } from "../../hooks/useSidebarPrefs";

describe("resolveTabRequest", () => {
  it("sends the old Discover ids to the area's Models page on Available", () => {
    expect(resolveTabRequest("online-models")).toEqual({ tab: "chat-models", modelsPreset: "available" });
    expect(resolveTabRequest("image-discover")).toEqual({ tab: "image-models", modelsPreset: "available" });
    expect(resolveTabRequest("video-discover")).toEqual({ tab: "video-models", modelsPreset: "available" });
  });

  it("sends the old chat My Models id to the chat Models page on Installed", () => {
    expect(resolveTabRequest("my-models")).toEqual({ tab: "chat-models", modelsPreset: "installed" });
  });

  it("passes a real tab through without a preset", () => {
    expect(resolveTabRequest("image-models")).toEqual({ tab: "image-models" });
    expect(resolveTabRequest("settings")).toEqual({ tab: "settings" });
  });

  it("keeps the preset of an explicit navigation", () => {
    expect(resolveTabRequest(modelsPageRequest("image", "installed"))).toEqual({ tab: "image-models", modelsPreset: "installed" });
    expect(resolveTabRequest(modelsPageRequest("video"))).toEqual({ tab: "video-models" });
  });

  it("redirects a stale legacy id inside a navigation object, keeping an explicit preset", () => {
    const stale = { tab: "image-discover" } as unknown as TabRequest;
    expect(resolveTabRequest(stale)).toEqual({ tab: "image-models", modelsPreset: "available" });
    const staleWithPreset = { tab: "my-models", modelsPreset: "available" } as unknown as TabRequest;
    expect(resolveTabRequest(staleWithPreset)).toEqual({ tab: "chat-models", modelsPreset: "available" });
  });

  it("ignores an empty request", () => {
    expect(resolveTabRequest(null)).toBeNull();
    expect(resolveTabRequest(undefined)).toBeNull();
  });

  it("only redirects to tabs that exist", () => {
    const tabIds = new Set(tabs.map((tab) => tab.id));
    for (const redirect of Object.values(LEGACY_TAB_REDIRECTS)) {
      expect(tabIds.has(redirect.tab), redirect.tab).toBe(true);
    }
  });
});

describe("Models page helpers", () => {
  it("maps each area to its page and back", () => {
    expect(modelsPageRequest("chat", "available")).toEqual({ tab: "chat-models", modelsPreset: "available" });
    expect(modelsAreaForTab("chat-models")).toBe("chat");
    expect(modelsAreaForTab("image-models")).toBe("image");
    expect(modelsAreaForTab("video-models")).toBe("video");
    expect(modelsAreaForTab("image-studio")).toBeNull();
  });

  it("knows the legacy ids", () => {
    expect(isLegacyTabId("video-discover")).toBe(true);
    expect(isLegacyTabId("video-models")).toBe(false);
    expect(isLegacyTabId(undefined)).toBe(false);
  });
});

describe("persisted sidebar ids", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("maps legacy ids and keeps the others", () => {
    expect(normalizeStoredTabId("image-discover")).toBe("image-models");
    expect(normalizeStoredTabId("my-models")).toBe("chat-models");
    expect(normalizeStoredTabId("voice-models")).toBe("voice-models");
    expect(normalizeStoredTabId(42)).toBeNull();
    expect(normalizeStoredTabId("")).toBeNull();
  });

  it("migrates lastChildByGroup", () => {
    expect(
      migrateLastChildByGroup({ chat: "online-models", images: "image-discover", video: "video-discover", voice: "voice-studio", tools: 7 }),
    ).toEqual({ chat: "chat-models", images: "image-models", video: "video-models", voice: "voice-studio" });
    expect(migrateLastChildByGroup(null)).toEqual({});
  });

  it("migrates what useSidebarPrefs reads from localStorage", () => {
    const stored = {
      mode: "tabs",
      collapsedGroups: ["tools"],
      lastChildByGroup: { chat: "my-models", images: "image-discover", video: "video-studio" },
    };
    vi.stubGlobal("window", {
      localStorage: { getItem: () => JSON.stringify(stored), setItem: () => {} },
    });
    expect(readStoredSidebarPrefs()).toEqual({
      mode: "tabs",
      collapsedGroups: ["tools"],
      lastChildByGroup: { chat: "chat-models", images: "image-models", video: "video-studio" },
    });
  });

  it("survives broken storage", () => {
    vi.stubGlobal("window", { localStorage: { getItem: () => "{not json", setItem: () => {} } });
    expect(readStoredSidebarPrefs()).toEqual({});
  });
});
