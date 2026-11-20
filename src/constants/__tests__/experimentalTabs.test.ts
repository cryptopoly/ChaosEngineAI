import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SubtabBar } from "../../components/SubtabBar";
import { resolveGroupTarget, visibleSidebarTabs } from "../../components/sidebarModel";
import { readStoredSidebarPrefs } from "../../hooks/useSidebarPrefs";
import { tabs, visibleTabRequest, type TabConfig } from "../tabs";

// Settings > General > "Show experimental tabs" (off by default). A tab flagged ``experimental`` is left out of
// the sidebar and the sub-tab bar until the switch is on, and a request for it lands on the Dashboard.

const sample: TabConfig[] = [
  { id: "conversion", label: "Conversion", labelKey: "tabs.conversion", caption: "", captionKey: "", group: "tools" },
  { id: "finetuning", label: "Fine-Tuning", labelKey: "tabs.fineTuning", caption: "", captionKey: "", group: "tools", experimental: true },
];

describe("Show experimental tabs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("leaves an experimental tab out of the sidebar until the switch is on", () => {
    expect(visibleSidebarTabs(sample, "Darwin", true).map((tab) => tab.id)).toEqual(["conversion"]);
    expect(visibleSidebarTabs(sample, "Darwin", true, false).map((tab) => tab.id)).toEqual(["conversion"]);
    expect(visibleSidebarTabs(sample, "Darwin", true, true).map((tab) => tab.id)).toEqual(["conversion", "finetuning"]);
  });

  it("opens every tab when the switch is on, and every tab that is not experimental when it is off", () => {
    for (const tab of tabs) {
      expect(visibleTabRequest(tab.id, true)).toBe(tab.id);
      if (!tab.experimental) expect(visibleTabRequest(tab.id, false)).toBe(tab.id);
      else expect(visibleTabRequest(tab.id, false)).toBe("dashboard");
    }
  });

  it("is remembered with the sidebar preferences and off when nothing is stored", () => {
    vi.stubGlobal("window", {
      localStorage: { getItem: () => JSON.stringify({ mode: "collapsible", showExperimentalTabs: true }), setItem: () => {} },
    });
    expect(readStoredSidebarPrefs().showExperimentalTabs).toBe(true);
    vi.stubGlobal("window", { localStorage: { getItem: () => null, setItem: () => {} } });
    expect(readStoredSidebarPrefs().showExperimentalTabs).toBeUndefined();
  });
});

// D-87: Fine-Tuning trains nothing yet, so it is experimental. Prompts moved into Chat and Plugins was removed, so
// neither is a tab any more. Conversion and Benchmarks stay listed.
describe("Fine-Tuning is experimental; Prompts and Plugins are gone", () => {
  it("hides only Fine-Tuning by default and lists neither Prompts nor Plugins", () => {
    const hidden = tabs.filter((tab) => tab.experimental).map((tab) => tab.id);
    expect(hidden).toEqual(["finetuning"]);
    const ids = tabs.map((tab) => tab.id as string);
    expect(ids).not.toContain("prompt-library");
    expect(ids).not.toContain("plugins");
    const listed = visibleSidebarTabs(tabs, "Darwin", true).map((tab) => tab.id);
    expect(listed).not.toContain("finetuning");
    expect(listed).toEqual(expect.arrayContaining(["conversion", "benchmarks", "benchmark-history"]));
    expect(visibleSidebarTabs(tabs, "Darwin", true, true).map((tab) => tab.id)).toContain("finetuning");
  });

  it("sends a request for Fine-Tuning to the Dashboard while the switch is off", () => {
    expect(visibleTabRequest("finetuning", false)).toBe("dashboard");
    expect(visibleTabRequest("finetuning", true)).toBe("finetuning");
  });

  it("does not reopen a remembered hidden or removed tab from the Tools group", () => {
    const tools = visibleSidebarTabs(tabs, "Darwin", true).filter((tab) => tab.group === "tools");
    expect(resolveGroupTarget("tools", tools, { tools: "plugins" })).toBe("conversion");
    expect(resolveGroupTarget("tools", tools, { tools: "finetuning" })).toBe("conversion");
    // Elsewhere Conversion is not listed, so with the switch off the Tools group has no page and is not shown.
    expect(visibleSidebarTabs(tabs, "Linux", true).filter((tab) => tab.group === "tools")).toEqual([]);
  });

  it("keeps Fine-Tuning out of the sub-tab bar unless the switch is on", () => {
    const props = { activeTab: "conversion" as const, onTabChange: () => {}, platform: "Darwin", onRememberLastChild: () => {} };
    const off = renderToStaticMarkup(createElement(SubtabBar, props));
    expect(off).not.toContain("Prompts");
    expect(off).not.toContain("Fine-Tuning");
    expect(off).not.toContain("Plugins");
    const on = renderToStaticMarkup(createElement(SubtabBar, { ...props, showExperimentalTabs: true }));
    expect(on).toContain("Fine-Tuning");
    expect(on).not.toContain("Plugins");
  });
});
