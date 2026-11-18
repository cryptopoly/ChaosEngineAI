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

// D-87 verdicts (overnight review): Fine-Tuning trains nothing yet and Plugins loads no outside code, so both are
// experimental. Prompts, Conversion and Benchmarks stay listed.
describe("Fine-Tuning and Plugins are experimental", () => {
  it("hides only those two by default", () => {
    const hidden = tabs.filter((tab) => tab.experimental).map((tab) => tab.id);
    expect(hidden).toEqual(["finetuning", "plugins"]);
    const listed = visibleSidebarTabs(tabs, "Darwin", true).map((tab) => tab.id);
    expect(listed).not.toContain("finetuning");
    expect(listed).not.toContain("plugins");
    expect(listed).toEqual(expect.arrayContaining(["prompt-library", "conversion", "benchmarks", "benchmark-history"]));
    expect(visibleSidebarTabs(tabs, "Darwin", true, true).map((tab) => tab.id)).toEqual(
      expect.arrayContaining(["finetuning", "plugins"]),
    );
  });

  it("sends a request for either one to the Dashboard while the switch is off", () => {
    expect(visibleTabRequest("finetuning", false)).toBe("dashboard");
    expect(visibleTabRequest("plugins", false)).toBe("dashboard");
    expect(visibleTabRequest("plugins", true)).toBe("plugins");
    expect(visibleTabRequest("prompt-library", false)).toBe("prompt-library");
  });

  it("does not reopen a remembered hidden tab from the Tools group", () => {
    const tools = visibleSidebarTabs(tabs, "Darwin", true).filter((tab) => tab.group === "tools");
    expect(resolveGroupTarget("tools", tools, { tools: "plugins" })).toBe("conversion");
    const toolsElsewhere = visibleSidebarTabs(tabs, "Linux", true).filter((tab) => tab.group === "tools");
    expect(resolveGroupTarget("tools", toolsElsewhere, { tools: "finetuning" })).toBe("prompt-library");
  });

  it("keeps them out of the sub-tab bar unless the switch is on", () => {
    const props = { activeTab: "conversion" as const, onTabChange: () => {}, platform: "Darwin", onRememberLastChild: () => {} };
    const off = renderToStaticMarkup(createElement(SubtabBar, props));
    expect(off).toContain("Prompts");
    expect(off).not.toContain("Fine-Tuning");
    expect(off).not.toContain("Plugins");
    const on = renderToStaticMarkup(createElement(SubtabBar, { ...props, showExperimentalTabs: true }));
    expect(on).toContain("Fine-Tuning");
    expect(on).toContain("Plugins");
  });
});
