import { afterEach, describe, expect, it, vi } from "vitest";
import { visibleSidebarTabs } from "../../components/sidebarModel";
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
