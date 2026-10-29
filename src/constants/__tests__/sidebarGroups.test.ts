import { describe, expect, it } from "vitest";
import { sidebarGroups } from "../sidebarGroups";
import { tabs } from "../tabs";
import { LEGACY_TAB_REDIRECTS } from "../tabRedirects";

describe("sidebarGroups", () => {
  it("each group's defaultChild is a real tab in that group", () => {
    for (const group of sidebarGroups) {
      const child = tabs.find((t) => t.id === group.defaultChild);
      expect(child, `group "${group.id}" defaultChild "${group.defaultChild}" not found`).toBeDefined();
      expect(child?.group, `defaultChild "${group.defaultChild}" must belong to group "${group.id}"`).toBe(group.id);
    }
  });

  it("every tab.group references a known group", () => {
    const groupIds = new Set(sidebarGroups.map((g) => g.id));
    for (const tab of tabs) {
      if (!tab.group) continue;
      expect(groupIds.has(tab.group), `tab "${tab.id}" references unknown group "${tab.group}"`).toBe(true);
    }
  });

  it("every group has at least one tab", () => {
    for (const group of sidebarGroups) {
      const children = tabs.filter((t) => t.group === group.id);
      expect(children.length, `group "${group.id}" has no tabs`).toBeGreaterThan(0);
    }
  });

  it("grouped tabs have a shortLabel", () => {
    for (const tab of tabs) {
      if (!tab.group) continue;
      expect(tab.shortLabel, `tab "${tab.id}" in group "${tab.group}" missing shortLabel`).toBeTruthy();
    }
  });

  it("has no Discover / My Models tabs left: one Models page per area (FU-129)", () => {
    const ids = tabs.map((tab) => tab.id as string);
    for (const legacy of Object.keys(LEGACY_TAB_REDIRECTS)) {
      expect(ids, `legacy tab "${legacy}" is still registered`).not.toContain(legacy);
    }
    for (const [group, page] of [["chat", "chat-models"], ["images", "image-models"], ["video", "video-models"]] as const) {
      const tab = tabs.find((candidate) => candidate.id === page);
      expect(tab?.group, `${page} must live in group ${group}`).toBe(group);
      expect(tab?.shortLabel).toBe("Models");
    }
  });

  it("opens Images and Video on their Models page and Chat on the chat", () => {
    const defaults = Object.fromEntries(sidebarGroups.map((group) => [group.id, group.defaultChild]));
    expect(defaults.images).toBe("image-models");
    expect(defaults.video).toBe("video-models");
    expect(defaults.chat).toBe("chat");
    for (const legacy of Object.keys(LEGACY_TAB_REDIRECTS)) {
      expect(Object.values(defaults)).not.toContain(legacy);
    }
  });
});
