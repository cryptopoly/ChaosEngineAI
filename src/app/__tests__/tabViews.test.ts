import { describe, expect, it } from "vitest";
import { tabViewFor } from "../tabViews";
import { MusicStudioView } from "../MusicTabs";
import { StoryboardView } from "../StoryboardView";
import { DashboardView } from "../SystemTabs";
import { HOME_STUDIO_TABS } from "../../constants/features";

describe("tabViewFor and the home studio flag", () => {
  it("sends Music and Storyboard ids to the dashboard when passed false (the default is on)", () => {
    for (const tab of HOME_STUDIO_TABS) {
      expect(tabViewFor(tab, false)).toBe(DashboardView);
    }
    expect(tabViewFor("music-studio")).not.toBe(DashboardView);
  });

  it("renders the placeholder views when the flag is on", () => {
    expect(tabViewFor("music-studio", true)).toBe(MusicStudioView);
    expect(tabViewFor("storyboard", true)).toBe(StoryboardView);
    for (const tab of HOME_STUDIO_TABS) {
      expect(tabViewFor(tab, true)).not.toBe(DashboardView);
    }
  });
});
