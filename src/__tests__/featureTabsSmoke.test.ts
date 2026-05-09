/**
 * Phase 0 import-smoke for the largest untested feature tabs.
 *
 * Each tab is well over 1k lines — a typo, broken import, or removed
 * export during the refactor would land silently because nothing else
 * pulls them into the test graph. These checks load each module and
 * assert the function reference is callable, which catches:
 *   - syntax / parse errors
 *   - missing/renamed sibling helpers
 *   - circular imports that fail at module-init time
 *
 * Full mounting tests are deferred — every tab takes 30+ deeply-nested
 * props and the goal here is the safety net, not behavioural coverage.
 * Phase 2 splits each tab into testable pieces and adds real renders.
 */

import { describe, expect, it } from "vitest";

import { HtmlChallengeTab } from "../features/chat/HtmlChallengeTab";
import { CompareView, compareTargets, cloneLaunchSettings } from "../features/chat/CompareView";
import { VideoStudioTab } from "../features/video/VideoStudioTab";
import { ImageStudioTab } from "../features/images/ImageStudioTab";

describe("feature tabs — import smoke", () => {
  it("HtmlChallengeTab is a function component", () => {
    expect(typeof HtmlChallengeTab).toBe("function");
    expect(HtmlChallengeTab.name).toBe("HtmlChallengeTab");
  });

  it("CompareView is a function component", () => {
    expect(typeof CompareView).toBe("function");
    expect(CompareView.name).toBe("CompareView");
  });

  it("CompareView module exports stable helper surface", () => {
    expect(compareTargets).toEqual(["a", "b", "c", "d"]);
    expect(typeof cloneLaunchSettings).toBe("function");
  });

  it("VideoStudioTab is a function component", () => {
    expect(typeof VideoStudioTab).toBe("function");
    expect(VideoStudioTab.name).toBe("VideoStudioTab");
  });

  it("ImageStudioTab is a function component", () => {
    expect(typeof ImageStudioTab).toBe("function");
    expect(ImageStudioTab.name).toBe("ImageStudioTab");
  });
});
