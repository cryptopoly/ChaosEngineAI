import { test, expect } from "./fixtures";
import { goTo } from "./helpers";

// Review stream D: the Video Gallery fetched every clip as a whole file the moment it opened (133 clips, about
// 100 MB, on the owner's library). A card now fetches its clip when it scrolls into view. Forty mocked clips:
// only the first rows are asked for, the last one only after scrolling to it.

function clip(i: number) {
  return {
    artifactId: `lazy-${i}`,
    modelId: "mock/video",
    modelName: "Mock video",
    prompt: `clip prompt ${i}`,
    width: 640,
    height: 360,
    numFrames: 24,
    fps: 24,
    steps: 4,
    guidance: 1,
    seed: i,
    createdAt: new Date(Date.UTC(2026, 10, 1, 12, 0, 0) - i * 60_000).toISOString(),
    durationSeconds: 2,
    clipDurationSeconds: 1,
    projectId: null,
    projectName: null,
  };
}

test("Video Gallery: clips load as their cards come into view, not all at once", async ({ page }) => {
  const asked = new Set<string>();
  await page.route("**/api/video/outputs", (route) =>
    route.fulfill({ json: { outputs: Array.from({ length: 40 }, (_, i) => clip(i + 1)) } }),
  );
  await page.route("**/api/video/outputs/*/file", (route) => {
    if (route.request().method() === "GET") asked.add(/outputs\/([^/]+)\/file/.exec(route.request().url())![1]);
    return route.fulfill({ status: 404, body: "" });
  });
  await goTo(page, "Video", "Gallery");
  const cards = page.locator(".video-output-card");
  await expect(cards).toHaveCount(40);
  await expect.poll(() => asked.size).toBeGreaterThan(0);
  await page.waitForTimeout(1500);
  expect(asked.size, `asked for ${[...asked].join(", ")}`).toBeLessThanOrEqual(16);
  expect(asked.has("lazy-1")).toBe(true);
  expect(asked.has("lazy-40")).toBe(false);
  // A card that was never in view still says it is loading, it has not failed.
  await expect(cards.last()).toContainText("Loading clip...");

  await cards.last().scrollIntoViewIfNeeded();
  await expect.poll(() => asked.has("lazy-40")).toBe(true);
  expect(asked.size).toBeLessThan(40);
});
