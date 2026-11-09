import { test, expect } from "./fixtures";
import { goTo } from "./helpers";
import type { Page } from "@playwright/test";

// 24 library items overflow the window: the page must scroll far enough to reach the last card, on a desktop
// and a phone. Library data is mocked at the network edge.

const sizes = [
  { name: "desktop", width: 1440, height: 800 },
  { name: "375 px", width: 375, height: 812 },
];

async function expectLastReachable(page: Page, cards: string) {
  const last = page.locator(cards).last();
  await expect(page.locator(cards)).toHaveCount(24);
  // Scroll the way a person does: the wheel over the cards (scrollIntoView ignores overflow:hidden boxes).
  const viewport = page.viewportSize()!;
  await page.mouse.move(viewport.width / 2, viewport.height - 60);
  for (let i = 0; i < 60; i += 1) await page.mouse.wheel(0, 400);
  await expect(last).toBeInViewport();
  // The first card is out of view after scrolling, so the page really did scroll.
  await expect(page.locator(cards).first()).not.toBeInViewport();
}

for (const size of sizes) {
  test.describe(size.name, () => {
    test.use({ viewport: { width: size.width, height: size.height } });

    test("Music and Audio library: the 24th card can be scrolled into view", async ({ page }) => {
      const items = Array.from({ length: 24 }, (_, i) => ({
        id: `m-${i}`, kind: i % 2 ? "sound" : "track", title: `Item ${i}`, caption: `caption ${i}`, lyrics: "", tags: [],
        createdAt: 1_790_000_000 + i, updatedAt: 1_790_000_000 + i, audio: "wav", modelName: "Mock music", seed: i,
        durationS: 12, sizeBytes: 1_000_000,
      }));
      await page.route(/\/api\/music\/library(\?.*)?$/, (route) => route.fulfill({ json: { items, count: items.length, diskBytes: 24_000_000 } }));
      await goTo(page, "Music and Audio", "Library");
      await expectLastReachable(page, ".music-library__item");
    });

    test("Voice library: the 24th card can be scrolled into view", async ({ page }) => {
      const items = Array.from({ length: 24 }, (_, i) => ({
        id: `v-${i}`, kind: "speech", title: `Clip ${i}`, text: `text ${i}`, tags: [], createdAt: 1_790_000_000 + i,
        updatedAt: 1_790_000_000 + i, audio: "wav", durationS: 4, sizeBytes: 400_000, voice: "af_heart", model: "kokoro",
      }));
      await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items } }));
      await goTo(page, "Voice", "Library");
      await expectLastReachable(page, ".voice-library__grid > *");
    });
  });
}
