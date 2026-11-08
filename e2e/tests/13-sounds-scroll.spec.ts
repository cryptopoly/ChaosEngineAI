import { test, expect } from "./fixtures";
import { goTo } from "./helpers";

// Four finished sounds in the Sounds studio: the page must scroll far enough to reach the last one.

test("Sounds studio: the last of four results can be scrolled into view", async ({ page }) => {
  const items = [1, 2, 3, 4].map((n) => ({
    id: `snd-${n}`, kind: "sound", title: "A squirrel chewing on some nuts", caption: "A squirrel chewing on some nuts",
    lyrics: "", tags: [], createdAt: 1_790_000_000 + n, updatedAt: 1_790_000_000 + n, audio: "flac", seed: 100 + n,
    durationS: 5, jobSeconds: 4, sizeBytes: 800_000, model: "stabilityai/stable-audio-open-small",
  }));
  await page.route("**/api/music/library/*/audio*", (route) => route.fulfill({ status: 200, contentType: "audio/flac", body: Buffer.alloc(64) }));
  await page.route("**/api/music/sound/generate", (route) =>
    route.fulfill({ json: { items, seed: 100, jobId: "j1", model: "stabilityai/stable-audio-open-small", seconds: 5 } }),
  );
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      if (family.kind !== "sound") continue;
      for (const variant of family.variants) Object.assign(variant, { availableLocally: true, weightsSource: "checkpoints", packInstalled: true, textEncoderReady: true });
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.runtime.sound = { ...body.runtime.sound, installed: true, unavailableReason: null, lane: "mps" };
    await route.fulfill({ response, json: body });
  });
  await page.setViewportSize({ width: 2000, height: 730 });
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();
  await page.getByRole("textbox").first().fill("a squirrel chewing on some nuts");
  await page.getByRole("button", { name: "Make sound" }).click();
  const rows = page.locator(".music-studio__results [role=listitem]");
  await expect(rows).toHaveCount(4);
  const last = rows.nth(3);
  await last.scrollIntoViewIfNeeded();
  await page.screenshot({ path: "screenshots/sounds-scroll.png" });
  const box = await last.boundingBox();
  const viewport = page.viewportSize()!;
  expect(box!.y + box!.height, "the last result ends inside the window").toBeLessThanOrEqual(viewport.height + 1);
  await expect(last.getByRole("button", { name: /storyboard/i })).toBeInViewport();
});
