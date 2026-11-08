import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";

// Music Models rows that are downloading or paused show a progress bar, the percent and "downloaded / total GB".
// The downloads API is mocked with partial progress; nothing is downloaded.

const TURBO = "ACE-Step/Ace-Step1.5";
const XL = "ACE-Step/acestep-v15-xl-turbo";
const SOUND = "stabilityai/stable-audio-open-small";

async function mockDownloads(page: Page) {
  await page.route("**/api/music/download/status", (route) =>
    route.fulfill({
      json: {
        downloads: [
          { repo: TURBO, state: "downloading", progress: 0.42, downloadedGb: 4.2, totalGb: 10.0, error: null },
          { repo: XL, state: "cancelled", progress: 0.25, downloadedGb: 5.0, totalGb: 20.0, error: null },
          { repo: SOUND, state: "downloading", progress: 0.6, downloadedGb: 1.5, totalGb: 2.6, error: null },
        ],
      },
    }),
  );
}

async function expectProgress(page: Page) {
  const rows = page.getByTestId("model-download-progress");
  await expect(rows).toHaveCount(3);

  const turbo = page.locator(`[data-model-id="${TURBO}"]`).getByTestId("model-download-progress");
  await expect(turbo).toBeVisible();
  await expect(turbo.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
  await expect(turbo).toContainText("Downloading 42%");
  await expect(turbo).toContainText("4.2 / 10.0 GB");

  const xl = page.locator(`[data-model-id="${XL}"]`).getByTestId("model-download-progress");
  await expect(xl).toContainText("Paused 25%");
  await expect(xl).toContainText("5.0 / 20.0 GB");
  await expect(page.locator(`[data-model-id="${XL}"]`)).toHaveAttribute("data-status", "paused");

  const sound = page.locator(`[data-model-id="${SOUND}"]`).getByTestId("model-download-progress");
  await expect(sound).toContainText("Downloading 60%");
  await expect(sound).toContainText("1.5 / 2.6 GB");

  // The Downloading chip counts the paused row too.
  await expect(page.getByRole("button", { name: /^Downloading\s*3$/ })).toBeVisible();
}

test("Music Models: downloading and paused rows show bar, percent and GB", async ({ page }) => {
  await mockDownloads(page);
  await goTo(page, "Music and Audio", "Models");
  await expectProgress(page);
});

test.describe("375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("Music Models: the progress line is visible and the page does not scroll sideways", async ({ page }) => {
    await mockDownloads(page);
    await goTo(page, "Music and Audio", "Models");
    await expectProgress(page);
    const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(sideways).toBeLessThanOrEqual(0);
  });
});
