import { test, expect } from "./fixtures";
import { goTo, openTab } from "./helpers";

// Music Studio's Music | Sounds switch, and the Stable Audio Open licence screen. The catalog is the
// backend's own (no model is downloaded or run here), so these also pin the licence wording shipped.

test("Music | Sounds switch shows the sound model with its attribution and the next step", async ({ page }) => {
  await goTo(page, "Music", "Studio");
  const mode = page.getByRole("group", { name: "Music or sound effects" });
  const music = mode.getByRole("button", { name: "Music", exact: true });
  const sounds = mode.getByRole("button", { name: "Sounds", exact: true });
  await expect(music).toBeVisible();
  await expect(sounds).toBeVisible();
  // Music mode first: the ACE-Step model, and the pack is not installed on a clean run.
  await expect(page.getByText("ACE-Step 1.5 turbo")).toBeVisible();
  await expect(page.getByText("The music pack is not installed yet.")).toBeVisible();

  await expect(music).toHaveAttribute("aria-pressed", "true");
  await sounds.click();
  await expect(sounds).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Stable Audio Open Small")).toBeVisible();
  await expect(page.getByText("Powered by Stability AI")).toBeVisible();
  await expect(page.getByText(/asks you to accept its licence first/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Music Models" })).toBeVisible();

  await music.click();
  await expect(page.getByText("ACE-Step 1.5 turbo")).toBeVisible();
  await expect(page.getByText("Powered by Stability AI")).toHaveCount(0);
});

test("Stable Audio Open licence screen: terms, attribution, Not now and I have accepted", async ({ page }) => {
  await goTo(page, "Music", "Models");
  // Two sound models are listed (Small and 1.0), each with its own licence gate: act on Small's row.
  const smallRow = page.locator('[data-model-id="stabilityai/stable-audio-open-small"]');
  await expect(smallRow).toBeVisible();
  await expect(smallRow.getByText("NEEDS LICENCE", { exact: false }).first()).toBeVisible();

  await smallRow.getByRole("button", { name: "Accept licence" }).click();
  const dialog = page.getByRole("dialog", { name: /Stable Audio Open Small/ });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Stability AI Community Licence");
  await expect(dialog).toContainText("What this licence asks of you");
  await expect(dialog).toContainText("US$1 million a year");
  await expect(dialog).toContainText("Powered by Stability AI");
  await expect(dialog).toContainText("The weights are never bundled with ChaosEngineAI.");
  await expect(dialog.getByRole("button", { name: "Read the full licence" })).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Open on Hugging Face" })).toBeVisible();

  // Not now leaves it unaccepted.
  await dialog.getByRole("button", { name: "Not now" }).click();
  await expect(dialog).toBeHidden();
  await expect(smallRow.getByRole("button", { name: "Accept licence" })).toBeVisible();

  // Accepting is remembered in this browser and the Studio's next step moves on to the download.
  await smallRow.getByRole("button", { name: "Accept licence" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "I have accepted" }).click();
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(smallRow.getByRole("button", { name: "Accept licence" })).toHaveCount(0);
  // Acceptance is per model: 1.0 still asks.
  await expect(page.locator('[data-model-id="stabilityai/stable-audio-open-1.0"]').getByRole("button", { name: "Accept licence" })).toBeVisible();

  await openTab(page, "Music", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();
  await expect(page.getByText(/asks you to accept its licence first/)).toHaveCount(0);
  await expect(page.getByText(/Download Stable Audio Open Small from Music Models/)).toBeVisible();
});
