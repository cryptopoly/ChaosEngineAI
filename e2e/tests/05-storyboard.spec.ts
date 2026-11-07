import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, seedProject } from "./helpers";

// Storyboard: create a project, add shots, the take and lock controls, the captions panel and the
// export panel (AI label, run estimate, Film list). Projects with takes and a voice clip are seeded
// through the backend's own API with a tiny still and a two-second tone; nothing is rendered, so no
// model is needed and the render buttons are never pressed.

const main = (page: Page) => page.getByRole("main");

async function openProject(page: Page, name: string) {
  await goTo(page, "Storyboard", "Projects");
  await page.getByRole("article", { name }).getByRole("button", { name: "Open", exact: true }).click();
  await expect(page.getByRole("heading", { name: `${name} · Board` })).toBeVisible();
}

async function openMixer(page: Page, name: string) {
  await openProject(page, name);
  await main(page).getByRole("button", { name: "Mixer", exact: true }).click();
  await expect(page.getByRole("heading", { name: `${name} · Mixer` })).toBeVisible();
}

test("create a project in a portrait shape and add shots", async ({ page }) => {
  await goTo(page, "Storyboard", "Projects");
  // Earlier specs seeded projects in the shared backend, so the list is not empty here.
  await main(page).getByRole("button", { name: "New project" }).first().click();

  const dialog = page.getByRole("dialog", { name: "New project" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Create" })).toBeDisabled();
  await dialog.getByRole("textbox").fill("Dawn Patrol");
  await dialog.getByRole("radio", { name: /9:16/ }).click();
  await expect(dialog.getByRole("radio", { name: /9:16/ })).toHaveAttribute("aria-checked", "true");
  await dialog.getByRole("button", { name: "Create" }).click();

  await expect(page.getByRole("heading", { name: "Dawn Patrol · Board" })).toBeVisible();
  await expect(page.getByText("0 shots · 0 s · 0 stale")).toBeVisible();
  await expect(page.getByText("No shots yet. Add a shot, then give it a prompt or import footage.")).toBeVisible();
  await expect(main(page).getByText(/9:16 · 24 fps/)).toBeVisible();

  await page.getByRole("button", { name: "Add shot" }).first().click();
  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();
  await page.getByRole("button", { name: "Add shot" }).first().click();
  await expect(page.getByText("2 shots · 8 s · 0 stale")).toBeVisible();
  await expect(page.getByText("Not made yet")).toHaveCount(2);

  // The run estimate strip: no run yet, so every figure is honest about not knowing.
  await expect(page.getByText("2 shots need a new take")).toBeVisible();
  await expect(page.getByText("Time: unknown until the first run")).toBeVisible();
  await expect(page.getByText("Memory: not estimated")).toBeVisible();
  await expect(page.getByText("Disk: not estimated")).toBeVisible();
  await expect(page.getByRole("button", { name: "Make missing" })).toBeEnabled();

  // Undo takes the last shot back out.
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();

  // Back on Projects the new card is listed.
  await main(page).getByRole("button", { name: "Projects", exact: true }).click();
  await expect(page.getByRole("article", { name: "Dawn Patrol" })).toContainText("1 shot");
});

test("takes: pick another take, then lock the shot and the strip follows", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Takes Film", shots: 1, takes: 2 });
  await openProject(page, project.name);

  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();
  const takes = page.getByRole("radiogroup", { name: "Takes" });
  await expect(takes.getByRole("radio")).toHaveCount(2);
  await expect(takes.getByRole("radio", { name: "Take 2 · Active take" })).toBeChecked();
  await takes.getByRole("radio", { name: "Take 1" }).click();
  await expect(takes.getByRole("radio", { name: "Take 1 · Active take" })).toBeChecked();
  await expect(takes.getByRole("radio", { name: "Take 2" })).not.toBeChecked();

  // Every unlocked shot has a take, so "Make missing" has nothing to do.
  await expect(page.getByText("Every unlocked shot has a take.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Make missing" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeEnabled();

  const lock = page.getByRole("button", { name: "Lock this shot" });
  await expect(lock).toHaveAttribute("aria-pressed", "false");
  await lock.click();
  const unlock = page.getByRole("button", { name: "Unlock this shot" });
  await expect(unlock).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Locked", { exact: true })).toBeVisible();
  // A locked shot is skipped, so there is nothing left for "Make all unlocked" either.
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeDisabled();
  await unlock.click();
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeEnabled();

  const redo = page.locator(".shot-card").getByRole("button", { name: "Redo" });
  await redo.click();
  await expect(redo).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Marked for redo")).toBeVisible();
});

test("Mixer: captions panel without speech, and with a voice clip", async ({ page, request }) => {
  const silent = await seedProject(request, { name: "Silent Film", shots: 1 });
  await openMixer(page, silent.name);
  const captions = page.locator(".captions-panel");
  await expect(captions).toContainText("Captions (0)");
  await captions.locator("summary").click();
  await expect(captions.getByText("Add speech to the voice lane to make captions from it.")).toBeVisible();
  await expect(captions.getByRole("button", { name: "Make from voice" })).toBeDisabled();
  await expect(captions.getByRole("button", { name: "Write .srt" })).toBeDisabled();
  await captions.getByRole("button", { name: "Add at playhead" }).click();
  await expect(captions).toContainText("Captions (1)");
  await expect(captions.getByRole("textbox", { name: "Caption text" })).toBeVisible();
  await expect(captions.getByRole("button", { name: "Write .srt" })).toBeEnabled();
  await captions.getByRole("button", { name: "Remove" }).click();
  await expect(captions).toContainText("Captions (0)");

  const spoken = await seedProject(request, { name: "Spoken Film", shots: 1, clonedVoice: true });
  await openMixer(page, spoken.name);
  const withVoice = page.locator(".captions-panel");
  await withVoice.locator("summary").click();
  await expect(withVoice.getByText("Add speech to the voice lane")).toHaveCount(0);
  await expect(withVoice.getByRole("button", { name: "Make from voice" })).toBeEnabled();
});

test("Mixer export panel: forced AI tag for a cloned voice, estimate strip and Film list", async ({ page, request }) => {
  // A project with no cloned voice lets the AI tag be switched off; one with a cloned voice does not.
  const plain = await seedProject(request, { name: "Plain Film", shots: 1 });
  await openMixer(page, plain.name);
  const exportPanel = page.getByRole("region", { name: "Export" });
  const tag = exportPanel.getByRole("switch", { name: "Add a machine-readable AI tag" });
  await expect(tag).toBeEnabled();
  await expect(tag).toBeChecked();
  await tag.click();
  await expect(tag).not.toBeChecked();
  await expect(exportPanel.getByText("This project uses a cloned voice")).toHaveCount(0);

  const cloned = await seedProject(request, { name: "Cloned Film", shots: 1, clonedVoice: true });
  await page.route(`**/api/storyboard/projects/${cloned.id}/films`, (route) =>
    route.fulfill({
      json: { films: [{ file: "exports/cloned-film.mp4", name: "cloned-film.mp4", bytes: 3_500_000, modifiedAt: 1_790_000_000 }] },
    }),
  );
  await openMixer(page, cloned.name);
  const panel = page.getByRole("region", { name: "Export" });
  const forced = panel.getByRole("switch", { name: "Add a machine-readable AI tag" });
  await expect(forced).toBeChecked();
  await expect(forced).toBeDisabled();
  await expect(panel.getByText("This project uses a cloned voice, so the AI tag is always written.")).toBeVisible();

  await expect(panel.getByRole("radiogroup", { name: "Shape" })).toBeVisible();
  await expect(panel.getByText("Time: unknown until the first run")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Export MP4" })).toBeVisible();
  await expect(panel.getByRole("button", { name: /Export for an editor/ })).toBeVisible();

  const films = panel.locator(".film-list");
  await expect(films.getByRole("heading", { name: "Film" })).toBeVisible();
  await expect(films).toContainText("cloned-film.mp4");
  await expect(films).toContainText("3.3 MB");
  await expect(films.getByRole("button")).toHaveCount(1);
});
