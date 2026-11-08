import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { RUN_HOME, goToProjects, openProject, openProjectTab, seedProject } from "./helpers";

// Projects (D-65): the one Projects entry, its list (search, templates), the project shell with its five
// tabs, the Cast with up to four pictures, and the Storyboard, Mixer and Export tabs. Projects with takes
// and a voice clip are seeded through the backend's own API with a tiny still and a two-second tone;
// nothing is rendered, so no model is needed and the render buttons are never pressed.

const main = (page: Page) => page.getByRole("main");
const shell = (page: Page) => page.locator(".project-shell");

async function newProjectDialog(page: Page) {
  await goToProjects(page);
  // Earlier specs seeded projects in the shared backend, so the list is not empty here.
  await main(page).getByRole("button", { name: "New project" }).first().click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await expect(dialog).toBeVisible();
  return dialog;
}

test("create a project in a portrait aspect ratio and add shots", async ({ page }) => {
  const dialog = await newProjectDialog(page);
  await expect(dialog.getByRole("button", { name: "Create" })).toBeDisabled();
  await dialog.getByRole("textbox").first().fill("Dawn Patrol");
  // Start from Blank by default; the format: Aspect ratio, Resolution (720p chosen), Frame rate.
  await expect(dialog.getByRole("radio", { name: /^Blank/ })).toHaveAttribute("aria-checked", "true");
  await expect(dialog.getByRole("radiogroup", { name: "Aspect ratio" })).toBeVisible();
  await expect(dialog.getByText("Shape", { exact: true })).toHaveCount(0);
  await expect(dialog.getByRole("radio", { name: /^720p/ })).toHaveAttribute("aria-checked", "true");
  await expect(dialog.getByRole("radiogroup", { name: "Frame rate" }).getByRole("radio", { name: "24 fps" })).toHaveAttribute("aria-checked", "true");
  await dialog.getByRole("radio", { name: /^9:16$/ }).click();
  await expect(dialog.getByRole("radio", { name: /^9:16$/ })).toHaveAttribute("aria-checked", "true");
  await dialog.getByRole("button", { name: "Create" }).click();

  // A new project opens on Overview, with its format in the header chips.
  await expect(shell(page).getByRole("heading", { name: "Dawn Patrol" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  await expect(shell(page).locator(".project-shell__chips")).toHaveText(/9:16\s*720p\s*24 fps/);
  await expect(page.getByRole("button", { name: "Write your shots" })).toBeVisible();

  await page.getByRole("button", { name: "Write your shots" }).click();
  await expect(page.getByRole("tab", { name: /^Storyboard/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("0 shots · 0 s · 0 stale")).toBeVisible();
  await expect(page.getByText(/No shots yet\. Add a shot and describe it/)).toBeVisible();

  await page.getByRole("button", { name: "Add shot" }).first().click();
  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();
  await page.getByRole("button", { name: "Add shot" }).first().click();
  await expect(page.getByText("2 shots · 8 s · 0 stale")).toBeVisible();
  await expect(page.getByText("Not made yet")).toHaveCount(2);

  // The run estimate: no run yet, so one chip says it is not known.
  await expect(page.getByText("2 shots need a new take")).toBeVisible();
  await expect(page.getByText("Estimate: unknown until the first run")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Make \d+ missing/ })).toBeEnabled();

  // Undo, in the project header, takes the last shot back out.
  await shell(page).getByRole("button", { name: "Undo" }).click();
  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();

  // The back link returns to the list, where the new card is listed with its format.
  await shell(page).getByRole("button", { name: "Projects", exact: true }).click();
  await expect(page.getByRole("article", { name: "Dawn Patrol" })).toContainText("1 shot");
  await expect(page.getByRole("article", { name: "Dawn Patrol" })).toContainText("9:16 · 720p · 24 fps");
});

test("new project: every resolution and frame rate is offered, and the dialog fits 375 px", async ({ page }) => {
  const dialog = await newProjectDialog(page);
  const resolution = dialog.getByRole("radiogroup", { name: "Resolution" });
  for (const name of ["480p", "720p", "1080p", "1440p", "2160p (4K)"]) {
    await expect(resolution.getByRole("radio", { name: new RegExp(`^${name.replace(/[()]/g, "\\$&")}`) })).toBeEnabled();
  }
  await expect(resolution.getByRole("radio", { name: /^480p/ })).toContainText("854 × 480");
  await expect(resolution.getByRole("radio", { name: /^2160p/ })).toContainText("3840 × 2160");
  const fps = dialog.getByRole("radiogroup", { name: "Frame rate" });
  for (const rate of [24, 25, 30, 48, 50, 60]) await expect(fps.getByRole("radio", { name: `${rate} fps` })).toBeEnabled();
  // The cost note is always there; with no video model installed the fit note says how to get a hint.
  await expect(dialog.getByText(/Higher resolutions render slower and use more memory\./)).toBeVisible();
  await expect(dialog.getByText(/Install a video model to see whether each size fits/)).toBeVisible();
  // Templates, including one without any character.
  const start = dialog.getByRole("radiogroup", { name: "Start from" });
  for (const name of ["Blank", "30 s ad", "Vertical social", "Short story", "Scene / b-roll"]) {
    await expect(start.getByRole("radio", { name: new RegExp(`^${name.replace("/", "\\/")}`) })).toBeVisible();
  }

  // Aspect ratio changes the pixel sizes shown.
  await dialog.getByRole("radio", { name: /^9:16$/ }).click();
  await expect(resolution.getByRole("radio", { name: /^1080p/ })).toContainText("1080 × 1920");

  // 375 px: nothing sticks out sideways.
  await page.setViewportSize({ width: 375, height: 812 });
  const fits = await dialog.evaluate((el) => {
    const box = el.getBoundingClientRect();
    return {
      inside: box.left >= 0 && box.right <= window.innerWidth,
      pageScrolls: document.documentElement.scrollWidth > window.innerWidth,
      dialogScrolls: el.scrollWidth > el.clientWidth,
    };
  });
  expect(fits).toEqual({ inside: true, pageScrolls: false, dialogScrolls: false });
  await dialog.getByRole("button", { name: "Cancel" }).click();
});

test("new project from the Vertical social template at 1080p 9:16", async ({ page }) => {
  const dialog = await newProjectDialog(page);
  await dialog.getByRole("textbox").first().fill("Night Market Reel");
  await dialog.getByRole("radio", { name: /^Vertical social/ }).click();
  await expect(dialog.getByRole("radio", { name: /^9:16$/ })).toHaveAttribute("aria-checked", "true");
  await expect(dialog.getByRole("radio", { name: /^1080p/ })).toHaveAttribute("aria-checked", "true");
  await expect(dialog.getByRole("radio", { name: /^1080p/ })).toContainText("1080 × 1920");
  await expect(dialog.getByRole("radiogroup", { name: "Frame rate" }).getByRole("radio", { name: "30 fps" })).toHaveAttribute("aria-checked", "true");
  // The template says what it set, before anything is created (P1-2).
  await expect(dialog.getByText("Vertical social set the format to 9:16, 1080p, 30 fps. Change any of them below.")).toBeVisible();
  await dialog.getByRole("button", { name: "Create" }).click();

  await expect(shell(page).getByRole("heading", { name: "Night Market Reel" })).toBeVisible();
  await expect(shell(page).locator(".project-shell__chips")).toHaveText(/9:16\s*1080p\s*30 fps/);
  await expect(page.getByText(/^3 shots · 0 with a clip · 0 still only · 0 stale/)).toBeVisible();
  // The next step for a template project is one obvious button.
  await expect(page.getByRole("heading", { name: "Make 3 shots" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Make 3 missing" })).toBeVisible();
  await openProjectTab(page, "Storyboard");
  await expect(page.getByRole("textbox", { name: /title/i }).first()).toHaveValue("Hook");
});

test("project list: search by name, and nothing found", async ({ page, request }) => {
  await seedProject(request, { name: "Searchable Seagull" });
  await seedProject(request, { name: "Searchable Pelican" });
  await goToProjects(page);
  const search = page.getByRole("searchbox", { name: "Search projects" });
  await search.fill("seagull");
  await expect(page.getByRole("article", { name: "Searchable Seagull" })).toBeVisible();
  await expect(page.getByRole("article", { name: "Searchable Pelican" })).toHaveCount(0);
  await search.fill("searchable");
  await expect(page.getByRole("article", { name: /^Searchable/ })).toHaveCount(2);
  await search.fill("no such film");
  await expect(page.getByText("No project matches no such film.")).toBeVisible();
  await search.press("Escape");
  await expect(search).toHaveValue("");
  await expect(page.getByRole("combobox", { name: "Sort" })).toHaveValue("edited");
});

test("project settings on Overview: pick 1080p and 50 fps", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Format Film", shots: 1 });
  await openProject(page, project.name, "Overview");
  const chips = shell(page).locator(".project-shell__chips");
  await expect(chips).toHaveText(/16:9\s*720p\s*24 fps/);
  // Collapsed once the project has shots; the summary still says what is set.
  await page.getByText("Project settings", { exact: true }).click();
  await page.getByRole("radio", { name: /^1080p/ }).click();
  await page.getByRole("radio", { name: "50 fps" }).click();
  await expect(chips).toHaveText(/16:9\s*1080p\s*50 fps/);
  // The voice is a picker over the real voice list, not a blank text box.
  await expect(page.getByText("No voice (automatic)")).toBeVisible();
  await page.getByRole("button", { name: "Change voice" }).click();
  await expect(page.getByRole("searchbox", { name: "Search voices" })).toBeVisible();
  await expect(page.getByRole("radio", { name: "No voice (automatic)" })).toBeChecked();
});

test("a project with no cast: add a shot and nothing asks for a character", async ({ page }) => {
  const dialog = await newProjectDialog(page);
  await dialog.getByRole("textbox").first().fill("Busy Street");
  await dialog.getByRole("button", { name: "Create" }).click();
  await expect(shell(page).getByRole("heading", { name: "Busy Street" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Add a cast (optional)" })).toBeVisible();
  await openProjectTab(page, "Storyboard");
  await page.getByRole("button", { name: "Add shot" }).first().click();
  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Make \d+ missing/ })).toBeEnabled();
  await openProjectTab(page, "Cast");
  await expect(page.getByText(/Optional\./)).toBeVisible();
  await expect(page.getByText(/None yet\./)).toHaveCount(2);
});

test("cast: a character with two pictures from the Image gallery and one from a shot still", async ({ page, request }) => {
  // Two saved images with real files, served as the Image gallery.
  const dir = path.join(RUN_HOME, "gallery");
  mkdirSync(dir, { recursive: true });
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    "base64",
  );
  const files = ["front.png", "side.png"].map((name) => {
    const file = path.join(dir, name);
    writeFileSync(file, png);
    return file;
  });
  const outputs = files.map((file, i) => ({
    artifactId: `cast-${i}`, modelId: "mock/image", modelName: "Mock image", prompt: `keeper ${i === 0 ? "front" : "side"}`,
    width: 1, height: 1, steps: 4, guidance: 1, seed: i, createdAt: new Date(Date.UTC(2026, 10, 2, 0, i)).toISOString(),
    durationSeconds: 1, previewUrl: "/none.png", imagePath: file, projectId: null, projectName: null,
  }));
  await page.route("**/api/images/outputs", (route) => route.fulfill({ json: { outputs } }));

  const project = await seedProject(request, { name: "Cast Film", shots: 1, takes: 1 });
  await openProject(page, project.name, "Cast");
  await page.getByRole("button", { name: "Add character" }).click();
  const card = page.locator(".cast-card").first();
  // A new card takes the focus on its name. Type the name, Tab, type the look, then go straight to Add picture:
  // nothing typed is lost (P0-1).
  await expect(card.getByRole("textbox", { name: "Name" })).toBeFocused();
  await page.keyboard.type("Keeper");
  await page.keyboard.press("Tab");
  await page.keyboard.type("Old man, wool hat");
  await card.getByRole("button", { name: "Add a picture of Keeper" }).click();
  await page.getByRole("dialog", { name: "Add a picture of Keeper" }).getByRole("button", { name: "Cancel" }).click();
  await expect(card.getByRole("textbox", { name: "What it looks like" })).toHaveValue("Old man, wool hat");
  await openProjectTab(page, "Overview");
  await openProjectTab(page, "Cast");
  await expect(card.getByRole("textbox", { name: "What it looks like" })).toHaveValue("Old man, wool hat");
  await expect(card.getByRole("textbox", { name: "Name" })).toHaveValue("Keeper");

  const addPicture = async (source: "Image gallery" | "From a shot", pick: string | RegExp) => {
    await card.getByRole("button", { name: "Add a picture of Keeper" }).click();
    const picker = page.getByRole("dialog", { name: "Add a picture of Keeper" });
    await picker.getByRole("tab", { name: source }).click();
    await picker.getByRole("button", { name: pick }).click();
    await expect(picker).toBeHidden();
  };
  await addPicture("Image gallery", "Use keeper front");
  await addPicture("Image gallery", "Use keeper side");
  await addPicture("From a shot", "Use Shot 1 · take 1");

  await expect(card.getByRole("button", { name: /^Remove picture/ })).toHaveCount(3);
  await expect(card.getByText("3 of 4")).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Cast/ })).toContainText("1");
  // A picture comes off again, and Undo puts it back.
  await card.getByRole("button", { name: "Remove picture 2" }).click();
  await expect(card.getByRole("button", { name: /^Remove picture/ })).toHaveCount(2);
  await shell(page).getByRole("button", { name: "Undo" }).click();
  await expect(card.getByRole("button", { name: /^Remove picture/ })).toHaveCount(3);
});

test("cast: pictures from a chosen file, a paste and a drop on the tab", async ({ page, request }) => {
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    "base64",
  );
  const project = await seedProject(request, { name: "Files Film" });
  await openProject(page, project.name, "Cast");
  await page.getByRole("button", { name: "Add place" }).click();
  const card = page.locator(".cast-card").first();
  await expect(card.getByRole("textbox", { name: "Name" })).toBeFocused();
  await page.keyboard.type("Harbour");
  const picker = page.getByRole("dialog", { name: "Add a picture of Harbour" });

  // Choose file: the native chooser's input, as the user's pick would fill it.
  await card.getByRole("button", { name: "Add a picture of Harbour" }).click();
  await expect(picker.getByRole("tab", { name: "This computer" })).toHaveAttribute("aria-selected", "true");
  await picker.locator('input[type="file"]').setInputFiles({ name: "harbour-front.png", mimeType: "image/png", buffer: png });
  await expect(picker).toBeHidden();
  await expect(card.getByRole("button", { name: /^Remove picture/ })).toHaveCount(1);

  // Paste: a picture pasted into the paste box.
  await card.getByRole("button", { name: "Add a picture of Harbour" }).click();
  const box = picker.getByRole("textbox", { name: "Paste a picture here" });
  await box.focus();
  await box.evaluate((el, b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const data = new DataTransfer();
    data.items.add(new File([bytes], "pasted.png", { type: "image/png" }));
    el.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
  }, png.toString("base64"));
  await expect(picker).toBeHidden();
  await expect(card.getByRole("button", { name: /^Remove picture/ })).toHaveCount(2);

  // Drop on the tab's empty space: a new character named after the file.
  await page.locator(".cast-tab__drop").evaluate((el, b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const data = new DataTransfer();
    data.items.add(new File([bytes], "old_lighthouse-keeper.png", { type: "image/png" }));
    for (const type of ["dragover", "drop"]) el.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
  }, png.toString("base64"));
  await expect(page.getByRole("article", { name: "old lighthouse keeper" })).toBeVisible();
  await expect(page.getByRole("article", { name: "old lighthouse keeper" }).getByRole("button", { name: /^Remove picture/ })).toHaveCount(1);
});

test("takes: pick another take, then lock the shot and the strip follows", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Takes Film", shots: 1, takes: 2 });
  await openProject(page, project.name, "Storyboard");

  await expect(page.getByText("1 shot · 4 s · 0 stale")).toBeVisible();
  const takes = page.getByRole("radiogroup", { name: "Takes" });
  await expect(takes.getByRole("radio")).toHaveCount(2);
  await expect(takes.getByRole("radio", { name: "Take 2 · Active take" })).toBeChecked();
  await takes.getByRole("radio", { name: "Take 1" }).click();
  await expect(takes.getByRole("radio", { name: "Take 1 · Active take" })).toBeChecked();
  await expect(takes.getByRole("radio", { name: "Take 2" })).not.toBeChecked();

  // The takes are stills with no clip, so the shot still needs its clip: Make missing counts it (P1-1), and the
  // card offers Make clip.
  await expect(page.getByText("1 shot needs work: 1 has a still and needs its clip")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Make 1 missing/ })).toBeEnabled();
  await expect(page.locator(".shot-card").getByRole("button", { name: "Make clip" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeEnabled();

  const lock = page.getByRole("button", { name: "Lock this shot" });
  await expect(lock).toHaveAttribute("aria-pressed", "false");
  await lock.click();
  const unlock = page.getByRole("button", { name: "Unlock this shot" });
  await expect(unlock).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Locked", { exact: true })).toBeVisible();
  // A locked shot is skipped, so there is nothing left for either Make button.
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Make missing", exact: true })).toBeDisabled();
  await unlock.click();
  await expect(page.getByRole("button", { name: "Make all unlocked" })).toBeEnabled();

  // The card's toggle says "New take", so it is not confused with Redo in the header.
  const newTake = page.locator(".shot-card").getByRole("button", { name: "New take" });
  await newTake.click();
  await expect(newTake).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("New take next run")).toBeVisible();
  // These takes are stills without a clip: they say so instead of "Done".
  await expect(page.locator(".shot-card__status")).toHaveText("Still only");
});

test("tabs: arrow keys move between the five tabs and the project reopens on its last tab", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Keys Film", shots: 1 });
  await openProject(page, project.name, "Overview");
  const tabs = page.getByRole("tablist", { name: "Project" });
  await expect(tabs.getByRole("tab")).toHaveText([/^Overview/, /^Cast/, /^Storyboard/, /^Mixer/, /^Export/]);
  await tabs.getByRole("tab", { name: "Overview" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(tabs.getByRole("tab", { name: /^Cast/ })).toHaveAttribute("aria-selected", "true");
  await expect(tabs.getByRole("tab", { name: /^Cast/ })).toBeFocused();
  await page.keyboard.press("End");
  await expect(tabs.getByRole("tab", { name: "Export" })).toHaveAttribute("aria-selected", "true");
  // Leave and come back through the sidebar: the project is still open, on Export.
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Dashboard/ }).click();
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Projects/ }).click();
  await expect(shell(page).getByRole("heading", { name: project.name })).toBeVisible();
  await expect(tabs.getByRole("tab", { name: "Export" })).toHaveAttribute("aria-selected", "true");
});

test("Mixer: captions panel without speech, and with a voice clip", async ({ page, request }) => {
  const silent = await seedProject(request, { name: "Silent Film", shots: 1 });
  await openProject(page, silent.name, "Mixer");
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
  // Export is its own tab now; the Mixer has no export panel.
  await expect(page.getByRole("button", { name: "Export MP4" })).toHaveCount(0);

  const spoken = await seedProject(request, { name: "Spoken Film", shots: 1, clonedVoice: true });
  await openProject(page, spoken.name, "Mixer");
  const withVoice = page.locator(".captions-panel");
  await withVoice.locator("summary").click();
  await expect(withVoice.getByText("Add speech to the voice lane")).toHaveCount(0);
  await expect(withVoice.getByRole("button", { name: "Make from voice" })).toBeEnabled();
});

test("Export tab: forced AI tag for a cloned voice, estimate strip, Film list and the bundle", async ({ page, request }) => {
  // A project with no cloned voice lets the AI tag be switched off; one with a cloned voice does not.
  const plain = await seedProject(request, { name: "Plain Film", shots: 1 });
  await openProject(page, plain.name, "Export");
  const exportPanel = page.getByRole("region", { name: "Export" });
  const tag = exportPanel.getByRole("switch", { name: "Add a machine-readable AI tag" });
  await expect(tag).toBeEnabled();
  await expect(tag).toBeChecked();
  await tag.click();
  await expect(tag).not.toBeChecked();
  await expect(exportPanel.getByText("This project uses a cloned voice")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Export bundle" })).toBeVisible();

  const cloned = await seedProject(request, { name: "Cloned Film", shots: 1, clonedVoice: true });
  await page.route(`**/api/storyboard/projects/${cloned.id}/films`, (route) =>
    route.fulfill({
      json: { films: [{ file: "exports/cloned-film.mp4", name: "cloned-film.mp4", bytes: 3_500_000, modifiedAt: 1_790_000_000 }] },
    }),
  );
  await openProject(page, cloned.name, "Export");
  const panel = page.getByRole("region", { name: "Export" });
  const forced = panel.getByRole("switch", { name: "Add a machine-readable AI tag" });
  await expect(forced).toBeChecked();
  await expect(forced).toBeDisabled();
  await expect(panel.getByText("This project uses a cloned voice, so the AI tag is always written.")).toBeVisible();

  await expect(panel.getByRole("radiogroup", { name: "Aspect ratio" })).toBeVisible();
  await expect(panel.getByRole("radiogroup", { name: "Shape" })).toHaveCount(0);

  // Upscale: Same size by default; 4K says it is upscaled from the project size and adds no detail.
  const upscale = panel.getByRole("radiogroup", { name: "Upscale" });
  await expect(upscale.getByRole("radio", { name: "Same size" })).toHaveAttribute("aria-checked", "true");
  await upscale.getByRole("radio", { name: "2160p (4K)" }).click();
  await expect(upscale.getByRole("radio", { name: "2160p (4K)" })).toHaveAttribute("aria-checked", "true");
  await expect(panel.getByText(/Upscaled from 1280 × 720 to 3840 × 2160 at export/)).toBeVisible();
  await upscale.getByRole("radio", { name: "Same size" }).click();
  await expect(panel.getByText(/Upscaled from/)).toHaveCount(0);
  await expect(panel.getByText("Estimate: unknown until the first run")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Before you export" })).toBeVisible();
  await expect(page.getByText("A cloned voice is used, so the AI tag is always written.")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Export MP4" })).toBeVisible();
  await expect(panel.getByRole("button", { name: /Export for an editor/ })).toBeVisible();

  const films = panel.locator(".film-list");
  await expect(films.getByRole("heading", { name: "Film" })).toBeVisible();
  await expect(films).toContainText("cloned-film.mp4");
  await expect(films).toContainText("3.3 MB");
  await expect(films.getByRole("button")).toHaveCount(1);
});

test("first-run walk: the guide ticks as the work happens, without ever needing a character", async ({ page, request }) => {
  // A project with two shots, the first with an imported still (as a gallery send would make it).
  const project = await seedProject(request, { name: "Walk Film", shots: 2, takes: 1 });
  await openProject(page, project.name, "Overview");
  const guide = page.locator(".overview-tab__next");
  // One shot has no take, one has a still but no clip: Make missing counts both and says why.
  await expect(guide.getByRole("heading", { name: "Make 2 shots" })).toBeVisible();
  await expect(guide).toContainText("1 has a still but no clip");
  const row = (name: string) => guide.locator(".overview-checklist__row", { hasText: name });
  await expect(row("Shots")).toHaveClass(/is-done/);
  await expect(row("Cast (optional)")).not.toHaveClass(/is-done/);
  await expect(row("A clip for every shot")).not.toHaveClass(/is-done/);

  // The Storyboard run bar agrees with the Overview.
  await openProjectTab(page, "Storyboard");
  const runBar = page.getByRole("region", { name: "Make shots" });
  await expect(runBar.getByRole("button", { name: /^Make 2 missing/ })).toBeVisible();
  await expect(runBar).toContainText("2 shots need work: 1 has a still and needs its clip");
  await expect(page.getByRole("group", { name: "Shot lengths" }).getByRole("button")).toHaveCount(2);

  // Use the still as a reference: the Cast gets an entry with that picture and the guide ticks Cast.
  await page.getByRole("button", { name: "More actions for shot 1" }).click();
  await page.getByRole("menuitem", { name: "Use still as reference" }).click();
  await expect(page.getByRole("tab", { name: /^Cast/ })).toContainText("1");
  await openProjectTab(page, "Cast");
  await expect(page.locator(".cast-card").getByRole("button", { name: /^Remove picture/ })).toHaveCount(1);
  await openProjectTab(page, "Overview");
  await expect(row("Cast (optional)")).toHaveClass(/is-done/);
});
