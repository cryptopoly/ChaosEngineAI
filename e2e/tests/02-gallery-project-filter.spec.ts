import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";

// Project dropdown in the Image gallery toolbar. Outputs are mocked at the API edge so no
// model or generated image is needed; nine projects so the search box appears (SEARCH_FROM = 8).

function output(i: number, project: { id: string; name: string } | null) {
  return {
    artifactId: `img-${i}`,
    modelId: "mock/model",
    modelName: "Mock model",
    prompt: `prompt ${i}`,
    width: 512,
    height: 512,
    steps: 4,
    guidance: 1,
    seed: i,
    createdAt: new Date(Date.UTC(2026, 10, 1, 0, i)).toISOString(),
    durationSeconds: 1,
    previewUrl: "/none.png",
    projectId: project?.id ?? null,
    projectName: project?.name ?? null,
  };
}

const PROJECT_NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India"];
const outputs = [
  ...PROJECT_NAMES.map((name, i) => output(i, { id: `p-${name.toLowerCase()}`, name })),
  output(20, null),
];

async function openGallery(page: Page) {
  await page.route("**/api/images/outputs", (route) => route.fulfill({ json: { outputs } }));
  await page.goto("/");
  // The backend's data dir is shared by the whole run, so the terms may already be accepted.
  await expect(page.locator("aside.sidebar")).toBeVisible();
  const agree = page.getByRole("dialog", { name: "Terms of use" }).getByRole("button", { name: "I agree" });
  if (await agree.isVisible({ timeout: 3000 }).catch(() => false)) await agree.click();
  await page.locator(".nav-group-header", { hasText: "Images" }).click();
  await page.locator(".nav-child-button", { hasText: "Gallery" }).click();
  await expect(page.getByRole("button", { name: /Project/ }).first()).toBeVisible();
}

test("opens with all projects, a no-project row and a search box", async ({ page }) => {
  await openGallery(page);
  const trigger = page.locator(".project-filter__button");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await trigger.click();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  const list = page.getByRole("listbox", { name: "Project" });
  await expect(list).toBeVisible();
  await expect(list.getByRole("option")).toHaveCount(2 + PROJECT_NAMES.length);
  await expect(list.getByRole("option", { name: /All projects/ })).toHaveAttribute("aria-selected", "true");
  await expect(list.getByRole("option", { name: /No project/ })).toBeVisible();
  await expect(page.getByRole("searchbox", { name: "Search projects" })).toBeFocused();
});

test("search narrows the list and Enter picks the first match", async ({ page }) => {
  await openGallery(page);
  const trigger = page.locator(".project-filter__button");
  await trigger.click();
  const search = page.getByRole("searchbox", { name: "Search projects" });
  await search.fill("ech");
  const list = page.getByRole("listbox", { name: "Project" });
  await expect(list.getByRole("option")).toHaveCount(1);
  await expect(list.getByRole("option", { name: /Echo/ })).toBeVisible();
  await search.fill("zzz");
  await expect(page.getByText("No project matches.")).toBeVisible();
  await search.fill("ech");
  await search.press("Enter");
  await expect(list).toBeHidden();
  await expect(trigger).toContainText("Echo");
  await expect(trigger).toBeFocused();
});

test("Escape closes the list and returns focus to the trigger", async ({ page }) => {
  await openGallery(page);
  const trigger = page.locator(".project-filter__button");
  await trigger.click();
  await expect(page.getByRole("listbox", { name: "Project" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("listbox", { name: "Project" })).toBeHidden();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
});

test("arrow keys move between rows from the search box and Enter on a row picks it", async ({ page }) => {
  await openGallery(page);
  const trigger = page.locator(".project-filter__button");
  await trigger.click();
  await page.keyboard.press("ArrowDown");
  const list = page.getByRole("listbox", { name: "Project" });
  await expect(list.getByRole("option").first()).toBeFocused();
  await page.keyboard.press("End");
  await expect(list.getByRole("option").last()).toBeFocused();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(list).toBeHidden();
  await expect(trigger).not.toContainText("All projects");
  await expect(trigger).toBeFocused();
});
