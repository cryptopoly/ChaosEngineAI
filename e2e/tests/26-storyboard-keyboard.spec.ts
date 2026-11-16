import { test, expect } from "./fixtures";
import type { APIRequestContext, Page } from "@playwright/test";
import { API_BASE, TOKEN, openProject } from "./helpers";

// FU-188 items 2 and 5, no model needed. The shot list has one roving Tab stop: Tab enters the current card and
// its controls, the other cards are an arrow key away, so 30 shots are about 10 Tab stops. The Plan tab shows a
// shot count suggestion from the brief's length.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Payload {
  project: { id: string; revision: number; shots: Array<{ id: string }> };
}

async function api(request: APIRequestContext, url: string, data: unknown): Promise<Payload> {
  const response = await request.post(`${API_BASE}${url}`, { headers, data });
  expect(response.ok(), `${url} -> ${response.status()} ${await response.text()}`).toBeTruthy();
  return (await response.json()) as Payload;
}

/** The sample film plus blank shots up to ``total``. */
async function longFilm(request: APIRequestContext, name: string, total: number): Promise<Payload> {
  let payload = await api(request, "/api/storyboard/sample", { name });
  while (payload.project.shots.length < total) {
    payload = await api(request, `/api/storyboard/projects/${payload.project.id}/ops`, {
      revision: payload.project.revision, op: "add_shot", args: { title: `Extra ${payload.project.shots.length + 1}` },
    });
  }
  return payload;
}

const shotCard = (page: Page, n: number) => page.locator("article.shot-card").nth(n - 1);
const focusedShotId = (page: Page) =>
  page.evaluate(() => (document.activeElement?.closest("[data-shot-id]") as HTMLElement | null)?.dataset.shotId ?? null);
const focusInList = (page: Page) => page.evaluate(() => Boolean(document.activeElement?.closest(".shot-list")));

test("30 shots are about 10 Tab stops; arrows move between cards, Tab enters the current card, Enter opens it", async ({ page, request }) => {
  test.setTimeout(90_000);
  const film = await longFilm(request, "Keyboard check", 30);
  await openProject(page, "Keyboard check", "Storyboard");
  await expect(page.locator("article.shot-card")).toHaveCount(30);

  // One card is a Tab stop, and only its controls; every other card's controls are out of the Tab order.
  await expect(page.locator("article.shot-card[tabindex='0']")).toHaveCount(1);
  const thirdTabbable = await shotCard(page, 3).evaluate((card) =>
    Array.from(card.querySelectorAll<HTMLElement>("button, input, select, summary, [tabindex]")).filter((el) => el.tabIndex >= 0).length,
  );
  expect(thirdTabbable).toBe(0);

  // Count the Tab stops from the first card to the first stop past the list.
  await shotCard(page, 1).focus();
  let stops = 1;
  for (; stops < 60; stops++) {
    await page.keyboard.press("Tab");
    if (!(await focusInList(page))) break;
  }
  expect(stops).toBeLessThanOrEqual(14);

  // Back to the cards: arrows move between them and the current card's controls follow.
  await shotCard(page, 1).focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await expect(shotCard(page, 3)).toBeFocused();
  await page.keyboard.press("Tab");
  expect(await focusedShotId(page)).toBe(film.project.shots[2].id);
  await expect(shotCard(page, 3).getByRole("textbox", { name: "Shot 3 title" })).toBeFocused();
  // Accessible names stay on the card.
  await expect(shotCard(page, 3)).toHaveAttribute("aria-label", /^Shot 3 · /);
  await page.keyboard.press("Shift+Tab");
  await expect(shotCard(page, 3)).toBeFocused();
  await page.keyboard.press("End");
  await expect(shotCard(page, 30)).toBeFocused();
  await page.keyboard.press("Home");
  await expect(shotCard(page, 1)).toBeFocused();

  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog").filter({ hasText: /Shot 1/ }).first()).toBeVisible();
});

test("the Plan tab suggests a shot count from the brief's length", async ({ page, request }) => {
  await api(request, "/api/storyboard/projects", { name: "Suggestion check" });
  await openProject(page, "Suggestion check");
  await page.getByRole("tablist", { name: "Project" }).getByRole("tab", { name: "Plan" }).click();
  await page.getByRole("textbox", { name: "Brief" }).fill("A 30 s ad for a small tea shop");
  await expect(page.getByText("An ad of 30 s is about 6 to 8 shots.")).toBeVisible();
  await expect(page.getByText("Suggestion:")).toBeVisible();
  await page.getByRole("button", { name: "Use 30 s" }).click();
  await expect(page.getByRole("button", { name: "30 s", exact: true })).toHaveAttribute("aria-pressed", "true");
});
