import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, RUN_HOME, TOKEN, openApp, openProject, openProjectTab } from "./helpers";

// S-N (v0.10.0 QA pass), no model needed: the Overview's next-step card for a paused batch and for a slow fresh
// film, a Length chip clicked straight after typing the brief, a long project name in the page header, and the
// Settings > Disk clean-up rows.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Project {
  id: string;
  revision: number;
  shots: Array<{ id: string }>;
  plan?: { filmSeconds?: number | null; brief?: string } | null;
  batches: Array<{ id: string; state: string }>;
}

async function post(request: APIRequestContext, url: string, data: unknown): Promise<{ project: Project }> {
  const response = await request.post(`${API_BASE}${url}`, { headers, data });
  expect(response.ok(), `${url} -> ${response.status()} ${await response.text()}`).toBeTruthy();
  return (await response.json()) as { project: Project };
}

async function load(request: APIRequestContext, id: string): Promise<Project> {
  const response = await request.get(`${API_BASE}/api/storyboard/projects/${id}`, { headers });
  return ((await response.json()) as { project: Project }).project;
}

function projectFile(id: string): string {
  const hit = (readdirSync(RUN_HOME, { recursive: true }) as string[]).find((rel) => rel.endsWith(path.join(id, "project.json")));
  if (!hit) throw new Error(`project.json of ${id} not found under the run home`);
  return path.join(RUN_HOME, hit);
}

async function planned(request: APIRequestContext, name: string, shots: number): Promise<Project> {
  const { project } = await post(request, "/api/storyboard/projects", { name });
  const list = Array.from({ length: shots }, (_, i) => ({ title: `Shot ${i + 1}`, stillPrompt: `a harbour, frame ${i + 1}`, motionPrompt: "", seconds: 4 }));
  const applied = await post(request, `/api/storyboard/projects/${project.id}/ops`, {
    revision: project.revision, op: "apply_plan", args: { mode: "replace", shots: list },
  });
  return applied.project;
}

test("Overview: a paused batch shows Resume, never a new Make; Resume carries on", async ({ page, request }) => {
  const project = await planned(request, "Paused overview", 3);
  const file = projectFile(project.id);
  const data = JSON.parse(readFileSync(file, "utf8"));
  data.batches.push({
    id: "b_dead2801", what: "still", force: false, draft: false, state: "running", createdAt: Date.now() / 1000 - 60, finishedAt: null,
    results: data.shots.map((s: { id: string }, i: number) => ({ shotId: s.id, kind: "still", state: i === 0 ? "running" : "queued", takeId: null, error: null })),
  });
  writeFileSync(file, JSON.stringify(data));
  await openProject(page, "Paused overview", "Overview");
  const next = page.locator(".overview-tab__next");
  await expect(next.getByRole("heading", { name: "Paused" })).toBeVisible();
  await expect(next).toContainText("Paused when the app closed; Resume carries on.");
  await expect(next).toContainText("0 of 3 steps done, 3 left");
  await expect(next.getByRole("button", { name: /^Make/ })).toHaveCount(0);
  await next.getByRole("button", { name: "Resume" }).click();
  // No still model in the run home: the resumed steps fail at once, so the record leaves "paused".
  await expect.poll(async () => (await load(request, project.id)).batches.at(-1)?.state).not.toBe("paused");
});

test("Overview: Make drafts is the primary button on a fresh film whose still model is slow", async ({ page, request }) => {
  await planned(request, "Slow overview", 4);
  // The run home has no still model; the estimate is answered as for a 10-minute still.
  await page.route("**/api/storyboard/projects/*/estimate", (route) =>
    route.fulfill({
      json: {
        what: "both", shots: 4, stills: 4, clips: 4, seconds: 2600, stillSeconds: 2400, clipSeconds: 200, peakGb: 12, totalGb: 64,
        diskMb: null, basis: "estimated", missing: [], imageModel: "Slow One", videoModel: "Vid",
      },
    }),
  );
  await openProject(page, "Slow overview", "Overview");
  const next = page.locator(".overview-tab__next");
  await expect(next.getByRole("button", { name: "Make drafts" })).toHaveClass(/primary-button/);
  await expect(next.getByRole("button", { name: "Make 4 missing" })).toHaveClass(/secondary-button/);
  await expect(next.locator(".primary-button")).toHaveCount(1);
  await expect(next).toContainText("The still model is slow.");
});

test("Plan: a Length chip clicked straight after typing the brief is not lost", async ({ page, request }) => {
  const { project } = await post(request, "/api/storyboard/projects", { name: "Chip race" });
  await openProject(page, "Chip race", "Plan");
  const brief = page.getByLabel("Brief");
  await brief.click();
  await brief.fill("A one-minute advert for a small tea shop.");
  // No blur first: the click's mousedown blurs the brief, which saves it, and the chip must still take the click.
  const chip = page.getByRole("group", { name: "Length" }).getByRole("button", { name: "30 s", exact: true });
  // A human click: the button is held about 150 ms, long enough for the brief's save to start between down and up.
  const box = await chip.boundingBox();
  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(150);
  await page.mouse.up();
  await expect(chip).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => (await load(request, project.id)).plan?.filmSeconds).toBe(30);
  expect((await load(request, project.id)).plan?.brief).toBe("A one-minute advert for a small tea shop.");
});

test("A long project name keeps to two lines in the page header at phone width", async ({ page, request }) => {
  const name = "Ein sehr langer Projektname über einen Leuchtturm am Ende der Welt, wo der Wind nie aufhört und die Möwen schreien, 灯台の物語";
  await post(request, "/api/storyboard/projects", { name });
  await page.setViewportSize({ width: 375, height: 812 });
  await openProject(page, name);
  const title = page.locator(".workspace-header__project h2");
  await expect(title).toHaveAttribute("title", name);
  const box = await title.boundingBox();
  const lineHeight = await title.evaluate((el) => parseFloat(getComputedStyle(el).lineHeight) || parseFloat(getComputedStyle(el).fontSize) * 1.2);
  expect(box!.height).toBeLessThanOrEqual(lineHeight * 2 + 2);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test("Settings > Disk: clean-up rows put the checkbox beside its text, and the panels keep their own height", async ({ page, request }) => {
  await post(request, "/api/storyboard/projects", { name: "Clean-up row" });
  await openApp(page);
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Settings/ }).click();
  await page.getByRole("tab", { name: /Disk/ }).first().click();
  const row = page.locator("label.cleanup-row", { hasText: "Clean-up row" });
  await expect(row).toBeVisible();
  const input = await row.locator("input").boundingBox();
  const text = await row.locator("strong").boundingBox();
  expect(input!.x + input!.width).toBeLessThanOrEqual(text!.x); // beside, not above
  expect(Math.abs(input!.y - text!.y)).toBeLessThan(12);
  const selectAll = await page.locator(".cleanup-select-all").first().locator("input").boundingBox();
  const group = await page.locator(".cleanup-group").first().boundingBox();
  expect(selectAll!.x - group!.x).toBeLessThan(40); // at the start of the line, not centred
  // Disk usage keeps its natural height: Clean up starts right below it, not halfway down the page.
  const disk = await page.locator(".settings-disk-stack > .panel").first().boundingBox();
  const scan = await page.getByRole("button", { name: "Scan disk usage" }).boundingBox();
  expect(disk!.y + disk!.height - (scan!.y + scan!.height)).toBeLessThan(60);
});

test("Board: Alt+arrow reaches a second card while one is selected", async ({ page, request }) => {
  await planned(request, "Board keys", 4);
  await openProject(page, "Board keys", "Board");
  const current = page.locator(".board-card[tabindex='0']");
  await current.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("Space");
  const first = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.cardId ?? null);
  await page.keyboard.press("Alt+ArrowRight");
  const second = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.cardId ?? null);
  expect(second).not.toBe(first);
  await page.keyboard.press("Space");
  await expect(page.locator(".board-card[aria-pressed='true']")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Order these 2" })).toBeVisible();
  await openProjectTab(page, "Overview");
});

test("Storyboard run bar: the Make buttons stay inside the estimate box in the side column", async ({ page, request }) => {
  await planned(request, "Run bar fit", 19);
  await page.route("**/api/storyboard/projects/*/estimate", (route) =>
    route.fulfill({
      json: {
        what: "both", shots: 19, stills: 19, clips: 19, seconds: 13920, stillSeconds: 9000, clipSeconds: 4920, peakGb: 17, totalGb: 51.8,
        diskMb: null, basis: "estimated", missing: [], imageModel: "Slow One", videoModel: "Vid",
      },
    }),
  );
  await page.setViewportSize({ width: 1280, height: 720 });
  await openProject(page, "Run bar fit", "Storyboard");
  const box = page.locator(".run-bar .run-estimate");
  const make = page.locator(".run-bar").getByRole("button", { name: /^Make 19 missing/ });
  await expect(make).toBeVisible();
  const outer = await box.boundingBox();
  const inner = await make.boundingBox();
  expect(inner!.x + inner!.width).toBeLessThanOrEqual(outer!.x + outer!.width + 0.5);
});
