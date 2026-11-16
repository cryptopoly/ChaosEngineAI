import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext, Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, RUN_HOME, TOKEN, openProject } from "./helpers";

// S-O items 2 to 6, no model needed: the Plan tab's shot rows as one Tab stop, a song longer than the film, the
// Length chip against the song, the Overview's Make drafts tooltip, and no preview probe on a project never rendered.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Project {
  id: string;
  revision: number;
  plan?: { filmSeconds?: number | null } | null;
  shots: Array<{ id: string }>;
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

async function planned(request: APIRequestContext, name: string, shots: number): Promise<Project> {
  const { project } = await post(request, "/api/storyboard/projects", { name });
  const list = Array.from({ length: shots }, (_, i) => ({ title: `Shot ${i + 1}`, stillPrompt: `a harbour, frame ${i + 1}`, motionPrompt: "", seconds: 4 }));
  return (await post(request, `/api/storyboard/projects/${project.id}/ops`, { revision: project.revision, op: "apply_plan", args: { mode: "replace", shots: list } })).project;
}

/** A 30 s mono WAV of a soft tone. Stdlib only. */
function writeSong(file: string): string {
  const rate = 8000;
  const n = rate * 30;
  const data = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) data.writeInt16LE(Math.round(0.2 * Math.sin((2 * Math.PI * 220 * i) / rate) * 32000), i * 2);
  const head = Buffer.alloc(44);
  head.write("RIFF", 0);
  head.writeUInt32LE(36 + data.length, 4);
  head.write("WAVEfmt ", 8);
  head.writeUInt32LE(16, 16);
  head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22);
  head.writeUInt32LE(rate, 24);
  head.writeUInt32LE(rate * 2, 28);
  head.writeUInt16LE(2, 32);
  head.writeUInt16LE(16, 34);
  head.write("data", 36);
  head.writeUInt32LE(data.length, 40);
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, Buffer.concat([head, data]));
  return file;
}

const row = (page: Page, n: number) => page.locator("li.plan-shot").nth(n - 1);

test("Plan: the shot rows are one Tab stop; arrows move between rows, Enter goes into the title", async ({ page, request }) => {
  test.setTimeout(60_000);
  await planned(request, "Plan keys", 19);
  await openProject(page, "Plan keys", "Plan");
  await expect(page.locator("li.plan-shot")).toHaveCount(19);
  await expect(page.locator("li.plan-shot[tabindex='0']")).toHaveCount(1);
  const thirdTabbable = await row(page, 3).evaluate((li) =>
    Array.from(li.querySelectorAll<HTMLElement>("button, input, textarea, select, [tabindex]")).filter((el) => el.tabIndex >= 0).length,
  );
  expect(thirdTabbable).toBe(0);

  // From the first row to the first stop past the list: the row and its five controls, not 5 stops per shot.
  await row(page, 1).focus();
  let stops = 1;
  for (; stops < 120; stops++) {
    await page.keyboard.press("Tab");
    const inList = await page.evaluate(() => Boolean(document.activeElement?.closest(".plan-shots__list")));
    if (!inList) break;
  }
  expect(stops).toBeLessThanOrEqual(8);

  await row(page, 1).focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await expect(row(page, 3)).toBeFocused();
  await expect(row(page, 3)).toHaveAttribute("aria-label", "Shot 3 · Shot 3");
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Title of shot 3")).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(row(page, 3)).toBeFocused();
  await page.keyboard.press("End");
  await expect(row(page, 19)).toBeFocused();
  await page.keyboard.press("Home");
  await expect(row(page, 1)).toBeFocused();
});

test("Song longer than the film: the note and Make a cut plan beside Time the lyrics; Use the song's length", async ({ page, request }) => {
  test.setTimeout(90_000);
  const film = await planned(request, "Song longer", 2);
  const length = await post(request, `/api/storyboard/projects/${film.id}/ops`, { revision: film.revision, op: "set_plan", args: { filmSeconds: 15 } });
  const song = writeSong(path.join(RUN_HOME, "fixtures", "song-30.wav"));
  await post(request, `/api/storyboard/projects/${film.id}/song`, { revision: length.project.revision, path: song });
  await openProject(page, "Song longer", "Plan");

  const longer = page.locator(".song-lyrics__longer");
  await expect(longer).toContainText("The song runs 0:30; the shots add up to 0:08.");
  await longer.getByRole("button", { name: "Make a cut plan" }).click();
  await expect(page.locator("#cut-plan-title")).toBeVisible({ timeout: 30_000 });

  // The Length chip says 15 s, the song 30 s.
  await expect(page.getByText("The song runs 30 s.")).toBeVisible();
  await page.getByRole("button", { name: "Use the song's length" }).click();
  await expect(page.getByRole("group", { name: "Length" }).getByRole("button", { name: "30 s", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => (await load(request, film.id)).plan?.filmSeconds).toBe(30);
  await expect(page.getByRole("button", { name: "Use the song's length" })).toHaveCount(0);
});

test("Overview: the Make drafts tooltip names the draft model and the time per draft", async ({ page, request }) => {
  await planned(request, "Draft tooltip", 4);
  const draft = { shots: 4, stills: 4, clips: 4, seconds: 36, stillSeconds: 36, clipSeconds: 200, basis: "estimated", imageModel: "Iris-3B Turbo", videoModel: "Vid", width: 640, height: 360 };
  await page.route("**/api/storyboard/projects/*/estimate", (route) =>
    route.fulfill({
      json: {
        what: "both", shots: 4, stills: 4, clips: 4, seconds: 2600, stillSeconds: 2400, clipSeconds: 200, peakGb: 12, totalGb: 64,
        diskMb: null, basis: "estimated", missing: [], imageModel: "Slow One", videoModel: "Vid",
        passes: { draft, final: { ...draft, imageModel: "Slow One", width: 1280, height: 720, seconds: 2600, stillSeconds: 2400 } },
      },
    }),
  );
  await openProject(page, "Draft tooltip", "Overview");
  const make = page.locator(".overview-tab__next").getByRole("button", { name: "Make drafts" });
  await expect(make).toHaveAttribute("title", "Quick stills with Iris-3B Turbo at 640 x 360. Mark the shots you like with Keep, then run the final pass. About 9 s per draft.");
  // The run bar says the same.
  await page.getByRole("tablist", { name: "Project" }).getByRole("tab", { name: /^Storyboard/ }).click();
  await expect(page.locator(".run-bar__passes").getByRole("button", { name: /^Make drafts/ })).toHaveAttribute("title", /About 9 s per draft\.$/);
});

test("Mixer: a project never rendered asks for no preview file", async ({ page, request }) => {
  await planned(request, "No preview", 2);
  const asked: string[] = [];
  page.on("request", (req) => {
    if (req.url().includes("preview.mp4")) asked.push(req.url());
  });
  const failed: number[] = [];
  page.on("response", (res) => {
    if (res.url().includes("/api/storyboard/") && res.status() === 404) failed.push(res.status());
  });
  await openProject(page, "No preview", "Mixer");
  await expect(page.locator(".mixer-tab, .mixer").first()).toBeVisible();
  await page.waitForTimeout(1500);
  expect(asked).toEqual([]);
  expect(failed).toEqual([]);
});
