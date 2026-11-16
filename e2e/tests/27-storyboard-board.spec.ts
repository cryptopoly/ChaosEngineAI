import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext, Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, RUN_HOME, TOKEN, openProject, openProjectTab } from "./helpers";

// Item 8 UI (S-M2), no model needed: the freestyle Board tab (cards, drag, keyboard, Order these, Undo), the song
// panel on Plan (a synthetic WAV, sections ruler, cut plan review and apply), the song ruler on the board, and the
// paused batch bar after a restart (written into the project file as a dead process leaves it).

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Project {
  id: string;
  revision: number;
  shots: Array<{ id: string; title: string }>;
  board?: { pos: Record<string, { x: number; y: number }> };
  song?: { sections: unknown[] } | null;
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

/** A 30 s mono WAV: a quiet tone for 8 s, then clicks at 120 BPM over a louder tone. Stdlib only. */
function writeSong(file: string): string {
  const rate = 11025;
  const seconds = 30;
  const n = rate * seconds;
  const data = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    let v = t < 8 ? 0.08 * Math.sin(2 * Math.PI * 220 * t) : 0.25 * Math.sin(2 * Math.PI * 330 * t);
    const sinceBeat = t >= 8 ? (t - 8) % 0.5 : 1;
    if (sinceBeat < 0.03) v += 0.8 * Math.exp(-sinceBeat / 0.005) * Math.sin(2 * Math.PI * 1000 * sinceBeat);
    data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, v)) * 32000), i * 2);
  }
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

function projectFile(id: string): string {
  const hit = (readdirSync(RUN_HOME, { recursive: true }) as string[]).find((rel) => rel.endsWith(path.join(id, "project.json")));
  if (!hit) throw new Error(`project.json of ${id} not found under the run home`);
  return path.join(RUN_HOME, hit);
}

const card = (page: Page, title: string) => page.locator(".board-card", { hasText: title });

test("Board: cards, drag, keyboard moves and Order these with Undo", async ({ page, request }) => {
  const { project } = await post(request, "/api/storyboard/sample", { name: "Board film" });
  await openProject(page, "Board film", "Board");
  await expect(page.locator(".board-card")).toHaveCount(4);
  await expect(page.locator(".board-card[tabindex='0']")).toHaveCount(1);

  // Drag the last shot to the left of the first: Order all puts it first.
  const last = project.shots[3];
  const first = await card(page, project.shots[0].title).boundingBox();
  const moving = await card(page, last.title).boundingBox();
  expect(first && moving).toBeTruthy();
  await page.mouse.move(moving!.x + 40, moving!.y + 30);
  await page.mouse.down();
  await page.mouse.move(first!.x - 160, first!.y + 30, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => Boolean((await load(request, project.id)).board?.pos[last.id])).toBe(true);
  await page.getByRole("button", { name: "Order all shots" }).click();
  await expect.poll(async () => (await load(request, project.id)).shots[0].id).toBe(last.id);
  await page.getByRole("button", { name: "Undo" }).click();
  await expect.poll(async () => (await load(request, project.id)).shots[0].id).toBe(project.shots[0].id);

  // Keyboard: arrows move the focus between cards; Space selects; Shift+arrow moves the selected card.
  const current = page.locator(".board-card[tabindex='0']");
  await current.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".board-card:focus")).toHaveAttribute("tabindex", "0");
  const focusedId = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.cardId ?? null);
  expect(focusedId).toBeTruthy();
  const before = (await load(request, project.id)).board?.pos[focusedId!];
  await page.keyboard.press("Space");
  await expect(page.locator(`[data-card-id="${focusedId}"]`)).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("Shift+ArrowDown");
  await page.keyboard.press("ArrowRight");
  await expect.poll(async () => (await load(request, project.id)).board?.pos[focusedId!]?.y).toBe((before?.y ?? 0) + 64);
  expect((await load(request, project.id)).board?.pos[focusedId!]?.x).toBe((before?.x ?? 0) + 8);
  await page.keyboard.press("Escape");
  await expect(page.locator(".board-card[aria-pressed='true']")).toHaveCount(0);

  // S-N: with a card selected, Alt+arrow still goes to the next card (plain arrows move the selected one), so two
  // cards can be selected from the keyboard and "Order these 2" is reachable.
  const moved = (await load(request, project.id)).board?.pos[focusedId!];
  await page.keyboard.press("Space");
  await page.keyboard.press("Alt+ArrowLeft");
  const secondId = await page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.cardId ?? null);
  expect(secondId).toBeTruthy();
  expect(secondId).not.toBe(focusedId);
  await page.keyboard.press("Space");
  await expect(page.locator(".board-card[aria-pressed='true']")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Order these 2" })).toBeVisible();
  expect((await load(request, project.id)).board?.pos[focusedId!]).toEqual(moved);
  await page.keyboard.press("Escape");

  // One shot selected: its card chat; none: the board (direction) chat.
  await card(page, project.shots[1].title).click();
  await expect(page.locator(".board-card-chat")).toBeVisible();
  await page.getByRole("button", { name: "Board chat" }).click();
  await expect(page.getByRole("heading", { name: "Direction" })).toBeVisible();
});

test("Song: attach a WAV, rename a section, review and apply a cut plan; the board shows the ruler", async ({ page, request }) => {
  test.setTimeout(90_000);
  const { project } = await post(request, "/api/storyboard/projects", { name: "Song film" });
  const song = writeSong(path.join(RUN_HOME, "fixtures", "song-27.wav"));
  await openProject(page, "Song film", "Plan");
  const panel = page.locator(".song-panel");
  await panel.getByRole("textbox", { name: /Or an audio file/ }).fill(song);
  await panel.getByRole("button", { name: "Attach" }).last().click();
  await expect(panel.getByRole("list", { name: "Song sections" })).toBeVisible({ timeout: 60_000 });
  await expect(panel).toContainText("0:30");

  // Rename the first section.
  await panel.locator("button.song-ruler__label").first().click();
  await panel.locator(".song-ruler__input").fill("verse");
  await panel.locator(".song-ruler__input").press("Enter");
  await expect(panel.locator("button.song-ruler__label").first()).toHaveText("verse");

  // The panel's own button (a song longer than the shots also offers one beside Time the lyrics, S-O).
  await panel.locator(".song-panel__actions").getByRole("button", { name: "Make a cut plan" }).click();
  const review = page.locator(".cut-plan");
  await expect(review).toContainText("Cut plan:");
  const use = review.getByRole("button", { name: /^Use (these \d+ shots|this shot)$/ });
  const n = Number((await use.textContent())?.match(/\d+/)?.[0] ?? 1);
  await use.click();
  await expect.poll(async () => (await load(request, project.id)).shots.length).toBe(n);

  await openProjectTab(page, "Board");
  await expect(page.locator(".board-viewport__song").getByRole("list", { name: "Song sections" })).toBeVisible();
  await expect(page.locator(".board-card")).toHaveCount(n);
});

test("Paused batch: the reason after a restart, steps left, and Cancel", async ({ page, request }) => {
  const { project } = await post(request, "/api/storyboard/sample", { name: "Paused film" });
  const file = projectFile(project.id);
  const data = JSON.parse(readFileSync(file, "utf8"));
  data.batches.push({
    id: "b_dead2701", what: "still", force: false, draft: false, state: "running", createdAt: Date.now() / 1000 - 60, finishedAt: null,
    results: data.shots.slice(0, 3).map((s: { id: string }, i: number) => ({ shotId: s.id, kind: "still", state: i === 0 ? "running" : "queued", takeId: null, error: null })),
  });
  writeFileSync(file, JSON.stringify(data));
  await openProject(page, "Paused film", "Storyboard");
  const bar = page.locator(".batch-bar--paused");
  await expect(bar).toContainText("Paused when the app closed; Resume carries on.");
  await expect(bar).toContainText("0 of 3 steps done, 3 left");
  await expect(bar.getByRole("button", { name: "Resume" })).toBeEnabled();
  await bar.getByRole("button", { name: "Cancel batch" }).click();
  await expect(bar).toHaveCount(0);
  await expect.poll(async () => (await load(request, project.id)).batches.at(-1)?.state).toBe("cancelled");
});
