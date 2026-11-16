import { writeFileSync } from "node:fs";
import path from "node:path";
import { deflateSync } from "node:zlib";
import { test, expect } from "./fixtures";
import type { APIRequestContext, Page } from "@playwright/test";
import { API_BASE, RUN_HOME, TOKEN, goToProjects, openProject, openProjectTab, writeFixtureMedia } from "./helpers";

// Storyboard 0.10 screens that need no model: the Plan tab with no chat model, the sample film (template and
// API), Keep for final with Undo, the take Compare dialog, the Play film (rough cut) dialog and its keys, and the
// Cast tab's style card and brand kit. The run's backend has a fresh HOME, so no chat, image or video model is
// installed; the sample film brings its own stills and clips.

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface Payload {
  project: { id: string; revision: number; name: string; shots: Array<{ id: string; activeTake: string | null }> };
}

async function api(request: APIRequestContext, url: string, data: unknown): Promise<Payload> {
  const response = await request.post(`${API_BASE}${url}`, { headers, data });
  expect(response.ok(), `${url} -> ${response.status()} ${await response.text()}`).toBeTruthy();
  return (await response.json()) as Payload;
}

/** The sample film copied in under ``name``, with a second take (an imported still) on its first shot. */
async function sampleWithTwoTakes(request: APIRequestContext, name: string): Promise<Payload> {
  let payload = await api(request, "/api/storyboard/sample", { name });
  const { still } = writeFixtureMedia();
  payload = await api(request, `/api/storyboard/projects/${payload.project.id}/ops`, {
    revision: payload.project.revision, op: "import_take", args: { shotId: payload.project.shots[0].id, still },
  });
  return payload;
}

/** A 48 x 32 PNG in three colour bands, so the style palette has colours to read. */
function bandsPng(): string {
  const bands = [[200, 120, 40], [240, 200, 120], [40, 90, 160]];
  const rows: number[] = [];
  for (let y = 0; y < 32; y++) {
    rows.push(0);
    for (let x = 0; x < 48; x++) rows.push(...bands[Math.floor((x * bands.length) / 48)]);
  }
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf: Buffer) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (kind: string, data: Buffer) => {
    const body = Buffer.concat([Buffer.from(kind, "ascii"), data]);
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const sum = Buffer.alloc(4);
    sum.writeUInt32BE(crc(body));
    return Buffer.concat([len, body, sum]);
  };
  const head = Buffer.alloc(13);
  head.writeUInt32BE(48, 0);
  head.writeUInt32BE(32, 4);
  head.set([8, 2, 0, 0, 0], 8);
  const file = path.join(RUN_HOME, "e2e-bands.png");
  writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", head),
    chunk("IDAT", deflateSync(Buffer.from(rows))), chunk("IEND", Buffer.alloc(0))]));
  return file;
}

const shotCard = (page: Page, n: number) => page.locator("article.shot-card").nth(n - 1);

test("Plan tab with no chat model installed: the callout, and Propose waits for a brief and a model", async ({ page, request }) => {
  const created = await api(request, "/api/storyboard/projects", { name: "Plan without a model" });
  expect(created.project.shots).toHaveLength(0);
  await openProject(page, "Plan without a model");
  await page.getByRole("tablist", { name: "Project" }).getByRole("tab", { name: "Plan" }).click();
  await expect(page.getByText(/No chat model is installed yet/)).toBeVisible();
  await expect(page.getByText(/No shots yet\. Write a brief and propose shots/)).toBeVisible();
  const propose = page.getByRole("button", { name: /^Propose/ });
  await expect(propose).toBeDisabled();
  await page.getByRole("textbox", { name: "Brief" }).fill("A calm advert for a tea shop · 京都の茶屋");
  await expect(propose).toBeDisabled(); // still no model
  await expect(page.getByText("Load a chat model first.")).toBeVisible();
});

test("the Sample film template copies a finished four-shot project", async ({ page }) => {
  await goToProjects(page);
  await page.getByRole("main").getByRole("button", { name: "New project" }).first().click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await dialog.getByRole("textbox").first().fill("Sample from template");
  await dialog.getByRole("radio", { name: /^Sample film/ }).click({ timeout: 5000 });
  await dialog.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name: "Sample from template" })).toBeVisible();
  await openProjectTab(page, "Storyboard");
  await expect(page.locator("article.shot-card")).toHaveCount(4);
  await expect(page.getByText(/^4 shots · /)).toBeVisible();
});

test("Keep for final, and Undo takes it back", async ({ page, request }) => {
  await api(request, "/api/storyboard/sample", { name: "Keep check" });
  await openProject(page, "Keep check", "Storyboard");
  const keep = shotCard(page, 1).getByRole("button", { name: "Keep for final" });
  await expect(keep).toHaveAttribute("aria-pressed", "false");
  await keep.click();
  await expect(keep).toHaveAttribute("aria-pressed", "true");
  await page.locator(".project-shell").getByRole("button", { name: "Undo" }).click();
  await expect(keep).toHaveAttribute("aria-pressed", "false");
});

test("Compare takes: side by side, Use this one switches the take, Escape closes", async ({ page, request }) => {
  await sampleWithTwoTakes(request, "Compare check");
  await openProject(page, "Compare check", "Storyboard");
  await shotCard(page, 1).getByRole("button", { name: "Compare", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: /Compare takes · Shot 1/ });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("table")).toBeVisible();
  const use = dialog.getByRole("button", { name: "Use this one" });
  await expect(use).toHaveCount(1); // the other side is the take in use
  await use.click();
  await expect(dialog.getByRole("status").or(dialog.getByText(/is now the shot's take/))).toContainText(/is now the shot's take/);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
});

test("Play film opens the rough cut; its keys work and Escape closes it", async ({ page, request }) => {
  const sample = await sampleWithTwoTakes(request, "Rough cut check");
  await api(request, `/api/storyboard/projects/${sample.project.id}/ops`, {
    revision: sample.project.revision, op: "add_caption", args: { start: 0.2, end: 1.6, text: "灯台が光る" },
  });
  await openProject(page, "Rough cut check", "Storyboard");
  const open = page.getByRole("button", { name: "Play film" }).first();
  await open.click();
  const dialog = page.getByRole("dialog", { name: "Rough cut" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(/4 shots/);
  const captions = dialog.locator("button.rough-cut__cc");
  await expect(captions).toBeEnabled();
  const before = await captions.getAttribute("aria-pressed");
  await dialog.press("c");
  await expect(captions).not.toHaveAttribute("aria-pressed", before ?? "");
  const scrubber = dialog.getByRole("slider", { name: "Position in the film" });
  await dialog.press("End");
  await expect(scrubber).not.toHaveAttribute("aria-valuenow", "0");
  await dialog.press("Home");
  await expect(scrubber).toHaveAttribute("aria-valuenow", "0");
  await dialog.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(open).toBeFocused();
});

test("Cast: a style card with its palette, and the brand kit's caption place", async ({ page, request }) => {
  let payload = await api(request, "/api/storyboard/projects", { name: "Brand check" });
  writeFixtureMedia();
  payload = await api(request, `/api/storyboard/projects/${payload.project.id}/ops`, {
    revision: payload.project.revision, op: "add_reference", args: { kind: "style", name: "House look", images: [bandsPng()] },
  });
  await openProject(page, "Brand check", "Cast");
  await expect(page.getByRole("article", { name: "House look" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Colours read from the pictures" }).first().getByRole("button", { name: /^Remove colour #/ })).toHaveCount(3);
  const kit = page.getByRole("region", { name: "Brand kit" });
  await expect(kit).toBeVisible();
  const top = kit.getByRole("button", { name: "Top", exact: true });
  await top.click();
  await expect(top).toHaveAttribute("aria-pressed", "true");
  await expect(kit.getByRole("button", { name: "Bottom", exact: true })).toHaveAttribute("aria-pressed", "false");
  await page.locator(".project-shell").getByRole("button", { name: "Undo" }).click();
  await expect(kit.getByRole("button", { name: "Bottom", exact: true })).toHaveAttribute("aria-pressed", "true");
});
