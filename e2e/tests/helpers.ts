import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { APIRequestContext, Page } from "@playwright/test";
import { expect } from "./fixtures";

/** Same values as playwright.config.ts; the backend is loopback-only with a dummy token. */
export const API_BASE = `http://127.0.0.1:${process.env.E2E_API_PORT ?? 8910}`;
export const TOKEN = "e2e-dummy-token";
/** The throwaway HOME the backend runs in; fixture files for seeded projects live under it. */
export const RUN_HOME = path.resolve(__dirname, "..", ".run", "home");
export const SHOTS_DIR = path.resolve(__dirname, "..", "screenshots");

/** Open the app and accept the terms if the shared backend has not had them accepted yet. */
export async function openApp(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.locator("aside.sidebar")).toBeVisible();
  const agree = page.getByRole("dialog", { name: "Terms of use" }).getByRole("button", { name: "I agree" });
  // isVisible() does not wait; on a cold start the dialog appears a moment after the sidebar.
  if (await agree.waitFor({ state: "visible", timeout: 5000 }).then(() => true, () => false)) await agree.click();
  await expect(page.getByRole("dialog", { name: "Terms of use" })).toBeHidden();
}

/** Click a sidebar group (when closed) and then one of its children. */
export async function openTab(page: Page, group: string, child: string): Promise<void> {
  const nav = page.getByRole("navigation", { name: "Primary" });
  const header = nav.locator(".nav-group-header", { hasText: group });
  if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
  await nav.locator(".nav-child-button", { hasText: new RegExp(`^${child}$`) }).click();
}

export async function goTo(page: Page, group: string, child: string): Promise<void> {
  await openApp(page);
  await openTab(page, group, child);
}

/** The one Projects entry (D-65): opens the project list, leaving a project that was still open. */
export async function goToProjects(page: Page): Promise<void> {
  await openApp(page);
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Projects/ }).click();
  const back = page.locator(".project-shell__back");
  if (await back.isVisible()) await back.click();
  await expect(page.getByRole("heading", { name: "Projects", exact: true }).last()).toBeVisible();
}

export type ProjectTab = "Overview" | "Cast" | "Storyboard" | "Mixer" | "Export";

/** Open a project from the list by its card, then one of its tabs. */
export async function openProject(page: Page, name: string, tab?: ProjectTab): Promise<void> {
  await goToProjects(page);
  await page.getByRole("button", { name: `Open ${name}`, exact: true }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name, exact: true })).toBeVisible();
  if (tab) await openProjectTab(page, tab);
}

export async function openProjectTab(page: Page, tab: ProjectTab): Promise<void> {
  const button = page.getByRole("tablist", { name: "Project" }).getByRole("tab", { name: new RegExp(`^${tab}`) });
  await button.click();
  await expect(button).toHaveAttribute("aria-selected", "true");
}

const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

function wav(seconds: number): Buffer {
  const rate = 16000;
  const n = rate * seconds;
  const data = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) data.writeInt16LE(Math.round(3000 * ((i % 80) - 40) / 40), i * 2);
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
  return Buffer.concat([head, data]);
}

/** Write a tiny still and a two-second tone into the run's HOME; returns their paths. */
export function writeFixtureMedia(): { still: string; tone: string } {
  mkdirSync(RUN_HOME, { recursive: true });
  const still = path.join(RUN_HOME, "e2e-still.png");
  const tone = path.join(RUN_HOME, "e2e-tone.wav");
  writeFileSync(still, PNG_1X1);
  writeFileSync(tone, wav(2));
  return { still, tone };
}

const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

interface ProjectPayload {
  project: { id: string; revision: number; shots: Array<{ id: string }> };
}

/**
 * Seed a Storyboard project through the backend's own API (no model needed): `shots` empty shots,
 * `takes` still takes on the first shot, and optionally a cloned-voice clip on the voice lane.
 */
export async function seedProject(
  request: APIRequestContext,
  opts: { name: string; shots?: number; takes?: number; clonedVoice?: boolean },
): Promise<{ id: string; name: string }> {
  const media = writeFixtureMedia();
  const post = async (url: string, data: unknown): Promise<ProjectPayload> => {
    const response = await request.post(`${API_BASE}${url}`, { headers, data });
    expect(response.ok(), `${url} -> ${response.status()}`).toBeTruthy();
    return (await response.json()) as ProjectPayload;
  };
  let payload = await post("/api/storyboard/projects", { name: opts.name, defaults: {} });
  const id = payload.project.id;
  const op = async (name: string, args: Record<string, unknown>) => {
    payload = await post(`/api/storyboard/projects/${id}/ops`, { revision: payload.project.revision, op: name, args });
  };
  for (let i = 0; i < (opts.shots ?? 0); i++) await op("add_shot", {});
  const first = payload.project.shots[0]?.id;
  for (let i = 0; i < (opts.takes ?? 0); i++) await op("import_take", { shotId: first, still: media.still });
  if (opts.clonedVoice) await op("add_audio", { kind: "voice", path: media.tone, clonedVoice: true, label: "Narration" });
  return { id, name: opts.name };
}
