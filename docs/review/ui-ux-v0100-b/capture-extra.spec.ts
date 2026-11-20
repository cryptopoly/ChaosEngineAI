// Screenshot recipe for the v0.10.0 UI/UX review B, pass 3 (world A only): the scenes pass 2 did not reach
// (Music Studio refusal, the Copy now panel, Send to storyboard, the shot editor with a vocal) plus two
// measurements printed to the log (where the Run benchmark button sits; whether the H3 buttons are in view on a phone).
// Run as capture-states.spec.ts, copied to e2e/tests/99-uiux-b-extra.spec.ts, with REVIEW_WORLD=A.
import { mkdirSync } from "node:fs";
import path from "node:path";
import { test } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openProject, openTab, seedProject, writeFixtureMedia } from "./helpers";

const OUT = process.env.REVIEW_OUT ?? path.resolve(__dirname, "..", "..", "docs", "review", "ui-ux-v0100-b", "shots");
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 375, height: 812 };
const GB = 1024 ** 3;

test.setTimeout(600_000);

async function shoot(page: Page, name: string, phone: boolean) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, `${phone ? "m" : "d"}-A-${name}.png`), fullPage: false });
}

async function scene(name: string, body: () => Promise<void>) {
  try {
    await body();
  } catch (error) {
    console.log(`SCENE NOT REACHED: ${name}: ${String(error).split("\n")[0].slice(0, 220)}`);
  }
}

async function musicInstalled(page: Page) {
  await page.route("**/api/busy-advice*", (route) =>
    route.fulfill({ json: { advice: { level: "ok", message: "", jobs: [], combinedPeakGb: 10, totalGb: 64, availableGb: 40, footprintGb: 10 } } }),
  );
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      for (const variant of family.variants) Object.assign(variant, { availableLocally: true, packInstalled: true, ready: true, textEncoderReady: true });
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime, { installed: true, unavailableReason: null });
    await route.fulfill({ response, json: body });
  });
}

function track(i: number, kind: "track" | "sound") {
  return {
    id: `m-${i}`, kind, title: `${kind === "sound" ? "Glass breaking" : "Harbour lights"} ${i}`, caption: "warm analog synths, slow tempo", lyrics: "", tags: [],
    createdAt: Date.UTC(2026, 10, 19, 9, i) / 1000, updatedAt: Date.UTC(2026, 10, 19, 9, i) / 1000, audio: "flac", modelName: "ACE-Step 1.5 turbo", seed: i,
    durationS: kind === "sound" ? 5 : 60, sizeBytes: 12_000_000, projectId: null, projectName: null,
  };
}

async function startTrack(page: Page) {
  await page.getByRole("group", { name: "Kind of track" }).getByRole("button", { name: "Instrumental" }).click();
  await page.getByRole("textbox", { name: /Describe the music/ }).fill("warm lo-fi piano, 85 bpm");
  await page.locator(".run-estimate").getByRole("button", { name: "Make track" }).click();
}

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });

    test(`extra scenes (${phone ? "phone" : "desktop"})`, async ({ page, request }) => {
      await musicInstalled(page);
      await page.route(/\/api\/music\/library(\?.*)?$/, (route) => route.fulfill({ json: { items: [track(1, "track"), track(2, "sound")], diskBytes: 24_000_000 } }));
      const film = await seedProject(request, { name: phone ? "Vocal Film (phone)" : "Vocal Film", shots: 2, takes: 1 });
      const media = writeFixtureMedia();

      await goTo(page, "Music and Audio", "Studio");
      await page.waitForTimeout(1500);
      await scene("music refusal", async () => {
        await page.route("**/api/music/generate", (route) =>
          route.fulfill({ status: 409, json: { detail: "The GPU has 0.2 GB free and this track needs about 1.0 GB. Another program may be using the GPU. Close it, or pick ACE-Step 1.5 turbo." } }),
        );
        await startTrack(page);
        await page.waitForTimeout(1500);
        await shoot(page, "music-studio-error", phone);
        await page.unroute("**/api/music/generate");
      });
      await scene("copy now", async () => {
        await page.route("**/api/music/generate", (route) =>
          route.fulfill({
            status: 409,
            json: { detail: { message: "Setting up the music model needs 6 weight file(s) copied: this PC did not allow a link.", needsCopy: true, fileCount: 6, totalBytes: 14.3 * GB, source: "E:\\AI Models\\hf\\snapshots", target: "C:\\Users\\me\\.chaosengine\\music\\checkpoints", sourceDrive: "E:", targetDrive: "C:" } },
          }),
        );
        const status = { phase: "running", running: true, bytesDone: 5.1 * GB, totalBytes: 14.3 * GB, fileCount: 6, filesDone: 2, currentFile: "acestep-v15-xl-turbo/model.safetensors", error: null, etaSeconds: 95 };
        await page.route("**/api/music/weights/copy/status", (route) => route.fulfill({ json: status }));
        await page.route("**/api/music/weights/copy", (route) => route.fulfill({ json: status }));
        await page.locator(".run-estimate").getByRole("button", { name: "Make track" }).click();
        await page.waitForTimeout(1500);
        await shoot(page, "music-copy-offer", phone);
        await page.getByRole("button", { name: "Copy now" }).click();
        await page.waitForTimeout(1800);
        await shoot(page, "music-copy-running", phone);
      });

      await openTab(page, "Music and Audio", "Library");
      await page.waitForTimeout(1200);
      await scene("send to storyboard menu", async () => {
        await page.getByRole("button", { name: /Send to storyboard/i }).first().click();
        await page.waitForTimeout(600);
        await shoot(page, "music-library-send-to-storyboard", phone);
        await page.keyboard.press("Escape");
      });

      await scene("shot editor with a vocal", async () => {
        await openProject(page, film.name, "Storyboard");
        await page.locator(".shot-card").first().focus();
        await page.keyboard.press("Enter");
        const editor = page.locator(".shot-editor");
        await editor.waitFor();
        await shoot(page, "project-shot-editor", phone);
        const vocal = editor.locator("fieldset.shot-editor__vocal");
        await vocal.scrollIntoViewIfNeeded();
        await vocal.getByRole("textbox", { name: "Vocal file (WAV or FLAC)" }).fill(media.tone);
        await vocal.getByRole("button", { name: "Attach vocal" }).click();
        await page.waitForTimeout(1200);
        await vocal.scrollIntoViewIfNeeded();
        await shoot(page, "project-shot-editor-vocal", phone);
        await page.keyboard.press("Escape");
      });
      await scene("export tab", async () => {
        await page.getByRole("tablist", { name: "Project" }).getByRole("tab", { name: /^Export/ }).click();
        await page.waitForTimeout(1500);
        await shoot(page, "project-export-view", phone);
      });

      await scene("benchmark run button position", async () => {
        await openTab(page, "Benchmarks", "Run");
        await page.waitForTimeout(1200);
        const facts = await page.evaluate(() => {
          const button = document.querySelector(".benchmark-run-btn");
          const box = button?.getBoundingClientRect();
          const scroller = button?.closest(".scrollable-panel-content") as HTMLElement | null;
          const sb = scroller?.getBoundingClientRect();
          return {
            viewport: [window.innerWidth, window.innerHeight],
            buttonTopBottom: box ? [Math.round(box.top), Math.round(box.bottom)] : null,
            scrollBoxTopBottom: sb ? [Math.round(sb.top), Math.round(sb.bottom)] : null,
            scrollBox: scroller ? [scroller.clientHeight, scroller.scrollHeight] : null,
          };
        });
        console.log(`BENCH RUN BUTTON (${phone ? "phone" : "desktop"}): ${JSON.stringify(facts)}`);
        await shoot(page, "bench-run-viewport", phone);
      });
    });
  });
}
