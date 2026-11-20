// Screenshot recipe for the v0.10.0 UI/UX review B, pass 2: the states that matter (see capture-baseline.spec.ts
// for the two worlds and how to run). Not a test of behaviour: every scene is wrapped so one failing selector does
// not stop the walk, and a scene that could not be reached is printed, not hidden.
//   REVIEW_WORLD=B scenes read the PC's model folder (read only, every mutating request answered 503 in the browser).
//   REVIEW_WORLD=A scenes use mocked music, library and gallery answers plus projects seeded in the throwaway HOME.
import { mkdirSync } from "node:fs";
import path from "node:path";
import { test } from "./fixtures";
import type { Page, Route } from "@playwright/test";
import { goTo, goToProjects, openProjectTab, openTab, seedProject, showExperimentalTabs, writeFixtureMedia, API_BASE, TOKEN } from "./helpers";

const WORLD = process.env.REVIEW_WORLD ?? "A";
const OUT = process.env.REVIEW_OUT ?? path.resolve(__dirname, "..", "..", "docs", "review", "ui-ux-v0100-b", "shots");
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 375, height: 812 };
const GB = 1024 ** 3;
const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

test.setTimeout(900_000);

async function shoot(page: Page, name: string, phone: boolean, grow = true) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(700);
  const file = path.join(OUT, `${phone ? "m" : "d"}-${WORLD}-${name}.png`);
  const size = phone ? PHONE : DESKTOP;
  if (grow) {
    const tallest = await page.evaluate(() => {
      let max = 0;
      for (const el of document.querySelectorAll<HTMLElement>("main, main *")) {
        const style = getComputedStyle(el);
        if (["auto", "scroll"].includes(style.overflowY) && el.scrollHeight > el.clientHeight + 1) {
          max = Math.max(max, el.scrollHeight + el.getBoundingClientRect().top / 0.9 + 40);
        }
      }
      return max;
    });
    const height = Math.min(Math.max(size.height, Math.ceil(tallest * 0.9)), 9000);
    if (height !== size.height) {
      await page.setViewportSize({ width: size.width, height });
      await page.waitForTimeout(400);
    }
  }
  await page.screenshot({ path: file, fullPage: false });
  await page.setViewportSize(size);
}

async function scene(name: string, body: () => Promise<void>) {
  try {
    await body();
  } catch (error) {
    console.log(`SCENE NOT REACHED: ${name}: ${String(error).split("\n")[0].slice(0, 220)}`);
  }
}

async function readOnly(page: Page) {
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const url = new URL(request.url()).pathname;
    const harmless = /\/api\/(terms|settings|auth|busy-advice|music\/estimate|chat\/studio\/price|images\/preview|video\/preview|images\/estimate|video\/estimate|licence-gate$)/.test(url);
    if (request.method() === "GET" || request.method() === "OPTIONS" || harmless) return route.fallback();
    return route.fulfill({ status: 503, json: { detail: "Review run: read-only." } });
  });
}

async function expandRow(page: Page, id: string) {
  const row = page.locator(`[data-model-id="${id}"]`);
  await row.scrollIntoViewIfNeeded();
  await row.locator(".models-card__toggle").first().click();
  await page.waitForTimeout(300);
}

// ---------------------------------------------------------------------------------------------------------------
// World B: the rows that are really on this PC
// ---------------------------------------------------------------------------------------------------------------

const H3_KEY = "minimax-h3";
const H3_SENTENCE =
  "I am not in the European Union, the United Kingdom, the Republic of Korea or the United States, and I will not use this model or what it makes there. I accept the MiniMax H3 Community License, including its Acceptable Use Policy.";

async function mockH3Gate(page: Page) {
  // The real backend answers the gate itself when H3 is released; with H3_RELEASED off it is mocked as spec 24 does,
  // but the licence text comes from the backend's own status when it has one.
  const real = await page.request.get(`${API_BASE}/api/licence-gate`, { headers }).then((r) => r.json()).catch(() => ({ models: [] }));
  const own = (real.models ?? []).find((m: { key: string }) => m.key === H3_KEY);
  const model = own ?? {
    key: H3_KEY, name: "MiniMax H3", kinds: ["video"],
    excludedTerritories: ["European Union", "United Kingdom", "Republic of Korea", "United States"],
    licenceName: "MiniMax H3 Community License",
    licenceUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/LICENSE",
    liveLicenceUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
    summaryLines: ["Licensed only outside the European Union, the United Kingdom, the Republic of Korea and the United States."],
    confirmSentence: H3_SENTENCE,
    note: "ChaosEngineAI does not include or distribute MiniMax H3. You download it from Hugging Face yourself, and you are the licensee. If you let other people generate with it (for example through remote connections or the agent bridge), you are offering a hosted service and the licence's terms for that apply to you. It runs on the stable-diffusion.cpp runtime (sd), which release builds may not include. Not tested by the maintainers.",
    confirmed: false, confirmedAt: null,
  };
  console.log(`H3 gate text: ${own ? "the backend's own" : "mocked as in spec 24 (H3_RELEASED is off)"}`);
  await page.route("**/api/licence-gate", (route: Route) => route.fulfill({ json: { models: [model] } }));
  await page.route("**/api/video/catalog", async (route: Route) => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ json: { ...body, territoryLimited: [H3_KEY] } });
  });
}

function worldB(phone: boolean) {
  test(`world B states (${phone ? "phone" : "desktop"})`, async ({ page }) => {
    await readOnly(page);
    await mockH3Gate(page);
    await goTo(page, "Images", "Models");
    await page.waitForTimeout(2500);

    await scene("image row details", async () => {
      await expandRow(page, "black-forest-labs/flux-3-image-klein");
      await expandRow(page, "black-forest-labs/FLUX.1-schnell-nunchaku-int4");
      await expandRow(page, "OpenMOSS-Team/MOSS-Image-SFT");
      await shoot(page, "image-models-rows-open", phone);
    });
    await scene("image models: installed filter", async () => {
      await page.getByRole("button", { name: /^Installed\s*\d+$/ }).click();
      await shoot(page, "image-models-installed", phone);
      await page.getByRole("button", { name: /^Needs attention\s*\d+$/ }).click();
      await shoot(page, "image-models-attention", phone);
    });

    await openTab(page, "Images", "Studio");
    await page.waitForTimeout(1500);
    const imageModel = page.locator("select").filter({ has: page.locator("option", { hasText: /Klein|FLUX/ }) }).first();
    for (const [label, name] of [
      [/^FLUX 3 Image Klein \(diffusers\)/, "image-studio-flux3"],
      [/^Qwen-Image-2\.1 Turbo/, "image-studio-qwen-turbo"],
      [/^MOSS-Image SFT/, "image-studio-moss"],
    ] as const) {
      await scene(name, async () => {
        const option = imageModel.locator("option").filter({ hasText: label }).first();
        await imageModel.selectOption((await option.getAttribute("value")) ?? "");
        await page.waitForTimeout(900);
        if (name === "image-studio-flux3") {
          await page.locator("summary", { hasText: "Options" }).first().click();
          await page.locator("summary", { hasText: "Expert" }).first().click();
        }
        await shoot(page, name, phone);
      });
    }
    await scene("image generate refused (409 not downloaded)", async () => {
      const option = imageModel.locator("option").filter({ hasText: /Nunchaku INT4/ }).first();
      await imageModel.selectOption((await option.getAttribute("value")) ?? "");
      await page.route("**/api/images/generate", (route) =>
        route.fulfill({
          status: 409,
          json: { detail: "FLUX.1 Schnell · Nunchaku INT4 (CUDA) is not downloaded. Download it in Image Studio first: it needs the 4-bit pack mit-han-lab/svdq-int4-flux.1-schnell (6.7 GB)." },
        }),
      );
      await page.getByPlaceholder(/Moody cinematic alleyway/).fill("A lighthouse on dark rocks at dusk, long exposure, 35mm photo");
      await page.getByRole("button", { name: "Generate image" }).click();
      await page.waitForTimeout(2500);
      await shoot(page, "image-studio-409", phone);
      const close = page.getByRole("button", { name: /^(Close|Dismiss|OK)$/ }).first();
      if (await close.isVisible().catch(() => false)) await close.click();
    });

    await openTab(page, "Video", "Models");
    await page.waitForTimeout(1500);
    await scene("H3 territory notice and licence screen", async () => {
      const notice = page.locator("details.territory-gate-notice");
      await notice.locator("summary").click();
      await shoot(page, "video-models-territory-notice", phone, false);
      await notice.getByRole("button", { name: "Review" }).click();
      await page.waitForTimeout(500);
      await shoot(page, "video-h3-licence-screen", phone, false);
      await page.getByRole("dialog").getByRole("checkbox").check();
      await shoot(page, "video-h3-licence-ticked", phone, false);
      await page.keyboard.press("Escape");
    });
    await scene("video row details", async () => {
      await expandRow(page, "Lightricks/LTX-2.5-Diffusers~nf4");
      await expandRow(page, "OzzyGT/LTX-2.3-Distilled-bnb-nf4");
      await expandRow(page, "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers");
      await expandRow(page, "Wan-AI/Wan2.2-TI2V-5B-Diffusers");
      await shoot(page, "video-models-rows-open", phone);
    });

    await openTab(page, "Video", "Studio");
    await page.waitForTimeout(1500);
    const videoModel = page.locator("select").filter({ has: page.locator("option", { hasText: /LTX|Wan/ }) }).first();
    for (const [label, name] of [
      [/^LTX-2\.3 · distilled 4-bit/, "video-studio-ltx23"],
      [/^LTX-2\.5 · distilled 4-bit/, "video-studio-ltx25"],
      [/^Kandinsky 6\.0 Lite/, "video-studio-kandinsky"],
    ] as const) {
      await scene(name, async () => {
        const option = videoModel.locator("option").filter({ hasText: label }).first();
        await videoModel.selectOption((await option.getAttribute("value")) ?? "");
        await page.waitForTimeout(900);
        if (name === "video-studio-ltx23") {
          await page.getByPlaceholder(/cinematic drone shot/).fill("A woman sings into a studio microphone, warm light, slow push in");
          await page.locator("summary", { hasText: "Options" }).first().click();
          await page.locator("summary", { hasText: "Expert" }).first().click();
        }
        await shoot(page, name, phone);
      });
    }
  });
}

// ---------------------------------------------------------------------------------------------------------------
// World A: mocked music, libraries and galleries; real Storyboard projects in the throwaway HOME
// ---------------------------------------------------------------------------------------------------------------

async function musicInstalled(page: Page) {
  await page.route("**/api/busy-advice*", (route) =>
    route.fulfill({ json: { advice: { level: "ok", message: "", jobs: [], combinedPeakGb: 10, totalGb: 64, availableGb: 40, footprintGb: 10 } } }),
  );
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      for (const variant of family.variants) Object.assign(variant, { availableLocally: true, packInstalled: true, ready: true, textEncoderReady: true, licenceAccepted: true });
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime, { installed: true, unavailableReason: null });
    Object.assign(body.runtime.sound, { installed: true, unavailableReason: null });
    await route.fulfill({ response, json: body });
  });
}

const LONG = "A very long title that a person typed without thinking about how much room a card has, with a second clause, and then a third one for good measure";

function track(i: number, kind: "track" | "sound", over: Record<string, unknown> = {}) {
  return {
    id: `m-${i}`, kind, title: `${kind === "sound" ? "Glass breaking on a stone floor" : "Harbour lights"} ${i}`, caption: "warm analog synths, slow tempo, female vocal, rain on a tin roof",
    lyrics: kind === "track" ? "[verse]\nThe boats come home at nine\n[chorus]\nHarbour lights, harbour lights" : "", tags: [],
    createdAt: Date.UTC(2026, 10, 19, 9, i) / 1000, updatedAt: Date.UTC(2026, 10, 19, 9, i) / 1000, audio: "flac",
    modelName: kind === "sound" ? "Stable Audio 3 Small SFX" : "ACE-Step 1.5 XL turbo (large)", seed: 1000 + i, durationS: kind === "sound" ? 5 : 180, sizeBytes: 42_000_000,
    projectId: null, projectName: null, ...over,
  };
}

function still(i: number, over: Record<string, unknown> = {}) {
  return {
    artifactId: `still-${i}`, modelId: "black-forest-labs/FLUX.2-klein-4B", modelName: "FLUX.2 Klein 4B", prompt: `A lighthouse on dark rocks at dusk, take ${i}`,
    width: 1024, height: 1024, steps: 4, guidance: 1, seed: i, createdAt: new Date(Date.UTC(2026, 10, 19, 9, i)).toISOString(), durationSeconds: 6.8,
    previewUrl: "/none.png", projectId: null, projectName: null, ...over,
  };
}

function clip(i: number, over: Record<string, unknown> = {}) {
  return {
    artifactId: `clip-${i}`, modelId: "Lightricks/LTX-2.5-Diffusers~nf4", modelName: "LTX-2.5 · distilled 4-bit on load (diffusers)", prompt: `A beam sweeps over the sea, take ${i}`,
    width: 768, height: 512, numFrames: 97, fps: 24, steps: 8, guidance: 1, seed: i, createdAt: new Date(Date.UTC(2026, 10, 19, 10, i)).toISOString(),
    durationSeconds: 61.3, clipDurationSeconds: 4.04, projectId: null, projectName: null, ...over,
  };
}

function worldA(phone: boolean) {
  test(`world A states (${phone ? "phone" : "desktop"})`, async ({ page, request }) => {
    await showExperimentalTabs(page);
    await musicInstalled(page);
    await page.route(/\/api\/music\/library(\?.*)?$/, (route) =>
      route.fulfill({ json: { items: [track(1, "track"), track(2, "sound"), track(3, "track", { title: LONG, caption: LONG }), track(4, "sound")], diskBytes: 168_000_000 } }),
    );
    await page.route("**/api/images/outputs", (route) =>
      route.fulfill({ json: { outputs: [still(1), still(2, { prompt: LONG }), still(3, { modelName: "FLUX 3 Image Klein (diffusers)", modelId: "black-forest-labs/flux-3-image-klein" })] } }),
    );
    await page.route("**/api/video/outputs", (route) => route.fulfill({ json: { outputs: [clip(1), clip(2, { prompt: LONG }), clip(3, { modelName: "Kandinsky 6.0 Lite · distilled" })] } }));
    await page.route("**/api/video/outputs/*/file", (route) => route.fulfill({ status: 404, body: "" }));

    const film = await seedProject(request, { name: phone ? "Harbour Film (phone)" : "Harbour Film", shots: 3, takes: 2, clonedVoice: true });
    const media = writeFixtureMedia();

    await goTo(page, "Music and Audio", "Studio");
    await page.waitForTimeout(2000);
    await scene("music studio form", async () => {
      await shoot(page, "music-studio-form", phone);
      const caption = page.locator("textarea").first();
      await caption.fill(`${LONG}. ${LONG}.`);
      await page.waitForTimeout(800);
      await shoot(page, "music-studio-long-text", phone);
    });
    await scene("music generate error", async () => {
      await page.route("**/api/music/generate", (route) =>
        route.fulfill({ status: 409, json: { detail: "The GPU has 0.2 GB free and this track needs about 1.0 GB. Another program may be using the GPU. Close it, or pick ACE-Step 1.5 turbo." } }),
      );
      await page.locator("textarea").first().fill("warm analog synths, slow tempo");
      await page.getByRole("button", { name: /^Make (track|song|instrumental)/ }).first().click();
      await page.waitForTimeout(2000);
      await shoot(page, "music-studio-error", phone);
      await page.unroute("**/api/music/generate");
    });
    await scene("music copy now panel", async () => {
      await page.route("**/api/music/generate", (route) =>
        route.fulfill({
          status: 409,
          json: { detail: { message: "Setting up the music model needs 6 weight file(s) copied: this PC did not allow a link.", needsCopy: true, fileCount: 6, totalBytes: 14.3 * GB, source: "E:\\AI Models\\hf\\snapshots", target: "C:\\Users\\me\\.chaosengine\\music\\checkpoints", sourceDrive: "E:", targetDrive: "C:" } },
        }),
      );
      await page.route("**/api/music/weights/copy/status", (route) =>
        route.fulfill({ json: { phase: "running", running: true, bytesDone: 5.1 * GB, totalBytes: 14.3 * GB, fileCount: 6, filesDone: 2, currentFile: "acestep-v15-xl-turbo/model.safetensors", error: null, etaSeconds: 95 } }),
      );
      await page.route("**/api/music/weights/copy", (route) => route.fulfill({ json: { started: true } }));
      await page.getByRole("button", { name: /^Make (track|song|instrumental)/ }).first().click();
      await page.waitForTimeout(1500);
      await shoot(page, "music-copy-offer", phone);
      await page.getByRole("button", { name: /Copy now/ }).click();
      await page.waitForTimeout(1800);
      await shoot(page, "music-copy-running", phone);
    });
    await scene("sounds studio form", async () => {
      await page.getByRole("button", { name: "Sounds", exact: true }).or(page.getByRole("tab", { name: "Sounds", exact: true })).first().click();
      await page.waitForTimeout(1500);
      await shoot(page, "sounds-studio-form", phone);
    });

    await openTab(page, "Music and Audio", "Models");
    await page.waitForTimeout(1500);
    await scene("music models: all ready", async () => {
      await expandRow(page, "ACE-Step/Ace-Step1.5");
      await shoot(page, "music-models-ready", phone);
    });
    await page.unroute("**/api/music/catalog");
    await page.unroute("**/api/music/runtime");
    await scene("music models: downloading, paused, failed", async () => {
      await page.route("**/api/music/download/status", (route) =>
        route.fulfill({
          json: {
            downloads: [
              { repo: "ACE-Step/Ace-Step1.5", state: "downloading", progress: 0.42, downloadedGb: 4.2, totalGb: 10.1, error: null },
              { repo: "ACE-Step/acestep-v15-xl-turbo", state: "cancelled", progress: 0.25, downloadedGb: 5.0, totalGb: 19.9, error: null },
              { repo: "stabilityai/stable-audio-open-small", state: "failed", progress: 0.1, downloadedGb: 0.3, totalGb: 2.6, error: "403 Client Error: this account has not accepted the licence for stabilityai/stable-audio-open-small." },
            ],
          },
        }),
      );
      await openTab(page, "Music and Audio", "Library");
      await openTab(page, "Music and Audio", "Models");
      await page.waitForTimeout(2500);
      await shoot(page, "music-models-downloading", phone);
    });
    await scene("music licence screen", async () => {
      const row = page.locator('[data-model-id="stabilityai/stable-audio-3-medium"]');
      await row.getByRole("button", { name: "Accept licence" }).click();
      await page.waitForTimeout(600);
      await shoot(page, "music-licence-screen", phone, false);
      await page.keyboard.press("Escape");
    });

    await openTab(page, "Music and Audio", "Library");
    await page.waitForTimeout(1500);
    await scene("music library with items", async () => {
      await shoot(page, "music-library-items", phone);
      const menu = page.getByRole("button", { name: /Storyboard/ }).first();
      await menu.click();
      await page.waitForTimeout(500);
      await shoot(page, "music-library-send-to-storyboard", phone, false);
      await page.keyboard.press("Escape");
    });

    await openTab(page, "Images", "Gallery");
    await page.waitForTimeout(1500);
    await scene("image gallery with items", async () => shoot(page, "image-gallery-items", phone));
    await openTab(page, "Video", "Gallery");
    await page.waitForTimeout(1500);
    await scene("video gallery with items", async () => shoot(page, "video-gallery-items", phone));

    // Projects (Storyboard)
    await goToProjects(page);
    await page.waitForTimeout(1200);
    await scene("projects list", async () => shoot(page, "projects-list", phone));
    await scene("new project dialog", async () => {
      await page.getByRole("button", { name: /^New project/ }).first().click();
      await page.waitForTimeout(500);
      await shoot(page, "projects-new-dialog", phone, false);
      await page.keyboard.press("Escape");
    });
    await scene("open project", async () => {
      await page.getByRole("button", { name: `Open ${film.name}`, exact: true }).click();
      await page.waitForTimeout(1200);
    });
    for (const tab of ["Overview", "Plan", "Cast", "Storyboard", "Board", "Mixer", "Export"] as const) {
      await scene(`project ${tab}`, async () => {
        await openProjectTab(page, tab);
        await page.waitForTimeout(1500);
        await shoot(page, `project-${tab.toLowerCase()}`, phone);
      });
    }
    await scene("shot editor with a vocal clip", async () => {
      await openProjectTab(page, "Storyboard");
      await page.waitForTimeout(800);
      const edit = page.getByRole("button", { name: /^(Edit shot|Edit)/ }).first();
      await edit.click();
      await page.waitForTimeout(800);
      const vocal = page.getByLabel(/Vocal for lip sync/).first();
      if (await vocal.isVisible().catch(() => false)) {
        await vocal.fill(media.tone);
        await vocal.blur();
        await page.waitForTimeout(1000);
      }
      await shoot(page, "project-shot-editor", phone);
      await page.keyboard.press("Escape");
    });

    await openTab(page, "Tools", "Fine-Tuning").catch(() => undefined);
  });
}

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });
    if (WORLD === "B") worldB(phone);
    else worldA(phone);
  });
}
