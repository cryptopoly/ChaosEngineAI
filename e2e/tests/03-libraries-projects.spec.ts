import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, seedProject } from "./helpers";
import { mockVoiceApi } from "./voiceMocks";

// Video, Music and Voice libraries with Storyboard projects: the Project dropdown, filtering, the
// "From project X" badge, the project header and "Open in Storyboard". Library data is mocked at the
// API edge (nothing is generated); the Storyboard project itself is seeded through the backend.

const PROJECT = "Harbour Film";
let projectId = "";

test.beforeAll(async ({ request }) => {
  projectId = (await seedProject(request, { name: PROJECT, shots: 1 })).id;
});

const created = (i: number) => new Date(Date.UTC(2026, 10, 1, 0, i)).toISOString();

function clip(i: number, tagged: boolean) {
  return {
    artifactId: `clip-${i}`,
    modelId: "mock/video",
    modelName: "Mock video",
    prompt: `clip prompt ${i}`,
    width: 640,
    height: 360,
    numFrames: 24,
    fps: 24,
    steps: 4,
    guidance: 1,
    seed: i,
    createdAt: created(i),
    durationSeconds: 2,
    clipDurationSeconds: 1,
    projectId: tagged ? projectId : null,
    projectName: tagged ? PROJECT : null,
  };
}

function still(i: number, tagged: boolean) {
  return {
    artifactId: `still-${i}`,
    modelId: "mock/image",
    modelName: "Mock image",
    prompt: `still ${i}`,
    width: 512,
    height: 512,
    steps: 4,
    guidance: 1,
    seed: i,
    createdAt: created(i),
    durationSeconds: 1,
    previewUrl: "/none.png",
    projectId: tagged ? projectId : null,
    projectName: tagged ? PROJECT : null,
  };
}

async function pickProject(page: Page, name: string | RegExp) {
  await page.locator(".project-filter__button").click();
  await page.getByRole("listbox", { name: "Project" }).getByRole("option", { name }).click();
}

test("Video gallery: project filter, badge, header counts and Open in Storyboard", async ({ page }) => {
  await page.route("**/api/video/outputs", (route) =>
    route.fulfill({ json: { outputs: [clip(1, true), clip(2, true), clip(3, false)] } }),
  );
  await page.route("**/api/video/outputs/*/file", (route) => route.fulfill({ status: 404, body: "" }));
  await page.route("**/api/images/outputs", (route) => route.fulfill({ json: { outputs: [still(1, true)] } }));
  await goTo(page, "Video", "Gallery");

  const cards = page.locator(".image-gallery-grid > *");
  await expect(cards).toHaveCount(3);
  await expect(page.locator(".project-header")).toHaveCount(0);
  await expect(page.locator(".project-card-badge")).toHaveCount(2);
  await expect(page.locator(".project-card-badge").first()).toHaveText(`From project ${PROJECT}`);

  await pickProject(page, new RegExp(PROJECT));
  await expect(cards).toHaveCount(2);
  const header = page.locator(".project-header");
  await expect(header).toContainText(PROJECT);
  await expect(header).toContainText("2 clips");
  await expect(header).toContainText("1 still");

  await pickProject(page, /No project/);
  await expect(cards).toHaveCount(1);
  await expect(page.locator(".project-card-badge")).toHaveCount(0);
  await expect(page.locator(".project-header")).toHaveCount(0);

  await pickProject(page, /All projects/);
  await expect(cards).toHaveCount(3);

  await pickProject(page, new RegExp(PROJECT));
  await header.getByRole("button", { name: "Open in Storyboard" }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name: PROJECT })).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Storyboard/ })).toHaveAttribute("aria-selected", "true");
});

function track(i: number, kind: "track" | "sound", tagged: boolean) {
  return {
    id: `m-${i}`,
    kind,
    title: `${kind === "sound" ? "Sound" : "Track"} ${i}`,
    caption: `caption ${i}`,
    lyrics: "",
    tags: [],
    createdAt: Date.UTC(2026, 10, 1, 0, i) / 1000,
    updatedAt: Date.UTC(2026, 10, 1, 0, i) / 1000,
    audio: "wav",
    modelName: "Mock music",
    seed: i,
    durationS: 12,
    sizeBytes: 1_000_000,
    projectId: tagged ? projectId : null,
    projectName: tagged ? PROJECT : null,
  };
}

test("Music library: project filter, badge, counts and Open in Storyboard", async ({ page }) => {
  const items = [track(1, "track", true), track(2, "sound", true), track(3, "track", false)];
  await page.route(/\/api\/music\/library(\?.*)?$/, (route) =>
    route.fulfill({ json: { items, count: items.length, diskBytes: 3_000_000 } }),
  );
  await goTo(page, "Music and Audio", "Library");

  const cards = page.locator(".music-library__item");
  await expect(cards).toHaveCount(3);
  await expect(page.getByText(`From project ${PROJECT}`)).toHaveCount(2);

  await pickProject(page, new RegExp(PROJECT));
  await expect(cards).toHaveCount(2);
  const header = page.locator(".project-header");
  await expect(header).toContainText(PROJECT);
  await expect(header).toContainText("1 track");
  await expect(header).toContainText("1 sound");

  await pickProject(page, /No project/);
  await expect(cards).toHaveCount(1);
  await expect(page.getByText("From project")).toHaveCount(0);

  await pickProject(page, new RegExp(PROJECT));
  await header.getByRole("button", { name: "Open in Storyboard" }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name: PROJECT })).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Storyboard/ })).toHaveAttribute("aria-selected", "true");
});

test("Music and Audio library: All / Music / Sound effects filter, counts and selection", async ({ page }) => {
  const items = [track(1, "track", false), track(2, "sound", false), track(3, "sound", false)];
  await page.route(/\/api\/music\/library(\?.*)?$/, (route) =>
    route.fulfill({ json: { items, count: items.length, diskBytes: 3_000_000 } }),
  );
  await goTo(page, "Music and Audio", "Library");

  const cards = page.locator(".music-library__item");
  const kinds = page.getByRole("group", { name: "Show" });
  await expect(cards).toHaveCount(3);
  await expect(kinds.getByRole("button", { name: "All (3)" })).toHaveAttribute("aria-pressed", "true");
  await expect(kinds.getByRole("button", { name: "Music (1)" })).toBeVisible();

  await kinds.getByRole("button", { name: "Sound effects (2)" }).click();
  await expect(cards).toHaveCount(2);
  await expect(page.getByText("Track 1", { exact: true })).toHaveCount(0);

  // A selected sound is dropped from the selection when the filter hides it.
  await page.getByRole("checkbox", { name: "Select Sound 2" }).check();
  await expect(page.getByRole("button", { name: "Delete selected (1)" })).toBeVisible();
  await kinds.getByRole("button", { name: "Music (1)" }).click();
  await expect(cards).toHaveCount(1);
  await expect(page.getByRole("button", { name: /Delete selected/ })).toHaveCount(0);

  await kinds.getByRole("button", { name: "All (3)" }).click();
  await expect(cards).toHaveCount(3);
});

function speech(i: number, kind: "speech" | "transcript", tagged: boolean) {
  return {
    id: `v-${i}`,
    kind,
    title: `${kind} ${i}`,
    preview: `preview ${i}`,
    createdAt: Date.UTC(2026, 10, 1, 0, i) / 1000,
    updatedAt: Date.UTC(2026, 10, 1, 0, i) / 1000,
    audio: kind === "speech" ? "wav" : null,
    speakers: {},
    speakerCount: 0,
    durationS: 3,
    projectId: tagged ? projectId : null,
    projectName: tagged ? PROJECT : null,
  };
}

test("Voice library: project filter, badge, counts and Open in Storyboard", async ({ page }) => {
  const items = [speech(1, "speech", true), speech(2, "transcript", true), speech(3, "speech", false)];
  await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items } }));
  await goTo(page, "Voice", "Library");

  await expect(page.getByText(`From project ${PROJECT}`)).toHaveCount(2);
  await pickProject(page, new RegExp(PROJECT));
  await expect(page.getByText("preview 3")).toHaveCount(0);
  await expect(page.getByText("preview 1")).toBeVisible();
  const header = page.locator(".project-header");
  await expect(header).toContainText("1 speech item");
  await expect(header).toContainText("1 transcript");

  await pickProject(page, /No project/);
  await expect(page.getByText("preview 3")).toBeVisible();
  await expect(page.getByText("preview 1")).toHaveCount(0);

  await pickProject(page, new RegExp(PROJECT));
  await header.getByRole("button", { name: "Open in Storyboard" }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name: PROJECT })).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Storyboard/ })).toHaveAttribute("aria-selected", "true");
});

test("Voice library: Open Studio goes to Voice Studio", async ({ page }) => {
  await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items: [speech(1, "speech", false)] } }));
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime ?? body, { installed: true });
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Voice", "Library");
  await page.getByRole("button", { name: "Open Studio" }).click();
  await expect(page.locator(".studio-shell__header").getByRole("heading", { name: "Voice Studio" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Speak" })).toHaveAttribute("aria-selected", "true");
});

test("Voice library speech cards: voice name and kind, AI-made on cloned speech, Speak again opens Speak with the text and voice", async ({ page }) => {
  const items = [
    { ...speech(1, "speech", false), voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset", preview: "Good evening, everyone." },
    { ...speech(2, "speech", false), voice: "clone:voice-1", voiceName: "Ada", voiceKind: "clone", clonedVoice: true },
    // Saved before the voice name was stored: the card works it out from the id.
    { ...speech(3, "speech", false), voice: "af_heart" },
  ];
  await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items } }));
  await page.route(/\/api\/voice\/library\/v-1$/, (route) =>
    route.fulfill({ json: { ...items[0], text: "Good evening, everyone. This is the whole text.", segments: [], audioPath: null } }),
  );
  await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b"], speakVoice: "af_heart" });
  await goTo(page, "Voice", "Library");

  const ryan = page.locator(".media-library-card", { hasText: "speech 1" });
  await expect(ryan.getByText("Natural", { exact: true })).toBeVisible();
  await expect(ryan).toContainText("Ryan · ");
  await expect(ryan).not.toContainText("qwen3:Ryan");
  const ada = page.locator(".media-library-card", { hasText: "speech 2" });
  await expect(ada.getByText("AI-made, cloned voice")).toBeVisible();
  await expect(ada.getByText("Cloned", { exact: true })).toBeVisible();
  const old = page.locator(".media-library-card", { hasText: "speech 3" });
  await expect(old).toContainText("Heart · ");
  await expect(old.getByText("Quick", { exact: true })).toBeVisible();

  await ryan.getByRole("button", { name: "Speak again" }).click();
  await expect(page.getByRole("tab", { name: "Speak" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByLabel("Text to speak")).toHaveValue("Good evening, everyone. This is the whole text.");
  await expect(page.getByRole("radio", { name: /^Ryan/ }).first()).toHaveAttribute("aria-checked", "true");
});

test.describe("Voice library at 375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("speech cards with badges and Speak again fit the width", async ({ page }) => {
    const items = [
      { ...speech(1, "speech", true), voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset" },
      { ...speech(2, "speech", false), voice: "clone:voice-1", voiceName: "Ada", voiceKind: "clone", clonedVoice: true },
    ];
    await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items } }));
    await goTo(page, "Voice", "Library");
    await expect(page.getByText("AI-made, cloned voice")).toBeVisible();
    await expect(page.getByRole("button", { name: "Speak again" })).toHaveCount(2);
    const wide = await page.evaluate(() =>
      [...document.querySelectorAll(".voice-library__grid *")]
        .filter((el) => el.getBoundingClientRect().right > window.innerWidth + 1)
        .map((el) => el.tagName),
    );
    expect(wide).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    await page.screenshot({ path: "test-results/voice-library-375.png", fullPage: true });
  });
});
