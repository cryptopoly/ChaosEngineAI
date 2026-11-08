import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi, speakAnswer } from "./voiceMocks";

// Voice Studio > Speak and Voices, and the Voice Models page rows (D-58). The voice pack, the voice list and the
// model list are mocked; /api/voice/speak is mocked and its request body is checked. No model runs.

const BOTH_CUSTOM = ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b"];

async function openSpeak(page: Page) {
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Speak" }).click();
  await expect(page.getByLabel("Text to speak")).toBeVisible();
}

const styleBox = (page: Page) => page.getByRole("textbox", { name: "Style", exact: true });

async function mockSpeak(page: Page): Promise<Array<Record<string, unknown>>> {
  const bodies: Array<Record<string, unknown>> = [];
  await page.route("**/api/voice/speak", (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ json: speakAnswer() });
  });
  return bodies;
}

test("a quick voice has no style: the box and chips are greyed out with a note, and the request carries no style", async ({ page }) => {
  await mockVoiceApi(page, { speakVoice: "af_heart" });
  const bodies = await mockSpeak(page);
  await openSpeak(page);

  // The voice last used in Speak wins over the natural voice the rule would pick.
  await expect(page.getByRole("radio", { name: /^Heart/ })).toHaveAttribute("aria-checked", "true");
  await expect(styleBox(page)).toBeDisabled();
  for (const chip of ["Neutral", "Angry", "Sad", "Excited", "Whisper", "Calm"]) {
    await expect(page.getByRole("button", { name: chip, exact: true })).toBeDisabled();
  }
  await expect(page.getByText("Quick voices have no style control.")).toBeVisible();
  await expect(page.getByText("Runs with Kokoro")).toBeVisible();

  await page.getByLabel("Text to speak").fill("Hello world.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  await expect(page.getByRole("button", { name: "MP3" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save to Library" })).toBeVisible();
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({ text: "Hello world.", voice: "af_heart", style: "", speed: 1, pitch: 0, seed: null });
});

test("first run with no natural voice: Speak starts on the best quick voice and offers to install a better one in place", async ({ page }) => {
  const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b"] });
  await openSpeak(page);
  await expect(page.getByRole("radio", { name: /^Heart/ })).toHaveAttribute("aria-checked", "true");
  const card = page.locator(".voice-install-card");
  await expect(card.getByRole("heading", { name: "Install a better voice" })).toBeVisible();
  await expect(card).toContainText("Qwen3-TTS 1.7B CustomVoice adds natural prosody");
  await expect(card).toContainText("Download 3.08 GB; 3.5 GB in memory, about 2.5 s per 10 s of speech (measured on Apple M4 Max, 64 GB, 2026-11-08).");
  await card.getByRole("button", { name: "Download 3.08 GB" }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);
  expect(seen.speakVoices).toEqual([]); // the rule's choice is not saved as the person's own

  // Once the model is there, Speak starts on the natural voice for the UI language and the card is gone.
  await page.unrouteAll({ behavior: "ignoreErrors" });
  await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b"] });
  await openSpeak(page);
  await expect(page.getByRole("radio", { name: /^Ryan/ })).toHaveAttribute("aria-checked", "true");
  await expect(page.locator(".voice-install-card")).toHaveCount(0);
});

test("Not now hides the install card, and a picked voice is saved as the Speak voice", async ({ page }) => {
  const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b"] });
  await openSpeak(page);
  await page.locator(".voice-install-card").getByRole("button", { name: "Not now" }).click();
  await expect(page.locator(".voice-install-card")).toHaveCount(0);
  await page.getByRole("radio", { name: /^Adam/ }).click();
  await expect.poll(() => seen.speakVoices).toEqual(["am_adam"]);
});

test("typing a style never swaps the model: the installed style model runs throughout", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM });
  await openSpeak(page);
  await expect(page.getByRole("radio", { name: /^Ryan/ })).toHaveAttribute("aria-checked", "true");
  const line = page.locator(".voice-speak__model");
  await expect(line).toContainText("Runs with Qwen3-TTS 1.7B CustomVoice:");
  await styleBox(page).fill("Slow and gentle.");
  await expect(line).toContainText("Runs with Qwen3-TTS 1.7B CustomVoice:");
  await styleBox(page).fill("");
  await expect(line).toContainText("Runs with Qwen3-TTS 1.7B CustomVoice:");
});

test("a cloned voice runs with the installed cloning model, and two installed models give a choice", async ({ page }) => {
  await mockVoiceApi(page, { clones: [ADA], installed: ["qwen3-tts-1.7b"] });
  const bodies = await mockSpeak(page);
  await openSpeak(page);
  await page.getByRole("radio", { name: /^Ada\b/ }).click();
  await expect(page.getByText("Runs with Qwen3-TTS 1.7B Base")).toBeVisible();
  await expect(page.getByLabel("Cloning model")).toHaveCount(0);
  await page.getByLabel("Text to speak").fill("Hello world.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  expect(bodies[0]).toMatchObject({ voice: "clone:voice-1", model: "qwen3-tts-1.7b" });
});

test("with both cloning models installed a select picks which one runs", async ({ page }) => {
  await mockVoiceApi(page, { clones: [ADA], installed: ["qwen3-tts-0.6b", "qwen3-tts-1.7b"] });
  const bodies = await mockSpeak(page);
  await openSpeak(page);
  await page.getByRole("radio", { name: /^Ada\b/ }).click();
  await expect(page.getByLabel("Cloning model")).toHaveValue("qwen3-tts-0.6b");
  await page.getByLabel("Cloning model").selectOption("qwen3-tts-1.7b");
  await page.getByLabel("Text to speak").fill("Hello world.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  expect(bodies[0]).toMatchObject({ model: "qwen3-tts-1.7b" });
});

test("Cancel tells the backend to stop, and the next Generate works at once", async ({ page }) => {
  await mockVoiceApi(page);
  const cancels: string[] = [];
  let calls = 0;
  await page.route("**/api/voice/speak/cancel", (route) => {
    cancels.push(route.request().method());
    return route.fulfill({ json: { cancelled: true } });
  });
  await page.route("**/api/voice/speak", async (route) => {
    calls += 1;
    if (calls === 1) return; // the first request hangs until Cancel aborts it
    await route.fulfill({ json: speakAnswer() });
  });
  await openSpeak(page);
  await page.getByLabel("Text to speak").fill("Hello world.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect.poll(() => cancels).toEqual(["POST"]);
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
});

test("the small model ignores style: greyed out with a way to download the style model", async ({ page }) => {
  const seen = await mockVoiceApi(page);
  await openSpeak(page);
  await page.getByRole("radio", { name: /^Ryan/ }).click();
  await expect(styleBox(page)).toBeDisabled();
  await expect(page.getByText("The small model ignores style.")).toBeVisible();
  await page.getByRole("button", { name: "Download Qwen3-TTS 1.7B CustomVoice for style" }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);
});

test("a preset with the style model: chips fill the box, speed, pitch and seed are sent", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM });
  const bodies = await mockSpeak(page);
  await openSpeak(page);
  await page.getByRole("radio", { name: /^Ryan/ }).click();

  const style = styleBox(page);
  await expect(style).toBeEnabled();
  await page.getByRole("button", { name: "Angry", exact: true }).click();
  await expect(style).toHaveValue("Speak in an angry tone.");
  await expect(page.locator(".voice-speak__model")).toContainText(
    "Runs with Qwen3-TTS 1.7B CustomVoice: 3.5 GB in memory, about 2.5 s per 10 s of speech (measured on Apple M4 Max, 64 GB, 2026-11-08).",
  );
  await expect(page.getByRole("radio", { name: /^Ryan/ })).toContainText("3.5 GB, 2.5 s per 10 s");

  await page.getByLabel(/^Pitch/).fill("3");
  await page.getByLabel(/^Speed/).fill("1.5");
  await page.getByText("Advanced").click();
  await page.getByLabel(/^Seed/).fill("42");
  await page.getByLabel("Text to speak").fill("Careful now.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({ voice: "qwen3:Ryan", style: "Speak in an angry tone.", pitch: 3, speed: 1.5, seed: 42 });
});

test("a missing model stops Generate and offers Download; the design voice needs its description", async ({ page }) => {
  const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b"] });
  const bodies = await mockSpeak(page);
  await openSpeak(page);
  await page.getByLabel("Text to speak").fill("Hello.");
  await page.getByRole("radio", { name: /^Ryan/ }).click();
  await expect(page.getByText(/Needs Qwen3-TTS 1.7B CustomVoice .*not downloaded yet/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Generate audio" })).toBeDisabled();
  await page.locator(".voice-speak__model").getByRole("button", { name: "Download", exact: true }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);

  await page.unroute("**/api/voice/models");
  await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-design-1.7b"] });
  await openSpeak(page);
  await page.getByRole("radio", { name: /Design a voice from a description/ }).click();
  await page.getByLabel("Text to speak").fill("Hello.");
  await expect(page.getByLabel("Describe the voice")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Generate audio" })).toBeDisabled();
  await page.getByLabel("Describe the voice").fill("A calm woman in her thirties.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  expect(bodies.at(-1)).toMatchObject({ voice: "design", style: "A calm woman in her thirties." });
});

test("Voices lists built-in and cloned voices, filters them, Use jumps to Speak, Delete asks first", async ({ page }) => {
  const seen = await mockVoiceApi(page, { clones: [ADA] });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  const list = page.locator(".voice-list");
  await expect(list.getByText("Heart", { exact: true })).toBeVisible();
  await expect(list.getByText("Ryan", { exact: true })).toBeVisible();
  await expect(list.getByText("Ada", { exact: true })).toBeVisible();
  for (const card of ["Clone a voice", "Design a voice", "Get more voices"]) {
    await expect(page.locator(".voice-newcards").getByText(card, { exact: true })).toBeVisible();
  }

  await page.getByRole("tab", { name: "Mine" }).click();
  await expect(list.getByText("Heart", { exact: true })).toHaveCount(0);
  await expect(list.getByText("Ada", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Built-in" }).click();
  await expect(list.getByText("Ada", { exact: true })).toHaveCount(0);
  await page.getByRole("tab", { name: "All" }).click();

  await list.locator("li", { hasText: "Aiden" }).getByRole("button", { name: "Use" }).click();
  await expect(page.getByLabel("Text to speak")).toBeVisible();
  await expect(page.getByRole("radio", { name: /^Aiden/ })).toHaveAttribute("aria-checked", "true");

  await page.getByRole("tab", { name: "Voices" }).click();
  await page.locator(".voice-newcards").getByRole("button", { name: "Describe a voice" }).click();
  await expect(page.getByRole("radio", { name: /Design a voice from a description/ })).toHaveAttribute("aria-checked", "true");

  await page.getByRole("tab", { name: "Voices" }).click();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "Delete the voice Ada" }).click();
  expect(seen.deleted).toEqual([]);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete the voice Ada" }).click();
  await expect.poll(() => seen.deleted).toEqual(["voice-1"]);
});

test("Voice Models lists the speech models with what each can do, and keeps the voice pack and settings", async ({ page }) => {
  const seen = await mockVoiceApi(page);
  await goTo(page, "Voice", "Models");
  await expect(page.getByRole("heading", { name: "Voice pack" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
  const row = page.locator(".voice-tts-row", { hasText: "Qwen3-TTS 1.7B CustomVoice" });
  await expect(row.getByText("Style", { exact: true })).toBeVisible();
  await expect(row.getByText("Speak", { exact: true })).toBeVisible();
  await expect(row.getByText("3.08 GB", { exact: false }).first()).toBeVisible();
  await expect(row.getByText("3.5 GB peak")).toBeVisible();
  await expect(row.getByText("2.5 s per 10 s")).toBeVisible();
  await expect(row.getByText("Apple M4 Max, 64 GB, 2026-11-08")).toBeVisible();
  await expect(page.locator(".voice-tts-row", { hasText: "0.6B Base" }).getByText("Clone", { exact: true })).toBeVisible();
  await expect(page.locator(".voice-tts-row", { hasText: "VoiceDesign" }).getByText("Design", { exact: true })).toBeVisible();
  await expect(page.locator(".voice-tts-row", { hasText: "0.6B CustomVoice" }).getByText("Downloaded")).toBeVisible();
  await row.getByRole("button", { name: "Download" }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);
});

test.describe("375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  async function expectFits(page: Page, scope: string) {
    const { sideways, wide } = await page.evaluate((root) => {
      const width = window.innerWidth;
      const wide = [...document.querySelectorAll(`${root} *`)]
        .filter((el) => {
          const box = el.getBoundingClientRect();
          return box.width > 0 && box.right > width + 1 && getComputedStyle(el).visibility !== "hidden";
        })
        .slice(0, 5)
        .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 40)} right=${Math.round(el.getBoundingClientRect().right)}`);
      return { sideways: document.documentElement.scrollWidth - width, wide };
    }, scope);
    expect(sideways, `page scrolls sideways (${wide.join("; ")})`).toBeLessThanOrEqual(0);
    expect(wide, `${scope} sticks out past the right edge`).toEqual([]);
  }

  test("Speak, Voices and Voice Models fit the width", async ({ page }) => {
    await mockVoiceApi(page, { clones: [ADA], installed: BOTH_CUSTOM });
    await openSpeak(page);
    await page.getByRole("radio", { name: /^Ryan/ }).click();
    await expectFits(page, ".voice-speak");
    await page.getByRole("tab", { name: "Voices" }).click();
    await expect(page.locator(".voice-list")).toBeVisible();
    await expectFits(page, ".voice-voices");
    await goTo(page, "Voice", "Models");
    await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
    await expectFits(page, ".voice-page");
  });
});
