import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";

// Voice cloning flow in Voice Studio > Clone: the standing statement, the consent gates, the consent
// sentence with its code, no way to save without a recording, and the AI label on everything made with
// a cloned voice. The voice pack is patched to look installed; no model runs and no microphone is used.

async function openClonePanel(page: Page) {
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime ?? body, { installed: true });
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Clone" }).click();
  await expect(page.locator(".voice-clone__statement")).toBeVisible();
}

test("clone panel states the rules, and the AI label cannot be turned off", async ({ page }) => {
  await openClonePanel(page);
  const statement = page.locator(".voice-clone__statement");
  await expect(statement).toContainText("Only copy your own voice");
  await expect(statement).toContainText("Not public figures, children or people who have died");
  await expect(statement).toContainText("labelled as AI-made inside the file");
  await expect(statement).toContainText("neither can be turned off");
  // Nothing in the panel offers a way to switch the label or the watermark off.
  await expect(page.locator(".voice-clone").getByText(/(remove|turn off|disable).*(label|watermark)/i)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
});

test("consent gates: the sentence needs a basis, a name and the confirmation; saving needs a recording", async ({ page }) => {
  await openClonePanel(page);
  await page.getByRole("button", { name: "New cloned voice" }).click();
  const getSentence = page.getByRole("button", { name: "Get the consent sentence" });
  await expect(getSentence).toBeDisabled();

  await page.getByRole("radio", { name: "My own voice" }).check();
  await expect(getSentence).toBeDisabled();
  await page.getByLabel("Speaker's name").fill("Ada Example");
  await expect(getSentence).toBeDisabled();
  const confirm = page.getByRole("checkbox", { name: /This is my own voice, or the speaker has given me permission/ });
  await confirm.check();
  await expect(getSentence).toBeEnabled();
  // Taking the confirmation back closes the gate again.
  await confirm.uncheck();
  await expect(getSentence).toBeDisabled();
  await confirm.check();

  await getSentence.click();
  const sentence = page.locator(".voice-clone__sentence");
  await expect(sentence).toBeVisible();
  await expect(page.getByText("Ada Example reads this sentence aloud, including the code one digit at a time:")).toBeVisible();
  const hint = page.getByText(/^Code: \d( \d)+\./);
  await expect(hint).toBeVisible();
  // The spoken sentence carries the same digits as the hint.
  const digits = ((await hint.textContent()) ?? "").match(/Code: ([\d ]+)\./)?.[1].replace(/ /g, "") ?? "";
  expect(digits.length).toBeGreaterThanOrEqual(4);
  expect(((await sentence.textContent()) ?? "").replace(/\D/g, "")).toContain(digits);

  // No recording, so the voice cannot be saved; the form is still cancellable.
  await expect(page.getByRole("button", { name: "Record" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Check and save voice" })).toBeDisabled();
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
});

test("a cloned voice speaks with the AI label shown and the optional spoken disclaimer is sent", async ({ page }) => {
  await page.route("**/api/voice/clone/models", (route) =>
    route.fulfill({
      json: {
        available: true,
        allowedHere: true,
        models: [
          {
            key: "mock-clone",
            name: "Mock cloning model",
            summary: "Mock.",
            repo: "mock/clone",
            revision: "0",
            sizeGb: 1,
            languages: "10 languages",
            default: true,
            licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/licence" },
            supported: true,
            installed: true,
            download: { state: "done", error: null, phase: null },
          },
        ],
      },
    }),
  );
  await page.route("**/api/voice/clone/voices", (route) =>
    route.fulfill({
      json: {
        voices: [
          {
            id: "voice-1",
            name: "Ada",
            createdAt: 1_790_000_000,
            durationS: 12,
            referenceText: "",
            consent: { speakerName: "Ada Example", basis: "own", consentedAt: "2026-11-01T10:00:00Z", appVersion: "0.0.0", code: "4721" },
          },
        ],
      },
    }),
  );
  let spoken: Record<string, unknown> | null = null;
  await page.route("**/api/voice/clone/speak", (route) => {
    spoken = route.request().postDataJSON() as Record<string, unknown>;
    return route.fulfill({
      json: {
        id: "speech-1",
        kind: "speech",
        title: "Hello",
        text: "Hello there",
        createdAt: 1_790_000_100,
        updatedAt: 1_790_000_100,
        audio: "wav",
        speakers: {},
        segments: [],
        durationS: 2,
      },
    });
  });
  await page.route("**/api/voice/library/speech-1/audio*", (route) => route.fulfill({ status: 200, contentType: "audio/wav", body: Buffer.alloc(64) }));
  await openClonePanel(page);

  await expect(page.getByText("own voice, consent recorded 2026-11-01")).toBeVisible();
  const generate = page.locator(".voice-clone").getByRole("button", { name: "Generate audio" });
  await expect(generate).toBeDisabled();
  await page.getByPlaceholder("Text to say in the cloned voice.").fill("Hello there");
  await expect(generate).toBeEnabled();
  await page.getByRole("checkbox", { name: /Start with a spoken line/ }).check();
  await generate.click();

  await expect(page.getByText("Labelled as AI-made")).toBeVisible();
  await expect(page.getByText("Saved to Library")).toBeVisible();
  expect(spoken).toMatchObject({ voiceId: "voice-1", text: "Hello there", model: "mock-clone", disclaimer: true });
});
