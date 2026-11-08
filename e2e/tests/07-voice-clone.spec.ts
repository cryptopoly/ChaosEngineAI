import { test, expect } from "./fixtures";
import type { Page, Route } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi, speakAnswer } from "./voiceMocks";

// Voice cloning flow in Voice Studio > Voices (and Speak for the cloned voice): the standing statement, the new-voice form (whose voice, the
// speaker's name, a voice sample from a file or the microphone, one permission tick), what the form sends,
// and the AI label on everything made with a cloned voice. The voice pack and the cloning model are patched
// to look installed; no model runs. The microphone test uses Chromium's fake audio device, not a real one.

const ADA_VOICE = ADA;

/** Mock the voice routes; returns the multipart body of each voice the form saves (as latin1 text). */
async function mockClone(page: Page, voices: unknown[] = []): Promise<string[]> {
  const saved: string[] = [];
  await mockVoiceApi(page, { clones: voices as Array<typeof ADA> });
  // The save route is the one the form posts to; the list route stays as mockVoiceApi set it.
  await page.route("**/api/voice/clone/voices", (route: Route) => {
    if (route.request().method() === "POST") {
      saved.push(route.request().postDataBuffer()?.toString("latin1") ?? "");
      return route.fulfill({ json: { ...ADA_VOICE, id: "voice-2", name: "Grace" } });
    }
    return route.fulfill({ json: { voices } });
  });
  return saved;
}

/** Clone a voice lives on the Voices screen of Voice Studio now. */
async function openClonePanel(page: Page) {
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  await expect(page.locator(".voice-clone__statement")).toBeVisible();
}

/** The value of one multipart text field. */
function field(body: string, name: string): string | undefined {
  return body.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]*)\\r\\n`))?.[1];
}

async function fillDetails(page: Page) {
  await page.getByRole("radio", { name: "Someone who gave me permission" }).check();
  await page.getByLabel("Speaker's name").fill("Grace Example");
  await page.getByLabel("Name for this voice (optional)").fill("Grace");
}

const confirmBox = (page: Page) =>
  page.getByRole("checkbox", { name: "This is my own voice, or the speaker has given me permission to clone it." });

test("clone panel states the rules, and the AI label cannot be turned off", async ({ page }) => {
  await mockClone(page);
  await openClonePanel(page);
  const statement = page.locator(".voice-clone__statement");
  await expect(statement).toContainText("Only clone your own voice or the voice of someone who has given you permission");
  await expect(statement).toContainText("Not public figures, children or people who have died");
  await expect(statement).toContainText("labelled as AI-made inside the file");
  await expect(statement).toContainText("neither can be turned off");
  // Nothing in the panel offers a way to switch the label or the watermark off.
  await expect(page.locator(".voice-voices").getByText(/(remove|turn off|disable).*(label|watermark)/i)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
});

test("an uploaded sample: Save needs whose voice, a name, the tick and a supported file, and sends them all", async ({ page }) => {
  const saved = await mockClone(page);
  await openClonePanel(page);
  await page.getByRole("button", { name: "New cloned voice" }).click();
  const save = page.getByRole("button", { name: "Save voice" });
  await expect(save).toBeDisabled();
  await expect(page.getByRole("button", { name: "Record" })).toBeVisible();

  await fillDetails(page);
  await expect(save).toBeDisabled();
  await confirmBox(page).check();
  await expect(save).toBeDisabled(); // still no sample

  const sampleInput = page.getByLabel("Voice sample file");
  await sampleInput.setInputFiles({ name: "clip.webm", mimeType: "audio/webm", buffer: Buffer.from("not used") });
  await expect(page.getByText("Choose a WAV, MP3, M4A, FLAC or OGG file.")).toBeVisible();
  await expect(save).toBeDisabled();

  await sampleInput.setInputFiles({ name: "grace.mp3", mimeType: "audio/mpeg", buffer: Buffer.from("ID3 fake mp3 bytes") });
  await expect(page.getByText("Choose a WAV, MP3, M4A, FLAC or OGG file.")).toHaveCount(0);
  await expect(save).toBeEnabled();
  // Taking the tick back closes the gate again.
  await confirmBox(page).uncheck();
  await expect(save).toBeDisabled();
  await confirmBox(page).check();

  await save.click();
  await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
  expect(saved).toHaveLength(1);
  const body = saved[0];
  expect(body).toContain('name="sample"; filename="grace.mp3"');
  expect(body).toContain("ID3 fake mp3 bytes");
  expect(field(body, "speakerName")).toBe("Grace Example");
  expect(field(body, "basis")).toBe("permission");
  expect(field(body, "name")).toBe("Grace");
  expect(field(body, "permissionConfirmed")).toBe("true");
});

// Chromium's fake microphone (a tone) for the recording tests; the other tests never open the microphone.
test.use({
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
  permissions: ["microphone"],
});

test.describe("microphone sample (Chromium fake audio device)", () => {
  test("a recording of 3 s or more is the sample and is sent as WAV", async ({ page }) => {
    const saved = await mockClone(page);
    await openClonePanel(page);
    await page.getByRole("button", { name: "New cloned voice" }).click();
    await page.getByRole("radio", { name: "My own voice" }).check();
    await page.getByLabel("Speaker's name").fill("Ada Example");
    await confirmBox(page).check();

    await page.getByRole("button", { name: "Record" }).click();
    await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
    await page.waitForTimeout(3600);
    await page.getByRole("button", { name: "Stop recording" }).click();
    await expect(page.getByRole("button", { name: "Record again" })).toBeVisible();
    await expect(page.getByLabel("Your recording")).toBeVisible();
    await page.getByRole("button", { name: "Save voice" }).click();
    await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
    expect(saved).toHaveLength(1);
    expect(saved[0]).toContain('name="sample"; filename="recording.wav"');
    expect(saved[0]).toContain("RIFF");
    expect(field(saved[0], "basis")).toBe("own");
    expect(field(saved[0], "name")).toBe("Ada Example");
  });

  test("a recording under 3 s is refused before saving", async ({ page }) => {
    await mockClone(page);
    await openClonePanel(page);
    await page.getByRole("button", { name: "New cloned voice" }).click();
    await page.getByRole("radio", { name: "My own voice" }).check();
    await page.getByLabel("Speaker's name").fill("Ada Example");
    await confirmBox(page).check();
    await page.getByRole("button", { name: "Record" }).click();
    await page.waitForTimeout(800);
    await page.getByRole("button", { name: "Stop recording" }).click();
    await expect(page.getByText("The recording is shorter than 3 seconds. Record a little more.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Save voice" })).toBeDisabled();
  });
});

test("a cloned voice speaks from the Speak screen with the AI label shown", async ({ page }) => {
  await mockClone(page, [ADA]);
  let spoken: Record<string, unknown> | null = null;
  await page.route("**/api/voice/speak", (route) => {
    spoken = route.request().postDataJSON() as Record<string, unknown>;
    return route.fulfill({
      json: speakAnswer({ voice: "clone:voice-1", kind: "clone", model: "qwen3-tts-0.6b", clonedVoice: true, libraryId: "speech-1" }),
    });
  });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Speak" }).click();

  await page.getByRole("radio", { name: /^Ada(?!m)/ }).click();
  await expect(page.getByText("own voice, confirmed 2026-11-01").first()).toBeVisible();
  await expect(page.locator(".voice-speak").getByText("Only clone your own voice")).toBeVisible();
  const generate = page.getByRole("button", { name: "Generate audio" });
  await expect(generate).toBeDisabled();
  await page.getByLabel("Text to speak").fill("Hello there");
  await expect(generate).toBeEnabled();
  await generate.click();

  await expect(page.locator(".voice-studio__result .badge", { hasText: "Labelled as AI-made" })).toBeVisible();
  await expect(page.getByText("Saved to Library")).toBeVisible();
  expect(spoken).toMatchObject({ voice: "clone:voice-1", text: "Hello there", style: "", pitch: 0, seed: null });
  expect(spoken).not.toHaveProperty("disclaimer");
});

test.describe("375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("the Voices screen and the new-voice form fit the width", async ({ page }) => {
    await mockClone(page, [ADA]);
    await openClonePanel(page);
    await page.getByRole("button", { name: "New cloned voice" }).click();
    await fillDetails(page);
    await page.getByLabel("Voice sample file").setInputFiles({ name: "grace.wav", mimeType: "audio/wav", buffer: Buffer.from("RIFF") });
    await confirmBox(page).check();
    await expect(page.getByRole("button", { name: "Save voice" })).toBeEnabled();
    const { sideways, wide } = await page.evaluate(() => {
      const width = window.innerWidth;
      const wide = [...document.querySelectorAll(".voice-voices *")]
        .filter((el) => {
          const box = el.getBoundingClientRect();
          return box.width > 0 && box.right > width + 1 && getComputedStyle(el).visibility !== "hidden";
        })
        .slice(0, 5)
        .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 40)} right=${Math.round(el.getBoundingClientRect().right)}`);
      return { sideways: document.documentElement.scrollWidth - width, wide };
    });
    expect(sideways, `page scrolls sideways (${wide.join("; ")})`).toBeLessThanOrEqual(0);
    expect(wide, "clone form elements stick out past the right edge").toEqual([]);
  });
});
