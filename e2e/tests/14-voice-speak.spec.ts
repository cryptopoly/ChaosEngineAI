import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi, speakAnswer } from "./voiceMocks";

// Voice Studio > Speak and Voices, and the Voice Models page rows (D-58, D-73). The voice runtime, the voice list and
// the model list are mocked; /api/voice/speak is mocked and its request body is checked. No model runs. Speak shows
// one voice row; the picker opens in a drawer behind Change voice.

const BOTH_CUSTOM = ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b"];

async function openSpeak(page: Page) {
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Speak" }).click();
  await expect(page.getByLabel("Text to speak")).toBeVisible();
}

/** Open the picker drawer on Speak (Change voice). */
async function openPicker(page: Page) {
  const drawer = page.getByRole("dialog", { name: "Choose a voice" });
  if (!(await drawer.isVisible())) await page.getByRole("button", { name: "Change voice" }).click();
  await expect(drawer).toBeVisible();
  return drawer;
}

/** Pick a voice in the drawer; a click closes it. */
async function chooseVoice(page: Page, name: RegExp) {
  const drawer = await openPicker(page);
  await drawer.getByRole("radio", { name }).first().click();
  await expect(drawer).toHaveCount(0);
}

/** The voice Speak will use, named on the voice row. */
async function expectSpeakVoice(page: Page, name: RegExp) {
  await expect(page.locator(".voice-row__name strong")).toHaveText(name);
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
  const saved: Array<Record<string, string>> = [];
  await page.route("**/api/voice/library/speech", (route) => {
    const form = route.request().postData() ?? "";
    const field = (name: string) => form.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]*)`))?.[1] ?? "";
    saved.push({ text: field("text"), voice: field("voice"), voiceName: field("voiceName"), voiceKind: field("voiceKind") });
    return route.fulfill({ json: { id: "1790000000_abcdef12", kind: "speech", title: "Hello world.", text: "Hello world.", createdAt: 1, updatedAt: 1, audio: ".wav", speakers: {}, segments: [] } });
  });
  await openSpeak(page);

  // The voice last used in Speak wins over the natural voice the rule would pick.
  await expectSpeakVoice(page, /^Heart/);
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
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({ text: "Hello world.", voice: "af_heart", style: "", speed: 1, pitch: 0, seed: null });
  // Every result is saved to the Library with the voice's name and kind; the row links there.
  await expect.poll(() => saved).toEqual([{ text: "Hello world.", voice: "af_heart", voiceName: "Heart", voiceKind: "builtin" }]);
  await expect(page.getByRole("button", { name: "Save to Library" })).toHaveCount(0);
  await page.getByRole("button", { name: "Saved to Library" }).click();
  await expect(page.locator("section").getByRole("heading", { name: "Voice Library" })).toBeVisible();
  await expect(page.getByRole("radio", { name: "Speech" })).toHaveAttribute("aria-checked", "true");
});

test("first run with no natural voice: Speak starts on the best quick voice and offers to install a better one in place", async ({ page }) => {
  const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b"] });
  await openSpeak(page);
  await expectSpeakVoice(page, /^Heart/);
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
  await expectSpeakVoice(page, /^Ryan/);
  await expect(page.locator(".voice-install-card")).toHaveCount(0);
});

test("Not now hides the install card, and a picked voice is saved as the Speak voice", async ({ page }) => {
  const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b"] });
  await openSpeak(page);
  await page.locator(".voice-install-card").getByRole("button", { name: "Not now" }).click();
  await expect(page.locator(".voice-install-card")).toHaveCount(0);
  await chooseVoice(page, /^Bella/);
  await expect.poll(() => seen.speakVoices).toEqual(["af_bella"]);
});

test("picker groups: Recommended, a graded Quick shortlist with an expander, Cloned, Designed, More", async ({ page }) => {
  await mockVoiceApi(page, { clones: [ADA], installed: [...BOTH_CUSTOM, "qwen3-tts-design-1.7b"], speakVoice: "af_heart" });
  const installs: unknown[] = [];
  await page.route("**/api/voice/install", (route) => {
    installs.push(route.request().postDataJSON());
    return route.fulfill({ json: { phase: "installing", phaseLabel: "Installing", phaseIndex: 1, phaseTotal: 5, percent: 10, message: "", error: null, log: [], startedAt: 0, finishedAt: 0, running: true, extras: ["chinese"] } });
  });
  await openSpeak(page);
  await openPicker(page);
  const picker = page.locator(".voice-picker");
  const titles = picker.locator(".voice-group__head > .voice-group__title");
  await expect(titles).toHaveText(["Recommended", "Quick", "Cloned", "Designed", "More"]);

  const recommended = picker.locator(".voice-picker__group--recommended .voice-card__title");
  await expect(recommended).toHaveText(["Ryan", "Heart", "Ada", "Design a voice from a description"]);

  const quick = picker.locator(".voice-picker__group--quick");
  await expect(quick).toContainText("Kokoro, part of the voice runtime. Fast, under 1 GB, speed only. 22 voices in 2 languages.");
  await expect(quick.locator(".voice-card__title")).toHaveText(["Heart", "Bella", "Nicole", "Fenrir", "Michael", "Puck"]);
  await quick.getByRole("button", { name: "All 20 American English voices" }).click();
  await expect(quick.locator(".voice-card__title")).toHaveCount(20);
  await quick.getByRole("button", { name: "Show the shortlist" }).click();
  await expect(quick.locator(".voice-card__title")).toHaveCount(6);

  // Mandarin needs the add-on here: the card installs it in place.
  await quick.getByLabel("Language").selectOption("Chinese (Mandarin)");
  await expect(quick.locator(".voice-card__main")).toHaveCount(0);
  await quick.getByRole("button", { name: "Add Mandarin Chinese voices" }).click();
  await expect.poll(() => installs.length).toBe(1);
  expect((installs[0] as { extras: string[] }).extras).toContain("chinese");

  await expect(picker.locator(".voice-picker__group--more .voice-card__title")).toContainText(["Aiden", "Get more voices"]);
  // The cloning statement lives in the New cloned voice form, not in the picker (D-73).
  await expect(picker).not.toContainText("labelled as AI-made inside the file");
  await picker.locator(".voice-picker__group--clones").getByRole("button", { name: "Open the form" }).click();
  await expect(page.getByRole("dialog", { name: "Choose a voice" })).toHaveCount(0);
  await expect(page.getByRole("tab", { name: "Voices" })).toHaveAttribute("aria-selected", "true");
});

test("typing a style never swaps the model: the installed style model runs throughout", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM });
  await openSpeak(page);
  await expectSpeakVoice(page, /^Ryan/);
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
  await chooseVoice(page, /^Ada\b/);
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
  await chooseVoice(page, /^Ada\b/);
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
  await chooseVoice(page, /^Ryan/);
  await expect(styleBox(page)).toBeDisabled();
  await expect(page.getByText("The small model ignores style.")).toBeVisible();
  await page.getByRole("button", { name: "Download Qwen3-TTS 1.7B CustomVoice for style" }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);
});

test("a preset with the style model: chips fill the box, speed, pitch and seed are sent", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM });
  const bodies = await mockSpeak(page);
  await openSpeak(page);
  await chooseVoice(page, /^Ryan/);

  const style = styleBox(page);
  await expect(style).toBeEnabled();
  await page.getByRole("button", { name: "Angry", exact: true }).click();
  await expect(style).toHaveValue("Speak in an angry tone.");
  await expect(page.locator(".voice-speak__model")).toContainText(
    "Runs with Qwen3-TTS 1.7B CustomVoice: 3.5 GB in memory, about 2.5 s per 10 s of speech (measured on Apple M4 Max, 64 GB, 2026-11-08).",
  );
  await expect(page.locator(".voice-row")).toContainText("3.5 GB, 2.5 s per 10 s");

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
  await chooseVoice(page, /^Ryan/);
  await expect(page.getByText(/Needs Qwen3-TTS 1.7B CustomVoice .*not downloaded yet/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Generate audio" })).toBeDisabled();
  await page.locator(".voice-speak__model").getByRole("button", { name: "Download", exact: true }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);

  await page.unroute("**/api/voice/models");
  await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-design-1.7b"] });
  await openSpeak(page);
  await chooseVoice(page, /Design a voice from a description/);
  await page.getByLabel("Text to speak").fill("Hello.");
  await expect(page.getByLabel("Describe the voice")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Generate audio" })).toBeDisabled();
  await page.getByLabel("Describe the voice").fill("A calm woman in her thirties.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect(page.getByRole("button", { name: "WAV" })).toBeVisible();
  expect(bodies.at(-1)).toMatchObject({ voice: "design", style: "A calm woman in her thirties." });
});

test("Voices groups every voice, filters by kind, Use for > Speak sets the Speak voice, Delete asks first", async ({ page }) => {
  const seen = await mockVoiceApi(page, { clones: [ADA] });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  const list = page.locator(".voice-voices");
  for (const name of ["Heart", "Ryan", "Ada", "Design a voice"]) {
    await expect(list.locator(".voice-list__name strong", { hasText: new RegExp(`^${name}$`) }).first()).toBeVisible();
  }
  await expect(page.locator(".voice-voices__group h4")).toHaveText(["Natural", "Quick", "Cloned", "Designed"]);
  for (const action of ["New cloned voice", "Design a voice", "Get more voices"]) {
    await expect(page.getByRole("button", { name: action, exact: true }).first()).toBeVisible();
  }
  await expect(page.getByRole("button", { name: "Add Mandarin Chinese voices" })).toBeVisible();
  // Every voice that runs has a sample button, cloned and designed ones too (FU-146).
  await expect(list.getByRole("button", { name: "Play a sample of Ada", exact: true })).toBeVisible();

  const rows = page.locator(".voice-list__row");
  await page.getByRole("tab", { name: "Cloned" }).click();
  await expect(rows).toHaveCount(1);
  await page.getByRole("tab", { name: "Quick" }).click();
  await expect(rows.filter({ hasText: "Ada" })).toHaveCount(0);
  await expect(rows.filter({ hasText: "Heart" })).toHaveCount(1);
  await page.getByRole("tab", { name: "Natural" }).click();
  await expect(rows).toHaveCount(2);
  await page.getByRole("tab", { name: "Designed" }).click();
  await expect(rows).toHaveCount(1);
  await page.getByRole("tab", { name: "All", exact: true }).click();

  const aiden = rows.filter({ hasText: "Aiden" });
  await aiden.getByText("Use for…").click();
  await aiden.getByRole("menuitemradio", { name: /^Speak/ }).click();
  await expect.poll(() => seen.speakVoices).toContain("qwen3:Aiden");
  await expect(page.locator('[data-use="speak"]')).toContainText("Aiden");
  await page.getByRole("tab", { name: "Speak" }).click();
  await expectSpeakVoice(page, /^Aiden/);

  await page.getByRole("tab", { name: "Voices" }).click();
  await page.locator(".voice-voices__actions").getByRole("button", { name: "Design a voice" }).click();
  await expectSpeakVoice(page, /Design a voice/);

  await page.getByRole("tab", { name: "Voices" }).click();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "Delete the voice Ada" }).click();
  expect(seen.deleted).toEqual([]);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete the voice Ada" }).click();
  await expect.poll(() => seen.deleted).toEqual(["voice-1"]);
});

test("the main voices: Use for sets read-aloud and calls with W6's rules; the chat settings are in Chat now", async ({ page }) => {
  const seen = await mockVoiceApi(page, { clones: [ADA], installed: BOTH_CUSTOM });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  const read = page.locator('[data-use="read"]');
  const call = page.locator('[data-use="call"]');
  await expect(read).toContainText("Heart");
  await expect(call).toContainText("(same as Chat)");
  const rows = page.locator(".voice-list__row");
  const bella = rows.filter({ hasText: "Bella" });
  await bella.getByText("Use for…").click();
  await bella.getByRole("menuitemradio", { name: /^Chat read-aloud/ }).click();
  await expect.poll(() => seen.ttsVoices).toEqual(["af_bella"]);
  await expect(read).toContainText("Bella");
  await expect(bella.locator(".badge", { hasText: /^Chat$/ })).toBeVisible();
  // D-72: a cloned voice reads aloud but is not used in calls; "Describe a voice" has no Use for menu.
  const ada = rows.filter({ hasText: "Ada" });
  await ada.getByText("Use for…").click();
  await expect(ada.getByRole("menuitemradio", { name: /^Chat read-aloud/ })).toBeEnabled();
  await expect(ada.getByRole("menuitemradio", { name: /^Voice calls/ })).toBeDisabled();
  await expect(ada.getByRole("menuitemradio", { name: /^Voice calls/ })).toContainText("Cloned voices are not used in voice calls.");
  await page.keyboard.press("Escape");
  await expect(rows.filter({ hasText: "Design a voice" }).getByText("Use for…")).toHaveCount(0);
  // Change on the calls slot opens the drawer with only the voices calls may use.
  await call.getByRole("button", { name: /Change the voice for Voice calls/ }).click();
  const drawer = page.getByRole("dialog", { name: "Choose the voice for Voice calls" });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole("radio", { name: /^Ada\b/ })).toHaveCount(0);
  await drawer.getByRole("radio", { name: /^Ryan/ }).first().click();
  await expect.poll(() => seen.callVoices).toEqual(["qwen3:Ryan"]);
  await expect(drawer).toHaveCount(0);
  await expect(call).toContainText("Ryan");
  // The chat behaviour settings and the agent switch left Voices (Chat's Voice popover, Settings > Agent access).
  await expect(page.getByText("Read every chat reply aloud when it finishes")).toHaveCount(0);
  await expect(page.getByText(/Allow agents and other computers/)).toHaveCount(0);
  await expect(page.getByText("Kokoro reads instead and Chat says so", { exact: false })).toBeVisible();
});

test("a cloned voice's sample is made through speak and not kept in the Library", async ({ page }) => {
  await mockVoiceApi(page, { clones: [ADA] });
  const bodies: Array<Record<string, unknown>> = [];
  const deleted: unknown[] = [];
  await page.route("**/api/voice/speak", (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ json: speakAnswer({ voice: "clone:voice-1", kind: "clone", clonedVoice: true, libraryId: "1790000000_abcdef12" }) });
  });
  await page.route("**/api/voice/library/delete", (route) => {
    deleted.push(route.request().postDataJSON());
    return route.fulfill({ json: { deleted: ["1790000000_abcdef12"] } });
  });
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Voices" }).click();
  await page.locator(".voice-list").getByRole("button", { name: "Play a sample of Ada", exact: true }).click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ voice: "clone:voice-1", text: "Hello! This is how I sound." });
  await expect.poll(() => deleted).toEqual([{ ids: ["1790000000_abcdef12"] }]);
});

test("without ffmpeg, Speak says so before Generate when speed or pitch is changed", async ({ page }) => {
  await mockVoiceApi(page, { installed: BOTH_CUSTOM, speakVoice: "qwen3:Ryan", needsFfmpeg: true });
  await mockSpeak(page);
  await openSpeak(page);
  await page.getByLabel("Text to speak").fill("Hello.");
  const generate = page.getByRole("button", { name: "Generate audio" });
  await expect(generate).toBeEnabled();
  await page.getByLabel(/^Speed/).fill("1.5");
  await expect(page.getByText("Speed and pitch for this voice need ffmpeg.", { exact: false })).toBeVisible();
  await expect(generate).toBeDisabled();
  await page.getByLabel(/^Speed/).fill("1");
  await expect(generate).toBeEnabled();
  // A quick voice changes speed itself; only pitch needs ffmpeg.
  await chooseVoice(page, /^Heart/);
  await page.getByLabel(/^Speed/).fill("1.5");
  await expect(generate).toBeEnabled();
  await page.getByLabel(/^Pitch/).fill("2");
  await expect(page.getByText("Changing the pitch needs ffmpeg.", { exact: false })).toBeVisible();
  await expect(generate).toBeDisabled();
});

test("Voice Models lists the speech models in the shared table: Kokoro first, Makes, Fit, Download and Delete", async ({ page }) => {
  const seen = await mockVoiceApi(page);
  const deleted: string[] = [];
  await page.route("**/api/voice/clone/models/delete-plan*", (route) =>
    route.fulfill({ json: { key: "qwen3-tts-custom-0.6b", name: "Qwen3-TTS 0.6B CustomVoice", remove: ["mock/qwen3-tts-custom-0.6b"], kept: [], watermark: false, sizeGb: 2.5 } }),
  );
  await page.route("**/api/voice/clone/models/delete", (route) => {
    deleted.push((route.request().postDataJSON() as { key: string }).key);
    return route.fulfill({ json: { key: "qwen3-tts-custom-0.6b", deleted: true, removed: [], kept: [], watermarkRemoved: false, freedGb: 2.5 } });
  });
  await goTo(page, "Voice", "Models");
  await expect(page.getByRole("heading", { name: "Voice runtime" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
  const rows = page.locator(".models-card");
  await expect(rows.first()).toHaveAttribute("data-model-id", "kokoro");
  await expect(rows.first()).toContainText("Part of the runtime");
  await expect(rows.first()).toContainText("the voice Chat falls back to");
  await expect(page.locator(".voice-models__family", { hasText: "Qwen3-TTS" })).toBeVisible();
  const row = page.locator('[data-model-id="qwen3-tts-custom-1.7b"]');
  await expect(row.getByText("Style", { exact: true }).first()).toBeVisible();
  await expect(row.getByText("3.08 GB", { exact: true })).toBeVisible();
  await expect(row.getByText("~3.5 GB peak")).toBeVisible();
  await expect(page.locator('[data-model-id="qwen3-tts-0.6b"]').getByText("Clone", { exact: true }).first()).toBeVisible();
  await expect(page.locator('[data-model-id="qwen3-tts-design-1.7b"]').getByText("Design", { exact: true }).first()).toBeVisible();
  await row.getByRole("button", { name: "Download 3.08 GB" }).click();
  await expect.poll(() => seen.downloads).toEqual(["qwen3-tts-custom-1.7b"]);
  // The detail has the measured cost.
  await row.getByRole("button", { name: /Qwen3-TTS 1.7B CustomVoice/ }).click();
  await expect(row).toContainText("3.5 GB in memory, about 2.5 s per 10 s of speech (measured on Apple M4 Max, 64 GB, 2026-11-08)");
  // Delete asks first, naming the model and its size.
  const installed = page.locator('[data-model-id="qwen3-tts-custom-0.6b"]');
  let asked = "";
  page.once("dialog", (dialog) => {
    asked = dialog.message();
    void dialog.dismiss();
  });
  await installed.getByRole("button", { name: "Delete model" }).click();
  await expect.poll(() => asked).toContain("Delete Qwen3-TTS 0.6B CustomVoice (2.50 GB)");
  expect(deleted).toEqual([]);
  page.once("dialog", (dialog) => void dialog.accept());
  await installed.getByRole("button", { name: "Delete model" }).click();
  await expect.poll(() => deleted).toEqual(["qwen3-tts-custom-0.6b"]);
});

test("Voice Studio opens on Speak, then on the mode used last", async ({ page }) => {
  await mockVoiceApi(page);
  await goTo(page, "Voice", "Studio");
  await expect(page.getByRole("tab", { name: "Speak" })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: "Transcribe" }).click();
  await page.locator(".studio-shell__header").getByRole("button", { name: "Models" }).click();
  await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
  await page.getByRole("button", { name: "Open Studio" }).click();
  await expect(page.getByRole("tab", { name: "Transcribe" })).toHaveAttribute("aria-selected", "true");
});

test("Models > Use in Speak lands on Speak with that model's voice; a cloning model opens Voices", async ({ page }) => {
  await mockVoiceApi(page, { installed: [...BOTH_CUSTOM, "qwen3-tts-design-1.7b"], speakVoice: "af_heart" });
  await goTo(page, "Voice", "Models");
  await page.locator('[data-model-id="qwen3-tts-custom-1.7b"]').getByRole("button", { name: "Use in Speak" }).click();
  await expect(page.getByRole("tab", { name: "Speak" })).toHaveAttribute("aria-selected", "true");
  await expectSpeakVoice(page, /^Ryan/);
  await expect(page.locator(".voice-speak__model").getByRole("button", { name: "Manage in Models" })).toBeVisible();

  await page.locator(".voice-speak__model").getByRole("button", { name: "Manage in Models" }).click();
  await page.locator('[data-model-id="qwen3-tts-design-1.7b"]').getByRole("button", { name: "Use in Speak" }).click();
  await expectSpeakVoice(page, /Design a voice/);

  await page.locator(".studio-shell__header").getByRole("button", { name: "Models" }).click();
  await page.locator('[data-model-id="qwen3-tts-0.6b"]').getByRole("button", { name: "Browse voices" }).click();
  await expect(page.getByRole("tab", { name: "Voices" })).toHaveAttribute("aria-selected", "true");
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
    await chooseVoice(page, /^Ryan/);
    await expectFits(page, ".voice-speak");
    await expectFits(page, ".studio-shell__header");
    // All 20 quick voices open at once still fit, in the drawer, which takes the full width.
    await openPicker(page);
    await page.locator(".voice-picker__group--quick").getByRole("button", { name: /^All 20/ }).click();
    await expect(page.locator(".voice-picker__group--quick .voice-card__main")).toHaveCount(20);
    await expectFits(page, ".drawer");
    await page.keyboard.press("Escape");
    await page.getByRole("tab", { name: "Voices" }).click();
    await expect(page.locator(".voice-main")).toBeVisible();
    await expectFits(page, ".voice-voices");
    await goTo(page, "Voice", "Models");
    await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
    await expectFits(page, ".voice-page");
  });
});
