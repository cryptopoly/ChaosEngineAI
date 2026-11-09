import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi } from "./voiceMocks";

// D-73 voice makeover: Delete on a speech model (confirm names the model and size, a refusal is shown), the Change
// voice drawer on Speak (keyboard open, Escape, focus back), the main-voices strip on Voices, and Chat's Voice
// popover writing the same settings keys. Everything at desktop width and at 375 px. The voice API is mocked; no
// model is downloaded, deleted or run.

const BOTH_CUSTOM = ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b"];

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

async function mockDelete(page: Page, refuse = false) {
  const deleted: string[] = [];
  await page.route("**/api/voice/clone/models/delete-plan*", (route) =>
    route.fulfill({
      json: {
        key: "qwen3-tts-custom-1.7b", name: "Qwen3-TTS 1.7B CustomVoice", remove: ["mock/qwen3-tts-custom-1.7b"],
        kept: [{ repo: "mock/shared-tokenizer", usedBy: ["Qwen3-TTS 0.6B Base"] }], watermark: false, sizeGb: 3.08,
      },
    }),
  );
  await page.route("**/api/voice/clone/models/delete", (route) => {
    deleted.push((route.request().postDataJSON() as { key: string }).key);
    if (refuse) return route.fulfill({ status: 409, json: { detail: "A voice model is loading or speaking (read-aloud, a call or Speak). Try again when it has finished." } });
    return route.fulfill({ json: { key: "qwen3-tts-custom-1.7b", deleted: true, removed: [], kept: [], watermarkRemoved: false, freedGb: 3.08 } });
  });
  return deleted;
}

for (const phone of [false, true]) {
  test.describe(phone ? "375 px" : "desktop", () => {
    // Desktop at the review's 1280 x 900 window, where Generate must be on the first screen.
    test.use({ viewport: phone ? { width: 375, height: 812 } : { width: 1280, height: 900 } });

    test("Delete on a speech model names it, its size and what another model keeps; a refusal is shown", async ({ page }) => {
      await mockVoiceApi(page, { installed: BOTH_CUSTOM });
      const deleted = await mockDelete(page, true);
      await goTo(page, "Voice", "Models");
      const row = page.locator('[data-model-id="qwen3-tts-custom-1.7b"]');
      await expect(row).toBeVisible();
      let asked = "";
      page.once("dialog", (dialog) => {
        asked = dialog.message();
        void dialog.accept();
      });
      await row.getByRole("button", { name: "Delete model" }).click();
      await expect.poll(() => asked).toContain("Delete Qwen3-TTS 1.7B CustomVoice (3.08 GB)");
      expect(asked).toContain("Kept: mock/shared-tokenizer, still used by Qwen3-TTS 0.6B Base.");
      expect(asked).toContain("Cloned voices and Library items stay.");
      await expect.poll(() => deleted).toEqual(["qwen3-tts-custom-1.7b"]);
      await expect(page.getByRole("alert").filter({ hasText: "A voice model is loading or speaking" })).toBeVisible();
      // Kokoro is part of the runtime: no Delete on its row.
      await expect(page.locator('[data-model-id="kokoro"]').getByRole("button", { name: "Delete model" })).toHaveCount(0);
      await expectFits(page, ".voice-page");
    });

    test("Speak: Change voice opens the picker drawer; Escape closes it and focus returns; a click picks and closes", async ({ page }) => {
      const seen = await mockVoiceApi(page, { installed: BOTH_CUSTOM, clones: [ADA], speakVoice: "af_heart" });
      await goTo(page, "Voice", "Studio");
      await page.getByRole("tab", { name: "Speak" }).click();
      await expect(page.locator(".voice-row__name strong")).toHaveText("Heart");
      // The form and Generate are on the first screen; the picker is not.
      await expect(page.locator(".voice-speak .voice-picker")).toHaveCount(0);
      if (!phone) await expect(page.getByRole("button", { name: "Generate audio" })).toBeInViewport();
      const change = page.getByRole("button", { name: "Change voice" });
      await change.focus();
      await page.keyboard.press("Enter");
      const drawer = page.getByRole("dialog", { name: "Choose a voice" });
      await expect(drawer).toBeVisible();
      // Focus lands on the chosen voice's card; arrow keys move without closing.
      await expect(drawer.locator('[data-voice-id="af_heart"]').first()).toBeFocused();
      await page.keyboard.press("ArrowRight");
      await expect(drawer).toBeVisible();
      await expectFits(page, ".drawer");
      await page.keyboard.press("Escape");
      await expect(drawer).toHaveCount(0);
      await expect(change).toBeFocused();
      await change.click();
      await drawer.getByRole("radio", { name: /^Ada\b/ }).first().click();
      await expect(drawer).toHaveCount(0);
      await expect(page.locator(".voice-row__name strong")).toHaveText("Ada");
      await expect.poll(() => seen.speakVoices.at(-1)).toBe("clone:voice-1");
      await expectFits(page, ".voice-speak");
    });

    test("Voices: the main-voices strip changes the read-aloud voice through the drawer", async ({ page }) => {
      const seen = await mockVoiceApi(page, { installed: BOTH_CUSTOM, clones: [ADA], ttsVoice: "af_heart" });
      await goTo(page, "Voice", "Studio");
      await page.getByRole("tab", { name: "Voices" }).click();
      const read = page.locator('[data-use="read"]');
      await expect(read).toContainText("Heart");
      await read.getByRole("button", { name: /Change the voice for Chat read-aloud/ }).click();
      const drawer = page.getByRole("dialog", { name: "Choose the voice for Chat read-aloud" });
      await expect(drawer).toBeVisible();
      // Describe a voice cannot read chat replies; a cloned voice can.
      await expect(drawer.getByRole("radio", { name: /Design a voice/ })).toHaveCount(0);
      await expect(drawer.getByRole("radio", { name: /^Ada\b/ }).first()).toBeVisible();
      await drawer.getByRole("radio", { name: /^Ryan/ }).first().click();
      await expect.poll(() => seen.ttsVoices).toEqual(["qwen3:Ryan"]);
      await expect(drawer).toHaveCount(0);
      await expect(read).toContainText("Ryan");
      await expectFits(page, ".voice-voices");
    });

    test("Chat: the Voice popover names the voices and writes the chat settings", async ({ page }) => {
      await mockVoiceApi(page, { installed: BOTH_CUSTOM, ttsVoice: "qwen3:Ryan", callVoice: "af_heart", ready: true });
      const posted: Array<Record<string, unknown>> = [];
      page.on("request", (request) => {
        if (request.url().endsWith("/api/voice/settings") && request.method() === "POST") posted.push(request.postDataJSON() as Record<string, unknown>);
      });
      await goTo(page, "Chat", "Chat");
      await page.locator(".voice-popover__button").click();
      const popover = page.locator(".voice-popover__panel");
      await expect(popover).toBeVisible();
      await expect(popover.locator(".voice-popover__line").nth(0)).toContainText("Ryan");
      await expect(popover.locator(".voice-popover__line").nth(1)).toContainText("Heart");
      await expect(popover).toContainText("Dictation");
      await expect(popover).toContainText("Needs:");
      await expect(popover).not.toContainText("(Voice Models)");
      await expectFits(page, ".voice-popover__panel");
      // The boxes follow the saved settings, so they are clicked and the request is checked.
      // The backend is shared across tests, so each box flips from whatever it holds now.
      const autoRead = popover.getByRole("checkbox", { name: "Read every chat reply aloud when it finishes" });
      const wasRead = await autoRead.isChecked();
      await autoRead.click();
      await expect.poll(() => posted.some((body) => body.autoRead === !wasRead)).toBe(true);
      const bargeIn = popover.getByRole("checkbox", { name: "Voice call: interrupt the model by speaking" });
      const wasBarge = await bargeIn.isChecked();
      await bargeIn.click();
      await expect.poll(() => posted.some((body) => body.bargeIn === !wasBarge)).toBe(true);
      // Change goes to Studio > Voices.
      await popover.getByRole("button", { name: "Change" }).first().click();
      await expect(page.getByRole("tab", { name: "Voices" })).toHaveAttribute("aria-selected", "true");
    });
  });
}
