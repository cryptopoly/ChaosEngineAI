import { test, expect } from "./fixtures";
import type { Page, Route } from "@playwright/test";
import { goTo } from "./helpers";

// D-76 / FU-166: the territory notice on Video > Models, the licence screen with its required tick, the MiniMax H3
// row appearing after Confirm and leaving after Withdraw. The real backend ships with H3_RELEASED off (nothing is
// listed), so the gate routes and the catalog row are mocked here; no weights are downloaded and nothing runs.
// The backend side of the gate (hidden rows, 403 refusals, UI-only confirm) is covered by tests/test_licence_gate.py.

const KEY = "minimax-h3";
const H3_ID = "leejet/MiniMax-H3-GGUF~fl2va-pruned-q4km";
// The tick sentence, live licence link and hosted-service notice as backend_service/licence_gate.py sends them (D-99).
const SENTENCE =
  "I am not in the European Union, the United Kingdom, the Republic of Korea or the United States, and I will not use this model or what it makes there. I accept the MiniMax H3 Community License, including its Acceptable Use Policy, and I am responsible for following it.";
const LIVE_LICENCE = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE";
const NOTICE =
  "ChaosEngineAI does not include or distribute MiniMax H3. You download it from Hugging Face yourself, and you are the licensee. If you let other people generate with it (for example through remote connections or the agent bridge), you are offering a hosted service and the licence's duties for services (sections V.2 and V.5) apply to you. MiniMax may update its Acceptable Use Policy; read the current licence on Hugging Face.";

function gateModel(confirmed: boolean) {
  return {
    key: KEY,
    name: "MiniMax H3",
    kinds: ["video"],
    excludedTerritories: ["European Union", "United Kingdom", "Republic of Korea", "United States"],
    licenceName: "MiniMax H3 Community License",
    licenceUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/LICENSE",
    liveLicenceUrl: LIVE_LICENCE,
    summaryLines: ["Licensed only outside the European Union, the United Kingdom, the Republic of Korea and the United States."],
    confirmSentence: SENTENCE,
    note: `${NOTICE} It runs on the stable-diffusion.cpp runtime (sd), which release builds may not include. Not tested by the maintainers.`,
    confirmed,
    confirmedAt: confirmed ? "2026-11-14T10:00:00+0000" : null,
  };
}

const H3_FAMILY = {
  id: "minimax-h3",
  name: "MiniMax H3",
  provider: "MiniMax",
  headline: "Video with stereo sound.",
  summary: "Territory-limited.",
  updatedLabel: "sd.cpp",
  badges: [],
  defaultVariantId: H3_ID,
  variants: [
    {
      id: H3_ID,
      familyId: "minimax-h3",
      name: "MiniMax H3 · FL2VA pruned GGUF Q4_K_M (sd.cpp)",
      provider: "MiniMax · leejet",
      repo: "leejet/MiniMax-H3-GGUF",
      link: "https://huggingface.co/leejet/MiniMax-H3-GGUF",
      runtime: "stable-diffusion.cpp (MiniMax H3)",
      styleTags: ["general", "audio"],
      taskSupport: ["txt2video"],
      territoryGate: KEY,
      licence: {
        id: "minimax-h3-community",
        name: "MiniMax H3 Community License",
        commercial: "limited",
        gated: false,
        territoryRestricted: true,
        summary: "Only outside the EU, the UK, the Republic of Korea and the USA.",
      },
      sizeGb: 35.5,
      recommendedResolution: "864x480",
      note: "Not tested by the maintainers.",
      availableLocally: false,
      hasLocalData: false,
      releaseDate: "2026-08",
    },
  ],
};

async function mockGate(page: Page) {
  const state = { confirmed: false, confirmBody: null as unknown };
  const status = () => ({ models: [gateModel(state.confirmed)] });
  await page.route("**/api/licence-gate", (route: Route) => route.fulfill({ json: status() }));
  await page.route("**/api/licence-gate/confirm", async (route: Route) => {
    state.confirmBody = route.request().postDataJSON();
    state.confirmed = true;
    await route.fulfill({ json: status() });
  });
  await page.route("**/api/licence-gate/withdraw", async (route: Route) => {
    state.confirmed = false;
    await route.fulfill({ json: status() });
  });
  await page.route("**/api/video/catalog", async (route: Route) => {
    const response = await route.fetch();
    const body = await response.json();
    const families = state.confirmed ? [...body.families, H3_FAMILY] : body.families;
    await route.fulfill({ json: { ...body, families, territoryLimited: [KEY] } });
  });
  return state;
}

async function walkThrough(page: Page) {
  const state = await mockGate(page);
  await goTo(page, "Video", "Models");
  const row = page.locator(`[data-model-id="${H3_ID}"]`);
  await expect(row).toHaveCount(0);

  const notice = page.locator("details.territory-gate-notice");
  await expect(notice).toBeVisible();
  await notice.locator("summary", { hasText: "Models with territory limits" }).click();
  await expect(notice).toContainText("Hidden until you confirm");
  await notice.getByRole("button", { name: "Review" }).click();

  const dialog = page.getByRole("dialog", { name: "MiniMax H3" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Not licensed for use in: European Union, United Kingdom, Republic of Korea, United States.");
  await expect(dialog).toContainText("does not check where you are");
  // D-99: the hosted-service notice, and the current licence linked beside the pinned one.
  await expect(dialog).toContainText("you are offering a hosted service");
  await expect(dialog.getByRole("button", { name: "Read the full licence" })).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Read the current licence on Hugging Face" })).toBeVisible();
  const confirm = dialog.getByRole("button", { name: "Confirm" });
  await expect(confirm).toBeDisabled();

  // Escape is "Not now": nothing is stored.
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  expect(state.confirmed).toBe(false);

  await notice.getByRole("button", { name: "Review" }).click();
  await dialog.getByRole("checkbox", { name: SENTENCE }).check();
  await expect(confirm).toBeEnabled();
  await confirm.click();
  await expect(dialog).toBeHidden();
  expect(state.confirmBody).toEqual({ key: KEY, confirmed: true, sentence: SENTENCE });

  await expect(row).toBeVisible();
  await expect(row.getByText("Territory-limited")).toBeVisible();
  await expect(notice).toContainText("Confirmed: listed below");

  await notice.getByRole("button", { name: "Withdraw" }).click();
  await expect(row).toHaveCount(0);
  await expect(notice.getByRole("button", { name: "Review" })).toBeVisible();
  // A background catalog refresh may still be passing through the mock.
  await page.unrouteAll({ behavior: "ignoreErrors" });
}

test("territory notice, tick-to-confirm screen, rows appear and Withdraw hides them", async ({ page }) => {
  await walkThrough(page);
});

test.describe("at 375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("the same holds on a phone, with no sideways scroll", async ({ page }) => {
    await walkThrough(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  });
});
