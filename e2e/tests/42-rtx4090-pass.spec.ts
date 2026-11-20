import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openProject, seedProject, writeFixtureMedia } from "./helpers";
import { ttsModels, voiceList } from "./voiceMocks";

// UI flows built on the Windows / RTX 4090 pass that had unit tests only (coverage audit 2026-11-20,
// docs/review/test-coverage-2026-11-20.md):
//   - Music > Studio: the Copy now panel (D-100) offered on a 409 that carries the copy facts, its progress, Done and
//     the load tried again, and Cancel back to "Copy again". Every music route is mocked: nothing is copied or run.
//   - Voice > Models on a PC (D-101, D-102 A): the Mac-only speech rows folded behind one closed line, the GPU row
//     shown without "(GPU)", and the Word timings (forced aligner) card whose Download starts the pinned repo and shows
//     a progress bar with Cancel. Voice routes and the model download routes are mocked: nothing downloads.
//   - Storyboard shot editor (D-103): the "Vocal for lip sync" field attaches a WAV through the real backend
//     (set_vocal_clip), shows a player, and Remove vocal clears it. No model runs.
// The MiniMax H3 tick, live licence link and hosted-service notice (D-99) are in 24-territory-gate.spec.ts.

const GB = 1024 ** 3;

// ---------------------------------------------------------------------------------------------------------------
// Copy now (D-100)
// ---------------------------------------------------------------------------------------------------------------

const COPY_FACTS = {
  message: "Setting up the music model needs 6 weight file(s) copied: this PC did not allow a link.",
  needsCopy: true,
  fileCount: 6,
  totalBytes: 14.3 * GB,
  source: "E:\\AI Models\\hf\\snapshots",
  target: "C:\\Users\\me\\.chaosengine\\music\\checkpoints",
  sourceDrive: "E:",
  targetDrive: "C:",
};

function copyStatus(phase: string, bytesDone: number, over: Record<string, unknown> = {}) {
  return {
    phase, running: phase === "running", bytesDone, totalBytes: COPY_FACTS.totalBytes, fileCount: 6, filesDone: 0,
    currentFile: phase === "running" ? "model.safetensors" : "", error: null, etaSeconds: phase === "running" ? 120 : null,
    ...over,
  };
}

async function mockMusicNeedingCopy(page: Page) {
  const seen = { generate: 0, copyStarts: 0, cancels: 0, polls: 0 };
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
    Object.assign(body.runtime ?? body, { installed: true, unavailableReason: null });
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/estimate", (route) =>
    route.fulfill({
      json: {
        estimate: { model: "ACE-Step/Ace-Step1.5", seconds: 60, coldStartSeconds: 20, warm: false, peakGb: 10, idleGb: 5, writesMb: 12, modelDiskGb: 10, basis: "estimated" },
        fit: null, totalGb: 64, queue: { running: null, waiting: [] },
      },
    }),
  );
  await page.route("**/api/music/generate", (route) => {
    seen.generate += 1;
    // The first load needs the copy; the retry after the copy stops with a plain 409, so no track has to be mocked.
    const detail = seen.generate === 1 ? COPY_FACTS : "The music model is busy in this mock.";
    return route.fulfill({ status: 409, json: { detail } });
  });
  return seen;
}

async function startTrackThatNeedsCopy(page: Page) {
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Kind of track" }).getByRole("button", { name: "Instrumental" }).click();
  await page.getByRole("textbox", { name: /Describe the music/ }).fill("warm lo-fi piano, 85 bpm");
  await page.locator(".run-estimate").getByRole("button", { name: "Make track" }).click();
  const panel = page.getByRole("group", { name: "The music model's files need to be copied" });
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("6 files (14.3 GB)");
  await expect(panel).toContainText("Developer Mode");
  await expect(panel).toContainText("same drive as the app's data (C:)");
  return panel;
}

test("Copy now: the offer, the progress bar, Done and the load tried again", async ({ page }) => {
  const seen = await mockMusicNeedingCopy(page);
  await page.route("**/api/music/weights/copy", (route) => {
    seen.copyStarts += 1;
    return route.fulfill({ json: copyStatus("running", 0) });
  });
  await page.route("**/api/music/weights/copy/status", (route) => {
    seen.polls += 1;
    return route.fulfill({ json: seen.polls < 3 ? copyStatus("running", 7.15 * GB) : copyStatus("done", COPY_FACTS.totalBytes) });
  });
  const panel = await startTrackThatNeedsCopy(page);
  await panel.getByRole("button", { name: "Copy now (14.3 GB, about 2 min)" }).click();
  await expect(panel.getByRole("progressbar", { name: "Copy progress" })).toBeVisible();
  await expect(panel).toContainText("7.2 of 14.3 GB · 50%");
  await expect(panel).toContainText("model.safetensors");
  await expect(panel.getByRole("button", { name: "Cancel" })).toBeVisible();
  // Done: the panel goes and the load that stopped is tried again by itself.
  await expect.poll(() => seen.generate, { timeout: 10_000 }).toBe(2);
  expect(seen.copyStarts).toBe(1);
  await expect(page.getByText("The music model is busy in this mock.", { exact: true })).toBeVisible();
  await expect(panel).toHaveCount(0);
});

test("Copy now: Cancel goes back to Copy again and says the finished files are kept", async ({ page }) => {
  const seen = await mockMusicNeedingCopy(page);
  let cancelled = false;
  await page.route("**/api/music/weights/copy", (route) => route.fulfill({ json: copyStatus("running", 0) }));
  await page.route("**/api/music/weights/copy/status", (route) =>
    route.fulfill({ json: cancelled ? copyStatus("cancelled", 3 * GB) : copyStatus("running", 3 * GB) }),
  );
  await page.route("**/api/music/weights/copy/cancel", (route) => {
    seen.cancels += 1;
    cancelled = true;
    return route.fulfill({ json: copyStatus("cancelled", 3 * GB) });
  });
  const panel = await startTrackThatNeedsCopy(page);
  await panel.getByRole("button", { name: /^Copy now/ }).click();
  await panel.getByRole("button", { name: "Cancel" }).click();
  await expect(panel.getByRole("button", { name: "Copy again (14.3 GB, about 2 min)" })).toBeVisible();
  await expect(panel).toContainText("Finished files are kept");
  expect(seen.cancels).toBe(1);
  expect(seen.generate).toBe(1);
});

// ---------------------------------------------------------------------------------------------------------------
// Voice on a PC: Mac-only fold line and the Word timings card (D-101, D-102 A)
// ---------------------------------------------------------------------------------------------------------------

const ALIGNER = {
  key: "qwen3-forced-aligner-0.6b", id: "Qwen/Qwen3-ForcedAligner-0.6B-hf", name: "Qwen3-ForcedAligner 0.6B",
  sizeGb: 1.84, licence: "Apache-2.0", vramGb: 1.9, installed: false,
};

async function mockVoiceOnAPc(page: Page) {
  const seen = { downloads: [] as string[], cancels: [] as string[] };
  let downloading = false;
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, {
      installed: true, lane: "portable", sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true,
      extras: [
        { id: "kokoro", label: "Kokoro quick voices", installed: true },
        { id: "stt-gpu", label: "Speech-to-text on the GPU", installed: true },
      ],
      sttModels: [
        {
          id: "Qwen/Qwen3-ASR-1.7B-hf", key: "qwen3-asr-1.7b", name: "Qwen3-ASR 1.7B", languages: "30 languages", sizeGb: 4.09,
          installed: true, default: true, backend: "transformers-asr", licence: "Apache-2.0", minVramGb: 5, vramBasis: "measured", gpuFacts: null,
        },
        { id: "org/whisper", key: "whisper-large-v3-turbo", name: "Whisper Large v3 Turbo", languages: "99 languages", sizeGb: 1.6, installed: true, default: false, backend: "faster-whisper", licence: "MIT" },
      ],
      aligner: ALIGNER,
    });
    await route.fulfill({ response, json: body });
  });
  // Five Mac-only rows (MLX lane, not downloaded) and one GPU row: on the portable lane the five fold away.
  const macRows = ttsModels([]).map((row) => ({ ...row, lanes: ["mlx"], supported: false }));
  const gpuRow = { ...ttsModels(["qwen3-tts-custom-1.7b"])[3], key: "qwen3-tts-custom-1.7b-cuda", name: "Qwen3-TTS 1.7B CustomVoice (GPU)", lanes: ["cuda"], supported: true };
  await page.route("**/api/voice/models", (route) => route.fulfill({ json: { models: [...macRows, gpuRow], available: true } }));
  await page.route("**/api/voice/voices", (route) => route.fulfill({ json: { voices: voiceList([], ["qwen3-tts-custom-0.6b"]) } }));
  await page.route("**/api/voice/clone/models", (route) => route.fulfill({ json: { available: true, allowedHere: true, models: [] } }));
  const job = () => ({ repo: ALIGNER.id, state: downloading ? "downloading" : "cancelled", progress: 0.25, downloadedGb: 0.46, totalGb: 1.84, error: null });
  await page.route("**/api/models/download", (route) => {
    seen.downloads.push((route.request().postDataJSON() as { repo: string }).repo);
    downloading = true;
    return route.fulfill({ json: { download: job() } });
  });
  await page.route("**/api/models/download/status", (route) => route.fulfill({ json: { downloads: downloading ? [job()] : [] } }));
  await page.route("**/api/models/download/cancel", (route) => {
    seen.cancels.push((route.request().postDataJSON() as { repo: string }).repo);
    downloading = false;
    return route.fulfill({ json: { download: job() } });
  });
  return seen;
}

test("Voice > Models on a PC: Mac-only rows behind one closed line, the GPU row without (GPU)", async ({ page }) => {
  await mockVoiceOnAPc(page);
  await goTo(page, "Voice", "Models");
  const fold = page.locator("details.voice-models__mac-only");
  await expect(fold).toBeVisible();
  await expect(fold.locator("summary")).toHaveText("5 models for Apple Silicon Macs");
  await expect(fold).not.toHaveAttribute("open", "");
  await expect(page.locator('[data-model-id="qwen3-tts-0.6b"]')).toHaveCount(0);
  await fold.locator("summary").click();
  await expect(fold).toContainText("They do not run on this PC");
  await expect(fold).toContainText("Qwen3-TTS 0.6B Base");
  const gpu = page.locator('[data-model-id="qwen3-tts-custom-1.7b-cuda"]');
  await expect(gpu).toBeVisible();
  await expect(gpu).toContainText("Qwen3-TTS 1.7B CustomVoice");
  await expect(gpu).not.toContainText("(GPU)");
});

test("Voice > Models on a PC: the Word timings card downloads the aligner with a progress bar and Cancel", async ({ page }) => {
  const seen = await mockVoiceOnAPc(page);
  await goTo(page, "Voice", "Models");
  const card = page.locator('[data-stt-aligner="qwen3-forced-aligner-0.6b"]');
  await expect(card).toBeVisible();
  await expect(card).toContainText("Word timings");
  await expect(card).toContainText("Qwen3-ForcedAligner 0.6B · 1.84 GB");
  await card.getByRole("button", { name: "Download 1.84 GB" }).click();
  await expect.poll(() => seen.downloads).toEqual([ALIGNER.id]);
  const bar = card.getByRole("progressbar", { name: "Downloading Qwen3-ForcedAligner 0.6B" });
  await expect(bar).toBeVisible();
  await expect(card).toContainText("25% · 0.46 of 1.84 GB");
  await card.getByRole("button", { name: "Cancel" }).click();
  await expect.poll(() => seen.cancels).toEqual([ALIGNER.id]);
  await expect(bar).toHaveCount(0);
});

// ---------------------------------------------------------------------------------------------------------------
// Storyboard: the lip-sync vocal field (D-103), real backend
// ---------------------------------------------------------------------------------------------------------------

test("Storyboard shot editor: attach a vocal for lip sync, see its player, remove it", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Lip-sync vocal", shots: 1 });
  const { tone } = writeFixtureMedia();
  await openProject(page, project.name, "Storyboard");
  await page.locator(".shot-card").first().focus();
  await page.keyboard.press("Enter");
  const editor = page.locator(".shot-editor");
  await expect(editor).toBeVisible();
  const vocal = editor.locator("fieldset.shot-editor__vocal");
  await expect(vocal.locator("legend")).toHaveText("Vocal for lip sync");
  await expect(vocal).toContainText("lasts 3 to 8 s");
  await expect(vocal.locator("audio")).toHaveCount(0);
  await vocal.getByRole("textbox", { name: "Vocal file (WAV or FLAC)" }).fill(tone);
  await vocal.getByRole("button", { name: "Attach vocal" }).click();
  await expect(vocal.locator("audio.shot-editor__vocal-player")).toBeVisible();
  await expect(vocal.getByRole("button", { name: "Change vocal" })).toBeVisible();
  await vocal.getByRole("button", { name: "Remove vocal" }).click();
  await expect(vocal.locator("audio")).toHaveCount(0);
  await expect(vocal.getByRole("button", { name: "Attach vocal" })).toBeVisible();
});
