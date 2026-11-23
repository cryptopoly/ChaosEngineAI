import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { LipSyncAudioPicker } from "../LipSyncAudioPicker";
import {
  EMPTY_LIP_SYNC,
  lipSyncClip,
  lipSyncEstimateSeconds,
  lipSyncPayloadFields,
  lipSyncTakeCount,
  type LipSyncBlock,
} from "../lipSyncLogic";

// The 4-bit LTX-2.3 row's measured figures (backend_service/catalog/video_models.py).
const BLOCK: LipSyncBlock = {
  secondsPerShot: 118,
  secondsPerShotWarm: 70,
  estimateRef: { width: 768, height: 512, frames: 121 },
};

describe("lip-sync logic", () => {
  it("defaults to 3 takes for a sung vocal and 1 for speech (D-110)", () => {
    expect(lipSyncTakeCount(null, true)).toBe(3);
    expect(lipSyncTakeCount(null, false)).toBe(1);
    expect(lipSyncTakeCount(4, false)).toBe(4);
    expect(lipSyncTakeCount(9, true)).toBe(4);
  });

  it("follows the backend's 8k+1 and 3 to 8 s rule", () => {
    expect(lipSyncClip(124, 24)).toEqual({ frames: 121, seconds: 5, ok: true });
    expect(lipSyncClip(49, 24).ok).toBe(false);
    expect(lipSyncClip(241, 24).ok).toBe(false);
  });

  it("multiplies the estimate by the takes, as the backend does", () => {
    expect(lipSyncEstimateSeconds(BLOCK, 768, 512, 121, 1)).toBe(118);
    expect(lipSyncEstimateSeconds(BLOCK, 768, 512, 121, 3)).toBe(288);
    expect(lipSyncEstimateSeconds(BLOCK, 768, 512, 121, 3, true)).toBe(240);
    expect(lipSyncEstimateSeconds({ engine: "mlx" }, 768, 512, 121, 3)).toBeNull();
  });
});

describe("lip-sync request fields", () => {
  it("adds the vocal, start, singing and takes only with a vocal on a lip-sync row", () => {
    const sung = { audioPath: "C:/v.wav", start: 2.5, singing: true, takes: 4 };
    expect(lipSyncPayloadFields(true, sung)).toEqual({ audioPath: "C:/v.wav", audioStartSeconds: 2.5, lipSyncSinging: true, lipSyncTakes: 4 });
    expect(lipSyncPayloadFields(true, { ...sung, singing: false, takes: null })).toEqual({ audioPath: "C:/v.wav", audioStartSeconds: 2.5 });
    expect(lipSyncPayloadFields(false, sung)).toEqual({});
    expect(lipSyncPayloadFields(true, EMPTY_LIP_SYNC)).toEqual({});
  });
});

describe("LipSyncAudioPicker", () => {
  const render = (value = EMPTY_LIP_SYNC, numFrames = 121, hasStartImage = true) =>
    renderToStaticMarkup(
      <LipSyncAudioPicker block={BLOCK} value={value} onChange={() => undefined} numFrames={numFrames} fps={24}
        width={768} height={512} hasStartImage={hasStartImage} />,
    );

  it("offers a vocal file and nothing else until one is chosen", () => {
    const html = render();
    expect(html).toContain("Vocal for lip sync");
    expect(html).toContain("3 to 8 s");
    expect(html).not.toContain("Takes");
  });

  it("shows the start, the singing switch, the takes and the estimate for a sung vocal", () => {
    const html = render({ audioPath: "C:/v/vocals.wav", start: 1.2, singing: true, takes: null });
    expect(html).toContain("vocals.wav");
    expect(html).toContain("Sung vocal");
    expect(html).toContain("Default (3)");
    expect(html).toContain("4 (best quality)");
    expect(html).toContain("About 5 min for 3 takes of a 5.0 s clip");
  });

  it("warns about a clip outside 3 to 8 s and a missing start image", () => {
    const html = render({ audioPath: "v.wav", start: 0, singing: false, takes: 1 }, 241, false);
    expect(html).toContain("lip sync needs 3 to 8 s");
    expect(html).toContain("add one above");
  });
});
