import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { emptyWorkspace as defaultWorkspace } from "../../../defaults";
import type { ChatPick, Recommendation } from "../../../types";
import { DashboardTab } from "../DashboardTab";

// D-105 B and E (FU-456, FU-457): the Dashboard names the catalogue's best-fit row with its own headroom, and a new
// user gets a "Start here" card.

const PICK: ChatPick = {
  variantId: "org/bonsai-GGUF", familyId: "bonsai", name: "Ternary Bonsai 2 27B GGUF", repo: "org/bonsai-GGUF",
  sizeGb: 7.8, paramsB: 27, availableLocally: false,
  fit: { status: "fits", needGb: 10.4, usableGb: 24, headroomPct: 57, reason: "x", basis: "estimated" },
  tokS: 233.4, tokSBasis: "measured", hardwareLabel: "NVIDIA GPU, 24 GB VRAM",
};

const SYSTEM = { ...defaultWorkspace.system, gpuVramTotalGb: 24, gpuUnifiedMemory: false, hardwareSummary: "Windows PC" };

function render(recommendation: Partial<Recommendation>, extra: Partial<Parameters<typeof DashboardTab>[0]> = {}): string {
  return renderToStaticMarkup(
    <DashboardTab
      system={SYSTEM}
      recommendation={{ ...defaultWorkspace.recommendation, title: "Recommended target: 14B GGUF", targetModel: "14B", ...recommendation }}
      runtime={defaultWorkspace.runtime}
      activity={[]}
      backendOnline
      onOpenChatModels={() => undefined}
      {...extra}
    />,
  );
}

describe("Dashboard best fit", () => {
  it("names the row, its download size, the GPU memory it needs and the measured speed", () => {
    const markup = render({ bestFit: PICK, chatPicks: [PICK], headroomPercent: 57, targetModel: PICK.name });
    expect(markup).toContain("Ternary Bonsai 2 27B GGUF");
    expect(markup).toContain("7.8 GB download · needs about 10 GB of 24 GB GPU memory · 233 tok/s, measured on this computer");
    expect(markup).toContain("Headroom for Ternary Bonsai 2 27B GGUF");
    expect(markup).not.toContain("14B GGUF");
    // The headroom bar appears once, not twice (Live stats no longer repeats it).
    expect(markup.match(/Headroom for/g)).toHaveLength(1);
  });

  it("without a pick keeps the size class and shows no headroom bar", () => {
    const markup = render({ bestFit: null, chatPicks: [], headroomPercent: null });
    expect(markup).toContain("Recommended target: 14B GGUF");
    expect(markup).not.toContain("Headroom for");
  });

  it("opens on Start here when no chat model is on disk, and not otherwise", () => {
    const fresh = render({ bestFit: PICK, headroomPercent: 57 }, { hasChatModel: false, onDownloadModel: () => undefined, onOpenChat: () => undefined });
    expect(fresh).toContain("Start here");
    expect(fresh).toContain("NVIDIA GPU, 24 GB VRAM");
    expect(fresh).toContain("Download (7.8 GB)");
    expect(fresh).toContain("Open Chat");
    const downloading = render(
      { bestFit: PICK, headroomPercent: 57 },
      {
        hasChatModel: false, onDownloadModel: () => undefined,
        activeDownloads: { [PICK.repo]: { repo: PICK.repo, state: "downloading", progress: 0.42, downloadedGb: 3.3, totalGb: 7.8, error: null } },
      },
    );
    expect(downloading).toContain("Downloading, 42%");
    expect(render({ bestFit: PICK }, { hasChatModel: true })).not.toContain("Start here");
    expect(render({ bestFit: PICK }, { hasChatModel: undefined })).not.toContain("Start here");
  });
});
