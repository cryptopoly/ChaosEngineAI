import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ChatPick } from "../../../types";
import type { ModelFit } from "../../../types/models";
import { makeChatFamily, makeChatVariant, makeLibraryItem } from "../../../components/models/__tests__/chatFixtures";
import { ChatModelsPage, type ChatModelsPageProps } from "../ChatModelsPage";

// D-105 D / FU-457: with no chat model on disk Chat Models opens on the recommended rows, not on filters.

const FIT: ModelFit = { status: "fits", needGb: 10.4, usableGb: 24, headroomPct: 57, reason: "Weights + cache", basis: "estimated" };
const PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24, gpuUnifiedMemory: false };

const best = makeChatVariant({
  id: "org/best-GGUF", familyId: "best", name: "Best 27B GGUF", repo: "org/best-GGUF", format: "GGUF", backend: "llama.cpp",
  sizeGb: 9.6, fit: FIT,
});
const other = makeChatVariant({
  id: "org/other-GGUF", familyId: "other", name: "Other 9B GGUF", repo: "org/other-GGUF", format: "GGUF", backend: "llama.cpp",
  sizeGb: 5.5, fit: { ...FIT, needGb: 7.5, headroomPct: 69 },
});
const notPicked = makeChatVariant({
  id: "org/big-GGUF", familyId: "big", name: "Big 70B GGUF", repo: "org/big-GGUF", format: "GGUF", backend: "llama.cpp",
  sizeGb: 40, fit: { ...FIT, status: "wont-fit", needGb: 44, headroomPct: -83 },
});

function pick(variant: typeof best, tokS: number | null, basis: ChatPick["tokSBasis"]): ChatPick {
  return {
    variantId: variant.id, familyId: variant.familyId, name: variant.name, repo: variant.repo, sizeGb: variant.sizeGb,
    paramsB: variant.paramsB, availableLocally: false, fit: variant.fit as ModelFit, tokS, tokSBasis: basis,
    hardwareLabel: "NVIDIA GPU, 24 GB VRAM",
  };
}

const PICKS = [pick(best, 98.2, "estimated"), pick(other, 151.4, "measured")];

function render(overrides: Partial<ChatModelsPageProps> = {}): string {
  const props: ChatModelsPageProps = {
    families: [
      makeChatFamily({ id: "best", name: "Best" }, [best]),
      makeChatFamily({ id: "other", name: "Other" }, [other]),
      makeChatFamily({ id: "big", name: "Big" }, [notPicked]),
    ],
    library: [],
    activeDownloads: {},
    searchInput: "",
    onSearchInputChange: vi.fn(),
    searchError: null,
    hubResults: [],
    expandedHubId: null,
    onToggleHubExpand: vi.fn(),
    hubFileCache: {},
    hubFileLoading: {},
    hubFileError: {},
    capFilter: null,
    onCapFilterChange: vi.fn(),
    formatFilter: null,
    onFormatFilterChange: vi.fn(),
    accelFilter: null,
    onAccelFilterChange: vi.fn(),
    backendFilter: null,
    onBackendFilterChange: vi.fn(),
    sort: "release",
    sortDir: "desc",
    onSortChange: vi.fn(),
    availability: "all",
    onAvailabilityChange: vi.fn(),
    accelCompat: { dflashModels: [], mtplxModels: [], tensorfoldModels: [], turboInstalled: false, turboquantMlxAvailable: false, appleSilicon: false },
    hostSystem: PC,
    enabledDirectoryCount: 1,
    fileRevealLabel: "Show in Explorer",
    onDownloadModel: vi.fn(),
    onCancelModelDownload: vi.fn(),
    onDeleteModelDownload: vi.fn(),
    onDeleteModel: vi.fn(),
    onPrepareCatalogConversion: vi.fn(),
    onOpenModelSelector: vi.fn(),
    onOpenFamilyDetails: vi.fn(),
    onOpenExternalUrl: vi.fn(),
    onRevealPath: vi.fn(),
    recommendedPicks: PICKS,
    ...overrides,
  };
  return renderToStaticMarkup(<ChatModelsPage {...props} />);
}

describe("Chat Models first run", () => {
  it("opens on the recommended rows, best first, with no filter bar", () => {
    const markup = render();
    expect(markup).toContain("Recommended for this computer (NVIDIA GPU, 24 GB VRAM)");
    expect(markup.match(/data-testid="chat-recommended-row"/g)).toHaveLength(2);
    expect(markup.indexOf("Best 27B GGUF")).toBeLessThan(markup.indexOf("Other 9B GGUF"));
    expect(markup).not.toContain("Big 70B GGUF");
    expect(markup).not.toContain("models-filter-bar");
    expect(markup).toContain("Show all 3 models and filters");
    expect(markup).toContain('aria-label="Download model"');
  });

  it("names size, the GPU memory it needs and the speed with its basis", () => {
    const markup = render();
    expect(markup).toContain("9.6 GB download · needs about 10 GB of 24 GB GPU memory · about 98 tok/s (estimate)");
    expect(markup).toContain("151 tok/s, measured on this computer");
  });

  it("shows the full page once a chat model is on disk, or with a search or filter set", () => {
    const installed = makeLibraryItem({ name: "org/best-GGUF", path: "C:/models/org/best-GGUF" });
    for (const markup of [render({ library: [installed] }), render({ searchInput: "best" }), render({ capFilter: "coding" }), render({ recommendedPicks: [] })]) {
      expect(markup).not.toContain('data-testid="chat-recommended-row"');
    }
    // Chat's "Browse models" opens on "available": with nothing on disk that is still the first-run list.
    expect(render({ availability: "available" })).toContain('data-testid="chat-recommended-row"');
  });

  it("titles the memory column GPU memory on a card with its own memory, RAM elsewhere", () => {
    const pc = render({ recommendedPicks: [], hostSystem: PC });
    expect(pc).toContain(">GPU memory");
    const mac = render({ recommendedPicks: [], hostSystem: { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null } });
    expect(mac).not.toContain(">GPU memory");
    expect(mac).toContain(">RAM");
  });
});
