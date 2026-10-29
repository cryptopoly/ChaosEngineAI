import { renderToStaticMarkup } from "react-dom/server";
import i18next from "i18next";
import { describe, expect, it, vi } from "vitest";
import type { DownloadStatus } from "../../../api";
import type { HubModel, LibraryItem, ModelFamily, ModelVariant } from "../../../types";
import { buildChatEntries, type ChatModelEntry } from "../../../components/models/adapters/chat";
import { resolveModelStatus, sortModelRows, type ModelRow } from "../../../components/models";
import {
  MAC,
  WINDOWS_CUDA,
  makeChatFamily,
  makeChatVariant,
  makeHubModel,
  makeLibraryItem,
} from "../../../components/models/__tests__/chatFixtures";
import { ChatModelsPage, chatLaunchActions, chatSortOptions, type ChatModelsPageProps } from "../ChatModelsPage";
import { hubLaunchKey, sortHubResults } from "../ChatHubResults";

const t = i18next.getFixedT("en", "library");

function render(overrides: Partial<ChatModelsPageProps> = {}): string {
  const props: ChatModelsPageProps = {
    families: [],
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
    accelCompat: { dflashModels: [], mtplxModels: [], tensorfoldModels: [], turboInstalled: false, turboquantMlxAvailable: false, appleSilicon: true },
    hostSystem: MAC,
    enabledDirectoryCount: 2,
    fileRevealLabel: "Show in Finder",
    onDownloadModel: vi.fn(),
    onCancelModelDownload: vi.fn(),
    onDeleteModelDownload: vi.fn(),
    onDeleteModel: vi.fn(),
    onPrepareCatalogConversion: vi.fn(),
    onOpenModelSelector: vi.fn(),
    onOpenFamilyDetails: vi.fn(),
    onOpenExternalUrl: vi.fn(),
    onRevealPath: vi.fn(),
    ...overrides,
  };
  return renderToStaticMarkup(<ChatModelsPage {...props} />);
}

/** The markup of one row, found by its data-model-id. */
function rowMarkup(markup: string, id: string): string {
  const start = markup.indexOf(`data-model-id="${id}"`);
  expect(start, `row ${id} not rendered`).toBeGreaterThan(-1);
  const next = markup.indexOf("data-model-id=", start + 1);
  const hub = markup.indexOf("chat-hub-results", start);
  const ends = [next, hub].filter((index) => index > -1);
  return markup.slice(start, ends.length ? Math.min(...ends) : undefined);
}

// ── Fixtures from the old Discover tests ────────────────────────────────

function coderVariant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return makeChatVariant({
    id: "Qwen/Qwen3-Coder-Next-FP8",
    familyId: "qwen3-coder",
    name: "Qwen3 Coder Next FP8",
    repo: "Qwen/Qwen3-Coder-Next-FP8",
    link: "https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8",
    paramsB: 75,
    sizeGb: 74.9,
    format: "Transformers",
    quantization: "FP8",
    capabilities: ["coding", "agents", "tool-use"],
    note: "Official FP8 repo.",
    contextWindow: "256K",
    estimatedMemoryGb: 60.0,
    estimatedCompressedMemoryGb: 40.0,
    launchMode: "convert",
    backend: "mlx",
    ...overrides,
  });
}

function coderFamily(variant: ModelVariant): ModelFamily {
  return makeChatFamily(
    {
      id: "qwen3-coder",
      name: "Qwen3 Coder",
      headline: "Code-specialised Qwen3 with strong agentic support.",
      capabilities: ["coding", "agents", "tool-use"],
      readme: ["Official Next repos replace older provisional placeholders."],
    },
    [variant],
  );
}

function renderCatalogDownload(download: DownloadStatus, extra: Partial<ChatModelsPageProps> = {}): string {
  const variant = coderVariant();
  return render({ families: [coderFamily(variant)], activeDownloads: { [variant.repo]: download }, ...extra });
}

const glm = makeHubModel();

describe("ChatModelsPage: catalog rows (old Discover tests)", () => {
  it("shows a preparing label instead of a misleading 0% download state", () => {
    const markup = renderCatalogDownload({ repo: "Qwen/Qwen3-Coder-Next-FP8", state: "downloading", progress: 0, downloadedGb: 0, totalGb: 74.9, error: null });
    expect(markup).toContain("Preparing download...");
    expect(markup).not.toContain("Downloading 0%");
  });

  it("renders inline failed-download details with retry actions", () => {
    const markup = renderCatalogDownload({
      repo: "Qwen/Qwen3-Coder-Next-FP8",
      state: "failed",
      progress: 0,
      downloadedGb: 0,
      totalGb: 74.9,
      error: "Qwen/Qwen3-Coder-Next-FP8 was not found on Hugging Face.",
    });
    const row = rowMarkup(markup, "catalog:Qwen/Qwen3-Coder-Next-FP8");
    expect(row).toContain("Qwen/Qwen3-Coder-Next-FP8 was not found on Hugging Face.");
    expect(row).toContain('aria-label="Retry download"');
    expect(row).toContain('aria-label="Delete download"');
  });

  it("shows in-flight progress on a collapsed row", () => {
    const markup = renderCatalogDownload({ repo: "Qwen/Qwen3-Coder-Next-FP8", state: "downloading", progress: 0.42, downloadedGb: 31, totalGb: 74.9, error: null });
    const row = rowMarkup(markup, "catalog:Qwen/Qwen3-Coder-Next-FP8");
    expect(row).toContain("Downloading 42%");
    expect(row).toContain('aria-label="Pause download"');
    expect(row).not.toContain("models-card__detail");
  });

  it("lists available models from the catalog with a Download button", () => {
    const variant = coderVariant();
    const row = rowMarkup(render({ families: [coderFamily(variant)] }), `catalog:${variant.id}`);
    expect(row).toContain('aria-label="Download model"');
    expect(row).toContain('data-status="available"');
  });
});

describe("ChatModelsPage: fit verdict (R14)", () => {
  function renderFit(variant: ModelVariant, availableMemoryGb: number | null): string {
    return rowMarkup(render({ families: [coderFamily(variant)], availableMemoryGb }), `catalog:${variant.id}`);
  }

  it("shows the backend fit verdict instead of the free-RAM guess", () => {
    const variant = {
      ...coderVariant({ estimatedMemoryGb: 60 }),
      fit: { status: "tight", needGb: 44, usableGb: 51.8, headroomPct: 15, reason: "Weights 40 GB + 4 GB cache", basis: "estimated" },
    } as ModelVariant;
    // 60 GB against 16 GB free would be "Too big" by the old guess.
    const row = renderFit(variant, 16);
    expect(row).toContain("fit-badge--tight");
    expect(row).toContain("Weights 40 GB + 4 GB cache");
    expect(row).not.toContain("memory-fit-badge--over");
  });

  it("falls back to the free-RAM guess when the backend sends no verdict", () => {
    const row = renderFit(coderVariant({ estimatedMemoryGb: 60 }), 16);
    expect(row).toContain("memory-fit-badge--over");
    expect(row).not.toContain('class="fit-badge ');
  });

  it("marks the family's best fit with the chat explanation", () => {
    const variant = coderVariant();
    const family = { ...coderFamily(variant), bestFitVariantId: variant.id } as ModelFamily;
    const row = rowMarkup(render({ families: [family] }), `catalog:${variant.id}`);
    expect(row).toContain("best-fit-ribbon");
    expect(row).toContain("among sizes released together, the largest that fits");
  });
});

describe("ChatModelsPage: Hugging Face results", () => {
  const unrelated = coderFamily(coderVariant());

  it("renders live hub results even when no catalog or local model matches", () => {
    const markup = render({ families: [unrelated], searchInput: "glm", hubResults: [glm] });
    expect(markup).toContain("HuggingFace Hub");
    expect(markup).toContain("GLM-4.7-Flash");
    expect(markup).toContain("Results from Hugging Face are below.");
    expect(markup).not.toContain('No models match "glm"');
  });

  it("orders live hub results by most recent update first", () => {
    const older = makeHubModel({ id: "org/older-model", repo: "org/older-model", name: "older-model", lastModified: "2026-04-10T10:00:00Z", downloads: 1000 });
    const newer = makeHubModel({ id: "org/newer-model", repo: "org/newer-model", name: "newer-model", lastModified: "2026-04-16T10:00:00Z", downloads: 10 });
    const markup = render({ families: [unrelated], searchInput: "model", hubResults: [older, newer] });
    expect(markup.indexOf("newer-model")).toBeLessThan(markup.indexOf("older-model"));
    expect(markup).toContain("sorted by most recent update");
    expect(sortHubResults([older, newer], null).map((model) => model.id)).toEqual(["org/newer-model", "org/older-model"]);
    expect(sortHubResults([older, newer], "GGUF")).toEqual([]);
  });

  it("shows in-flight download progress in a collapsed hub card", () => {
    const markup = render({
      families: [unrelated],
      searchInput: "glm",
      hubResults: [glm],
      activeDownloads: { [glm.repo]: { repo: glm.repo, state: "downloading", progress: 0.17, downloadedGb: 3.2, totalGb: 19, error: null } },
    });
    expect(markup).toContain("Downloading 17%");
  });

  it("collapses large weight shard lists by default in an expanded hub card", () => {
    const repo = "org/big-model";
    const big = makeHubModel({ id: repo, repo, name: "big-model" });
    const markup = render({
      families: [unrelated],
      searchInput: "big",
      hubResults: [big],
      expandedHubId: repo,
      hubFileCache: {
        [repo]: {
          repo,
          files: Array.from({ length: 12 }, (_, index) => ({
            path: `model-${String(index).padStart(5, "0")}-of-00012.safetensors`,
            sizeBytes: 9_500_000_000,
            sizeGb: 9.5,
            kind: "weight" as const,
          })),
          totalSizeBytes: 12 * 9_500_000_000,
          totalSizeGb: 12 * 9.5,
          license: "apache-2.0",
          tags: ["text-generation", "safetensors"],
          pipelineTag: "text-generation",
          lastModified: "2026-04-16T10:00:00Z",
        },
      },
    });
    expect(markup).toContain('<details class="hub-file-group hub-file-group--collapsible">');
    expect(markup).toContain("Weights (12)");
    expect(markup).toContain("Largest shard 9.5 GB");
  });

  it("preselects the hub model's own files in the launch modal (R3)", () => {
    const local = makeLibraryItem({ name: "zai-org/GLM-4.7-Flash", path: "/cache/hub/models--zai-org--GLM-4.7-Flash", format: "MLX" });
    expect(hubLaunchKey(glm, [local], [])).toBe(`library:${local.path}`);
    const variant = makeChatVariant({ id: "glm-flash", repo: glm.repo });
    expect(hubLaunchKey(glm, [], [makeChatFamily({}, [variant])])).toBe("catalog:glm-flash");
    expect(hubLaunchKey(glm, [], [])).toBeUndefined();
  });
});

// ── Fixtures from the old My Models tests ───────────────────────────────

function bf16Variant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return makeChatVariant({
    id: "Qwen/Qwen3.6-35B-A3B",
    familyId: "qwen-3-6",
    name: "Qwen3.6 35B A3B",
    repo: "Qwen/Qwen3.6-35B-A3B",
    link: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    paramsB: 35,
    sizeGb: 67,
    format: "Transformers",
    quantization: "BF16",
    capabilities: ["reasoning", "coding", "vision", "agents"],
    note: "Hybrid reasoning checkpoint.",
    contextWindow: "262K",
    estimatedMemoryGb: 30.3,
    estimatedCompressedMemoryGb: 20.6,
    launchMode: "convert",
    backend: "mlx",
    ...overrides,
  });
}

function hfCacheItem(overrides: Partial<LibraryItem> = {}): LibraryItem {
  return makeLibraryItem({
    name: "Qwen/Qwen3.6-35B-A3B",
    path: "/Users/dan/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B",
    format: "HF cache",
    sourceKind: "HF cache",
    quantization: "BF16",
    backend: "mlx",
    sizeGb: 4.7,
    broken: true,
    brokenReason: "No .gguf, .safetensors, or pytorch weights found in HF cache entry",
    ...overrides,
  });
}

function renderLocal(item: LibraryItem, variant: ModelVariant, activeDownloads: Record<string, DownloadStatus> = {}): string {
  const markup = render({ families: [makeChatFamily({ id: variant.familyId }, [variant])], library: [item], activeDownloads });
  return rowMarkup(markup, `catalog:${variant.id}`);
}

describe("ChatModelsPage: local files (old My Models tests)", () => {
  it("shows resumable progress for partial HF cache entries before a download resumes", () => {
    const row = renderLocal(
      hfCacheItem({ sizeGb: 23.9, brokenReason: "Hugging Face download is incomplete: partial blob files are still present." }),
      bf16Variant({ sizeGb: 67 }),
    );
    expect(row).toContain("Paused 36%");
    expect(row).toContain("23.9 / 67.0 GB downloaded.");
    expect(row).toContain('aria-label="Resume download"');
    expect(row).toContain('aria-label="Delete download"');
    expect(row).not.toContain("BROKEN");
  });

  it("shows active download progress instead of a broken badge for partial HF cache entries", () => {
    const row = renderLocal(hfCacheItem(), bf16Variant(), {
      "Qwen/Qwen3.6-35B-A3B": { repo: "Qwen/Qwen3.6-35B-A3B", state: "downloading", progress: 0.125, downloadedGb: 8.4, totalGb: 66.99, error: null },
    });
    expect(row).toContain("Downloading 13%");
    expect(row).toContain("8.4 / 67.0 GB downloaded.");
    expect(row).toContain('aria-label="Pause download"');
    expect(row).toContain('aria-label="Cancel download"');
    expect(row).not.toContain("BROKEN");
    expect(row).not.toContain("No .gguf, .safetensors, or pytorch weights found");
  });

  it("still shows the broken badge when there is no active download for the HF cache entry", () => {
    const row = renderLocal(hfCacheItem(), bf16Variant());
    expect(row).toContain("BROKEN");
    expect(row).toContain("No .gguf, .safetensors, or pytorch weights found in HF cache entry");
    expect(row).toContain('aria-label="Retry download"');
    expect(row).toContain('aria-label="Delete download"');
    expect(row).toContain('aria-label="Delete model"');
    expect(row).not.toContain("Chat in a new thread");
    expect(row).not.toContain("Load for server");
  });

  it("renders download-only rows without a reveal button or a download:// path", () => {
    const repo = "org/hub-only-model";
    const markup = render({
      activeDownloads: { [repo]: { repo, state: "downloading", progress: 0.03, downloadedGb: 2.0, totalGb: 67, error: null } },
    });
    const row = rowMarkup(markup, `library:download://${repo}`);
    expect(row).toContain("Downloading 3%");
    expect(row).toContain('aria-label="Pause download"');
    expect(row).toContain('aria-label="Cancel download"');
    expect(row).not.toContain(`>download://${repo}`);
    expect(row).not.toContain("Show in Finder");
  });

  it("maps a local GGUF folder to its download repo for progress", () => {
    const variant = bf16Variant({
      id: "unsloth/Qwen3.6-35B-A3B-GGUF",
      repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
      name: "Qwen3.6-35B-A3B-GGUF",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: "Q4_K_S",
    });
    const item = makeLibraryItem({
      name: "Qwen3.6-35B-A3B-GGUF",
      path: "/Users/dan/AI_Models/unsloth/Qwen3.6-35B-A3B-GGUF",
      format: "GGUF",
      sourceKind: "Directory",
      quantization: "Q4_K_S",
      backend: "llama.cpp",
      sizeGb: 21.7,
      broken: true,
      brokenReason: "GGUF download is incomplete: main model weights are still downloading.",
    });
    const row = renderLocal(item, variant, {
      "unsloth/Qwen3.6-35B-A3B-GGUF": { repo: "unsloth/Qwen3.6-35B-A3B-GGUF", state: "downloading", progress: 0.01, downloadedGb: 2.55, totalGb: 494.14, error: null },
    });
    expect(row).toContain("Downloading 1%");
    expect(row).toContain("2.5 / 494.1 GB downloaded.");
    expect(row).toContain('aria-label="Pause download"');
    expect(row).toContain('aria-label="Cancel download"');
    expect(row).not.toContain("Chat in a new thread");
  });

  it("offers only Delete model and Reveal on a broken local model with no known repo", () => {
    const item = makeLibraryItem({
      name: "broken-thing",
      path: "/m/broken-thing",
      sourceKind: "Directory",
      format: "PyTorch",
      backend: null,
      quantization: null,
      broken: true,
      brokenReason: "Missing config.json",
    });
    const row = rowMarkup(render({ library: [item] }), `library:${item.path}`);
    expect(row).toContain("BROKEN");
    expect(row).toContain("Missing config.json");
    expect(row).toContain('aria-label="Delete model"');
    expect(row).toContain('aria-label="Show in Finder"');
    expect(row).not.toContain('aria-label="Retry download"');
    expect(row).not.toContain('aria-label="Delete download"');
  });

  it("shows source kind, directory label and disk totals for local models", () => {
    const item = makeLibraryItem({ name: "custom", path: "/m/custom", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null, directoryLabel: "External SSD", sizeGb: 10 });
    const markup = render({ library: [item], enabledDirectoryCount: 3 });
    const row = rowMarkup(markup, `library:${item.path}`);
    expect(row).toContain(">Directory</span>");
    expect(row).toContain("External SSD");
    expect(markup).toContain("10.0 GB on disk · 3 directories");
  });
});

describe("ChatModelsPage: launch actions (lead's decision)", () => {
  it("puts Chat in a new thread first and Use in this chat / Load for server in the menu of an installed row", () => {
    const item = makeLibraryItem();
    const variant = makeChatVariant();
    const row = rowMarkup(render({ families: [makeChatFamily({}, [variant])], library: [item] }), `catalog:${variant.id}`);
    expect(row).toContain('aria-label="Chat in a new thread"');
    expect(row).toContain(">Use in this chat</button>");
    expect(row).toContain(">Load for server</button>");
    expect(row).toContain('aria-label="Delete model"');
    expect(row).toContain('aria-label="Show in Finder"');
  });

  it("opens the launch modal with this row's model for each action", () => {
    const item = makeLibraryItem();
    const entries = buildChatEntries([makeChatFamily()], [item], {});
    const onOpen = vi.fn();
    const { launch, overflow } = chatLaunchActions(entries[0], onOpen, t);
    launch.onClick();
    overflow.forEach((action) => action.onClick());
    expect(onOpen.mock.calls).toEqual([
      ["thread", `library:${item.path}`],
      ["chat", `library:${item.path}`],
      ["server", `library:${item.path}`],
    ]);
    // A catalog row with no files keys by catalog id (the modal maps it to local files when it finds them).
    const catalogOnly = buildChatEntries([makeChatFamily({}, [makeChatVariant()])], [], {});
    chatLaunchActions(catalogOnly[0], onOpen, t).launch.onClick();
    expect(onOpen).toHaveBeenLastCalledWith("thread", "catalog:mlx-community/Qwen3-4B-4bit");
  });

  it("disables launching a model the host cannot run, but keeps it listed with a chip", () => {
    const item = makeLibraryItem();
    const markup = render({ families: [makeChatFamily()], library: [item], hostSystem: WINDOWS_CUDA });
    const row = rowMarkup(markup, "catalog:mlx-community/Qwen3-4B-4bit");
    expect(row).toContain("Cannot run on this machine");
    expect(row).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Chat in a new thread"|aria-label="Chat in a new thread"[^>]*disabled=""/);
    expect(row).toContain('aria-label="Delete model"');
  });

  it("offers Convert on an installed catalog model that needs converting (R4)", () => {
    const variant = coderVariant();
    const item = makeLibraryItem({
      name: "Qwen/Qwen3-Coder-Next-FP8",
      path: "/cache/hub/models--Qwen--Qwen3-Coder-Next-FP8",
      format: "Transformers",
      quantization: "FP8",
      sizeGb: 74.9,
    });
    const onConvert = vi.fn();
    const row = rowMarkup(render({ families: [coderFamily(variant)], library: [item], onPrepareCatalogConversion: onConvert }), `catalog:${variant.id}`);
    expect(row).toContain('aria-label="Convert model"');
    expect(rowMarkup(render({ families: [coderFamily(variant)] }), `catalog:${variant.id}`)).not.toContain("Convert model");
  });
});

describe("ChatModelsPage: platform gating (R19)", () => {
  it("hides catalog models the host cannot run when nothing of them is on disk", () => {
    const mlx = makeChatVariant();
    const gguf = makeChatVariant({ id: "org/m-GGUF", repo: "org/m-GGUF", name: "M GGUF", format: "GGUF", backend: "llama.cpp" });
    const markup = render({ families: [makeChatFamily({}, [mlx, gguf])], hostSystem: WINDOWS_CUDA });
    expect(markup).not.toContain(`data-model-id="catalog:${mlx.id}"`);
    expect(markup).toContain('data-model-id="catalog:org/m-GGUF"');
  });
});

describe("ChatModelsPage: favourites", () => {
  it("shows a star on rows with local files and lifts favourites to the top", () => {
    const first = makeLibraryItem({ name: "aaa-model", path: "/m/aaa-model", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null });
    const second = makeLibraryItem({ name: "zzz-model", path: "/m/zzz-model", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null });
    const markup = render({
      library: [first, second],
      sort: "name",
      sortDir: "asc",
      favoriteModelRefs: ["/m/zzz-model"],
      onToggleFavoriteModel: vi.fn(),
    });
    expect(markup.indexOf('data-model-id="library:/m/zzz-model"')).toBeLessThan(markup.indexOf('data-model-id="library:/m/aaa-model"'));
    expect(rowMarkup(markup, "library:/m/zzz-model")).toContain('aria-label="Remove from favourites"');
    expect(rowMarkup(markup, "library:/m/aaa-model")).toContain('aria-label="Mark as favourite"');
  });

  it("offers no star on a catalog model that is not on disk", () => {
    const variant = makeChatVariant();
    const row = rowMarkup(render({ families: [makeChatFamily({}, [variant])], onToggleFavoriteModel: vi.fn() }), `catalog:${variant.id}`);
    expect(row).not.toContain("favourite");
  });
});

describe("ChatModelsPage: search, filters and presets", () => {
  const custom = makeLibraryItem({ name: "custom-local", path: "/m/custom-local", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null, directoryLabel: "External SSD" });
  const variant = makeChatVariant();
  const families = [makeChatFamily({}, [variant])];

  it("filters catalog and local rows with one search box", () => {
    const markup = render({ families, library: [custom], searchInput: "external ssd" });
    expect(markup).toContain('data-model-id="library:/m/custom-local"');
    expect(markup).not.toContain(`data-model-id="catalog:${variant.id}"`);
    const byCatalog = render({ families, library: [custom], searchInput: "qwen3 4b" });
    expect(byCatalog).toContain(`data-model-id="catalog:${variant.id}"`);
    expect(byCatalog).not.toContain('data-model-id="library:/m/custom-local"');
  });

  it("opens on installed or available rows from the presets", () => {
    const installed = render({ families, library: [custom], availability: "installed" });
    expect(installed).toContain('data-model-id="library:/m/custom-local"');
    expect(installed).not.toContain(`data-model-id="catalog:${variant.id}"`);
    const available = render({ families, library: [custom], availability: "available" });
    expect(available).toContain(`data-model-id="catalog:${variant.id}"`);
    expect(available).not.toContain('data-model-id="library:/m/custom-local"');
  });

  it("filters by capability, format and backend on each row's own data", () => {
    const gguf = makeChatVariant({ id: "org/coder-GGUF", repo: "org/coder-GGUF", name: "Coder GGUF", format: "GGUF", backend: "llama.cpp", capabilities: ["coding"] });
    const all = [makeChatFamily({}, [variant, gguf])];
    const coding = render({ families: all, library: [custom], capFilter: "coding" });
    expect(coding).toContain('data-model-id="catalog:org/coder-GGUF"');
    expect(coding).not.toContain(`data-model-id="catalog:${variant.id}"`);
    // An unmatched local model has no capabilities, as on the old My Models tab.
    expect(coding).not.toContain('data-model-id="library:/m/custom-local"');
    const ggufOnly = render({ families: all, library: [custom], formatFilter: "GGUF" });
    expect(ggufOnly).toContain('data-model-id="catalog:org/coder-GGUF"');
    expect(ggufOnly).not.toContain(`data-model-id="catalog:${variant.id}"`);
    const llama = render({ families: all, library: [custom], backendFilter: "llama.cpp" });
    expect(llama).toContain('data-model-id="catalog:org/coder-GGUF"');
    expect(llama).not.toContain('data-model-id="library:/m/custom-local"');
  });

  it("hides the Apple-only acceleration chips on other machines", () => {
    const mac = render({ families });
    expect(mac).toContain(">MTPLX");
    expect(mac).toContain(">TensorFold");
    const pc = render({
      families,
      hostSystem: WINDOWS_CUDA,
      accelCompat: { dflashModels: [], mtplxModels: [], tensorfoldModels: [], turboInstalled: false, turboquantMlxAvailable: false, appleSilicon: false },
    });
    expect(pc).not.toContain(">MTPLX");
    expect(pc).toContain("TurboQuant · memory saver");
  });

  it("keeps Run from Hugging Face on the page", () => {
    expect(render({ families })).toContain("Run from Hugging Face");
  });
});

describe("chat sorts", () => {
  function rows(entries: ChatModelEntry[]): ModelRow<ChatModelEntry>[] {
    return entries.map((entry) => ({ entry, status: resolveModelStatus(entry, { download: entry.source.download }) }));
  }
  const older = makeLibraryItem({ name: "older", path: "/m/older", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null, lastModified: "2026-01-01 10:00", sizeGb: 5 });
  const newer = makeLibraryItem({ name: "newer", path: "/m/newer", sourceKind: "Directory", format: "PyTorch", backend: null, quantization: null, lastModified: "2026-06-01 10:00", sizeGb: 2 });
  const shortCtx = makeChatVariant({ id: "short", repo: "org/short", name: "Short", contextWindow: "8K", estimatedCompressedMemoryGb: 1 });
  const longCtx = makeChatVariant({ id: "long", repo: "org/long", name: "Long", contextWindow: "128K", estimatedCompressedMemoryGb: 9 });
  const entries = buildChatEntries([makeChatFamily({}, [shortCtx, longCtx])], [older, newer], {});
  const options = chatSortOptions(t);
  const sortBy = (id: string, dir: "asc" | "desc") =>
    sortModelRows(rows(entries), options.find((option) => option.id === id), dir).map((row) => row.entry.name);

  it("sorts by recently modified with catalog-only rows last", () => {
    expect(sortBy("modified", "desc").slice(0, 2)).toEqual(["newer", "older"]);
    expect(sortBy("modified", "asc").slice(0, 2)).toEqual(["older", "newer"]);
    expect(sortBy("modified", "asc").slice(2).sort()).toEqual(["Long", "Short"]);
  });

  it("sorts by context and compressed RAM", () => {
    expect(sortBy("context", "desc")[0]).toBe("Long");
    expect(sortBy("compressed", "desc")[0]).toBe("Long");
    expect(sortBy("compressed", "asc")[0]).toBe("Short");
  });
});
