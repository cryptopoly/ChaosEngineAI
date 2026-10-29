import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { DownloadStatus } from "../../../api";
import type { VideoModelFamily, VideoModelVariant } from "../../../types";
import type { ModelAvailabilityFilter } from "../../../components/models";
import {
  CUDA_PC,
  MAC,
  downloadState,
  makeLongLiveVariant,
  makeVideoFamily,
  makeVideoVariant,
  wan14bRows,
} from "../../../components/models/__tests__/videoFixtures";
import { VideoModelsPage, type VideoModelsPageProps } from "../VideoModelsPage";

function render(overrides: Partial<VideoModelsPageProps> = {}): string {
  const props: VideoModelsPageProps = {
    catalog: [],
    latest: [],
    searchInput: "",
    onSearchInputChange: vi.fn(),
    taskFilter: "all",
    onTaskFilterChange: vi.fn(),
    sort: "release",
    onSortChange: vi.fn(),
    availability: "all",
    onAvailabilityChange: vi.fn(),
    activeVideoDownloads: {},
    selectedVideoVariant: null,
    fileRevealLabel: "Show in Finder",
    hostSystem: MAC,
    longLiveStatus: null,
    installingLongLive: false,
    longLiveJob: null,
    onActiveTabChange: vi.fn(),
    onOpenVideoStudio: vi.fn(),
    onVideoDownload: vi.fn(),
    onCancelVideoDownload: vi.fn(),
    onDeleteVideoDownload: vi.fn(),
    onOpenExternalUrl: vi.fn(),
    onRevealPath: vi.fn(),
    onRefreshLongLiveStatus: vi.fn(),
    onInstallLongLive: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as VideoModelsPageProps["onInstallLongLive"],
    ...overrides,
  };
  return renderToStaticMarkup(<VideoModelsPage {...props} />);
}

/** The markup of one row, found by its data-model-id. */
function rowMarkup(markup: string, id: string): string {
  const start = markup.indexOf(`data-model-id="${id}"`);
  expect(start, `row ${id} not rendered`).toBeGreaterThan(-1);
  const next = markup.indexOf("data-model-id=", start + 1);
  return markup.slice(start, next === -1 ? undefined : next);
}

function rowOrder(markup: string): string[] {
  return Array.from(markup.matchAll(/data-model-id="([^"]+)"/g)).map((match) => match[1]);
}

const Q4 = "city96/Wan2.1-T2V-14B-gguf-q4km";
const Q6 = "city96/Wan2.1-T2V-14B-gguf-q6k";
const Q8 = "city96/Wan2.1-T2V-14B-gguf-q8";
const BASE = "Wan-AI/Wan2.1-T2V-14B-Diffusers";

const ltxInstalled = makeVideoVariant({
  id: "dgrauet/ltx-2.3-mlx-q4",
  familyId: "ltx-2",
  name: "LTX-2.3 · distilled q4 (ltx-2-mlx)",
  provider: "Lightricks · dgrauet",
  repo: "dgrauet/ltx-2.3-mlx-q4",
  runtime: "ltx-2-mlx (MLX native)",
  styleTags: ["general", "fast", "mlx"],
  availableLocally: true,
  hasLocalData: true,
  localPath: "/hf/models--dgrauet--ltx-2.3-mlx-q4/snapshots/1",
  releaseDate: "2026-03",
});
const ltx25 = makeVideoVariant({
  id: "dgrauet/ltx-2.5-mlx-q4",
  familyId: "ltx-2",
  name: "LTX-2.5 · distilled q4 (ltx-2-mlx)",
  provider: "Lightricks · dgrauet",
  repo: "dgrauet/ltx-2.5-mlx-q4",
  runtime: "ltx-2-mlx (MLX native)",
  styleTags: ["general", "fast", "mlx"],
  note: "LTX-2.5 distilled. Gated: accept the licence on Hugging Face first.",
  releaseDate: "2026-08",
});

function catalog(): VideoModelFamily[] {
  return [
    makeVideoFamily({ id: "wan-2-1", name: "Wan 2.1" }, wan14bRows()),
    makeVideoFamily({ id: "ltx-2", name: "LTX-2 (MLX)" }, [ltxInstalled, ltx25]),
  ];
}

const downloads: Record<string, DownloadStatus> = {
  // Keyed by variant id (R12); the repo is the one being fetched.
  [Q6]: downloadState("city96/Wan2.1-T2V-14B-gguf", "cancelled", { progress: 0.36, downloadedGb: 4.5, totalGb: 12.5 }),
  [Q8]: downloadState("city96/Wan2.1-T2V-14B-gguf", "downloading"),
};

describe("VideoModelsPage", () => {
  it("lists installed and not-installed models on one page with bucket counts", () => {
    const markup = render({ catalog: catalog(), activeVideoDownloads: downloads });
    for (const id of [BASE, Q4, Q6, Q8, ltxInstalled.id, ltx25.id]) rowMarkup(markup, id);
    expect(markup).toContain("6 of 6 models · 2 installed");
    expect(markup).toMatch(/data-bucket="installed"[^>]*>.*?<span class="models-availability__count">2</);
    expect(markup).toMatch(/data-bucket="available"[^>]*>.*?<span class="models-availability__count">2</);
    expect(markup).toMatch(/data-bucket="downloading"[^>]*>.*?<span class="models-availability__count">2</);
    expect(markup).toContain("Video Models");
    expect(markup).toContain(">Open Studio</button>");
  });

  it("reads the download of a row by its variant id, not its repo (R12)", () => {
    // A status stored under the shared repo alone does not mark every
    // variant of that repo as downloading.
    const markup = render({
      catalog: catalog(),
      activeVideoDownloads: { [BASE]: downloadState(BASE, "downloading") },
    });
    expect(rowMarkup(markup, BASE)).toContain('data-status="downloading"');
    expect(rowMarkup(markup, Q4)).toContain('data-status="available"');
  });

  it("offers Resume and Delete download on a paused row, labelled for the shared GGUF (R10)", () => {
    const row = rowMarkup(render({ catalog: catalog(), activeVideoDownloads: downloads }), Q6);
    expect(row).toContain('aria-label="Resume download"');
    expect(row).toContain('aria-label="Delete shared GGUF download"');
    expect(row).not.toContain('aria-label="Download model"');
    expect(row).toContain('data-status="paused"');
  });

  it("offers Pause and Cancel on a running download", () => {
    const row = rowMarkup(render({ catalog: catalog(), activeVideoDownloads: downloads }), Q8);
    expect(row).toContain('aria-label="Pause download"');
    expect(row).toContain('aria-label="Cancel download"');
  });

  it("offers Generate, Delete model, Reveal and the model card on an installed row", () => {
    const row = rowMarkup(render({ catalog: catalog() }), ltxInstalled.id);
    expect(row).toContain('aria-label="Generate"');
    expect(row).toContain('aria-label="Delete model"');
    expect(row).toContain('aria-label="Show in Finder"');
    expect(row).toContain('aria-label="Open model card"');
    expect(row).toContain('data-status="installed"');
  });

  it("offers Download on the not-installed LTX-2.5 row and keeps the gated note in its detail", () => {
    const row = rowMarkup(render({ catalog: catalog() }), ltx25.id);
    expect(row).toContain('aria-label="Download model"');
    expect(row).not.toContain('aria-label="Generate"');
    expect(row).toContain('aria-expanded="false"');
    expect(row).toContain("<small>Lightricks · dgrauet · LTX-2 (MLX)</small>");
    expect(row).not.toContain("Gated: accept the licence");
  });

  it("shows the shared-storage badge on rows with files", () => {
    const markup = render({ catalog: catalog(), activeVideoDownloads: { [Q4]: downloadState("city96/Wan2.1-T2V-14B-gguf", "cancelled") } });
    expect(rowMarkup(markup, Q4)).toContain("shares storage · +2");
    // Same siblings, but nothing on disk: no badge.
    expect(rowMarkup(render({ catalog: catalog() }), Q4)).not.toContain("shares storage");
  });

  it("marks the row the runtime has loaded", () => {
    const row = rowMarkup(render({ catalog: catalog(), loadedVideoVariantId: ltxInstalled.id }), ltxInstalled.id);
    expect(row).toContain("Loaded in memory");
    expect(rowMarkup(render({ catalog: catalog(), loadedVideoVariantId: ltxInstalled.id }), BASE)).not.toContain("Loaded in memory");
  });

  it("shows an incomplete row with its reason and Resume", () => {
    const partial = makeVideoVariant({
      id: "partial",
      name: "Partial Model",
      repo: "org/partial",
      hasLocalData: true,
      localDataRepos: ["org/partial"],
      primaryLocalRepo: "org/partial",
      localStatusReason: "Text encoder weights are missing.",
    });
    const row = rowMarkup(render({ catalog: [makeVideoFamily({}, [partial])] }), "partial");
    expect(row).toContain('data-status="partial"');
    expect(row).toContain('aria-label="Resume download"');
    expect(row).toContain('aria-label="Delete model"');
    expect(row).toContain("Text encoder weights are missing.");
  });

  it("explains a gated download failure and links to Hugging Face and Settings", () => {
    const markup = render({
      catalog: catalog(),
      activeVideoDownloads: {
        [ltx25.id]: downloadState(ltx25.repo, "failed", {
          error: "403 Client Error. Cannot access gated repo for url https://huggingface.co/dgrauet/ltx-2.5-mlx-q4",
        }),
      },
    });
    const row = rowMarkup(markup, ltx25.id);
    expect(row).toContain("LTX-2.5 · distilled q4 (ltx-2-mlx) is gated on Hugging Face");
    expect(row).toContain(">Hugging Face</button>");
    expect(row).toContain(">Settings</button>");
    expect(row).toContain('aria-label="Retry download"');
    expect(row).toContain('aria-label="Delete download"');
  });

  it("sorts by size using the GGUF file of each quant (R13)", () => {
    const markup = render({ catalog: [makeVideoFamily({}, wan14bRows())], sort: "size" });
    expect(rowOrder(markup)).toEqual([BASE, Q8, Q6, Q4]);
  });

  it("filters by the availability preset", () => {
    const show = (availability: ModelAvailabilityFilter) =>
      render({ catalog: catalog(), activeVideoDownloads: downloads, availability });
    const installedOnly = show("installed");
    expect(installedOnly).toContain(`data-model-id="${BASE}"`);
    expect(installedOnly).not.toContain(`data-model-id="${ltx25.id}"`);
    const availableOnly = show("available");
    expect(availableOnly).toContain(`data-model-id="${ltx25.id}"`);
    expect(availableOnly).not.toContain(`data-model-id="${BASE}"`);
    expect(show("downloading")).toContain(`data-model-id="${Q6}"`);
    expect(availableOnly).toContain("Showing: Available");
  });

  it("says nothing is installed yet, with a way to the available models", () => {
    const markup = render({ catalog: [makeVideoFamily({}, [ltx25])], availability: "installed" });
    expect(markup).toContain("No video models installed yet");
    expect(markup).toContain("Show available models");
  });

  it("hides a model this host cannot run unless it is installed, then marks it", () => {
    const mlxAvailable = makeVideoVariant({ ...ltx25 });
    const mlxInstalled = makeVideoVariant({ ...ltxInstalled });
    const markup = render({ catalog: [makeVideoFamily({ id: "ltx-2" }, [mlxAvailable, mlxInstalled])], hostSystem: CUDA_PC });
    expect(markup).not.toContain(`data-model-id="${ltx25.id}"`);
    const row = rowMarkup(markup, ltxInstalled.id);
    expect(row).toContain("Cannot run on this machine");
    expect(row).toMatch(/disabled=""[^>]*aria-label="Generate"|aria-label="Generate"[^>]*disabled=""/);
    expect(row).toContain('aria-label="Delete model"');
  });

  it("shows CUDA accelerator pills on a CUDA PC only", () => {
    const wan = makeVideoVariant();
    const onMac = rowMarkup(render({ catalog: [makeVideoFamily({}, [wan])], hostSystem: MAC }), wan.id);
    const onPc = rowMarkup(render({ catalog: [makeVideoFamily({}, [wan])], hostSystem: CUDA_PC }), wan.id);
    expect(onMac).not.toContain("accelerator-card");
    expect(onPc).toContain("accelerator-card");
  });

  it("shows the fit verdict and the family's best fit", () => {
    const fitted = makeVideoVariant({
      id: "fits",
      name: "Fits Model",
      repo: "org/fits",
      fit: { status: "fits", needGb: 14, usableGb: 51.8, headroomPct: 73, reason: "14 GB peak", basis: "estimated" },
    } as Partial<VideoModelVariant>);
    const family = { ...makeVideoFamily({}, [fitted]), bestFitVariantId: "fits" } as VideoModelFamily;
    const row = rowMarkup(render({ catalog: [family] }), "fits");
    expect(row).toContain("best-fit-ribbon");
    expect(row).toContain("fit-badge--fits");
  });

  describe("LongLive", () => {
    const longLive = makeLongLiveVariant();
    const family = makeVideoFamily({ id: "longlive", name: "LongLive 1.3B" }, [longLive]);

    it("is hidden on a Mac (CUDA only)", () => {
      expect(render({ catalog: [family], hostSystem: MAC })).not.toContain(`data-model-id="${longLive.id}"`);
    });

    it("offers Install, the install note and no download or delete when not installed", () => {
      const row = rowMarkup(render({ catalog: [family], hostSystem: CUDA_PC }), longLive.id);
      expect(row).toContain('aria-label="Install"');
      expect(row).toContain("LongLive installs into an isolated venv");
      expect(row).not.toContain('aria-label="Download model"');
      expect(row).not.toContain('aria-label="Delete model"');
      expect(row).toContain('data-status="available"');
    });

    it("shows Installing while the installer runs", () => {
      const row = rowMarkup(render({ catalog: [family], hostSystem: CUDA_PC, installingLongLive: true }), longLive.id);
      expect(row).toContain('data-status="downloading"');
      expect(row).toContain('aria-label="Installing"');
      expect(row).toMatch(/disabled=""[^>]*aria-label="Installing"|aria-label="Installing"[^>]*disabled=""/);
    });

    it("is installed when the runtime probe says it can generate", () => {
      const row = rowMarkup(
        render({
          catalog: [family],
          hostSystem: CUDA_PC,
          longLiveStatus: { activeEngine: "longlive", realGenerationAvailable: true, message: "ready" },
        }),
        longLive.id,
      );
      expect(row).toContain('data-status="installed"');
      expect(row).toContain('aria-label="Generate"');
      expect(row).not.toContain("LongLive installs into an isolated venv");
      expect(row).not.toContain('aria-label="Delete model"');
    });

    it("ignores download statuses for LongLive", () => {
      const row = rowMarkup(
        render({
          catalog: [family],
          hostSystem: CUDA_PC,
          activeVideoDownloads: { [longLive.id]: downloadState(longLive.repo, "failed", { error: "404" }) },
        }),
        longLive.id,
      );
      expect(row).toContain('data-status="available"');
      expect(row).toContain('aria-label="Install"');
    });
  });
});
