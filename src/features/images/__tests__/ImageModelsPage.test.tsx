import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { DownloadStatus } from "../../../api";
import type { ImageModelFamily, ImageModelVariant } from "../../../types";
import type { ModelAvailabilityFilter } from "../../../components/models";
import { makeImageFamily, makeImageVariant } from "../../../components/models/__tests__/fixtures";
import { ImageModelsPage, type ImageModelsPageProps } from "../ImageModelsPage";

const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };

function render(overrides: Partial<ImageModelsPageProps> = {}): string {
  const props: ImageModelsPageProps = {
    catalog: [],
    latest: [],
    searchInput: "",
    onSearchInputChange: vi.fn(),
    taskFilter: "all",
    onTaskFilterChange: vi.fn(),
    accessFilter: "all",
    onAccessFilterChange: vi.fn(),
    sort: "release",
    onSortChange: vi.fn(),
    availability: "all",
    onAvailabilityChange: vi.fn(),
    activeImageDownloads: {},
    selectedImageVariant: null,
    fileRevealLabel: "Show in Finder",
    hostSystem: MAC,
    onActiveTabChange: vi.fn(),
    onOpenImageStudio: vi.fn(),
    onImageDownload: vi.fn(),
    onCancelImageDownload: vi.fn(),
    onDeleteImageDownload: vi.fn(),
    onOpenExternalUrl: vi.fn(),
    onRevealPath: vi.fn(),
    ...overrides,
  };
  return renderToStaticMarkup(<ImageModelsPage {...props} />);
}

/** The markup of one row, found by its data-model-id. */
function rowMarkup(markup: string, id: string): string {
  const start = markup.indexOf(`data-model-id="${id}"`);
  expect(start, `row ${id} not rendered`).toBeGreaterThan(-1);
  const next = markup.indexOf("data-model-id=", start + 1);
  return markup.slice(start, next === -1 ? undefined : next);
}

function paused(repo: string): DownloadStatus {
  return { repo, state: "cancelled", progress: 0.36, downloadedGb: 8.6, totalGb: 23.8, error: null };
}

const installed = makeImageVariant({
  id: "flux-dev",
  name: "FLUX.1 Dev",
  repo: "black-forest-labs/FLUX.1-dev",
  availableLocally: true,
  hasLocalData: true,
  localPath: "/models/flux-dev",
  sharedRepoSiblings: 1,
  sharedRepoSiblingIds: ["flux-dev-mflux"],
  releaseDate: "2024-08",
});
const pausedVariant = makeImageVariant({ id: "sd35", name: "SD 3.5 Medium", repo: "stabilityai/sd35", releaseDate: "2024-10" });
const failedGated = makeImageVariant({ id: "gated", name: "Gated Model", repo: "org/gated", gated: true, releaseDate: "2025-01" });
const available = makeImageVariant({ id: "qwen", name: "Qwen-Image", repo: "Qwen/Qwen-Image", releaseDate: "2025-08" });
const tracked = makeImageVariant({
  id: "baidu/ERNIE-Image",
  name: "ERNIE-Image",
  repo: "baidu/ERNIE-Image",
  trackedOnly: true,
  releaseDate: "2026-02",
});

function catalog(): ImageModelFamily[] {
  return [makeImageFamily({ id: "flux", name: "FLUX" }, [installed, pausedVariant, failedGated, available])];
}

const downloads: Record<string, DownloadStatus> = {
  "stabilityai/sd35": paused("stabilityai/sd35"),
  "org/gated": {
    repo: "org/gated",
    state: "failed",
    progress: 0,
    downloadedGb: 0,
    totalGb: null,
    error: "403 Client Error. Cannot access gated repo for url https://huggingface.co/org/gated",
  },
};

describe("ImageModelsPage", () => {
  it("lists installed and not-installed models on one page with bucket counts", () => {
    const markup = render({ catalog: catalog(), latest: [tracked], activeImageDownloads: downloads });
    for (const id of ["flux-dev", "sd35", "gated", "qwen", "baidu/ERNIE-Image"]) rowMarkup(markup, id);
    expect(markup).toContain("5 of 5 models · 1 installed");
    expect(markup).toMatch(/data-bucket="installed"[^>]*>.*?<span>Installed<\/span><span class="models-availability__count">1</);
    expect(markup).toMatch(/data-bucket="available"[^>]*>.*?<span class="models-availability__count">2</);
    expect(markup).toMatch(/data-bucket="downloading"[^>]*>.*?<span class="models-availability__count">1</);
    expect(markup).toMatch(/data-bucket="attention"[^>]*>.*?<span class="models-availability__count">1</);
  });

  it("offers Resume and Delete download on a paused row (not Download model)", () => {
    const row = rowMarkup(render({ catalog: catalog(), activeImageDownloads: downloads }), "sd35");
    expect(row).toContain('aria-label="Resume download"');
    expect(row).toContain('aria-label="Delete download"');
    expect(row).not.toContain('aria-label="Download model"');
    expect(row).toContain("Paused: Paused 36% / 8.6 / 23.8 GB");
  });

  it("offers Generate, Delete model and Reveal on an installed row, with the shared-storage badge", () => {
    const row = rowMarkup(render({ catalog: catalog(), activeImageDownloads: downloads }), "flux-dev");
    expect(row).toContain('aria-label="Generate"');
    expect(row).toContain('aria-label="Delete model"');
    expect(row).toContain('aria-label="Show in Finder"');
    expect(row).toContain("shares storage · +1");
    expect(row).toContain('data-status="installed"');
  });

  it("explains a gated download failure and links to Hugging Face and Settings", () => {
    const row = rowMarkup(render({ catalog: catalog(), activeImageDownloads: downloads }), "gated");
    expect(row).toContain("Gated Model is gated on Hugging Face");
    expect(row).toContain(">Hugging Face</button>");
    expect(row).toContain(">Settings</button>");
    expect(row).toContain('aria-label="Retry download"');
  });

  it("disables the download of a tracked-only row and says why (FU-061)", () => {
    const row = rowMarkup(render({ latest: [tracked] }), "baidu/ERNIE-Image");
    expect(row).toContain("Watching upstream");
    expect(row).toMatch(/aria-label="Tracked only — not yet launchable"[^>]*>|disabled=""[^>]*aria-label="Tracked only/);
    expect(row).toContain('disabled=""');
    expect(row).not.toContain('aria-label="Download model"');
  });

  it("filters by the availability preset", () => {
    const show = (availability: ModelAvailabilityFilter) =>
      render({ catalog: catalog(), latest: [tracked], activeImageDownloads: downloads, availability });
    const installedOnly = show("installed");
    expect(installedOnly).toContain('data-model-id="flux-dev"');
    expect(installedOnly).not.toContain('data-model-id="qwen"');
    const availableOnly = show("available");
    expect(availableOnly).toContain('data-model-id="qwen"');
    expect(availableOnly).toContain('data-model-id="baidu/ERNIE-Image"');
    expect(availableOnly).not.toContain('data-model-id="flux-dev"');
    expect(show("downloading")).toContain('data-model-id="sd35"');
    expect(show("attention")).toContain('data-model-id="gated"');
    expect(availableOnly).toContain("Showing: Available");
  });

  it("says nothing is installed yet, with a way to the available models", () => {
    const markup = render({ catalog: [makeImageFamily({}, [available])], availability: "installed" });
    expect(markup).toContain("No image models installed yet");
    expect(markup).toContain("Show available models");
  });

  it("hides a model this host cannot run unless it is installed, then marks it", () => {
    const macOnly = makeImageVariant({ id: "mflux-a", name: "Mac Only", repo: "org/mflux-a", runtime: "mflux" });
    const macOnlyInstalled = makeImageVariant({
      id: "mflux-b",
      name: "Mac Only Installed",
      repo: "org/mflux-b",
      runtime: "mflux",
      availableLocally: true,
      hasLocalData: true,
    });
    const markup = render({ catalog: [makeImageFamily({}, [macOnly, macOnlyInstalled])], hostSystem: CUDA_PC });
    expect(markup).not.toContain('data-model-id="mflux-a"');
    const row = rowMarkup(markup, "mflux-b");
    expect(row).toContain("Cannot run on this machine");
    expect(row).toMatch(/disabled=""[^>]*aria-label="Generate"|aria-label="Generate"[^>]*disabled=""/);
    expect(row).toContain('aria-label="Delete model"');
  });

  it("shows the fit verdict and the family's best fit", () => {
    const fitted = makeImageVariant({
      id: "fits",
      name: "Fits Model",
      repo: "org/fits",
      fit: { status: "fits", needGb: 14, usableGb: 51.8, headroomPct: 73, reason: "14 GB peak at 1024px", basis: "estimated" },
    } as Partial<ImageModelVariant>);
    const big = makeImageVariant({
      id: "big",
      name: "Big Model",
      repo: "org/big",
      fit: { status: "wont-fit", needGb: 70, usableGb: 51.8, headroomPct: -35, reason: "70 GB peak", basis: "measured" },
    } as Partial<ImageModelVariant>);
    const family = { ...makeImageFamily({}, [fitted, big]), bestFitVariantId: "fits" } as ImageModelFamily;
    const markup = render({ catalog: [family] });
    const fitsRow = rowMarkup(markup, "fits");
    expect(fitsRow).toContain("best-fit-ribbon");
    expect(fitsRow).toContain("fit-badge--fits");
    expect(fitsRow).toContain("est.");
    const bigRow = rowMarkup(markup, "big");
    expect(bigRow).not.toContain("best-fit-ribbon");
    expect(bigRow).toContain("fit-badge--wont-fit");
  });

  it("keeps the provider and family name as the subtitle and the note out of the row", () => {
    const row = rowMarkup(render({ catalog: catalog() }), "qwen");
    expect(row).toContain("<small>Black Forest Labs · FLUX</small>");
    expect(row).not.toContain("Four-step distilled FLUX.");
    expect(row).toContain('aria-expanded="false"');
  });
});
