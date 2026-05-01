import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { OnlineModelsTab } from "./OnlineModelsTab";
import type { DownloadStatus } from "../../api";
import type { HubModel, ModelFamily, ModelVariant } from "../../types";

function makeVariant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return {
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
    availableLocally: false,
    launchMode: "convert",
    backend: "mlx",
    ...overrides,
  };
}

function makeFamily(variant: ModelVariant): ModelFamily {
  return {
    id: "qwen3-coder",
    name: "Qwen3 Coder",
    provider: "Qwen",
    headline: "Code-specialised Qwen3 with strong agentic support.",
    summary: "Purpose-built coding model.",
    description: "A coding-focused official checkpoint.",
    updatedLabel: "Updated Feb 2026",
    popularityLabel: "639k downloads",
    likesLabel: "1.3k likes",
    badges: ["Coding"],
    capabilities: ["coding", "agents", "tool-use"],
    defaultVariantId: variant.id,
    variants: [variant],
    readme: ["Official Next repos replace older provisional placeholders."],
  };
}

function renderTab(downloadState: DownloadStatus): string {
  const variant = makeVariant();
  const family = makeFamily(variant);
  return renderToStaticMarkup(
    <OnlineModelsTab
      searchResults={[family]}
      searchInput="qwen coder"
      onSearchInputChange={() => {}}
      searchError={null}
      localVariantCount={0}
      discoverCapFilter={null}
      onDiscoverCapFilterChange={() => {}}
      discoverFormatFilter={null}
      onDiscoverFormatFilterChange={() => {}}
      expandedFamilyId={family.id}
      onExpandedFamilyIdChange={() => {}}
      expandedVariantId={null}
      onExpandedVariantIdChange={() => {}}
      onDetailFamilyIdChange={() => {}}
      library={[]}
      activeDownloads={{ [variant.repo]: downloadState }}
      onDownloadModel={vi.fn()}
      onCancelModelDownload={vi.fn()}
      onDeleteModelDownload={vi.fn()}
      onPrepareCatalogConversion={vi.fn()}
      onOpenModelSelector={vi.fn()}
      onOpenExternalUrl={vi.fn()}
      hubResults={[]}
      expandedHubId={null}
      onToggleHubExpand={() => {}}
      hubFileCache={{}}
      hubFileLoading={{}}
      hubFileError={{}}
    />,
  );
}

function renderHubOnlyTab(hubResults: HubModel[]): string {
  return renderToStaticMarkup(
    <OnlineModelsTab
      searchResults={[]}
      searchInput="glm"
      onSearchInputChange={() => {}}
      searchError={null}
      localVariantCount={0}
      discoverCapFilter={null}
      onDiscoverCapFilterChange={() => {}}
      discoverFormatFilter={null}
      onDiscoverFormatFilterChange={() => {}}
      expandedFamilyId={null}
      onExpandedFamilyIdChange={() => {}}
      expandedVariantId={null}
      onExpandedVariantIdChange={() => {}}
      onDetailFamilyIdChange={() => {}}
      library={[]}
      activeDownloads={{}}
      onDownloadModel={vi.fn()}
      onCancelModelDownload={vi.fn()}
      onDeleteModelDownload={vi.fn()}
      onPrepareCatalogConversion={vi.fn()}
      onOpenModelSelector={vi.fn()}
      onOpenExternalUrl={vi.fn()}
      hubResults={hubResults}
      expandedHubId={null}
      onToggleHubExpand={() => {}}
      hubFileCache={{}}
      hubFileLoading={{}}
      hubFileError={{}}
    />,
  );
}

function renderExpandedHubTab(hubResults: HubModel[], fileCount = 12): string {
  const repo = hubResults[0]?.repo ?? "org/model";
  return renderToStaticMarkup(
    <OnlineModelsTab
      searchResults={[]}
      searchInput="glm"
      onSearchInputChange={() => {}}
      searchError={null}
      localVariantCount={0}
      discoverCapFilter={null}
      onDiscoverCapFilterChange={() => {}}
      discoverFormatFilter={null}
      onDiscoverFormatFilterChange={() => {}}
      expandedFamilyId={null}
      onExpandedFamilyIdChange={() => {}}
      expandedVariantId={null}
      onExpandedVariantIdChange={() => {}}
      onDetailFamilyIdChange={() => {}}
      library={[]}
      activeDownloads={{}}
      onDownloadModel={vi.fn()}
      onCancelModelDownload={vi.fn()}
      onDeleteModelDownload={vi.fn()}
      onPrepareCatalogConversion={vi.fn()}
      onOpenModelSelector={vi.fn()}
      onOpenExternalUrl={vi.fn()}
      hubResults={hubResults}
      expandedHubId={repo}
      onToggleHubExpand={() => {}}
      hubFileCache={{
        [repo]: {
          repo,
          files: Array.from({ length: fileCount }, (_, index) => ({
            path: `model-${String(index).padStart(5, "0")}-of-${String(fileCount).padStart(5, "0")}.safetensors`,
            sizeBytes: 9_500_000_000,
            sizeGb: 9.5,
            kind: "weight" as const,
          })),
          totalSizeBytes: fileCount * 9_500_000_000,
          totalSizeGb: fileCount * 9.5,
          license: "apache-2.0",
          tags: ["text-generation", "safetensors"],
          pipelineTag: "text-generation",
          lastModified: "2026-04-16T10:00:00Z",
        },
      }}
      hubFileLoading={{}}
      hubFileError={{}}
    />,
  );
}

describe("OnlineModelsTab", () => {
  it("shows a preparing label instead of a misleading 0% download state", () => {
    const markup = renderTab({
      repo: "Qwen/Qwen3-Coder-Next-FP8",
      state: "downloading",
      progress: 0,
      downloadedGb: 0,
      totalGb: 74.9,
      error: null,
    });

    expect(markup).toContain("Preparing download...");
    expect(markup).not.toContain("Downloading 0%");
  });

  it("renders inline failed-download details with retry actions", () => {
    const markup = renderTab({
      repo: "Qwen/Qwen3-Coder-Next-FP8",
      state: "failed",
      progress: 0,
      downloadedGb: 0,
      totalGb: 74.9,
      error: "Qwen/Qwen3-Coder-Next-FP8 was not found on Hugging Face.",
    });

    expect(markup).toContain("Qwen/Qwen3-Coder-Next-FP8 was not found on Hugging Face.");
    expect(markup).toContain("aria-label=\"Retry download\"");
    expect(markup).toContain("aria-label=\"Delete download\"");
  });

  it("renders live hub results even when no curated families match", () => {
    const markup = renderHubOnlyTab([
      {
        id: "zai-org/GLM-4.7-Flash",
        repo: "zai-org/GLM-4.7-Flash",
        name: "GLM-4.7-Flash",
        provider: "zai-org",
        link: "https://huggingface.co/zai-org/GLM-4.7-Flash",
        format: "Transformers",
        tags: ["text-generation"],
        downloads: 729180,
        likes: 1665,
        downloadsLabel: "729,180 downloads",
        likesLabel: "1,665 likes",
        lastModified: "2026-04-16T10:00:00Z",
        updatedLabel: "Updated Apr 16",
        availableLocally: false,
        launchMode: "convert",
        backend: "mlx",
      },
    ]);

    expect(markup).toContain("HuggingFace Hub");
    expect(markup).toContain("GLM-4.7-Flash");
    expect(markup).not.toContain('No models match "glm"');
  });

  it("orders live hub results by most recent update first", () => {
    const markup = renderHubOnlyTab([
      {
        id: "org/older-model",
        repo: "org/older-model",
        name: "older-model",
        provider: "org",
        link: "https://huggingface.co/org/older-model",
        format: "Transformers",
        tags: ["text-generation"],
        downloads: 1000,
        likes: 50,
        downloadsLabel: "1,000 downloads",
        likesLabel: "50 likes",
        lastModified: "2026-04-10T10:00:00Z",
        updatedLabel: "Updated Apr 10",
        availableLocally: false,
        launchMode: "convert",
        backend: "mlx",
      },
      {
        id: "org/newer-model",
        repo: "org/newer-model",
        name: "newer-model",
        provider: "org",
        link: "https://huggingface.co/org/newer-model",
        format: "Transformers",
        tags: ["text-generation"],
        downloads: 10,
        likes: 1,
        downloadsLabel: "10 downloads",
        likesLabel: "1 like",
        lastModified: "2026-04-16T10:00:00Z",
        updatedLabel: "Updated Apr 16",
        availableLocally: false,
        launchMode: "convert",
        backend: "mlx",
      },
    ]);

    expect(markup.indexOf("newer-model")).toBeLessThan(markup.indexOf("older-model"));
    expect(markup).toContain("sorted by most recent update");
  });

  it("surfaces in-flight download progress in the collapsed curated card header", () => {
    const variant = makeVariant();
    const family = makeFamily(variant);
    const markup = renderToStaticMarkup(
      <OnlineModelsTab
        searchResults={[family]}
        searchInput=""
        onSearchInputChange={() => {}}
        searchError={null}
        localVariantCount={0}
        discoverCapFilter={null}
        onDiscoverCapFilterChange={() => {}}
        discoverFormatFilter={null}
        onDiscoverFormatFilterChange={() => {}}
        expandedFamilyId={null}
        onExpandedFamilyIdChange={() => {}}
        expandedVariantId={null}
        onExpandedVariantIdChange={() => {}}
        onDetailFamilyIdChange={() => {}}
        library={[]}
        activeDownloads={{
          [variant.repo]: {
            repo: variant.repo,
            state: "downloading",
            progress: 0.42,
            downloadedGb: 31,
            totalGb: 74.9,
            error: null,
          },
        }}
        onDownloadModel={vi.fn()}
        onCancelModelDownload={vi.fn()}
        onDeleteModelDownload={vi.fn()}
        onPrepareCatalogConversion={vi.fn()}
        onOpenModelSelector={vi.fn()}
        onOpenExternalUrl={vi.fn()}
        hubResults={[]}
        expandedHubId={null}
        onToggleHubExpand={() => {}}
        hubFileCache={{}}
        hubFileLoading={{}}
        hubFileError={{}}
      />,
    );

    // Header progress should be visible even though the card is collapsed.
    expect(markup).toContain("Downloading 42%");
    expect(markup).not.toContain('class="discover-card expanded"');
  });

  it("surfaces in-flight download progress in the collapsed hub card header", () => {
    const markup = renderToStaticMarkup(
      <OnlineModelsTab
        searchResults={[]}
        searchInput="glm"
        onSearchInputChange={() => {}}
        searchError={null}
        localVariantCount={0}
        discoverCapFilter={null}
        onDiscoverCapFilterChange={() => {}}
        discoverFormatFilter={null}
        onDiscoverFormatFilterChange={() => {}}
        expandedFamilyId={null}
        onExpandedFamilyIdChange={() => {}}
        expandedVariantId={null}
        onExpandedVariantIdChange={() => {}}
        onDetailFamilyIdChange={() => {}}
        library={[]}
        activeDownloads={{
          "zai-org/GLM-4.7-Flash": {
            repo: "zai-org/GLM-4.7-Flash",
            state: "downloading",
            progress: 0.17,
            downloadedGb: 3.2,
            totalGb: 19,
            error: null,
          },
        }}
        onDownloadModel={vi.fn()}
        onCancelModelDownload={vi.fn()}
        onDeleteModelDownload={vi.fn()}
        onPrepareCatalogConversion={vi.fn()}
        onOpenModelSelector={vi.fn()}
        onOpenExternalUrl={vi.fn()}
        hubResults={[
          {
            id: "zai-org/GLM-4.7-Flash",
            repo: "zai-org/GLM-4.7-Flash",
            name: "GLM-4.7-Flash",
            provider: "zai-org",
            link: "https://huggingface.co/zai-org/GLM-4.7-Flash",
            format: "Transformers",
            tags: ["text-generation"],
            downloads: 729180,
            likes: 1665,
            downloadsLabel: "729,180 downloads",
            likesLabel: "1,665 likes",
            lastModified: "2026-04-16T10:00:00Z",
            updatedLabel: "Updated Apr 16",
            availableLocally: false,
            launchMode: "convert",
            backend: "mlx",
          },
        ]}
        expandedHubId={null}
        onToggleHubExpand={() => {}}
        hubFileCache={{}}
        hubFileLoading={{}}
        hubFileError={{}}
      />,
    );

    expect(markup).toContain("Downloading 17%");
  });

  it("collapses large weight shard lists by default in expanded hub cards", () => {
    const markup = renderExpandedHubTab([
      {
        id: "org/big-model",
        repo: "org/big-model",
        name: "big-model",
        provider: "org",
        link: "https://huggingface.co/org/big-model",
        format: "Transformers",
        tags: ["text-generation", "safetensors"],
        downloads: 100,
        likes: 5,
        downloadsLabel: "100 downloads",
        likesLabel: "5 likes",
        lastModified: "2026-04-16T10:00:00Z",
        updatedLabel: "Updated Apr 16",
        availableLocally: false,
        launchMode: "convert",
        backend: "mlx",
      },
    ]);

    expect(markup).toContain('<details class="hub-file-group hub-file-group--collapsible">');
    expect(markup).toContain("Weights (12)");
    expect(markup).toContain("Largest shard 9.5 GB");
  });
});
