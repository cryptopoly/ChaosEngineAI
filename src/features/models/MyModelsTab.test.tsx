import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { DownloadStatus } from "../../api";
import { MyModelsTab, type LibraryRow } from "./MyModelsTab";
import type { LibraryItem, ModelVariant } from "../../types";

function makeVariant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return {
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
    availableLocally: false,
    launchMode: "convert",
    backend: "mlx",
    ...overrides,
  };
}

function makeItem(overrides: Partial<LibraryItem> = {}): LibraryItem {
  return {
    name: "Qwen/Qwen3.6-35B-A3B",
    path: "/Users/dan/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B",
    format: "HF cache",
    sourceKind: "HF cache",
    quantization: "BF16",
    backend: "mlx",
    modelType: "text",
    sizeGb: 4.7,
    lastModified: "2026-04-16 17:00",
    actions: ["Run Chat", "Run Server", "Cache Preview", "Delete"],
    broken: true,
    brokenReason: "No .gguf, .safetensors, or pytorch weights found in HF cache entry",
    ...overrides,
  };
}

function makeLocalGgufItem(overrides: Partial<LibraryItem> = {}): LibraryItem {
  return {
    name: "Qwen3.6-35B-A3B-GGUF",
    path: "/Users/dan/AI_Models/unsloth/Qwen3.6-35B-A3B-GGUF",
    format: "GGUF",
    sourceKind: "Directory",
    quantization: "Q4_K_S",
    backend: "llama.cpp",
    modelType: "text",
    sizeGb: 21.7,
    lastModified: "2026-04-16 17:30",
    actions: ["Run Chat", "Run Server", "Cache Preview", "Delete"],
    broken: true,
    brokenReason: "GGUF download is incomplete: main model weights are still downloading.",
    ...overrides,
  };
}

function renderTab(row: LibraryRow, activeDownloads: Record<string, DownloadStatus>): string {
  return renderToStaticMarkup(
    <MyModelsTab
      filteredLibraryRows={[row]}
      libraryTotalSizeGb={row.item.sizeGb}
      enabledDirectoryCount={1}
      librarySearchInput=""
      onLibrarySearchInputChange={() => {}}
      libraryCapFilter={null}
      onLibraryCapFilterChange={() => {}}
      libraryFormatFilter={null}
      onLibraryFormatFilterChange={() => {}}
      libraryBackendFilter={null}
      onLibraryBackendFilterChange={() => {}}
      strategyCompat={undefined}
      activeDownloads={activeDownloads}
      expandedLibraryPath={null}
      onExpandedLibraryPathChange={() => {}}
      fileRevealLabel="Reveal"
      onDownloadModel={vi.fn()}
      onCancelModelDownload={vi.fn()}
      onDeleteModelDownload={vi.fn()}
      onPrepareLibraryConversion={vi.fn()}
      onOpenModelSelector={vi.fn()}
      onRevealPath={vi.fn()}
      onDeleteModel={vi.fn()}
      librarySortKey="modified"
      librarySortDir="desc"
      onLibrarySortKeyChange={() => {}}
      onLibrarySortDirChange={() => {}}
    />,
  );
}

describe("MyModelsTab", () => {
  it("shows resumable progress for partial HF cache entries even before an active download resumes", () => {
    const row: LibraryRow = {
      item: makeItem({
        sizeGb: 23.9,
        brokenReason: "Hugging Face download is incomplete: partial blob files are still present.",
      }),
      matchedVariant: makeVariant({ sizeGb: 67 }),
      displayFormat: "Transformers",
      displayQuantization: "BF16",
      displayBackend: "mlx",
      sourceKind: "HF cache",
      estimatedRamGb: null,
      estimatedCompressedGb: null,
    };

    const markup = renderTab(row, {});

    expect(markup).toContain("Paused 36%");
    expect(markup).toContain("23.9 / 67.0 GB downloaded.");
    expect(markup).toContain("RESUME");
    expect(markup).toContain("DELETE");
    expect(markup).not.toContain("BROKEN");
  });

  it("shows active download progress instead of a broken badge for partial HF cache entries", () => {
    const row: LibraryRow = {
      item: makeItem(),
      matchedVariant: makeVariant(),
      displayFormat: "Transformers",
      displayQuantization: "BF16",
      displayBackend: "mlx",
      sourceKind: "HF cache",
      estimatedRamGb: null,
      estimatedCompressedGb: null,
    };

    const markup = renderTab(row, {
      "Qwen/Qwen3.6-35B-A3B": {
        repo: "Qwen/Qwen3.6-35B-A3B",
        state: "downloading",
        progress: 0.125,
        downloadedGb: 8.4,
        totalGb: 66.99,
        error: null,
      },
    });

    expect(markup).toContain("Downloading 13%");
    expect(markup).toContain("8.4 / 67.0 GB downloaded.");
    expect(markup).toContain("PAUSE");
    expect(markup).toContain("CANCEL");
    expect(markup).not.toContain("BROKEN");
    expect(markup).not.toContain("No .gguf, .safetensors, or pytorch weights found");
  });

  it("still shows the broken badge when there is no active download for the HF cache entry", () => {
    const row: LibraryRow = {
      item: makeItem(),
      matchedVariant: makeVariant(),
      displayFormat: "Transformers",
      displayQuantization: "BF16",
      displayBackend: "mlx",
      sourceKind: "HF cache",
      estimatedRamGb: null,
      estimatedCompressedGb: null,
    };

    const markup = renderTab(row, {});

    expect(markup).toContain("BROKEN");
    expect(markup).toContain("No .gguf, .safetensors, or pytorch weights found in HF cache entry");
    expect(markup).toContain("RETRY");
    expect(markup).toContain("DELETE");
    expect(markup).not.toContain("CHAT");
    expect(markup).not.toContain("SERVER");
  });

  it("maps local GGUF directories to matched download repos for retry state", () => {
    const row: LibraryRow = {
      item: makeLocalGgufItem(),
      matchedVariant: makeVariant({
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        id: "unsloth/Qwen3.6-35B-A3B-GGUF",
        name: "Qwen3.6-35B-A3B-GGUF",
        format: "GGUF",
        backend: "llama.cpp",
        quantization: "Q4_K_S",
      }),
      displayFormat: "GGUF",
      displayQuantization: "Q4_K_S",
      displayBackend: "llama.cpp",
      sourceKind: "Directory",
      estimatedRamGb: null,
      estimatedCompressedGb: null,
    };

    const markup = renderTab(row, {
      "unsloth/Qwen3.6-35B-A3B-GGUF": {
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        state: "downloading",
        progress: 0.01,
        downloadedGb: 2.55,
        totalGb: 494.14,
        error: null,
      },
    });

    expect(markup).toContain("Downloading 1%");
    expect(markup).toContain("2.5 / 494.1 GB downloaded.");
    expect(markup).toContain("PAUSE");
    expect(markup).toContain("CANCEL");
    expect(markup).not.toContain("CHAT");
    expect(markup).not.toContain("SERVER");
  });
});
