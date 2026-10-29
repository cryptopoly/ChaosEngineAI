import { describe, expect, it } from "vitest";
import {
  buildChatEntries,
  chatEntryMatchesQuery,
  chatFavoriteRef,
  findLibraryItemForRepo,
  libraryItemIsRepo,
  makeChatAccelPredicate,
  prepareChatJoin,
  type ChatAccelCompat,
  type ChatModelEntry,
} from "../adapters/chat";
import { resolveModelStatus } from "../modelStatus";
import { buildModelRows } from "../modelFilters";
import { MAC, WINDOWS_CUDA, makeChatFamily, makeChatVariant, makeLibraryItem, transfer } from "./chatFixtures";

const NOW = Date.parse("2026-10-29T12:00:00Z");

function byId(entries: ChatModelEntry[], id: string): ChatModelEntry {
  const entry = entries.find((candidate) => candidate.id === id);
  expect(entry, `row ${id}`).toBeDefined();
  return entry as ChatModelEntry;
}

function stateOf(entry: ChatModelEntry) {
  return resolveModelStatus(entry, { download: entry.source.download }).state;
}

const qwenMlx = makeChatVariant();
const qwenGguf = makeChatVariant({
  id: "lmstudio-community/Qwen3-4B-GGUF",
  name: "Qwen3 4B GGUF",
  repo: "lmstudio-community/Qwen3-4B-GGUF",
  link: "https://huggingface.co/lmstudio-community/Qwen3-4B-GGUF",
  format: "GGUF",
  quantization: "Q4_K_M",
  backend: "llama.cpp",
});
const family = makeChatFamily({}, [qwenMlx, qwenGguf]);

describe("chat join rule (R1)", () => {
  it("merges a local model into its catalog row when each is the other's best match and the repo is the same", () => {
    const item = makeLibraryItem();
    const entries = buildChatEntries([family], [item], {}, { now: NOW });
    const row = byId(entries, `catalog:${qwenMlx.id}`);
    expect(row.source.item).toBe(item);
    expect(row.localPath).toBe(item.path);
    expect(row.source.launchKey).toBe(`library:${item.path}`);
    expect(stateOf(row)).toBe("installed");
    // No second row for the same files.
    expect(entries.some((entry) => entry.id === `library:${item.path}`)).toBe(false);
    expect(entries).toHaveLength(2);
  });

  it("lists a local model with no catalog match as its own installed row", () => {
    // Not GGUF: the fuzzy score gives any GGUF folder 14 points against any
    // GGUF variant (format alone), which is why rule 2 also needs identity.
    const item = makeLibraryItem({
      name: "my-private-finetune",
      path: "/Users/dan/AI_Models/private/my-private-finetune",
      sourceKind: "Directory",
      format: "PyTorch",
      backend: null,
      quantization: null,
      directoryLabel: "AI_Models",
    });
    const entries = buildChatEntries([family], [item], {}, { now: NOW });
    const row = byId(entries, `library:${item.path}`);
    expect(row.source.kind).toBe("library");
    expect(row.source.variant).toBeNull();
    expect(stateOf(row)).toBe("installed");
    expect(row.repo).toBe("");
    expect(stateOf(byId(entries, `catalog:${qwenMlx.id}`))).toBe("available");
    expect(entries).toHaveLength(3);
  });

  it("lists a catalog variant that is not on disk as available", () => {
    const entries = buildChatEntries([family], [], {}, { now: NOW });
    expect(entries.map((entry) => entry.id)).toEqual([`catalog:${qwenMlx.id}`, `catalog:${qwenGguf.id}`]);
    expect(entries.map(stateOf)).toEqual(["available", "available"]);
    expect(byId(entries, `catalog:${qwenMlx.id}`).source.launchKey).toBe(`catalog:${qwenMlx.id}`);
  });

  it("adds a download row for a transfer whose repo no other row shows", () => {
    const download = transfer("org/hub-only-model", "downloading", { progress: 0.03, downloadedGb: 2, totalGb: 67 });
    const entries = buildChatEntries([family], [], { [download.repo]: download }, { now: NOW });
    const row = byId(entries, "library:download://org/hub-only-model");
    expect(row.source.kind).toBe("download");
    expect(row.localPath).toBeNull();
    expect(row.source.launchKey).toBeNull();
    expect(stateOf(row)).toBe("downloading");
    expect(row.source.modifiedAt).toBe(NOW);
  });

  it("keeps a fuzzy match without a confirmed identity as two rows, so the catalog row never claims to be installed", () => {
    const item = makeLibraryItem({
      name: "Qwen3-4B-Instruct-GGUF",
      path: "/Users/dan/AI_Models/other/Qwen3-4B-Instruct-GGUF",
      sourceKind: "Directory",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: null,
    });
    const entries = buildChatEntries([makeChatFamily({}, [qwenGguf])], [item], {}, { now: NOW });
    const library = byId(entries, `library:${item.path}`);
    expect(library.source.variant).toBe(qwenGguf); // display only
    expect(library.capabilities).toEqual(qwenGguf.capabilities);
    expect(stateOf(library)).toBe("installed");
    expect(stateOf(byId(entries, `catalog:${qwenGguf.id}`))).toBe("available");
  });

  it("merges on the backend's own availableLocally check when the folder does not name the repo", () => {
    const variant = { ...qwenGguf, availableLocally: true };
    const item = makeLibraryItem({
      name: "Qwen3-4B-GGUF",
      path: "/Users/dan/AI_Models/Qwen3-4B-GGUF",
      sourceKind: "Directory",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: "Q4_K_M",
    });
    const entries = buildChatEntries([makeChatFamily({}, [variant])], [item], {}, { now: NOW });
    expect(entries).toHaveLength(1);
    expect(byId(entries, `catalog:${variant.id}`).source.item).toBe(item);
  });

  it("merges a folder whose path ends in org/name (the old GGUF directory case)", () => {
    const item = makeLibraryItem({
      name: "Qwen3-4B-GGUF",
      path: "/Users/dan/AI_Models/lmstudio-community/Qwen3-4B-GGUF",
      sourceKind: "Directory",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: "Q4_K_M",
    });
    expect(libraryItemIsRepo(item, qwenGguf.repo)).toBe(true);
    const entries = buildChatEntries([makeChatFamily({}, [qwenGguf])], [item], {}, { now: NOW });
    expect(entries).toHaveLength(1);
    expect(byId(entries, `catalog:${qwenGguf.id}`).localPath).toBe(item.path);
  });

  it("merges only the variant's best local match when two local models want the same row", () => {
    const cache = makeLibraryItem({
      name: "lmstudio-community/Qwen3-4B-GGUF",
      path: "/cache/hub/models--lmstudio-community--Qwen3-4B-GGUF",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: "Q4_K_M",
    });
    const copy = makeLibraryItem({
      name: "Qwen3-4B-GGUF-copy",
      path: "/Users/dan/AI_Models/copies/Qwen3-4B-GGUF-copy",
      sourceKind: "Directory",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: "Q4_K_M",
    });
    const entries = buildChatEntries([makeChatFamily({}, [qwenGguf])], [copy, cache], {}, { now: NOW });
    expect(byId(entries, `catalog:${qwenGguf.id}`).source.item).toBe(cache);
    expect(byId(entries, `library:${copy.path}`).source.kind).toBe("library");
    expect(entries).toHaveLength(2);

    // Both pass the identity check when the backend says the variant is on
    // disk; only the variant's best match may merge, or the other vanishes.
    const onDisk = { ...qwenGguf, availableLocally: true };
    const both = buildChatEntries([makeChatFamily({}, [onDisk])], [copy, cache], {}, { now: NOW });
    expect(byId(both, `catalog:${onDisk.id}`).source.item).toBe(cache);
    expect(byId(both, `library:${copy.path}`).source.item).toBe(copy);
    expect(both).toHaveLength(2);
  });

  it("lists every catalog variant and every local model exactly once", () => {
    const items = [
      makeLibraryItem(),
      makeLibraryItem({ name: "custom-a", path: "/m/custom-a", sourceKind: "Directory", format: "GGUF", backend: "llama.cpp", quantization: null }),
      makeLibraryItem({ name: "custom-b", path: "/m/custom-b", sourceKind: "Directory", format: "GGUF", backend: "llama.cpp", quantization: null }),
    ];
    const entries = buildChatEntries([family], items, {}, { now: NOW });
    const ids = entries.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const variant of family.variants) expect(ids).toContain(`catalog:${variant.id}`);
    const shownItems = entries.map((entry) => entry.source.item).filter(Boolean);
    for (const item of items) expect(shownItems).toContain(item);
  });

  it("accepts a precomputed join from the same arrays", () => {
    const item = makeLibraryItem();
    const join = prepareChatJoin([family], [item]);
    const entries = buildChatEntries([family], [item], {}, { now: NOW, join });
    expect(byId(entries, `catalog:${qwenMlx.id}`).source.item).toBe(item);
  });
});

describe("chat status resolution", () => {
  it("shows a broken local model as broken even when the catalog says it is on disk", () => {
    const variant = { ...qwenMlx, availableLocally: true };
    const item = makeLibraryItem({ broken: true, brokenReason: "No .safetensors weights found" });
    const entries = buildChatEntries([makeChatFamily({}, [variant])], [item], {}, { now: NOW });
    const row = byId(entries, `catalog:${variant.id}`);
    expect(stateOf(row)).toBe("broken");
    expect(row.broken?.reason).toBe("No .safetensors weights found");
  });

  it("turns a partial Hugging Face cache entry into a paused download with progress from the bytes on disk", () => {
    const variant = { ...qwenMlx, sizeGb: 67 };
    const item = makeLibraryItem({ broken: true, sizeGb: 23.9, brokenReason: "Hugging Face download is incomplete: partial blob files are still present." });
    const entries = buildChatEntries([makeChatFamily({}, [variant])], [item], {}, { now: NOW });
    const row = byId(entries, `catalog:${variant.id}`);
    expect(stateOf(row)).toBe("paused");
    expect(row.source.download?.repo).toBe(variant.repo);
    expect(row.source.download?.progress).toBeCloseTo(23.9 / 67);
  });

  it("lets an active download win over installed files", () => {
    const item = makeLibraryItem();
    const entries = buildChatEntries([family], [item], { [qwenMlx.repo]: transfer(qwenMlx.repo, "downloading") }, { now: NOW });
    expect(stateOf(byId(entries, `catalog:${qwenMlx.id}`))).toBe("downloading");
  });

  it("shows a download on one row only: the catalog row of its repo, not a local model that merely resembles it", () => {
    const item = makeLibraryItem({
      name: "Qwen3-4B-Instruct-GGUF",
      path: "/Users/dan/AI_Models/other/Qwen3-4B-Instruct-GGUF",
      sourceKind: "Directory",
      format: "GGUF",
      backend: "llama.cpp",
      quantization: null,
    });
    const downloads = { [qwenGguf.repo]: transfer(qwenGguf.repo, "downloading") };
    const entries = buildChatEntries([makeChatFamily({}, [qwenGguf])], [item], downloads, { now: NOW });
    expect(stateOf(byId(entries, `catalog:${qwenGguf.id}`))).toBe("downloading");
    expect(stateOf(byId(entries, `library:${item.path}`))).toBe("installed");
    expect(entries.filter((entry) => entry.source.download)).toHaveLength(1);
  });

  it("marks the row of the loaded model", () => {
    const item = makeLibraryItem();
    const entries = buildChatEntries([family], [item], {}, { now: NOW, loadedModel: { ref: item.name, path: item.path } });
    const row = byId(entries, `catalog:${qwenMlx.id}`);
    expect(resolveModelStatus(row, { download: row.source.download }).loaded).toBe(true);
    expect(byId(entries, `catalog:${qwenGguf.id}`).loaded).toBe(false);
  });
});

describe("chat search (R8)", () => {
  const custom = makeLibraryItem({
    name: "my-finetune",
    path: "/Volumes/External/models/my-finetune",
    sourceKind: "Directory",
    format: "GGUF",
    backend: "llama.cpp",
    quantization: null,
    directoryLabel: "External SSD",
  });
  const entries = buildChatEntries([family], [makeLibraryItem(), custom], {}, { now: NOW });

  it("matches catalog rows per variant with the old Discover token rule", () => {
    expect(chatEntryMatchesQuery(byId(entries, `catalog:${qwenGguf.id}`), "qwen gguf")).toBe(true);
    expect(chatEntryMatchesQuery(byId(entries, `catalog:${qwenMlx.id}`), "qwen gguf")).toBe(false);
    // Family text counts for every variant.
    expect(chatEntryMatchesQuery(byId(entries, `catalog:${qwenGguf.id}`), "hybrid reasoning")).toBe(true);
  });

  it("matches local files on name, path, format, backend and directory label (old My Models rule)", () => {
    const row = byId(entries, `library:${custom.path}`);
    expect(chatEntryMatchesQuery(row, "external ssd")).toBe(true);
    expect(chatEntryMatchesQuery(row, "/volumes/external")).toBe(true);
    expect(chatEntryMatchesQuery(row, "llama.cpp")).toBe(true);
    expect(chatEntryMatchesQuery(row, "qwen")).toBe(false);
    // A merged row also matches on its local path.
    expect(chatEntryMatchesQuery(byId(entries, `catalog:${qwenMlx.id}`), "models--mlx-community")).toBe(true);
  });

  it("treats an empty query as a match", () => {
    expect(chatEntryMatchesQuery(byId(entries, `library:${custom.path}`), "  ")).toBe(true);
  });
});

describe("chat acceleration filter (R7)", () => {
  const compat: ChatAccelCompat = {
    dflashModels: ["Qwen/Qwen3-4B"],
    mtplxModels: ["mlx-community/Qwen3-4B-4bit"],
    tensorfoldModels: [],
    turboInstalled: false,
    turboquantMlxAvailable: true,
  };
  const predicate = makeChatAccelPredicate(compat);

  it("uses the catalog rule on rows without files", () => {
    const entries = buildChatEntries([family], [], {}, { now: NOW });
    expect(predicate(byId(entries, `catalog:${qwenGguf.id}`), "turboquant")).toBe(true);
    expect(predicate(byId(entries, `catalog:${qwenMlx.id}`), "turboquant")).toBe(false);
    expect(predicate(byId(entries, `catalog:${qwenMlx.id}`), "mtplx")).toBe(true);
  });

  it("uses the installed-model rule on rows with files", () => {
    const entries = buildChatEntries([family], [makeLibraryItem()], {}, { now: NOW });
    const merged = byId(entries, `catalog:${qwenMlx.id}`);
    // MLX + TurboQuant MLX available: yes; turbo binary missing does not matter for MLX.
    expect(predicate(merged, "turboquant")).toBe(true);
    expect(predicate(merged, "mtplx")).toBe(true);
    const gguf = makeLibraryItem({ name: "qwen3-4b-dflash.gguf", path: "/m/qwen3-4b-dflash.gguf", sourceKind: "File", format: "GGUF", backend: "llama.cpp" });
    const ggufRow = buildChatEntries([], [gguf], {}, { now: NOW })[0];
    // DFlash never applies to GGUF; TurboQuant needs the turbo binary.
    expect(predicate(ggufRow, "dflash")).toBe(false);
    expect(predicate(ggufRow, "turboquant")).toBe(false);
  });
});

describe("chat platform gating (R19)", () => {
  it("hides an MLX model the host cannot run unless it is on disk", () => {
    const item = makeLibraryItem();
    const onWindows = buildChatEntries([family], [item], {}, { now: NOW, host: WINDOWS_CUDA });
    const merged = byId(onWindows, `catalog:${qwenMlx.id}`);
    expect(merged.hostCompatible).toBe(false);
    const rows = buildModelRows({
      entries: onWindows,
      resolveStatus: (entry) => resolveModelStatus(entry, { download: entry.source.download }),
      availability: "all",
    }).rows.map((row) => row.entry.id);
    expect(rows).toContain(`catalog:${qwenMlx.id}`); // installed: stays, with the chip
    expect(rows).toContain(`catalog:${qwenGguf.id}`); // llama.cpp runs anywhere

    const notInstalled = buildChatEntries([family], [], {}, { now: NOW, host: WINDOWS_CUDA });
    const visible = buildModelRows({
      entries: notInstalled,
      resolveStatus: (entry) => resolveModelStatus(entry),
      availability: "all",
    }).rows.map((row) => row.entry.id);
    expect(visible).toEqual([`catalog:${qwenGguf.id}`]);
    expect(buildChatEntries([family], [], {}, { now: NOW, host: MAC }).every((entry) => entry.hostCompatible)).toBe(true);
  });
});

describe("chat favourites", () => {
  it("keeps the old My Models favourite id so stored favourites still match", () => {
    expect(chatFavoriteRef(makeLibraryItem(), null)).toBe("mlx-community/Qwen3-4B-4bit");
    const dir = makeLibraryItem({ name: "my-model", path: "/m/my-model" });
    expect(chatFavoriteRef(dir, qwenGguf)).toBe(qwenGguf.repo);
    expect(chatFavoriteRef(dir, null)).toBe("/m/my-model");
  });

  it("pins favourite rows above the others whatever the sort", () => {
    const custom = makeLibraryItem({ name: "zz-custom", path: "/m/zz-custom", sourceKind: "Directory", format: "GGUF", backend: "llama.cpp", quantization: null });
    const entries = buildChatEntries([family], [custom], {}, { now: NOW });
    const favourite = byId(entries, `library:${custom.path}`).source.favoriteRef;
    const rows = buildModelRows({
      entries,
      resolveStatus: (entry) => resolveModelStatus(entry, { download: entry.source.download }),
      availability: "all",
      sort: { id: "name", label: "Name", defaultDir: "asc", compare: (a, b) => a.entry.name.localeCompare(b.entry.name) },
      sortDir: "asc",
      pinned: (entry) => entry.source.favoriteRef === favourite,
    }).rows;
    expect(rows[0].entry.id).toBe(`library:${custom.path}`);
  });
});

describe("findLibraryItemForRepo (R3)", () => {
  it("finds the local files of a Hugging Face repo by cache folder, name or path", () => {
    const cache = makeLibraryItem();
    const folder = makeLibraryItem({ name: "GLM-4.7-Flash", path: "C:\\Models\\zai-org\\GLM-4.7-Flash", sourceKind: "Directory" });
    expect(findLibraryItemForRepo([folder, cache], "mlx-community/Qwen3-4B-4bit")).toBe(cache);
    expect(findLibraryItemForRepo([cache, folder], "zai-org/GLM-4.7-Flash")).toBe(folder);
    expect(findLibraryItemForRepo([cache], "org/unknown")).toBeNull();
  });
});
