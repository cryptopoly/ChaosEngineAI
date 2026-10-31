import { describe, expect, it, vi } from "vitest";
import type { VideoModelVariant } from "../../../types";
import { buildDeleteConfirmMessage, confirmAndDelete } from "../deleteConfirm";
import { resolveModelStatus } from "../modelStatus";
import { buildModelRows, type ModelSortOption } from "../modelFilters";
import {
  buildVideoEntries,
  buildVideoRepoUsers,
  collectVideoVariants,
  hasVisibleLongLiveRow,
  isLongLiveRepo,
  videoDeleteConfirmEntry,
  videoEntryMatchesQuery,
  videoRowActions,
  videoSizeSortKey,
  videoVariantToEntry,
  type VideoModelEntry,
} from "../adapters/video";
import {
  CUDA_PC,
  MAC,
  downloadState,
  makeLongLiveVariant,
  makeVideoFamily,
  makeVideoVariant,
  wan14bRows,
} from "./videoFixtures";

function wanEntries(): VideoModelEntry[] {
  return buildVideoEntries([makeVideoFamily({ id: "wan-2-1", name: "Wan 2.1" }, wan14bRows())], []);
}

function byId(entries: VideoModelEntry[], id: string): VideoModelEntry {
  const entry = entries.find((candidate) => candidate.id === id);
  if (!entry) throw new Error(`no entry ${id}`);
  return entry;
}

describe("collectVideoVariants", () => {
  it("lists every curated variant with its family name, then tracked rows not already curated (R26)", () => {
    const wan = makeVideoVariant();
    const family = makeVideoFamily({}, [wan]);
    const duplicate = makeVideoVariant({ id: "latest-wan", repo: "wan-ai/wan2.1-t2v-1.3b-diffusers" });
    const fresh = makeVideoVariant({ id: "org/new-video", repo: "org/new-video", name: "New Video" });
    const twice = makeVideoVariant({ id: "org/new-video-2", repo: "ORG/new-video", name: "New dup" });
    const variants = collectVideoVariants([family], [duplicate, fresh, twice]);
    expect(variants.map((variant) => variant.id)).toEqual([wan.id, "org/new-video"]);
    expect(variants[0].familyName).toBe("Wan 2.1");
  });
});

describe("videoVariantToEntry", () => {
  it("keys the download by variant id and deletes the local repo (R12)", () => {
    const [, q4] = wan14bRows();
    const entry = videoVariantToEntry(q4, { host: MAC });
    expect(entry).toMatchObject({
      id: "city96/Wan2.1-T2V-14B-gguf-q4km",
      area: "video",
      repo: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      downloadKey: "city96/Wan2.1-T2V-14B-gguf-q4km",
      deleteTarget: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      hostCompatible: true,
      longLive: false,
      fit: null,
      bestFit: false,
    });
    expect(entry.downloadKey).not.toBe(entry.repo);
    expect(entry.shared).toEqual({
      count: 2,
      key: "city96/Wan2.1-T2V-14B-gguf",
      siblingNames: ["city96/Wan2.1-T2V-14B-gguf-q6k", "city96/Wan2.1-T2V-14B-gguf-q8"],
    });
  });

  it("deletes the shared GGUF repo when only the GGUF files are on disk", () => {
    const variant = makeVideoVariant({
      repo: "Lightricks/LTX-Video",
      ggufRepo: "city96/LTX-Video-gguf",
      ggufFile: "ltx-video-2b-v0.9-Q6_K.gguf",
      hasLocalData: true,
      localDataRepos: ["city96/LTX-Video-gguf"],
      primaryLocalRepo: "city96/LTX-Video-gguf",
    });
    expect(videoVariantToEntry(variant).deleteTarget).toBe("city96/LTX-Video-gguf");
  });

  it("sizes a GGUF quant by its own file, not the shared repo (R13)", () => {
    const [base, q4, q6, q8] = wan14bRows();
    expect(videoSizeSortKey(q4)).toBe(10.1);
    expect(videoSizeSortKey(q6)).toBe(12.5);
    expect(videoSizeSortKey(q8)).toBe(15.9);
    expect(videoSizeSortKey(base)).toBe(57);
    expect(videoVariantToEntry(q4).sizeLabel).toBe("10.1 GB");
  });

  it("marks only the row the runtime has loaded", () => {
    const [base, q4] = wan14bRows();
    expect(videoVariantToEntry(q4, { loadedVariantId: q4.id }).loaded).toBe(true);
    expect(videoVariantToEntry(base, { loadedVariantId: q4.id }).loaded).toBe(false);
  });

  it("hides Mac-only rows on a PC and CUDA-only rows on a Mac", () => {
    const mlx = makeVideoVariant({ id: "mlx", repo: "prince-canuma/LTX-2-distilled", runtime: "mlx-video (MLX native)" });
    expect(videoVariantToEntry(mlx, { host: MAC }).hostCompatible).toBe(true);
    expect(videoVariantToEntry(mlx, { host: CUDA_PC }).hostCompatible).toBe(false);
    const longLive = makeLongLiveVariant();
    expect(videoVariantToEntry(longLive, { host: MAC }).hostCompatible).toBe(false);
    expect(videoVariantToEntry(longLive, { host: CUDA_PC }).hostCompatible).toBe(true);
  });

  it("shows CUDA accelerator pills only where they apply", () => {
    const wan = makeVideoVariant();
    expect(videoVariantToEntry(wan, { host: MAC }).accelerators).toEqual([]);
    expect(videoVariantToEntry(wan, { host: CUDA_PC }).accelerators?.length).toBeGreaterThan(0);
    const mlx = makeVideoVariant({ repo: "prince-canuma/LTX-2-distilled", runtime: "mlx-video (MLX native)" });
    expect(videoVariantToEntry(mlx, { host: CUDA_PC }).accelerators).toEqual([]);
  });

  it("reads fit defensively and marks the family's best fit", () => {
    const variant = makeVideoVariant({
      fit: { status: "tight", needGb: 40, usableGb: 51.8, headroomPct: 23, reason: "40 GB peak", basis: "estimated" },
    } as Partial<VideoModelVariant>);
    const family = { ...makeVideoFamily({}, [variant]), bestFitVariantId: variant.id };
    const entry = videoVariantToEntry(variant, { family });
    expect(entry.fit).toMatchObject({ status: "tight", basis: "estimated" });
    expect(entry.bestFit).toBe(true);
  });

  it("takes LongLive's install state from the runtime probe only", () => {
    expect(isLongLiveRepo("NVlabs/LongLive-1.3B")).toBe(true);
    expect(isLongLiveRepo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")).toBe(false);
    // A stale catalog flag must not make LongLive look installed.
    const variant = makeLongLiveVariant({ availableLocally: true, hasLocalData: true });
    const notReady = videoVariantToEntry(variant, { longLiveReady: false });
    expect(notReady).toMatchObject({ longLive: true, availableLocally: false, hasLocalData: false, installedByRuntime: false });
    expect(resolveModelStatus(notReady).state).toBe("available");
    const ready = videoVariantToEntry(variant, { longLiveReady: true });
    expect(resolveModelStatus(ready).state).toBe("installed");
    expect(resolveModelStatus(notReady, { installing: true }).state).toBe("downloading");
  });
});

describe("hasVisibleLongLiveRow (R25)", () => {
  it("is true only when a LongLive row can show on this host", () => {
    const family = makeVideoFamily({ id: "longlive" }, [makeLongLiveVariant()]);
    expect(hasVisibleLongLiveRow(buildVideoEntries([family], [], { host: CUDA_PC }))).toBe(true);
    expect(hasVisibleLongLiveRow(buildVideoEntries([family], [], { host: MAC }))).toBe(false);
    expect(hasVisibleLongLiveRow(wanEntries())).toBe(false);
  });
});

describe("videoRowActions (R12)", () => {
  it("downloads with (repo, variant id), pauses the running repo and cancels the delete repo", () => {
    const entries = wanEntries();
    const q4 = byId(entries, "city96/Wan2.1-T2V-14B-gguf-q4km");
    const callbacks = { onVideoDownload: vi.fn(), onCancelVideoDownload: vi.fn(), onDeleteVideoDownload: vi.fn() };
    const running = downloadState("city96/Wan2.1-T2V-14B-gguf", "downloading");
    const actions = videoRowActions(q4, running, callbacks);
    actions.download();
    expect(callbacks.onVideoDownload).toHaveBeenCalledWith("Wan-AI/Wan2.1-T2V-14B-Diffusers", "city96/Wan2.1-T2V-14B-gguf-q4km");
    actions.pause();
    expect(callbacks.onCancelVideoDownload).toHaveBeenCalledWith("city96/Wan2.1-T2V-14B-gguf");
    actions.cancel();
    expect(callbacks.onDeleteVideoDownload).toHaveBeenCalledWith("city96/Wan2.1-T2V-14B-gguf");
    expect(actions.deleteKind).toBe("gguf");
  });

  it("pauses the variant's repo when the download has no repo of its own", () => {
    const entries = wanEntries();
    const q6 = byId(entries, "city96/Wan2.1-T2V-14B-gguf-q6k");
    const callbacks = { onVideoDownload: vi.fn(), onCancelVideoDownload: vi.fn(), onDeleteVideoDownload: vi.fn() };
    const actions = videoRowActions(q6, null, callbacks);
    actions.pause();
    expect(callbacks.onCancelVideoDownload).toHaveBeenCalledWith("Wan-AI/Wan2.1-T2V-14B-Diffusers");
    expect(actions.deleteKind).toBe("primary");
  });

  it("names a text-encoder repo delete as such", () => {
    const variant = makeVideoVariant({
      id: "prince-canuma/LTX-2.3-distilled",
      repo: "prince-canuma/LTX-2.3-distilled",
      textEncoderRepo: "prince-canuma/LTX-2-distilled",
      hasLocalData: true,
      localDataRepos: ["prince-canuma/LTX-2-distilled"],
      primaryLocalRepo: "prince-canuma/LTX-2-distilled",
    });
    const entry = videoVariantToEntry(variant);
    const actions = videoRowActions(entry, null, {
      onVideoDownload: vi.fn(),
      onCancelVideoDownload: vi.fn(),
      onDeleteVideoDownload: vi.fn(),
    });
    expect(actions.deleteRepo).toBe("prince-canuma/LTX-2-distilled");
    expect(actions.deleteKind).toBe("textEncoder");
  });
});

describe("videoDeleteConfirmEntry", () => {
  it("names every row that loses files when a quant's base repo is deleted", () => {
    const entries = wanEntries();
    const users = buildVideoRepoUsers(entries);
    const q4 = byId(entries, "city96/Wan2.1-T2V-14B-gguf-q4km");
    const confirmEntry = videoDeleteConfirmEntry(q4, "Wan-AI/Wan2.1-T2V-14B-Diffusers", users);
    expect(confirmEntry.deleteTarget).toBe("Wan-AI/Wan2.1-T2V-14B-Diffusers");
    expect(confirmEntry.shared?.count).toBe(3);
    const message = buildDeleteConfirmMessage(confirmEntry);
    expect(message).toContain('Delete the files of "Wan 2.1 14B · GGUF q4km"?');
    expect(message).toContain("3 other models use the same download and will be removed too");
    expect(message).toContain("• Wan 2.1 14B");
    expect(message).toContain("• Wan 2.1 14B · GGUF q6k");
    expect(message).toContain("• Wan 2.1 14B · GGUF q8");
  });

  it("names the GGUF siblings and the GGUF repo, not the base snapshot, for a shared GGUF delete", () => {
    const entries = wanEntries();
    const users = buildVideoRepoUsers(entries);
    const base = byId(entries, "Wan-AI/Wan2.1-T2V-14B-Diffusers");
    const q4 = byId(entries, "city96/Wan2.1-T2V-14B-gguf-q4km");
    const ggufDelete = videoDeleteConfirmEntry(q4, "city96/Wan2.1-T2V-14B-gguf", users);
    expect(ggufDelete.localPath).toBeNull();
    const message = buildDeleteConfirmMessage(ggufDelete);
    expect(message).toContain("This removes city96/Wan2.1-T2V-14B-gguf from disk");
    expect(message).toContain("2 other models use the same download");
    expect(message).not.toContain("• Wan 2.1 14B\n");
    // The base row's own delete shows its snapshot path.
    expect(videoDeleteConfirmEntry(base, "Wan-AI/Wan2.1-T2V-14B-Diffusers", users).localPath).toBe(base.localPath);
  });

  it("asks before deleting and deletes nothing on a no", () => {
    const entries = wanEntries();
    const users = buildVideoRepoUsers(entries);
    const base = byId(entries, "Wan-AI/Wan2.1-T2V-14B-Diffusers");
    const onDelete = vi.fn();
    const confirm = vi.fn((_message: string) => false);
    expect(confirmAndDelete(videoDeleteConfirmEntry(base, base.deleteTarget, users), onDelete, confirm)).toBe(false);
    expect(confirm.mock.calls[0][0]).toContain("3 other models use the same download");
    expect(onDelete).not.toHaveBeenCalled();
    confirmAndDelete(videoDeleteConfirmEntry(base, base.deleteTarget, users), onDelete, () => true);
    expect(onDelete).toHaveBeenCalledWith("Wan-AI/Wan2.1-T2V-14B-Diffusers");
  });
});

describe("video size sort", () => {
  it("orders GGUF quants by their own file size", () => {
    const entries = wanEntries();
    const sizeSort: ModelSortOption<VideoModelEntry> = {
      id: "size",
      label: "Size",
      defaultDir: "desc",
      value: (row) => row.entry.sizeGb,
    };
    const { rows } = buildModelRows({
      entries,
      resolveStatus: (entry) => resolveModelStatus(entry),
      availability: "all",
      sort: sizeSort,
      sortDir: "asc",
    });
    expect(rows.map((row) => row.entry.id)).toEqual([
      "city96/Wan2.1-T2V-14B-gguf-q4km",
      "city96/Wan2.1-T2V-14B-gguf-q6k",
      "city96/Wan2.1-T2V-14B-gguf-q8",
      "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ]);
  });
});

describe("videoEntryMatchesQuery", () => {
  it("matches the row's own text or its family's text", () => {
    const wan = makeVideoVariant();
    const other = makeVideoVariant({ id: "other", name: "Other", repo: "org/other", note: "", styleTags: [] });
    const family = makeVideoFamily({ summary: "Open text-to-video weights" }, [wan, other]);
    const entries = buildVideoEntries([family], []);
    const families = new Map([[family.id, family]]);
    expect(videoEntryMatchesQuery(entries[0], "1.3b", families)).toBe(true);
    expect(videoEntryMatchesQuery(entries[1], "weights", families)).toBe(true);
    expect(videoEntryMatchesQuery(entries[1], "nothing-like-this", families)).toBe(false);
    expect(videoEntryMatchesQuery(entries[1], "", families)).toBe(true);
  });
});
