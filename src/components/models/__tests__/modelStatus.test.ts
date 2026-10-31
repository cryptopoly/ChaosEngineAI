import { describe, expect, it } from "vitest";
import type { DownloadStatus } from "../../../api";
import {
  countBuckets,
  isVisibleOnHost,
  matchesAvailability,
  resolveModelStatus,
  statusSortRank,
} from "../modelStatus";

function download(state: DownloadStatus["state"], overrides: Partial<DownloadStatus> = {}): DownloadStatus {
  return { repo: "org/model", state, progress: 0.4, downloadedGb: 2, totalGb: 5, error: null, ...overrides };
}

const nothing = { availableLocally: false, hasLocalData: false };

describe("resolveModelStatus", () => {
  it("is available when nothing is on disk or in flight", () => {
    const status = resolveModelStatus(nothing);
    expect(status).toMatchObject({ state: "available", bucket: "available", chip: "incomplete", loaded: false });
  });

  it("is installed when the backend validated the files", () => {
    expect(resolveModelStatus({ availableLocally: true, hasLocalData: true })).toMatchObject({
      state: "installed",
      bucket: "installed",
      chip: "installed",
    });
  });

  it("is installed when the download completed before the catalog refreshed", () => {
    expect(resolveModelStatus(nothing, { download: download("completed") }).state).toBe("installed");
  });

  it("marks an installed model in memory as loaded (a chip, not a bucket)", () => {
    const status = resolveModelStatus({ availableLocally: true, hasLocalData: true, loaded: true });
    expect(status).toMatchObject({ state: "installed", bucket: "installed", chip: "loaded", loaded: true });
  });

  it("does not call a model that is not installed loaded", () => {
    expect(resolveModelStatus({ ...nothing, loaded: true }).loaded).toBe(false);
  });

  it("puts downloading and paused in the Downloading bucket", () => {
    expect(resolveModelStatus(nothing, { download: download("downloading") })).toMatchObject({
      state: "downloading",
      bucket: "downloading",
    });
    expect(resolveModelStatus(nothing, { download: download("cancelled") })).toMatchObject({
      state: "paused",
      bucket: "downloading",
      chip: "paused",
    });
  });

  it("lets an active transfer win over installed files", () => {
    // A shared repo can be valid for one variant while a sibling's
    // download runs: the row needs its Pause control.
    expect(
      resolveModelStatus({ availableLocally: true, hasLocalData: true }, { download: download("downloading") }).state,
    ).toBe("downloading");
  });

  it("keeps an installed model installed when a later download failed", () => {
    expect(
      resolveModelStatus({ availableLocally: true, hasLocalData: true }, { download: download("failed", { error: "x" }) }).state,
    ).toBe("installed");
  });

  it("puts failed, broken and partial rows in Needs attention", () => {
    expect(resolveModelStatus(nothing, { download: download("failed") })).toMatchObject({ state: "failed", bucket: "attention" });
    expect(resolveModelStatus({ ...nothing, broken: { reason: "no weights" } })).toMatchObject({
      state: "broken",
      bucket: "attention",
    });
    expect(resolveModelStatus({ availableLocally: false, hasLocalData: true })).toMatchObject({
      state: "partial",
      bucket: "attention",
    });
  });

  it("uses the runtime probe for installed (LongLive) and its install job for downloading", () => {
    expect(resolveModelStatus({ ...nothing, installedByRuntime: true }).state).toBe("installed");
    expect(resolveModelStatus({ ...nothing, installedByRuntime: false }).state).toBe("available");
    expect(resolveModelStatus(nothing, { installing: true }).state).toBe("downloading");
  });

  it("carries the download it came from", () => {
    const active = download("downloading");
    expect(resolveModelStatus(nothing, { download: active }).download).toBe(active);
  });
});

describe("status buckets", () => {
  it("matches the All filter for every bucket and otherwise only its own", () => {
    expect(matchesAvailability("installed", "all")).toBe(true);
    expect(matchesAvailability("installed", "installed")).toBe(true);
    expect(matchesAvailability("installed", "available")).toBe(false);
    expect(matchesAvailability("attention", "attention")).toBe(true);
  });

  it("counts every bucket and the total", () => {
    const counts = countBuckets([
      resolveModelStatus({ availableLocally: true, hasLocalData: true }),
      resolveModelStatus(nothing),
      resolveModelStatus(nothing),
      resolveModelStatus(nothing, { download: download("cancelled") }),
      resolveModelStatus({ availableLocally: false, hasLocalData: true }),
    ]);
    expect(counts).toEqual({ all: 5, installed: 1, available: 2, downloading: 1, attention: 1 });
  });

  it("ranks installed first and available last for the Status sort", () => {
    const installed = resolveModelStatus({ availableLocally: true, hasLocalData: true });
    const loaded = resolveModelStatus({ availableLocally: true, hasLocalData: true, loaded: true });
    const available = resolveModelStatus(nothing);
    expect(statusSortRank(loaded)).toBeLessThan(statusSortRank(installed));
    expect(statusSortRank(installed)).toBeLessThan(statusSortRank(available));
  });
});

describe("isVisibleOnHost", () => {
  it("hides a row the host cannot run when nothing of it is on disk", () => {
    expect(isVisibleOnHost({ hostCompatible: false }, resolveModelStatus(nothing))).toBe(false);
  });

  it("keeps an installed or partial row the host cannot run, so it can be deleted", () => {
    expect(isVisibleOnHost({ hostCompatible: false }, resolveModelStatus({ availableLocally: true, hasLocalData: true }))).toBe(true);
    expect(isVisibleOnHost({ hostCompatible: false }, resolveModelStatus({ availableLocally: false, hasLocalData: true }))).toBe(true);
  });

  it("shows every row the host can run", () => {
    expect(isVisibleOnHost({ hostCompatible: true }, resolveModelStatus(nothing))).toBe(true);
  });
});
