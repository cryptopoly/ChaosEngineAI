import { describe, expect, it } from "vitest";
import type { DownloadStatus } from "../../api";
import { downloadProgressPercent } from "../downloads";

function row(partial: Partial<DownloadStatus>): DownloadStatus {
  return { repo: "a/b", state: "downloading", progress: 0, downloadedGb: 0, totalGb: 10, error: null, ...partial };
}

describe("downloadProgressPercent()", () => {
  it("rounds the progress to a percent", () => {
    expect(downloadProgressPercent(row({ progress: 0.426, downloadedGb: 4.26 }))).toBe(43);
  });
  it("shows at least 1 % once bytes arrived, 0 % before", () => {
    expect(downloadProgressPercent(row({ progress: 0.001, downloadedGb: 0.01 }))).toBe(1);
    expect(downloadProgressPercent(row({}))).toBe(0);
  });
  it("never shows 100 % until the download completed", () => {
    expect(downloadProgressPercent(row({ progress: 1, downloadedGb: 10 }))).toBe(99);
    expect(downloadProgressPercent(row({ state: "completed", progress: 1 }))).toBe(100);
  });
  it("keeps the percent of a paused download", () => {
    expect(downloadProgressPercent(row({ state: "cancelled", progress: 0.5, downloadedGb: 5 }))).toBe(50);
  });
  it("is indeterminate without a known total or a record", () => {
    expect(downloadProgressPercent(row({ totalGb: null, downloadedGb: 2 }))).toBeNull();
    expect(downloadProgressPercent(null)).toBeNull();
  });
});
