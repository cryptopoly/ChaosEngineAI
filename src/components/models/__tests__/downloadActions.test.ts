import { describe, expect, it } from "vitest";
import { downloadActionsFor } from "../downloadActions";

const kinds = (state: Parameters<typeof downloadActionsFor>[0]["state"], options?: Parameters<typeof downloadActionsFor>[1]) =>
  downloadActionsFor({ state }, options).map((action) => action.kind);

describe("downloadActionsFor", () => {
  it("offers Download on an available row", () => {
    expect(kinds("available")).toEqual(["download"]);
  });

  it("offers Pause and Cancel while downloading", () => {
    expect(kinds("downloading")).toEqual(["pause", "cancel"]);
  });

  it("offers Resume and Delete download on a paused row, never Download model (R10)", () => {
    const actions = downloadActionsFor({ state: "paused" });
    expect(actions.map((action) => action.kind)).toEqual(["resume", "deleteDownload"]);
    expect(actions[0]).toMatchObject({ icon: "resume", defaultLabel: "Resume download" });
    expect(actions.map((action) => action.defaultLabel)).not.toContain("Download model");
  });

  it("offers Retry and Delete download on a failed or broken row", () => {
    expect(kinds("failed")).toEqual(["retry", "deleteDownload"]);
    expect(kinds("broken")).toEqual(["retry", "deleteDownload"]);
  });

  it("offers Resume and Delete model on partial local data", () => {
    expect(kinds("partial")).toEqual(["resume", "deleteModel"]);
  });

  it("offers Delete model on an installed row when the area can delete", () => {
    expect(kinds("installed")).toEqual(["deleteModel"]);
    expect(kinds("installed", { canDeleteModel: false })).toEqual([]);
    expect(kinds("partial", { canDeleteModel: false })).toEqual(["resume"]);
  });

  it("gives a tracked-only row one disabled button whatever its state (FU-061)", () => {
    for (const state of ["available", "installed", "downloading", "paused"] as const) {
      const actions = downloadActionsFor({ state }, { trackedOnly: true });
      expect(actions).toHaveLength(1);
      expect(actions[0]).toMatchObject({ kind: "trackedOnly", disabled: true });
    }
  });

  it("marks destructive actions as danger", () => {
    const danger = downloadActionsFor({ state: "downloading" }).filter((action) => action.danger).map((action) => action.kind);
    expect(danger).toEqual(["cancel"]);
  });
});
