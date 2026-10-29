import { describe, expect, it, vi } from "vitest";
import { buildDeleteConfirmMessage, confirmAndDelete } from "../deleteConfirm";

const single = {
  name: "FLUX.1 Schnell",
  repo: "black-forest-labs/FLUX.1-schnell",
  localPath: "/Users/me/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell",
  deleteTarget: "black-forest-labs/FLUX.1-schnell",
  shared: null,
};

describe("buildDeleteConfirmMessage", () => {
  it("names the model and what leaves the disk", () => {
    const message = buildDeleteConfirmMessage(single);
    expect(message).toContain('Delete the files of "FLUX.1 Schnell"?');
    expect(message).toContain(single.localPath);
    expect(message).toContain("cannot be undone");
    expect(message).not.toContain("same download");
  });

  it("falls back to the delete target when there is no local path", () => {
    expect(buildDeleteConfirmMessage({ ...single, localPath: null })).toContain("black-forest-labs/FLUX.1-schnell");
  });

  it("names every sibling that shares the download", () => {
    const message = buildDeleteConfirmMessage({
      ...single,
      shared: { count: 2, key: single.repo, siblingNames: ["FLUX.1 Schnell · mflux", "FLUX.1 Schnell · Nunchaku"] },
    });
    expect(message).toContain("2 other models use the same download and will be removed too");
    expect(message).toContain("• FLUX.1 Schnell · mflux");
    expect(message).toContain("• FLUX.1 Schnell · Nunchaku");
  });

  it("uses the singular for one sibling and summarises a long list", () => {
    expect(
      buildDeleteConfirmMessage({ ...single, shared: { count: 1, key: single.repo, siblingNames: ["Other"] } }),
    ).toContain("1 other model uses the same download");
    const many = Array.from({ length: 9 }, (_, index) => `Variant ${index + 1}`);
    const message = buildDeleteConfirmMessage({ ...single, shared: { count: 9, key: single.repo, siblingNames: many } });
    expect(message).toContain("• Variant 6");
    expect(message).not.toContain("• Variant 7");
    expect(message).toContain("and 3 more");
  });
});

describe("confirmAndDelete", () => {
  it("deletes nothing when the user says no", () => {
    const onDelete = vi.fn();
    const confirm = vi.fn(() => false);
    expect(confirmAndDelete(single, onDelete, confirm)).toBe(false);
    expect(confirm).toHaveBeenCalledOnce();
    expect(onDelete).not.toHaveBeenCalled();
  });

  it("deletes the entry's target on a yes", () => {
    const onDelete = vi.fn();
    expect(confirmAndDelete(single, onDelete, () => true)).toBe(true);
    expect(onDelete).toHaveBeenCalledWith("black-forest-labs/FLUX.1-schnell");
  });

  it("asks through window.confirm by default", () => {
    const confirm = vi.fn(() => false);
    vi.stubGlobal("window", { confirm });
    try {
      const onDelete = vi.fn();
      confirmAndDelete(single, onDelete);
      expect(confirm).toHaveBeenCalledOnce();
      expect(onDelete).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
