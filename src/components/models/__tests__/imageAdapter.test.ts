import { describe, expect, it } from "vitest";
import type { ImageModelVariant } from "../../../types";
import { makeImageFamily, makeImageVariant } from "./fixtures";
import { buildImageEntries, collectImageVariants, imageEntryMatchesQuery, imageVariantToEntry } from "../adapters/image";

const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };

describe("collectImageVariants", () => {
  it("lists every curated variant with its family name, then tracked rows not already curated (R26)", () => {
    const schnell = makeImageVariant();
    const mflux = makeImageVariant({ id: "flux-schnell-mflux", name: "FLUX.1 Schnell · mflux", runtime: "mflux" });
    const family = makeImageFamily({}, [schnell, mflux]);
    const trackedDuplicate = makeImageVariant({
      id: "latest-flux",
      familyId: "latest",
      repo: "Black-Forest-Labs/FLUX.1-Schnell",
      source: "latest",
    });
    const trackedNew = makeImageVariant({ id: "baidu/ERNIE-Image", repo: "baidu/ERNIE-Image", name: "ERNIE-Image", trackedOnly: true });
    const trackedTwice = makeImageVariant({ id: "baidu/ERNIE-Image-2", repo: "baidu/ernie-image", name: "ERNIE dup" });
    const variants = collectImageVariants([family], [trackedDuplicate, trackedNew, trackedTwice]);
    expect(variants.map((variant) => variant.id)).toEqual([schnell.id, mflux.id, "baidu/ERNIE-Image"]);
    expect(variants[0].familyName).toBe("FLUX Fast");
  });
});

describe("imageVariantToEntry", () => {
  it("maps the fields the shared page reads", () => {
    const entry = imageVariantToEntry(makeImageVariant({ license: "apache-2.0", gated: false }), { host: MAC });
    expect(entry).toMatchObject({
      id: "black-forest-labs/FLUX.1-schnell",
      area: "image",
      downloadKey: "black-forest-labs/FLUX.1-schnell",
      deleteTarget: "black-forest-labs/FLUX.1-schnell",
      hostCompatible: true,
      gated: false,
      license: "apache-2.0",
      releaseSortKey: "2024-08",
      tasks: ["txt2img"],
      fit: null,
      bestFit: false,
      shared: null,
    });
    expect(entry.source.id).toBe("black-forest-labs/FLUX.1-schnell");
  });

  it("reads fit defensively and marks the family's best fit", () => {
    const variant = makeImageVariant({
      fit: { status: "fits", needGb: 14, usableGb: 51.8, headroomPct: 73, reason: "14 GB peak", basis: "measured" },
    } as Partial<ImageModelVariant>);
    const family = { ...makeImageFamily({}, [variant]), bestFitVariantId: variant.id };
    const entry = imageVariantToEntry(variant, { family });
    expect(entry.fit).toMatchObject({ status: "fits", basis: "measured" });
    expect(entry.bestFit).toBe(true);
    const otherPick = { ...family, bestFitVariantId: "other" };
    expect(imageVariantToEntry(variant, { family: otherPick }).bestFit).toBe(false);
  });

  it("keeps a Mac-only row compatible on a Mac and not on a CUDA PC", () => {
    const mflux = makeImageVariant({ id: "mflux", runtime: "mflux (MLX native)" });
    expect(imageVariantToEntry(mflux, { host: MAC }).hostCompatible).toBe(true);
    expect(imageVariantToEntry(mflux, { host: CUDA_PC }).hostCompatible).toBe(false);
  });

  it("shows CUDA accelerator pills only on a host with an NVIDIA GPU (host-aware pills)", () => {
    const flux = makeImageVariant();
    expect(imageVariantToEntry(flux, { host: MAC }).accelerators).toEqual([]);
    expect(imageVariantToEntry(flux, { host: CUDA_PC }).accelerators).toEqual(expect.arrayContaining(["nunchaku", "sageattention"]));
    const mflux = makeImageVariant({ runtime: "mflux" });
    expect(imageVariantToEntry(mflux, { host: CUDA_PC }).accelerators).toEqual([]);
  });

  it("marks the row the runtime has loaded", () => {
    const variant = makeImageVariant({ availableLocally: true });
    expect(imageVariantToEntry(variant, { loadedRepo: variant.repo }).loaded).toBe(true);
    expect(imageVariantToEntry(variant, { loadedRepo: "other/repo" }).loaded).toBe(false);
  });
});

describe("buildImageEntries", () => {
  it("names the variants that share a download", () => {
    const schnell = makeImageVariant({
      sharedRepoSiblings: 1,
      sharedRepoSiblingIds: ["flux-schnell-mflux"],
      sharedRepoKey: "black-forest-labs/FLUX.1-schnell",
    });
    const mflux = makeImageVariant({
      id: "flux-schnell-mflux",
      name: "FLUX.1 Schnell · mflux",
      sharedRepoSiblings: 1,
      sharedRepoSiblingIds: ["black-forest-labs/FLUX.1-schnell"],
    });
    const entries = buildImageEntries([makeImageFamily({}, [schnell, mflux])], []);
    expect(entries[0].shared).toEqual({
      count: 1,
      key: "black-forest-labs/FLUX.1-schnell",
      siblingNames: ["FLUX.1 Schnell · mflux"],
    });
    expect(entries[1].shared?.siblingNames).toEqual(["FLUX.1 Schnell"]);
  });
});

describe("imageEntryMatchesQuery", () => {
  it("matches the row's own text or its family's text", () => {
    const schnell = makeImageVariant();
    const dev = makeImageVariant({ id: "dev", name: "Dev", repo: "org/dev", note: "", styleTags: [] });
    const family = makeImageFamily({ summary: "Distilled FLUX models" }, [schnell, dev]);
    const entries = buildImageEntries([family], []);
    const families = new Map([[family.id, family]]);
    expect(imageEntryMatchesQuery(entries[0], "schnell", families)).toBe(true);
    // "distilled" is only in the family summary: both rows of the family match.
    expect(imageEntryMatchesQuery(entries[1], "distilled", families)).toBe(true);
    expect(imageEntryMatchesQuery(entries[1], "nothing-like-this", families)).toBe(false);
    expect(imageEntryMatchesQuery(entries[1], "", families)).toBe(true);
  });
});
