export type ImageGalleryRuntimeFilter = "all" | "diffusers" | "placeholder" | "warning";
export type ImageGalleryOrientationFilter = "all" | "square" | "portrait" | "landscape";
export type ImageGallerySort = "newest" | "oldest";
export type ImageDiscoverTaskFilter = "all" | "txt2img" | "img2img" | "inpaint";
export type ImageDiscoverAccessFilter = "all" | "open" | "gated";
/** Discover sort axis. ``release`` = most recently released first (prefers
 * the curated releaseDate, falls back to HF createdAt). ``size`` and ``ram``
 * sort largest first using the same metadata that powers the row labels.
 * ``likes`` = HF stars/hearts desc. ``downloads`` = HF downloads desc.
 * Variants without the relevant metadata sort to the bottom. */
export type DiscoverSort = "release" | "size" | "ram" | "likes" | "downloads";
