/**
 * Chat catalog + local library → Models page entries (FU-129, R1).
 *
 * The old Discover tab listed catalog variants; the old My Models tab
 * listed the disk scan, joined to the catalog by a fuzzy name match
 * (``libraryVariantMatchScore``, threshold 12). One page needs one join
 * rule, so a model is never listed twice and never dropped:
 *
 *   1. Every catalog variant is a row ("catalog" row).
 *   2. A local model (chat library item) is merged into a catalog row when
 *      the two are each other's best match (the item's best variant is
 *      that variant, and the variant's best item is that item) AND the
 *      identity is confirmed: the backend already says the variant is on
 *      disk (``availableLocally``, its own name check), or the item IS
 *      that repo (a Hugging Face cache folder ``models--org--name``, an
 *      item named ``org/name``, or a folder whose path ends in
 *      ``/org/name``). The merged row shows the catalog variant with the
 *      item's files (path, size on disk, format and quant as found,
 *      broken state, Reveal, Delete, favourite).
 *   3. Every other local model is its own "library" row. It keeps its best
 *      catalog match only for display (capabilities, context, note), as
 *      the old My Models tab did; its status comes from its own files.
 *   4. An active download whose repo no row shows yet (a Hugging Face
 *      search result, Run from Hugging Face) is a "download" row with a
 *      ``download://<repo>`` path until the files reach the library scan.
 *
 * Installed state, in this order: the library scan when the row has local
 * files (a broken item is not installed), else the catalog's
 * ``availableLocally``; an active download wins over both (shared status
 * ladder, ``resolveModelStatus``). Each download is shown on exactly one
 * row: the catalog row of its repo, else the library row that is that
 * repo, else a download row.
 */

import type { DownloadStatus } from "../../../api";
import type { LibraryItem, ModelFamily, ModelVariant, SystemStats } from "../../../types";
import {
  chatVariantPlatformGate,
  discoverSearchTokens,
  estimateLibraryItemCompressedGb,
  estimateLibraryItemResidentGb,
  formatReleaseLabel,
  inferHfRepoFromLocalPath,
  isVariantCompatibleWithHost,
  libraryItemBackend,
  libraryItemFormat,
  libraryItemQuantization,
  libraryItemSourceKind,
  libraryVariantMatchScore,
  parseContextK,
  sizeLabel,
} from "../../../utils";
import { candidateKeys } from "../../runtimeSupport";
import { isTensorfoldRepo } from "../../tensorfoldSupport";
import { readBestFitVariantId, readVariantFit } from "../fit";
import type { ModelEntry } from "../modelEntry";

/** Same threshold as ``findLibraryItemForVariant`` / ``findCatalogVariantForLibraryItem``. */
export const CHAT_MATCH_THRESHOLD = 12;

export type ChatRowKind = "catalog" | "library" | "download";

export interface ChatRowSource {
  kind: ChatRowKind;
  /** Catalog row: its variant. Library row: the closest catalog match,
   * for display only. Download row: ``null``. */
  variant: ModelVariant | null;
  family: ModelFamily | null;
  /** Local files (merged or library row); a placeholder with a
   * ``download://<repo>`` path for a download row. */
  item: LibraryItem | null;
  displayFormat: string;
  displayQuantization: string | null;
  displayBackend: string;
  sourceKind: string | null;
  paramsB: number | null;
  contextWindow: string | null;
  ramGb: number | null;
  compressedGb: number | null;
  /** Epoch ms of the files' last change, for the "Recently modified" sort. */
  modifiedAt: number | null;
  /** The download shown on this row (live, or inferred from a partial
   * Hugging Face cache entry). */
  download: DownloadStatus | null;
  /** Persisted favourite id (the old My Models rule). */
  favoriteRef: string | null;
  /** Key for the launch modal: ``library:<path>`` or ``catalog:<id>``. */
  launchKey: string | null;
  /** The catalog variant needs converting before it runs (MLX). */
  canConvert: boolean;
  /** Normalised search tokens of the family + variant (catalog rows). */
  catalogTokens: Set<string> | null;
  /** Lower-cased text the old My Models search looked at (rows with files). */
  libraryHaystack: string | null;
}

export type ChatModelEntry = ModelEntry<ChatRowSource>;

export type HostLike = Pick<SystemStats, "platform" | "arch" | "gpuVramTotalGb"> | null | undefined;

export interface ChatEntryContext {
  host?: HostLike;
  /** ``workspace.runtime.loadedModel``. */
  loadedModel?: { ref?: string | null; path?: string | null; canonicalRepo?: string | null } | null;
  /** For download rows; defaults to ``Date.now()``. */
  now?: number;
  /** ``prepareChatJoin(families, library)`` computed from the same arrays,
   * so a page can memoise the fuzzy match apart from download polling. */
  join?: ChatJoin;
}

/** Sort ids of the chat Models page; ``useModels`` keeps the choice. */
export type ChatModelsSort =
  | "release"
  | "name"
  | "format"
  | "backend"
  | "size"
  | "ram"
  | "compressed"
  | "context"
  | "modified"
  | "status";

export const CHAT_MODELS_DEFAULT_SORT: ChatModelsSort = "release";

// ── Identity ──────────────────────────────────────────────────────────

function normalisePath(path: string): string {
  return path.replace(/\\/g, "/").toLowerCase();
}

/** The Hugging Face repo the item's own files belong to, when its name or
 * path says so: an HF cache folder or an item named ``org/name``. */
export function libraryItemRepo(item: LibraryItem): string | null {
  const fromCache = inferHfRepoFromLocalPath(item.path);
  if (fromCache) return fromCache;
  const name = item.name.trim();
  if (/^[^/\s]+\/[^/\s]+$/.test(name)) return name;
  return null;
}

/** True when the item's files are those of ``repo``: the HF cache folder
 * or name is the repo, or the folder path ends in ``/org/name``. */
export function libraryItemIsRepo(item: LibraryItem, repo: string): boolean {
  const target = repo.trim().toLowerCase();
  if (!target) return false;
  if (libraryItemRepo(item)?.toLowerCase() === target) return true;
  return normalisePath(item.path).replace(/\/+$/, "").endsWith(`/${target}`);
}

/** Persisted favourite id of a local model. Unchanged from the old My
 * Models tab so stored favourites keep matching: the inferred HF repo,
 * else the matched catalog repo, else an ``org/name`` item name, else the
 * path. */
export function chatFavoriteRef(item: LibraryItem, matchedVariant: ModelVariant | null): string | null {
  const repo =
    inferHfRepoFromLocalPath(item.path) ??
    matchedVariant?.repo ??
    (item.name.includes("/") ? item.name : null);
  if (repo && repo.trim()) return repo.trim();
  if (item.path) return item.path;
  return null;
}

/** A broken Hugging Face cache entry whose files are still partial is a
 * paused download, with progress from the bytes on disk (old My Models). */
export function inferredPartialDownload(
  item: LibraryItem,
  repo: string | null,
  totalGbHint: number | null | undefined,
): DownloadStatus | null {
  if (!repo || !item.broken) return null;
  const reason = (item.brokenReason ?? "").toLowerCase();
  const isPartialHfCache =
    libraryItemSourceKind(item).toLowerCase() === "hf cache" &&
    (reason.includes("partial blob") || reason.includes("incomplete"));
  if (!isPartialHfCache) return null;
  const downloadedGb = Math.max(0, item.sizeGb ?? 0);
  const totalGb = totalGbHint && totalGbHint > 0 ? totalGbHint : null;
  const progress = totalGb ? Math.max(0, Math.min(1, downloadedGb / totalGb)) : 0;
  return { repo, state: "cancelled", progress, downloadedGb, totalGb, error: null };
}

function isTransfer(download: DownloadStatus | null | undefined): download is DownloadStatus {
  return download?.state === "downloading" || download?.state === "cancelled" || download?.state === "failed";
}

function parseModified(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value.includes("T") ? value : value.replace(" ", "T"));
  return Number.isNaN(parsed) ? null : parsed;
}

function positiveOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

// ── Search ────────────────────────────────────────────────────────────

function catalogSearchTokens(family: ModelFamily | null, variant: ModelVariant): Set<string> {
  const fragments: Array<string | null | undefined> = [
    family?.name,
    family?.provider,
    family?.headline,
    family?.summary,
    family?.description,
    ...(family?.capabilities ?? []),
    ...(family?.readme ?? []),
    variant.name,
    variant.repo,
    variant.format,
    variant.quantization,
    variant.note,
    variant.contextWindow,
    ...variant.capabilities,
  ];
  return new Set(discoverSearchTokens(fragments.filter(Boolean).join(" ")));
}

function libraryHaystack(
  item: LibraryItem,
  displayFormat: string,
  displayQuantization: string | null,
  displayBackend: string,
  sourceKind: string | null,
): string {
  return `${item.name} ${item.path} ${displayFormat} ${displayQuantization ?? ""} ${displayBackend} ${sourceKind ?? ""} ${item.directoryLabel ?? ""}`.toLowerCase();
}

/** One search box for the whole page (R8):
 *  - a catalog row matches when every word of the query appears in its
 *    family's text or its own variant's text (the old Discover token
 *    rule, now per variant instead of per family);
 *  - a row with local files also matches on the old My Models text
 *    (name, path, format, quant, backend, source, directory label);
 *  - the same text drives the Hugging Face search, shown under the table. */
export function chatEntryMatchesQuery(entry: ChatModelEntry, query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return true;
  const { catalogTokens, libraryHaystack: haystack } = entry.source;
  if (catalogTokens) {
    const tokens = discoverSearchTokens(trimmed);
    if (tokens.length > 0 && tokens.every((token) => catalogTokens.has(token))) return true;
  }
  if (haystack && haystack.includes(trimmed.toLowerCase())) return true;
  return false;
}

// ── Acceleration filter (R7) ──────────────────────────────────────────

export type ChatAccelId = "dflash" | "mtplx" | "tensorfold" | "turboquant";

export const CHAT_ACCEL_FILTERS: ReadonlyArray<{ id: ChatAccelId; label: string; color: string; appleOnly: boolean }> = [
  { id: "dflash", label: "DFlash", color: "#a78bfa", appleOnly: false },
  { id: "mtplx", label: "MTPLX", color: "#f472b6", appleOnly: true },
  { id: "tensorfold", label: "TensorFold", color: "#34d399", appleOnly: true },
  { id: "turboquant", label: "TurboQuant", color: "#60a5fa", appleOnly: false },
];

export interface ChatAccelCompat {
  dflashModels: string[];
  mtplxModels: string[];
  /** Exact repo ids TensorFold serves. */
  tensorfoldModels: string[];
  /** ``llama-server-turbo`` is installed. */
  turboInstalled: boolean;
  /** TurboQuant is available for MLX. */
  turboquantMlxAvailable: boolean;
  /** MTPLX and TensorFold run on Apple Silicon only; their chips hide elsewhere. */
  appleSilicon?: boolean;
}

function keySetOf(refs: string[]): Set<string> {
  const keys = new Set<string>();
  for (const ref of refs) for (const key of candidateKeys([ref])) keys.add(key);
  return keys;
}

function overlaps(values: Array<string | null | undefined>, keys: Set<string>): boolean {
  if (keys.size === 0) return false;
  return candidateKeys(values).some((key) => keys.has(key));
}

/** Predicate for the acceleration chips, each rule on its own data:
 *  - a row with local files uses the old My Models rule (what can run
 *    now: DFlash on non-GGUF with a registered drafter, TurboQuant only
 *    when its binary / MLX support is installed, MTPLX on MLX by the
 *    folder's own name, TensorFold by exact repo);
 *  - a catalog row without files uses the old Discover rule per variant
 *    (DFlash / MTPLX by repo key, TensorFold by exact repo, TurboQuant for
 *    GGUF builds). */
export function makeChatAccelPredicate(compat: ChatAccelCompat): (entry: ChatModelEntry, accel: string) => boolean {
  const dflashKeys = keySetOf(compat.dflashModels);
  const mtplxKeys = keySetOf(compat.mtplxModels);
  return (entry, accel) => {
    const source = entry.source;
    const item = source.kind === "download" ? null : source.item;
    if (item) {
      const backend = source.displayBackend.toLowerCase();
      const isGguf = backend.includes("llama") || source.displayFormat === "GGUF";
      const isMlx = backend.includes("mlx") || source.displayFormat === "MLX";
      switch (accel) {
        case "dflash":
          if (isGguf) return false;
          return overlaps([item.name, source.variant?.repo], dflashKeys);
        case "turboquant":
          return (isGguf && compat.turboInstalled) || (isMlx && compat.turboquantMlxAvailable);
        case "mtplx":
          // Baked-in MTP heads: match the folder's own name only, never the
          // fuzzy catalog match (it would attach unrelated repacks).
          if (!isMlx) return false;
          return overlaps([item.name], mtplxKeys);
        case "tensorfold":
          if (isGguf) return false;
          return isTensorfoldRepo(compat.tensorfoldModels, [source.variant?.repo, item.name, inferHfRepoFromLocalPath(item.path)]);
        default:
          return true;
      }
    }
    const variant = source.variant;
    if (!variant) return false;
    switch (accel) {
      case "dflash":
        return overlaps([variant.repo], dflashKeys);
      case "mtplx":
        return overlaps([variant.repo], mtplxKeys);
      case "tensorfold":
        return isTensorfoldRepo(compat.tensorfoldModels, [variant.repo]);
      case "turboquant":
        return variant.format === "GGUF";
      default:
        return true;
    }
  };
}

// ── Join ──────────────────────────────────────────────────────────────

export interface ChatJoin {
  /** Best catalog variant of each library item (score >= threshold). */
  bestVariantOfItem: Map<LibraryItem, ModelVariant>;
  /** Best library item of each catalog variant (score >= threshold). */
  bestItemOfVariant: Map<ModelVariant, LibraryItem>;
}

/** Scores every (item, variant) pair once and keeps both best matches,
 * with the same first-wins tie rule as ``findLibraryItemForVariant`` and
 * ``findCatalogVariantForLibraryItem``. */
export function matchLibraryToCatalog(library: LibraryItem[], variants: ModelVariant[]): ChatJoin {
  const bestVariantOfItem = new Map<LibraryItem, ModelVariant>();
  const bestItemOfVariant = new Map<ModelVariant, LibraryItem>();
  const itemBest = new Map<LibraryItem, number>();
  const variantBest = new Map<ModelVariant, number>();
  for (const item of library) {
    for (const variant of variants) {
      const score = libraryVariantMatchScore(item, variant);
      const currentItemBest = itemBest.get(item);
      if (currentItemBest === undefined || score > currentItemBest) {
        itemBest.set(item, score);
        bestVariantOfItem.set(item, variant);
      }
      const currentVariantBest = variantBest.get(variant);
      if (currentVariantBest === undefined || score > currentVariantBest) {
        variantBest.set(variant, score);
        bestItemOfVariant.set(variant, item);
      }
    }
  }
  for (const [item, score] of itemBest) if (score < CHAT_MATCH_THRESHOLD) bestVariantOfItem.delete(item);
  for (const [variant, score] of variantBest) if (score < CHAT_MATCH_THRESHOLD) bestItemOfVariant.delete(variant);
  return { bestVariantOfItem, bestItemOfVariant };
}

function flattenCatalog(families: ModelFamily[]): { variants: ModelVariant[]; familyOf: Map<ModelVariant, ModelFamily> } {
  const familyOf = new Map<ModelVariant, ModelFamily>();
  const variants: ModelVariant[] = [];
  for (const family of families) {
    for (const variant of family.variants) {
      familyOf.set(variant, family);
      variants.push(variant);
    }
  }
  return { variants, familyOf };
}

function isRealItem(item: LibraryItem): boolean {
  return !item.path.startsWith("download://");
}

/** The fuzzy match of every local model against the catalog (the costly
 * part of ``buildChatEntries``). */
export function prepareChatJoin(families: ModelFamily[], library: LibraryItem[]): ChatJoin {
  return matchLibraryToCatalog(library.filter(isRealItem), flattenCatalog(families).variants);
}

/** Rule 2: does this item merge into this variant's row? */
export function shouldMergeLibraryItem(
  item: LibraryItem,
  variant: ModelVariant,
  join: ChatJoin,
): boolean {
  if (join.bestVariantOfItem.get(item) !== variant) return false;
  if (join.bestItemOfVariant.get(variant) !== item) return false;
  return Boolean(variant.availableLocally) || libraryItemIsRepo(item, variant.repo);
}

function isLoaded(
  loaded: ChatEntryContext["loadedModel"],
  item: LibraryItem | null,
  variant: ModelVariant | null,
): boolean {
  if (!loaded) return false;
  if (item) return Boolean(loaded.path && loaded.path === item.path);
  if (!variant) return false;
  const refs = [loaded.ref, loaded.canonicalRepo].filter((ref): ref is string => Boolean(ref)).map((ref) => ref.toLowerCase());
  return refs.includes(variant.repo.toLowerCase()) || refs.includes(variant.id.toLowerCase());
}

function capabilitiesOf(variant: ModelVariant | null, family: ModelFamily | null): string[] {
  if (!variant) return [];
  if (variant.capabilities.length > 0) return [...variant.capabilities];
  return [...(family?.capabilities ?? [])];
}

function providerFromRepo(repo: string | null): string {
  if (!repo || !repo.includes("/")) return "";
  return repo.split("/")[0] ?? "";
}

/** Build every row of the chat Models page (rules at the top of the file). */
export function buildChatEntries(
  families: ModelFamily[],
  library: LibraryItem[],
  activeDownloads: Record<string, DownloadStatus>,
  ctx: ChatEntryContext = {},
): ChatModelEntry[] {
  const { variants, familyOf } = flattenCatalog(families);
  const realItems = library.filter(isRealItem);
  const join = ctx.join ?? matchLibraryToCatalog(realItems, variants);

  const mergedItemOf = new Map<ModelVariant, LibraryItem>();
  const mergedItems = new Set<LibraryItem>();
  for (const item of realItems) {
    const variant = join.bestVariantOfItem.get(item);
    if (variant && shouldMergeLibraryItem(item, variant, join)) {
      mergedItemOf.set(variant, item);
      mergedItems.add(item);
    }
  }

  const catalogRepos = new Set(variants.map((variant) => variant.repo.toLowerCase()));
  const shownRepos = new Set<string>(catalogRepos);
  const downloadFor = (repo: string | null | undefined): DownloadStatus | null =>
    repo ? activeDownloads[repo] ?? null : null;

  const entries: ChatModelEntry[] = [];

  // Rules 1 + 2: catalog rows, with their merged local files.
  for (const variant of variants) {
    const family = familyOf.get(variant) ?? null;
    const item = mergedItemOf.get(variant) ?? null;
    const ownRepo = item ? libraryItemRepo(item) : null;
    if (ownRepo) shownRepos.add(ownRepo.toLowerCase());
    const download =
      downloadFor(variant.repo) ??
      (ownRepo && ownRepo.toLowerCase() !== variant.repo.toLowerCase() ? downloadFor(ownRepo) : null) ??
      (item ? inferredPartialDownload(item, ownRepo ?? variant.repo, variant.sizeGb) : null);
    const displayFormat = item ? libraryItemFormat(item, variant) : variant.format;
    const displayQuantization = item ? libraryItemQuantization(item, variant) : variant.quantization;
    const displayBackend = item ? libraryItemBackend(item, variant) : variant.backend;
    const sourceKind = item ? libraryItemSourceKind(item) : null;
    const ramGb = item ? estimateLibraryItemResidentGb(item, variant) : positiveOrNull(variant.estimatedMemoryGb);
    const compressedGb = item
      ? estimateLibraryItemCompressedGb(item, variant)
      : positiveOrNull(variant.estimatedCompressedMemoryGb);
    const sizeGb = item ? item.sizeGb : variant.sizeGb;
    const fit = readVariantFit(variant);
    entries.push({
      id: `catalog:${variant.id}`,
      area: "chat",
      name: variant.name,
      provider: family?.provider ?? providerFromRepo(variant.repo),
      familyId: family?.id ?? variant.familyId,
      familyName: family?.name ?? null,
      note: variant.note || null,
      repo: variant.repo,
      link: variant.link || null,
      downloadKey: variant.repo,
      deleteTarget: item?.path ?? variant.repo,
      availableLocally: item ? !item.broken : Boolean(variant.availableLocally),
      hasLocalData: Boolean(item),
      broken: item?.broken ? { reason: item.brokenReason ?? "" } : null,
      loaded: isLoaded(ctx.loadedModel, item, variant),
      localPath: item?.path ?? null,
      hostCompatible: isVariantCompatibleWithHost(chatVariantPlatformGate({ backend: displayBackend }), ctx.host),
      capabilities: capabilitiesOf(variant, family),
      sizeGb: positiveOrNull(sizeGb),
      sizeLabel: sizeLabel(sizeGb),
      memory: { label: ramGb ? `~${ramGb.toFixed(1)} GB` : null, peakGb: ramGb },
      releaseSortKey: variant.releaseDate ?? "",
      releaseLabel: formatReleaseLabel(variant.releaseLabel, variant.releaseDate) || null,
      fit,
      bestFit: Boolean(family && readBestFitVariantId(family) === variant.id),
      source: {
        kind: "catalog",
        variant,
        family,
        item,
        displayFormat,
        displayQuantization,
        displayBackend,
        sourceKind,
        paramsB: positiveOrNull(variant.paramsB),
        contextWindow: variant.contextWindow || null,
        ramGb,
        compressedGb,
        modifiedAt: item ? parseModified(item.lastModified) : null,
        download,
        favoriteRef: item ? chatFavoriteRef(item, variant) : null,
        launchKey: item ? `library:${item.path}` : `catalog:${variant.id}`,
        canConvert: variant.launchMode === "convert",
        catalogTokens: catalogSearchTokens(family, variant),
        libraryHaystack: item ? libraryHaystack(item, displayFormat, displayQuantization, displayBackend, sourceKind) : null,
      },
    });
  }

  // Rule 3: local models no catalog row took.
  for (const item of realItems) {
    if (mergedItems.has(item)) continue;
    const matched = join.bestVariantOfItem.get(item) ?? null;
    const family = matched ? familyOf.get(matched) ?? null : null;
    const ownRepo = libraryItemRepo(item);
    // A download of a catalog repo shows on that catalog row only.
    const ownsDownload = Boolean(ownRepo && !catalogRepos.has(ownRepo.toLowerCase()));
    if (ownRepo && ownsDownload) shownRepos.add(ownRepo.toLowerCase());
    const download =
      (ownsDownload ? downloadFor(ownRepo) : null) ??
      inferredPartialDownload(item, ownRepo, matched?.sizeGb);
    const displayFormat = libraryItemFormat(item, matched);
    const displayQuantization = libraryItemQuantization(item, matched);
    const displayBackend = libraryItemBackend(item, matched);
    const sourceKind = libraryItemSourceKind(item);
    const ramGb = estimateLibraryItemResidentGb(item, matched);
    entries.push({
      id: `library:${item.path}`,
      area: "chat",
      name: item.name,
      provider: providerFromRepo(ownRepo),
      familyId: family?.id ?? null,
      familyName: null,
      note: matched?.note || null,
      repo: ownRepo ?? "",
      link: ownRepo ? `https://huggingface.co/${ownRepo}` : null,
      downloadKey: ownRepo ?? "",
      deleteTarget: item.path,
      availableLocally: !item.broken,
      hasLocalData: true,
      broken: item.broken ? { reason: item.brokenReason ?? "" } : null,
      loaded: isLoaded(ctx.loadedModel, item, null),
      localPath: item.path,
      hostCompatible: isVariantCompatibleWithHost(chatVariantPlatformGate({ backend: displayBackend }), ctx.host),
      capabilities: capabilitiesOf(matched, family),
      sizeGb: positiveOrNull(item.sizeGb),
      sizeLabel: sizeLabel(item.sizeGb),
      memory: { label: ramGb ? `~${ramGb.toFixed(1)} GB` : null, peakGb: ramGb },
      releaseSortKey: matched?.releaseDate ?? "",
      releaseLabel: matched ? formatReleaseLabel(matched.releaseLabel, matched.releaseDate) || null : null,
      fit: null,
      bestFit: false,
      source: {
        kind: "library",
        variant: matched,
        family,
        item,
        displayFormat,
        displayQuantization,
        displayBackend,
        sourceKind,
        paramsB: null,
        contextWindow: matched?.contextWindow || null,
        ramGb,
        compressedGb: estimateLibraryItemCompressedGb(item, matched),
        modifiedAt: parseModified(item.lastModified),
        download,
        favoriteRef: chatFavoriteRef(item, matched),
        launchKey: `library:${item.path}`,
        canConvert: false,
        catalogTokens: null,
        libraryHaystack: libraryHaystack(item, displayFormat, displayQuantization, displayBackend, sourceKind),
      },
    });
  }

  // Rule 4: downloads no row shows yet.
  const now = ctx.now ?? Date.now();
  for (const download of Object.values(activeDownloads)) {
    if (!isTransfer(download)) continue;
    if (shownRepos.has(download.repo.toLowerCase())) continue;
    shownRepos.add(download.repo.toLowerCase());
    const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
    const item: LibraryItem = {
      name: download.repo,
      path: `download://${download.repo}`,
      format: "",
      sourceKind: "HF cache",
      quantization: null,
      backend: null,
      modelType: "text",
      sizeGb: downloadedGb,
      lastModified: new Date(now).toISOString().replace("T", " ").slice(0, 16),
      actions: [],
      broken: false,
      brokenReason: null,
    };
    entries.push({
      id: `library:${item.path}`,
      area: "chat",
      name: download.repo,
      provider: providerFromRepo(download.repo),
      familyId: null,
      familyName: null,
      note: null,
      repo: download.repo,
      link: `https://huggingface.co/${download.repo}`,
      downloadKey: download.repo,
      deleteTarget: download.repo,
      availableLocally: false,
      hasLocalData: false,
      broken: null,
      loaded: false,
      localPath: null,
      hostCompatible: true,
      capabilities: [],
      sizeGb: positiveOrNull(downloadedGb),
      sizeLabel: sizeLabel(downloadedGb),
      memory: null,
      releaseSortKey: "",
      releaseLabel: null,
      fit: null,
      bestFit: false,
      source: {
        kind: "download",
        variant: null,
        family: null,
        item,
        displayFormat: "",
        displayQuantization: null,
        displayBackend: "",
        sourceKind: null,
        paramsB: null,
        contextWindow: null,
        ramGb: null,
        compressedGb: null,
        modifiedAt: now,
        download,
        favoriteRef: null,
        launchKey: null,
        canConvert: false,
        catalogTokens: null,
        libraryHaystack: `${download.repo} ${item.path}`.toLowerCase(),
      },
    });
  }

  return entries;
}

/** Context length in thousands of tokens, for the Context sort. */
export function chatContextSortKey(entry: ChatModelEntry): number | null {
  const window = entry.source.contextWindow;
  if (!window) return null;
  const value = parseContextK(window);
  return value > 0 ? value : null;
}

/** The chat library item a Hugging Face search result is, for the launch
 * modal preset (R3): the same repo by HF cache / name / path, else the
 * backend's own rule (the repo id inside the item's name or path). */
export function findLibraryItemForRepo(library: LibraryItem[], repo: string): LibraryItem | null {
  const exact = library.find((item) => isRealItem(item) && libraryItemIsRepo(item, repo));
  if (exact) return exact;
  const target = repo.trim().toLowerCase();
  if (!target) return null;
  return (
    library.find(
      (item) =>
        isRealItem(item) &&
        (item.name.toLowerCase().includes(target) || normalisePath(item.path).includes(target)),
    ) ?? null
  );
}
