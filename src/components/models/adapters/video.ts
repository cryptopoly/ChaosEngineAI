/**
 * Video catalog → Models page entries.
 *
 * Mirrors ``adapters/image.ts`` with the video differences (R12, R13,
 * R25):
 *   - downloads are keyed by variant id (``handleVideoDownload`` writes the
 *     status under the variant id and under the repo it fetched), so
 *     ``downloadKey`` is the variant id, never the repo;
 *   - Pause goes to the repo of the running download (``download.repo``),
 *     Cancel / Delete to ``videoDeleteRepoForVariant`` (which may be a
 *     shared GGUF or text-encoder repo), Download to ``(repo, variantId)``;
 *   - size sorts by the GGUF file first (FU-054), so three quants of one
 *     repo sort by their own file;
 *   - LongLive is installed by its own installer, so its status comes
 *     from the LongLive runtime probe, not from the Hugging Face cache.
 */

import type { DownloadStatus } from "../../../api";
import type { VideoModelFamily, VideoModelVariant } from "../../../types";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  formatReleaseLabel,
  imageOrVideoVariantPlatformGate,
  isVariantCompatibleWithHost,
  videoDeleteRepoForVariant,
  videoDeleteRepoKind,
  videoDiscoverFamilyMatchesQuery,
  videoDiscoverMemoryEstimate,
  videoDiscoverVariantMatchesQuery,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
  type VideoDeleteRepoKind,
} from "../../../utils";
import { acceleratorsForHost, type AcceleratorHost } from "../AcceleratorPills";
import { readBestFitVariantId, readVariantFit } from "../fit";
import type { ModelEntry, ModelSharedDownload } from "../modelEntry";

export interface VideoModelEntry extends ModelEntry<VideoModelVariant> {
  /** A LongLive row: installed by its own installer, status from the
   * LongLive runtime probe, no download / delete buttons. */
  longLive: boolean;
}

export interface VideoEntryContext {
  family?: VideoModelFamily | null;
  host?: AcceleratorHost;
  /** Id of the variant the video runtime has in memory. */
  loadedVariantId?: string | null;
  /** LongLive runtime probe says it can generate. */
  longLiveReady?: boolean;
  /** Variant id → display name, for shared-download siblings. */
  nameById?: Map<string, string>;
}

/**
 * LongLive ships through a dedicated installer (isolated venv, GitHub
 * clone, weights from ``Efficient-Large-Model/LongLive-1.3B``), not
 * through ``snapshot_download``. The catalog repo ``NVlabs/LongLive-1.3B``
 * is only a routing key; it does not resolve on Hugging Face.
 */
export function isLongLiveRepo(repo: string | null | undefined): boolean {
  return repo?.startsWith("NVlabs/LongLive") ?? false;
}

/** Curated variants (each tagged with its family name) followed by the
 * tracked rows whose repo the curated catalog does not list (R26). */
export function collectVideoVariants(catalog: VideoModelFamily[], latest: VideoModelVariant[]): VideoModelVariant[] {
  const curated = catalog.flatMap((family) =>
    family.variants.map((variant) => ({ ...variant, familyName: variant.familyName ?? family.name })),
  );
  const curatedRepos = new Set(curated.map((variant) => variant.repo.toLowerCase()));
  const seenTracked = new Set<string>();
  const tracked = latest.filter((variant) => {
    const repo = variant.repo.toLowerCase();
    if (curatedRepos.has(repo) || seenTracked.has(repo)) return false;
    seenTracked.add(repo);
    return true;
  });
  return [...curated, ...tracked];
}

function firstPositive(values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

/** Size used by the Size sort (R13): the pinned GGUF file first, then what
 * is on disk, the weights, the repo, the catalog estimate. */
export function videoSizeSortKey(variant: VideoModelVariant): number | null {
  return firstPositive([variant.ggufFileGb, variant.onDiskGb, variant.coreWeightsGb, variant.repoSizeGb, variant.sizeGb]);
}

/** Fields an older or newer backend may send that the video type does not
 * declare yet; read defensively. */
function optionalFlags(variant: VideoModelVariant): { trackedOnly: boolean; gated: boolean | null } {
  const raw = variant as VideoModelVariant & { trackedOnly?: unknown; gated?: unknown };
  return {
    trackedOnly: raw.trackedOnly === true,
    gated: typeof raw.gated === "boolean" ? raw.gated : null,
  };
}

export function videoVariantToEntry(variant: VideoModelVariant, ctx: VideoEntryContext = {}): VideoModelEntry {
  const family = ctx.family ?? null;
  const longLive = isLongLiveRepo(variant.repo);
  const memory = videoDiscoverMemoryEstimate(variant);
  const primarySize = videoPrimarySizeLabel(variant);
  const secondarySize = videoSecondarySizeLabel(variant);
  const siblings = variant.sharedRepoSiblings ?? 0;
  const siblingIds = variant.sharedRepoSiblingIds ?? [];
  const { trackedOnly, gated } = optionalFlags(variant);
  return {
    id: variant.id,
    area: "video",
    name: variant.name,
    provider: variant.provider,
    familyId: variant.familyId,
    familyName: variant.familyName ?? family?.name ?? null,
    note: variant.note || null,
    repo: variant.repo,
    link: variant.link,
    // R12: the active-download map is keyed by variant id.
    downloadKey: variant.id,
    deleteTarget: videoDeleteRepoForVariant(variant),
    // LongLive's files are not in the Hugging Face cache: only the runtime
    // probe decides whether it is installed.
    availableLocally: longLive ? false : Boolean(variant.availableLocally),
    hasLocalData: longLive ? false : Boolean(variant.hasLocalData),
    installedByRuntime: longLive ? Boolean(ctx.longLiveReady) : null,
    loaded: Boolean(ctx.loadedVariantId && ctx.loadedVariantId === variant.id),
    localPath: variant.localPath ?? null,
    hostCompatible: isVariantCompatibleWithHost(imageOrVideoVariantPlatformGate(variant), ctx.host),
    trackedOnly,
    gated,
    license: variant.license ?? null,
    runtime: variant.runtime,
    tasks: [...variant.taskSupport],
    styleTags: [...variant.styleTags],
    accelerators: acceleratorsForHost(variant, ctx.host),
    sizeGb: videoSizeSortKey(variant),
    sizeLabel: compactModelSizeLabel(primarySize),
    sizeTitle: [primarySize, secondarySize].filter(Boolean).join(" / ") || null,
    memory: memory ? { label: memory.label, title: memory.title, peakGb: memory.estimatedPeakGb } : null,
    releaseSortKey: variant.releaseDate ?? variant.createdAt ?? variant.lastModified ?? "",
    releaseLabel: compactReleaseLabel(formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt)),
    likes: variant.likes ?? null,
    downloads: variant.downloads ?? null,
    likesLabel: variant.likesLabel ?? null,
    downloadsLabel: variant.downloadsLabel ?? null,
    fit: readVariantFit(variant),
    bestFit: Boolean(family && readBestFitVariantId(family) === variant.id),
    shared:
      siblings > 0
        ? {
            count: siblings,
            key: variant.sharedRepoKey ?? variant.repo,
            siblingNames: siblingIds.map((id) => ctx.nameById?.get(id) ?? id),
          }
        : null,
    source: variant,
    longLive,
  };
}

export function buildVideoEntries(
  catalog: VideoModelFamily[],
  latest: VideoModelVariant[],
  ctx: Omit<VideoEntryContext, "family" | "nameById"> = {},
): VideoModelEntry[] {
  const familyByVariantId = new Map<string, VideoModelFamily>();
  const nameById = new Map<string, string>();
  for (const family of catalog) {
    for (const variant of family.variants) {
      familyByVariantId.set(variant.id, family);
      nameById.set(variant.id, variant.name);
    }
  }
  for (const variant of latest) {
    if (!nameById.has(variant.id)) nameById.set(variant.id, variant.name);
  }
  return collectVideoVariants(catalog, latest).map((variant) =>
    videoVariantToEntry(variant, { ...ctx, family: familyByVariantId.get(variant.id) ?? null, nameById }),
  );
}

/** Search used by the old Video Discover tab: a row matches when its own
 * text (name, family, provider, repo, engine, size, tasks, tags, note) or
 * its family's text contains the query. */
export function videoEntryMatchesQuery(
  entry: VideoModelEntry,
  query: string,
  familyById: Map<string, VideoModelFamily>,
): boolean {
  if (!query) return true;
  if (videoDiscoverVariantMatchesQuery(entry.source, query)) return true;
  const family = entry.familyId ? familyById.get(entry.familyId) : undefined;
  return Boolean(
    family && family.variants.some((variant) => variant.id === entry.id) && videoDiscoverFamilyMatchesQuery(family, query),
  );
}

/** A LongLive row the page can show on this host (R25): only then is the
 * LongLive runtime worth probing. */
export function hasVisibleLongLiveRow(entries: VideoModelEntry[]): boolean {
  return entries.some((entry) => entry.longLive && entry.hostCompatible);
}

// ── Delete target and confirm ─────────────────────────────────────────

/** Repos a variant needs on disk: its pipeline repo, its pinned GGUF repo,
 * its text-encoder repo and whatever the backend found locally for it. */
export function videoRepoDependencies(variant: VideoModelVariant): string[] {
  const repos = [variant.repo, variant.ggufRepo, variant.textEncoderRepo, ...(variant.localDataRepos ?? [])];
  return Array.from(new Set(repos.filter((repo): repo is string => Boolean(repo && repo.trim()))));
}

/** Repo → the variants that need it, for naming everything a delete
 * takes away. */
export function buildVideoRepoUsers(entries: VideoModelEntry[]): Map<string, VideoModelEntry[]> {
  const users = new Map<string, VideoModelEntry[]>();
  for (const entry of entries) {
    for (const repo of videoRepoDependencies(entry.source)) {
      const list = users.get(repo);
      if (list) list.push(entry);
      else users.set(repo, [entry]);
    }
  }
  return users;
}

/**
 * What deleting ``repo`` for ``entry`` takes from the other rows.
 *
 * The backend's ``sharedRepoSiblings`` (FU-054, shown as the "shares
 * storage" badge) groups rows by their GGUF repo or main local repo. A
 * video delete can go wider: deleting the base pipeline repo of a GGUF
 * quant also breaks the base row and the other quants, and deleting a
 * shared text encoder breaks every row that loads it. The confirm names
 * all of them.
 */
export function videoSharedForDelete(
  entry: VideoModelEntry,
  repo: string,
  repoUsers: Map<string, VideoModelEntry[]>,
): ModelSharedDownload | null {
  const names = new Map<string, string>();
  for (const user of repoUsers.get(repo) ?? []) {
    if (user.id !== entry.id) names.set(user.id, user.name);
  }
  if (entry.shared && entry.shared.key === repo) {
    const ids = entry.source.sharedRepoSiblingIds ?? [];
    ids.forEach((id, index) => {
      if (id !== entry.id && !names.has(id)) names.set(id, entry.shared?.siblingNames[index] ?? id);
    });
  }
  if (names.size === 0) return null;
  return { count: names.size, key: repo, siblingNames: Array.from(names.values()) };
}

/** The entry ``confirmAndDelete`` should describe for a delete of
 * ``repo``: that repo as the target, the snapshot path only when it is
 * that repo's, and every row the delete affects. */
export function videoDeleteConfirmEntry(
  entry: VideoModelEntry,
  repo: string,
  repoUsers: Map<string, VideoModelEntry[]>,
): Pick<VideoModelEntry, "name" | "repo" | "localPath" | "deleteTarget" | "shared"> {
  const snapshotRepo = entry.source.primaryLocalRepo ?? entry.source.repo;
  return {
    name: entry.name,
    repo: entry.repo,
    localPath: repo === snapshotRepo ? entry.localPath : null,
    deleteTarget: repo,
    shared: videoSharedForDelete(entry, repo, repoUsers),
  };
}

// ── Row handlers (R12) ────────────────────────────────────────────────

export interface VideoDownloadCallbacks {
  onVideoDownload: (repo: string, modelId?: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
}

export interface VideoRowActions {
  /** Repo Cancel / Delete act on. */
  deleteRepo: string;
  deleteKind: VideoDeleteRepoKind;
  /** Download, Resume and Retry. */
  download: () => void;
  pause: () => void;
  /** Cancel an in-flight download or delete a paused / failed one. */
  cancel: () => void;
}

/** The download handlers of one row, with the video keys (R12). */
export function videoRowActions(
  entry: VideoModelEntry,
  download: Pick<DownloadStatus, "repo"> | null | undefined,
  callbacks: VideoDownloadCallbacks,
): VideoRowActions {
  const variant = entry.source;
  const deleteRepo = videoDeleteRepoForVariant(variant, download ?? undefined);
  return {
    deleteRepo,
    deleteKind: videoDeleteRepoKind(variant, deleteRepo),
    download: () => callbacks.onVideoDownload(variant.repo, variant.id),
    pause: () => callbacks.onCancelVideoDownload(download?.repo ?? variant.repo),
    cancel: () => callbacks.onDeleteVideoDownload(deleteRepo),
  };
}
