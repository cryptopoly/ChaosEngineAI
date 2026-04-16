import { useEffect, useRef, useState } from "react";
import {
  downloadModel,
  cancelDownload,
  deleteModelDownload,
  getDownloadStatus,
  searchHubModels,
  listHubFiles,
} from "../api";
import type { DownloadStatus } from "../api";
import {
  defaultVariantForFamily,
  buildDownloadStatusMap,
  modelFamilyMatchesDiscoverQuery,
  pendingDownloadStatus,
  failedDownloadStatus,
} from "../utils";
import type {
  HubFileListResponse,
  HubModel,
  ModelFamily,
  WorkspaceData,
} from "../types";

export function useModels(
  backendOnline: boolean,
  activeChatId: string,
  curatedFamilies: ModelFamily[],
  setError: (msg: string | null) => void,
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>,
) {
  const [searchInput, setSearchInput] = useState("");
  const [searchResults, setSearchResults] = useState<ModelFamily[]>(curatedFamilies);
  const [hubResults, setHubResults] = useState<HubModel[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [expandedHubId, setExpandedHubId] = useState<string | null>(null);
  const [hubFileCache, setHubFileCache] = useState<Record<string, HubFileListResponse>>({});
  const [hubFileLoading, setHubFileLoading] = useState<Record<string, boolean>>({});
  const [hubFileError, setHubFileError] = useState<Record<string, string>>({});
  const [detailFamilyId, setDetailFamilyId] = useState<string | null>(null);
  const [selectedFamilyId, setSelectedFamilyId] = useState("");
  const [selectedVariantId, setSelectedVariantId] = useState("");
  const [expandedFamilyId, setExpandedFamilyId] = useState<string | null>(null);
  const [expandedVariantId, setExpandedVariantId] = useState<string | null>(null);
  const [activeDownloads, setActiveDownloads] = useState<Record<string, DownloadStatus>>({});
  const [discoverCapFilter, setDiscoverCapFilter] = useState<string | null>(null);
  const [discoverFormatFilter, setDiscoverFormatFilter] = useState<string | null>(null);

  // Keep curated families in sync when workspace refreshes (without
  // retriggering the search effect, which would cancel in-flight API calls).
  const curatedRef = useRef(curatedFamilies);
  curatedRef.current = curatedFamilies;

  useEffect(() => {
    const localSearch = searchInput.trim();
    if (!localSearch) {
      setSearchResults(curatedRef.current);
      setHubResults([]);
      setSearchError(null);
      return;
    }
    setSearchResults(curatedRef.current.filter((family) => modelFamilyMatchesDiscoverQuery(family, localSearch)));
  }, [curatedFamilies, searchInput]); // eslint-disable-line react-hooks/exhaustive-deps

  // Hub search effect — curated families are always filtered locally so the
  // Discover list stays responsive even if the backend search path is stale.
  useEffect(() => {
    const query = searchInput.trim();
    if (!query) {
      setHubResults([]);
      setSearchError(null);
      return;
    }
    let cancelled = false;
    const timeout = window.setTimeout(() => {
      void (async () => {
        if (!backendOnline) {
          if (!cancelled) {
            setHubResults([]);
            setSearchError("Backend search is offline. Showing local catalog matches only.");
          }
          return;
        }
        try {
          const hubModels = await searchHubModels(query);
          if (!cancelled) {
            setHubResults(hubModels);
            setSearchError(null);
          }
        } catch (searchError) {
          if (!cancelled) {
            setHubResults([]);
            setSearchError(
              searchError instanceof Error
                ? `${searchError.message} Showing local catalog matches only.`
                : "Could not search models right now. Showing local catalog matches only.",
            );
          }
        }
      })();
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [backendOnline, searchInput]);

  // Search result selection sync
  useEffect(() => {
    if (!searchResults.length) {
      setSelectedFamilyId("");
      setSelectedVariantId("");
      return;
    }
    const familyValid = searchResults.some((family) => family.id === selectedFamilyId);
    const nextFamilyId = familyValid ? selectedFamilyId : searchResults[0].id;
    if (nextFamilyId !== selectedFamilyId) {
      setSelectedFamilyId(nextFamilyId);
    }
    const family = searchResults.find((f) => f.id === nextFamilyId) ?? searchResults[0];
    if (family && !family.variants.some((v) => v.id === selectedVariantId)) {
      const dv = defaultVariantForFamily(family);
      if (dv) setSelectedVariantId(dv.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchResults]);

  // Download status polling
  const hasActiveDownloads = Object.values(activeDownloads).some((d) => d.state === "downloading");
  useEffect(() => {
    if (!hasActiveDownloads || !backendOnline) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const statuses = await getDownloadStatus();
          setActiveDownloads(buildDownloadStatusMap(statuses));
          if (statuses.some((s) => s.state === "completed")) {
            void refreshWorkspace(activeChatId || undefined);
          }
        } catch { /* ignore */ }
      })();
    }, 2000);
    return () => window.clearInterval(interval);
  }, [hasActiveDownloads, backendOnline, activeChatId, refreshWorkspace]);

  async function handleDownloadModel(repo: string) {
    try {
      setActiveDownloads((prev) => ({ ...prev, [repo]: pendingDownloadStatus(repo, prev[repo]) }));
      const download = await downloadModel(repo);
      setActiveDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
      setActiveDownloads((prev) => ({ ...prev, [repo]: failedDownloadStatus(repo, String(err)) }));
    }
  }

  async function handleCancelModelDownload(repo: string) {
    try {
      const download = await cancelDownload(repo);
      setActiveDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not pause download");
    }
  }

  async function handleDeleteModelDownload(repo: string) {
    try {
      await deleteModelDownload(repo);
      const statuses = await getDownloadStatus();
      setActiveDownloads(buildDownloadStatusMap(statuses));
      await refreshWorkspace(activeChatId || undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete download");
    }
  }

  async function toggleHubExpand(repo: string) {
    const next = expandedHubId === repo ? null : repo;
    setExpandedHubId(next);
    if (next && !hubFileCache[repo] && !hubFileLoading[repo]) {
      setHubFileLoading((current) => ({ ...current, [repo]: true }));
      setHubFileError((current) => {
        const { [repo]: _omit, ...rest } = current;
        return rest;
      });
      try {
        const payload = await listHubFiles(repo);
        setHubFileCache((current) => ({ ...current, [repo]: payload }));
      } catch (err) {
        setHubFileError((current) => ({
          ...current,
          [repo]: err instanceof Error ? err.message : "Could not load file list.",
        }));
      } finally {
        setHubFileLoading((current) => {
          const { [repo]: _omit, ...rest } = current;
          return rest;
        });
      }
    }
  }

  return {
    searchInput,
    setSearchInput,
    searchResults,
    setSearchResults,
    hubResults,
    searchError,
    expandedHubId,
    hubFileCache,
    hubFileLoading,
    hubFileError,
    detailFamilyId,
    setDetailFamilyId,
    selectedFamilyId,
    setSelectedFamilyId,
    selectedVariantId,
    setSelectedVariantId,
    expandedFamilyId,
    setExpandedFamilyId,
    expandedVariantId,
    setExpandedVariantId,
    activeDownloads,
    setActiveDownloads,
    discoverCapFilter,
    setDiscoverCapFilter,
    discoverFormatFilter,
    setDiscoverFormatFilter,
    hasActiveDownloads,
    handleDownloadModel,
    handleCancelModelDownload,
    handleDeleteModelDownload,
    toggleHubExpand,
  };
}
