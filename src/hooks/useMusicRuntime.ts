/**
 * Music catalog, runtime, the music pack install and weight downloads: what the Music Models page and the
 * Studio header read. Polls downloads while one is running and the install job while it is running, then
 * re-reads the catalog and runtime so "Installed" appears without a reload.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cancelMusicDownload,
  cancelMusicInstall,
  deleteMusicDownload,
  downloadMusicModel,
  getMusicCatalog,
  getMusicDownloadStatus,
  getMusicInstallStatus,
  getMusicRuntime,
  startMusicInstall,
} from "../api";
import type { DownloadStatus } from "../api";
import type { MusicInstallJob, MusicModelFamily, MusicRuntimeStatus } from "../types";
import { flattenVariants, pickDefaultVariant } from "../features/music/musicLogic";

const DOWNLOAD_POLL_MS = 2000;
const INSTALL_POLL_MS = 1500;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function useMusicRuntime(backendOnline: boolean, setError: (message: string | null) => void) {
  const [families, setFamilies] = useState<MusicModelFamily[]>([]);
  const [runtime, setRuntime] = useState<MusicRuntimeStatus | null>(null);
  const [downloads, setDownloads] = useState<Record<string, DownloadStatus>>({});
  const [installJob, setInstallJob] = useState<MusicInstallJob | null>(null);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const downloadingRef = useRef(false);

  const refreshRuntime = useCallback(async () => {
    try {
      const next = await getMusicRuntime();
      setRuntime(next);
      setInstallJob((previous) => (previous?.running ? previous : next.install));
    } catch (err) {
      setError(message(err));
    }
  }, [setError]);

  const refreshCatalog = useCallback(async () => {
    try {
      const next = await getMusicCatalog();
      setFamilies(next.families);
      setCatalogLoaded(true);
    } catch (err) {
      setError(message(err));
    }
  }, [setError]);

  const refreshDownloads = useCallback(async () => {
    try {
      const rows = await getMusicDownloadStatus();
      setDownloads(Object.fromEntries(rows.map((row) => [row.repo, row])));
      return rows;
    } catch {
      return [];
    }
  }, []);

  const refreshMusicData = useCallback(async () => {
    await Promise.all([refreshCatalog(), refreshRuntime(), refreshDownloads()]);
  }, [refreshCatalog, refreshRuntime, refreshDownloads]);

  useEffect(() => {
    if (backendOnline) void refreshMusicData();
  }, [backendOnline, refreshMusicData]);

  // Poll downloads while one runs; the catalog is re-read when the last one finishes.
  const anyDownloading = Object.values(downloads).some((row) => row.state === "downloading");
  useEffect(() => {
    if (!anyDownloading) {
      if (downloadingRef.current) {
        downloadingRef.current = false;
        void refreshCatalog();
        void refreshRuntime();
      }
      return;
    }
    downloadingRef.current = true;
    const timer = window.setInterval(() => void refreshDownloads(), DOWNLOAD_POLL_MS);
    return () => window.clearInterval(timer);
  }, [anyDownloading, refreshDownloads, refreshCatalog, refreshRuntime]);

  // Poll the install job while it runs; a finished job re-reads the runtime and catalog.
  const installRunning = Boolean(installJob?.running);
  useEffect(() => {
    if (!installRunning) return;
    const timer = window.setInterval(() => {
      void getMusicInstallStatus()
        .then((job) => {
          setInstallJob(job);
          if (!job.running) void refreshMusicData();
        })
        .catch(() => undefined);
    }, INSTALL_POLL_MS);
    return () => window.clearInterval(timer);
  }, [installRunning, refreshMusicData]);

  const variants = useMemo(() => flattenVariants(families), [families]);
  const defaultVariant = useMemo(() => pickDefaultVariant(families), [families]);
  const selectedVariant = useMemo(
    () => variants.find((variant) => variant.id === selectedModelId) ?? defaultVariant,
    [variants, selectedModelId, defaultVariant],
  );
  const installedVariants = useMemo(
    () => variants.filter((variant) => variant.availableLocally && (variant.missingRequires ?? []).length === 0),
    [variants],
  );

  const startInstall = useCallback(async () => {
    try {
      setInstallJob(await startMusicInstall());
    } catch (err) {
      setError(message(err));
    }
  }, [setError]);

  const cancelInstall = useCallback(async () => {
    try {
      setInstallJob(await cancelMusicInstall());
    } catch (err) {
      setError(message(err));
    }
  }, [setError]);

  const startDownload = useCallback(
    async (repo: string) => {
      try {
        const row = await downloadMusicModel(repo);
        setDownloads((previous) => ({ ...previous, [row.repo]: row }));
      } catch (err) {
        setError(message(err));
      }
    },
    [setError],
  );

  const pauseDownload = useCallback(
    async (repo: string) => {
      try {
        const row = await cancelMusicDownload(repo);
        setDownloads((previous) => ({ ...previous, [row.repo]: row }));
      } catch (err) {
        setError(message(err));
      }
    },
    [setError],
  );

  const deleteDownload = useCallback(
    async (repo: string) => {
      try {
        await deleteMusicDownload(repo);
        setDownloads((previous) => {
          const next = { ...previous };
          delete next[repo];
          return next;
        });
        await Promise.all([refreshCatalog(), refreshRuntime()]);
      } catch (err) {
        setError(message(err));
      }
    },
    [setError, refreshCatalog, refreshRuntime],
  );

  return {
    families,
    variants,
    runtime,
    downloads,
    installJob,
    catalogLoaded,
    selectedModelId,
    setSelectedModelId,
    selectedVariant,
    installedVariants,
    hasInstalledModel: installedVariants.length > 0,
    refreshMusicData,
    refreshRuntime,
    refreshCatalog,
    startInstall,
    cancelInstall,
    startDownload,
    pauseDownload,
    deleteDownload,
  };
}

export type UseMusicRuntime = ReturnType<typeof useMusicRuntime>;
