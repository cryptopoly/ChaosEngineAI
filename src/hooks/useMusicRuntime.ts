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
  saveMusicEmbedTag,
  saveMusicSettings,
  startMusicInstall,
} from "../api";
import type { DownloadStatus } from "../api";
import type { MusicInstallJob, MusicModelFamily, MusicRuntimeStatus } from "../types";
import { DOWNLOAD_POLL_MS, MAX_POLL_FAILURES, flattenVariants, pickDefaultVariant, pollDelay } from "../features/music/musicLogic";

const INSTALL_POLL_MS = 1500;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function useMusicRuntime(backendOnline: boolean, setErrorProp: (message: string | null) => void) {
  const errorRef = useRef(setErrorProp);
  errorRef.current = setErrorProp;
  const setError = useCallback((text: string | null) => errorRef.current(text), []);
  const [families, setFamilies] = useState<MusicModelFamily[]>([]);
  const [runtime, setRuntime] = useState<MusicRuntimeStatus | null>(null);
  const [downloads, setDownloads] = useState<Record<string, DownloadStatus>>({});
  const [installJob, setInstallJob] = useState<MusicInstallJob | null>(null);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const downloadingRef = useRef(false);
  /** Set when the download status stopped answering and polling gave up. */
  const [downloadPollError, setDownloadPollError] = useState<string | null>(null);

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

  /** One poll that reports whether the status endpoint answered. */
  const pollDownloads = useCallback(async (): Promise<string | null> => {
    try {
      const rows = await getMusicDownloadStatus();
      setDownloads(Object.fromEntries(rows.map((row) => [row.repo, row])));
      return null;
    } catch (err) {
      return message(err);
    }
  }, []);

  const refreshMusicData = useCallback(async () => {
    setDownloadPollError(null);
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
    if (downloadPollError) return; // gave up; a manual refresh clears this and polling resumes
    // Each failed poll waits longer (2, 4, 8 ... 30 s); after a run of failures it stops and says so.
    let failures = 0;
    let timer: number | undefined;
    let stopped = false;
    const tick = async () => {
      const failure = await pollDownloads();
      if (stopped) return;
      failures = failure === null ? 0 : failures + 1;
      if (failure === null) setDownloadPollError(null);
      if (failures >= MAX_POLL_FAILURES) {
        setDownloadPollError(failure);
        return;
      }
      timer = window.setTimeout(() => void tick(), pollDelay(failures));
    };
    timer = window.setTimeout(() => void tick(), pollDelay(0));
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [anyDownloading, downloadPollError, pollDownloads, refreshCatalog, refreshRuntime]);

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

  /** D-34: the AI tag in new files. Lives in the app settings; the runtime reports it back. */
  const setEmbedTag = useCallback(
    async (on: boolean) => {
      try {
        await saveMusicEmbedTag(on);
        await refreshRuntime();
      } catch (err) {
        setError(message(err));
      }
    },
    [setError, refreshRuntime],
  );

  const setIdleMinutes = useCallback(
    async (minutes: number) => {
      try {
        await saveMusicSettings({ idleUnloadMinutes: minutes });
        await refreshRuntime();
      } catch (err) {
        setError(message(err));
      }
    },
    [setError, refreshRuntime],
  );

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
    downloadPollError,
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
    setEmbedTag,
    setIdleMinutes,
    startDownload,
    pauseDownload,
    deleteDownload,
  };
}

export type UseMusicRuntime = ReturnType<typeof useMusicRuntime>;
