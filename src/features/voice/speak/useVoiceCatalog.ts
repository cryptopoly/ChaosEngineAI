import { useCallback, useEffect, useMemo, useState } from "react";
import { splitForLane } from "../models/voiceModelsLogic";
import { cancelCloneDownload, deleteSpeechModel, downloadCloneModel, getCloneModels } from "../clone/cloneApi";
import { getVoiceModels, getVoices } from "../../../api";
import type { VoiceEntry, VoiceModelRow } from "../../../types";

/** Kokoro's measured peak and fit (it has no catalog row; the Models table shows it first). */
export interface KokoroInfo {
  peakGb: number | null;
  fit: unknown;
}

export interface VoiceCatalog {
  voices: VoiceEntry[];
  models: VoiceModelRow[];
  /** Preset, designed and cloned voices run on Apple Silicon Macs for now. */
  modelsAvailable: boolean;
  /** Cloned voices can be listed and used from this window. */
  cloneAllowed: boolean;
  loaded: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  download: (key: string) => Promise<void>;
  /** Stop a running download, then list again. */
  cancelDownload?: (key: string) => Promise<void>;
  /** Delete a speech model's files, then list again. Rejects with the backend's reason (in use, downloading). */
  remove?: (key: string) => Promise<void>;
  kokoro?: KokoroInfo | null;
  /** On a PC: the speech models that only run on an Apple Silicon Mac, left out of ``models`` (D-102 A). */
  macOnly?: VoiceModelRow[];
  /** On a PC: how many voices were left out of ``voices`` because only those models speak them. */
  hiddenVoices?: number;
}

/**
 * Voices and speech models for the Speak, Voices and Models screens; polls while a model downloads. On the portable
 * lane (a PC) the Mac-only models and their voices are left out and counted instead (``splitForLane``).
 */
export function useVoiceCatalog(enabled = true, lane?: string | null): VoiceCatalog {
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const [models, setModels] = useState<VoiceModelRow[]>([]);
  const [kokoro, setKokoro] = useState<KokoroInfo | null>(null);
  const [modelsAvailable, setModelsAvailable] = useState(false);
  const [cloneAllowed, setCloneAllowed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [voiceList, modelList, clone] = await Promise.allSettled([getVoices(), getVoiceModels(), getCloneModels()]);
    if (voiceList.status === "fulfilled") setVoices(voiceList.value);
    if (modelList.status === "fulfilled") {
      setModels(modelList.value.models);
      setModelsAvailable(modelList.value.available);
      setKokoro(modelList.value.kokoro ?? null);
    }
    if (clone.status === "fulfilled") setCloneAllowed(clone.value.available && clone.value.allowedHere);
    const failed = [voiceList, modelList].find((r) => r.status === "rejected");
    setError(failed && failed.status === "rejected" ? (failed.reason instanceof Error ? failed.reason.message : String(failed.reason)) : null);
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (enabled) void refresh();
  }, [enabled, refresh]);

  const downloading = models.some((m) => m.download.state === "downloading");
  useEffect(() => {
    if (!downloading) return;
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [downloading, refresh]);

  const download = useCallback(
    async (key: string) => {
      try {
        await downloadCloneModel(key);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
      await refresh();
    },
    [refresh],
  );

  const cancelDownload = useCallback(
    async (key: string) => {
      try {
        await cancelCloneDownload(key);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (key: string) => {
      try {
        await deleteSpeechModel(key);
      } finally {
        await refresh();
      }
    },
    [refresh],
  );

  const split = useMemo(() => splitForLane(models, voices, lane), [models, voices, lane]);
  return {
    voices: split.voices,
    models: split.models,
    macOnly: split.macOnly,
    hiddenVoices: split.hiddenVoices,
    modelsAvailable,
    cloneAllowed,
    loaded,
    error,
    refresh,
    download,
    cancelDownload,
    remove,
    kokoro,
  };
}
