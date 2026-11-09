import { useCallback, useEffect, useState } from "react";
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
}

/** Voices and speech models for the Speak, Voices and Models screens; polls while a model downloads. */
export function useVoiceCatalog(enabled = true): VoiceCatalog {
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

  return { voices, models, modelsAvailable, cloneAllowed, loaded, error, refresh, download, cancelDownload, remove, kokoro };
}
