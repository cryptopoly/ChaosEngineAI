import { useEffect, useState } from "react";
import {
  getCachePreview,
  getTauriBackendInfo,
  restartManagedBackend,
  shutdownServer,
  stopManagedBackend,
  updateSettings,
  installPipPackage,
  installSystemPackage,
} from "../api";
import { emptyWorkspace, emptyLaunchPreferences, emptyPreview } from "../defaults";
import {
  settingsDraftFromWorkspace,
  syncStoppedBackend,
} from "../utils";
import type {
  ConversionResult,
  LaunchPreferences,
  ModelDirectorySetting,
  PreviewMetrics,
  TauriBackendInfo,
  WorkspaceData,
} from "../types";
import type { DataDirRestartPrompt, SettingsDraft } from "../types/chat";

export function useSettings(
  workspace: WorkspaceData,
  setWorkspace: React.Dispatch<React.SetStateAction<WorkspaceData>>,
  backendOnline: boolean,
  setBackendOnline: (online: boolean) => void,
  tauriBackend: TauriBackendInfo | null,
  setTauriBackend: (info: TauriBackendInfo | null) => void,
  setError: (msg: string | null) => void,
  setBusyAction: (action: string | null) => void,
  activeChatId: string,
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>,
  refreshImageData: () => Promise<void>,
) {
  const [settingsDraft, setSettingsDraft] = useState<SettingsDraft>(settingsDraftFromWorkspace(emptyWorkspace.settings));
  const [launchSettings, setLaunchSettings] = useState<LaunchPreferences>(emptyLaunchPreferences);
  const [preview, setPreview] = useState<PreviewMetrics>(emptyPreview);
  const [previewControls, setPreviewControls] = useState({
    bits: emptyLaunchPreferences.cacheBits,
    fp16Layers: emptyLaunchPreferences.fp16Layers,
    numLayers: emptyPreview.numLayers,
    numHeads: emptyPreview.numHeads,
    hiddenSize: emptyPreview.hiddenSize,
    contextTokens: emptyLaunchPreferences.contextTokens,
    paramsB: emptyPreview.paramsB,
    strategy: emptyLaunchPreferences.cacheStrategy,
  });
  const [dataDirRestartPrompt, setDataDirRestartPrompt] = useState<DataDirRestartPrompt | null>(null);
  const [newDirectoryLabel, setNewDirectoryLabel] = useState("");
  const [newDirectoryPath, setNewDirectoryPath] = useState("");
  const [conversionDraft, setConversionDraft] = useState({
    modelRef: "",
    path: "",
    hfRepo: "",
    outputPath: "",
    quantize: true,
    qBits: 4,
    qGroupSize: 64,
    dtype: "float16",
  });
  const [lastConversion, setLastConversion] = useState<ConversionResult | null>(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [serverModelKey, setServerModelKey] = useState("");
  const [installingPackage, setInstallingPackage] = useState<string | null>(null);

  // Cache preview calculation
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void (async () => {
        const nextPreview = await getCachePreview(previewControls);
        setPreview(nextPreview);
      })();
    }, 220);
    return () => window.clearTimeout(timeout);
  }, [previewControls]);

  // Sync previewControls from launchSettings
  useEffect(() => {
    setPreviewControls((current) => {
      if (
        current.bits === launchSettings.cacheBits &&
        current.fp16Layers === launchSettings.fp16Layers &&
        current.contextTokens === launchSettings.contextTokens &&
        current.strategy === launchSettings.cacheStrategy
      ) return current;
      return {
        ...current,
        bits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        contextTokens: launchSettings.contextTokens,
        strategy: launchSettings.cacheStrategy,
      };
    });
  }, [launchSettings.contextTokens, launchSettings.fp16Layers, launchSettings.cacheBits, launchSettings.cacheStrategy]);

  // Sync settings draft model directories
  useEffect(() => {
    if (!workspace.settings?.modelDirectories) return;
    setSettingsDraft((current) => {
      const dirs = (current.modelDirectories ?? []);
      let changed = false;
      const updated = dirs.map((directory) => {
        const latest = workspace.settings.modelDirectories.find((item) => item.id === directory.id);
        if (latest && (directory.exists !== latest.exists || directory.modelCount !== latest.modelCount)) {
          changed = true;
          return { ...directory, exists: latest.exists, modelCount: latest.modelCount };
        }
        return directory;
      });
      return changed ? { ...current, modelDirectories: updated } : current;
    });
  }, [workspace.settings?.modelDirectories]);

  function updateLaunchSetting<K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) {
    setLaunchSettings((current) => ({ ...current, [key]: value }));
  }

  function updateConversionDraft<K extends keyof typeof conversionDraft>(key: K, value: (typeof conversionDraft)[K]) {
    setConversionDraft((current) => ({ ...current, [key]: value }));
  }

  function handleAddDirectory() {
    const path = newDirectoryPath.trim();
    if (!path) {
      setError("Enter a directory path before adding it.");
      return;
    }
    const fallbackLabel = path.split("/").filter(Boolean).pop() || "Custom directory";
    const nextDirectory: ModelDirectorySetting = {
      id: `user-${Date.now()}`,
      label: newDirectoryLabel.trim() || fallbackLabel,
      path,
      enabled: true,
      source: "user",
    };
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: [...current.modelDirectories, nextDirectory],
    }));
    setNewDirectoryLabel("");
    setNewDirectoryPath("");
  }

  function handleToggleDirectory(directoryId: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.map((directory) =>
        directory.id === directoryId ? { ...directory, enabled: !directory.enabled } : directory,
      ),
    }));
  }

  function handleRemoveDirectory(directoryId: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.filter((directory) => directory.id !== directoryId),
    }));
  }

  function handleUpdateDirectoryPath(directoryId: string, nextPath: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.map((directory) =>
        directory.id === directoryId ? { ...directory, path: nextPath } : directory,
      ),
    }));
  }

  async function pickDirectory(currentPath?: string): Promise<string | null> {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      let defaultPath = currentPath?.trim() || undefined;
      if (!defaultPath) {
        try {
          const { homeDir } = await import("@tauri-apps/api/path");
          defaultPath = await homeDir();
        } catch { /* picker will open at its default */ }
      }
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select model directory",
        defaultPath,
      });
      return typeof selected === "string" && selected ? selected : null;
    } catch (err) {
      console.error("Folder picker failed", err);
      return null;
    }
  }

  async function handlePickDataDirectory() {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const picked = await tauriInvoke<string | null>("pick_directory");
      if (picked) {
        setSettingsDraft((current) => ({ ...current, dataDirectory: picked }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open the directory picker.");
    }
  }

  async function handlePickOutputDirectory(field: "imageOutputsDirectory" | "videoOutputsDirectory") {
    // Same Tauri command the data-directory picker uses — gives us a native
    // folder dialog that returns an absolute path. Falls back gracefully if
    // the user cancels (`null`) or the dialog raises (no Tauri runtime).
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const picked = await tauriInvoke<string | null>("pick_directory");
      if (picked) {
        setSettingsDraft((current) => ({ ...current, [field]: picked }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open the directory picker.");
    }
  }

  async function handlePickImageOutputsDirectory() {
    await handlePickOutputDirectory("imageOutputsDirectory");
  }

  async function handlePickVideoOutputsDirectory() {
    await handlePickOutputDirectory("videoOutputsDirectory");
  }

  async function handleSaveSettings() {
    setBusyAction("Saving settings...");
    try {
      const prevSettings = workspace.settings ?? { preferredServerPort: 8876, allowRemoteConnections: false };
      const response = await updateSettings({
        modelDirectories: settingsDraft.modelDirectories,
        preferredServerPort: settingsDraft.preferredServerPort,
        allowRemoteConnections: settingsDraft.allowRemoteConnections,
        autoStartServer: settingsDraft.autoStartServer,
        launchPreferences: launchSettings,
        remoteProviders: (settingsDraft.remoteProviders ?? []).map((p) => ({
          id: p.id,
          label: p.label,
          apiBase: p.apiBase,
          apiKey: p.apiKey ?? "",
          model: p.model,
        })),
        ...(settingsDraft.huggingFaceToken
          ? { huggingFaceToken: settingsDraft.huggingFaceToken }
          : {}),
        ...(settingsDraft.dataDirectory !== (workspace.settings?.dataDirectory ?? "")
          ? { dataDirectory: settingsDraft.dataDirectory }
          : {}),
        // Always send the per-modality overrides — empty string is a valid
        // "use the default" signal that the backend persists explicitly.
        ...(settingsDraft.imageOutputsDirectory !== (workspace.settings?.imageOutputsDirectory ?? "")
          ? { imageOutputsDirectory: settingsDraft.imageOutputsDirectory }
          : {}),
        ...(settingsDraft.videoOutputsDirectory !== (workspace.settings?.videoOutputsDirectory ?? "")
          ? { videoOutputsDirectory: settingsDraft.videoOutputsDirectory }
          : {}),
      });
      const settings = response.settings;
      setSettingsDraft(settingsDraftFromWorkspace(settings));
      setLaunchSettings(settings.launchPreferences);
      await refreshWorkspace(activeChatId || undefined);
      if (response.restartRequired) {
        setDataDirRestartPrompt({ migration: response.migrationSummary ?? null });
      }
      const restartRequired =
        settings.preferredServerPort !== prevSettings.preferredServerPort ||
        settings.allowRemoteConnections !== prevSettings.allowRemoteConnections;
      if (restartRequired && tauriBackend?.managedByTauri) {
        setBusyAction("Restarting server to apply changes...");
        const runtimeInfo = await restartManagedBackend();
        if (runtimeInfo) {
          setTauriBackend(runtimeInfo);
          if (runtimeInfo.started) {
            await refreshWorkspace(activeChatId || undefined);
            await refreshImageData();
            setError(null);
            setTauriBackend((await getTauriBackendInfo(true)) ?? runtimeInfo);
            setBackendOnline(true);
          }
        }
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to save settings.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStopServer() {
    setBusyAction("Stopping server...");
    try {
      if (tauriBackend?.managedByTauri) {
        const runtimeInfo = await stopManagedBackend();
        if (!runtimeInfo) throw new Error("The desktop sidecar could not be stopped.");
        setTauriBackend(runtimeInfo);
        setBackendOnline(false);
        setWorkspace((current) => syncStoppedBackend(current, runtimeInfo));
      } else {
        try { await shutdownServer(); } catch { /* Expected */ }
        setBackendOnline(false);
        setWorkspace((current) => syncStoppedBackend(current, null));
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to stop the API service.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRestartServer() {
    setBusyAction("Restarting server...");
    try {
      if (tauriBackend?.managedByTauri) {
        const runtimeInfo = await restartManagedBackend();
        if (!runtimeInfo) throw new Error("The desktop sidecar could not be restarted.");
        setTauriBackend(runtimeInfo);
        // The Rust side already waits BACKEND_START_TIMEOUT (12 s) for the
        // port to respond, but on slow machines the backend may still be
        // finishing FastAPI init.  Retry a few times before giving up.
        let online = runtimeInfo.started;
        if (!online) {
          const { checkBackend } = await import("../api");
          for (let i = 0; i < 5; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            online = await checkBackend();
            if (online) break;
          }
        }
        if (!online) {
          throw new Error(runtimeInfo.startupError ?? "The API service did not come back online.");
        }
        await refreshWorkspace(activeChatId || undefined);
        await refreshImageData();
        setError(null);
        setTauriBackend((await getTauriBackendInfo(true)) ?? runtimeInfo);
        setBackendOnline(true);
      } else {
        try { await shutdownServer(); } catch { /* Expected */ }
        setBackendOnline(false);
        const { checkBackend } = await import("../api");
        let came_back = false;
        for (let i = 0; i < 15; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          const online = await checkBackend();
          if (online) { came_back = true; break; }
        }
        if (came_back) {
          await refreshWorkspace(activeChatId || undefined);
          await refreshImageData();
          setError(null);
          setBackendOnline(true);
        } else {
          setError("Server was stopped. Please restart it manually, then it will reconnect.");
        }
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to restart the API service.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleInstallPackage(strategyId: string) {
    // Strategies that need the turbo binary (llama-server-turbo) for GGUF.
    const needsTurboBinary = strategyId === "rotorquant" || strategyId === "turboquant";

    const pipPackageMap: Record<string, string> = {
      rotorquant: "turboquant",
      turboquant: "turboquant-mlx",
      triattention: "triattention",
      "dflash-mlx": "dflash-mlx",
      dflash: "dflash",
    };
    const pipName = pipPackageMap[strategyId];
    if (!pipName) {
      if (strategyId === "chaosengine") {
        setError(
          "ChaosEngine is not on PyPI. Desktop builds can bundle a vendored vendor/ChaosEngine checkout during npm run stage:runtime. For source/dev installs, clone https://github.com/cryptopoly/ChaosEngine and install it into the backend runtime with ./.venv/bin/python3 -m pip install -e /path/to/ChaosEngine, then restart ChaosEngineAI.",
        );
      }
      return;
    }
    setInstallingPackage(strategyId);
    setError(null);
    try {
      // If this strategy needs llama-server-turbo and it's not installed,
      // build it first (clones + compiles the TurboQuant fork).
      if (needsTurboBinary) {
        const turboInstalled = workspace?.system?.llamaServerTurboPath;
        if (!turboInstalled) {
          const turboResult = await installSystemPackage("llama-server-turbo");
          if (!turboResult.ok) {
            setError(`llama-server-turbo build failed: ${turboResult.output.slice(0, 300)}`);
            return;
          }
        }
      }

      const result = await installPipPackage(pipName);
      if (result.ok) {
        await refreshWorkspace(activeChatId || undefined);
      } else {
        setError(`Install failed: ${result.output.slice(0, 300)}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Install failed.");
    } finally {
      setInstallingPackage(null);
    }
  }

  async function handleInstallSystemPackage(packageName: string) {
    setInstallingPackage(packageName);
    setError(null);
    try {
      const result = await installSystemPackage(packageName);
      if (result.ok) {
        await refreshWorkspace(activeChatId || undefined);
      } else {
        setError(`Install failed: ${result.output.slice(0, 300)}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Install failed.");
    } finally {
      setInstallingPackage(null);
    }
  }

  return {
    settingsDraft,
    setSettingsDraft,
    launchSettings,
    setLaunchSettings,
    preview,
    setPreview,
    previewControls,
    setPreviewControls,
    dataDirRestartPrompt,
    setDataDirRestartPrompt,
    newDirectoryLabel,
    setNewDirectoryLabel,
    newDirectoryPath,
    setNewDirectoryPath,
    conversionDraft,
    setConversionDraft,
    lastConversion,
    setLastConversion,
    systemPrompt,
    setSystemPrompt,
    serverModelKey,
    setServerModelKey,
    installingPackage,
    updateLaunchSetting,
    updateConversionDraft,
    handleAddDirectory,
    handleToggleDirectory,
    handleRemoveDirectory,
    handleUpdateDirectoryPath,
    pickDirectory,
    handlePickDataDirectory,
    handlePickImageOutputsDirectory,
    handlePickVideoOutputsDirectory,
    handleSaveSettings,
    handleStopServer,
    handleRestartServer,
    handleInstallPackage,
    handleInstallSystemPackage,
  };
}
