import { isTauri } from "@tauri-apps/api/core";
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
  StrategyInstallLog,
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
    numKvHeads: emptyPreview.numKvHeads,
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
  const [installLogs, setInstallLogs] = useState<Record<string, StrategyInstallLog>>({});

  function installLabelFor(strategyId: string): string {
    const labels: Record<string, string> = {
      rotorquant: "RotorQuant",
      turboquant: "TurboQuant",
      triattention: "TriAttention",
      "dflash-mlx": "DFlash",
      dflash: "DFlash",
      chaosengine: "ChaosEngine",
    };
    return labels[strategyId] ?? strategyId;
  }

  function beginInstallLog(strategyId: string) {
    setInstallLogs((current) => ({
      ...current,
      [strategyId]: {
        strategyId,
        label: installLabelFor(strategyId),
        status: "running",
        startedAt: new Date().toLocaleString(),
        finishedAt: null,
        steps: [],
      },
    }));
  }

  function addInstallLogStep(strategyId: string, stepId: string, label: string, command: string) {
    setInstallLogs((current) => {
      const existing = current[strategyId] ?? {
        strategyId,
        label: installLabelFor(strategyId),
        status: "running" as const,
        startedAt: new Date().toLocaleString(),
        finishedAt: null,
        steps: [],
      };
      return {
        ...current,
        [strategyId]: {
          ...existing,
          status: "running",
          finishedAt: null,
          steps: [
            ...existing.steps,
            { id: stepId, label, command, status: "running", output: "" },
          ],
        },
      };
    });
  }

  function finishInstallLogStep(strategyId: string, stepId: string, status: "success" | "failed", output: string) {
    setInstallLogs((current) => {
      const existing = current[strategyId];
      if (!existing) return current;
      return {
        ...current,
        [strategyId]: {
          ...existing,
          steps: existing.steps.map((step) =>
            step.id === stepId ? { ...step, status, output } : step,
          ),
        },
      };
    });
  }

  function finishInstallLog(strategyId: string, status: "success" | "failed") {
    setInstallLogs((current) => {
      const existing = current[strategyId];
      if (!existing) return current;
      return {
        ...current,
        [strategyId]: {
          ...existing,
          status,
          finishedAt: new Date().toLocaleString(),
        },
      };
    });
  }

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
        requireApiAuth: settingsDraft.requireApiAuth,
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
      // Prefer the Tauri command path whenever we're in the desktop shell,
      // NOT just when ``tauriBackend.managedByTauri`` is true. That flag is
      // ``false`` by default (in ``BackendRuntimeInfo::default()``) and only
      // flips to ``true`` after the Rust ``bootstrap()`` thread finishes.
      // If the user clicks Stop/Restart before the initial runtime_info
      // fetch returns with the updated state (or hits the backend during a
      // probe crash that blanks the cached state), the old logic fell into
      // the "web fallback" branch which just POSTs /api/server/shutdown and
      // then polls for a respawn that never happens. Using ``isTauri()``
      // lets Rust handle both managed and attached cases correctly.
      if (isTauri()) {
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
      // See handleStopServer: use isTauri() so a stale or not-yet-populated
      // managedByTauri flag doesn't trap us in the web-fallback path where
      // nothing re-spawns the Python sidecar.
      if (isTauri()) {
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
    const pipCommandMap: Record<string, string> = {
      rotorquant: "./.venv/bin/python3 -m pip install turboquant",
      turboquant: "./.venv/bin/python3 -m pip install turboquant-mlx-full",
      triattention: "./.venv/bin/python3 -m pip install 'triattention @ git+https://github.com/WeianMao/triattention.git'",
      "dflash-mlx": "./.venv/bin/python3 -m pip install 'dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@f825ffb268e50d531e8b6524413b0847334a14dd'",
      dflash: "./.venv/bin/python3 -m pip install dflash",
    };
    const pipName = pipPackageMap[strategyId];
    if (!pipName) {
      beginInstallLog(strategyId);
      if (strategyId === "chaosengine") {
        const message = "ChaosEngine is not on PyPI. Desktop builds can bundle a vendored vendor/ChaosEngine checkout during npm run stage:runtime. For source/dev installs, clone https://github.com/cryptopoly/ChaosEngine and install it into the backend runtime with ./.venv/bin/python3 -m pip install -e /path/to/ChaosEngine, then restart ChaosEngineAI.";
        addInstallLogStep(strategyId, "manual", "Manual install required", "./.venv/bin/python3 -m pip install -e /path/to/ChaosEngine");
        finishInstallLogStep(strategyId, "manual", "failed", message);
        finishInstallLog(strategyId, "failed");
        setError(message);
      } else {
        const message = `No installer is configured for ${strategyId}.`;
        addInstallLogStep(strategyId, "manual", "No installer configured", strategyId);
        finishInstallLogStep(strategyId, "manual", "failed", message);
        finishInstallLog(strategyId, "failed");
        setError(message);
      }
      return;
    }
    beginInstallLog(strategyId);
    setInstallingPackage(strategyId);
    setError(null);
    try {
      // If this strategy needs llama-server-turbo and it's not installed,
      // build it first (clones + compiles the TurboQuant fork).
      if (needsTurboBinary) {
        const turboInstalled = workspace?.system?.llamaServerTurboPath;
        if (!turboInstalled) {
          addInstallLogStep(strategyId, "llama-server-turbo", "Build llama-server-turbo", "./scripts/build-llama-turbo.sh");
          const turboResult = await installSystemPackage("llama-server-turbo");
          finishInstallLogStep(
            strategyId,
            "llama-server-turbo",
            turboResult.ok ? "success" : "failed",
            turboResult.output,
          );
          if (!turboResult.ok) {
            finishInstallLog(strategyId, "failed");
            setError(`llama-server-turbo build failed: ${turboResult.output.slice(0, 300)}`);
            return;
          }
        }
      }

      addInstallLogStep(strategyId, "pip", "Install Python package", pipCommandMap[strategyId] ?? `./.venv/bin/python3 -m pip install ${pipName}`);
      const result = await installPipPackage(pipName);
      finishInstallLogStep(strategyId, "pip", result.ok ? "success" : "failed", result.output);
      if (result.ok) {
        await refreshWorkspace(activeChatId || undefined);
        finishInstallLog(strategyId, "success");
      } else {
        finishInstallLog(strategyId, "failed");
        setError(`Install failed: ${result.output.slice(0, 300)}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Install failed.";
      addInstallLogStep(strategyId, "request", "Install request failed", "ChaosEngineAI install request");
      finishInstallLogStep(strategyId, "request", "failed", message);
      finishInstallLog(strategyId, "failed");
      setError(message);
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
    installLogs,
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
