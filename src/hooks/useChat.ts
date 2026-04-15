import { useEffect, useRef, useState } from "react";
import {
  checkBackend,
  createSession,
  deleteSession,
  deleteSessionDocument,
  generateChatStream,
  getTauriBackendInfo,
  restartManagedBackend,
  uploadSessionDocument,
  updateSession,
} from "../api";
import {
  upsertSession,
  sortSessions,
  titleFromPrompt,
  syncRuntime,
} from "../utils";
import type {
  ChatSession,
  ChatThinkingMode,
  LaunchPreferences,
  LoadModelActionResult,
  ModelVariant,
  TabId,
  WorkspaceData,
} from "../types";
import type { ChatModelOption } from "../types/chat";

export function useChat(
  workspace: WorkspaceData,
  setWorkspace: React.Dispatch<React.SetStateAction<WorkspaceData>>,
  backendOnline: boolean,
  setBackendOnline: (online: boolean) => void,
  setError: (msg: string | null) => void,
  launchSettings: LaunchPreferences,
  systemPrompt: string,
  setActiveTab: (tab: TabId) => void,
  handleLoadModel: (payload: {
    modelRef: string;
    modelName?: string;
    canonicalRepo?: string | null;
    source?: string;
    backend?: string;
    path?: string;
    nextTab?: TabId;
    busyLabel?: string;
  }) => Promise<LoadModelActionResult>,
  defaultChatVariant: ModelVariant | null,
  threadModelOptions: ChatModelOption[],
  launchCacheLabel: string,
  loadedModelCacheLabel: string,
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>,
) {
  const [activeChatId, setActiveChatId] = useState("");
  const [threadTitleDraft, setThreadTitleDraft] = useState("");
  const [draftMessage, setDraftMessage] = useState("");
  const [chatBusySessionId, setChatBusySessionId] = useState<string | null>(null);
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const [enableTools, setEnableTools] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const sortedChatSessions = sortSessions(workspace.chatSessions);
  const activeChat = workspace.chatSessions.find((session) => session.id === activeChatId) ?? sortedChatSessions[0];
  const activeThinkingMode: ChatThinkingMode = activeChat?.thinkingMode === "auto" ? "auto" : "off";
  const activeRuntimeProfileMatchesLaunchSettings = (() => {
    const loaded = workspace.runtime.loadedModel;
    if (!loaded) return false;
    return (
      loaded.cacheBits === launchSettings.cacheBits &&
      loaded.fp16Layers === launchSettings.fp16Layers &&
      loaded.fusedAttention === launchSettings.fusedAttention &&
      loaded.cacheStrategy === launchSettings.cacheStrategy &&
      loaded.fitModelInMemory === launchSettings.fitModelInMemory &&
      loaded.contextTokens === launchSettings.contextTokens &&
      (loaded.speculativeDecoding ?? false) === launchSettings.speculativeDecoding &&
      (loaded.treeBudget ?? 0) === launchSettings.treeBudget
    );
  })();

  // Chat session validity
  useEffect(() => {
    if (!workspace.chatSessions.some((session) => session.id === activeChatId)) {
      setActiveChatId(workspace.chatSessions[0]?.id ?? "");
    }
  }, [workspace.chatSessions, activeChatId]);

  // Active chat title sync
  useEffect(() => {
    const nextActiveChat = workspace.chatSessions.find((session) => session.id === activeChatId) ?? workspace.chatSessions[0];
    setThreadTitleDraft(nextActiveChat?.title ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChatId]);

  function sessionModelPayload(session?: ChatSession | null) {
    if (session?.modelRef) {
      return {
        modelRef: session.modelRef,
        modelName: session.model,
        canonicalRepo: session.canonicalRepo ?? undefined,
        source: session.modelSource ?? "catalog",
        path: session.modelPath ?? undefined,
        backend: session.modelBackend ?? "auto",
      };
    }
    if (workspace.runtime.loadedModel) {
      return {
        modelRef: workspace.runtime.loadedModel.ref,
        modelName: workspace.runtime.loadedModel.name,
        canonicalRepo: workspace.runtime.loadedModel.canonicalRepo ?? undefined,
        source: workspace.runtime.loadedModel.source,
        path: workspace.runtime.loadedModel.path ?? undefined,
        backend: workspace.runtime.loadedModel.backend,
      };
    }
    if (defaultChatVariant) {
      return {
        modelRef: defaultChatVariant.id,
        modelName: defaultChatVariant.name,
        canonicalRepo: defaultChatVariant.repo,
        source: "catalog",
        backend: defaultChatVariant.backend,
      };
    }
    return null;
  }

  function mergeSessionMetadata(session: ChatSession, patch: Partial<ChatSession>): ChatSession {
    return { ...session, ...patch };
  }

  function appendOptimisticTurn(sessionId: string, prompt: string) {
    const updatedAt = new Date().toLocaleString();
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              updatedAt,
              messages: [
                ...session.messages,
                { role: "user" as const, text: prompt, metrics: null },
                { role: "assistant" as const, text: "", reasoning: "", reasoningDone: true, metrics: null },
              ],
            }
          : session,
      ),
    }));
  }

  function replaceOptimisticAssistant(sessionId: string, prompt: string, text: string) {
    const updatedAt = new Date().toLocaleString();
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((session) => {
        if (session.id !== sessionId) return session;
        const messages = [...session.messages];
        const last = messages[messages.length - 1];
        const previous = messages[messages.length - 2];
        if (
          last?.role === "assistant" &&
          !last.text &&
          !last.metrics &&
          previous?.role === "user" &&
          previous.text === prompt
        ) {
          messages[messages.length - 1] = { ...last, text };
        } else {
          messages.push({ role: "user", text: prompt, metrics: null });
          messages.push({ role: "assistant", text, metrics: null });
        }
        return { ...session, updatedAt, messages };
      }),
    }));
  }

  function rollbackOptimisticTurn(sessionId: string, prompt: string) {
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((session) => {
        if (session.id !== sessionId) return session;
        const messages = [...session.messages];
        const last = messages[messages.length - 1];
        const previous = messages[messages.length - 2];
        if (
          last?.role === "assistant" &&
          !last.text &&
          !last.metrics &&
          previous?.role === "user" &&
          previous.text === prompt
        ) {
          return { ...session, messages: messages.slice(0, -2) };
        }
        return session;
      }),
    }));
  }

  async function persistSessionChanges(sessionId: string, patch: Partial<ChatSession>) {
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((session) =>
        session.id === sessionId ? mergeSessionMetadata(session, patch) : session,
      ),
    }));

    if (!backendOnline) return;

    try {
      const session = await updateSession(sessionId, {
        title: patch.title,
        model: patch.model,
        modelRef: patch.modelRef,
        canonicalRepo: patch.canonicalRepo,
        modelSource: patch.modelSource,
        modelPath: patch.modelPath,
        modelBackend: patch.modelBackend,
        thinkingMode: patch.thinkingMode,
        pinned: patch.pinned,
        cacheStrategy: patch.cacheStrategy,
        cacheBits: patch.cacheBits,
        fp16Layers: patch.fp16Layers,
        fusedAttention: patch.fusedAttention,
        fitModelInMemory: patch.fitModelInMemory,
        contextTokens: patch.contextTokens,
        speculativeDecoding: patch.speculativeDecoding,
        dflashDraftModel: patch.dflashDraftModel,
        treeBudget: patch.treeBudget,
      });
      setWorkspace((current) => ({
        ...current,
        chatSessions: upsertSession(current.chatSessions, session),
      }));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to update thread.");
    }
  }

  async function handleCreateSession() {
    const fallbackModel = sessionModelPayload(activeChat);
    if (!backendOnline) {
      const loaded = workspace.runtime.loadedModel;
      const localSession: ChatSession = {
        id: `local-${Date.now()}`,
        title: "New chat",
        updatedAt: new Date().toLocaleString(),
        pinned: false,
        model: fallbackModel?.modelName ?? "Choose a model",
        modelRef: fallbackModel?.modelRef ?? null,
        canonicalRepo: fallbackModel?.canonicalRepo ?? null,
        modelSource: fallbackModel?.source ?? "catalog",
        modelPath: fallbackModel?.path ?? null,
        modelBackend: fallbackModel?.backend ?? "auto",
        thinkingMode: activeThinkingMode,
        cacheLabel: loaded ? loadedModelCacheLabel : launchCacheLabel,
        cacheStrategy: loaded?.cacheStrategy ?? launchSettings.cacheStrategy,
        cacheBits: loaded?.cacheBits ?? launchSettings.cacheBits,
        fp16Layers: loaded?.fp16Layers ?? launchSettings.fp16Layers,
        fusedAttention: loaded?.fusedAttention ?? launchSettings.fusedAttention,
        fitModelInMemory: loaded?.fitModelInMemory ?? launchSettings.fitModelInMemory,
        contextTokens: loaded?.contextTokens ?? launchSettings.contextTokens,
        speculativeDecoding: loaded?.speculativeDecoding ?? launchSettings.speculativeDecoding,
        dflashDraftModel: loaded?.dflashDraftModel ?? null,
        treeBudget: loaded?.treeBudget ?? launchSettings.treeBudget,
        messages: [],
      };
      setWorkspace((current) => ({ ...current, chatSessions: [localSession, ...current.chatSessions] }));
      setActiveChatId(localSession.id);
      setThreadTitleDraft(localSession.title);
      return;
    }

    try {
      const session = await createSession("New chat");
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, session) }));
      setActiveChatId(session.id);
      setThreadTitleDraft(session.title);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to create session.");
    }
  }

  async function ensureBackendAvailable(preferredChatId?: string): Promise<{
    online: boolean;
    startupError: string | null;
  }> {
    if (backendOnline) {
      return { online: true, startupError: null };
    }

    let online = await checkBackend();
    if (online) {
      setBackendOnline(true);
      return { online: true, startupError: null };
    }

    const runtimeInfo = await getTauriBackendInfo(true);
    if (!runtimeInfo?.managedByTauri) {
      return { online: false, startupError: runtimeInfo?.startupError ?? null };
    }

    const restartInfo = await restartManagedBackend();
    const startupError = restartInfo?.startupError ?? runtimeInfo.startupError ?? null;
    if (!restartInfo) {
      return { online: false, startupError };
    }

    for (let attempt = 0; attempt < 5; attempt += 1) {
      online = await checkBackend();
      if (online) {
        setBackendOnline(true);
        await refreshWorkspace(preferredChatId);
        return { online: true, startupError: null };
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    return { online: false, startupError };
  }

  async function handleRenameActiveThread() {
    if (!activeChat) return;
    const nextTitle = threadTitleDraft.trim();
    if (!nextTitle || nextTitle === activeChat.title) {
      setThreadTitleDraft(activeChat.title);
      return;
    }
    await persistSessionChanges(activeChat.id, { title: nextTitle, updatedAt: new Date().toLocaleString() });
  }

  async function handleToggleThreadPin(session: ChatSession) {
    await persistSessionChanges(session.id, {
      pinned: !session.pinned,
      updatedAt: new Date().toLocaleString(),
    });
  }

  async function handleThinkingModeChange(nextMode: ChatThinkingMode) {
    if (!activeChat) return;
    if ((activeChat.thinkingMode ?? "off") === nextMode) return;
    await persistSessionChanges(activeChat.id, {
      thinkingMode: nextMode,
      updatedAt: new Date().toLocaleString(),
    });
  }

  async function handleDeleteSession(sessionId: string) {
    try {
      await deleteSession(sessionId);
      setWorkspace((current) => ({
        ...current,
        chatSessions: current.chatSessions.filter((s) => s.id !== sessionId),
      }));
      if (activeChatId === sessionId) {
        const remaining = workspace.chatSessions.filter((s) => s.id !== sessionId);
        setActiveChatId(remaining[0]?.id ?? "");
      }
    } catch {
      setError("Failed to delete session.");
    }
  }

  async function handleSelectThreadModel(nextKey: string) {
    const nextOption = threadModelOptions.find((option) => option.key === nextKey);
    if (!activeChat || !nextOption) return;
    await persistSessionChanges(activeChat.id, {
      model: nextOption.model,
      modelRef: nextOption.modelRef,
      canonicalRepo: nextOption.canonicalRepo ?? null,
      modelSource: nextOption.source,
      modelPath: nextOption.path ?? null,
      modelBackend: nextOption.backend,
      updatedAt: new Date().toLocaleString(),
    });
  }

  async function handleLoadActiveThreadModel() {
    const modelSelection = sessionModelPayload(activeChat);
    if (!modelSelection) {
      setError("Choose a model for this thread before loading it.");
      return;
    }
    await handleLoadModel({
      modelRef: modelSelection.modelRef,
      modelName: modelSelection.modelName,
      canonicalRepo: modelSelection.canonicalRepo,
      source: modelSelection.source,
      backend: modelSelection.backend,
      path: modelSelection.path,
      nextTab: "chat",
    });
  }

  function handleCopyMessage(text: string) {
    void navigator.clipboard.writeText(text).catch(() => {});
  }

  function handleDeleteMessage(index: number) {
    if (!activeChat) return;
    const sessionId = activeChat.id;
    const updatedMessages = activeChat.messages.filter((_, i) => i !== index);
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((s) =>
        s.id === sessionId ? { ...s, messages: updatedMessages } : s,
      ),
    }));
    void updateSession(sessionId, { messages: updatedMessages })
      .then((session) => {
        setWorkspace((current) => ({
          ...current,
          chatSessions: upsertSession(current.chatSessions, session),
        }));
      })
      .catch(() => {});
  }

  async function handleRetryMessage(index: number) {
    if (!activeChat) return;
    const messages = activeChat.messages;
    const target = messages[index];
    if (!target || target.role !== "assistant") return;

    let userIdx = index - 1;
    while (userIdx >= 0 && messages[userIdx].role !== "user") userIdx--;
    if (userIdx < 0) return;
    const userText = messages[userIdx].text;

    // Remove the user message + assistant response, then re-send.
    // Pass the text directly to sendMessage to avoid state-timing races.
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((s) =>
        s.id === activeChat.id ? { ...s, messages: s.messages.slice(0, userIdx) } : s,
      ),
    }));
    await sendMessage(userText);
  }

  const DOC_EXTENSIONS = new Set([
    "pdf", "txt", "md", "rst", "csv", "json", "yaml", "yml", "toml",
    "py", "js", "ts", "tsx", "jsx", "rs", "go", "java", "c", "cpp",
    "h", "hpp", "rb", "php", "swift", "kt", "html", "css", "sh",
  ]);

  async function handleChatFileDrop(files: FileList) {
    if (!files?.length || !activeChat) return;
    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        if (file.size > 10 * 1024 * 1024) {
          setError("Image must be under 10MB");
          continue;
        }
        await new Promise<void>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => {
            const b64 = (reader.result as string).split(",")[1];
            if (b64) setPendingImages((prev) => [...prev, b64]);
            resolve();
          };
          reader.readAsDataURL(file);
        });
      } else {
        const ext = (file.name.split(".").pop() ?? "").toLowerCase();
        if (!DOC_EXTENSIONS.has(ext)) {
          setError(`Unsupported file type: .${ext}`);
          continue;
        }
        try {
          await uploadSessionDocument(activeChat.id, file);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Upload failed");
        }
      }
    }
    await refreshWorkspace(activeChat.id);
  }

  async function sendMessage(overrideText?: string) {
    const rawPrompt = overrideText ?? draftMessage;
    const trimmed = rawPrompt.trim();
    if (!trimmed) return;

    const sendingSessionId = activeChat?.id ?? null;
    const threadModel = sessionModelPayload(activeChat);
    const pendingImagesSnapshot = pendingImages.length > 0 ? [...pendingImages] : [];
    const loadedIdentifiers = [
      workspace.runtime.loadedModel?.ref,
      workspace.runtime.loadedModel?.canonicalRepo,
      workspace.runtime.loadedModel?.runtimeTarget,
      workspace.runtime.loadedModel?.path,
    ].filter(Boolean);
    const threadIdentifiers = [threadModel?.modelRef, threadModel?.canonicalRepo, threadModel?.path].filter(Boolean);
    const threadModelAlreadyLoaded =
      threadIdentifiers.length > 0 &&
      threadIdentifiers.some((identifier) => loadedIdentifiers.includes(identifier));
    const shouldLoadThreadModel =
      threadModel &&
      (!workspace.runtime.loadedModel ||
        !threadModelAlreadyLoaded ||
        !activeRuntimeProfileMatchesLaunchSettings);
    const requiresPreload = shouldLoadThreadModel || (!workspace.runtime.loadedModel && Boolean(defaultChatVariant));
    // Always show the user message immediately — even when a model
    // preload is needed.  The assistant placeholder is added later
    // when streaming actually begins.
    const optimisticTurnAdded = Boolean(sendingSessionId);
    let streamStarted = false;

    const restoreComposer = () => {
      if (overrideText == null) {
        setDraftMessage(rawPrompt);
      }
      setPendingImages(pendingImagesSnapshot);
    };

    setDraftMessage("");
    setPendingImages([]);
    setChatBusySessionId(sendingSessionId);
    setError(null);

    if (optimisticTurnAdded && sendingSessionId) {
      appendOptimisticTurn(sendingSessionId, trimmed);
    }

    const { online: isOnline, startupError } = await ensureBackendAvailable(sendingSessionId ?? undefined);

    if (!isOnline) {
      const offlineMessage = startupError
        ? `The desktop shell is offline, so this local fallback kept the chat moving. Restart the FastAPI sidecar to route prompts through the runtime endpoints. Startup error: ${startupError}`
        : "The desktop shell is offline, so this local fallback kept the chat moving. Start the FastAPI sidecar to route prompts through the runtime endpoints.";
      if (startupError) {
        setError(`API service restart failed: ${startupError}`);
      }
      if (optimisticTurnAdded && sendingSessionId) {
        replaceOptimisticAssistant(sendingSessionId, trimmed, offlineMessage);
        setActiveChatId(sendingSessionId);
      } else {
        const fallbackSession = activeChat ?? {
          id: `local-${Date.now()}`,
          title: titleFromPrompt(trimmed, workspace.chatSessions),
          updatedAt: new Date().toLocaleString(),
          model: threadModel?.modelName ?? "Choose a model",
          modelRef: threadModel?.modelRef ?? null,
          canonicalRepo: threadModel?.canonicalRepo ?? null,
          modelSource: threadModel?.source ?? "catalog",
          modelPath: threadModel?.path ?? null,
          modelBackend: threadModel?.backend ?? "auto",
          thinkingMode: activeThinkingMode,
          cacheLabel: launchCacheLabel,
          cacheStrategy: launchSettings.cacheStrategy,
          cacheBits: launchSettings.cacheBits,
          fp16Layers: launchSettings.fp16Layers,
          fusedAttention: launchSettings.fusedAttention,
          fitModelInMemory: launchSettings.fitModelInMemory,
          contextTokens: launchSettings.contextTokens,
          speculativeDecoding: launchSettings.speculativeDecoding,
          dflashDraftModel: null,
          treeBudget: launchSettings.treeBudget,
          messages: [],
        };
        const updatedSession: ChatSession = {
          ...fallbackSession,
          updatedAt: new Date().toLocaleString(),
          messages: [
            ...fallbackSession.messages,
            { role: "user", text: trimmed, metrics: null },
            {
              role: "assistant",
              text: offlineMessage,
              metrics: null,
            },
          ],
        };
        setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, updatedSession) }));
        setActiveChatId(updatedSession.id);
      }
      setDraftMessage("");
      setChatBusySessionId(null);
      return;
    }

    try {
      let sessionId = activeChat?.id;
      const needsProfileReload = threadModelAlreadyLoaded && !activeRuntimeProfileMatchesLaunchSettings;

      if (shouldLoadThreadModel && threadModel) {
        const loadResult = await handleLoadModel({
          modelRef: threadModel.modelRef,
          modelName: threadModel.modelName,
          canonicalRepo: threadModel.canonicalRepo,
          source: threadModel.source,
          backend: threadModel.backend,
          path: threadModel.path,
          busyLabel: needsProfileReload ? "Reloading model for updated launch settings..." : undefined,
        });
        if (!loadResult.ok) {
          restoreComposer();
          return;
        }
      } else if (!workspace.runtime.loadedModel && defaultChatVariant) {
        const loadResult = await handleLoadModel({
          modelRef: defaultChatVariant.id,
          modelName: defaultChatVariant.name,
          canonicalRepo: defaultChatVariant.repo,
          source: "catalog",
          backend: defaultChatVariant.backend,
        });
        if (!loadResult.ok) {
          restoreComposer();
          return;
        }
      }
      if (!sessionId) {
        const session = await createSession(activeChat?.title ?? "New chat");
        setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, session) }));
        setActiveChatId(session.id);
        sessionId = session.id;
        setChatBusySessionId(session.id);
      }

      const streamPayload = {
        sessionId,
        title: threadTitleDraft.trim() || activeChat?.title,
        prompt: trimmed,
        images: pendingImagesSnapshot.length > 0 ? pendingImagesSnapshot : undefined,
        modelRef: threadModel?.modelRef,
        modelName: threadModel?.modelName,
        canonicalRepo: threadModel?.canonicalRepo,
        source: threadModel?.source,
        path: threadModel?.path,
        backend: threadModel?.backend,
        thinkingMode: activeThinkingMode,
        temperature: launchSettings.temperature,
        maxTokens: launchSettings.maxTokens,
        systemPrompt: systemPrompt || undefined,
        cacheBits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        fusedAttention: launchSettings.fusedAttention,
        cacheStrategy: launchSettings.cacheStrategy,
        fitModelInMemory: launchSettings.fitModelInMemory,
        contextTokens: launchSettings.contextTokens,
        speculativeDecoding: launchSettings.speculativeDecoding,
        treeBudget: launchSettings.treeBudget,
        enableTools: enableTools || undefined,
      };

      const streamingChatId = sessionId ?? sendingSessionId;
      if (streamingChatId && (!optimisticTurnAdded || streamingChatId !== sendingSessionId)) {
        appendOptimisticTurn(streamingChatId, trimmed);
      }

      const streamAbort = new AbortController();
      streamAbortRef.current = streamAbort;
      streamStarted = true;
      await generateChatStream(streamPayload, {
        onToken: (token) => {
          if (streamingChatId) {
            setWorkspace((current) => ({
              ...current,
              chatSessions: current.chatSessions.map((s) => {
                if (s.id !== streamingChatId) return s;
                const msgs = [...s.messages];
                const last = msgs[msgs.length - 1];
                if (last?.role === "assistant") {
                  msgs[msgs.length - 1] = { ...last, text: last.text + token };
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
        onReasoning: (reasoning) => {
          if (streamingChatId) {
            setWorkspace((current) => ({
              ...current,
              chatSessions: current.chatSessions.map((s) => {
                if (s.id !== streamingChatId) return s;
                const msgs = [...s.messages];
                const last = msgs[msgs.length - 1];
                if (last?.role === "assistant") {
                  msgs[msgs.length - 1] = {
                    ...last,
                    reasoning: `${last.reasoning ?? ""}${reasoning}`,
                    reasoningDone: false,
                  };
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
        onReasoningDone: () => {
          if (streamingChatId) {
            setWorkspace((current) => ({
              ...current,
              chatSessions: current.chatSessions.map((s) => {
                if (s.id !== streamingChatId) return s;
                const msgs = [...s.messages];
                const last = msgs[msgs.length - 1];
                if (last?.role === "assistant") {
                  msgs[msgs.length - 1] = { ...last, reasoningDone: true };
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
        onDone: (response) => {
          setWorkspace((current) =>
            syncRuntime(
              { ...current, chatSessions: upsertSession(current.chatSessions, response.session) },
              response.runtime,
            ),
          );
          setActiveChatId(response.session.id);
        },
        onError: (errMsg) => {
          setError(`Chat error: ${errMsg}`);
          if (streamingChatId) {
            setWorkspace((current) => ({
              ...current,
              chatSessions: current.chatSessions.map((s) => {
                if (s.id !== streamingChatId) return s;
                let msgs = [...s.messages];
                // Remove the empty assistant placeholder, but keep the user message
                if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant" && !msgs[msgs.length - 1].text) {
                  msgs = msgs.slice(0, -1);
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
      }, streamAbort);
    } catch (actionError) {
      // Aborted by user — not a real error
      if (actionError instanceof DOMException && actionError.name === "AbortError") {
        if (!streamStarted && sendingSessionId && optimisticTurnAdded) {
          rollbackOptimisticTurn(sendingSessionId, trimmed);
        }
        setChatBusySessionId(null);
        streamAbortRef.current = null;
        return;
      }
      if (!streamStarted) {
        if (sendingSessionId && optimisticTurnAdded) {
          rollbackOptimisticTurn(sendingSessionId, trimmed);
        }
        restoreComposer();
      }
      const detail = actionError instanceof Error ? actionError.message : "Unknown error";
      if (detail.includes("Load a model") || detail.includes("409")) {
        setError("No model is loaded. Select a model and try again.");
      } else {
        setError(`Chat error: ${detail}`);
      }
      if (sendingSessionId) {
        setWorkspace((current) => ({
          ...current,
          chatSessions: current.chatSessions.map((s) => {
            if (s.id !== sendingSessionId) return s;
            let msgs = [...s.messages];
            // Remove the empty assistant placeholder, but keep the user message
            if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant" && !msgs[msgs.length - 1].text) {
              msgs = msgs.slice(0, -1);
            }
            return { ...s, messages: msgs };
          }),
        }));
      }
    } finally {
      setChatBusySessionId(null);
      streamAbortRef.current = null;
    }
  }

  function cancelGeneration() {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    setChatBusySessionId(null);
  }

  return {
    activeChatId,
    setActiveChatId,
    threadTitleDraft,
    setThreadTitleDraft,
    draftMessage,
    setDraftMessage,
    chatBusySessionId,
    pendingImages,
    setPendingImages,
    enableTools,
    setEnableTools,
    chatScrollRef,
    sortedChatSessions,
    activeChat,
    activeThinkingMode,
    activeRuntimeProfileMatchesLaunchSettings,
    sessionModelPayload,
    persistSessionChanges,
    handleCreateSession,
    handleRenameActiveThread,
    handleToggleThreadPin,
    handleThinkingModeChange,
    handleDeleteSession,
    handleSelectThreadModel,
    handleLoadActiveThreadModel,
    handleCopyMessage,
    handleDeleteMessage,
    handleRetryMessage,
    handleChatFileDrop,
    sendMessage,
    cancelGeneration,
    deleteSessionDocument,
  };
}
