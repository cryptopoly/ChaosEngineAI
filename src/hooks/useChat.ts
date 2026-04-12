import { useEffect, useRef, useState } from "react";
import {
  checkBackend,
  createSession,
  deleteSession,
  deleteSessionDocument,
  generateChatStream,
  uploadSessionDocument,
  updateSession,
} from "../api";
import { mockWorkspace } from "../mockData";
import {
  upsertSession,
  sortSessions,
  titleFromPrompt,
  syncRuntime,
} from "../utils";
import type {
  ChatSession,
  LaunchPreferences,
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
    source?: string;
    backend?: string;
    path?: string;
    nextTab?: TabId;
  }) => Promise<void>,
  defaultChatVariant: ModelVariant | null,
  threadModelOptions: ChatModelOption[],
  launchCacheLabel: string,
  loadedModelCacheLabel: string,
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>,
) {
  const [activeChatId, setActiveChatId] = useState(mockWorkspace.chatSessions[0]?.id ?? "");
  const [threadTitleDraft, setThreadTitleDraft] = useState(mockWorkspace.chatSessions[0]?.title ?? "");
  const [draftMessage, setDraftMessage] = useState("");
  const [chatBusySessionId, setChatBusySessionId] = useState<string | null>(null);
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const sortedChatSessions = sortSessions(workspace.chatSessions);
  const activeChat = workspace.chatSessions.find((session) => session.id === activeChatId) ?? sortedChatSessions[0];

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
        source: session.modelSource ?? "catalog",
        path: session.modelPath ?? undefined,
        backend: session.modelBackend ?? "auto",
      };
    }
    if (workspace.runtime.loadedModel) {
      return {
        modelRef: workspace.runtime.loadedModel.ref,
        modelName: workspace.runtime.loadedModel.name,
        source: workspace.runtime.loadedModel.source,
        path: workspace.runtime.loadedModel.path ?? undefined,
        backend: workspace.runtime.loadedModel.backend,
      };
    }
    if (defaultChatVariant) {
      return {
        modelRef: defaultChatVariant.id,
        modelName: defaultChatVariant.name,
        source: "catalog",
        backend: defaultChatVariant.backend,
      };
    }
    return null;
  }

  function mergeSessionMetadata(session: ChatSession, patch: Partial<ChatSession>): ChatSession {
    return { ...session, ...patch };
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
        modelSource: patch.modelSource,
        modelPath: patch.modelPath,
        modelBackend: patch.modelBackend,
        pinned: patch.pinned,
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
      const localSession: ChatSession = {
        id: `local-${Date.now()}`,
        title: "New chat",
        updatedAt: new Date().toLocaleString(),
        pinned: false,
        model: fallbackModel?.modelName ?? "Choose a model",
        modelRef: fallbackModel?.modelRef ?? null,
        modelSource: fallbackModel?.source ?? "catalog",
        modelPath: fallbackModel?.path ?? null,
        modelBackend: fallbackModel?.backend ?? "auto",
        cacheLabel:
          workspace.runtime.loadedModel ? loadedModelCacheLabel : launchCacheLabel,
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

    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((s) =>
        s.id === activeChat.id ? { ...s, messages: s.messages.slice(0, userIdx) } : s,
      ),
    }));
    setDraftMessage(userText);
    await new Promise((r) => setTimeout(r, 50));
    await sendMessage();
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

  async function sendMessage() {
    const trimmed = draftMessage.trim();
    if (!trimmed) return;

    setDraftMessage("");
    if (activeChat) {
      setWorkspace((current) => ({
        ...current,
        chatSessions: current.chatSessions.map((s) =>
          s.id === activeChat.id
            ? { ...s, messages: [...s.messages, { role: "user" as const, text: trimmed, metrics: null }], updatedAt: new Date().toLocaleString() }
            : s,
        ),
      }));
    }

    const sendingSessionId = activeChat?.id ?? null;
    setChatBusySessionId(sendingSessionId);
    setError(null);

    const threadModel = sessionModelPayload(activeChat);

    const isOnline = backendOnline || await checkBackend();
    if (isOnline && !backendOnline) setBackendOnline(true);

    if (!isOnline) {
      const fallbackSession = activeChat ?? {
        id: `local-${Date.now()}`,
        title: titleFromPrompt(trimmed),
        updatedAt: new Date().toLocaleString(),
        model: threadModel?.modelName ?? "Choose a model",
        modelRef: threadModel?.modelRef ?? null,
        modelSource: threadModel?.source ?? "catalog",
        modelPath: threadModel?.path ?? null,
        modelBackend: threadModel?.backend ?? "auto",
        cacheLabel: launchCacheLabel,
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
            text: "The desktop shell is offline, so this local fallback kept the chat moving. Start the FastAPI sidecar to route prompts through the runtime endpoints.",
            metrics: null,
          },
        ],
      };
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, updatedSession) }));
      setActiveChatId(updatedSession.id);
      setDraftMessage("");
      setChatBusySessionId(null);
      return;
    }

    try {
      let sessionId = activeChat?.id;
      const shouldLoadThreadModel =
        threadModel &&
        (!workspace.runtime.loadedModel ||
          ![workspace.runtime.loadedModel.ref, workspace.runtime.loadedModel.runtimeTarget].includes(threadModel.modelRef));

      if (shouldLoadThreadModel && threadModel) {
        await handleLoadModel({
          modelRef: threadModel.modelRef,
          modelName: threadModel.modelName,
          source: threadModel.source,
          backend: threadModel.backend,
          path: threadModel.path,
        });
      } else if (!workspace.runtime.loadedModel && defaultChatVariant) {
        await handleLoadModel({
          modelRef: defaultChatVariant.id,
          modelName: defaultChatVariant.name,
          source: "catalog",
          backend: defaultChatVariant.backend,
        });
      }
      if (!sessionId) {
        const session = await createSession(activeChat?.title ?? "New chat");
        setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, session) }));
        setActiveChatId(session.id);
        sessionId = session.id;
      }

      const imagesToSend = pendingImages.length > 0 ? [...pendingImages] : undefined;
      setPendingImages([]);

      const streamPayload = {
        sessionId,
        title: threadTitleDraft.trim() || activeChat?.title,
        prompt: trimmed,
        images: imagesToSend,
        modelRef: threadModel?.modelRef,
        modelName: threadModel?.modelName,
        source: threadModel?.source,
        path: threadModel?.path,
        backend: threadModel?.backend,
        temperature: launchSettings.temperature,
        maxTokens: launchSettings.maxTokens,
        systemPrompt: systemPrompt || undefined,
        cacheBits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        fusedAttention: launchSettings.fusedAttention,
        cacheStrategy: launchSettings.cacheStrategy,
        fitModelInMemory: launchSettings.fitModelInMemory,
        contextTokens: launchSettings.contextTokens,
      };

      const streamingChatId = sessionId ?? sendingSessionId;
      if (streamingChatId) {
        setWorkspace((current) => ({
          ...current,
          chatSessions: current.chatSessions.map((s) =>
            s.id === streamingChatId
              ? { ...s, messages: [...s.messages, { role: "assistant" as const, text: "", metrics: null }] }
              : s,
          ),
        }));
      }

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
                if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant" && !msgs[msgs.length - 1].text) {
                  msgs = msgs.slice(0, -1);
                }
                if (msgs.length > 0 && msgs[msgs.length - 1].role === "user" && msgs[msgs.length - 1].text === trimmed) {
                  msgs = msgs.slice(0, -1);
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
      });
      setDraftMessage("");
    } catch (actionError) {
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
            if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant" && !msgs[msgs.length - 1].text) {
              msgs = msgs.slice(0, -1);
            }
            if (msgs.length > 0 && msgs[msgs.length - 1].role === "user" && msgs[msgs.length - 1].text === trimmed) {
              msgs = msgs.slice(0, -1);
            }
            return { ...s, messages: msgs };
          }),
        }));
      }
    } finally {
      setChatBusySessionId(null);
    }
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
    chatScrollRef,
    sortedChatSessions,
    activeChat,
    sessionModelPayload,
    persistSessionChanges,
    handleCreateSession,
    handleRenameActiveThread,
    handleToggleThreadPin,
    handleDeleteSession,
    handleSelectThreadModel,
    handleLoadActiveThreadModel,
    handleCopyMessage,
    handleDeleteMessage,
    handleRetryMessage,
    handleChatFileDrop,
    sendMessage,
    deleteSessionDocument,
  };
}
