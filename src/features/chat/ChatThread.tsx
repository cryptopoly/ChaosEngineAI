import type { Ref } from "react";
import { useState } from "react";
import { CitationBadge } from "../../components/CitationBadge";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import { PromptPhaseIndicator } from "../../components/PromptPhaseIndicator";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import { RichMarkdown } from "../../components/RichMarkdown";
import { ToolCallCard } from "../../components/ToolCallCard";
import type { ChatSession, ChatMessageVariant, LaunchPreferences, ModelLoadingState, WarmModel } from "../../types";
import { number } from "../../utils";
import { VariantPickerButton } from "./VariantPickerButton";
import {
  requestedCacheLabel,
  requestedSpeculativeMode,
  resolvedCacheBits,
  resolvedCacheLabel,
  resolvedCacheStrategy,
  resolvedDraftModel,
  resolvedFp16Layers,
  resolvedSpeculativeMode,
  resolvedTreeBudget,
  runtimeOutcomeWarning,
} from "./runtimeDetails";

/**
 * Phase 2.1: extracted from ChatTab.tsx. Renders the streaming message
 * list including assistant reasoning panels, prompt-phase indicator,
 * panic / thermal banners, tool calls, citations, the per-turn metrics
 * fold-out, and the model-loading placeholder. Drag-drop on the scroll
 * container forwards files via `onChatFileDrop`.
 */
export interface ChatThreadProps {
  activeChat: ChatSession | undefined;
  chatBusySessionId: string | null;
  chatScrollRef: Ref<HTMLDivElement>;
  serverLoading: ModelLoadingState | null;
  engineLabel: string;
  launchSettings: LaunchPreferences;
  busy: boolean;
  onChatFileDrop: (files: FileList) => void;
  onCopyMessage: (text: string) => void;
  onRetryMessage: (index: number) => void;
  onDeleteMessage: (index: number) => void;
  /** Phase 2.4: fork-from-here action on assistant messages. */
  onForkAtMessage: (index: number) => void;
  /** Phase 2.5: warm models available for variant generation. */
  warmModels: WarmModel[];
  /** Phase 2.5: kick off variant generation against an alternate model. */
  onAddVariant: (messageIndex: number, warm: WarmModel) => void;
  onDetailsToggle: (opened: boolean) => void;
  onCancelGeneration: () => void;
  onLoadModel: (payload: {
    modelRef: string;
    modelName?: string;
    canonicalRepo?: string | null;
    source?: string;
    backend?: string;
    path?: string;
    busyLabel?: string;
    cacheStrategy?: string;
    cacheBits?: number;
    fp16Layers?: number;
    fusedAttention?: boolean;
    fitModelInMemory?: boolean;
    contextTokens?: number;
    speculativeDecoding?: boolean;
    treeBudget?: number;
  }) => void;
}

export function ChatThread({
  activeChat,
  chatBusySessionId,
  chatScrollRef,
  serverLoading,
  engineLabel,
  launchSettings,
  busy,
  onChatFileDrop,
  onCopyMessage,
  onRetryMessage,
  onDeleteMessage,
  onForkAtMessage,
  warmModels,
  onAddVariant,
  onDetailsToggle,
  onCancelGeneration,
  onLoadModel,
}: ChatThreadProps) {
  return (
    <div
      className="message-list message-scroll"
      ref={chatScrollRef}
      onDragOver={(event) => {
        event.preventDefault();
        event.currentTarget.classList.add("drag-over");
      }}
      onDragLeave={(event) => {
        event.currentTarget.classList.remove("drag-over");
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.currentTarget.classList.remove("drag-over");
        if (event.dataTransfer?.files) {
          void onChatFileDrop(event.dataTransfer.files);
        }
      }}
    >
      {activeChat?.messages.length ? (
        activeChat.messages.map((message, index) => {
          const isStreamingMessage = chatBusySessionId === activeChat?.id && index === activeChat.messages.length - 1 && !message.metrics;
          const messageSpeculativeMode = message.metrics ? resolvedSpeculativeMode(message.metrics) : null;
          const messageDraftModel = message.metrics ? resolvedDraftModel(message.metrics) : null;
          const messageRequestedCache = message.metrics ? requestedCacheLabel(message.metrics) : null;
          const messageRequestedSpeculativeMode = message.metrics ? requestedSpeculativeMode(message.metrics) : null;
          const messageRuntimeWarning = message.metrics ? runtimeOutcomeWarning(message.metrics) : null;
          const actualFitInMemory = message.metrics?.fitModelInMemory;
          const requestedFitInMemory = message.metrics?.requestedFitModelInMemory;
          const fitInMemoryLabel = actualFitInMemory == null ? "Unknown" : actualFitInMemory ? "On" : "Off";
          const requestedFitInMemoryLabel = requestedFitInMemory == null ? null : requestedFitInMemory ? "On" : "Off";
          return (
            <div className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-header">
                <span className="eyebrow">{message.role === "assistant" ? "Agent" : "User"}</span>
                {!isStreamingMessage ? (
                  <div className="message-actions">
                    <button
                      type="button"
                      className="message-action-btn"
                      title="Copy message"
                      onClick={() => onCopyMessage(message.text)}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </svg>
                    </button>
                    {message.role === "assistant" ? (
                      <button
                        type="button"
                        className="message-action-btn"
                        title="Retry response"
                        onClick={() => void onRetryMessage(index)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="23 4 23 10 17 10" />
                          <polyline points="1 20 1 14 7 14" />
                          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                      </button>
                    ) : null}
                    {message.role === "assistant" ? (
                      <button
                        type="button"
                        className="message-action-btn"
                        title="Fork from here (creates a new thread)"
                        onClick={() => void onForkAtMessage(index)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <circle cx="6" cy="3" r="2" />
                          <circle cx="6" cy="21" r="2" />
                          <circle cx="18" cy="6" r="2" />
                          <path d="M6 5v14" />
                          <path d="M6 12c0-3 6-3 12-6" />
                        </svg>
                      </button>
                    ) : null}
                    {message.role === "assistant" && warmModels.length > 1 ? (
                      <VariantPickerButton
                        warmModels={warmModels}
                        currentModelRef={message.metrics?.modelRef ?? activeChat?.modelRef ?? null}
                        onPick={(warm) => onAddVariant(index, warm)}
                      />
                    ) : null}
                    <button
                      type="button"
                      className="message-action-btn message-action-delete"
                      title="Delete message"
                      onClick={() => onDeleteMessage(index)}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        <line x1="10" y1="11" x2="10" y2="17" />
                        <line x1="14" y1="11" x2="14" y2="17" />
                      </svg>
                    </button>
                  </div>
                ) : null}
              </div>
              {message.role === "assistant" ? (
                <ReasoningPanel
                  text={message.reasoning}
                  streaming={isStreamingMessage && message.reasoningDone !== true}
                />
              ) : null}
              {message.role === "assistant" && isStreamingMessage && message.streamPhase ? (
                <PromptPhaseIndicator phase={message.streamPhase} />
              ) : null}
              {message.role === "assistant" && message.thermalWarning ? (
                <div className={`panic-banner panic-banner--thermal panic-banner--${message.thermalWarning.state}`} role="alert">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
                  </svg>
                  <div className="panic-banner__body">
                    <strong className="panic-banner__title">Thermal throttle</strong>
                    <p className="panic-banner__message">{message.thermalWarning.message}</p>
                  </div>
                </div>
              ) : null}
              {message.role === "assistant" && message.panic ? (
                <div className="panic-banner" role="alert">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                  <div className="panic-banner__body">
                    <strong className="panic-banner__title">System memory critical</strong>
                    <p className="panic-banner__message">{message.panic.message}</p>
                    {message.panic.availableGb != null && message.panic.pressurePercent != null ? (
                      <small className="panic-banner__metrics">
                        {message.panic.availableGb.toFixed(1)} GB free · pressure {message.panic.pressurePercent.toFixed(0)}%
                      </small>
                    ) : null}
                  </div>
                  {isStreamingMessage ? (
                    <button
                      className="secondary-button panic-banner__cancel"
                      type="button"
                      onClick={onCancelGeneration}
                    >
                      Cancel
                    </button>
                  ) : null}
                </div>
              ) : null}
              {message.role === "assistant" ? (
                <div className={`markdown-content${isStreamingMessage && !message.streamPhase ? " streaming-cursor" : ""}`}>
                  <RichMarkdown>{message.text || "​"}</RichMarkdown>
                </div>
              ) : (
                <p>{message.text}</p>
              )}
              {message.toolCalls?.length ? (
                <div style={{ margin: "4px 0" }}>
                  {message.toolCalls.map((tc) => (
                    <ToolCallCard key={tc.id} toolCall={tc} />
                  ))}
                </div>
              ) : null}
              {message.citations?.length ? (
                <CitationBadge citations={message.citations} />
              ) : null}
              {message.role === "assistant" && message.variants?.length ? (
                <div className="variant-stack">
                  <div className="variant-stack__heading">
                    <strong>Comparing responses</strong>
                    <small>Same prompt routed through alternate warm models.</small>
                  </div>
                  {message.variants.map((variant, vIdx) => (
                    <VariantCard key={`${variant.modelRef}-${vIdx}`} variant={variant} />
                  ))}
                </div>
              ) : null}
              {message.metrics ? (
                <details className="message-details" onToggle={(event) => void onDetailsToggle(event.currentTarget.open)}>
                  <summary>
                    <span>Model details</span>
                    <small className="message-meta">
                      {(message.metrics.model ?? activeChat?.model) || "Unknown"} | {number(message.metrics.tokS)} tok/s
                      {message.metrics.dflashAcceptanceRate != null ? ` | DFLASH ${number(message.metrics.dflashAcceptanceRate)} avg accepted` : ""}
                      {messageSpeculativeMode && messageSpeculativeMode !== "Off" ? ` | ${messageSpeculativeMode}` : ""}
                      {messageRuntimeWarning ? ` | ${messageRuntimeWarning}` : ""}
                      {" | "}{number(message.metrics.responseSeconds ?? 0)} s
                    </small>
                  </summary>
                  <div className="message-detail-grid">
                    <div>
                      <span className="eyebrow">Model</span>
                      <p>{message.metrics.model ?? activeChat?.model}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Runtime</span>
                      <p>{message.metrics.engineLabel ?? engineLabel}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Cache</span>
                      <p>{resolvedCacheLabel(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Strategy</span>
                      <p>{resolvedCacheStrategy(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Cache bits</span>
                      <p>{resolvedCacheBits(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">FP16 layers</span>
                      <p>{resolvedFp16Layers(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Backend</span>
                      <p>{message.metrics.backend ?? activeChat?.modelBackend ?? "Auto"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Context</span>
                      <p>{message.metrics.contextTokens?.toLocaleString() ?? launchSettings.contextTokens.toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Fit in memory</span>
                      <p>{fitInMemoryLabel}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Tokens</span>
                      <p>{message.metrics.totalTokens} total</p>
                    </div>
                    <div>
                      <span className="eyebrow">Response time</span>
                      <p>{number(message.metrics.responseSeconds ?? 0)} s</p>
                    </div>
                    <div>
                      <span className="eyebrow">Decode speed</span>
                      <p>{number(message.metrics.tokS)} tok/s</p>
                    </div>
                    <div>
                      <span className="eyebrow">DFlash / DDTree</span>
                      <p>{messageSpeculativeMode}</p>
                    </div>
                    {messageRequestedCache && messageRequestedCache !== resolvedCacheLabel(message.metrics) ? (
                      <div>
                        <span className="eyebrow">Requested cache</span>
                        <p>{messageRequestedCache}</p>
                      </div>
                    ) : null}
                    {requestedFitInMemoryLabel && requestedFitInMemory !== actualFitInMemory ? (
                      <div>
                        <span className="eyebrow">Requested fit</span>
                        <p>{requestedFitInMemoryLabel}</p>
                      </div>
                    ) : null}
                    {messageRequestedSpeculativeMode && messageRequestedSpeculativeMode !== "Off" ? (
                      <div>
                        <span className="eyebrow">Requested DFlash / DDTree</span>
                        <p>{messageRequestedSpeculativeMode}</p>
                      </div>
                    ) : null}
                    {messageRuntimeWarning ? (
                      <div>
                        <span className="eyebrow">Runtime status</span>
                        <p>{messageRuntimeWarning}</p>
                      </div>
                    ) : null}
                    <div>
                      <span className="eyebrow">Tree budget</span>
                      <p>{resolvedTreeBudget(message.metrics)}</p>
                    </div>
                    {message.metrics.dflashAcceptanceRate != null ? (
                      <div>
                        <span className="eyebrow">DFLASH acceptance</span>
                        <p>{number(message.metrics.dflashAcceptanceRate)} avg tokens</p>
                      </div>
                    ) : null}
                    {messageDraftModel ? (
                      <div>
                        <span className="eyebrow">Draft model</span>
                        <p>{messageDraftModel}</p>
                      </div>
                    ) : null}
                  </div>
                  <button
                    className="secondary-button message-reload-settings"
                    type="button"
                    disabled={busy}
                    title="Load the exact model and runtime settings used for this response"
                    onClick={() => {
                      const ref = message.metrics!.modelRef ?? activeChat?.modelRef;
                      if (!ref) return;
                      void onLoadModel({
                        modelRef: ref,
                        modelName: message.metrics!.model ?? activeChat?.model,
                        canonicalRepo: message.metrics!.canonicalRepo ?? activeChat?.canonicalRepo ?? null,
                        source: message.metrics!.modelSource ?? activeChat?.modelSource ?? "library",
                        backend: message.metrics!.backend ?? activeChat?.modelBackend ?? "auto",
                        path: message.metrics!.modelPath ?? activeChat?.modelPath ?? undefined,
                        cacheStrategy: message.metrics!.cacheStrategy ?? activeChat?.cacheStrategy ?? undefined,
                        cacheBits: message.metrics!.cacheBits ?? activeChat?.cacheBits ?? undefined,
                        fp16Layers: message.metrics!.fp16Layers ?? activeChat?.fp16Layers ?? undefined,
                        fusedAttention: message.metrics!.fusedAttention ?? activeChat?.fusedAttention ?? undefined,
                        fitModelInMemory: message.metrics!.fitModelInMemory ?? activeChat?.fitModelInMemory ?? undefined,
                        contextTokens: message.metrics!.contextTokens ?? activeChat?.contextTokens ?? undefined,
                        speculativeDecoding: message.metrics!.speculativeDecoding ?? activeChat?.speculativeDecoding ?? undefined,
                        treeBudget: message.metrics!.treeBudget ?? activeChat?.treeBudget ?? undefined,
                      });
                    }}
                  >
                    Reload these settings
                  </button>
                </details>
              ) : null}
            </div>
          );
        })
      ) : (
        <div className="empty-state">
          <p>Send a message to start the conversation.</p>
        </div>
      )}
      {serverLoading ? (
        <div className="message-bubble assistant">
          <span className="eyebrow">Agent</span>
          <div className="model-loading-chat">
            <ModelLoadingProgress loading={serverLoading} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Phase 2.5: renders a single sibling response under the primary
 * assistant bubble. Includes the model name, decode tok/s if known,
 * the response markdown, and a collapsible reasoning panel when
 * the model emitted thinking tokens.
 */
function VariantCard({ variant }: { variant: ChatMessageVariant }) {
  const tokS = variant.metrics?.tokS;
  const responseSeconds = variant.metrics?.responseSeconds;
  return (
    <div className="variant-card">
      <div className="variant-card__header">
        <span className="variant-card__model">{variant.modelName}</span>
        {tokS != null ? (
          <small className="variant-card__metric">{number(tokS)} tok/s</small>
        ) : null}
        {responseSeconds != null ? (
          <small className="variant-card__metric">{number(responseSeconds)} s</small>
        ) : null}
      </div>
      {variant.reasoning ? (
        <ReasoningPanel text={variant.reasoning} streaming={false} />
      ) : null}
      <div className="markdown-content">
        <RichMarkdown>{variant.text || "​"}</RichMarkdown>
      </div>
    </div>
  );
}
