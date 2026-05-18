import type { Ref } from "react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { CitationBadge } from "../../components/CitationBadge";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import { PromptPhaseIndicator } from "../../components/PromptPhaseIndicator";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import { RichMarkdown } from "../../components/RichMarkdown";
import { AcceptedTokenOverlay } from "../../components/AcceptedTokenOverlay";
import { ChatPerfStrip } from "../../components/ChatPerfStrip";
import { LogprobSummary } from "../../components/LogprobSummary";
import { SubstrateRoutingBadge } from "../../components/SubstrateRoutingBadge";
import { ToolCallCard } from "../../components/ToolCallCard";
import { ChatEmptyStateBanner } from "./ChatEmptyStateBanner";
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
  /** FU-056 follow-up: when true, the empty-state CTA points the
   * user at Discover. When false (models present), it points at
   * Models. Always renders inside the empty thread when there are
   * no messages — so users on a fresh install never see a blank
   * Chat tab with no path forward. */
  noChatModelsInstalled?: boolean;
  loadedModelRef?: string | null;
  onBrowseDiscover?: () => void;
  onOpenModels?: () => void;
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
  /** Phase 3.6: re-run the message through a critique pass. */
  onDelveMessage: (messageIndex: number) => void;
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
  onDelveMessage,
  onDetailsToggle,
  onCancelGeneration,
  onLoadModel,
  noChatModelsInstalled = false,
  loadedModelRef = null,
  onBrowseDiscover,
  onOpenModels,
}: ChatThreadProps) {
  const { t } = useTranslation("chat");
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
          const fitInMemoryLabel = actualFitInMemory == null
            ? t("thread.fitInMemory.unknown", { defaultValue: "Unknown" })
            : actualFitInMemory
              ? t("thread.fitInMemory.on", { defaultValue: "On" })
              : t("thread.fitInMemory.off", { defaultValue: "Off" });
          const requestedFitInMemoryLabel = requestedFitInMemory == null
            ? null
            : requestedFitInMemory
              ? t("thread.fitInMemory.on", { defaultValue: "On" })
              : t("thread.fitInMemory.off", { defaultValue: "Off" });
          return (
            <div className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-header">
                <span className="eyebrow">
                  {message.role === "assistant"
                    ? t("thread.roleAgent", { defaultValue: "Agent" })
                    : t("thread.roleUser", { defaultValue: "User" })}
                </span>
                {!isStreamingMessage ? (
                  <div className="message-actions">
                    <button
                      type="button"
                      className="message-action-btn"
                      title={t("thread.copyMessageTooltip", { defaultValue: "Copy message" })}
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
                        title={t("thread.retryTooltip", { defaultValue: "Retry response" })}
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
                        title={t("thread.forkTooltip", { defaultValue: "Fork from here (creates a new thread)" })}
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
                    {message.role === "assistant" && index > 0 ? (
                      <button
                        type="button"
                        className="message-action-btn"
                        title={t("thread.delveTooltip", { defaultValue: "Delve — re-read with a critic's eye and propose a revised answer" })}
                        onClick={() => void onDelveMessage(index)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <circle cx="11" cy="11" r="8" />
                          <line x1="21" y1="21" x2="16.65" y2="16.65" />
                          <line x1="11" y1="8" x2="11" y2="14" />
                          <line x1="8" y1="11" x2="14" y2="11" />
                        </svg>
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="message-action-btn message-action-delete"
                      title={t("thread.deleteMessageTooltip", { defaultValue: "Delete message" })}
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
                    <strong className="panic-banner__title">
                      {t("thread.thermalThrottleTitle", { defaultValue: "Thermal throttle" })}
                    </strong>
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
                    <strong className="panic-banner__title">
                      {t("thread.memoryCriticalTitle", { defaultValue: "System memory critical" })}
                    </strong>
                    <p className="panic-banner__message">{message.panic.message}</p>
                    {message.panic.availableGb != null && message.panic.pressurePercent != null ? (
                      <small className="panic-banner__metrics">
                        {t("thread.memoryStats", {
                          available: message.panic.availableGb.toFixed(1),
                          pressure: message.panic.pressurePercent.toFixed(0),
                          defaultValue: "{available} GB free · pressure {pressure}%",
                        })}
                      </small>
                    ) : null}
                  </div>
                  {isStreamingMessage ? (
                    <button
                      className="secondary-button panic-banner__cancel"
                      type="button"
                      onClick={onCancelGeneration}
                    >
                      {t("thread.cancel", { defaultValue: "Cancel" })}
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
                    <strong>{t("thread.variants.heading", { defaultValue: "Comparing responses" })}</strong>
                    <small>{t("thread.variants.subtitle", { defaultValue: "Same prompt routed through alternate warm models." })}</small>
                  </div>
                  {message.variants.map((variant, vIdx) => (
                    <VariantCard key={`${variant.modelRef}-${vIdx}`} variant={variant} />
                  ))}
                </div>
              ) : null}
              {message.role === "assistant" && message.metrics ? (
                <div className="message-runtime-strip">
                  <SubstrateRoutingBadge metrics={message.metrics} />
                  <ChatPerfStrip metrics={message.metrics} />
                </div>
              ) : null}
              {message.role === "assistant" && message.tokenLogprobs?.length ? (
                <LogprobSummary entries={message.tokenLogprobs} />
              ) : null}
              {message.role === "assistant" && message.metrics?.acceptedSpans?.length ? (
                <AcceptedTokenOverlay metrics={message.metrics} />
              ) : null}
              {message.metrics ? (
                <details className="message-details" onToggle={(event) => void onDetailsToggle(event.currentTarget.open)}>
                  <summary>
                    <span>{t("thread.details.summary", { defaultValue: "Model details" })}</span>
                    <small className="message-meta">
                      {(message.metrics.model ?? activeChat?.model) || t("thread.details.unknown", { defaultValue: "Unknown" })} | {number(message.metrics.tokS)} tok/s
                      {message.metrics.dflashAcceptanceRate != null ? ` | ${t("thread.details.dflashAccepted", { value: number(message.metrics.dflashAcceptanceRate), defaultValue: "DFLASH {value} avg accepted" })}` : ""}
                      {messageSpeculativeMode && messageSpeculativeMode !== "Off" ? ` | ${messageSpeculativeMode}` : ""}
                      {messageRuntimeWarning ? ` | ${messageRuntimeWarning}` : ""}
                      {" | "}{number(message.metrics.responseSeconds ?? 0)} s
                    </small>
                  </summary>
                  <div className="message-detail-grid">
                    <div>
                      <span className="eyebrow">{t("thread.details.model", { defaultValue: "Model" })}</span>
                      <p>{message.metrics.model ?? activeChat?.model}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.runtime", { defaultValue: "Runtime" })}</span>
                      <p>{message.metrics.engineLabel ?? engineLabel}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.cache", { defaultValue: "Cache" })}</span>
                      <p>{resolvedCacheLabel(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.strategy", { defaultValue: "Strategy" })}</span>
                      <p>{resolvedCacheStrategy(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.cacheBits", { defaultValue: "Cache bits" })}</span>
                      <p>{resolvedCacheBits(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.fp16Layers", { defaultValue: "FP16 layers" })}</span>
                      <p>{resolvedFp16Layers(message.metrics)}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.backend", { defaultValue: "Backend" })}</span>
                      <p>{message.metrics.backend ?? activeChat?.modelBackend ?? t("thread.details.auto", { defaultValue: "Auto" })}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.context", { defaultValue: "Context" })}</span>
                      <p>{message.metrics.contextTokens?.toLocaleString() ?? launchSettings.contextTokens.toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.fitInMemory", { defaultValue: "Fit in memory" })}</span>
                      <p>{fitInMemoryLabel}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.tokens", { defaultValue: "Tokens" })}</span>
                      <p>{t("thread.details.tokensTotal", { count: message.metrics.totalTokens, defaultValue: "{count} total" })}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.responseTime", { defaultValue: "Response time" })}</span>
                      <p>{number(message.metrics.responseSeconds ?? 0)} s</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.decodeSpeed", { defaultValue: "Decode speed" })}</span>
                      <p>{number(message.metrics.tokS)} tok/s</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("thread.details.dflashDdtree", { defaultValue: "DFlash / DDTree" })}</span>
                      <p>{messageSpeculativeMode}</p>
                    </div>
                    {messageRequestedCache && messageRequestedCache !== resolvedCacheLabel(message.metrics) ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.requestedCache", { defaultValue: "Requested cache" })}</span>
                        <p>{messageRequestedCache}</p>
                      </div>
                    ) : null}
                    {requestedFitInMemoryLabel && requestedFitInMemory !== actualFitInMemory ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.requestedFit", { defaultValue: "Requested fit" })}</span>
                        <p>{requestedFitInMemoryLabel}</p>
                      </div>
                    ) : null}
                    {messageRequestedSpeculativeMode && messageRequestedSpeculativeMode !== "Off" ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.requestedDflashDdtree", { defaultValue: "Requested DFlash / DDTree" })}</span>
                        <p>{messageRequestedSpeculativeMode}</p>
                      </div>
                    ) : null}
                    {messageRuntimeWarning ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.runtimeStatus", { defaultValue: "Runtime status" })}</span>
                        <p>{messageRuntimeWarning}</p>
                      </div>
                    ) : null}
                    <div>
                      <span className="eyebrow">{t("thread.details.treeBudget", { defaultValue: "Tree budget" })}</span>
                      <p>{resolvedTreeBudget(message.metrics)}</p>
                    </div>
                    {message.metrics.dflashAcceptanceRate != null ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.dflashAcceptance", { defaultValue: "DFLASH acceptance" })}</span>
                        <p>{t("thread.details.avgTokens", { value: number(message.metrics.dflashAcceptanceRate), defaultValue: "{value} avg tokens" })}</p>
                      </div>
                    ) : null}
                    {messageDraftModel ? (
                      <div>
                        <span className="eyebrow">{t("thread.details.draftModel", { defaultValue: "Draft model" })}</span>
                        <p>{messageDraftModel}</p>
                      </div>
                    ) : null}
                  </div>
                  <button
                    className="secondary-button message-reload-settings"
                    type="button"
                    disabled={busy}
                    title={t("thread.loadExactSettingsTooltip", { defaultValue: "Load the exact model and runtime settings used for this response" })}
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
                    {t("thread.reloadSettings", { defaultValue: "Reload these settings" })}
                  </button>
                </details>
              ) : null}
            </div>
          );
        })
      ) : (
        <div className="empty-state">
          {/* FU-056 follow-up: redirect users with no installed chat
              model to Discover, and users with no loaded model to
              Models. The auto-load-largest-MLX-variant behaviour was
              both confusing on Apple Silicon (15+ GB silent download)
              and broken on Windows/Linux (MLX backend doesn't exist
              there). Banner stays visible until a model is loaded, but
              hides during the active load — the ModelLoadingProgress
              bubble below conveys state and a "Load Model" CTA next to
              a live progress bar reads as broken. */}
          {!loadedModelRef && !serverLoading && onBrowseDiscover && onOpenModels ? (
            <ChatEmptyStateBanner
              noChatModelsInstalled={noChatModelsInstalled}
              onBrowseDiscover={onBrowseDiscover}
              onOpenModels={onOpenModels}
            />
          ) : !loadedModelRef ? null : (
            <p>{t("thread.emptyState", { defaultValue: "Send a message to start the conversation." })}</p>
          )}
        </div>
      )}
      {serverLoading ? (
        <div className="message-bubble assistant">
          <span className="eyebrow">{t("thread.roleAgent", { defaultValue: "Agent" })}</span>
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
