/**
 * Per-slot result panel — reasoning, stream, rendered iframe / code view,
 * file actions, repair / retry / change-model buttons.
 *
 * The composition root keeps ownership of frame / shell / stream refs
 * (passed in as attach functions) and of the actual run/retry/repair
 * orchestration.
 */

import { type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { ReasoningPanel } from "../../../components/ReasoningPanel";
import { compareTargetLabels, type CompareTarget } from "../CompareView";
import {
  type ChallengeSlot,
  type ChallengeSlotState,
  type HtmlChallengeManifest,
  type HtmlChallengeManifestSlot,
  htmlValidationForState,
  htmlValidationLabels,
  validationBadgeClass,
  validationMessage,
} from "../htmlChallengeHelpers";
import {
  BrowserIcon,
  CollapseIcon,
  ExpandIcon,
  OpenFileIcon,
} from "../htmlChallengeIcons";
import { highlightHtmlCode, previewSrcDoc } from "../htmlChallengeMarkup";
import { fileActionPath } from "./htmlChallengeTabHelpers";

interface ChallengeSlotPanelProps {
  slot: ChallengeSlot;
  state: ChallengeSlotState;
  manifest: HtmlChallengeManifest | null;
  manifestSlot?: HtmlChallengeManifestSlot;
  modelLabel: string;
  subtitle: string;
  waitingLabel: string;
  busy: boolean;
  isExpanded: boolean;
  showLatestButton: boolean;
  retryable: boolean;
  repairable: boolean;
  hasRetryPayload: boolean;
  canChangeModel: boolean;
  isCodeView: boolean;
  previewBackground: string | null;
  fileRevealLabel: string;
  settingsSummary: string;
  onSetExpanded: (slotId: CompareTarget | null) => void;
  onScrollStreamToBottom: (slotId: CompareTarget) => void;
  onToggleCodeView: (slotId: CompareTarget) => void;
  onChangeModel: (slotId: CompareTarget) => void;
  onRetrySlot: () => void;
  onRepairSlot: (mode: "continue" | "repair") => void;
  onRevealPath: (path: string) => void;
  onOpenFilePath: (path: string) => void;
  onAttachStream: (slotId: CompareTarget, element: HTMLPreElement | null) => void;
  onAttachFrame: (slotId: CompareTarget, element: HTMLIFrameElement | null) => void;
  onAttachFrameShell: (slotId: CompareTarget, element: HTMLDivElement | null) => void;
  onStreamScroll: (slotId: CompareTarget) => void;
  onMarkPreviewActive: (slotId: CompareTarget) => void;
  onFocusPreviewFrame: (slotId: CompareTarget) => void;
  onForwardPreviewKey: (slotId: CompareTarget, event: ReactKeyboardEvent<HTMLElement>) => void;
}

export function ChallengeSlotPanel({
  slot,
  state,
  manifest,
  manifestSlot,
  modelLabel,
  subtitle,
  waitingLabel,
  busy,
  isExpanded,
  showLatestButton,
  retryable,
  repairable,
  hasRetryPayload,
  canChangeModel,
  isCodeView,
  previewBackground,
  fileRevealLabel,
  settingsSummary,
  onSetExpanded,
  onScrollStreamToBottom,
  onToggleCodeView,
  onChangeModel,
  onRetrySlot,
  onRepairSlot,
  onRevealPath,
  onOpenFilePath,
  onAttachStream,
  onAttachFrame,
  onAttachFrameShell,
  onStreamScroll,
  onMarkPreviewActive,
  onFocusPreviewFrame,
  onForwardPreviewKey,
}: ChallengeSlotPanelProps) {
  const { t } = useTranslation("chat");
  const validation = htmlValidationForState(state);
  const actionPath = fileActionPath(state, manifest?.folderPath);

  const renderValidationBadge = () => {
    if (!validation) return null;
    const status = validation.status;
    const message = validationMessage(validation);
    return (
      <span
        className={`badge ${validationBadgeClass(status)}`}
        title={message || validation.label || htmlValidationLabels[status]}
      >
        {validation.label || htmlValidationLabels[status]}
      </span>
    );
  };

  const renderFileActions = () => {
    const validationBadge = renderValidationBadge();
    const canExpand = Boolean(state.html && validation?.status !== "no-html");
    const canViewCode = Boolean(state.html && validation?.status !== "no-html");
    if (!state.filename && !actionPath && !validationBadge && !canExpand) return null;
    return (
      <div className="html-challenge-file-row">
        {state.filename ? <span className="badge success">{state.filename}</span> : null}
        {validationBadge}
        {actionPath ? (
          <>
            <button
              className="secondary-button html-challenge-icon-button"
              type="button"
              aria-label={fileRevealLabel}
              title={fileRevealLabel}
              onClick={() => onRevealPath(actionPath)}
            >
              <OpenFileIcon />
            </button>
            <button
              className="secondary-button html-challenge-icon-button"
              type="button"
              aria-label={t("htmlChallenge.openBrowser", { defaultValue: "Open in default browser" })}
              title={t("htmlChallenge.openBrowser", { defaultValue: "Open in default browser" })}
              onClick={() => onOpenFilePath(actionPath)}
            >
              <BrowserIcon />
            </button>
          </>
        ) : null}
        {canExpand ? (
          <button
            className="secondary-button html-challenge-icon-button"
            type="button"
            aria-label={isExpanded ? "Collapse preview" : "Expand preview"}
            title={isExpanded ? "Collapse preview" : "Expand preview"}
            onClick={() => onSetExpanded(isExpanded ? null : slot.id)}
          >
            {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
          </button>
        ) : null}
        {canViewCode ? (
          <button
            className="secondary-button html-challenge-code-toggle"
            type="button"
            onClick={() => onToggleCodeView(slot.id)}
          >
            {isCodeView ? "View Render" : "View HTML Code"}
          </button>
        ) : null}
      </div>
    );
  };

  const panelActions = isExpanded || showLatestButton || retryable || canChangeModel ? (
    <div className="html-challenge-panel-actions">
      {isExpanded ? (
        <button className="secondary-button" type="button" onClick={() => onSetExpanded(null)}>
          Collapse
        </button>
      ) : null}
      {showLatestButton ? (
        <button className="secondary-button" type="button" onClick={() => onScrollStreamToBottom(slot.id)}>
          Latest
        </button>
      ) : null}
      {canChangeModel ? (
        <button className="secondary-button" type="button" disabled={busy} onClick={() => onChangeModel(slot.id)}>
          Change Model
        </button>
      ) : null}
      {retryable ? (
        <>
          {repairable ? (
            <>
              <button
                className="secondary-button"
                type="button"
                disabled={busy || !manifest?.id || !hasRetryPayload}
                onClick={() => onRepairSlot("continue")}
              >
                Continue HTML
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={busy || !manifest?.id || !hasRetryPayload}
                onClick={() => onRepairSlot("repair")}
              >
                Repair HTML
              </button>
            </>
          ) : null}
          <button
            className="secondary-button"
            type="button"
            disabled={busy || !manifest?.id || !hasRetryPayload}
            onClick={onRetrySlot}
          >
            Retry
          </button>
        </>
      ) : null}
    </div>
  ) : null;

  return (
    <Panel
      title={compareTargetLabels[slot.id]}
      subtitle={subtitle}
      className={`html-challenge-preview-panel${isExpanded ? " html-challenge-preview-panel--expanded" : ""}`}
      actions={panelActions}
    >
      <div className="html-challenge-panel-body">
        {modelLabel ? (
          <div className="html-challenge-meta">
            <strong>{modelLabel}</strong>
            <span className="html-challenge-settings-summary">{settingsSummary}</span>
          </div>
        ) : null}
        <ReasoningPanel
          className="html-challenge-reasoning"
          text={state.reasoning}
          streaming={!state.reasoningDone}
        />
        {state.error ? (
          <p style={{ color: "#f87171" }}>{state.error}</p>
        ) : state.deleted ? (
          <div className="html-challenge-deleted">
            <strong>File deleted</strong>
            <span>{state.filename ?? "The saved HTML file"} is no longer in this challenge folder.</span>
            <div className="html-challenge-file-row">
              {state.filename ? <span className="badge warning">{state.filename}</span> : null}
              {manifest?.folderPath ? (
                <button
                  className="secondary-button html-challenge-file-button"
                  type="button"
                  onClick={() => onRevealPath(manifest.folderPath)}
                >
                  Open Folder
                </button>
              ) : null}
            </div>
          </div>
        ) : state.done && !state.html ? (
          <div className="html-challenge-empty-result">
            <strong>No HTML output</strong>
            <span>{validationMessage(validation) || "This model finished without a renderable HTML page."}</span>
            {renderFileActions()}
          </div>
        ) : state.html && validation?.status === "no-html" ? (
          <div className="html-challenge-empty-result">
            <strong>No HTML output</strong>
            <span>{validationMessage(validation) || "The saved output did not contain a standalone HTML document."}</span>
            {renderFileActions()}
          </div>
        ) : state.html ? (
          <>
            {renderFileActions()}
            {isCodeView ? (
              <pre className="html-challenge-stream html-challenge-code-view">
                <code dangerouslySetInnerHTML={{ __html: highlightHtmlCode(state.html) }} />
              </pre>
            ) : (
              <div
                ref={(element) => onAttachFrameShell(slot.id, element)}
                className={`html-challenge-frame-shell${isExpanded ? " html-challenge-frame-shell--expanded" : ""}`}
                style={previewBackground ? { background: previewBackground } : undefined}
                tabIndex={0}
                onPointerEnter={() => onMarkPreviewActive(slot.id)}
                onPointerDownCapture={() => onMarkPreviewActive(slot.id)}
                onMouseDownCapture={() => onFocusPreviewFrame(slot.id)}
                onKeyDown={(event) => onForwardPreviewKey(slot.id, event)}
                onKeyUp={(event) => onForwardPreviewKey(slot.id, event)}
              >
                {/* Iframe fills the shell directly so the page sees the
                    actual rendered pixel size as its viewport — the same
                    way a desktop browser hands its window to a page on
                    resize. No fixed 1280x720 stage, no transform scaling. */}
                <iframe
                  ref={(element) => onAttachFrame(slot.id, element)}
                  className="html-challenge-frame"
                  title={`${compareTargetLabels[slot.id]} HTML preview`}
                  srcDoc={previewSrcDoc(state.html, slot.id)}
                  sandbox="allow-scripts"
                  scrolling="yes"
                  tabIndex={0}
                  /* `allowTransparency` is a legacy WebKit attribute that
                     lets the iframe's default document canvas show the
                     frame's CSS background instead of opaque white. Pair
                     it with `:where(html, body) { background: transparent }`
                     inside the doc and `background: transparent` on the
                     iframe element so the frame-shell colour shows through
                     any region the model HTML doesn't paint. */
                  // @ts-expect-error legacy attribute, still honored by WebKit
                  allowtransparency="true"
                  onFocus={() => onMarkPreviewActive(slot.id)}
                />
              </div>
            )}
          </>
        ) : state.text ? (
          <pre
            ref={(element) => onAttachStream(slot.id, element)}
            className="html-challenge-stream"
            onScroll={() => onStreamScroll(slot.id)}
          >
            <code dangerouslySetInnerHTML={{ __html: highlightHtmlCode(state.text) }} />
          </pre>
        ) : state.loading ? (
          <p className="muted-text">{state.loadingMessage ?? "Loading model..."}</p>
        ) : busy ? (
          <p className="muted-text">{waitingLabel}</p>
        ) : null}
      </div>
    </Panel>
  );
}
