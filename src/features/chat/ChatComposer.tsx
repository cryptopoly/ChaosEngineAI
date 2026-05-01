import type { Dispatch, SetStateAction } from "react";
import { TemperatureChip } from "../../components/TemperatureChip";
import type { ChatSession, ChatThinkingMode, LaunchPreferences } from "../../types";
import type { SlashCommand } from "./slashCommands";

/**
 * Phase 2.1: extracted from ChatTab.tsx. The composer area — image
 * previews, slash-command popover, textarea, attach / thinking effort /
 * tools / send / stop buttons, plus the per-thread temperature chip.
 *
 * Slash-menu state and the temperature override are owned by the
 * parent (ChatTab) so the data flow stays unidirectional and so other
 * consumers (e.g. the upcoming compare view) can reuse the chip
 * without re-implementing the localStorage glue.
 */
export type ReasoningEffortLevel = "low" | "medium" | "high";

export interface ChatComposerProps {
  draftMessage: string;
  pendingImages: string[];
  loadedModelRef: string | undefined;
  thinkingMode: ChatThinkingMode;
  reasoningEffort: ReasoningEffortLevel;
  enableTools: boolean;
  chatBusySessionId: string | null;
  activeChat: ChatSession | undefined;
  launchSettings: LaunchPreferences;
  temperatureOverride: number | null;
  showSlashMenu: boolean;
  slashMatches: SlashCommand[];
  slashIndex: number;
  setSlashIndex: Dispatch<SetStateAction<number>>;
  onDraftMessageChange: (message: string) => void;
  onPendingImagesChange: Dispatch<SetStateAction<string[]>>;
  onSendMessage: () => void;
  onCancelGeneration: () => void;
  onClearDraft: () => void;
  onChatFileDrop: (files: FileList) => void;
  onToggleTools: (enabled: boolean) => void;
  onSetError: (msg: string | null) => void;
  onTemperatureOverrideChange: (value: number | null) => void;
  runSlashCommand: (cmd: SlashCommand) => void;
  handleEffortOff: () => void;
  handleEffortChange: (level: ReasoningEffortLevel) => void;
}

export function ChatComposer({
  draftMessage,
  pendingImages,
  loadedModelRef,
  thinkingMode,
  reasoningEffort,
  enableTools,
  chatBusySessionId,
  activeChat,
  launchSettings,
  temperatureOverride,
  showSlashMenu,
  slashMatches,
  slashIndex,
  setSlashIndex,
  onDraftMessageChange,
  onPendingImagesChange,
  onSendMessage,
  onCancelGeneration,
  onClearDraft,
  onChatFileDrop,
  onToggleTools,
  onSetError,
  onTemperatureOverrideChange,
  runSlashCommand,
  handleEffortOff,
  handleEffortChange,
}: ChatComposerProps) {
  return (
    <div className="composer">
      {pendingImages.length > 0 ? (
        <div className="composer-image-previews">
          {pendingImages.map((img, i) => (
            <div key={i} className="composer-image-thumb">
              <img src={`data:image/png;base64,${img}`} alt={`Attachment ${i + 1}`} />
              <button
                className="composer-image-remove"
                type="button"
                onClick={() => onPendingImagesChange((prev) => prev.filter((_, j) => j !== i))}
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      ) : null}
      <div className="composer-input-wrap">
        {showSlashMenu ? (
          <div className="slash-command-menu" role="listbox" aria-label="Slash commands">
            {slashMatches.map((cmd, idx) => (
              <button
                key={cmd.command}
                type="button"
                role="option"
                aria-selected={idx === slashIndex}
                className={`slash-command-menu__item${idx === slashIndex ? " slash-command-menu__item--active" : ""}`}
                onMouseEnter={() => setSlashIndex(idx)}
                onClick={() => runSlashCommand(cmd)}
              >
                <span className="slash-command-menu__command">{cmd.command}</span>
                <span className="slash-command-menu__desc">{cmd.description}</span>
              </button>
            ))}
          </div>
        ) : null}
        <textarea
          className="text-area"
          placeholder={
            loadedModelRef
              ? "Type a message... (Enter to send, Shift+Enter for new line, / for commands)"
              : "Load a model first — pick one from My Models or Discover, then hit CHAT."
          }
          rows={3}
          value={draftMessage}
          onChange={(event) => onDraftMessageChange(event.target.value)}
          onKeyDown={(event) => {
            if (showSlashMenu) {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setSlashIndex((current) => (current + 1) % slashMatches.length);
                return;
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setSlashIndex((current) => (current - 1 + slashMatches.length) % slashMatches.length);
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                const target = slashMatches[slashIndex];
                if (target) runSlashCommand(target);
                return;
              }
              if (event.key === "Escape") {
                event.preventDefault();
                onDraftMessageChange("");
                return;
              }
              if (event.key === "Tab") {
                event.preventDefault();
                const target = slashMatches[slashIndex];
                if (target) onDraftMessageChange(`${target.command} `);
                return;
              }
            }
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              // Mirror the Send button's disabled state — no-op when no
              // model is loaded so users don't trigger a confusing 500.
              if (!loadedModelRef) return;
              void onSendMessage();
            }
          }}
          onDrop={(event) => {
            const files = event.dataTransfer?.files;
            if (!files?.length) return;
            event.preventDefault();
            void onChatFileDrop(files);
          }}
          onDragOver={(event) => event.preventDefault()}
        />
      </div>
      <div className="button-row composer-button-row">
        <div className="composer-button-group composer-button-group--left">
          <label className="secondary-button composer-attach-btn" title="Attach image">
            <input
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(event) => {
                const files = event.target.files;
                if (!files) return;
                for (const file of Array.from(files)) {
                  if (file.size > 10 * 1024 * 1024) { onSetError("Image must be under 10MB"); continue; }
                  const reader = new FileReader();
                  reader.onload = () => {
                    const b64 = (reader.result as string).split(",")[1];
                    if (b64) onPendingImagesChange((prev) => [...prev, b64]);
                  };
                  reader.readAsDataURL(file);
                }
                event.target.value = "";
              }}
            />
            {"📎"}
          </label>
          <div
            className="composer-mode-control"
            title="Choose how much reasoning the model performs before answering. Off = direct answers; Low / Medium / High = increasing reasoning depth for capable models."
          >
            <span className="composer-mode-label">Thinking</span>
            <div className="thread-mode-toggle composer-thinking-toggle" role="group" aria-label="Thinking mode">
              <button
                type="button"
                className={`thread-mode-button${thinkingMode === "off" ? " thread-mode-button--active" : ""}`}
                disabled={chatBusySessionId === activeChat?.id}
                onClick={handleEffortOff}
                title="No reasoning — model answers directly"
              >
                Off
              </button>
              <button
                type="button"
                className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "low" ? " thread-mode-button--active" : ""}`}
                disabled={chatBusySessionId === activeChat?.id}
                onClick={() => handleEffortChange("low")}
                title="Brief reasoning"
              >
                Low
              </button>
              <button
                type="button"
                className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "medium" ? " thread-mode-button--active" : ""}`}
                disabled={chatBusySessionId === activeChat?.id}
                onClick={() => handleEffortChange("medium")}
                title="Default reasoning depth"
              >
                Med
              </button>
              <button
                type="button"
                className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === "high" ? " thread-mode-button--active" : ""}`}
                disabled={chatBusySessionId === activeChat?.id}
                onClick={() => handleEffortChange("high")}
                title="Extended reasoning"
              >
                High
              </button>
            </div>
          </div>
          <TemperatureChip
            defaultValue={launchSettings.temperature}
            override={temperatureOverride}
            onChange={onTemperatureOverrideChange}
            disabled={chatBusySessionId === activeChat?.id}
          />
          <button
            className={`secondary-button${enableTools ? " active-toggle" : ""}`}
            type="button"
            onClick={() => onToggleTools(!enableTools)}
            title={enableTools ? "Tools enabled (web search, code, calculator, file reader)" : "Enable agent tools"}
            style={{
              background: enableTools ? "#1e3a5f" : undefined,
              borderColor: enableTools ? "#3b82f6" : undefined,
              color: enableTools ? "#8fb4ff" : undefined,
              fontSize: 12,
              padding: "4px 10px",
            }}
          >
            {enableTools ? "Tools ON" : "Tools"}
          </button>
        </div>
        <div className="composer-button-group composer-button-group--right">
          <button className="secondary-button" type="button" onClick={onClearDraft}>
            Clear
          </button>
          {chatBusySessionId !== null ? (
            <button className="secondary-button" type="button" onClick={onCancelGeneration} style={{ background: "#7f1d1d", borderColor: "#dc2626", color: "#fca5a5" }}>
              Stop
            </button>
          ) : (
            <button
              className="primary-button"
              type="button"
              onClick={() => void onSendMessage()}
              disabled={!loadedModelRef}
              title={!loadedModelRef ? "Load a model first to send messages" : undefined}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
