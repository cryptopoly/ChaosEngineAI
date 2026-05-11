/**
 * Header / setup section of the HTML Challenge tab — title +
 * prompt inputs, history dropdown, layout toggle, file actions, and
 * the primary action button (Run / Cancel / Use Prompt).
 *
 * The composition root keeps run/retry orchestration and full state
 * ownership; this is just the chrome around it.
 */

import { useTranslation } from "react-i18next";
import {
  type HtmlChallengeLayoutMode,
  type HtmlChallengeManifest,
  stackedLayoutLabel,
} from "../htmlChallengeHelpers";
import { ChallengeHistoryCombobox } from "./ChallengeHistoryCombobox";

interface ChallengeSetupPanelProps {
  title: string;
  prompt: string;
  busy: boolean;
  manifest: HtmlChallengeManifest | null;
  challenges: HtmlChallengeManifest[];
  selectedChallengeId: string;
  historySearch: string;
  historyOpen: boolean;
  loadingChallengeId: string | null;
  layoutMode: HtmlChallengeLayoutMode;
  slotCount: number;
  completedChallenge: boolean;
  completedValidChallenge: boolean;
  canRunChallenge: boolean;
  onTitleChange: (value: string) => void;
  onPromptChange: (value: string) => void;
  onLayoutModeChange: (mode: HtmlChallengeLayoutMode) => void;
  onHistorySearchChange: (value: string) => void;
  onHistoryOpenChange: (open: boolean) => void;
  onLoadChallenge: (id: string) => void;
  onDeleteChallenge: (id: string, label: string) => void;
  onNewChallenge: () => void;
  onAddSlot: () => void;
  onCancelChallenge: () => void;
  onRunChallenge: () => void;
  onUsePromptInNewChallenge: () => void;
  onRevealPath: (path: string) => void;
  onOpenFilePath: (path: string) => void;
}

export function ChallengeSetupPanel({
  title,
  prompt,
  busy,
  manifest,
  challenges,
  selectedChallengeId,
  historySearch,
  historyOpen,
  loadingChallengeId,
  layoutMode,
  slotCount,
  completedChallenge,
  completedValidChallenge,
  canRunChallenge,
  onTitleChange,
  onPromptChange,
  onLayoutModeChange,
  onHistorySearchChange,
  onHistoryOpenChange,
  onLoadChallenge,
  onDeleteChallenge,
  onNewChallenge,
  onAddSlot,
  onCancelChallenge,
  onRunChallenge,
  onUsePromptInNewChallenge,
  onRevealPath,
  onOpenFilePath,
}: ChallengeSetupPanelProps) {
  const { t } = useTranslation("chat");
  return (
    <section className="panel html-challenge-setup-panel html-challenge-setup-panel--compact">
      <div className="html-challenge-setup-actions">
        {challenges.length > 0 ? (
          <div className="html-challenge-history-row">
            <button
              className="secondary-button"
              type="button"
              disabled={busy || (!manifest && !selectedChallengeId)}
              onClick={onNewChallenge}
            >
              New Challenge
            </button>
            <ChallengeHistoryCombobox
              challenges={challenges}
              selectedChallengeId={selectedChallengeId}
              historySearch={historySearch}
              historyOpen={historyOpen}
              busy={busy}
              loadingChallengeId={loadingChallengeId}
              onHistorySearchChange={onHistorySearchChange}
              onHistoryOpenChange={onHistoryOpenChange}
              onLoadChallenge={onLoadChallenge}
              onDeleteChallenge={onDeleteChallenge}
            />
          </div>
        ) : null}
        <div className="html-challenge-setup-actions-spacer" />
        <div className="html-challenge-layout-toggle" aria-label="HTML challenge layout">
          <button
            className={layoutMode === "row" ? "active" : ""}
            type="button"
            onClick={() => onLayoutModeChange("row")}
          >
            Row
          </button>
          <button
            className={layoutMode === "stacked" ? "active" : ""}
            type="button"
            onClick={() => onLayoutModeChange("stacked")}
          >
            {stackedLayoutLabel(slotCount)}
          </button>
        </div>
        {manifest?.folderPath ? (
          <button
            className="secondary-button"
            type="button"
            onClick={() => onRevealPath(manifest.folderPath)}
          >
            Open Folder
          </button>
        ) : null}
        {manifest?.settingsPath ? (
          <button
            className="secondary-button"
            type="button"
            onClick={() => onOpenFilePath(manifest.settingsPath!)}
          >
            Open Settings
          </button>
        ) : null}
        {!completedChallenge ? (
          <button className="secondary-button" type="button" onClick={onAddSlot} disabled={busy || slotCount >= 4}>
            Add model
          </button>
        ) : null}
        {busy ? (
          <button className="secondary-button" type="button" onClick={onCancelChallenge}>Cancel</button>
        ) : completedValidChallenge ? (
          <button
            className="primary-button"
            type="button"
            onClick={onUsePromptInNewChallenge}
          >
            Use Prompt in New Challenge
          </button>
        ) : (
          <button
            className="primary-button"
            type="button"
            onClick={onRunChallenge}
            disabled={!canRunChallenge}
          >
            {manifest ? "Run New Challenge" : "Run Challenge"}
          </button>
        )}
      </div>
      <div className="html-challenge-controls">
        <input
          className="text-input"
          type="text"
          value={title}
          onChange={(event) => onTitleChange(event.target.value)}
          placeholder={t("htmlChallenge.titlePlaceholder", { defaultValue: "Challenge title" })}
          disabled={busy}
        />
        <textarea
          className="text-input html-challenge-prompt"
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder={t("htmlChallenge.promptPlaceholder", { defaultValue: "Prompt all selected models with the same webpage challenge..." })}
          disabled={busy}
        />
      </div>
    </section>
  );
}
