/**
 * Prompt library picker for the HTML Challenge tab (Option C layout):
 * a tab strip of categories + a free-text search box + a card grid of
 * curated single-page prompts. Selecting a card hands the full prompt
 * (and a suggested title) back to the composition root, which drops it
 * into the challenge title + prompt fields.
 */

import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  CHALLENGE_PROMPT_CATEGORIES,
  type ChallengePrompt,
  type ChallengePromptCategoryId,
  challengePromptCountByCategory,
  filterChallengePrompts,
} from "./challengePromptLibrary";

type TabId = ChallengePromptCategoryId | "all";

interface ChallengePromptLibraryModalProps {
  open: boolean;
  onSelect: (entry: ChallengePrompt) => void;
  onClose: () => void;
}

export function ChallengePromptLibraryModal({
  open,
  onSelect,
  onClose,
}: ChallengePromptLibraryModalProps) {
  const { t } = useTranslation("chat");
  const [activeTab, setActiveTab] = useState<TabId>("all");
  const [search, setSearch] = useState("");

  // Reset the tab + search each time the picker is reopened so it never
  // resurfaces stale filter state from a previous visit.
  useEffect(() => {
    if (open) {
      setActiveTab("all");
      setSearch("");
    }
  }, [open]);

  // Close on Escape while the modal is open.
  useEffect(() => {
    if (!open) {
      return;
    }
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const counts = useMemo(() => challengePromptCountByCategory(), []);
  const results = useMemo(() => filterChallengePrompts(activeTab, search), [activeTab, search]);

  if (!open) {
    return null;
  }

  const tabs: { id: TabId; label: string; count: number }[] = [
    {
      id: "all",
      label: t("htmlChallenge.promptLibrary.tabs.all", { defaultValue: "All" }),
      count: counts.games + counts.simulations + counts["tech-demos"] + counts["creative-tools"],
    },
    ...CHALLENGE_PROMPT_CATEGORIES.map((category) => ({
      id: category.id,
      label: category.label,
      count: counts[category.id],
    })),
  ];

  const categoryLabel = (id: ChallengePromptCategoryId): string =>
    CHALLENGE_PROMPT_CATEGORIES.find((category) => category.id === id)?.label ?? id;

  return (
    <div
      className="modal-overlay challenge-prompt-library"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="modal-content modal-wide"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={t("htmlChallenge.promptLibrary.title", { defaultValue: "Prompt library" })}
      >
        <div className="modal-header">
          <h3>{t("htmlChallenge.promptLibrary.title", { defaultValue: "Prompt library" })}</h3>
          <p>
            {t("htmlChallenge.promptLibrary.subtitle", {
              defaultValue: "Pick a ready-made challenge, or close to write your own.",
            })}
          </p>
        </div>

        <div className="challenge-prompt-tabs" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`challenge-prompt-tab${activeTab === tab.id ? " active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              <span className="challenge-prompt-tab-count">{tab.count}</span>
            </button>
          ))}
        </div>

        <div className="challenge-prompt-search-row">
          <input
            className="text-input"
            type="search"
            value={search}
            autoFocus
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t("htmlChallenge.promptLibrary.searchPlaceholder", {
              defaultValue: "Search prompts (name, mechanic, keyword)...",
            })}
          />
          <span className="challenge-prompt-result-count">
            {t("htmlChallenge.promptLibrary.resultCount", {
              defaultValue: "{count} shown",
              count: results.length,
            })}
          </span>
        </div>

        <div className="modal-body challenge-prompt-body">
          {results.length === 0 ? (
            <p className="muted-text challenge-prompt-empty">
              {t("htmlChallenge.promptLibrary.empty", {
                defaultValue: "No prompts match your search.",
              })}
            </p>
          ) : (
            <div className="challenge-prompt-grid">
              {results.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  className="challenge-prompt-card"
                  onClick={() => onSelect(entry)}
                  title={entry.prompt}
                >
                  <span className="challenge-prompt-card-title">{entry.title}</span>
                  <span className="challenge-prompt-card-category">{categoryLabel(entry.category)}</span>
                  <span className="challenge-prompt-card-summary">{entry.summary}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button type="button" className="secondary-button" onClick={onClose}>
            {t("htmlChallenge.promptLibrary.close", { defaultValue: "Cancel" })}
          </button>
        </div>
      </div>
    </div>
  );
}
