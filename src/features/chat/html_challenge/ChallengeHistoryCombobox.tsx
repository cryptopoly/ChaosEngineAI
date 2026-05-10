/**
 * Searchable history combobox for the HTML Challenge tab.
 *
 * Renders the input + dropdown list of past challenges with per-row
 * delete buttons. The parent owns the search/open state so it can also
 * reset them when entering a new challenge or loading from history.
 */

import {
  type HtmlChallengeManifest,
  challengeHistoryLabel,
  displayChallengeTitle,
  formatChallengeDate,
  fuzzyIncludes,
} from "../htmlChallengeHelpers";

interface ChallengeHistoryComboboxProps {
  challenges: HtmlChallengeManifest[];
  selectedChallengeId: string;
  historySearch: string;
  historyOpen: boolean;
  busy: boolean;
  loadingChallengeId: string | null;
  onHistorySearchChange: (value: string) => void;
  onHistoryOpenChange: (open: boolean) => void;
  onLoadChallenge: (id: string) => void;
  onDeleteChallenge: (id: string, label: string) => void;
}

export function ChallengeHistoryCombobox({
  challenges,
  selectedChallengeId,
  historySearch,
  historyOpen,
  busy,
  loadingChallengeId,
  onHistorySearchChange,
  onHistoryOpenChange,
  onLoadChallenge,
  onDeleteChallenge,
}: ChallengeHistoryComboboxProps) {
  const selectedChallenge = challenges.find((challenge) => challenge.id === selectedChallengeId) ?? null;
  const historyInputValue = historyOpen || historySearch
    ? historySearch
    : selectedChallenge ? challengeHistoryLabel(selectedChallenge) : "";
  const filteredChallenges = challenges.filter((challenge) =>
    fuzzyIncludes(challengeHistoryLabel(challenge), historySearch),
  );

  return (
    <div className="html-challenge-history-combobox">
      <input
        className="text-input html-challenge-history-input"
        type="search"
        value={historyInputValue}
        placeholder="Search previous challenges..."
        disabled={busy || Boolean(loadingChallengeId)}
        onFocus={() => {
          onHistoryOpenChange(true);
          onHistorySearchChange("");
        }}
        onChange={(event) => {
          onHistorySearchChange(event.target.value);
          onHistoryOpenChange(true);
        }}
        onBlur={() => {
          window.setTimeout(() => {
            onHistoryOpenChange(false);
            onHistorySearchChange("");
          }, 120);
        }}
      />
      {historyOpen && !busy && !loadingChallengeId ? (
        <div className="html-challenge-history-menu" role="listbox">
          {filteredChallenges.map((challenge) => (
            <div
              key={challenge.id}
              role="option"
              aria-selected={challenge.id === selectedChallengeId}
              className={`html-challenge-history-option${challenge.id === selectedChallengeId ? " active" : ""}`}
            >
              <button
                type="button"
                className="html-challenge-history-option-main"
                onMouseDown={(event) => {
                  event.preventDefault();
                  onLoadChallenge(challenge.id);
                }}
              >
                <span>{displayChallengeTitle(challenge)}</span>
                <small>{formatChallengeDate(challenge.createdAt)}</small>
              </button>
              <button
                type="button"
                className="html-challenge-history-option-delete"
                aria-label={`Move "${displayChallengeTitle(challenge)}" to trash`}
                title="Move to trash"
                disabled={busy}
                onMouseDown={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onDeleteChallenge(challenge.id, displayChallengeTitle(challenge));
                }}
              >
                ×
              </button>
            </div>
          ))}
          {filteredChallenges.length === 0 ? (
            <p className="html-challenge-history-empty">No matching challenges.</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
