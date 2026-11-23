/**
 * Chat Models on first run (D-105 D, FU-457): with no chat model on disk the page opens on a short list of the
 * catalogue's recommended rows for this computer (``Recommendation.chatPicks``, the same pick and fit the
 * Dashboard shows) instead of two screens of filters. The full list with every filter is one click away.
 */

import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import {
  FitBadge,
  ModelDownloadControls,
  type ResolvedModelStatus,
} from "../../components/models";
import { ModelDownloadProgress } from "../../components/models/ModelDownloadProgress";
import type { ChatModelEntry } from "../../components/models/adapters/chat";
import type { ChatPick } from "../../types";
import { pickSummary } from "../dashboard/chatPick";
import { chatLaunchActions, type ChatLaunchAction } from "./ChatModelsPage";

export interface ChatRecommendedListProps {
  /** Catalog rows of the picks, best first, with the pick each came from. */
  rows: Array<{ entry: ChatModelEntry; pick: ChatPick }>;
  resolveStatus: (entry: ChatModelEntry) => ResolvedModelStatus;
  /** Rows the full list would show on this computer. */
  totalCount: number;
  hardwareLabel: string | null;
  gpu: boolean;
  onShowAll: () => void;
  onDownloadModel: (repo: string) => void;
  onCancelModelDownload: (repo: string) => void;
  onDeleteModelDownload: (repo: string) => void;
  onOpenModelSelector: (action: ChatLaunchAction, preselectedKey?: string) => void;
}

export type PickBadge = "fastest" | "newest" | "popular";

/** Simple picks over the shown rows: the one clearly fastest, newest and most downloaded. Needs at least two
 *  rows with a value and a strict leader, so a tie or a single row earns no badge. Release dates are
 *  `YYYY-MM` (or longer); only the month is compared. */
export function pickBadges(picks: ChatPick[]): Map<string, PickBadge[]> {
  const out = new Map<string, PickBadge[]>();
  const award = (badge: PickBadge, scored: Array<{ id: string; score: number }>) => {
    if (scored.length < 2) return;
    const sorted = [...scored].sort((a, b) => b.score - a.score);
    if (sorted[0].score > sorted[1].score) out.set(sorted[0].id, [...(out.get(sorted[0].id) ?? []), badge]);
  };
  award(
    "fastest",
    picks.flatMap((p) => (p.tokS && p.tokS > 0 ? [{ id: p.variantId, score: p.tokS }] : [])),
  );
  award(
    "newest",
    picks.flatMap((p) => {
      const match = /^(\d{4})-(\d{2})/.exec(p.releaseDate ?? "");
      return match ? [{ id: p.variantId, score: Number(match[1]) * 12 + Number(match[2]) }] : [];
    }),
  );
  award(
    "popular",
    picks.flatMap((p) => (p.downloads && p.downloads > 0 ? [{ id: p.variantId, score: p.downloads }] : [])),
  );
  return out;
}

export function ChatRecommendedList({
  rows,
  resolveStatus,
  totalCount,
  hardwareLabel,
  gpu,
  onShowAll,
  onDownloadModel,
  onCancelModelDownload,
  onDeleteModelDownload,
  onOpenModelSelector,
}: ChatRecommendedListProps) {
  const { t } = useTranslation("library");
  const { t: tDashboard } = useTranslation("dashboard");
  const { t: tCommon } = useTranslation("common");
  const showAll = (
    <button className="secondary-button" type="button" onClick={onShowAll}>
      {t("models.chat.recommended.showAll", { defaultValue: "Show all {count} models and filters", count: totalCount })}
    </button>
  );
  const badges = pickBadges(rows.map(({ pick }) => pick));
  const popularTitle = (pick: ChatPick) =>
    t("models.chat.recommended.popularTitle", {
      defaultValue: "{count} downloads in the last 30 days on Hugging Face (as of {date})",
      count: pick.downloads ?? 0,
      date: (pick.statsAsOf ?? "").slice(0, 10),
    });
  return (
    <div className="models-page chat-models-page chat-recommended">
      <Panel
        title={tCommon("tabs.chatModels", { defaultValue: "Chat Models" })}
        subtitle={
          hardwareLabel
            ? t("models.chat.recommended.subtitle", {
                defaultValue: "Recommended for this computer ({hardware})",
                hardware: hardwareLabel,
              })
            : t("models.chat.recommended.subtitleBare", { defaultValue: "Recommended for this computer" })
        }
        actions={showAll}
        className="models-page__panel"
      >
        <p className="muted-text">
          {t("models.chat.recommended.intro", {
            defaultValue:
              "No chat model is downloaded yet. These fit this computer best: download one, then open Chat. The first is the Dashboard's pick.",
          })}
        </p>
      </Panel>
      <ol className="chat-recommended__list">
        {rows.map(({ entry, pick }, index) => {
          const status = resolveStatus(entry);
          const repo = status.download?.repo ?? entry.downloadKey;
          const { launch, overflow } = chatLaunchActions(entry, onOpenModelSelector, t);
          return (
            <li key={entry.id} className="chat-recommended__row" data-testid="chat-recommended-row">
              <div className="chat-recommended__main">
                <div className="chat-recommended__title">
                  <strong>{entry.name}</strong>
                  {index === 0 ? (
                    <span className="badge accent">{t("models.chat.recommended.best", { defaultValue: "Best fit" })}</span>
                  ) : null}
                  {(badges.get(pick.variantId) ?? []).map((badge) => (
                    <span
                      key={badge}
                      className="badge"
                      data-testid={`chat-recommended-${badge}`}
                      title={badge === "popular" ? popularTitle(pick) : undefined}
                    >
                      {badge === "fastest"
                        ? t("models.chat.recommended.fastest", { defaultValue: "Fastest" })
                        : badge === "newest"
                          ? t("models.chat.recommended.newest", { defaultValue: "Newest" })
                          : t("models.chat.recommended.popular", { defaultValue: "Most popular" })}
                    </span>
                  ))}
                  <FitBadge fit={entry.fit} />
                </div>
                <p className="muted-text">{pickSummary(pick, gpu, tDashboard)}</p>
                {entry.note ? <p className="chat-recommended__note">{entry.note}</p> : null}
                <ModelDownloadProgress status={status} />
                {status.state === "failed" && status.download?.error ? (
                  <p className="callout error">{status.download.error}</p>
                ) : null}
              </div>
              <ModelDownloadControls
                status={status}
                canDownload={Boolean(repo)}
                onDownload={() => onDownloadModel(repo)}
                onPause={() => onCancelModelDownload(repo)}
                onCancel={() => onDeleteModelDownload(repo)}
                launch={launch}
                overflow={overflow}
              />
            </li>
          );
        })}
      </ol>
      <div className="button-row chat-recommended__footer">{showAll}</div>
    </div>
  );
}
