/**
 * The tool cards of a reply, and the bubble for a Studio turn this window does not stream (S-J, FU-208): a turn
 * that went on after a reload, or a Keep going nudge. The bubble shows the plan cards it waits on and the reply so
 * far (text, reasoning, tool cards) from the detached-turn poll, with the same cards as a streamed turn.
 */
import { useTranslation } from "react-i18next";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import { RichMarkdown } from "../../components/RichMarkdown";
import { ToolCallCard } from "../../components/ToolCallCard";
import type { ToolCallInfo } from "../../types";
import type { DetachedTurn } from "./detachedTurn";
import { StudioJobCard } from "./StudioJobCard";
import { StudioPlanCard } from "./StudioPlanCard";
import { mergeToolCalls, refusalReason, studioJobId } from "./studioChat";

export function ToolCallList({ calls }: { calls: ToolCallInfo[] }) {
  return (
    <div style={{ margin: "4px 0" }}>
      {mergeToolCalls(calls).map((tc) =>
        tc.renderAs === "studio" && (studioJobId(tc) || tc.data?.declined || refusalReason(tc)) ? (
          <StudioJobCard key={tc.id} toolCall={tc} />
        ) : (
          <ToolCallCard key={tc.id} toolCall={tc} />
        ),
      )}
    </div>
  );
}

export function DetachedTurnBubble({ turn, sessionId }: { turn: DetachedTurn; sessionId: string | undefined }) {
  const { t } = useTranslation("chat");
  const partial = turn.partial;
  const shown = Boolean(partial && (partial.text || partial.reasoning || partial.toolCalls.length));
  return (
    <div className="message-bubble assistant message-bubble--detached" data-testid="detached-turn">
      <div className="message-header">
        <span className="eyebrow">{t("thread.roleAgent", { defaultValue: "Agent" })}</span>
      </div>
      {partial?.reasoning ? <ReasoningPanel text={partial.reasoning} streaming={!partial.text} /> : null}
      {turn.plans.length || !shown ? (
        <p className="message-detached-note" role="status">
          {turn.plans.length
            ? t("studioChat.detached.waiting", { defaultValue: "This reply is waiting for your answer to the plan below." })
            : t("studioChat.detached.running", { defaultValue: "This reply is still being made. It shows here when it is done." })}
        </p>
      ) : null}
      {partial?.text ? (
        <div className="markdown-content streaming-cursor">
          <RichMarkdown>{partial.text}</RichMarkdown>
        </div>
      ) : null}
      {turn.plans.length ? (
        <div className="studio-plan-stack">
          {turn.plans.map((plan) => (
            <StudioPlanCard key={plan.planId} plan={plan} sessionId={sessionId}
              onEditByChat={() => document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus()} />
          ))}
        </div>
      ) : null}
      {partial?.toolCalls.length ? <ToolCallList calls={partial.toolCalls} /> : null}
    </div>
  );
}
