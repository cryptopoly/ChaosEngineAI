/**
 * The "Rework with the model" conversation about one shot (S8), shared by the Plan tab's shot row and the freestyle
 * board's card chat (S-M2). The model answers with a new prompt and why; `use` applies only the fields it changed,
 * as one edit (Undo works). The state lives in the caller, so the chat survives closing its panel.
 */
import { useState } from "react";
import { reworkStoryboardShot, type StoryboardChatTurn, type StoryboardRework } from "../../api/storyboardPlan";
import type { StoryboardShot } from "../../types";
import { reworkEdit } from "./planLogic";
import { failureOf, useElapsed } from "./usePlan";
import type { UseStoryboardResult } from "./useStoryboard";

export type ReworkTurn = StoryboardChatTurn & { suggestion?: StoryboardRework; applied?: boolean };

export interface ReworkState {
  turns: ReworkTurn[];
  ask: string;
  setAsk: (value: string) => void;
  pending: boolean;
  error: string | null;
  elapsed: number;
  send: () => Promise<void>;
  use: (index: number, suggestion: StoryboardRework) => Promise<void>;
  /** The newest suggestion not used yet. */
  waiting: ReworkTurn | undefined;
}

export function useRework(
  story: UseStoryboardResult,
  projectId: string,
  shot: StoryboardShot,
  canAsk: boolean,
  onModelMissing: () => void,
): ReworkState {
  const [turns, setTurns] = useState<ReworkTurn[]>([]);
  const [ask, setAsk] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const elapsed = useElapsed(pending);

  async function send() {
    const message = ask.trim();
    if (!message || pending) return;
    if (!canAsk) {
      onModelMissing();
      return;
    }
    const history: StoryboardChatTurn[] = turns.map(({ role, text }) => ({ role, text }));
    setTurns((current) => [...current, { role: "user", text: message }]);
    setAsk("");
    setPending(true);
    setError(null);
    try {
      const answer = await reworkStoryboardShot(projectId, shot.id, message, history);
      setTurns((current) => [...current, { role: "assistant", text: answer.why, suggestion: answer }]);
    } catch (err) {
      const failure = failureOf(err);
      if (failure.code === "no-model") onModelMissing();
      setError(failure.message);
    } finally {
      setPending(false);
    }
  }

  async function use(index: number, suggestion: StoryboardRework) {
    const args = reworkEdit(shot, suggestion.shot);
    if (Object.keys(args).length > 1 && !(await story.op("edit_shot", args))) return;
    setTurns((current) => current.map((turn, i) => (i === index ? { ...turn, applied: true } : turn)));
  }

  const waiting = [...turns].reverse().find((turn) => turn.suggestion && !turn.applied);
  return { turns, ask, setAsk, pending, error, elapsed, send, use, waiting };
}
