/**
 * How long the chat waits in "prompt_eval" (no first token) before it gives up and cancels the turn.
 *
 * A plain turn streams, so 60 s without a token means a stuck prefill. A tool turn (tools, Studio tools or skills)
 * does not stream its rounds: the backend sends the reply only after the model's whole first round, thinking
 * included, which on a long thread with a large model takes well over a minute (S-P: Qwen3.8-27B, "OK, go ahead"
 * after a cut plan was cancelled at 60 s and the reply came back empty). No watchdog then: Stop stays available.
 */
export const PROMPT_EVAL_TIMEOUT_MS = 60_000;

export function promptEvalTimeoutMs(turn: { enableTools?: boolean; studioTools?: boolean; skills?: boolean }): number | null {
  return turn.enableTools || turn.studioTools || turn.skills ? null : PROMPT_EVAL_TIMEOUT_MS;
}
