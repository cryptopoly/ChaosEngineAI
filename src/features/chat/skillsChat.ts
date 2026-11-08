/**
 * D-62: skills in Chat. The per-chat Skills switch (off by default, remembered per chat) and the count of
 * skills switched on in Settings, which decides whether the switch shows at all. Settings fires
 * SKILLS_CHANGED_EVENT after an import, a switch or a delete so an open composer follows.
 */
import { useEffect, useState } from "react";
import { listSkills } from "../../api/skills";

const KEY_PREFIX = "chaosengine.chat.skills.";
export const SKILLS_CHANGED_EVENT = "chaosengine:skills-changed";
const CHAT_SWITCH_EVENT = "chaosengine:chat-skills-switch";

export function readChatSkillsEnabled(sessionId: string | null | undefined): boolean {
  if (!sessionId) return false;
  try {
    return window.localStorage.getItem(KEY_PREFIX + sessionId) === "1";
  } catch {
    return false;
  }
}

export function writeChatSkillsEnabled(sessionId: string | null | undefined, enabled: boolean): void {
  if (!sessionId) return;
  try {
    if (enabled) window.localStorage.setItem(KEY_PREFIX + sessionId, "1");
    else window.localStorage.removeItem(KEY_PREFIX + sessionId);
  } catch {
    // storage blocked: the switch stays off
  }
  notify(CHAT_SWITCH_EVENT);
}

function notify(name: string): void {
  try {
    window.dispatchEvent(new Event(name));
  } catch {
    // no window (tests)
  }
}

export function notifySkillsChanged(): void {
  notify(SKILLS_CHANGED_EVENT);
}

/** The per-chat switch, following writes from anywhere in the app. */
export function useChatSkillsEnabled(sessionId: string | null | undefined): boolean {
  const [enabled, setEnabled] = useState(() => readChatSkillsEnabled(sessionId));
  useEffect(() => {
    const sync = () => setEnabled(readChatSkillsEnabled(sessionId));
    sync();
    window.addEventListener(CHAT_SWITCH_EVENT, sync);
    return () => window.removeEventListener(CHAT_SWITCH_EVENT, sync);
  }, [sessionId]);
  return enabled;
}

/** How many skills are switched on in Settings (0 while unknown or offline). */
export function useEnabledSkillCount(): number {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      listSkills()
        .then((skills) => {
          if (!cancelled) setCount(skills.filter((s) => s.enabled).length);
        })
        .catch(() => {
          if (!cancelled) setCount(0);
        });
    };
    load();
    window.addEventListener(SKILLS_CHANGED_EVENT, load);
    return () => {
      cancelled = true;
      window.removeEventListener(SKILLS_CHANGED_EVENT, load);
    };
  }, []);
  return count;
}
