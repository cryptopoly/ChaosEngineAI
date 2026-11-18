import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../api";
import { mergeEvents, type ModelEvent } from "../features/server/modelEvents";

export interface ModelEventsState {
  events: ModelEvent[];
  connected: boolean;
  paused: boolean;
  /** Events that arrived while paused; shown on Resume. */
  pending: number;
  pause: () => void;
  resume: () => void;
}

/** Parse complete server-sent events out of ``buffer``; returns the events and the unfinished rest. */
export function parseSseChunk(buffer: string): { events: ModelEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ModelEvent[] = [];
  for (const block of blocks) {
    const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    try {
      const parsed = JSON.parse(dataLine.slice(6)) as ModelEvent;
      if (parsed && typeof parsed.id === "number") events.push(parsed);
    } catch {
      // a malformed line is skipped
    }
  }
  return { events, rest };
}

/**
 * Structured model events (D-88): the backlog and then a live tail from
 * ``/api/server/events/stream`` while ``enabled``. Reconnects after 3 s.
 * Pause holds new events back and counts them.
 */
export function useModelEvents(enabled: boolean): ModelEventsState {
  const [events, setEvents] = useState<ModelEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [pending, setPending] = useState(0);
  const pausedRef = useRef(false);
  const heldRef = useRef<ModelEvent[]>([]);

  const accept = useCallback((incoming: ModelEvent[]) => {
    if (incoming.length === 0) return;
    if (pausedRef.current) {
      heldRef.current = mergeEvents(heldRef.current, incoming);
      setPending(heldRef.current.length);
      return;
    }
    setEvents((current) => mergeEvents(current, incoming));
  }, []);

  const pause = useCallback(() => {
    pausedRef.current = true;
    setPaused(true);
  }, []);

  const resume = useCallback(() => {
    pausedRef.current = false;
    setPaused(false);
    const held = heldRef.current;
    heldRef.current = [];
    setPending(0);
    if (held.length) setEvents((current) => mergeEvents(current, held));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let abort: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const schedule = () => {
      if (cancelled || timer) return;
      timer = setTimeout(() => {
        timer = null;
        void connect();
      }, 3000);
    };

    async function connect() {
      if (cancelled) return;
      const controller = new AbortController();
      abort = controller;
      try {
        const response = await apiFetch("/api/server/events/stream", { signal: controller.signal });
        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
        setConnected(true);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parsed = parseSseChunk(buffer);
          buffer = parsed.rest;
          accept(parsed.events);
        }
      } catch {
        // reconnect below
      } finally {
        if (!cancelled) setConnected(false);
        abort = null;
        schedule();
      }
    }

    void connect();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      abort?.abort();
      setConnected(false);
    };
  }, [enabled, accept]);

  return { events, connected, paused, pending, pause, resume };
}
