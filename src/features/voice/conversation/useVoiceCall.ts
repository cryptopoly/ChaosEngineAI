/**
 * Conversation mode: a hands-free voice call with the loaded chat model.
 *
 * The microphone streams 16 kHz audio; every 32 ms frame gets a Silero VAD
 * probability from the backend; the endpointer decides when a turn starts
 * and ends. A finished turn is transcribed and sent as a chat message, and
 * the reply is spoken sentence by sentence while it is still being written.
 * Speaking over the model (barge-in) stops the speech and the generation.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { detectSpeech, transcribeBlob } from "../../../api";
import type { ChatMessage } from "../../../types";
import { encodeWav, micErrorMessage, pcm16Base64, startMicStream, type MicStream } from "../audio";
import { speechPlayer, useSpeechPlayer } from "../speechPlayer";
import { Endpointer, FRAME_SAMPLES } from "./endpointer";
import { SpeechTextTracker } from "./speechText";

export type CallPhase = "starting" | "listening" | "hearing" | "transcribing" | "thinking" | "speaking" | "error";

/** Up to this many frames (128 ms) go to the backend in one VAD request. */
const VAD_BATCH = 4;
/** While the model speaks, its own voice can leak into the microphone, so
 * an interruption needs clearer and longer speech; the longer look-back
 * keeps the words said before the interruption was recognised. */
const BARGE_IN = { threshold: 0.7, minSpeechMs: 256, preRollMs: 1000 };
const LISTEN = { threshold: 0.5, minSpeechMs: 160, preRollMs: 400 };

export interface VoiceCallOptions {
  active: boolean;
  messages: ChatMessage[];
  platform: string;
  settings: {
    sttModel: string;
    language: string;
    ttsVoice: string;
    speed: number;
    vadSilenceMs: number;
    bargeIn: boolean;
  };
  /** Send a voice turn; resolves when the reply has finished streaming. */
  send: (text: string) => Promise<void> | void;
  /** Ask the backend to stop the reply; ``send`` still resolves when the stream ends. */
  cancel: () => void;
}

export interface VoiceCall {
  phase: CallPhase;
  muted: boolean;
  level: number;
  error: string | null;
  /** What the user said last. */
  userText: string;
  /** The reply being spoken (or the last one). */
  replyText: string;
  interrupted: boolean;
  toggleMute: () => void;
}

export const CALL_OWNER = "voice-call";

export function useVoiceCall({ active, messages, platform, settings, send, cancel }: VoiceCallOptions): VoiceCall {
  const [phase, setPhaseState] = useState<CallPhase>("starting");
  const [muted, setMuted] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [userText, setUserText] = useState("");
  const [replyText, setReplyText] = useState("");
  const [interrupted, setInterrupted] = useState(false);
  const player = useSpeechPlayer();

  const phaseRef = useRef<CallPhase>("starting");
  const mutedRef = useRef(false);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const sendRef = useRef(send);
  sendRef.current = send;
  const cancelRef = useRef(cancel);
  cancelRef.current = cancel;
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const endpointer = useRef(new Endpointer());
  const tracker = useRef(new SpeechTextTracker());
  /** Bumped per turn so a late result from an interrupted turn is dropped. */
  const turnRef = useRef(0);
  /** Message count when the turn was sent; the reply is the message after it. */
  const sentAtRef = useRef(-1);
  const replyDoneRef = useRef(false);
  /** The send in flight; a new turn waits for an interrupted one to finish saving. */
  const sendingRef = useRef<Promise<void>>(Promise.resolve());

  const setPhase = useCallback((next: CallPhase) => {
    phaseRef.current = next;
    setPhaseState(next);
    const talking = next === "thinking" || next === "speaking";
    endpointer.current.update(talking ? BARGE_IN : LISTEN);
  }, []);

  const backToListening = useCallback(() => {
    endpointer.current.reset();
    sentAtRef.current = -1;
    setPhase("listening");
  }, [setPhase]);

  const runTurn = useCallback(
    async (audio: Float32Array) => {
      const turn = ++turnRef.current;
      setPhase("transcribing");
      const { sttModel, language, ttsVoice, speed } = settingsRef.current;
      let text = "";
      try {
        const result = await transcribeBlob(encodeWav(audio), "turn.wav", {
          model: sttModel || undefined,
          language: language || undefined,
        });
        text = result.text.trim();
      } catch (err) {
        if (turn !== turnRef.current) return;
        setError(err instanceof Error ? err.message : String(err));
        backToListening();
        return;
      }
      if (turn !== turnRef.current) return;
      if (!text) {
        backToListening();
        return;
      }
      setError(null);
      setUserText(text);
      setReplyText("");
      setInterrupted(false);
      tracker.current.reset();
      replyDoneRef.current = false;
      sentAtRef.current = -1;
      setPhase("thinking");
      await sendingRef.current;
      if (turn !== turnRef.current) return;
      sentAtRef.current = messagesRef.current.length;
      speechPlayer.startQueue(CALL_OWNER, ttsVoice, speed);
      const sending = Promise.resolve(sendRef.current(text)).catch(() => {});
      sendingRef.current = sending;
      try {
        await sending;
      } finally {
        if (turn === turnRef.current) {
          const reply = replyFor(messagesRef.current, sentAtRef.current);
          for (const piece of tracker.current.next(reply, true)) speechPlayer.enqueue(piece);
          replyDoneRef.current = true;
          speechPlayer.endQueue();
        }
      }
    },
    [backToListening, setPhase],
  );

  const onFrame = useCallback(
    (frame: Float32Array, probability: number) => {
      const current = phaseRef.current;
      if (mutedRef.current || current === "starting" || current === "transcribing" || current === "error") return;
      const talking = current === "thinking" || current === "speaking";
      if (talking && !settingsRef.current.bargeIn) return;
      const event = endpointer.current.push(frame, probability);
      if (!event) return;
      if (event.type === "start") {
        if (talking) {
          // Barge-in: stop the speech and the reply, keep what was written.
          turnRef.current += 1;
          speechPlayer.stop();
          cancelRef.current();
          setInterrupted(true);
        }
        setPhase("hearing");
        endpointer.current.update(LISTEN);
      } else {
        void runTurn(event.audio);
      }
    },
    [runTurn, setPhase],
  );

  // Microphone and VAD pump.
  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let mic: MicStream | null = null;
    const session = `call-${Date.now()}`;
    let first = true;
    let pending = new Float32Array(0);
    const frames: Float32Array[] = [];
    let inFlight = false;

    const pump = async () => {
      if (inFlight || stopped || frames.length === 0) return;
      inFlight = true;
      const batch = frames.splice(0, VAD_BATCH);
      const joined = new Float32Array(batch.length * FRAME_SAMPLES);
      batch.forEach((f, i) => joined.set(f, i * FRAME_SAMPLES));
      try {
        const probs = await detectSpeech(session, pcm16Base64(joined), first);
        first = false;
        if (!stopped) batch.forEach((f, i) => onFrame(f, probs[i] ?? 0));
      } catch (err) {
        if (!stopped) {
          setError(err instanceof Error ? err.message : String(err));
          setPhase("error");
        }
        stopped = true;
        mic?.stop();
      } finally {
        inFlight = false;
      }
      void pump();
    };

    endpointer.current.update({ silenceMs: settingsRef.current.vadSilenceMs });
    void startMicStream({
      onLevel: (value) => setLevel(mutedRef.current ? 0 : value),
      onAudio: (samples) => {
        const merged = new Float32Array(pending.length + samples.length);
        merged.set(pending);
        merged.set(samples, pending.length);
        let offset = 0;
        while (merged.length - offset >= FRAME_SAMPLES) {
          frames.push(merged.slice(offset, offset + FRAME_SAMPLES));
          offset += FRAME_SAMPLES;
        }
        pending = merged.slice(offset);
        // Never let a slow backend build up a long backlog.
        if (frames.length > 64) frames.splice(0, frames.length - 64);
        void pump();
      },
    })
      .then((stream) => {
        if (stopped) {
          stream.stop();
          return;
        }
        mic = stream;
        setPhase("listening");
      })
      .catch((err) => {
        setError(micErrorMessage(err, platform));
        setPhase("error");
      });

    return () => {
      stopped = true;
      mic?.stop();
      turnRef.current += 1;
      endpointer.current.reset();
      if (speechPlayer.getState().ownerId === CALL_OWNER) speechPlayer.stop();
    };
  }, [active, onFrame, platform, setPhase]);

  useEffect(() => {
    endpointer.current.update({ silenceMs: settings.vadSilenceMs });
  }, [settings.vadSilenceMs]);

  // Feed the reply to the speech queue as it streams in.
  useEffect(() => {
    if (sentAtRef.current < 0) return;
    const current = phaseRef.current;
    if (current !== "thinking" && current !== "speaking") return;
    const reply = replyFor(messages, sentAtRef.current);
    setReplyText(reply);
    if (replyDoneRef.current) return;
    for (const piece of tracker.current.next(reply)) speechPlayer.enqueue(piece);
  }, [messages]);

  // Speech state drives speaking -> listening.
  useEffect(() => {
    const current = phaseRef.current;
    if (current !== "thinking" && current !== "speaking") return;
    if (player.ownerId !== CALL_OWNER && player.status !== "idle") return;
    if (player.status === "playing" && current === "thinking") setPhase("speaking");
    if (player.status === "error") {
      setError(player.error);
      backToListening();
    } else if (player.status === "idle" && replyDoneRef.current) {
      backToListening();
    }
  }, [backToListening, player, setPhase]);

  const toggleMute = useCallback(() => {
    mutedRef.current = !mutedRef.current;
    setMuted(mutedRef.current);
    endpointer.current.reset();
    if (mutedRef.current && phaseRef.current === "hearing") setPhase("listening");
  }, [setPhase]);

  return { phase, muted, level, error, userText, replyText, interrupted, toggleMute };
}

/** The assistant reply that follows the message count recorded at send time. */
export function replyFor(messages: ChatMessage[], sentAt: number): string {
  if (sentAt < 0) return "";
  for (let i = messages.length - 1; i >= sentAt; i -= 1) {
    if (messages[i].role === "assistant") return messages[i].text ?? "";
    if (messages[i].role === "user") break;
  }
  return "";
}
