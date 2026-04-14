export const BENCHMARK_PROMPTS: Array<{ id: string; label: string; prompt: string }> = [
  {
    id: "balanced",
    label: "Balanced summary",
    prompt: "Summarize the trade-offs of this local runtime profile for a desktop user in six concise bullets.",
  },
  {
    id: "reasoning",
    label: "Reasoning check",
    prompt: "Explain how you would choose between cache efficiency, context length, and answer quality for local inference.",
  },
  {
    id: "coding",
    label: "Coding reply",
    prompt: "Review a small backend service architecture and list practical improvements for reliability, testing, and latency.",
  },
];
