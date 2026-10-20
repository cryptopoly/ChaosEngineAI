import type { SidebarGroupId, TabId } from "../types";

export interface SidebarGroup {
  id: SidebarGroupId;
  label: string;
  caption: string;
  defaultChild: TabId;
}

export const sidebarGroups: SidebarGroup[] = [
  { id: "chat", label: "Chat", caption: "Chat, compare, and language models", defaultChild: "chat" },
  { id: "images", label: "Images", caption: "Image generation", defaultChild: "image-models" },
  { id: "video", label: "Video", caption: "Video generation", defaultChild: "video-models" },
  { id: "voice", label: "Voice", caption: "STT, TTS, and voice chat", defaultChild: "voice-studio" },
  { id: "benchmarks", label: "Benchmarks", caption: "Performance tests", defaultChild: "benchmarks" },
  { id: "tools", label: "Tools", caption: "Conversion, fine-tuning, prompts, plugins", defaultChild: "conversion" },
];
