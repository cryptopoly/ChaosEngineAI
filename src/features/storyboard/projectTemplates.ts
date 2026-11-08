/**
 * Start-from templates for a new project (D-65): client-side data only. A template fills the format and a few
 * titled shots with an example still prompt; nothing else. Blank is the default, and "Scene / b-roll" has no
 * character at all: a cast is optional everywhere.
 */
import type { StoryboardExportPreset, StoryboardResolution } from "../../types";

export interface TemplateShot {
  /** i18n key under `templates.shots.` and its English text. */
  key: string;
  title: string;
  stillPrompt: string;
  seconds: number;
}

export interface ProjectTemplate {
  id: "blank" | "ad30" | "vertical" | "story" | "broll";
  label: string;
  aspect: StoryboardExportPreset;
  resolution: StoryboardResolution;
  fps: number;
  shots: TemplateShot[];
}

export const PROJECT_TEMPLATES: ProjectTemplate[] = [
  { id: "blank", label: "Blank", aspect: "16:9", resolution: "720p", fps: 24, shots: [] },
  {
    id: "ad30",
    label: "30 s ad",
    aspect: "16:9",
    resolution: "1080p",
    fps: 30,
    shots: [
      { key: "adHook", title: "Hook", stillPrompt: "A striking close-up that stops the scroll", seconds: 5 },
      { key: "adProblem", title: "The problem", stillPrompt: "Someone struggling with an everyday task", seconds: 5 },
      { key: "adProduct", title: "The product", stillPrompt: "The product on a clean table in soft light", seconds: 5 },
      { key: "adBenefit", title: "The benefit", stillPrompt: "The same task made easy, a relaxed smile", seconds: 5 },
      { key: "adProof", title: "Proof", stillPrompt: "Several happy people using it in different places", seconds: 5 },
      { key: "adCall", title: "Call to action", stillPrompt: "The product with empty space for a message", seconds: 5 },
    ],
  },
  {
    id: "vertical",
    label: "Vertical social",
    aspect: "9:16",
    resolution: "1080p",
    fps: 30,
    shots: [
      { key: "socialHook", title: "Hook", stillPrompt: "A bold, close moment filling the frame", seconds: 3 },
      { key: "socialShow", title: "Show it", stillPrompt: "The main idea shown in one clear picture", seconds: 4 },
      { key: "socialEnd", title: "Payoff", stillPrompt: "A satisfying final reveal", seconds: 3 },
    ],
  },
  {
    id: "story",
    label: "Short story",
    aspect: "16:9",
    resolution: "720p",
    fps: 24,
    shots: [
      { key: "storyOpen", title: "Opening", stillPrompt: "A wide view of where the story happens", seconds: 4 },
      { key: "storyMeet", title: "Meet the hero", stillPrompt: "The main character at work, early morning", seconds: 4 },
      { key: "storyTurn", title: "The turn", stillPrompt: "Something unexpected arrives", seconds: 4 },
      { key: "storyPeak", title: "The peak", stillPrompt: "The moment everything is at stake", seconds: 4 },
      { key: "storyEnd", title: "Ending", stillPrompt: "A quiet last look back", seconds: 4 },
    ],
  },
  {
    id: "broll",
    label: "Scene / b-roll",
    aspect: "16:9",
    resolution: "1080p",
    fps: 24,
    shots: [
      { key: "brollWide", title: "Wide street", stillPrompt: "A busy city street at dusk, wide shot", seconds: 4 },
      { key: "brollCrowd", title: "Crowd", stillPrompt: "A nightclub crowd moving under coloured lights", seconds: 4 },
      { key: "brollDetail", title: "Detail", stillPrompt: "Rain on a window with neon reflections, close up", seconds: 4 },
    ],
  },
];

export function templateById(id: string): ProjectTemplate {
  return PROJECT_TEMPLATES.find((template) => template.id === id) ?? PROJECT_TEMPLATES[0];
}

/** The `add_shot` arguments a template adds after the project is made, in order. */
export function templateShotArgs(
  template: ProjectTemplate,
  translate: (shot: TemplateShot) => { title: string; stillPrompt: string },
): { title: string; stillPrompt: string; seconds: number }[] {
  return template.shots.map((shot) => ({ ...translate(shot), seconds: shot.seconds }));
}
