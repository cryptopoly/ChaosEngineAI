import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

const skillCount = { value: 0 };
vi.mock("../skillsChat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../skillsChat")>()),
  useEnabledSkillCount: () => skillCount.value,
}));

import type { Skill } from "../../../api/skills";
import { deleteSkill, importSkill, listSkills, setSkillEnabled } from "../../../api/skills";
import { SkillCard, SkillsPanel, formatBytes } from "../../settings/SkillsPanel";
import { ChatSkillsSwitch } from "../ChatSkillsSwitch";
import { readChatSkillsEnabled, writeChatSkillsEnabled } from "../skillsChat";

function memoryStorage() {
  const store = new Map<string, string>();
  return {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  };
}

function skill(over: Partial<Skill> = {}): Skill {
  return {
    id: "storyboard-house-style",
    name: "storyboard-house-style",
    description: "House style for storyboard shots.",
    enabled: false,
    importedAt: "2026-11-08 10:00:00",
    source: "storyboard-house-style",
    files: [{ path: "SKILL.md", bytes: 1200 }, { path: "palette.md", bytes: 300 }],
    skipped: [{ path: "logo.png", reason: "not a text file" }],
    ...over,
  };
}

const noop = () => undefined;

beforeEach(() => {
  vi.stubGlobal("window", { localStorage: memoryStorage(), dispatchEvent: () => true });
  skillCount.value = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("per-chat Skills switch", () => {
  it("is off by default and remembered per chat", () => {
    expect(readChatSkillsEnabled("s1")).toBe(false);
    writeChatSkillsEnabled("s1", true);
    expect(readChatSkillsEnabled("s1")).toBe(true);
    expect(readChatSkillsEnabled("s2")).toBe(false);
    writeChatSkillsEnabled("s1", false);
    expect(readChatSkillsEnabled("s1")).toBe(false);
    writeChatSkillsEnabled(undefined, true);
    expect(readChatSkillsEnabled(undefined)).toBe(false);
  });

  it("hides when no skill is switched on in Settings or there is no chat", () => {
    expect(renderToStaticMarkup(<ChatSkillsSwitch sessionId="s1" />)).toBe("");
    skillCount.value = 2;
    expect(renderToStaticMarkup(<ChatSkillsSwitch sessionId={undefined} />)).toBe("");
  });

  it("shows as a switch, on when this chat has it on", () => {
    skillCount.value = 1;
    const off = renderToStaticMarkup(<ChatSkillsSwitch sessionId="s1" />);
    expect(off).toContain('role="switch"');
    expect(off).toContain('aria-checked="false"');
    expect(off).toContain(">Skills<");
    writeChatSkillsEnabled("s1", true);
    const on = renderToStaticMarkup(<ChatSkillsSwitch sessionId="s1" />);
    expect(on).toContain('aria-checked="true"');
    expect(on).toContain("Skills ON");
    expect(on).toContain("Scripts in skills never run");
  });
});

describe("Settings skills panel", () => {
  it("says scripts never run and that new skills start off", () => {
    const html = renderToStaticMarkup(<SkillsPanel backendOnline />);
    expect(html).toContain("Scripts in a skill are never run");
    expect(html).toContain("New skills start switched off");
    expect(html).toContain("No skills imported yet.");
    expect(html).toContain("Choose folder");
    expect(html).toContain("Choose .zip");
  });

  it("shows a skill's description, files and skipped files before it is switched on", () => {
    const html = renderToStaticMarkup(
      <SkillCard skill={skill()} busy={false} confirming={false} onToggle={noop} onAskDelete={noop} onCancelDelete={noop} onDelete={noop} />,
    );
    expect(html).toContain("House style for storyboard shots.");
    expect(html).toContain("2 files");
    expect(html).toContain("palette.md");
    expect(html).toContain("1.2 KB");
    expect(html).toContain("logo.png");
    expect(html).toContain("not a text file");
    expect(html).toContain("Available in Chat");
    expect(html).not.toContain("checked=\"\"");
    expect(html).toContain(">Delete<");
  });

  it("asks before deleting", () => {
    const html = renderToStaticMarkup(
      <SkillCard skill={skill({ enabled: true })} busy={false} confirming onToggle={noop} onAskDelete={noop} onCancelDelete={noop} onDelete={noop} />,
    );
    expect(html).toContain("Delete storyboard-house-style for good");
    expect(html).toContain("Cancel");
    expect(html).toContain("checked=\"\"");
  });

  it("formats sizes", () => {
    expect(formatBytes(12)).toBe("12 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
  });
});

describe("skills API client", () => {
  it("calls the skills routes", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      void init;
      if (url.endsWith("/api/skills")) return new Response(JSON.stringify({ skills: [skill()] }), { status: 200 });
      if (url.endsWith("/api/skills/import")) return new Response(JSON.stringify({ skill: skill() }), { status: 200 });
      if (url.endsWith("/enabled")) return new Response(JSON.stringify({ skill: skill({ enabled: true }) }), { status: 200 });
      return new Response(JSON.stringify({ deleted: "storyboard-house-style" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    expect((await listSkills())[0].id).toBe("storyboard-house-style");
    expect((await importSkill("/tmp/skill")).name).toBe("storyboard-house-style");
    expect((await setSkillEnabled("storyboard-house-style", true)).enabled).toBe(true);
    await deleteSkill("storyboard-house-style");
    const calls = fetchMock.mock.calls
      .map(([url, init]) => [String(url).replace(/^.*\/api/, "/api"), init?.method ?? "GET"])
      .filter(([url]) => url.startsWith("/api/skills"));
    expect(calls).toEqual([
      ["/api/skills", "GET"],
      ["/api/skills/import", "POST"],
      ["/api/skills/storyboard-house-style/enabled", "POST"],
      ["/api/skills/storyboard-house-style", "DELETE"],
    ]);
    const importCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/skills/import"));
    expect(JSON.parse(String(importCall?.[1]?.body))).toEqual({ path: "/tmp/skill" });
  });

  it("keeps the backend's sentence as the error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "No SKILL.md at the top of the skill." }), { status: 400 })));
    await expect(importSkill("/tmp/x")).rejects.toThrow("No SKILL.md at the top of the skill.");
  });
});
