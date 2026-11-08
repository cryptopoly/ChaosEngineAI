# Skills

A skill teaches the chat model a procedure: a house style, a checklist, a workflow for a tool. It is a
folder with a `SKILL.md` and, if needed, more text files. The layout is the one Anthropic Agent Skills use,
so a skill written for another agent imports unchanged.

```
storyboard-house-style/
  SKILL.md        front matter (name, description), then markdown
  palette.md      optional extra files the skill mentions
```

`SKILL.md` starts with front matter:

```
---
name: storyboard-house-style
description: House style for storyboard shots - shot list format, camera words, lighting and palette.
---
```

Two samples live in [docs/examples/skills](../examples/skills/): `plain-summaries` (general) and
`storyboard-house-style` (with an extra file).

## Import

Settings > Skills. Type the path of a skill folder or a `.zip`, or choose one with the file dialog, then
Import. A zip may hold the files at its top or inside one folder.

The import is strict, because the folder or archive may come from anywhere:

- Only text files are kept (by extension, valid UTF-8, no NUL bytes). Binaries and hidden files are
  skipped and listed.
- Symbolic links, absolute paths, `..` entries, encrypted entries and over-compressed entries (zip bombs)
  reject the whole import.
- Caps: 64 text files, 256 KB per file, 2 MB in all, 512 entries looked at, 16 MB for the zip.
- The skill's id is a slug of its `name`. Importing a skill whose id is already there is refused; delete
  the old one first.

Skills are stored in `~/.chaosengine/skills/<id>/` and **start switched off**. The card shows the
description and the file list so you can check them before you switch the skill on.

**Scripts in a skill are never run.** A script file is stored as text and the model can only read it.

## In Chat

When at least one skill is switched on in Settings, the composer shows a **Skills** switch (tool-capable
models only). It is off by default and remembered per chat. With it on:

- the system prompt lists each switched-on skill's name and a one-line description (cut at 160
  characters, at most 24 skills), so a small model keeps its context free;
- the model gets two read-only tools: `load_skill(name)` returns `SKILL.md` without its front matter
  plus the list of other files, and `read_skill_file(name, path)` returns one of those files. Answers are
  cut at 12,000 characters. An unknown name answers with the list of valid names.

Skill text reaches the model as reference material marked as an imported file, never as the user's words.
The tools only read stored text, so they do not ask for approval. Skills work in streamed chats, alone or
with the Studio tools.

## API

| Route | What |
|-------|------|
| `GET /api/skills` | All imported skills with their files and skipped files. |
| `POST /api/skills/import` `{path}` | Import a folder or `.zip` on this computer. 400 with a reason, 409 for a duplicate id. |
| `POST /api/skills/{id}/enabled` `{enabled}` | Switch a skill on or off for Chat. |
| `DELETE /api/skills/{id}` | Delete a skill. |

## Tests

`tests/test_skills_import.py` (validation and store), `tests/test_skills_chat.py` (routes, prompt, tools,
chat stream), `tests/test_skills_e2e_phase.py` (E2E phase 19 in process), `src/features/chat/__tests__/skillsChat.test.tsx`
(UI), `e2e/tests/16-skills.spec.ts` (Playwright: import, switch, 375 px).
