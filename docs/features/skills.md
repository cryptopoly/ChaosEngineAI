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
description, the file list and the security scan so you can check them before you switch the skill on.

**Scripts in a skill are never run.** A script file is stored as text and the model can only read it.

## Security scan (D-68)

Every import is scanned, and the result shows in the import preview and again on the skill's card: a
rating (**Low**, **Medium** or **High**, as a word with an icon, never colour alone) and a list of
findings, each with the file, the line, the rule id and a short explanation.

**It is a heuristic, not a guarantee.** A fixed set of local rules reads the text; nothing is run and
nothing leaves the computer. Low means no rule matched, not that the skill is safe. Read a skill before
you switch it on.

| Rule | Severity | Looks for |
|------|----------|-----------|
| `override-instructions` | High | "ignore all previous instructions", "disregard your rules", "new system prompt" |
| `hide-from-user` | High | "do not tell the user about", "without telling the user", "keep this secret", "silently send" |
| `run-commands` | Medium | asking to run something in a shell or terminal, to call tools on its own, `sudo` |
| `dangerous-command` | High | download piped into a shell (`curl ... \| sh`), `rm -rf` on `/`, `~` or `$HOME` |
| `read-secrets` | High | reading or collecting SSH keys, `.env`, API keys, tokens, passwords, wallets, `/etc/passwd` |
| `send-data-out` | High | sending or uploading something to a web address, webhook or paste site |
| `url` / `url-action` | Low / Medium | every web address is listed; Medium when the line asks to fetch from it or send to it |
| `ip-address` / `ip-address-action` | Low / Medium | the same for IPv4 addresses (version numbers such as 0.1.5.1 are not counted) |
| `hidden-direction-or-tag-characters` | High | bidirectional controls and Unicode tag characters that hide or reorder text |
| `zero-width-characters` | Medium | zero-width spaces and joiners |
| `mixed-script-word` | Medium | a word that mixes Latin letters with look-alike Cyrillic, Greek, Armenian or Cherokee letters |
| `encoded-blob` | Medium | base64 runs of 100+ characters, hex runs of 96+, long `\x..` escapes |
| `bundled-script` | Medium | script files kept as text or skipped (`.sh`, `.py`, `.js`, `.ps1`, ...) |
| `bundled-executable` | Medium | program files that were skipped (`.exe`, `.dll`, `.so`, `.dylib`, `.app`, ...) |
| `large-text` | Medium | SKILL.md over 40,000 characters, or all files over 400,000 |
| `long-description` | Low | a description over 500 characters (Chat shows the model only the first 160) |
| `name-mismatch` | Low | the skill's name differs from the folder or archive it came in |

The rating is the worst severity found: any High finding makes the skill High, any Medium makes it
Medium, otherwise Low. Hidden characters show in a finding's excerpt as `<U+202E>` and the like. At most
200 findings are kept.

**A High skill needs one more step.** Its card shows an "I read the findings" box; "Available in Chat"
stays greyed out until you tick it. Nothing is blocked for good: you decide. The API enforces the same
step (see below). Switching a skill off never needs it.

The two samples in `docs/examples/skills` rate Low with no findings.

### What the scan cannot catch

- Wording the rules do not know: a paraphrase ("pay no attention to what came before"), another
  language, an instruction split across lines or files, or text built up from pieces.
- Meaning. A step that looks ordinary ("summarise the files in the home folder") can still be harmful in
  context; the rules match words, not intent.
- What a model might be talked into by text that reads as normal prose, and anything in a file the
  import skipped (only the skipped file's name is checked).
- Remote content. A link is listed, never opened, so whatever it points to is unchecked.

### Adding a rule

Rules live in `backend_service/skills/scan.py` as two tables: `LINE_RULES` (a regular expression tried
on every line) and `SKILL_RULES` (a function over the whole skill). Each row has an id, a severity and a
one-line explanation. Add a row, add a positive and a negative case to `tests/test_skills_scan.py`, and
bump `RULES_VERSION`: stored skills are scanned again under the new rules the next time they are listed.

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
| `GET /api/skills` | All imported skills with their files, skipped files and `scan` (re-scanned when the rules changed). |
| `POST /api/skills/import` `{path}` | Import a folder or `.zip` on this computer; the answer carries the `scan`. 400 with a reason, 409 for a duplicate id. |
| `POST /api/skills/{id}/enabled` `{enabled, acknowledgeFindings?}` | Switch a skill on or off for Chat. Switching on a skill rated High answers 409 unless `acknowledgeFindings` is `true`. |

`scan` is `{rating, findings[], truncated, scannedAt, rulesVersion}`; each finding is
`{file, line, rule, severity, message, excerpt}` (`line` is 0 for a finding about a whole file).
| `DELETE /api/skills/{id}` | Delete a skill. |

## Tests

`tests/test_skills_import.py` (validation and store), `tests/test_skills_scan.py` (each scan rule with a
positive and a negative case, the samples rate Low, a crafted skill rates High, rescan on new rules, the
High confirm in the store and routes), `tests/test_skills_chat.py` (routes, prompt, tools, chat stream),
`tests/test_skills_e2e_phase.py` (E2E phase 19 in process, including the scan check),
`src/features/chat/__tests__/skillsChat.test.tsx` (UI: rating chip, findings, the High confirm, the Chat
tooltip), `e2e/tests/16-skills.spec.ts` (Playwright: import, switch, a crafted bad skill rated High with
the confirm, 375 px).
