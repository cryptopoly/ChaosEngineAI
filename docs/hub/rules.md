---
title: Working agreement
order: 5
summary: How Claude and its agents work on this project.
---
# Working agreement

How Claude and its team of agents work on ChaosEngineAI. The generic rules come from the owner's working agreement; the project-specific ones are in the section below them and win where they differ.

## 1. Be objective and challenge the owner

- Challenge any request that is wrong, unworkable, or workable only with big caveats. Say it once, before building, with the reason and one or two alternatives. After the owner decides, follow the decision.
- Label pushback: **wrong**, **costly**, **risky** or **unclear**.
- Say when evidence is thin, a source is secondary, or something is a guess.
- The owner can overrule. Claude then records the decision and moves on.

## 2. Start of every project

1. **Landscape brief** (one page, sources and confidence tags): comparable products, what users praise and complain about, pricing, gaps to own.
2. **Questions**, one batch of at most 8, ranked, each with a recommended default so work can start meanwhile. Cover audience, goals and success measures, style, scope, platforms, architecture limits, budget and deadline, legal and licence limits, accessibility, and what to avoid.
3. **Tools check:** search for skills, plugins and MCP servers that help, listing source, purpose and risk. They run with our permissions, so they are proposed and **never installed without the owner's yes**. Prefer official sources and pinned versions.
4. **Scaffold the hub** and a decision log before any real work, and record the answers to the questions.

## 3. The project hub

- Every project has one navigable HTML hub, generated from short markdown files, so any new chat, model or person can pick the project up with little friction.
- Markdown is the source of truth; HTML is the readable view and opens from disk with no server.
- A one-screen [status and handover](status.md) page is read first in every session.
- [Decisions](decisions.md) hold open items with a recommendation on top and a collapsed dated log below.
- The hub is rebuilt after every batch of changes, by `python3 tools/hub_tracker.py && python3 tools/build_hub.py`, never by hand. The follow-up rows live in `CLAUDE.md` (see [tracker](tracker/README.md)); everything else lives in this folder.

## 4. Decisions and questions

- Every question or decision raised in chat is logged on the decisions page the same turn, with a recommendation and a reason. Decided items move to the dated log. The open list stays short.
- Priorities: P1 blocks launch or reworks built work; P2 is needed before the next milestone; P3 is later. Parked ideas go to [ideas](ideas.md).
- Ask in chat only what changes the work. If the answer would not change the plan, state the default and proceed.
- One owner for shared coordination files (the lead). Sub-agents propose decision-log changes in their reports; they do not edit it.

## 5. Parallel work and checkpoints

- **One worktree per agent or stream**, off the latest main. Never stash, reset, rebase with autostash, or discard work in a shared tree. Only the lead touches the main checkout.
- **Checkpoint:** small commits, pushed early. Each stream keeps a [progress file](progress/README.md) written so a fresh agent can resume after an outage or a dropped connection.
- **Generated artefacts** (the built hub, contact sheets, caches) are rebuilt by the lead rather than committed by every agent. Large binaries stay out of git or in LFS, with a backup plan.
- **Process hygiene:** a unique scratch directory per agent, a cap on concurrent heavy processes, never kill processes by name, clean up before finishing, never let tests write to the user's real settings or data.
- *Limit:* parallel work has coordination cost. Split by area; review time grows with agent count.

## 6. Right model for the task

| Work | Default |
|---|---|
| Doc chores, simple research, summaries, formatting | Smallest model |
| Bounded implementation, reviews, QA sweeps, prototypes | Mid model |
| Design-heavy or cross-cutting work, hard debugging, high cost of error | Large model |

- Use the cheapest model that passes. Record the model per task. If output fails the bar, say so and move up or down.
- The owner sets the budget. Paid generations (voices, images, APIs) get a cap per task and a log.

## 7. Verify, do not trust

- An agent's report is a **claim**. The lead checks the important ones: tests on a clean copy of main, screenshots actually looked at, numbers re-measured.
- Definition of done: see [testing](testing.md).
- For visual work: before and after side by side, looked at full size, iterated at least twice.
- One command runs every test.

## 8. Communication

- Short and precise: headline, what changed, what needs the owner, risks. Detail goes into files and the hub.
- Say early when something is blocked, costly or unavailable.

## 9. Autonomy, safety and scope

1. **Safe without asking:** edits in the project, tests, research, commits per project policy.
2. **Always ask first:** force-pushes and history rewrites, deleting data, installing third-party tools, spending paid credits or money, publishing or posting, messaging people, changing global settings.
3. **Security and licences:** no secrets in the repo; a ledger of third-party assets and licences in [risks](risks.md); disclose AI use where a platform requires it; check names and terms for clashes before committing to them.
4. **Scope control:** a freeze point; new ideas go to the vault unless the owner promotes them.
5. **Risk and assumptions log** in [risks](risks.md).
6. **Retro after each phase** (five lines): what worked, what failed, one rule to change.
7. **Human checkpoints** (playtest or usability session, listening pass, native-speaker review, legal and tax advice) are scheduled and listed as reminders in the [backlog](backlog.md), because agents cannot judge feel, language accuracy or legal standing.

## 10. Lessons, written as rules

- Research claims carry confidence tags (solid, likely, contested, unverified) and sources; invented or licensed material is labelled.
- Content gated by funding or release stage is flagged in data and excluded from public builds by a check script, not by hope.
- Every screen or state can be launched directly by a flag, and every flag is documented in [testing](testing.md).
- Each new kind of artefact has a checklist, so later additions meet the same bar.
- Reminders for the owner sit at the top of the backlog; open work for Claude and for the owner are listed separately on the status page.
- Keep the most-read pages short; move detail to linked pages.

## 11. ChaosEngineAI rules (these win over the generic ones)

- **Editorial rule:** no code, comment, UI string, doc or commit message names or compares against other desktop AI apps (the full list and the three allowed exceptions are at the top of `CLAUDE.md`). When a question mentions one, log it here by what it is, for example "an MLX server", not by the app's name.
- **No worktrees.** Work in the main checkout (`/Users/dan/ChaosEngineAI`); the owner runs and tests from there, and edits in a worktree are invisible to them. This replaces the generic one-worktree-per-agent rule; parallel agents still keep one progress file each and the lead owns shared files.
- **Hold pushes until tested.** No push, pull request, merge into `staging` or re-tag until the work is finished and tested; every push runs GitHub builds that cost Actions minutes. Keep committing locally; ask before any push.
- **No attribution lines** in commit messages or pull request text.
- **Icons:** monochrome SVG only, no emoji in the UI.
- **New feature gate:** a user-visible feature, engine, model family, install endpoint or cache or spec-dec strategy lands with an E2E check in `scripts/e2e_test_suite.py` (see [testing](testing.md)).
- **Prefer newer and faster models** when choosing defaults or replacements, unless a gated download or a missing measurement makes the older one the safer default.
- **Commit style:** conventional prefixes (`feat`, `fix`, `docs`, `chore`); tracker rows, release notes and tests go in the same commits as the code.
