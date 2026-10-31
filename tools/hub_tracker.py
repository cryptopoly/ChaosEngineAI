#!/usr/bin/env python3
"""Generate docs/hub/tracker/*.md from the Follow-Ups Tracker table in CLAUDE.md.

CLAUDE.md stays the source of truth for the FU-### rows; this script only reads it.
The generated pages are committed, so a fresh clone builds the hub without running
anything first, and tests/test_hub_tracker.py fails when they drift from CLAUDE.md.

  python3 tools/hub_tracker.py            rewrite docs/hub/tracker/
  python3 tools/hub_tracker.py --check    exit 1 when the committed pages are out of date

Standard library only, Python 3.11+.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = ROOT / "CLAUDE.md"
OUT_DIR = ROOT / "docs" / "hub" / "tracker"
BLOB = "https://github.com/cryptopoly/ChaosEngineAI/blob/staging/"
TREE = "https://github.com/cryptopoly/ChaosEngineAI/tree/staging/"

_ROW_START = re.compile(r"^\|\s*(~~)?(FU-\d+)(~~)?\s*\|")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_OPEN_ITEMS = re.compile(r"\*\*Open[:*(]|\bOpen:|\bOpen \(|\bstill open\b", re.I)
_SHIPPED = re.compile(r"^\W*(?:\*\*)?\s*(shipped|fully shipped|closed|obsoleted|dropped|superseded)", re.I)


@dataclass
class Row:
    fu: str
    struck: bool
    item: str
    status: str
    notes: str

    @property
    def closed(self) -> bool:
        """Struck through, or marked shipped with nothing left open in its notes."""
        if self.struck:
            return True
        return bool(_SHIPPED.match(self.status)) and not _OPEN_ITEMS.search(self.notes)

    @property
    def headline(self) -> str:
        text = _plain(self.status)
        for stop in (". ", " — ", "; "):
            cut = text.find(stop)
            if 12 < cut < 110:
                return text[:cut].rstrip(".")
        return text[:110].rstrip() + ("…" if len(text) > 110 else "")


def _plain(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    return re.sub(r"[*`~]|\\\|", lambda m: "|" if m.group(0) == "\\|" else "", text).strip()


def _clean_cell(text: str) -> str:
    return text.strip().replace("\\|", "|").replace("~~", "")


def _fix_link(match: re.Match[str]) -> str:
    label, target = match.group(1), match.group(2)
    if re.match(r"^(https?:|mailto:|#)", target):
        return match.group(0)
    path, _, line = target.partition(":")
    anchor = f"#L{line}" if line.isdigit() else ""
    last = path.rstrip("/").rsplit("/", 1)[-1]
    base = TREE if path.endswith("/") or "." not in last else BLOB
    return f"[{label}]({base}{path.lstrip('./')}{anchor})"


def _body(text: str) -> str:
    return _LINK.sub(_fix_link, _clean_cell(text))


def parse(markdown: str) -> list[Row]:
    rows: list[Row] = []
    for line in markdown.splitlines():
        match = _ROW_START.match(line)
        if not match:
            continue
        inner = line.strip()[1:]
        if inner.endswith("|") and not inner.endswith("\\|"):
            inner = inner[:-1]
        cells = _CELL_SPLIT.split(inner)
        if len(cells) < 4:
            continue
        # A few rows carry an unescaped pipe inside the notes; the first three cells are always clean.
        rows.append(Row(
            fu=match.group(2),
            struck=bool(match.group(1)),
            item=_clean_cell(cells[1]),
            status=_clean_cell(cells[2]),
            notes="|".join(cells[3:]).strip().replace("\\|", "|").replace("~~", ""),
        ))
    return rows


def _number(fu: str) -> int:
    return int(fu.split("-")[1])


def _details(row: Row) -> str:
    summary = f"{row.fu} · {_plain(row.item)} — {row.headline}"
    summary = summary.replace("<", "&lt;").replace(">", "&gt;")
    parts = [f"**Status or trigger:** {_body(row.status)}", f"**Notes:** {_body(row.notes)}"]
    return "<details>\n<summary>" + summary + "</summary>\n\n" + "\n\n".join(parts) + "\n\n</details>\n"


def _page(title: str, order: int, summary: str, intro: str, rows: list[Row]) -> str:
    out = [f"---\ntitle: {title}\norder: {order}\nsummary: {summary}\n---\n", f"# {title}\n", intro + "\n"]
    out.extend(_details(row) for row in rows)
    return "\n".join(out)


def build() -> dict[str, str]:
    rows = parse(CLAUDE_MD.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("no FU rows found in CLAUDE.md")
    ordered = sorted(rows, key=lambda r: -_number(r.fu))
    open_rows = [r for r in ordered if not r.closed]
    closed_rows = [r for r in ordered if r.closed]
    generated = "Generated from the Follow-Ups Tracker in `CLAUDE.md` by `tools/hub_tracker.py`; edit the row there, not here."
    pages = {
        "open.md": _page(
            "Open and live follow-ups", 2,
            f"{len(open_rows)} rows with work, decisions or watch items still open, newest first.",
            f"{generated} Rows with open work, a pending decision, a watch-list entry, or a shipped item that still lists open points. "
            f"Newest first; open a row for the full note.\n",
            open_rows,
        ),
        "closed.md": _page(
            "Closed follow-ups", 3,
            f"{len(closed_rows)} shipped, obsoleted or dropped rows, newest first.",
            f"{generated} Shipped, obsoleted or dropped rows (struck through, or marked shipped with no open points left). Kept as the history of why things are the way they are.\n",
            closed_rows,
        ),
    }
    index = [
        "---\ntitle: Follow-up tracker\norder: 1\nsummary: Every FU row from CLAUDE.md, split into open and closed.\n---\n",
        "# Follow-up tracker\n",
        f"{generated}\n",
        f"- [Open and live follow-ups](open.md): **{len(open_rows)}** rows.",
        f"- [Closed follow-ups](closed.md): **{len(closed_rows)}** rows.\n",
        "## Newest ten\n",
        "| Row | Item | Status |\n|---|---|---|",
    ]
    for row in ordered[:10]:
        page = "closed.md" if row.closed else "open.md"
        index.append(f"| [{row.fu}]({page}) | {_plain(row.item)[:90].replace('|', '/')} | {row.headline.replace('|', '/')} |")
    pages["README.md"] = "\n".join(index) + "\n"
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the committed pages are out of date")
    args = parser.parse_args()
    pages = build()
    if args.check:
        stale = [name for name, text in pages.items() if not (OUT_DIR / name).exists() or (OUT_DIR / name).read_text(encoding="utf-8") != text]
        if stale:
            print("out of date: " + ", ".join(stale) + "; run python3 tools/hub_tracker.py", file=sys.stderr)
            return 1
        print(f"tracker pages up to date ({len(pages)} files)")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(pages)} pages to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
