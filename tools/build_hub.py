#!/usr/bin/env python3
"""Build the project hub: docs/hub/*.md  ->  docs/site/*.html

Standard library only, Python 3.11+. The output opens from disk (file://):
relative links only, no scripts or fonts from the web.

  python3 tools/build_hub.py            build the site
  python3 tools/build_hub.py --check    run the checks only, write nothing (exit 1 on errors)
  python3 tools/build_hub.py --strict   build, then exit 1 if there were errors
  python3 tools/build_hub.py --serve    build, then serve docs/site on http://127.0.0.1:8000
                                        (--serve 9000 picks the port)
  options: --root DIR (project root), --src DIR (default docs/hub), --out DIR (default docs/site)

Optional docs/hub/hub.json:
  {"project_name": "...", "tagline": "...", "nav_order": ["status", "decisions"],
   "hide_folders": ["drafts"], "accent": "#2f6f8f", "font": "fonts/MyFont.woff2"}
Optional front matter per page: title, order, status, summary, example.
`example: true` marks template content that has not been replaced yet; the build errors until it is removed.
Errors: missing link targets, missing images, links that leave the hub folder.
Warnings: everything else (unknown nav_order entries, web iframes).
"""
from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

STATUS_WORDS = {"todo", "in progress", "done", "blocked", "open", "decided", "draft", "dropped", "accepted", "proposed"}
INDEX_NAMES = {"readme", "index"}
CSS_PATH = "_hub/hub.css"


@dataclass
class Page:
    src: Path
    rel: str                      # posix path relative to the hub folder, e.g. "progress/README.md"
    out: str = ""                 # built path, e.g. "progress/index.html"
    meta: dict = field(default_factory=dict)
    body: str = ""
    title: str = ""

    @property
    def is_index(self) -> bool:
        return Path(self.rel).stem.lower() in INDEX_NAMES

    @property
    def hidden(self) -> bool:      # built, but not in the sidebar
        return Path(self.rel).name.startswith("_")


ALLOW_EXAMPLES = False   # --allow-examples: build the untouched template without errors


class Report:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print("warning:", msg, file=sys.stderr)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print("error:", msg, file=sys.stderr)


# --------------------------------------------------------------------------- loading

def parse_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[m.end():]


def out_path(rel: str) -> str:
    p = Path(rel)
    parent = p.parent.as_posix()
    name = "index.html" if p.stem.lower() in INDEX_NAMES else p.stem + ".html"
    return name if parent == "." else f"{parent}/{name}"


def load_pages(src: Path, hidden_dirs: set[str], rep: Report) -> dict[str, Page]:
    pages: dict[str, Page] = {}
    for f in sorted(src.rglob("*.md")):
        rel = f.relative_to(src).as_posix()
        if any(part in hidden_dirs or part.startswith(".") for part in Path(rel).parts[:-1]):
            continue
        meta, body = parse_front_matter(f.read_text(encoding="utf-8"))
        page = Page(f, rel, out_path(rel), meta, body)
        if str(meta.get("example", "")).lower() == "true" and not ALLOW_EXAMPLES:
            rep.error(f"{rel}: still marked 'example: true'; replace the template content, then remove that line")
        h1 = re.search(r"^#\s+(.+?)\s*$", body, re.M)
        page.title = meta.get("title") or (re.sub(r"[`*_]", "", h1.group(1)) if h1 else
                                           Path(rel).stem.replace("_", " ").replace("-", " ").title())
        pages[rel] = page
    outs: dict[str, str] = {}
    for p in pages.values():
        if p.out in outs:
            rep.error(f"{p.rel} and {outs[p.out]} both build to {p.out}")
        outs[p.out] = p.rel
    return pages


# --------------------------------------------------------------------------- markdown

class Ctx:
    """Per-page rendering state."""
    def __init__(self, page: Page, pages: dict[str, Page], src: Path, rep: Report) -> None:
        self.page, self.pages, self.src, self.rep = page, pages, src, rep
        self.ids: dict[str, int] = {}
        self.assets: set[str] = set()   # referenced hub files (images etc.)


def slugify(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^\w\s-]", "", html.unescape(s).lower())
    return re.sub(r"[\s_]+", "-", s).strip("-") or "section"


def resolve_target(target: str, ctx: Ctx, kind: str) -> str:
    """Rewrite a relative link/image target; check that it exists inside the hub."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "//")):
        return target
    path, frag = (target.split("#", 1) + [""])[:2]
    if not path:
        return target
    here = posixpath.dirname(ctx.page.rel)
    full = posixpath.normpath(posixpath.join(here, path))
    where = f"{ctx.page.rel}: {kind} '{target}'"
    if full.startswith("..") or full.startswith("/"):
        ctx.rep.error(f"{where} leaves the hub folder")
        return target
    if full.lower().endswith(".md"):
        dest = ctx.pages.get(full)
        if not dest:
            ctx.rep.error(f"{where} points to a page that does not exist")
            return target
        new = dest.out
    else:
        if not (ctx.src / full).exists():
            ctx.rep.error(f"{where} points to a file that does not exist")
            return target
        ctx.assets.add(full)
        new = full
    rel = posixpath.relpath(new, posixpath.dirname(ctx.page.out) or ".")
    return rel + ("#" + frag if frag else "")


def badges(htm: str) -> str:
    """Turn P1/P2/P3 and 'P1, todo' style status words into badges (outside tags)."""
    def piece(t: str) -> str:
        t = re.sub(r"\b(P[123])\b", lambda m: f'<span class="badge {m.group(1).lower()}">{m.group(1)}</span>', t)
        return re.sub(r"(?<=</span>, )(todo|in progress|done|blocked)\b",
                      lambda m: f'<span class="badge {m.group(1).replace(" ", "-")}">{m.group(1)}</span>', t)
    parts = re.split(r"(<[^>]+>)", htm)
    out, skip = [], 0
    for p in parts:
        if p.startswith("<"):
            if re.match(r"<(code|a|span)\b", p):
                skip += 1
            elif re.match(r"</(code|a|span)>", p):
                skip = max(0, skip - 1)
            out.append(p)
        else:
            out.append(p if skip else piece(p))
    return "".join(out)


def inline(text: str, ctx: Ctx) -> str:
    stash: list[str] = []

    def keep(s: str) -> str:
        stash.append(s)
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(r"(`+)(.+?)\1", lambda m: keep(f"<code>{html.escape(m.group(2).strip(), quote=False)}</code>"), text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~])", lambda m: keep(html.escape(m.group(1))), text)
    text = html.escape(text, quote=False)
    text = re.sub(r"&lt;(/?(?:br|kbd|sub|sup))\s*/?&gt;", r"<\1>", text)
    text = re.sub(r"&lt;(https?://[^\s&]+)&gt;", r'<a class="ext" href="\1">\1</a>', text)

    def img(m: re.Match) -> str:
        src = resolve_target(m.group(2), ctx, "image")
        return keep(f'<img src="{html.escape(src)}" alt="{html.escape(m.group(1))}" loading="lazy">')
    text = re.sub(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)', img, text)

    def link(m: re.Match) -> str:
        href = resolve_target(html.unescape(m.group(2)), ctx, "link")
        ext = ' class="ext"' if re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I) else ""
        title = f' title="{html.escape(m.group(3))}"' if m.group(3) else ""
        return f'<a{ext} href="{html.escape(href)}"{title}>{m.group(1)}</a>'
    text = re.sub(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)', link, text)

    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    text = re.sub(r"(?: {2,}|\\)\n", "<br>\n", text)
    text = badges(text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


RE_FENCE = re.compile(r"^\s*(```+|~~~+)\s*([\w+-]*)")
RE_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
RE_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
RE_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
RE_TSEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
RE_RAW = re.compile(r"^\s*(</?details[^>]*>|<summary>.*</summary>|<iframe\b.*>\s*(</iframe>)?)\s*$", re.I)


def is_block_start(lines: list[str], i: int) -> bool:
    ln = lines[i]
    return bool(RE_FENCE.match(ln) or RE_HEAD.match(ln) or RE_HR.match(ln) or RE_ITEM.match(ln)
                or ln.lstrip().startswith(">") or RE_RAW.match(ln) or is_table(lines, i))


def is_table(lines: list[str], i: int) -> bool:
    return "|" in lines[i] and i + 1 < len(lines) and bool(RE_TSEP.match(lines[i + 1])) and "|" in lines[i + 1] + lines[i]


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)]


def render_table(rows: list[str], ctx: Ctx) -> str:
    head, sep, body = split_row(rows[0]), split_row(rows[1]), rows[2:]
    align = ["right" if s.endswith(":") and not s.startswith(":") else "center" if s.startswith(":") and s.endswith(":")
             else "" for s in sep]

    def cell(tag: str, text: str, n: int) -> str:
        plain = re.sub(r"[*_`]", "", text).strip().rstrip(".").lower()
        content = inline(text, ctx)
        if tag == "td" and plain in STATUS_WORDS:
            content = f'<span class="badge {plain.replace(" ", "-")}">{html.escape(plain)}</span>'
        style = f' style="text-align:{align[n]}"' if n < len(align) and align[n] else ""
        return f"<{tag}{style}>{content}</{tag}>"

    out = ['<div class="tablewrap"><table>', "<thead><tr>" + "".join(cell("th", c, n) for n, c in enumerate(head)) + "</tr></thead><tbody>"]
    for r in body:
        cells = split_row(r)
        out.append("<tr>" + "".join(cell("td", c, n) for n, c in enumerate(cells)) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def render_list(lines: list[str], i: int, ctx: Ctx) -> tuple[str, int]:
    first = RE_ITEM.match(lines[i])
    base = len(first.group(1))
    ordered = first.group(2)[0].isdigit()
    start = int(re.match(r"\d+", first.group(2)).group(0)) if ordered else 1
    items: list[list[str]] = []
    n = len(lines)
    while i < n:
        m = RE_ITEM.match(lines[i])
        if not m or len(m.group(1)) != base or m.group(2)[0].isdigit() != ordered:
            break
        offset = len(m.group(1)) + len(m.group(2)) + 1
        item = [m.group(3)]
        i += 1
        while i < n:
            ln = lines[i]
            if not ln.strip():
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and (len(lines[j]) - len(lines[j].lstrip())) > base:
                    item.extend([""] * (j - i))
                    i = j
                    continue
                break
            ind = len(ln) - len(ln.lstrip())
            if ind <= base:
                if RE_ITEM.match(ln) or is_block_start(lines, i):
                    break
                item.append(ln.strip())          # lazy paragraph continuation
            else:
                item.append(ln[min(ind, offset):] if ind >= offset else ln.lstrip())
            i += 1
        items.append(item)
        while i < n and not lines[i].strip() and i + 1 < n and RE_ITEM.match(lines[i + 1]) \
                and len(RE_ITEM.match(lines[i + 1]).group(1)) == base:
            i += 1
    tag = "ol" if ordered else "ul"
    out = [f'<{tag}{f" start={start}" if ordered and start != 1 else ""}>']
    for it in items:
        box = re.match(r"\[( |x|X)\]\s+(.*)", it[0])
        cls = ""
        if box:
            checked = box.group(1) != " "
            it[0] = box.group(2)
            cls = ' class="task done"' if checked else ' class="task"'
            prefix = f'<input type="checkbox" disabled{" checked" if checked else ""}> '
        else:
            prefix = ""
        inner = render_blocks(it, ctx)
        inner = re.sub(r"^<p>(.*?)</p>", lambda m: m.group(1), inner, count=1, flags=re.S)
        out.append(f"<li{cls}>{prefix}{inner}</li>")
    out.append(f"</{tag}>")
    return "\n".join(out), i


def render_blocks(lines: list[str], ctx: Ctx) -> str:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = RE_FENCE.match(ln)
        if m:
            fence, lang, buf = m.group(1), m.group(2), []
            indent = len(ln) - len(ln.lstrip())
            i += 1
            while i < n and not lines[i].strip().startswith(fence[:3]):
                buf.append(lines[i][indent:] if lines[i][:indent].strip() == "" else lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(buf), quote=False)}</code></pre>")
            continue
        m = RE_HEAD.match(ln)
        if m:
            level, text = len(m.group(1)), inline(m.group(2), ctx)
            slug = slugify(text)
            ctx.ids[slug] = ctx.ids.get(slug, 0) + 1
            if ctx.ids[slug] > 1:
                slug += f"-{ctx.ids[slug] - 1}"
            out.append(f'<h{level} id="{slug}">{text}<a class="anchor" href="#{slug}" aria-label="Link to this section">#</a></h{level}>')
            i += 1
            continue
        if RE_HR.match(ln):
            out.append("<hr>")
            i += 1
            continue
        if RE_RAW.match(ln):
            s = ln.strip()
            if s.lower().startswith("<summary>"):
                s = f"<summary>{inline(s[9:-10], ctx)}</summary>"
            elif s.lower().startswith("<iframe"):
                src = re.search(r'src="([^"]*)"', s)
                if src and re.match(r"^(https?:)?//", src.group(1)):
                    ctx.rep.warn(f"{ctx.page.rel}: iframe loads a web address ({src.group(1)}); the hub will not work offline")
                elif src:
                    s = s.replace(src.group(0), f'src="{html.escape(resolve_target(src.group(1), ctx, "iframe"))}"')
                s = f'<div class="frame">{s}</div>'
            out.append(s)
            i += 1
            continue
        if ln.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = render_blocks(buf, ctx)
            kind = re.match(r"^<p>(?:<strong>)?\s*(note|warning|tip)\b", inner, re.I)
            cls = f' class="callout {kind.group(1).lower()}"' if kind else ""
            out.append(f"<blockquote{cls}>\n{inner}\n</blockquote>")
            continue
        if is_table(lines, i):
            buf = [lines[i], lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                buf.append(lines[i])
                i += 1
            out.append(render_table(buf, ctx))
            continue
        if RE_ITEM.match(ln):
            htm, i = render_list(lines, i, ctx)
            out.append(htm)
            continue
        buf = [ln.rstrip("\n")]
        i += 1
        while i < n and lines[i].strip() and not is_block_start(lines, i):
            buf.append(lines[i])
            i += 1
        out.append(f"<p>{inline(chr(10).join(buf).strip(), ctx)}</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------- navigation

def sort_key(name: str, order: str | None, title: str, nav_order: list[str]):
    pos = nav_order.index(name) if name in nav_order else 10_000
    try:
        o = float(order) if order is not None else 9_999
    except ValueError:
        o = 9_999
    return (pos, o, title.lower())


def build_tree(pages: dict[str, Page], nav_order: list[str]) -> dict:
    root: dict = {"pages": [], "dirs": {}, "index": None}
    for p in pages.values():
        if p.hidden:
            continue
        node = root
        for d in Path(p.rel).parts[:-1]:
            node = node["dirs"].setdefault(d, {"pages": [], "dirs": {}, "index": None})
        if p.is_index:
            node["index"] = p
        else:
            node["pages"].append(p)
    return root


def render_nav(node: dict, current: Page, nav_order: list[str], top: bool = True) -> str:
    here = posixpath.dirname(current.out) or "."

    def href(p: Page) -> str:
        return posixpath.relpath(p.out, here)

    items: list[tuple[tuple, str]] = []
    for p in node["pages"]:
        cur = ' aria-current="page" class="current"' if p is current else ""
        items.append((sort_key(Path(p.rel).stem, p.meta.get("order"), p.title, nav_order),
                      f'<li><a href="{href(p)}"{cur}>{html.escape(p.title)}</a></li>'))
    for name, sub in node["dirs"].items():
        idx = sub["index"]
        label = html.escape(idx.title if idx else name.replace("_", " ").replace("-", " ").title())
        if idx:
            cur = ' aria-current="page" class="current"' if idx is current else ""
            label = f'<a href="{href(idx)}"{cur}>{label}</a>'
        opens = " open" if current.rel.startswith(name + "/") else ""
        inner = render_nav(sub, current, nav_order, False)
        items.append((sort_key(name, idx.meta.get("order") if idx else None, name, nav_order),
                      f"<li><details{opens}><summary>{label}</summary>{inner}</details></li>"))
    items.sort(key=lambda t: t[0])
    return "<ul>" + "".join(h for _, h in items) + "</ul>"


def breadcrumbs(page: Page, pages: dict[str, Page], project: str) -> str:
    here = posixpath.dirname(page.out) or "."
    crumbs = [f'<a href="{posixpath.relpath("index.html", here)}">{html.escape(project)}</a>']
    parts = Path(page.rel).parts[:-1]
    for k in range(len(parts)):
        folder = "/".join(parts[:k + 1])
        idx = next((q for q in pages.values() if q.is_index and posixpath.dirname(q.rel) == folder), None)
        label = html.escape(idx.title if idx else parts[k].replace("_", " ").title())
        crumbs.append(f'<a href="{posixpath.relpath(idx.out, here)}">{label}</a>' if idx else f"<span>{label}</span>")
    if not (page.is_index and not parts):
        crumbs.append(f"<span>{html.escape(page.title)}</span>")
    return '<nav class="crumbs" aria-label="Breadcrumb">' + " <i>/</i> ".join(crumbs) + "</nav>"


# --------------------------------------------------------------------------- output

def git_commit(root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def page_html(page: Page, body: str, ctx: dict) -> str:
    here = posixpath.dirname(page.out) or "."
    css = posixpath.relpath(CSS_PATH, here)
    status = page.meta.get("status", "")
    status_html = (f'<span class="badge {re.sub(r"[^a-z]+", "-", status.lower())}">{html.escape(status)}</span>'
                   if status else "")
    home = posixpath.relpath("index.html", here)
    tagline = f'<div class="tagline">{html.escape(ctx["tagline"])}</div>' if ctx["tagline"] else ""
    nav = render_nav(ctx["tree"], page, ctx["nav_order"])
    built = f'Built on {ctx["built"]}' + (f' &middot; commit <code>{ctx["commit"]}</code>' if ctx["commit"] else "")
    source = f' &middot; source <code>docs/hub/{html.escape(page.rel)}</code>' if page.rel != "(home)" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page.title)} &middot; {html.escape(ctx["project"])}</title>
<link rel="stylesheet" href="{css}">
</head>
<body>
<input type="checkbox" id="navtoggle" class="navtoggle">
<header class="top"><label for="navtoggle" class="burger" aria-label="Menu">&#9776;</label>
<a class="brand" href="{home}">{html.escape(ctx["project"])}</a></header>
<div class="layout">
<aside class="side"><a class="brand" href="{home}">{html.escape(ctx["project"])}</a>{tagline}
<nav aria-label="Pages">{nav}</nav></aside>
<main>
{breadcrumbs(page, ctx["pages"], ctx["project"])}
{status_html}
<article>
{body}
</article>
<footer>{built}{source}</footer>
</main>
</div>
<script>addEventListener("beforeprint",function(){{document.querySelectorAll("main details").forEach(function(d){{d.open=true}})}})</script>
</body>
</html>
"""


def home_body(ctx: dict) -> str:
    cards = []
    def collect(node: dict) -> list[tuple[tuple, Page]]:
        found = [(sort_key(Path(p.rel).stem, p.meta.get("order"), p.title, ctx["nav_order"]), p) for p in node["pages"]]
        for name, sub in node["dirs"].items():
            if sub["index"]:
                found.append((sort_key(name, sub["index"].meta.get("order"), name, ctx["nav_order"]), sub["index"]))
        return found
    for _, p in sorted(collect(ctx["tree"]), key=lambda t: t[0]):
        status = p.meta.get("status", "")
        badge = f'<span class="badge {re.sub(r"[^a-z]+", "-", status.lower())}">{html.escape(status)}</span>' if status else ""
        cards.append(f'<a class="card" href="{p.out}"><strong>{html.escape(p.title)}</strong>{badge}'
                     f'<span>{html.escape(p.meta.get("summary", ""))}</span></a>')
    tag = f"<p class=lead>{html.escape(ctx['tagline'])}</p>" if ctx["tagline"] else ""
    return f'<h1>{html.escape(ctx["project"])}</h1>{tag}<div class="cards">{"".join(cards)}</div>'


def make_css(accent: str, font: str) -> str:
    face = ""
    if font:
        face = f'@font-face{{font-family:"HubFont";src:url("{posixpath.basename(font)}");font-display:swap}}\n'
    return face + CSS.replace("__ACCENT__", accent).replace(
        "__FONT__", '"HubFont", ' if font else "")


def build(root: Path, src: Path, out: Path, check_only: bool, rep: Report) -> None:
    cfg: dict = {}
    cfg_file = src / "hub.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            rep.error(f"hub.json is not valid JSON: {e}")
    project = cfg.get("project_name") or root.name
    nav_order = [str(s) for s in cfg.get("nav_order", [])]
    hidden = {str(s) for s in cfg.get("hide_folders", [])}
    accent = cfg.get("accent") or "#2f6f8f"
    if not re.match(r"^#[0-9a-fA-F]{3,8}$", accent):
        rep.warn(f"accent '{accent}' is not a #hex colour; using the default")
        accent = "#2f6f8f"

    pages = load_pages(src, hidden, rep)
    if not pages:
        rep.error(f"no markdown pages found in {src}")
        return
    known = {Path(p.rel).stem for p in pages.values()} | {Path(p.rel).parts[0] for p in pages.values() if "/" in p.rel}
    for name in nav_order:
        if name not in known:
            rep.warn(f"hub.json nav_order lists '{name}' but no such page or folder exists")

    home = pages.get("index.md")
    if not home:
        home = Page(src / "index.md", "index.md", "index.html", {}, "", project)
        home.rel = "(home)"
    tree = build_tree({k: v for k, v in pages.items() if v is not home}, nav_order)
    ctx = {"project": project, "tagline": cfg.get("tagline", ""), "nav_order": nav_order, "pages": pages, "tree": tree,
           "built": datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": git_commit(root)}

    results: list[tuple[Page, str]] = []
    assets: set[str] = set()
    for p in pages.values():
        c = Ctx(p, pages, src, rep)
        results.append((p, render_blocks(p.body.splitlines(), c)))
        assets |= c.assets
    if home.rel == "(home)":
        results.append((home, home_body(ctx)))

    if check_only:
        return
    if out.exists():
        shutil.rmtree(out)
    for p, body in results:
        dest = out / p.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page_html(p, body, ctx), encoding="utf-8")
    for f in src.rglob("*"):                                   # copy images and other files as they are
        rel = f.relative_to(src)
        if f.is_dir() or f.suffix.lower() == ".md" or f.name == "hub.json" or any(
                part.startswith(".") or part in hidden for part in rel.parts):
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
    font = cfg.get("font", "")
    if font and not (src / font).exists():
        rep.warn(f"hub.json font '{font}' does not exist")
        font = ""
    css = out / CSS_PATH
    css.parent.mkdir(parents=True, exist_ok=True)
    css.write_text(make_css(accent, font), encoding="utf-8")
    if font:
        shutil.copy2(src / font, css.parent / posixpath.basename(font))
    print(f"built {len(results)} pages into {out}")


def serve(out: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"serving {out} at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the project hub (docs/hub -> docs/site).")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="project root")
    ap.add_argument("--src", type=Path, help="markdown folder (default docs/hub)")
    ap.add_argument("--out", type=Path, help="output folder (default docs/site)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if there were errors")
    ap.add_argument("--check", action="store_true", help="checks only, write nothing; exit 1 on errors")
    ap.add_argument("--allow-examples", action="store_true", help="do not error on pages still marked 'example: true' (template testing only)")
    ap.add_argument("--serve", nargs="?", const=8000, type=int, metavar="PORT", help="serve the built site")
    a = ap.parse_args()
    global ALLOW_EXAMPLES
    ALLOW_EXAMPLES = a.allow_examples
    root = a.root.resolve()
    src = (a.src or root / "docs" / "hub").resolve()
    out = (a.out or root / "docs" / "site").resolve()
    if not src.is_dir():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 1
    if out == src or src in out.parents or out in src.parents or out == root:
        print("error: --out must be a separate folder that does not contain the sources", file=sys.stderr)
        return 1
    rep = Report()
    build(root, src, out, a.check, rep)
    print(f"{len(rep.warnings)} warnings, {len(rep.errors)} errors")
    if (a.strict or a.check) and rep.errors:
        return 1
    if a.serve and not a.check:
        serve(out, a.serve)
    return 0


CSS = """\
:root{--bg:#fbfaf8;--panel:#f1eee9;--card:#fff;--text:#24211d;--muted:#6d665e;--line:#ddd7ce;--accent:__ACCENT__;
--link:var(--accent);--code:#efebe4;--note:#e8f1f6;--warn:#fbeccd;--tip:#e4f2e6;
--p1:#b3261e;--p2:#a0600a;--p3:#2f6b8f;--ok:#2e7d32;--shadow:0 1px 2px rgba(0,0,0,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#171614;--panel:#1f1d1b;--card:#22201d;--text:#e8e4dd;--muted:#a39b90;
--line:#37332e;--link:color-mix(in srgb,var(--accent) 55%,white);--code:#2a2724;--note:#1c2a33;--warn:#33291a;--tip:#1d2e20;
--p1:#f08a82;--p2:#e8b25a;--p3:#7fb6d9;--ok:#7bc47f;--shadow:none}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 __FONT__system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
a{color:var(--link);text-underline-offset:2px}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:100vh}
.side{background:var(--panel);border-right:1px solid var(--line);padding:22px 18px;position:sticky;top:0;height:100vh;overflow:auto;font-size:.93rem}
.brand{display:block;font-weight:700;font-size:1.15rem;color:var(--text);text-decoration:none;letter-spacing:-.01em}
.tagline{color:var(--muted);font-size:.85rem;margin:2px 0 18px}
.side nav{margin-top:16px}
.side ul{list-style:none;margin:0;padding:0}
.side ul ul{margin-left:12px;border-left:1px solid var(--line);padding-left:8px}
.side li a,.side summary{display:block;padding:4px 8px;border-radius:6px;color:var(--text);text-decoration:none}
.side summary{cursor:pointer;font-weight:600;color:var(--muted);text-transform:uppercase;font-size:.74rem;letter-spacing:.06em;margin-top:8px}
.side summary a{display:inline;padding:0;color:inherit}
.side li a:hover{background:color-mix(in srgb,var(--accent) 12%,transparent)}
.side a.current{background:color-mix(in srgb,var(--accent) 18%,transparent);font-weight:600}
main{padding:28px clamp(18px,4vw,56px) 40px;max-width:60rem;width:100%}
.crumbs{font-size:.82rem;color:var(--muted);margin-bottom:14px}
.crumbs a{color:var(--muted)}.crumbs i{font-style:normal;opacity:.5;margin:0 3px}
h1,h2,h3,h4{line-height:1.25;letter-spacing:-.01em;scroll-margin-top:16px}
h1{font-size:2rem;margin:.2em 0 .5em}
h2{font-size:1.45rem;margin:1.8em 0 .5em;padding-bottom:.25em;border-bottom:1px solid var(--line)}
h3{font-size:1.15rem;margin:1.5em 0 .4em}
h4{font-size:1rem;margin:1.2em 0 .3em}
.anchor{margin-left:.4em;font-size:.8em;opacity:0;text-decoration:none;color:var(--muted)}
h1:hover .anchor,h2:hover .anchor,h3:hover .anchor,h4:hover .anchor{opacity:.7}
p{margin:.7em 0}.lead{font-size:1.15rem;color:var(--muted)}
code{background:var(--code);padding:.12em .38em;border-radius:5px;font:.88em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:10px;overflow:auto;line-height:1.45;border:1px solid var(--line)}
pre code{background:none;padding:0;font-size:.85rem}
ul,ol{padding-left:1.5em}li{margin:.25em 0}li>ul,li>ol{margin:.2em 0}
li.task{list-style:none;margin-left:-1.3em}li.task input{margin-right:.5em;accent-color:var(--accent)}
li.task.done{color:var(--muted)}
img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line)}
hr{border:0;border-top:1px solid var(--line);margin:2em 0}
.tablewrap{overflow-x:auto;margin:1em 0;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.93rem}
th,td{padding:8px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{background:var(--panel);font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
tr:last-child td{border-bottom:0}
blockquote{margin:1em 0;padding:.4em 1em;border-left:4px solid var(--line);color:var(--muted)}
blockquote.callout{border-radius:0 10px 10px 0;color:var(--text);background:var(--note);border-left-color:var(--p3)}
blockquote.warning{background:var(--warn);border-left-color:var(--p2)}
blockquote.tip{background:var(--tip);border-left-color:var(--ok)}
details{margin:.6em 0}
summary{cursor:pointer}
main details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.5em 1em;box-shadow:var(--shadow)}
main details>summary{font-weight:600;padding:.2em 0}
main details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:.6em}
main details details{background:var(--bg)}
.frame iframe{width:100%;min-height:360px;border:1px solid var(--line);border-radius:10px}
.badge{display:inline-block;font-size:.72em;font-weight:700;line-height:1;padding:.28em .6em;border-radius:99px;vertical-align:.08em;
letter-spacing:.03em;text-transform:uppercase;background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--muted)}
.badge.p1,.badge.blocked{background:color-mix(in srgb,var(--p1) 16%,transparent);color:var(--p1)}
.badge.p2,.badge.open,.badge.in-progress,.badge.draft,.badge.proposed{background:color-mix(in srgb,var(--p2) 16%,transparent);color:var(--p2)}
.badge.p3{background:color-mix(in srgb,var(--p3) 16%,transparent);color:var(--p3)}
.badge.done,.badge.decided,.badge.accepted{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px;margin-top:22px}
.card{display:flex;flex-direction:column;gap:6px;padding:16px;background:var(--card);border:1px solid var(--line);border-radius:12px;
text-decoration:none;color:var(--text);box-shadow:var(--shadow)}
.card:hover{border-color:var(--accent)}.card span:last-child{color:var(--muted);font-size:.88rem}
footer{margin-top:3em;padding-top:1em;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem}
.top,.navtoggle{display:none}
@media (max-width:820px){
.layout{display:block}
.top{display:flex;gap:12px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.burger{font-size:1.4rem;cursor:pointer;line-height:1}
.side{display:none;position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
.side>.brand,.side>.tagline{display:none}.side nav{margin:0}
.navtoggle:checked~.layout .side{display:block}
main{padding:20px 16px 32px}}
@media print{
.side,.top,.anchor{display:none}.layout{display:block}
body{background:#fff;color:#000;font-size:11pt}main{max-width:none;padding:0}
main details{break-inside:avoid;box-shadow:none}a{color:inherit}pre{white-space:pre-wrap}}
"""

if __name__ == "__main__":
    sys.exit(main())
