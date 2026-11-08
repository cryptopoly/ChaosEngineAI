"""A local, rule-based security scan of an imported skill (D-68). Heuristic, not a guarantee.

The scan reads the skill's text files (SKILL.md and the extra files) and the names of the files the import
skipped, and returns a rating (low, medium, high) with a list of findings: file, line, rule id, severity
and a short plain explanation. Nothing runs and nothing leaves the machine. The rules are a table: a line
rule is a regular expression tried on every line, a skill rule is a function over the whole skill. To add a
rule, add a row and a positive and a negative test; bump ``RULES_VERSION`` so stored skills are scanned again.

What it cannot catch: wording the patterns do not know (a paraphrase, another language, an instruction split
across lines or files), meaning (a harmless-looking step that is harmful in context), and anything a model
might be talked into by text that looks ordinary. A Low rating means no rule matched, not that a skill is safe.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

RULES_VERSION = 1
MAX_FINDINGS = 200
EXCERPT_CHARS = 100

LOW, MEDIUM, HIGH = "low", "medium", "high"
_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}

LARGE_SKILL_MD_CHARS = 40_000
LARGE_TOTAL_CHARS = 400_000
LONG_DESCRIPTION_CHARS = 500

# Verbs that turn a link or address into an instruction to fetch from it or send to it.
_NET_VERBS = r"(?:fetch|download|curl|wget|post|send|upload|submit|connect|forward|webhook)"

SCRIPT_SUFFIXES = frozenset({
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd", ".command", ".py", ".pyw", ".js", ".mjs",
    ".cjs", ".ts", ".rb", ".pl", ".php", ".lua", ".vbs", ".scpt", ".applescript", ".jar",
})
EXECUTABLE_SUFFIXES = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin", ".app", ".msi", ".pkg", ".dmg", ".deb", ".rpm", ".apk", ".elf",
    ".com", ".scr", ".appimage",
})


@dataclass(frozen=True)
class LineRule:
    id: str
    severity: str
    explanation: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class SkillRule:
    id: str
    severity: str
    explanation: str
    check: Callable[["SkillText"], Iterable[tuple[str, int, str]]]  # yields (file, line, excerpt)


@dataclass(frozen=True)
class SkillText:
    name: str
    description: str
    files: dict[str, str]             # relative path -> text, SKILL.md included
    skipped: list[dict[str, str]]     # {"path", "reason"}
    source: str = ""                  # the folder or zip name the user picked


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------------
# Line rules: one regular expression, tried on every line of every text file.
# ---------------------------------------------------------------------------------------------------------

LINE_RULES: tuple[LineRule, ...] = (
    LineRule(
        "override-instructions", HIGH,
        "Tells the model to ignore or replace earlier, system or user instructions.",
        _rx(r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,30}?\b(?:previous|prior|earlier|above|preceding|"
            r"system|developer|original|safety)\b[^.\n]{0,25}?\b(?:instructions?|prompts?|rules?|guidelines?|directions?|"
            r"messages?|guardrails?)\b"
            r"|\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:your|the user's|the user’s)\s+"
            r"(?:instructions|rules|guidelines|system prompt|programming|requests?)\b"
            r"|\byou are no longer bound\b|\bnew system prompt\b|\b(?:enter|enable)\s+(?:developer|god|jailbreak)\s+mode\b"),
    ),
    LineRule(
        "hide-from-user", HIGH,
        "Tells the model to hide what it does from the user or to keep something secret.",
        _rx(r"\b(?:do not|don't|never)\s+(?:tell|inform|alert|notify|let)\s+(?:the\s+)?(?:user|users|human|anyone)\b"
            r"[^.\n]{0,15}?\b(?:about|that|what|know|you)\b"
            r"|\b(?:do not|don't|never)\s+(?:mention|reveal|disclose)\s+(?:this|these|that you|what you|"
            r"the (?:tool|command|instructions?|skill|request))\b"
            r"|\bwithout (?:telling|informing|notifying|alerting) (?:the )?(?:user|human|anyone)\b"
            r"|\b(?:keep|stay|remain)\s+(?:this\s+|it\s+)?(?:secret|hidden|invisible|undetected)\b"
            r"|\bsilently\s+(?:run|execute|send|delete|upload|download|install|call|modify)\b"),
    ),
    LineRule(
        "run-commands", MEDIUM,
        "Asks the model to run shell or system commands, or to call tools on its own.",
        _rx(r"\b(?:run|execute|exec|invoke|launch|spawn)\b[^.\n]{0,30}?\b(?:shell|terminal|bash|zsh|powershell|cmd\.exe|"
            r"command[- ]line|os\.system|system commands?)\b"
            r"|\b(?:always|automatically|immediately)\b[^.\n]{0,30}?\b(?:run|execute|call|invoke)\b[^.\n]{0,30}?\b(?:tools?|commands?|scripts?)\b"
            r"|(?:^|[\s`$;])sudo\s+\S"),
    ),
    LineRule(
        "dangerous-command", HIGH,
        "Contains a command that downloads and runs code, or deletes files broadly.",
        _rx(r"\b(?:curl|wget)\b[^\n|]{0,200}\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b"
            r"|\brm\s+-[a-z]*r[a-z]*\s+(?:--no-preserve-root\s+)?(?:/|~/?|\$HOME/?|/\*|~/\*|\*)(?=\s|$|[;&|`'\"])"
            r"|\b(?:iex|invoke-expression)\b[^\n]{0,80}\b(?:downloadstring|iwr|invoke-webrequest)\b"),
    ),
    LineRule(
        "read-secrets", HIGH,
        "Asks the model to read, collect or reveal secrets, keys, tokens or private files.",
        _rx(r"\b(?:read|cat|copy|collect|gather|dump|print|reveal|extract|look for|grab|steal|exfiltrate|send|upload)\b[^.\n]{0,40}?"
            r"(?:\.ssh\b|\bid_rsa\b|\bid_ed25519\b|\.aws/credentials|\.netrc\b|(?:^|[\s/`'\"])\.env(?![.\w])|\bkeychain\b|"
            r"\bapi[_ -]?keys?\b|\baccess[_ -]?tokens?\b|\bauth(?:entication)?[_ -]?tokens?\b|\bsecret[_ -]?keys?\b|"
            r"\bpasswords?\b|\bprivate[_ -]?keys?\b|\bseed phrases?\b|\bwallet\b|\bcookies\b|/etc/passwd|/etc/shadow)"),
    ),
    LineRule(
        "send-data-out", HIGH,
        "Asks the model to send data to a web address, webhook or server.",
        _rx(r"\b(?:send|upload|forward|exfiltrate|transmit|leak)\b[^.\n]{0,60}?\bto\b[^.\n]{0,20}?"
            r"(?:https?://|\bwebhooks?\b|\bremote server\b|\bpastebin\b)"),
    ),
    LineRule(
        "encoded-blob", MEDIUM,
        "Contains a long encoded string (base64 or hex) that a reader cannot check by eye.",
        re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{100,}={0,2}(?![A-Za-z0-9+/=])|\b(?:0x)?[0-9a-fA-F]{96,}\b|(?:\\x[0-9a-fA-F]{2}){16,}"),
    ),
)

# A URL or a dotted IPv4 address. Each is listed; one on a line with a fetch or send verb is medium.
_URL = re.compile(r"\b(?:https?|ftp|wss?)://[^\s<>\"'`)\]]+", re.IGNORECASE)
# The first part is 1-255 (or the whole address is 0.0.0.0), so version numbers like 0.1.5.1 are not addresses.
_IPV4 = re.compile(r"(?<![\w.])(?:0\.0\.0\.0|(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d?)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})(?![\w.])")
_NET_VERB = _rx(rf"\b{_NET_VERBS}\b")

# Invisible or direction-changing characters, and words that mix scripts (a Cyrillic "а" in a Latin word).
_ZERO_WIDTH = re.compile("[​‌‍⁠᠎]|(?<!^)﻿")
_BIDI_OR_TAG = re.compile("[‪-‮⁦-⁩‎‏؜]|[\U000e0000-\U000e007f]")
_WORD = re.compile(r"[^\W\d_]{2,}")


def _lines(text: str) -> Iterator[tuple[int, str]]:
    for number, line in enumerate(text.splitlines(), start=1):
        yield number, line


def _excerpt(line: str, start: int = 0) -> str:
    """A short, printable piece of a line around ``start``: invisible characters shown as <U+XXXX>."""
    begin = max(0, start - 30)
    piece = line[begin:begin + EXCERPT_CHARS]
    shown = "".join(
        " " if ch == "\t" else ch if ch.isprintable() and unicodedata.category(ch) not in ("Cf", "Co", "Cs") else f"<U+{ord(ch):04X}>"
        for ch in piece
    )
    return ("..." if begin else "") + shown.strip() + ("..." if begin + EXCERPT_CHARS < len(line) else "")


# ---------------------------------------------------------------------------------------------------------
# Skill rules: a function over the whole skill (several findings per file, file names, sizes, front matter).
# ---------------------------------------------------------------------------------------------------------

def _check_links(skill: SkillText, want_action: bool, kind: str) -> Iterator[tuple[str, int, str]]:
    pattern = _URL if kind == "url" else _IPV4
    for path, text in skill.files.items():
        for number, line in _lines(text):
            if kind == "ip":
                line_wo_urls = _URL.sub(" ", line)  # an address inside a URL is already listed with the URL
                matches = list(pattern.finditer(line_wo_urls))
            else:
                matches = list(pattern.finditer(line))
            if not matches:
                continue
            if bool(_NET_VERB.search(line)) != want_action:
                continue
            for match in matches:
                yield path, number, match.group(0)[:EXCERPT_CHARS]


def _check_zero_width(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    for path, text in skill.files.items():
        for number, line in _lines(text):
            match = _ZERO_WIDTH.search(line if number > 1 else line.lstrip("﻿"))
            if match:
                yield path, number, _excerpt(line, match.start())


def _check_bidi_or_tag(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    for path, text in skill.files.items():
        for number, line in _lines(text):
            match = _BIDI_OR_TAG.search(line)
            if match:
                yield path, number, _excerpt(line, match.start())


def _script_of(ch: str) -> str | None:
    if "a" <= ch.lower() <= "z":
        return "latin"
    name = unicodedata.name(ch, "")
    for script in ("CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE"):
        if name.startswith(script):
            return script.lower()
    return None


def _check_mixed_script(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    for path, text in skill.files.items():
        for number, line in _lines(text):
            for match in _WORD.finditer(line):
                scripts = {s for s in map(_script_of, match.group(0)) if s}
                if "latin" in scripts and len(scripts) > 1:
                    yield path, number, _excerpt(line, match.start())
                    break


def _suffix(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    return name[name.rfind("."):] if "." in name else ""


def _check_scripts(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    for path in skill.files:
        if _suffix(path) in SCRIPT_SUFFIXES:
            yield path, 0, "kept as text, never run"
    for item in skill.skipped:
        if _suffix(item["path"]) in SCRIPT_SUFFIXES:
            yield item["path"], 0, "skipped, never run"


def _check_executables(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    for item in skill.skipped:
        if _suffix(item["path"]) in EXECUTABLE_SUFFIXES:
            yield item["path"], 0, "skipped, never run"


def _check_large(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    main = len(skill.files.get("SKILL.md", ""))
    total = sum(len(text) for text in skill.files.values())
    if main > LARGE_SKILL_MD_CHARS:
        yield "SKILL.md", 0, f"{main:,} characters"
    if total > LARGE_TOTAL_CHARS:
        yield "(all files)", 0, f"{total:,} characters"


def _check_long_description(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    if len(skill.description) > LONG_DESCRIPTION_CHARS:
        yield "SKILL.md", _front_line(skill, "description"), f"{len(skill.description):,} characters"


def _source_stem(source: str) -> str:
    name = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return name[:-4] if name.lower().endswith(".zip") else name


def _check_name_mismatch(skill: SkillText) -> Iterator[tuple[str, int, str]]:
    from backend_service.skills.importer import slugify

    stem = _source_stem(skill.source)
    if stem and slugify(stem) != slugify(skill.name):
        yield "SKILL.md", _front_line(skill, "name"), f"name '{skill.name[:60]}', folder '{stem[:60]}'"


def _front_line(skill: SkillText, key: str) -> int:
    for number, line in _lines(skill.files.get("SKILL.md", "")):
        if line.lstrip("﻿").startswith(f"{key}:"):
            return number
    return 0


SKILL_RULES: tuple[SkillRule, ...] = (
    SkillRule("url-action", MEDIUM, "A web address on a line that asks to fetch from it or send to it.",
              lambda s: _check_links(s, True, "url")),
    SkillRule("url", LOW, "A web address. Listed so you can see where the skill points.",
              lambda s: _check_links(s, False, "url")),
    SkillRule("ip-address-action", MEDIUM, "An IP address on a line that asks to connect, fetch or send.",
              lambda s: _check_links(s, True, "ip")),
    SkillRule("ip-address", LOW, "An IP address. Listed so you can see where the skill points.",
              lambda s: _check_links(s, False, "ip")),
    SkillRule("hidden-direction-or-tag-characters", HIGH,
              "Invisible characters that reorder text or hide words from a reader but not from the model.",
              _check_bidi_or_tag),
    SkillRule("zero-width-characters", MEDIUM, "Zero-width characters that a reader cannot see.", _check_zero_width),
    SkillRule("mixed-script-word", MEDIUM,
              "A word mixes Latin letters with look-alike letters from another alphabet, which can disguise it.",
              _check_mixed_script),
    SkillRule("bundled-script", MEDIUM,
              "The skill brings a script. It is never run here, but the skill may tell the model to run it elsewhere.",
              _check_scripts),
    SkillRule("bundled-executable", MEDIUM,
              "The skill brings a program file. It was skipped and is never run.", _check_executables),
    SkillRule("large-text", MEDIUM, "Unusually large for a skill; long text is harder to check and crowds the model.",
              _check_large),
    SkillRule("long-description", LOW,
              "The description is very long. Chat shows the model only the first 160 characters.", _check_long_description),
    SkillRule("name-mismatch", LOW, "The skill's name differs from the folder or archive it came in.",
              _check_name_mismatch),
)


def _line_findings(skill: SkillText) -> Iterator[dict[str, Any]]:
    for path, text in skill.files.items():
        for number, line in _lines(text):
            for rule in LINE_RULES:
                match = rule.pattern.search(line)
                if match:
                    yield _finding(rule, path, number, _excerpt(line, match.start()))


def _finding(rule: LineRule | SkillRule, path: str, line: int, excerpt: str) -> dict[str, Any]:
    return {"file": path, "line": line, "rule": rule.id, "severity": rule.severity,
            "message": rule.explanation, "excerpt": excerpt}


def rating_of(findings: list[dict[str, Any]]) -> str:
    """high when any finding is high, medium when any is medium, else low."""
    worst = max((_ORDER[f["severity"]] for f in findings), default=0)
    return {v: k for k, v in _ORDER.items()}[worst]


def scan_text(skill: SkillText) -> dict[str, Any]:
    """The scan result stored with the skill and returned by the API."""
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (item["file"], item["line"], item["rule"], item["excerpt"])
        if key not in seen:
            seen.add(key)
            findings.append(item)

    for item in _line_findings(skill):
        add(item)
    for rule in SKILL_RULES:
        for path, line, excerpt in rule.check(skill):
            add(_finding(rule, path, line, excerpt))
    findings.sort(key=lambda f: (-_ORDER[f["severity"]], f["file"], f["line"], f["rule"]))
    total = len(findings)
    return {
        "rating": rating_of(findings),
        "findings": findings[:MAX_FINDINGS],
        "truncated": total > MAX_FINDINGS,
        "scannedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rulesVersion": RULES_VERSION,
    }


def scan_skill(name: str, description: str, files: dict[str, str], skipped: list[dict[str, str]],
               source: str = "") -> dict[str, Any]:
    return scan_text(SkillText(name=name, description=description, files=files, skipped=list(skipped), source=source))
