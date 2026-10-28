#!/usr/bin/env python3
"""Pre-release dependency audit: known vulnerabilities and out-of-date packages.

Reads the exact versions the app ships and tests with:

  * ``release-constraints.txt``  Python packages (release builds install with it)
  * ``src-tauri/Cargo.lock``     Rust crates
  * ``package-lock.json``        npm packages

and asks osv.dev (the open vulnerability database behind GitHub advisories,
PyPA and RustSec) whether any pinned version has a published advisory. Then it
compares the pins with the newest releases and reports how far behind they are.

    .venv/bin/python scripts/audit-dependencies.py              # gate: fail on high / critical
    .venv/bin/python scripts/audit-dependencies.py --fail-on moderate
    .venv/bin/python scripts/audit-dependencies.py --no-outdated   # advisories only
    .venv/bin/python scripts/audit-dependencies.py --json report.json

Exit code 0 = nothing blocking, 1 = a shipped dependency has an advisory at or
above the threshold, 2 = the audit could not run (no network). Out-of-date
packages are reported but never block.

Dev-only npm packages (vite, vitest, ...) never ship, so their advisories are
warnings unless ``--include-dev`` is given.

An advisory with no fix yet (or one we accept) goes in
``scripts/dependency-audit-ignore.json`` by id, with a reason and an expiry date;
the entry stops working on that date so the decision gets revisited. Entries
name one advisory, so a new advisory against the same package still blocks.

Standard library only, so it runs on any machine with Python 3.9 or newer.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "release-constraints.txt"
CARGO_LOCK = ROOT / "src-tauri" / "Cargo.lock"
CARGO_TOML = ROOT / "src-tauri" / "Cargo.toml"
PACKAGE_JSON = ROOT / "package.json"
PACKAGE_LOCK = ROOT / "package-lock.json"
IGNORE_FILE = ROOT / "scripts" / "dependency-audit-ignore.json"

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/"
USER_AGENT = "ChaosEngineAI-dependency-audit (https://github.com/cryptopoly/ChaosEngineAI)"

SEVERITY_RANK = {"none": 0, "unknown": 1, "low": 2, "moderate": 3, "high": 4, "critical": 5}
FAIL_LEVELS = ("low", "moderate", "high", "critical")


class AuditError(RuntimeError):
    """The audit could not be completed (network or an unreadable source)."""


@dataclass(frozen=True)
class Dep:
    ecosystem: str  # OSV names: PyPI, crates.io, npm
    name: str
    version: str
    dev: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}=={self.version}" if self.ecosystem == "PyPI" else f"{self.name}@{self.version}"


@dataclass
class Flaw:
    """One vulnerability. GHSA / PYSEC / CVE / RUSTSEC ids that name the same flaw are one Flaw."""

    ids: list[str]  # the record id first, then its aliases
    severity: str
    summary: str
    fixed_in: str | None
    notice: str = ""  # "unmaintained", "unsound", ... for informational advisories
    ignored_reason: str | None = None
    ignored_until: str | None = None
    ignore_expired: bool = False


@dataclass
class Finding:
    """Everything osv.dev knows about one pinned version."""

    dep: Dep
    flaws: list[Flaw]

    @property
    def severity(self) -> str:
        return max((f.severity for f in self.flaws), key=lambda s: SEVERITY_RANK[s], default="none")


# --------------------------------------------------------------------------
# Reading the pinned versions
# --------------------------------------------------------------------------


def read_python_pins(path: Path = CONSTRAINTS) -> list[Dep]:
    deps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if match:
            deps.append(Dep("PyPI", match.group(1), match.group(2)))
    return deps


def read_cargo_pins(path: Path = CARGO_LOCK) -> list[Dep]:
    """Registry crates only; path and git dependencies have nothing to look up."""
    deps = []
    for block in path.read_text(encoding="utf-8").split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        version = re.search(r'^version = "([^"]+)"', block, re.M)
        source = re.search(r'^source = "registry\+', block, re.M)
        if name and version and source:
            deps.append(Dep("crates.io", name.group(1), version.group(1)))
    return deps


def read_npm_pins(path: Path = PACKAGE_LOCK) -> list[Dep]:
    packages = json.loads(path.read_text(encoding="utf-8")).get("packages", {})
    seen: dict[tuple[str, str], Dep] = {}
    for key, info in packages.items():
        if not key or info.get("link") or not info.get("version"):
            continue
        name = key.rsplit("node_modules/", 1)[-1]
        dep = Dep("npm", name, info["version"], dev=bool(info.get("dev")))
        existing = seen.get((name, dep.version))
        # A package installed for both shipped and dev code counts as shipped.
        if existing is None or (existing.dev and not dep.dev):
            seen[(name, dep.version)] = dep
    return list(seen.values())


def read_direct_npm(package_json: Path = PACKAGE_JSON) -> set[str]:
    data = json.loads(package_json.read_text(encoding="utf-8"))
    return set(data.get("dependencies", {})) | set(data.get("devDependencies", {}))


def read_direct_crates(cargo_toml: Path = CARGO_TOML) -> set[str]:
    names: set[str] = set()
    in_deps = False
    for raw in cargo_toml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_deps = bool(re.match(r"^\[(?:target\..+\.)?(?:dev-|build-)?dependencies\]$", line))
            continue
        match = re.match(r"^([A-Za-z0-9_\-]+)\s*=", line)
        if in_deps and match:
            names.add(match.group(1))
    return names


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    try:  # python.org builds on macOS ship without a CA bundle; certifi has one
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the platform store
        return ssl.create_default_context()


def http_json(url: str, payload: dict[str, Any] | None = None, retries: int = 3, timeout: float = 30.0) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
        if attempt + 1 < retries:
            import time

            time.sleep(1.5 * (attempt + 1))
    hint = " (this Python has no CA bundle; run it with the project's .venv/bin/python)" if "CERTIFICATE_VERIFY_FAILED" in str(last) else ""
    raise AuditError(f"could not reach {urllib.parse.urlparse(url).netloc}: {last}{hint}")


def query_advisories(deps: list[Dep], fetch: Callable[..., Any] = http_json, chunk: int = 500) -> dict[Dep, list[str]]:
    """Advisory ids per dependency, from OSV's batch endpoint."""
    found: dict[Dep, list[str]] = {}
    for start in range(0, len(deps), chunk):
        batch = deps[start:start + chunk]
        queries = [{"package": {"name": d.name, "ecosystem": d.ecosystem}, "version": d.version} for d in batch]
        response = fetch(OSV_BATCH, {"queries": queries})
        results = (response or {}).get("results")
        if not isinstance(results, list) or len(results) != len(batch):
            raise AuditError("osv.dev returned an unexpected batch response")
        for dep, result in zip(batch, results):
            ids = [v["id"] for v in (result or {}).get("vulns", []) if v.get("id")]
            if ids:
                found[dep] = ids
    return found


def fetch_records(ids: Iterable[str], fetch: Callable[..., Any] = http_json) -> dict[str, dict[str, Any]]:
    unique = sorted(set(ids))
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(lambda i: (i, fetch(OSV_VULN + urllib.parse.quote(i, safe=""))), unique))
    return {i: r for i, r in records if isinstance(r, dict)}


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------


def version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for token in re.split(r"[.+-]", raw.lstrip("vV=^~")):
        digits = re.match(r"\d+", token)
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def is_newer(candidate: str, installed: str) -> bool:
    return version_tuple(candidate) > version_tuple(installed)


def behind_kind(installed: str, latest: str) -> str | None:
    """'major', 'minor', 'patch' or None when the pin is current."""
    a, b = version_tuple(installed), version_tuple(latest)
    if not a or not b or b <= a:
        return None
    if b[0] != a[0]:
        return "major"
    if len(a) > 1 and len(b) > 1 and b[1] != a[1]:
        return "minor"
    return "patch"


# --------------------------------------------------------------------------
# Reading advisory records
# --------------------------------------------------------------------------


_CVSS_WEIGHTS = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "CIA": {"H": 0.56, "L": 0.22, "N": 0.0},
}


def cvss3_score(vector: str) -> float | None:
    """CVSS v3.x base score from a vector string, or None if it is not a v3 vector."""
    if not vector.startswith(("CVSS:3.0/", "CVSS:3.1/")):
        return None
    try:
        m = dict(part.split(":", 1) for part in vector.split("/")[1:])
        changed = m["S"] == "C"
        privileges = {"N": 0.85, "L": 0.68 if changed else 0.62, "H": 0.5 if changed else 0.27}[m["PR"]]
        iss = 1 - (1 - _CVSS_WEIGHTS["CIA"][m["C"]]) * (1 - _CVSS_WEIGHTS["CIA"][m["I"]]) * (1 - _CVSS_WEIGHTS["CIA"][m["A"]])
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15 if changed else 6.42 * iss
        exploitability = 8.22 * _CVSS_WEIGHTS["AV"][m["AV"]] * _CVSS_WEIGHTS["AC"][m["AC"]] * privileges * _CVSS_WEIGHTS["UI"][m["UI"]]
    except (KeyError, ValueError):
        return None
    if impact <= 0:
        return 0.0
    total = min((1.08 if changed else 1.0) * (impact + exploitability), 10.0)
    scaled = round(total * 100000)  # round up to one decimal the way the CVSS 3.1 spec does
    return scaled / 100000.0 if scaled % 10000 == 0 else (scaled // 10000 + 1) / 10.0


def cvss_label(score: float) -> str:
    return "critical" if score >= 9.0 else "high" if score >= 7.0 else "moderate" if score >= 4.0 else "low" if score > 0 else "none"


def record_severity(record: dict[str, Any]) -> tuple[str, str]:
    """(severity, notice). Severity is none / unknown / low / moderate / high / critical.

    GitHub advisories carry a label; RustSec and PyPA records carry a CVSS vector.
    """
    spec = record.get("database_specific") or {}
    label = str(spec.get("severity") or "").lower()
    notice = str(spec.get("informational") or "")
    for affected in record.get("affected", []):
        notice = notice or str((affected.get("database_specific") or {}).get("informational") or "")
        label = label or str((affected.get("database_specific") or {}).get("severity") or "").lower()
    if label == "medium":
        label = "moderate"
    if label in SEVERITY_RANK and label != "unknown":
        return label, notice
    scores = [cvss3_score(str(s.get("score", ""))) for s in record.get("severity", [])]
    scores += [cvss3_score(str((a.get("database_specific") or {}).get("cvss") or "")) for a in record.get("affected", [])]
    known = [s for s in scores if s is not None]
    if known:
        return cvss_label(max(known)), notice
    # RustSec advisories that are about maintenance, not a flaw.
    return ("none" if notice == "unmaintained" else "unknown"), notice


def fixed_version(records: list[dict[str, Any]], dep: Dep) -> str | None:
    """Lowest release that clears every advisory (the highest of each advisory's own fix)."""
    needed: list[str] = []
    for record in records:
        options = []
        for affected in record.get("affected", []):
            if str((affected.get("package") or {}).get("name", "")).lower() != dep.name.lower():
                continue
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    fix = event.get("fixed")
                    if fix and is_newer(fix, dep.version):
                        options.append(fix)
        if options:
            needed.append(min(options, key=version_tuple))
    return max(needed, key=version_tuple) if needed else None


def build_finding(dep: Dep, records: list[dict[str, Any]]) -> Finding | None:
    """Collapse the advisory records of one package version into distinct flaws."""
    live = [r for r in records if not r.get("withdrawn")]
    clusters: list[list[dict[str, Any]]] = []
    for record in live:
        names = {record.get("id", ""), *record.get("aliases", [])} - {""}
        joined = [c for c in clusters if any(names & ({r.get("id", ""), *r.get("aliases", [])}) for r in c)]
        merged = [record]
        for cluster in joined:
            merged += cluster
            clusters.remove(cluster)
        clusters.append(merged)
    flaws = []
    for cluster in clusters:
        top, top_sev, notice = cluster[0], "none", ""
        for record in cluster:
            severity, note = record_severity(record)
            if SEVERITY_RANK[severity] > SEVERITY_RANK[top_sev] or top_sev == "none":
                top, top_sev, notice = record, severity, note
        ids: list[str] = []
        for record in [top, *cluster]:
            for ident in [record.get("id", ""), *record.get("aliases", [])]:
                if ident and ident not in ids:
                    ids.append(ident)
        ids.sort(key=lambda ident: 0 if ident.startswith(("GHSA-", "RUSTSEC-")) else 1)  # the names people search for first
        summary = (top.get("summary") or top.get("details", "")[:100]).strip().replace("\n", " ")
        flaws.append(Flaw(ids, top_sev, summary, fixed_version(cluster, dep), notice))
    flaws.sort(key=lambda f: -SEVERITY_RANK[f.severity])
    return Finding(dep, flaws) if flaws else None


# --------------------------------------------------------------------------
# Accepted risks
# --------------------------------------------------------------------------


def load_ignores(path: Path = IGNORE_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8")).get("ignore", [])
    for entry in entries:
        if not entry.get("id") or not entry.get("reason") or not entry.get("expires"):
            raise AuditError(f"{path.name}: every entry needs an advisory id, a reason and an expires date")
        try:
            dt.date.fromisoformat(str(entry["expires"]))
        except ValueError as exc:
            raise AuditError(f"{path.name}: expires must be YYYY-MM-DD ({entry['id']})") from exc
    return entries


def apply_ignores(findings: list[Finding], ignores: list[dict[str, Any]], today: dt.date) -> list[dict[str, Any]]:
    """Mark accepted flaws. Returns the ignore entries that matched nothing.

    An entry names one advisory id (any alias will do), so a new advisory against
    the same package is never hidden by an old decision.
    """
    used: set[int] = set()
    for finding in findings:
        for flaw in finding.flaws:
            for index, entry in enumerate(ignores):
                if entry["id"] not in flaw.ids:
                    continue
                used.add(index)
                if dt.date.fromisoformat(str(entry["expires"])) >= today:
                    flaw.ignored_reason = entry["reason"]
                    flaw.ignored_until = str(entry["expires"])
                else:
                    flaw.ignore_expired = True
                break
    return [entry for index, entry in enumerate(ignores) if index not in used]


def blocks(dep: Dep, flaw: Flaw, threshold: str, include_dev: bool) -> bool:
    if flaw.ignored_reason or (dep.dev and not include_dev):
        return False
    return SEVERITY_RANK[flaw.severity] >= SEVERITY_RANK[threshold]


# --------------------------------------------------------------------------
# Out-of-date report
# --------------------------------------------------------------------------


def latest_pypi(name: str, fetch: Callable[..., Any] = http_json) -> str | None:
    data = fetch(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    return (data or {}).get("info", {}).get("version")


def latest_npm(name: str, fetch: Callable[..., Any] = http_json) -> str | None:
    data = fetch(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}/latest")
    return (data or {}).get("version")


def latest_crate(name: str, fetch: Callable[..., Any] = http_json) -> str | None:
    data = fetch(f"https://crates.io/api/v1/crates/{urllib.parse.quote(name)}")
    crate = (data or {}).get("crate", {})
    return crate.get("max_stable_version") or crate.get("max_version")


@dataclass
class Outdated:
    dep: Dep
    latest: str
    kind: str


def find_outdated(
    python: list[Dep], crates: list[Dep], npm: list[Dep],
    direct_crates: set[str], direct_npm: set[str], fetch: Callable[..., Any] = http_json,
) -> list[Outdated]:
    """Every pinned Python package, plus the crates and npm packages we depend on directly."""
    jobs: list[tuple[Dep, Callable[..., Any]]] = [(d, latest_pypi) for d in python]
    jobs += [(d, latest_crate) for d in crates if d.name in direct_crates]
    jobs += [(d, latest_npm) for d in npm if d.name in direct_npm]

    def look(job: tuple[Dep, Callable[..., Any]]) -> Outdated | None:
        dep, lookup = job
        try:
            latest = lookup(dep.name, fetch)
        except AuditError:
            return None  # one registry hiccup should not fail the audit
        kind = behind_kind(dep.version, latest) if latest else None
        return Outdated(dep, latest, kind) if kind and latest else None

    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        return [o for o in pool.map(look, jobs) if o]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Report:
    checked: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    unused_ignores: list[dict[str, Any]] = field(default_factory=list)
    outdated: list[Outdated] | None = None
    threshold: str = "high"
    include_dev: bool = False

    def blockers(self) -> list[tuple[Dep, Flaw]]:
        return [
            (f.dep, flaw) for f in self.findings for flaw in f.flaws
            if blocks(f.dep, flaw, self.threshold, self.include_dev)
        ]


def run_audit(
    threshold: str = "high", include_dev: bool = False, check_outdated: bool = True,
    today: dt.date | None = None, fetch: Callable[..., Any] = http_json,
) -> Report:
    python, crates, npm = read_python_pins(), read_cargo_pins(), read_npm_pins()
    report = Report(
        checked={"PyPI": len(python), "crates.io": len(crates), "npm": len(npm)},
        threshold=threshold, include_dev=include_dev,
    )
    hits = query_advisories(python + crates + npm, fetch)
    records = fetch_records([i for ids in hits.values() for i in ids], fetch)
    for dep, ids in hits.items():
        finding = build_finding(dep, [records[i] for i in ids if i in records])
        if finding:
            report.findings.append(finding)
    report.findings.sort(key=lambda f: (-SEVERITY_RANK[f.severity], f.dep.ecosystem, f.dep.name))
    report.unused_ignores = apply_ignores(report.findings, load_ignores(), today or dt.date.today())
    if check_outdated:
        report.outdated = find_outdated(
            python, crates, npm, read_direct_crates(), read_direct_npm(), fetch,
        )
    return report


def render(report: Report) -> str:
    out: list[str] = []
    total = sum(report.checked.values())
    out.append(f"Dependency audit: {total} pinned packages "
               f"({', '.join(f'{n} {k}' for k, n in report.checked.items())}) checked against osv.dev")
    blockers = report.blockers()
    if blockers:
        out.append(f"\nBLOCKING ({len(blockers)}): advisory at or above {report.threshold} in a shipped package")
        for dep, flaw in blockers:
            fix = f"upgrade to {flaw.fixed_in} or newer" if flaw.fixed_in else "no fixed release yet"
            expired = "  [ignore entry expired]" if flaw.ignore_expired else ""
            out.append(f"  {flaw.severity.upper():8} {dep.ecosystem} {dep.label}: {flaw.summary[:90]}")
            out.append(f"           {', '.join(flaw.ids[:4])}{' ...' if len(flaw.ids) > 4 else ''}  ->  {fix}{expired}")
    rest = [(f.dep, flaw) for f in report.findings for flaw in f.flaws if (f.dep, flaw) not in blockers]
    if rest:
        counts: dict[str, int] = {}
        for dep, flaw in rest:
            tag = "accepted" if flaw.ignored_reason else ("dev-only" if dep.dev and not report.include_dev else flaw.severity)
            counts[tag] = counts.get(tag, 0) + 1
        out.append(f"\nNot blocking ({len(rest)}): " + ", ".join(f"{n} {tag}" for tag, n in sorted(counts.items())))
        for dep, flaw in rest:
            why = f"accepted until {flaw.ignored_until}" if flaw.ignored_reason else (flaw.notice or flaw.severity)
            out.append(f"  {dep.ecosystem} {dep.label}: {flaw.summary[:70]} [{why}]")
    if not report.findings:
        out.append("\nNo published advisories for any pinned version.")
    for entry in report.unused_ignores:
        out.append(f"\nNote: ignore entry {entry['id']} matches nothing now; remove it.")
    if report.outdated is not None:
        majors = [o for o in report.outdated if o.kind == "major"]
        out.append(f"\nOut of date (informational): {len(report.outdated)} behind the newest release, "
                   f"{len(majors)} by a major version")
        for o in sorted(report.outdated, key=lambda o: (o.kind != "major", o.dep.ecosystem, o.dep.name))[:25]:
            out.append(f"  {o.kind:5} {o.dep.ecosystem} {o.dep.name} {o.dep.version} -> {o.latest}")
        if len(report.outdated) > 25:
            out.append(f"  ... and {len(report.outdated) - 25} more (use --json for the full list)")
    out.append("\nRESULT: " + (f"FAIL ({len(blockers)} blocking)" if blockers else "PASS"))
    return "\n".join(out)


def to_json(report: Report) -> dict[str, Any]:
    def flaw_json(dep: Dep, flaw: Flaw) -> dict[str, Any]:
        return {
            "ecosystem": dep.ecosystem, "package": dep.name, "version": dep.version, "dev": dep.dev,
            "severity": flaw.severity, "ids": flaw.ids, "summary": flaw.summary, "fixedIn": flaw.fixed_in,
            "notice": flaw.notice, "ignored": flaw.ignored_reason, "ignoredUntil": flaw.ignored_until, "ignoreExpired": flaw.ignore_expired,
            "blocking": blocks(dep, flaw, report.threshold, report.include_dev),
        }

    return {
        "checked": report.checked, "threshold": report.threshold,
        "blocking": len(report.blockers()),
        "findings": [flaw_json(f.dep, flaw) for f in report.findings for flaw in f.flaws],
        "outdated": None if report.outdated is None else [
            {"ecosystem": o.dep.ecosystem, "package": o.dep.name, "version": o.dep.version,
             "latest": o.latest, "behind": o.kind} for o in report.outdated
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fail-on", choices=FAIL_LEVELS, default="high", help="lowest severity that blocks (default: high)")
    parser.add_argument("--include-dev", action="store_true", help="let advisories in dev-only npm packages block too")
    parser.add_argument("--no-outdated", action="store_true", help="skip the newest-release comparison")
    parser.add_argument("--json", metavar="FILE", help="also write the full report as JSON")
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.fail_on, args.include_dev, not args.no_outdated)
    except AuditError as exc:
        print(f"Dependency audit could not run: {exc}", file=sys.stderr)
        print("It needs network access to osv.dev; re-run when online.", file=sys.stderr)
        return 2
    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(report), indent=2) + "\n", encoding="utf-8")
    return 1 if report.blockers() else 0


if __name__ == "__main__":
    sys.exit(main())
