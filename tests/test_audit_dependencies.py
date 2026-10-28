"""Unit tests for ``scripts/audit-dependencies.py`` (no network: OSV answers are faked)."""
from __future__ import annotations

import datetime as dt
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit-dependencies.py"
    spec = importlib.util.spec_from_file_location("audit_dependencies", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_dependencies"] = module
    spec.loader.exec_module(module)
    return module


ad = _load_script()
TODAY = dt.date(2026, 10, 28)


def ghsa(ident: str, severity: str, package: str, fixed: str | None, aliases: list[str] | None = None, summary: str = "flaw") -> dict:
    events = [{"introduced": "0"}] + ([{"fixed": fixed}] if fixed else [])
    return {
        "id": ident, "aliases": aliases or [], "summary": summary,
        "database_specific": {"severity": severity},
        "affected": [{"package": {"name": package}, "ranges": [{"type": "ECOSYSTEM", "events": events}]}],
    }


class ParsingTests(unittest.TestCase):
    def _write(self, text: str, name: str) -> Path:
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_python_pins_skip_comments_and_blank_lines(self):
        path = self._write("# comment\n\npypdf==6.21.2\nImageIO==2.37.3\n", "c.txt")
        self.assertEqual([(d.name, d.version) for d in ad.read_python_pins(path)], [("pypdf", "6.21.2"), ("ImageIO", "2.37.3")])

    def test_cargo_pins_keep_registry_crates_only(self):
        lock = (
            '[[package]]\nname = "tokio"\nversion = "1.54.1"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n\n'
            '[[package]]\nname = "chaosengineai"\nversion = "0.9.9"\n\n'
            '[[package]]\nname = "forked"\nversion = "0.1.0"\nsource = "git+https://example.com/forked"\n'
        )
        deps = ad.read_cargo_pins(self._write(lock, "Cargo.lock"))
        self.assertEqual([(d.ecosystem, d.name, d.version) for d in deps], [("crates.io", "tokio", "1.54.1")])

    def test_npm_pins_use_the_last_node_modules_segment_and_flag_dev_only(self):
        lock = {"packages": {
            "": {"name": "app"},
            "node_modules/react": {"version": "18.3.1"},
            "node_modules/vite": {"version": "7.3.7", "dev": True},
            "node_modules/a/node_modules/@scope/pkg": {"version": "1.0.0", "dev": True},
            "node_modules/@scope/pkg": {"version": "1.0.0"},
            "node_modules/linked": {"link": True},
        }}
        by_name = {d.name: d for d in ad.read_npm_pins(self._write(json.dumps(lock), "package-lock.json"))}
        self.assertEqual(set(by_name), {"react", "vite", "@scope/pkg"})
        self.assertTrue(by_name["vite"].dev)
        self.assertFalse(by_name["react"].dev)
        # Installed for both shipped and dev code: counts as shipped.
        self.assertFalse(by_name["@scope/pkg"].dev)

    def test_direct_crates_come_from_every_dependency_table(self):
        toml = (
            '[package]\nname = "x"\n\n[dependencies]\ntauri = { version = "2" }\nserde = "1"\n\n'
            '[target.\'cfg(target_os = "linux")\'.dependencies]\nlibc = "0.2"\n\n[build-dependencies]\ntauri-build = "2"\n\n[profile.release]\nlto = true\n'
        )
        self.assertEqual(ad.read_direct_crates(self._write(toml, "Cargo.toml")), {"tauri", "serde", "libc", "tauri-build"})

    def test_the_real_pins_parse(self):
        self.assertTrue(any(d.name == "pypdf" for d in ad.read_python_pins()))
        self.assertTrue(any(d.name == "tokio" for d in ad.read_cargo_pins()))
        self.assertTrue(any(d.name == "react" and not d.dev for d in ad.read_npm_pins()))


class VersionTests(unittest.TestCase):
    def test_version_tuple_and_comparison(self):
        self.assertEqual(ad.version_tuple("2026.10.19"), (2026, 10, 19))
        self.assertEqual(ad.version_tuple("4.13.0.92"), (4, 13, 0, 92))
        self.assertTrue(ad.is_newer("6.21.2", "6.10.2"))  # not a string comparison
        self.assertFalse(ad.is_newer("6.9.0", "6.21.2"))

    def test_how_far_behind(self):
        self.assertEqual(ad.behind_kind("12.2.0", "13.0.0"), "major")
        self.assertEqual(ad.behind_kind("3.13.5", "3.14.4"), "minor")
        self.assertEqual(ad.behind_kind("3.14.1", "3.14.4"), "patch")
        self.assertIsNone(ad.behind_kind("3.14.4", "3.14.4"))
        self.assertIsNone(ad.behind_kind("3.14.4", "3.14.0"))


class SeverityTests(unittest.TestCase):
    def test_cvss3_scores_match_the_specification(self):
        cases = {
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H": 9.8,
            "CVSS:3.1/AV:L/AC:L/PR:L/UI:R/S:U/C:H/I:H/A:H": 7.3,
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H": 7.5,
            "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H": 9.9,
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N": 6.1,
            "CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N": 1.8,
        }
        for vector, score in cases.items():
            with self.subTest(vector=vector):
                self.assertEqual(ad.cvss3_score(vector), score)

    def test_cvss_that_cannot_be_scored_is_none(self):
        self.assertIsNone(ad.cvss3_score("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"))
        self.assertIsNone(ad.cvss3_score("not a vector"))
        self.assertIsNone(ad.cvss3_score("CVSS:3.1/AV:N/AC:L"))

    def test_score_to_label(self):
        self.assertEqual([ad.cvss_label(s) for s in (0.0, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0)],
                         ["none", "low", "moderate", "moderate", "high", "high", "critical"])

    def test_a_github_label_wins_and_medium_means_moderate(self):
        self.assertEqual(ad.record_severity(ghsa("GHSA-1", "HIGH", "x", None)), ("high", ""))
        self.assertEqual(ad.record_severity(ghsa("GHSA-2", "MEDIUM", "x", None)), ("moderate", ""))

    def test_a_rustsec_record_is_scored_from_its_vector(self):
        record = {"id": "RUSTSEC-1", "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"}]}
        self.assertEqual(ad.record_severity(record), ("high", ""))

    def test_a_maintenance_notice_is_not_a_flaw(self):
        record = {"id": "RUSTSEC-2", "affected": [{"database_specific": {"informational": "unmaintained"}}]}
        self.assertEqual(ad.record_severity(record), ("none", "unmaintained"))

    def test_an_unrated_flaw_is_unknown(self):
        self.assertEqual(ad.record_severity({"id": "X"})[0], "unknown")


class FindingTests(unittest.TestCase):
    dep = ad.Dep("PyPI", "pillow", "12.2.0")

    def test_aliases_of_one_flaw_count_once(self):
        records = [
            ghsa("GHSA-aaaa", "HIGH", "pillow", "13.0.0", aliases=["CVE-2026-1", "PYSEC-1"], summary="heap overflow"),
            ghsa("PYSEC-1", "HIGH", "pillow", "13.0.0", aliases=["GHSA-aaaa"]),
            ghsa("GHSA-bbbb", "LOW", "pillow", "12.3.0", summary="another"),
        ]
        finding = ad.build_finding(self.dep, records)
        self.assertEqual(len(finding.flaws), 2)
        self.assertEqual(finding.flaws[0].severity, "high")
        self.assertEqual(finding.flaws[0].ids[0], "GHSA-aaaa")
        self.assertEqual(set(finding.flaws[0].ids), {"GHSA-aaaa", "CVE-2026-1", "PYSEC-1"})
        self.assertEqual(finding.severity, "high")

    def test_withdrawn_advisories_are_ignored(self):
        record = ghsa("GHSA-x", "CRITICAL", "pillow", "13.0.0")
        record["withdrawn"] = "2026-01-01T00:00:00Z"
        self.assertIsNone(ad.build_finding(self.dep, [record]))

    def test_the_fix_clears_every_advisory(self):
        # One advisory is fixed in 12.3.0 (or 13.0.0 on the next branch), another in 13.0.0.
        first = ghsa("GHSA-1", "HIGH", "pillow", "12.3.0")
        first["affected"][0]["ranges"][0]["events"] += [{"fixed": "13.0.0"}]
        second = ghsa("GHSA-2", "HIGH", "pillow", "13.0.0")
        self.assertEqual(ad.fixed_version([first, second], self.dep), "13.0.0")
        self.assertEqual(ad.fixed_version([first], self.dep), "12.3.0")

    def test_a_fix_older_than_the_pin_is_not_a_fix(self):
        self.assertIsNone(ad.fixed_version([ghsa("GHSA-1", "HIGH", "pillow", "12.0.0")], self.dep))

    def test_other_packages_in_the_record_are_skipped(self):
        record = ghsa("GHSA-1", "HIGH", "someone-else", "99.0.0")
        self.assertIsNone(ad.fixed_version([record], self.dep))


class PolicyTests(unittest.TestCase):
    def _finding(self, severity: str = "high", dev: bool = False, ident: str = "GHSA-1") -> "ad.Finding":
        dep = ad.Dep("npm", "vite", "7.3.6", dev=dev)
        return ad.Finding(dep, [ad.Flaw([ident], severity, "flaw", None)])

    def test_threshold_and_dev_only(self):
        shipped, dev = self._finding("high"), self._finding("high", dev=True)
        self.assertTrue(ad.blocks(shipped.dep, shipped.flaws[0], "high", False))
        self.assertFalse(ad.blocks(shipped.dep, shipped.flaws[0], "critical", False))
        self.assertFalse(ad.blocks(dev.dep, dev.flaws[0], "high", False))
        self.assertTrue(ad.blocks(dev.dep, dev.flaws[0], "high", True))
        low = self._finding("moderate")
        self.assertFalse(ad.blocks(low.dep, low.flaws[0], "high", False))
        self.assertTrue(ad.blocks(low.dep, low.flaws[0], "moderate", False))

    def test_unknown_and_maintenance_notices_never_block(self):
        for severity in ("unknown", "none"):
            finding = self._finding(severity)
            self.assertFalse(ad.blocks(finding.dep, finding.flaws[0], "low", True))

    def test_an_active_ignore_accepts_the_flaw(self):
        finding = self._finding(ident="GHSA-1")
        entry = {"id": "GHSA-1", "reason": "signature-gated", "expires": "2026-12-31"}
        leftovers = ad.apply_ignores([finding], [entry], TODAY)
        flaw = finding.flaws[0]
        self.assertEqual((flaw.ignored_reason, flaw.ignored_until), ("signature-gated", "2026-12-31"))
        self.assertEqual(leftovers, [])
        self.assertFalse(ad.blocks(finding.dep, flaw, "high", False))

    def test_an_expired_ignore_blocks_again(self):
        finding = self._finding(ident="GHSA-1")
        ad.apply_ignores([finding], [{"id": "GHSA-1", "reason": "r", "expires": "2026-10-27"}], TODAY)
        self.assertTrue(finding.flaws[0].ignore_expired)
        self.assertTrue(ad.blocks(finding.dep, finding.flaws[0], "high", False))

    def test_an_ignore_works_through_any_alias(self):
        finding = ad.Finding(ad.Dep("PyPI", "x", "1"), [ad.Flaw(["GHSA-1", "CVE-2026-9"], "high", "f", None)])
        ad.apply_ignores([finding], [{"id": "CVE-2026-9", "reason": "r", "expires": "2027-01-01"}], TODAY)
        self.assertTrue(finding.flaws[0].ignored_reason)

    def test_a_new_advisory_on_the_same_package_is_not_hidden(self):
        dep = ad.Dep("crates.io", "zip", "4.6.1")
        old, new = ad.Flaw(["RUSTSEC-OLD"], "high", "old", None), ad.Flaw(["RUSTSEC-NEW"], "critical", "new", None)
        finding = ad.Finding(dep, [old, new])
        ad.apply_ignores([finding], [{"id": "RUSTSEC-OLD", "reason": "r", "expires": "2027-01-01"}], TODAY)
        self.assertFalse(ad.blocks(dep, old, "high", False))
        self.assertTrue(ad.blocks(dep, new, "high", False))

    def test_unused_entries_are_returned_for_cleanup(self):
        entry = {"id": "GHSA-gone", "reason": "r", "expires": "2027-01-01"}
        self.assertEqual(ad.apply_ignores([self._finding()], [entry], TODAY), [entry])

    def test_an_entry_without_a_reason_or_date_is_rejected(self):
        path = Path(tempfile.mkdtemp()) / "ignore.json"
        for bad in ({"id": "X", "expires": "2027-01-01"}, {"id": "X", "reason": "r"}, {"reason": "r", "expires": "2027-01-01"},
                    {"id": "X", "reason": "r", "expires": "soon"}):
            path.write_text(json.dumps({"ignore": [bad]}), encoding="utf-8")
            with self.subTest(entry=bad), self.assertRaises(ad.AuditError):
                ad.load_ignores(path)

    def test_the_shipped_ignore_file_is_valid_and_not_expired(self):
        for entry in ad.load_ignores():
            self.assertGreaterEqual(dt.date.fromisoformat(entry["expires"]), dt.date.today(), entry["id"])


class QueryTests(unittest.TestCase):
    deps = [ad.Dep("PyPI", f"p{i}", "1.0") for i in range(7)]

    def test_batches_are_chunked_and_matched_back_by_position(self):
        calls = []

        def fetch(url, payload=None, **_):
            calls.append(len(payload["queries"]))
            return {"results": [{"vulns": [{"id": f"V-{q['package']['name']}"}]} if q["package"]["name"] in ("p1", "p5") else {} for q in payload["queries"]]}

        found = ad.query_advisories(self.deps, fetch, chunk=3)
        self.assertEqual(calls, [3, 3, 1])
        self.assertEqual({d.name: ids for d, ids in found.items()}, {"p1": ["V-p1"], "p5": ["V-p5"]})

    def test_a_malformed_batch_answer_is_an_error_not_a_pass(self):
        for bad in ({}, {"results": []}, {"results": [{}]}, None):
            with self.subTest(bad=bad), self.assertRaises(ad.AuditError):
                ad.query_advisories(self.deps[:2], lambda *a, **k: bad)


class RunTests(unittest.TestCase):
    pins = [ad.Dep("PyPI", "pillow", "12.2.0"), ad.Dep("PyPI", "idna", "3.15")]

    def _run(self, records: dict, **kwargs):
        def fetch(url, payload=None, **_):
            if url == ad.OSV_BATCH:
                return {"results": [{"vulns": [{"id": "GHSA-p"}]} if q["package"]["name"] == "pillow" else {} for q in payload["queries"]]}
            if url.startswith(ad.OSV_VULN):
                return records.get(url.rsplit("/", 1)[1])
            raise AssertionError(url)

        with mock.patch.object(ad, "read_python_pins", return_value=self.pins), \
                mock.patch.object(ad, "read_cargo_pins", return_value=[]), \
                mock.patch.object(ad, "read_npm_pins", return_value=[]), \
                mock.patch.object(ad, "load_ignores", return_value=kwargs.pop("ignores", [])):
            return ad.run_audit(check_outdated=False, today=TODAY, fetch=fetch, **kwargs)

    def test_a_high_advisory_blocks_and_names_the_fix(self):
        report = self._run({"GHSA-p": ghsa("GHSA-p", "HIGH", "pillow", "13.0.0", summary="overflow")})
        self.assertEqual([(d.name, f.fixed_in) for d, f in report.blockers()], [("pillow", "13.0.0")])
        text = ad.render(report)
        self.assertIn("BLOCKING (1)", text)
        self.assertIn("upgrade to 13.0.0 or newer", text)
        self.assertIn("RESULT: FAIL", text)

    def test_a_clean_run_passes(self):
        report = self._run({}, threshold="critical")
        self.assertEqual(report.blockers(), [])
        self.assertIn("RESULT: PASS", ad.render(report))

    def test_an_accepted_advisory_passes_and_is_listed(self):
        ignores = [{"id": "GHSA-p", "reason": "r", "expires": "2027-01-01"}]
        report = self._run({"GHSA-p": ghsa("GHSA-p", "CRITICAL", "pillow", "13.0.0")}, ignores=ignores)
        self.assertEqual(report.blockers(), [])
        self.assertIn("accepted until 2027-01-01", ad.render(report))

    def test_json_report(self):
        report = self._run({"GHSA-p": ghsa("GHSA-p", "HIGH", "pillow", "13.0.0")})
        data = ad.to_json(report)
        self.assertEqual(data["blocking"], 1)
        self.assertEqual(data["findings"][0]["package"], "pillow")
        self.assertTrue(data["findings"][0]["blocking"])
        self.assertIsNone(data["outdated"])


class MainTests(unittest.TestCase):
    def _main(self, report=None, error=None, argv=()):
        out, err = io.StringIO(), io.StringIO()
        target = mock.patch.object(ad, "run_audit", side_effect=error) if error else mock.patch.object(ad, "run_audit", return_value=report)
        with target, redirect_stdout(out), redirect_stderr(err):
            code = ad.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_exit_codes(self):
        dep = ad.Dep("PyPI", "x", "1")
        blocked = ad.Report({"PyPI": 1}, [ad.Finding(dep, [ad.Flaw(["G"], "high", "f", None)])], threshold="high")
        clean = ad.Report({"PyPI": 1}, [], threshold="high")
        self.assertEqual(self._main(blocked)[0], 1)
        self.assertEqual(self._main(clean)[0], 0)

    def test_no_network_is_exit_2_with_a_message(self):
        code, _, err = self._main(error=ad.AuditError("could not reach api.osv.dev"))
        self.assertEqual(code, 2)
        self.assertIn("could not reach api.osv.dev", err)

    def test_options_reach_the_audit(self):
        with mock.patch.object(ad, "run_audit", return_value=ad.Report({"PyPI": 0})) as run, redirect_stdout(io.StringIO()):
            ad.main(["--fail-on", "moderate", "--include-dev", "--no-outdated"])
        run.assert_called_once_with("moderate", True, False)


class OutdatedTests(unittest.TestCase):
    def test_only_direct_crates_and_npm_packages_are_looked_up(self):
        python = [ad.Dep("PyPI", "pillow", "12.2.0"), ad.Dep("PyPI", "idna", "3.20")]
        crates = [ad.Dep("crates.io", "tokio", "1.0.0"), ad.Dep("crates.io", "indirect", "1.0.0")]
        npm = [ad.Dep("npm", "react", "18.3.1"), ad.Dep("npm", "scheduler", "0.23.0")]
        latest = {"pillow": "13.0.0", "idna": "3.20", "tokio": "1.54.1", "react": "19.3.1"}

        def fetch(url, payload=None, **_):
            name = url.rsplit("/", 2)[-2] if url.endswith("/json") else url.rsplit("/", 1)[-1]
            if "pypi.org" in url:
                return {"info": {"version": latest[url.split("/pypi/")[1].split("/")[0]]}}
            if "crates.io" in url:
                return {"crate": {"max_stable_version": latest[name]}}
            return {"version": latest[url.rsplit("/", 2)[-2]]}

        result = ad.find_outdated(python, crates, npm, {"tokio"}, {"react"}, fetch)
        self.assertEqual({(o.dep.name, o.kind) for o in result}, {("pillow", "major"), ("tokio", "minor"), ("react", "major")})

    def test_a_registry_hiccup_does_not_fail_the_audit(self):
        def fetch(*_a, **_k):
            raise ad.AuditError("registry down")

        self.assertEqual(ad.find_outdated([ad.Dep("PyPI", "x", "1.0")], [], [], set(), set(), fetch), [])


if __name__ == "__main__":
    unittest.main()
