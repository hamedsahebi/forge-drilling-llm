"""Trust layer: check every extracted daily-report value against the PDF, the other reports and the sensors.

No LLM, and no extracted value is changed. Each value gets one verdict:
    fail       a check contradicts it, or the report prints a value the extraction missed
    conflict   copied correctly, but the report disagrees with itself or the sensors: a person should look
    pass       a strong check confirms it (printed label, time-log row, casing row, continuity, sensor)
    pdf_read   read from the PDF by code (table by word position, Code or Non-Prod column), not written by the LLM;
               no other check confirms it. A fail or conflict still wins, and a pass stays pass.
    weak       only found somewhere in the report text, which also happens by chance
    unchecked  no check applies
Each verdict also has `source`: table, code, column or llm.

Writes outputs/trust/checks.json, verdicts.json and summary.json; build_db.py loads them into SQLite.
Summary counts leave out D01, D02 and D05, the prompt-tuning reports.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import fitz

from parse_pdf import casing_rows, page_text, read_npt_column
from run_prototype import DOCS, OUT, RAW, clock
from sensor_truth import DST as SENSOR_FILE, load as load_sensor

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "outputs" / "trust"
DEV = {"D01", "D02", "D05"}
SENSOR_WELL = "78B-32"
DEPTH_TOL_FT, SENSOR_TOL_FT, HOURS_TOL = 1.0, 50.0, 0.01
CROSS_CHECKS = {"continuity", "md_vs_time_log", "sensor_depth"}  # these compare a value with something else the report or sensors say

NUM = r"(\d[\d,]*(?:\.\d+)?)"
# label -> regex per template; group 1 is the printed value, and no group 1 means the label is there but blank
LABELS = {
    ("header", "report_number"): [rf"Report\s+No:\s*{NUM}", rf"RPT\s+NUM\.?:\s*{NUM}"],
    ("header", "report_date"): [r"Report\s+For\s+(\d{2}-[A-Za-z]{3}-\d{2})", r"RPT\s+DATE:\s*(\d{2}/\d{2}/\d{4})"],
    ("header", "rig"): [r"Rig:\s+(.*?)\s+Spud\s+Date"],
    ("depth", "md_ft"): [rf"Measured\s+Depth\s+\(ft\):\s*(?:{NUM})?", rf"MD/TVD:\s*(?!24\s+HR)(?:{NUM})?"],
    ("depth", "tvd_ft"): [rf"Vertical\s+Depth\s+\(ft\):\s*(?:{NUM})?"],
    ("depth", "footage_24h_ft"): [rf"Hole\s+Made\s+\(ft\)\s*/\s*Hrs:\s*(?:{NUM})?", rf"24\s+HR\s+FTG:\s*(?:{NUM})?(?=\s|$)"],
}
NUMERIC = {"header": ["report_number"], "depth": ["md_ft", "tvd_ft", "footage_24h_ft", "hole_size_in"],
           "casing": ["od_in", "weight_ppf", "shoe_md_ft"],
           "mud": ["depth_ft", "density_ppg", "funnel_viscosity_s", "pv_cp", "yp_lbf_100ft2", "ph", "chlorides_mg_l"],
           "bits": ["size_in", "depth_in_ft", "depth_out_ft", "wob_klbf", "rpm", "rop_ft_hr"],
           "bha": ["od_in", "id_in", "length_ft"], "surveys": ["md_ft", "inclination_deg", "azimuth_deg", "tvd_ft"],
           "operations": ["hours", "end_md_ft"]}
TEXT = {"header": ["well_name", "operator", "contractor", "rig"], "depth": ["current_operation", "planned_operation"],
        "casing": ["section", "grade"], "bits": ["bit_number", "manufacturer", "bit_type", "serial_number"],
        "bha": ["component"], "operations": ["start", "end", "description"]}
UNCHECKED = {"header": ["report_type"], "operations": ["activity", "is_npt"], "events": ["event_type", "depth_ft", "time", "description"]}


def number(s: str | None) -> float | None:
    return float(s.replace(",", "")) if s else None


def parse_date(s: str) -> str:
    return datetime.strptime(s, "%d-%b-%y" if "-" in s else "%m/%d/%Y").date().isoformat()


def source_sets(text: str) -> tuple[set[float], set[str]]:
    """All numbers and words in the report. Fractions count as decimals, because reports print 8-3/4 for 8.75."""
    nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    for whole, a, b in re.findall(r"(?<![\d/])(\d+)[- ](\d+)/(\d+)", text):
        if int(b):
            nums.add(int(whole) + int(a) / int(b))
    return nums, set(re.findall(r"[a-z0-9]+", text.lower()))


class Report:
    def __init__(self, path: Path, pdf: fitz.Document):
        self.id = path.stem
        self.rec = json.loads(path.read_text(encoding="utf-8"))
        self.data = self.rec["data"]
        self.pages = [p - 1 for p in self.rec["pages"]]
        self.text = "\n".join(page_text(pdf[i]) for i in self.pages)
        self.head = re.sub(r"\s+", " ", page_text(pdf[self.pages[0]]))
        self.log = read_npt_column(pdf, self.pages) or []
        self.nums, self.words = source_sets(self.text)
        self.checks: list[dict] = []

    def add(self, table, row, field, value, check, status, evidence):
        self.checks.append({"report_id": self.id, "tbl": table, "row": row, "field": field, "value": value,
                            "check_name": check, "status": status, "evidence": evidence})

    def printed(self, table: str, field: str):
        """(label found, printed value) from the first page, read with the label regex alone."""
        for pat in LABELS[(table, field)]:
            m = re.search(pat, self.head)
            if m:
                return True, m.group(1) if m.groups() else None
        return False, None

    def printed_date(self) -> str | None:
        found, v = self.printed("header", "report_date")
        return parse_date(v) if found and v else None

    def printed_number(self, table: str, field: str) -> float | None:
        found, v = self.printed(table, field)
        return number(v) if found else None


def check_labels(r: Report) -> None:
    """Header fields against the value printed next to their label."""
    for (table, field), _ in LABELS.items():
        found, raw = r.printed(table, field)
        if not found:
            continue
        saved = r.data[table].get(field)
        if field == "report_date":
            want = parse_date(raw) if raw else None
        elif field == "rig":
            want = raw
        else:
            want = number(raw)
        if field in ("rig", "report_date"):
            same = (saved or None) == (want or None)
        else:
            same = saved == want or (saved is not None and want is not None and abs(saved - want) < 1e-6)
        if want is None and saved is None:
            continue
        r.add(table, None, field, saved, "printed_label", "pass" if same else "fail", f"report prints {raw!r}")


def check_source(r: Report) -> None:
    """Every extracted value should appear somewhere in the report. Weak: finding it proves little, missing it is telling."""
    groups = {"header": [r.data["header"]], "depth": [r.data["depth"]]}
    groups.update({t: r.data[t] for t in ("casing", "mud", "bits", "bha", "surveys", "operations")})
    for table, rows in groups.items():
        for i, row in enumerate(rows):
            n = None if table in ("header", "depth") else i
            for f in NUMERIC.get(table, []):
                v = row.get(f)
                if v is not None:
                    ok = any(abs(v - x) < 1e-6 for x in r.nums)
                    r.add(table, n, f, v, "source_match", "weak" if ok else "fail", "number found in report text" if ok else "number not in report text")
            for f in TEXT.get(table, []):
                v = row.get(f)
                if v:
                    words = re.findall(r"[a-z0-9]+", str(v).lower())
                    miss = [w for w in words if w not in r.words]
                    ok = not miss and bool(words)
                    r.add(table, n, f, v, "source_match", "weak" if ok else "fail", "all words found in report text" if ok else f"not in report text: {miss[:5]}")


def check_time_log(r: Report) -> None:
    """Time-log rows against the From, To, Elapsed triples read from word positions, and their arithmetic."""
    ops = r.data["operations"]
    pool = defaultdict(list)
    for row in r.log:
        pool[(clock(row["start"]), clock(row["end"]))].append(row)
    if r.log:
        for i, o in enumerate(ops):
            hit = pool.get((clock(o.get("start")), clock(o.get("end"))))
            row = hit.pop(0) if hit else None
            ok = row is not None
            for f in ("start", "end"):
                r.add("operations", i, f, o.get(f), "time_log_row", "pass" if ok else "fail", "same From and To in PDF row" if ok else "no such row in PDF")
            if ok and o.get("hours") is not None:
                same = abs(o["hours"] - row["hours"]) <= HOURS_TOL
                r.add("operations", i, "hours", o["hours"], "time_log_row", "pass" if same else "fail", f"PDF elapsed {row['hours']}")
        for v in pool.values():
            for row in v:
                r.add("operations", None, "row", f"{row['start']}-{row['end']}", "time_log_row", "fail", f"PDF row on page {row['page']} was not extracted")
    total = sum(o.get("hours") or 0 for o in ops)
    if ops:
        r.add("operations", None, "hours_total", total, "hours_24", "pass" if abs(total - 24) <= 0.5 else "fail", f"rows sum to {total} h")
    last = next((o["end_md_ft"] for o in reversed(ops) if o.get("end_md_ft") is not None), None)
    md = r.data["depth"].get("md_ft")
    if last is not None and md is not None:
        ok = abs(last - md) <= DEPTH_TOL_FT
        r.add("depth", None, "md_ft", md, "md_vs_time_log", "pass" if ok else "fail", f"last time-log End MD is {last}")


def check_casing(r: Report, pdf: fitz.Document) -> None:
    """Saved casing rows against the table read by word position. Rows are matched on section name.
    Skipped when the saved rows already come from that table (run_prototype.apply_casing_table)."""
    if r.rec["meta"].get("casing_source") == "table":
        return
    rows = casing_rows(pdf, r.pages)
    if not rows:
        return
    left = list(rows)
    for i, c in enumerate(r.data["casing"]):
        row = next((x for x in left if x.get("section") == c.get("section")), None)
        if row is None:
            r.add("casing", i, "section", c.get("section"), "casing_row", "fail", "no such section left in the PDF table")
            continue
        left.remove(row)
        r.add("casing", i, "section", c.get("section"), "casing_row", "pass", "section is in the PDF table")
        for f, k in (("od_in", "size"), ("shoe_md_ft", "bottom_md"), ("weight_ppf", "weight")):
            want, got = number(row.get(k)), c.get(f)
            ok = want == got or (want is not None and got is not None and abs(want - got) < 1e-6)
            if want is not None or got is not None:
                r.add("casing", i, f, got, "casing_row", "pass" if ok else "fail", f"PDF column {k} prints {row.get(k)}")
        if row.get("grade") or c.get("grade"):
            ok = row.get("grade") == c.get("grade")
            r.add("casing", i, "grade", c.get("grade"), "casing_row", "pass" if ok else "fail", f"PDF column grade prints {row.get('grade')}")
    for row in left:
        r.add("casing", None, "row", row.get("section"), "casing_row", "fail", "PDF row was not extracted")


def check_continuity(reports: list[Report]) -> None:
    """Consecutive reports: depth grows by the footage, and hole size does not grow as the well deepens."""
    by_num = sorted((r for r in reports if r.printed_number("header", "report_number") is not None),
                    key=lambda r: r.printed_number("header", "report_number"))
    for a, b in zip(by_num, by_num[1:]):
        if b.printed_number("header", "report_number") - a.printed_number("header", "report_number") != 1:
            continue
        md0, md1, ftg = a.data["depth"].get("md_ft"), b.data["depth"].get("md_ft"), b.data["depth"].get("footage_24h_ft")
        if md0 is None or md1 is None:
            continue
        if ftg is not None:
            ok = abs(md1 - md0 - ftg) <= DEPTH_TOL_FT
            evidence = f"previous MD {md0} + footage {ftg} = {md0 + ftg}"
            b.add("depth", None, "md_ft", md1, "continuity", "pass" if ok else "fail", evidence)
            b.add("depth", None, "footage_24h_ft", ftg, "continuity", "pass" if ok else "fail", evidence)
        elif md1 < md0 - DEPTH_TOL_FT:
            b.add("depth", None, "md_ft", md1, "continuity", "fail", f"MD fell from previous {md0}")
    smallest = None
    for r in by_num:
        size = r.data["depth"].get("hole_size_in")
        if size is None:
            continue
        if smallest is not None:
            ok = size <= smallest + 0.01
            # weak when it holds: a wrong size smaller than the earlier ones also holds
            r.add("depth", None, "hole_size_in", size, "hole_size_sequence", "weak" if ok else "fail", f"smallest earlier hole size {smallest}")
        smallest = size if smallest is None else min(smallest, size)


def check_sensor(reports: list[Report]) -> None:
    """Report depth against the Pason hole depth at the end of the printed report date, which the LLM did not supply."""
    if not SENSOR_FILE.exists():
        return
    times, cols = load_sensor()
    depth = cols["hole_depth_ft"]
    for r in reports:
        day, md = r.printed_date(), r.data["depth"].get("md_ft")
        if not day or md is None:
            continue
        start = datetime.fromisoformat(day)
        end = start + timedelta(days=1)
        if times[0] > start:  # the sensors must cover the whole report day
            continue
        i = bisect_right(times, end) - 1
        while i >= 0 and depth[i] is None:
            i -= 1
        if i < 0 or end - times[i] > timedelta(hours=1):
            continue
        ok = abs(md - depth[i]) <= SENSOR_TOL_FT
        r.add("depth", None, "md_ft", md, "sensor_depth", "pass" if ok else "fail", f"Pason hole depth {depth[i]:.0f} ft at {times[i]:%m-%d %H:%M}")


def extracted_values(r: Report):
    """Every non-null extracted value, as (table, row, field, value)."""
    d = r.data
    for table in ("header", "depth"):
        for f in NUMERIC.get(table, []) + TEXT.get(table, []) + UNCHECKED.get(table, []):
            if d[table].get(f) not in (None, ""):
                yield table, None, f, d[table][f]
    if d["header"].get("report_date"):
        yield "header", None, "report_date", d["header"]["report_date"]
    for table in ("casing", "mud", "bits", "bha", "surveys", "operations", "events"):
        for i, row in enumerate(d[table]):
            for f in NUMERIC.get(table, []) + TEXT.get(table, []) + UNCHECKED.get(table, []):
                if row.get(f) not in (None, ""):
                    yield table, i, f, row[f]


def value_source(r: Report, t: str, i: int | None, f: str) -> str:
    """Where a saved value came from: a PDF table, the Code or Non-Prod column read by code, or the LLM."""
    meta = r.rec["meta"]
    if t in ("casing", "mud", "bits") and meta.get(f"{t}_source") == "table":
        return "table"
    if t == "operations" and i is not None and i < len(r.data["operations"]):
        o = r.data["operations"][i]
        if f == "activity" and o.get("activity_source") == "code":
            return "code"
        if f == "is_npt" and o.get("npt_source") == "column":
            return "column"
    return "llm"


def verdicts(r: Report) -> list[dict]:
    seen = defaultdict(list)
    for c in r.checks:
        seen[(c["tbl"], c["row"], c["field"])].append(c)
    out = []
    keys = {(t, i, f): v for t, i, f, v in extracted_values(r)}
    for k in seen:
        keys.setdefault(k, seen[k][0]["value"])  # values the extraction missed, and report-level checks
    for (t, i, f), v in keys.items():
        cs = seen.get((t, i, f), [])
        st = {c["status"] for c in cs}
        verdict = "fail" if "fail" in st else "pass" if "pass" in st else "weak" if "weak" in st else "unchecked"
        failed = {c["check_name"] for c in cs if c["status"] == "fail"}
        if failed and failed <= CROSS_CHECKS and any(c["check_name"] == "printed_label" and c["status"] == "pass" for c in cs):
            verdict = "conflict"
        source = value_source(r, t, i, f)
        if source != "llm" and verdict in ("weak", "unchecked"):
            verdict = "pdf_read"
        out.append({"report_id": r.id, "tbl": t, "row": i, "field": f, "value": v, "verdict": verdict, "source": source,
                    "failed_checks": ",".join(sorted({c["check_name"] for c in cs if c["status"] == "fail"}))})
    return out


if __name__ == "__main__":
    all_checks, all_verdicts, doc_of = [], [], {}
    for doc in DOCS:
        if doc["kind"] != "ddr":
            continue
        pdf = fitz.open(RAW / doc["path"])
        reports = [Report(f, pdf) for f in sorted((OUT / doc["id"] / "reports").glob("*.json"))]
        for r in reports:
            doc_of[r.id] = doc["id"]
            check_labels(r)
            check_source(r)
            check_time_log(r)
            check_casing(r, pdf)
        check_continuity(reports)
        if doc["well"] == SENSOR_WELL:
            check_sensor(reports)
        for r in reports:
            all_checks += r.checks
            all_verdicts += verdicts(r)
    scored_c = [c for c in all_checks if doc_of[c["report_id"]] not in DEV]
    scored_v = [v for v in all_verdicts if doc_of[v["report_id"]] not in DEV]
    table = Counter((v["tbl"], v["verdict"]) for v in scored_v)
    summary = {"reports_scored": len({v["report_id"] for v in scored_v}),
               "verdicts": dict(Counter(v["verdict"] for v in scored_v)),
               "by_source": dict(Counter(v["source"] for v in scored_v)),
               "by_table": {t: {s: table[(t, s)] for s in ("pass", "pdf_read", "weak", "conflict", "fail", "unchecked") if table[(t, s)]}
                            for t in sorted({v["tbl"] for v in scored_v})},
               "by_check": {k: dict(Counter(c["status"] for c in scored_c if c["check_name"] == k))
                            for k in sorted({c["check_name"] for c in scored_c})}}
    DST.mkdir(parents=True, exist_ok=True)
    for name, obj in (("checks", all_checks), ("verdicts", all_verdicts), ("summary", summary)):
        (DST / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
