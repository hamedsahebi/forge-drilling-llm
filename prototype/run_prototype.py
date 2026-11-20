"""Prototype: extract structured data and figures from 10 public Utah FORGE drilling documents.

Phase 1 (text LLM): daily reports -> DailyReport JSON; long reports -> WellSummary (map-reduce).
Phase 2 (vision LLM): figures -> FigureInfo JSON (image + caption + text-layer hints).
Outputs are cached under data/processed/<doc_id>/ so the run can be resumed.

    python run_prototype.py [--phase text|figures|all] [--only DOC_ID] [--out DIR] [--no-seed]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import fitz

import llm
from llm import TEXT_MODEL, VISION_MODEL, extract
from parse_pdf import bit_rows, casing_rows, find_figures, mud_rows, page_text, read_activity_codes, read_npt_column, split_reports
from pydantic import BaseModel, create_model

from schema import (BhaComponent, Bit, Casing, DailyReport, FigureInfo, MudCheck, Operation, PartHeader,
                    ActivityClass, RowLabels, Survey, WellFacts, WellNarrative, WellSummary)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "forge"
OUT = ROOT / "data" / "processed"

_16A = "16A78-32_DDR/16A(78)-32_Daily_Reports"
DOCS = [
    dict(id="D01", well="16A(78)-32", kind="ddr", path=f"{_16A}/drilling/Utah_Forge_FORGE_16A_(78)-32_Drilling-C_10232020_10232020_26_reporttmp.pdf"),
    dict(id="D02", well="16A(78)-32", kind="ddr", path=f"{_16A}/drilling/11_04_2020_Utah_Forge_FORGE_16A_[78]-32_Drilling-C_11420201142020_25_reporttmp.pdf"),
    dict(id="D03", well="16A(78)-32", kind="ddr", path=f"{_16A}/completion/Utah_Forge_FORGE_16A_(78)-32_Completion-C_01082021_01082021_17_reporttmp.pdf"),
    dict(id="D04", well="56-32", kind="ddr", path="56-32_DDR/DailyReportDetailRpt_210205202148.pdf"),
    dict(id="D05", well="56-32", kind="ddr", path="56-32_DDR/DailyReportDetailRpt_210215064844.pdf"),
    dict(id="D06", well="56-32", kind="ddr", path="56-32_DDR/DailyReportDetailRpt_210309095105.pdf"),
    dict(id="D07", well="78B-32", kind="ddr", path="78B-32_DDR.pdf"),            # 93-page compilation
    dict(id="D08", well="56-32", kind="long_report", path="56-32_EOWR.pdf"),     # End-of-Well Report
    dict(id="D09", well="78B-32", kind="plot_document", path="78B-32_Pason_graph.pdf", pages=[0, 10, 60, 150]),
    dict(id="D10", well="78B-32", kind="plot_document", path="78B-32_Mud_Log.pdf", pages=[2, 5, 12, 20]),
]

DDR_SYSTEM = """You are a drilling engineer converting a daily drilling/completion report into structured data.
Rules:
- The text is a monospace rendering of the page: table values sit vertically under their column headers.
- Copy values exactly as written; convert dates to YYYY-MM-DD. Depths in ft, sizes in inches.
- If a value is not present in the report, use null (or an empty list). Never invent values, never use 0
  as a placeholder, never copy a value from a neighbouring column or row.
- operations: one entry per time-log row. Choose `activity` from the allowed classes.
  is_npt = true only if the row is marked non-productive or describes unplanned trouble/repair/waiting.
- events: drilling problems or incidents (stuck pipe, losses, tight hole, failures, kicks, HSE). Empty list if none.
- bha: list every component row with OD, ID and length."""

SUMMARY_SYSTEM = """You are a drilling engineer reading part of an End-of-Well report.
Take values from this excerpt only; use null / empty lists for anything it does not state. Never invent values."""

FACTS_TASK = """Fill the well facts stated in the excerpt: well name, purpose, spud and TD dates (YYYY-MM-DD),
total depth MD/TVD in ft as a plain number (a total depth the text states in words or with commas counts), rig,
every casing string (size, weight, grade, shoe depth) and every bit run (number, size, make, type, serial,
depth in/out, ROP) found in text or tables."""

NARRATIVE_TASK = """Fill `events` with drilling problems the excerpt describes (failures, stuck pipe, losses,
cementing problems), with depth if given, and `lessons_learned` with recommendations or lessons the authors state.
Planned procedures are not lessons."""

FIGURE_SYSTEM = """You are a drilling engineer reading a figure from a well report.
Classify the figure, read its axes, series and any clearly legible numbers.
TEXT-LAYER WORDS are machine-readable text found inside the figure area - prefer them over reading pixels for numbers.
Only report values you can actually read. Describe what an engineer learns from the figure.
Then structure your reading: list every named curve/track/component in `series` and every legible
number in `key_values`, each written as "<quantity> = <value> <unit>". Fill axes for charts.
Take every number from your transcription of this figure - never from these instructions.
Glossary: MSE = mechanical specific energy, ROP = rate of penetration, WOB = weight on bit, BHA = bottom-hole
assembly, HWDP = heavy-weight drill pipe, DC = drill collar, SPP = standpipe pressure, BOPE = blowout preventer equipment."""


def _table_task(name: str, row_cls: type[BaseModel], hint: str = "") -> tuple[type[BaseModel], str]:
    cls = create_model(f"Part_{name}", **{name: (list[row_cls], ...)})
    fields = ", ".join(row_cls.model_fields)
    return cls, (f"Fill `{name}`: one entry for EVERY row of that table in the report, in order. "
                 f"Fields per row: {fields}. {hint} Empty list if the report has no such table.")


# Per-table extraction: a small model reads one table at a time far more accurately than a whole report.
SECTION_TASKS = [
    (PartHeader, "Fill the report header (well, report number, date, operator, contractor, rig, "
                 "report_type = drilling or completion only), "
                 "depth status (MD, TVD, footage/hole made in 24 h, hole size, current and planned operations) "
                 "and a one-sentence summary."),
    _table_task("operations", Operation, "hours = elapsed hours of the time-log row."),
    _table_task("casing", Casing, "shoe_md_ft = bottom/shoe measured depth."),
    _table_task("mud", MudCheck),
    _table_task("bits", Bit, "Bit rows are often in a combined table (e.g. 'Bit/BHA/Workstring Information' or "
                             "'DRILL BITS') - every numbered bit row counts. manufacturer = make; "
                             "bit_type = model or type (PDC, tricone ...); rop_ft_hr = average ROP."),
    _table_task("bha", BhaComponent, "Only rows that list a component; if the BHA is a single summary line, "
                                    "list each named component with null dimensions."),
    _table_task("surveys", Survey),
]

# Physically plausible ranges; values outside are LLM decoding errors -> set to null and logged.
PLAUSIBLE = {"md_ft": (0, 40000), "tvd_ft": (0, 40000), "footage_24h_ft": (0, 10000), "hole_size_in": (2, 40),
             "od_in": (1, 40), "id_in": (0.5, 30), "weight_ppf": (1, 200), "shoe_md_ft": (0, 40000),
             "density_ppg": (6, 22), "funnel_viscosity_s": (20, 300), "pv_cp": (0, 150), "yp_lbf_100ft2": (0, 150),
             "ph": (0, 14), "chlorides_mg_l": (0, 250000), "size_in": (2, 40), "depth_in_ft": (0, 40000),
             "depth_out_ft": (0, 40000), "wob_klbf": (0, 150), "rpm": (0, 400), "rop_ft_hr": (0, 1000),
             "length_ft": (0, 40000), "inclination_deg": (0, 180), "azimuth_deg": (0, 360), "hours": (0, 24),
             "end_md_ft": (0, 40000), "depth_ft": (0, 40000)}


def sanity_check(obj, path: str = "", flags: list | None = None) -> list:
    flags = [] if flags is None else flags
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PLAUSIBLE and isinstance(v, (int, float)) and not PLAUSIBLE[k][0] <= v <= PLAUSIBLE[k][1]:
                flags.append(f"{path}{k}={v}")
                obj[k] = None
            else:
                sanity_check(v, f"{path}{k}.", flags)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sanity_check(v, f"{path}{i}.", flags)
    return flags


NPT_SYSTEM = """You label rows of a drilling time log. For each row decide:
- is_npt: true ONLY for unplanned lost time: repairs, equipment failure, fishing, stuck pipe, losses,
  well control, waiting on parts/weather/orders. Planned work is never NPT.
- event_type: the MOST SPECIFIC problem type if the row describes an unplanned problem or incident, else null.
  Use "other" only when no listed type fits.
Examples:
"POOH to change bit after planned run length." -> is_npt false, event null
"Shut down overnight during rig move, resume at first light." -> is_npt false, event null
"Pre-job safety meeting with crews." -> is_npt false, event null
"ROP increase at 2,950', sandstone in samples." -> is_npt false, event null
"Repaired hydraulic leak on top drive." -> is_npt true, event equipment_failure
"Worked stuck pipe, jarred free at 3,200'." -> is_npt true, event stuck_pipe
"Lost 40 bbl to formation, pumped LCM pill." -> is_npt true, event lost_circulation
"Overpull of 40k while pulling through 5,100'-4,800', backreamed." -> is_npt true, event tight_hole
"Mud pump #1 down, replaced liner and swab." -> is_npt true, event equipment_failure
"No progress at 6,200', POOH, found bit cones missing." -> is_npt true, event bit_or_bha_failure
"Lay down BHA and release rental equipment." (end of well) -> is_npt false, event null
"Row marked Non-Prod: wait on cement crew." -> is_npt true, event other"""


def label_rows(ops: list[dict], meta: dict) -> list[dict]:
    """Second pass: classify NPT and problem events row by row (focused prompt + few-shot examples)."""
    if not ops:
        return []
    rows = "\n".join(f"{i}: {o.get('description') or ''}" for i, o in enumerate(ops))
    res, m = extract(RowLabels, NPT_SYSTEM, f"TIME LOG ROWS:\n{rows}")
    for k in ("attempts", "seconds", "prompt_tokens", "output_tokens"):
        meta[k] += m.get(k) or 0
    events = []
    for lab in (res.labels if res else []):
        if 0 <= lab.row < len(ops):
            o = ops[lab.row]
            o["is_npt"] = o["llm_is_npt"] = lab.is_npt
            if lab.event_type:
                events.append({"event_type": lab.event_type.value, "depth_ft": o.get("end_md_ft"),
                               "time": o.get("start"), "description": o.get("description") or ""})
    return events


ACTIVITY_SYSTEM = """You classify rows of a drilling time log by the work done in the row (not the reason for the time).
Classes:
- drilling: making new hole - rotary or slide drilling, drilling out cement, float equipment or shoe track,
  reaming or washing, opening or enlarging the hole, cutting core.
- tripping: running in or pulling out of the hole with drill pipe or a work string, wiper trips, picking up,
  making up, laying down or breaking out the bit, BHA, tools or drill pipe.
- casing: running, pulling or landing casing, liner or tubing.
- cementing: the cement job - pressure testing lines for it, pumping spacer, cement and plugs, displacing.
- circulating: circulating or conditioning mud, pumping sweeps or pills, filling pipe.
- bop_testing: pressure or function testing the BOP, wellhead or choke equipment.
- logging_survey: wireline logging, directional surveys, formation integrity or leak-off tests.
- rig_up_down: rigging up or down the rig or a service company's equipment, nippling up or down the BOP or
  wellhead, moving the rig, loading out equipment.
- rig_repair: repairing failed rig or service equipment, also while waiting on a technician or parts for it.
- waiting: idle time - waiting on daylight, weather, orders, crews or equipment; shut down for the night.
- wellbore_problem: work caused by a hole problem - stuck pipe, lost circulation, well control, fishing.
- completion: completion and well-test work - packers, perforating, stimulation, injection tests, monitoring
  well pressure.
- other: safety meetings, routine rig service, slipping and cutting drilling line, housekeeping, anything else.
A row that names several jobs takes the class of the main work in it. Return one label for every row."""


def classify_rows(ops: list[dict], meta: dict) -> None:
    """Third pass: activity class per row from its description (focused prompt with class definitions).
    Sets activity and llm_activity; apply_activity_code() later overrides it where the operator code maps."""
    if not ops:
        return
    # one required field per row: with a free list of {row, activity} the model skipped a row and renumbered the rest
    cls = create_model("RowActivities", **{f"r{i}": (ActivityClass, ...) for i in range(len(ops))})
    rows = "\n".join(f"r{i}: {o.get('description') or ''}" for i, o in enumerate(ops))
    res, m = extract(cls, ACTIVITY_SYSTEM, f"TIME LOG ROWS:\n{rows}")
    for k in ("attempts", "seconds", "prompt_tokens", "output_tokens"):
        meta[k] += m.get(k) or 0
    meta["activity_pass"] = res is not None
    for i, o in enumerate(ops if res else []):
        o["activity"] = o["llm_activity"] = getattr(res, f"r{i}").value


ONLY_REPORTS: set[str] | None = None  # --reports: extract only these daily reports
MODE = "sectioned"  # or "single" (one call for the whole DailyReport schema) - compared in the paper


def extract_ddr(text: str, mode: str) -> tuple[DailyReport | None, dict]:
    if mode == "single":
        return extract(DailyReport, DDR_SYSTEM, f"REPORT TEXT:\n{text}")
    merged, meta = {}, {"model": TEXT_MODEL, "mode": mode, "seed": llm.SEED, "attempts": 0, "seconds": 0.0,
                        "prompt_tokens": 0, "output_tokens": 0}
    for cls, task in SECTION_TASKS:
        res, m = extract(cls, DDR_SYSTEM + "\n\nTASK: " + task, f"REPORT TEXT:\n{text}")
        for k in ("attempts", "seconds", "prompt_tokens", "output_tokens"):
            meta[k] += m.get(k) or 0
        if res is None:
            meta.setdefault("failed_sections", []).append(cls.__name__)
            continue
        merged.update(res.model_dump(mode="json"))
    merged.setdefault("header", {})
    merged.setdefault("depth", {})
    merged["events"] = label_rows(merged.get("operations", []), meta)
    classify_rows(merged.get("operations", []), meta)
    meta["sanity_flags"] = sanity_check(merged)
    return DailyReport.model_validate(merged), meta


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clock(t: str | None) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", t or "")
    return (int(m[1]), int(m[2])) if m else None


def apply_npt_column(doc: dict) -> None:
    """Set is_npt from the operator's Non-Prod column on saved reports. A blank column is only trusted when the
    document has at least one mark: otherwise it cannot be told from a column nobody fills, so the LLM flag stays."""
    pdf = fitz.open(RAW / doc["path"])
    recs = [(f, json.loads(f.read_text(encoding="utf-8"))) for f in sorted((OUT / doc["id"] / "reports").glob("*.json"))]
    read = [read_npt_column(pdf, [p - 1 for p in rec["pages"]]) for _, rec in recs]
    in_use = any(row["marked"] for rows in read for row in rows or [])
    for (f, rec), rows in zip(recs, read):
        if not rec["data"]:
            continue
        by_time: dict[tuple, list[dict]] = {}
        for row in rows or []:
            by_time.setdefault((clock(row["start"]), clock(row["end"])), []).append(row)
        for o in rec["data"]["operations"]:
            o.setdefault("llm_is_npt", o["is_npt"])
            hit = by_time.get((clock(o.get("start")), clock(o.get("end"))))
            row = hit.pop(0) if hit else None
            o["is_npt"], o["npt_source"] = (row["marked"], "column") if in_use and row else (o["llm_is_npt"], "llm")
        rec["meta"]["npt_column"] = {"found": rows is not None, "in_use": in_use, "rows": len(rows or []),
                                     "marks": sum(r["marked"] for r in rows or []),
                                     "unmatched_rows": [f"{r['start']}-{r['end']}" for v in by_time.values() for r in v]}
        _save(f, rec)


def _num(s: str | None) -> float | None:
    try:
        return float(s.replace(",", "")) if s else None
    except ValueError:
        return None


def apply_casing_table(doc: dict) -> None:
    """Replace the LLM's casing rows with the Casing/Tubular table read by word position, where the page has one.
    The LLM often takes the OH Diam. column as the size and the top depth as the shoe. Its rows stay in meta."""
    pdf = fitz.open(RAW / doc["path"])
    for f in sorted((OUT / doc["id"] / "reports").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec["data"]:
            continue
        rec["meta"].setdefault("llm_casing", rec["data"]["casing"])
        rows = casing_rows(pdf, [p - 1 for p in rec["pages"]])
        if rows:
            rec["data"]["casing"] = [{"section": r.get("section"), "od_in": _num(r.get("size")),
                                      "weight_ppf": _num(r.get("weight")), "grade": r.get("grade"),
                                      "shoe_md_ft": _num(r.get("bottom_md"))} for r in rows]
        else:
            rec["data"]["casing"] = rec["meta"]["llm_casing"]
        rec["meta"]["casing_source"] = "table" if rows else "llm"
        _save(f, rec)


def apply_mud_table(doc: dict) -> None:
    """Replace the LLM's mud checks with the Mud Information table read by word position, where the page has one.
    The LLM missed checks and repeated others. Its rows stay in meta.llm_mud."""
    pdf = fitz.open(RAW / doc["path"])
    for f in sorted((OUT / doc["id"] / "reports").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec["data"]:
            continue
        rec["meta"].setdefault("llm_mud", rec["data"]["mud"])
        rows = mud_rows(pdf, [p - 1 for p in rec["pages"]])
        rec["data"]["mud"] = ([{k: _num(r.get(k)) for k in ("depth_ft", "density_ppg", "funnel_viscosity_s", "pv_cp",
                                                            "yp_lbf_100ft2", "ph", "chlorides_mg_l")} for r in rows]
                              if rows else rec["meta"]["llm_mud"])
        rec["meta"]["mud_source"] = "table" if rows else "llm"
        _save(f, rec)


def apply_bit_table(doc: dict) -> None:
    """Replace the LLM's bits with the Bit/BHA/Workstring table read by word position, where the page has one.
    The LLM missed bits and took drilling hours as ROP. The table prints no serial number. LLM rows stay in
    meta.llm_bits."""
    pdf = fitz.open(RAW / doc["path"])
    for f in sorted((OUT / doc["id"] / "reports").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec["data"]:
            continue
        rec["meta"].setdefault("llm_bits", rec["data"]["bits"])
        rows = bit_rows(pdf, [p - 1 for p in rec["pages"]])
        rec["data"]["bits"] = ([{"bit_number": r.get("bit_number"), "size_in": _num(r.get("size_in")),
                                 "manufacturer": r.get("manufacturer"), "bit_type": r.get("bit_type"), "serial_number": None,
                                 **{k: _num(r.get(k)) for k in ("depth_in_ft", "depth_out_ft", "wob_klbf", "rpm", "rop_ft_hr")}}
                                for r in rows] if rows else rec["meta"]["llm_bits"])
        rec["meta"]["bits_source"] = "table" if rows else "llm"
        _save(f, rec)


# Operator activity codes. Numbered codes are phase-activity[-sub] (e.g. 3-6-4). A three-digit middle part with a
# leading zero is the code plus a sub digit (03-051 -> 05).
# IADC_ACTIVITY: legacy IADC daily drilling report (tour sheet) main codes, from memory; check against the IADC manual.
# OPERATOR_ACTIVITY: codes outside that list, or IADC codes with no single class here. Meanings were read from the
# rows of the reports that are not hand-labelled, and confirmed by a drilling engineer (Hamed, 2026-11-06).
# Codes not listed keep the LLM's class: reaming 3, coring 4 (rows are labelled by step: trips with the core barrel
# are tripping, cutting core is drilling), directional 20, 21, 27, and the catch-all codes 99 and OTHER.
IADC_ACTIVITY = {1: "rig_up_down", 2: "drilling", 5: "circulating", 6: "tripping", 7: "other", 8: "rig_repair",
                 9: "other", 10: "logging_survey", 11: "logging_survey", 13: "waiting", 15: "bop_testing",
                 16: "logging_survey", 17: "cementing", 18: "cementing", 19: "wellbore_problem"}
OPERATOR_ACTIVITY = {12: "casing",          # run casing (sub-code 2 is cementing, see SUB_ACTIVITY)
                     14: "rig_up_down",     # nipple up / install BOP
                     28: "drilling",        # drill out shoe track
                     34: "tripping",        # make up / pick up / lay down BHA
                     46: "logging_survey",  # formation integrity test
                     97: "other"}           # clean rig floor
SUB_ACTIVITY = {(12, 2): "cementing"}       # 5-12-2: mix and pump cement
TEXT_ACTIVITY = {"RIGU": "rig_up_down", "TRPI": "tripping", "LOG": "logging_survey", "REPR": "rig_repair",
                 "WOD": "waiting", "SERV": "other", "CUTDL": "other",  # WOD wait on daylight, SERV rig service
                 "CMTP": "cementing"}


def code_activity(code: str) -> str | None:
    code = code.strip().upper()
    if code in TEXT_ACTIVITY:
        return TEXT_ACTIVITY[code]
    m = re.fullmatch(r"\d{1,2}-(\d{1,3})(?:-(\d+))?", code)
    if not m:
        return None
    mid = int(m[1][:2] if len(m[1]) == 3 and m[1][0] == "0" else m[1])
    sub = int(m[2]) if m[2] else None
    return SUB_ACTIVITY.get((mid, sub)) or OPERATOR_ACTIVITY.get(mid) or IADC_ACTIVITY.get(mid)


def apply_activity_code(doc: dict) -> None:
    """Set activity from the operator's Code column where the code maps to one class; otherwise keep the LLM's.
    The LLM's class stays in llm_activity."""
    pdf = fitz.open(RAW / doc["path"])
    for f in sorted((OUT / doc["id"] / "reports").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec["data"]:
            continue
        rows = read_activity_codes(pdf, [p - 1 for p in rec["pages"]])
        by_time: dict[tuple, list[dict]] = {}
        for row in rows or []:
            by_time.setdefault((clock(row["start"]), clock(row["end"])), []).append(row)
        mapped = 0
        for o in rec["data"]["operations"]:
            o.setdefault("llm_activity", o["activity"])
            hit = by_time.get((clock(o.get("start")), clock(o.get("end"))))
            row = hit.pop(0) if hit else None
            o["activity_code"] = row["code"] if row else None
            act = code_activity(row["code"]) if row else None
            o["activity"], o["activity_source"] = (act, "code") if act else (o["llm_activity"], "llm")
            mapped += bool(act)
        rec["meta"]["activity_code"] = {"found": rows is not None, "rows": len(rows or []), "mapped": mapped}
        _save(f, rec)


def run_text(doc: dict) -> None:
    pdf = fitz.open(RAW / doc["path"])
    out = OUT / doc["id"]
    pages = doc.get("pages", list(range(len(pdf))))
    if doc["kind"] == "ddr":
        for n, grp in enumerate(split_reports(pdf, pages), 1):
            dst = out / "reports" / f"{doc['id']}_r{n:03d}.json"
            if dst.exists() or (ONLY_REPORTS and dst.stem not in ONLY_REPORTS):
                continue
            text = "\n\n".join(page_text(pdf[i]) for i in grp)
            res, meta = extract_ddr(text[:30000], MODE)
            _save(dst, {"doc_id": doc["id"], "well": doc["well"], "source": doc["path"],
                        "pages": [i + 1 for i in grp], "meta": meta,
                        "data": res.model_dump(mode="json") if res else None})
            print(f"  {dst.name}: {meta['seconds']:.0f}s ok={res is not None}", flush=True)
        apply_npt_column(doc)
        apply_casing_table(doc)
        apply_mud_table(doc)
        apply_bit_table(doc)
        apply_activity_code(doc)
    elif doc["kind"] == "long_report":
        chunk, start = "", pages[0]
        chunks = []
        for i in pages:
            t = re.sub(r" {3,}", "  ", page_text(pdf[i]))  # prose pages: drop alignment padding, save tokens
            if len(chunk) + len(t) > 12000 and chunk:
                chunks.append((start, i - 1, chunk))
                chunk, start = "", i
            chunk += f"\n[page {i + 1}]\n{t}"
        chunks.append((start, pages[-1], chunk))
        for a, b, text in chunks:
            dst = out / "summary_parts" / f"{doc['id']}_p{a + 1:03d}-{b + 1:03d}.json"
            if dst.exists():
                continue
            # two focused calls per chunk, same lesson as the daily reports: one mixed schema -> nulls
            merged, meta = {}, {"model": TEXT_MODEL, "attempts": 0, "seconds": 0.0}
            for cls, task in [(WellFacts, FACTS_TASK), (WellNarrative, NARRATIVE_TASK)]:
                res, m = extract(cls, SUMMARY_SYSTEM + "\n\nTASK: " + task,
                                 f"EXCERPT (pages {a + 1}-{b + 1}):\n{text}")
                meta["attempts"] += m["attempts"]
                meta["seconds"] += m["seconds"]
                if res:
                    merged.update(res.model_dump(mode="json"))
            meta["sanity_flags"] = sanity_check(merged)
            _save(dst, {"doc_id": doc["id"], "pages": [a + 1, b + 1], "meta": meta,
                        "data": WellSummary.model_validate(merged).model_dump(mode="json")})
            print(f"  {dst.name}: {meta['seconds']:.0f}s ok=True", flush=True)


def run_figures(doc: dict) -> None:
    if doc["kind"] == "ddr":
        return  # daily reports here carry no figures besides the logo
    pdf = fitz.open(RAW / doc["path"])
    out = OUT / doc["id"]
    pages = doc.get("pages", list(range(len(pdf))))
    figs = find_figures(pdf, doc["id"], pages, out / "figures", full_page=doc["kind"] == "plot_document")
    for f in figs:
        dst = Path(f.image_path).with_suffix(".json")
        if dst.exists():
            continue
        prompt = (f"CAPTION: {f.caption or 'none'}\nDOCUMENT: {doc['path']} (well {doc['well']}), page {f.page}\n"
                  f"TEXT-LAYER WORDS:\n{f.text_hint[:2500] or 'none (raster image)'}")
        res, meta = extract(FigureInfo, FIGURE_SYSTEM, prompt, images=[f.image_path],
                            model=VISION_MODEL, num_ctx=16384, num_predict=1500)
        _save(dst, {**asdict(f), "well": doc["well"], "meta": meta,
                    "data": res.model_dump(mode="json") if res else None})
        print(f"  {dst.name}: {meta['seconds']:.0f}s ok={res is not None}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["text", "figures", "all"])
    ap.add_argument("--only")
    ap.add_argument("--relabel", action="store_true", help="rerun only the NPT/event pass on saved reports")
    ap.add_argument("--reclassify", action="store_true", help="rerun only the activity pass on saved reports")
    ap.add_argument("--npt-column", action="store_true", help="reread only the operator's Non-Prod column on saved reports")
    ap.add_argument("--casing-table", action="store_true", help="reread only the Casing/Tubular table on saved reports")
    ap.add_argument("--mud-table", action="store_true", help="reread only the Mud Information table on saved reports")
    ap.add_argument("--bit-table", action="store_true", help="reread only the Bit/BHA/Workstring table on saved reports")
    ap.add_argument("--activity-code", action="store_true", help="reread only the time-log Code column on saved reports")
    ap.add_argument("--out", help="write extractions here instead of data/processed (repeatability runs)")
    ap.add_argument("--no-seed", action="store_true", help="leave the sampling seed to Ollama")
    ap.add_argument("--reports", help="comma-separated report ids to extract (e.g. D07_r002,D07_r005)")
    ap.add_argument("--cold-calls", action="store_true", help="unload the model after every call (keep_alive 0)")
    a = ap.parse_args()
    if a.reports:
        ONLY_REPORTS = set(a.reports.split(","))
    if a.cold_calls:
        llm.KEEP_ALIVE = 0
    if a.out:
        OUT = Path(a.out).resolve()
    if a.no_seed:
        llm.SEED = None
    if a.relabel:
        for f in sorted(OUT.glob("*/reports/*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            if not rec["data"] or (a.only and rec["doc_id"] != a.only):
                continue
            meta = {k: 0 for k in ("attempts", "seconds", "prompt_tokens", "output_tokens")}
            rec["data"]["events"] = label_rows(rec["data"]["operations"], meta)
            rec["meta"]["seconds"] += meta["seconds"]
            _save(f, rec)
            print(f"  {f.name}: {len(rec['data']['events'])} events", flush=True)
        a.npt_column = True
    if a.reclassify:
        for f in sorted(OUT.glob("*/reports/*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            if not rec["data"] or (a.only and rec["doc_id"] != a.only) or (ONLY_REPORTS and f.stem not in ONLY_REPORTS):
                continue
            meta = {k: 0 for k in ("attempts", "seconds", "prompt_tokens", "output_tokens")}
            classify_rows(rec["data"]["operations"], meta)
            rec["meta"]["activity_pass"] = meta.get("activity_pass")
            rec["meta"]["seconds"] += meta["seconds"]
            _save(f, rec)
            print(f"  {f.name}: {meta['seconds']:.0f}s ok={meta.get('activity_pass')}", flush=True)
        a.activity_code = True
    if a.npt_column or a.casing_table or a.mud_table or a.bit_table or a.activity_code:
        for d in DOCS:
            if d["kind"] == "ddr" and (not a.only or d["id"] == a.only):
                if a.npt_column:
                    apply_npt_column(d)
                if a.casing_table:
                    apply_casing_table(d)
                if a.mud_table:
                    apply_mud_table(d)
                if a.bit_table:
                    apply_bit_table(d)
                if a.activity_code:
                    apply_activity_code(d)
        raise SystemExit
    docs = [d for d in DOCS if (not a.only or d["id"] == a.only) and (not ONLY_REPORTS or any(r.startswith(d["id"]) for r in ONLY_REPORTS))]
    t0 = time.time()
    if a.phase in ("text", "all"):
        print(f"== Phase 1: text extraction with {TEXT_MODEL}")
        for d in docs:
            print(d["id"], d["path"], flush=True)
            run_text(d)
    if a.phase in ("figures", "all"):
        print(f"== Phase 2: figure extraction with {VISION_MODEL}")
        for d in docs:
            print(d["id"], d["path"], flush=True)
            run_figures(d)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
