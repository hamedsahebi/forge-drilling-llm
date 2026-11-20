"""Load all extracted JSON into one SQLite database (outputs/drilling_reports.db) + CSV exports."""
from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path

from chart_reader import CURVES, OUT as CHART_READINGS, SCALE
from run_prototype import DOCS

WELLS = {d["id"]: d["well"] for d in DOCS}  # the same well names as the other tables, so joins on well work
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"
DB = OUTD / "drilling_reports.db"
# Figure kinds whose numbers the vision model reads off a curve or log; scored far below the deterministic reader
CHART_KINDS = {"time_vs_depth", "drilling_parameter_chart", "pressure_chart", "directional_plot", "other_chart",
               "well_log", "mud_log"}

TABLES = {
    "reports": "report_id TEXT PRIMARY KEY, doc_id, well, source, pages, report_number INT, report_date, report_type, rig, md_ft REAL, tvd_ft REAL, footage_24h_ft REAL, hole_size_in REAL, current_operation, planned_operation, summary, seconds REAL",
    "operations": "report_id, well, report_date, start, end, hours REAL, end_md_ft REAL, activity, activity_source, activity_code, llm_activity, is_npt INT, npt_source, llm_is_npt INT, description",
    "events": "report_id, well, report_date, event_type, depth_ft REAL, time, description",
    "bits": "report_id, well, report_date, bit_number, size_in REAL, manufacturer, bit_type, serial_number, depth_in_ft REAL, depth_out_ft REAL, wob_klbf REAL, rpm REAL, rop_ft_hr REAL",
    "bha": "report_id, well, report_date, ord INT, component, od_in REAL, id_in REAL, length_ft REAL",
    "surveys": "report_id, well, report_date, md_ft REAL, inclination_deg REAL, azimuth_deg REAL, tvd_ft REAL",
    "mud": "report_id, well, report_date, depth_ft REAL, density_ppg REAL, funnel_viscosity_s REAL, pv_cp REAL, yp_lbf_100ft2 REAL, ph REAL, chlorides_mg_l REAL",
    "casing": "report_id, well, report_date, section, od_in REAL, weight_ppf REAL, grade, shoe_md_ft REAL",
    "figures": "figure_id TEXT PRIMARY KEY, doc_id, well, page INT, source, caption, kind, title, x_axis, y_axis, series, key_values, unverified_values, value_source, transcription, description, image_path, seconds REAL",
    "chart_readings": "well, doc_id, page INT, time, variable, value REAL, kind, scale_max REAL",
    "well_summary": "doc_id, well, field, value",
    "eowr_casing": "doc_id, well, pages, section, od_in REAL, weight_ppf REAL, grade, shoe_md_ft REAL",
    "eowr_bits": "doc_id, well, pages, bit_number, size_in REAL, manufacturer, bit_type, serial_number, depth_in_ft REAL, depth_out_ft REAL, wob_klbf REAL, rpm REAL, rop_ft_hr REAL",
    "eowr_events": "doc_id, well, pages, event_type, depth_ft REAL, time, description",
    "eowr_lessons": "doc_id, well, pages, lesson",
    "checks": "report_id, tbl, row INT, field, value, check_name, status, evidence",
    "trust": "report_id, tbl, row INT, field, value, verdict, source, failed_checks",
}


def build() -> sqlite3.Connection:
    OUTD.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    for t, cols in TABLES.items():
        con.execute(f"CREATE TABLE {t} ({cols})")

    def ins(table: str, row: dict) -> None:
        con.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))

    for f in sorted(PROC.glob("*/reports/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        d = r["data"]
        if not d:
            continue
        rid, well, date = f.stem, r["well"], d["header"].get("report_date")
        base = {"report_id": rid, "well": well, "report_date": date}
        ins("reports", {"report_id": rid, "doc_id": r["doc_id"], "well": well, "source": r["source"],
                        "pages": json.dumps(r["pages"]), "report_number": d["header"].get("report_number"),
                        "report_date": date, "report_type": d["header"].get("report_type"),
                        "rig": d["header"].get("rig"), **d["depth"], "summary": d.get("summary"),
                        "seconds": r["meta"]["seconds"]})
        for o in d["operations"]:
            ins("operations", {**base, **o, "is_npt": int(o["is_npt"])})
        for e in d["events"]:
            ins("events", {**base, **e})
        for b in d["bits"]:
            ins("bits", {**base, **b})
        for c in d["bha"]:
            c = dict(c)
            ins("bha", {**base, "ord": c.pop("order"), **c})
        for s in d["surveys"]:
            ins("surveys", {**base, **s})
        for m in d["mud"]:
            ins("mud", {**base, **m})
        for c in d["casing"]:
            ins("casing", {**base, **c})

    for f in sorted(PROC.glob("*/figures/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        d = r["data"] or {}
        values, kind = json.dumps(d.get("key_values", [])), d.get("kind")
        # value_source says how far a figure's numbers can be trusted. The saved JSON keeps the model's full answer.
        if r["doc_id"] == "D09":  # read exactly in chart_readings, so nothing the vision model said about it is kept
            source, kind = "replaced", "drilling_parameter_chart"
            d = {"series": [{"name": v, "unit": None, "trend": None} for v, _, _ in CURVES.values()],
                 "description": f"Pason drilling-parameter time log, page {r['page']}. Curves: "
                                f"{', '.join(v for v, _, _ in CURVES.values())}. Values are in table chart_readings."}
        else:
            source = None if kind is None else "chart" if kind in CHART_KINDS else "label"
        ins("figures", {"figure_id": f.stem, "doc_id": r["doc_id"], "well": r["well"], "page": r["page"],
                        "source": r["source"], "caption": r["caption"], "kind": kind,
                        "title": d.get("title"), "x_axis": d.get("x_axis"), "y_axis": d.get("y_axis"),
                        "series": json.dumps(d.get("series", [])), "value_source": source,
                        "transcription": d.get("transcription"),
                        "key_values": values if source == "label" else "[]",
                        "unverified_values": values if source == "chart" else "[]",
                        "description": d.get("description"),
                        "image_path": str(f.with_suffix(".png").relative_to(ROOT)),  # saved paths are machine-specific
                        "seconds": r["meta"]["seconds"]})

    # Merge map-reduce parts of long reports: scalars = first non-null, lists = concatenated
    for doc_dir in sorted(PROC.glob("*/summary_parts")):
        merged: dict = {}
        spans: dict[tuple, dict] = {}  # (field, row) -> page ranges of the chunks that gave the row
        for f in sorted(doc_dir.glob("*.json")):
            part = json.loads(f.read_text(encoding="utf-8"))
            d = part["data"] or {}
            for k, v in d.items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                    for row in v:
                        spans.setdefault((k, json.dumps(row, sort_keys=True)), {})[f"{part['pages'][0]}-{part['pages'][1]}"] = None
                elif v is not None and merged.get(k) is None:
                    merged[k] = v
        # chunked reading repeats and fragments table rows: drop near-empty rows and exact duplicates
        for k in ("casing", "bits", "events"):
            seen, keep = set(), []
            for row in merged.get(k, []):
                key = json.dumps(row, sort_keys=True)
                if sum(v not in (None, "", []) for v in row.values()) >= 2 and key not in seen:
                    seen.add(key)
                    keep.append(row)
            merged[k] = keep
        merged["lessons_learned"] = list(dict.fromkeys(merged.get("lessons_learned", [])))
        doc_id = doc_dir.parent.name
        (doc_dir.parent / "well_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
        base = {"doc_id": doc_id, "well": WELLS[doc_id]}
        for k, v in merged.items():
            if not isinstance(v, list):
                ins("well_summary", {**base, "field": k, "value": str(v)})
        for k, table in (("casing", "eowr_casing"), ("bits", "eowr_bits"), ("events", "eowr_events"), ("lessons_learned", "eowr_lessons")):
            for row in merged.get(k, []):
                pages = ", ".join(spans[(k, json.dumps(row, sort_keys=True))])
                ins(table, {**base, "pages": pages, **(row if isinstance(row, dict) else {"lesson": row})})

    # Deterministic readings of the Pason graph (D09), if chart_reader.py has run. A curve pinned at the top of its
    # track only shows that the true value is above the scale, so it gets no value and MAX/AVG cannot use the limit.
    if CHART_READINGS.exists():
        with gzip.open(CHART_READINGS, "rt", newline="") as f:
            for r in csv.DictReader(f):
                ins("chart_readings", {"well": WELLS["D09"], "doc_id": "D09", "page": int(r["page"]), "time": r["time"],
                                       "variable": r["variable"], "kind": r["kind"],
                                       "value": None if r["kind"] == "at_max" else float(r["value"]),
                                       "scale_max": SCALE.get(r["variable"])})
        con.execute("CREATE INDEX chart_readings_idx ON chart_readings (variable, time)")

    # Trust layer results, if trust.py has run
    for table, name in (("checks", "checks"), ("trust", "verdicts")):
        path = OUTD / "trust" / f"{name}.json"
        for row in json.loads(path.read_text(encoding="utf-8")) if path.exists() else []:
            ins(table, row)
    con.commit()

    csv_dir = OUTD / "csv"
    csv_dir.mkdir(exist_ok=True)
    for t in TABLES:
        if t == "chart_readings":  # about 100k rows, already in outputs/chart_eval/
            continue
        cur = con.execute(f"SELECT * FROM {t}")
        with open(csv_dir / f"{t}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([c[0] for c in cur.description])
            w.writerows(cur)
    return con


if __name__ == "__main__":
    con = build()
    for t in TABLES:
        print(f"{t:14s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:6d} rows")
    print("->", DB)
