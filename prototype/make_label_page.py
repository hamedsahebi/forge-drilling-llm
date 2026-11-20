"""Build the hand-labelling page (outputs/label_page.html) for the evaluation reports.

Each report shows its PDF pages next to the model's extracted values, pre-filled for correction. The page
saves corrections to the artifact's database; read them back into data/labels/ before scoring.
    python make_label_page.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import fitz

from build_db import OUTD, PROC
from run_prototype import DOCS, RAW
from schema import ActivityClass

TEMPLATE = Path(__file__).with_name("label_page_template.html")
OUT = OUTD / "label_page.html"
# Evaluation set: every non-dev report outside D07, plus 12 of D07's 35 spread evenly through the well
REPORTS = ["D03_r001", "D04_r001", "D06_r001"] + [f"D07_r{n:03d}" for n in range(2, 36, 3)]
# Blind set (2026-11-06): fields start empty, no model values shown. D07 reports not looked at while the table readers
# and the activity-code table were built, so they test those readers and the LLM independently of the first labels.
BLIND = ["D07_r013", "D07_r019", "D07_r022", "D07_r024", "D07_r033", "D07_r034"]
HEADER = ["report_number", "report_date", "rig", "report_type"]
DEPTH = ["md_ft", "tvd_ft", "footage_24h_ft", "hole_size_in"]
TABLES = {  # free text (descriptions, summary, planned operation) is left out
    "operations": ["start", "end", "hours", "end_md_ft", "activity", "is_npt"],
    "casing": ["section", "od_in", "id_in", "weight_ppf", "grade", "shoe_md_ft"],  # id_in: not in the extraction schema
    "mud": ["depth_ft", "density_ppg", "funnel_viscosity_s", "pv_cp", "yp_lbf_100ft2", "ph", "chlorides_mg_l"],
    "bits": ["bit_number", "size_in", "manufacturer", "bit_type", "serial_number", "depth_in_ft", "depth_out_ft",
             "wob_klbf", "rpm", "rop_ft_hr"],
    "bha": ["component", "od_in", "id_in", "length_ft"],
    "surveys": ["md_ft", "inclination_deg", "azimuth_deg", "tvd_ft"],
}


def text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:g}" if isinstance(v, float) else str(v)


def report(rid: str, blind: bool = False) -> dict:
    doc_id = rid.split("_")[0]
    saved = json.loads((PROC / doc_id / "reports" / f"{rid}.json").read_text(encoding="utf-8"))
    d = saved["data"]
    pdf = fitz.open(RAW / next(x for x in DOCS if x["id"] == doc_id)["path"])
    pages = ["data:image/jpeg;base64," + base64.b64encode(pdf[p - 1].get_pixmap(dpi=110).tobytes("jpg", jpg_quality=70)).decode()
             for p in saved["pages"]]
    header = {f: text(d["header"].get(f)) for f in HEADER} | {f: text(d["depth"].get(f)) for f in DEPTH}
    tables = {t: [{f: text(r.get(f)) for f in cols} for r in d[t]] for t, cols in TABLES.items()}
    if blind:
        header, tables = {f: "" for f in header}, {t: [] for t in TABLES}
    return {"id": rid, "blind": blind, "well": saved["well"], "date": None if blind else d["header"].get("report_date"), "pdf_pages": saved["pages"],
            "pages": pages, "model": {"header": header, "tables": tables}}


if __name__ == "__main__":
    data = {"reports": [report(r) for r in REPORTS] + [report(r, blind=True) for r in BLIND], "header": HEADER + DEPTH, "tables": TABLES,
            "activities": [a.value for a in ActivityClass]}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("{{DATA}}", blob), encoding="utf-8")
    print(f"{len(REPORTS)} + {len(BLIND)} blind reports -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
