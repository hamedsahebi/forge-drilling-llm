"""Score the NPT flags against the operator's Non-Prod marks, and check those marks against the NPT hours
the same reports state in their Comments. Run after `run_prototype.py --npt-column`.

Writes outputs/npt_eval/summary.json. D01, D02 and D05 are dev reports and are not scored.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

from parse_pdf import read_npt_column
from run_prototype import DOCS, OUT, RAW, clock

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "outputs" / "npt_eval" / "summary.json"
DEV = {"D01", "D02", "D05"}
STATED_RE = re.compile(r"Total hours of NPT on (?:daily )?report(?: \d+)?(?: due to \S+)? ([\d.]+) HRS", re.I)


def score(rows: list[dict]) -> dict:
    tp = sum(r["marked"] and r["llm"] for r in rows)
    fp = sum(r["llm"] and not r["marked"] for r in rows)
    fn = sum(r["marked"] and not r["llm"] for r in rows)
    p, q = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"rows": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": len(rows) - tp - fp - fn,
            "precision": round(p, 3), "recall": round(q, 3), "f1": round(2 * p * q / max(p + q, 1e-9), 3),
            "hours_marked": sum(r["hours"] for r in rows if r["marked"]),
            "hours_llm": sum(r["hours"] for r in rows if r["llm"]),
            "hours_both": sum(r["hours"] for r in rows if r["marked"] and r["llm"])}


def evaluate(doc: dict) -> dict:
    pdf = fitz.open(RAW / doc["path"])
    rows, reports = [], []
    for f in sorted((OUT / doc["id"] / "reports").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        pages = [p - 1 for p in rec["pages"]]
        read = read_npt_column(pdf, pages) or []
        ops = {}
        for o in rec["data"]["operations"]:
            ops.setdefault((clock(o.get("start")), clock(o.get("end"))), []).append(o)
        for r in read:
            hit = ops.get((clock(r["start"]), clock(r["end"])))
            if hit:
                rows.append({**r, "llm": hit.pop(0)["llm_is_npt"]})
        stated = [float(v) for i in pages for v in STATED_RE.findall(pdf[i].get_text())]
        reports.append({"report": f.stem, "pages": rec["pages"], "rows": len(read),
                        "hours_logged": sum(r["hours"] for r in read),
                        "marks": sum(r["marked"] for r in read),
                        "hours_marked": sum(r["hours"] for r in read if r["marked"]),
                        "hours_stated": sum(stated) if stated else None})
    return {"score": score(rows), "reports": reports}


if __name__ == "__main__":
    out = {}
    for d in DOCS:
        if d["kind"] != "ddr" or d["id"] in DEV:
            continue
        res = evaluate(d)
        if not any(r["marks"] for r in res["reports"]):
            out[d["id"]] = {"marks": 0}
            print(f"{d['id']}: no marks, not scored")
            continue
        out[d["id"]] = res
        print(d["id"], res["score"])
        for r in res["reports"]:
            if r["marks"] or r["hours_stated"] is not None:
                print(f"  {r['report']} p{r['pages'][0]}: rows {r['rows']}, marks {r['marks']}, "
                      f"marked {r['hours_marked']} h, stated {r['hours_stated']} h")
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, indent=1), encoding="utf-8")
