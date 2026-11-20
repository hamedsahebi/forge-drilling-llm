"""Score the saved model extraction against the hand labels in data/labels/labels/.

Rows are matched on content (see match_rows; `_m`, the model row a label row came from, breaks ties). Each labelled field is counted as one of:
  correct  - both filled and equal, or both empty
  wrong    - both filled, different values
  missed   - label filled, model empty (includes rows the labeller added)
  extra    - label empty, model filled (includes model rows the labeller removed)
Accuracy = correct / all fields; recall = filled-and-equal / label filled; precision = filled-and-equal / model filled.
Left out of the totals and reported separately: `id_in` (casing, not in the extraction schema) and the BHA table
(the labelling rule empties it when a report gives names only, so its "extra" rows are a convention, not model errors).
    python score_labels.py            # writes outputs/label_scores.json
    python score_labels.py --proc ../outputs/repeat/B3/run1 --out ../outputs/repeat/B3/run1_scores.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_db import OUTD, PROC
from make_label_page import DEPTH, HEADER, TABLES, text

LABELS = Path(__file__).resolve().parents[1] / "data" / "labels" / "labels"
OUT = OUTD / "label_scores.json"
NOT_IN_SCHEMA = {("casing", "id_in")}
SEPARATE_TABLES = {"bha"}
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def norm(v: str):
    v = (v or "").strip()
    if not v:
        return None
    m = TIME_RE.match(v)
    if m:  # 0:00 == 00:00 == 24:00
        return ("t", (int(m[1]) * 60 + int(m[2])) % 1440)
    try:
        return ("n", round(float(v.replace(",", "")), 6))
    except ValueError:
        return ("s", re.sub(r"\s+", " ", v.lower()))


def outcome(label: str, model: str) -> str:
    a, b = norm(label), norm(model)
    if a is None:
        return "correct" if b is None else "extra"
    if b is None:
        return "missed"
    return "correct" if a == b else "wrong"


def model_fields(rid: str) -> dict:
    d = json.loads((PROC / rid.split("_")[0] / "reports" / f"{rid}.json").read_text(encoding="utf-8"))["data"]
    header = {f: text(d["header"].get(f)) for f in HEADER} | {f: text(d["depth"].get(f)) for f in DEPTH}
    tables = {t: [{f: text(r.get(f)) for f in cols} for r in d[t]] for t, cols in TABLES.items()}
    for row, o in zip(tables["operations"], d["operations"]):
        row["_src"] = o.get("activity_source") or "llm"  # code or llm; not a scored field
    return {"header": header, "tables": tables}


def match_rows(lrows: list[dict], mrows: list[dict], cols: list[str]) -> dict[int, int]:
    """Label row -> model row, greedy on the number of filled fields that agree; `_m` breaks ties.
    `_m` alone is not enough: it points at the model rows as they were when labelled, and code steps such as
    apply_casing_table() can replace them later."""
    cand = []
    for li, l in enumerate(lrows):
        for mi, m in enumerate(mrows):
            same = sum(1 for f in cols if norm(l.get(f, "")) is not None and norm(l.get(f, "")) == norm(m.get(f, "")))
            if same or l.get("_m") == mi:
                cand.append((-same, l.get("_m") != mi, li, mi))
    pair, taken = {}, set()
    for _, _, li, mi in sorted(cand):
        if li not in pair and mi not in taken:
            pair[li] = mi
            taken.add(mi)
    return pair


def score_report(rid: str, lab: dict) -> list[tuple]:
    """(table, field, outcome, label, model) for every labelled field of one report."""
    mod = model_fields(rid)
    out = [("header", f, outcome(lab["header"].get(f, ""), mod["header"][f]), lab["header"].get(f, ""), mod["header"][f])
           for f in HEADER + DEPTH]
    for t, cols in TABLES.items():
        mrows, lrows = mod["tables"][t], lab["tables"].get(t, [])
        pair, used = match_rows(lrows, mrows, cols), set()
        for li, row in enumerate(lrows):
            m = pair.get(li)
            mrow = mrows[m] if m is not None else {}
            if m is not None:
                used.add(m)
            out += [(t, f + ("@" + mrow.get("_src", "none") if t == "operations" and f == "activity" else ""),
                     outcome(row.get(f, ""), mrow.get(f, "")), row.get(f, ""), mrow.get(f, "")) for f in cols]
        for i, mrow in enumerate(mrows):  # removed model rows (and any the label file does not mention)
            if i not in used:
                out += [(t, f, outcome("", mrow[f]), "", mrow[f]) for f in cols if mrow[f]]
    return out


def summarise(rows) -> dict:
    c = defaultdict(int)
    for _, _, o, *_ in rows:
        c[o] += 1
    n, hit = sum(c.values()), c["correct"]
    lab_filled = c["wrong"] + c["missed"] + sum(1 for r in rows if r[2] == "correct" and norm(r[3]) is not None)
    mod_filled = c["wrong"] + c["extra"] + sum(1 for r in rows if r[2] == "correct" and norm(r[4]) is not None)
    tp = lab_filled - c["wrong"] - c["missed"]
    return {"fields": n, **{k: c[k] for k in ("correct", "wrong", "missed", "extra")},
            "accuracy": round(hit / n, 3) if n else None,
            "recall": round(tp / lab_filled, 3) if lab_filled else None,
            "precision": round(tp / mod_filled, 3) if mod_filled else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", help="score this extraction directory instead of data/processed")
    ap.add_argument("--out", help="write the scores here instead of outputs/label_scores.json")
    a = ap.parse_args()
    PROC = Path(a.proc).resolve() if a.proc else PROC
    OUT = Path(a.out).resolve() if a.out else OUT
    all_rows, per_report = [], {}
    for f in sorted(LABELS.glob("*.json")):
        lab = json.loads(f.read_text(encoding="utf-8"))
        if lab.get("status") != "done":
            print(f"skip {f.stem}: status {lab.get('status')}")
            continue
        rows = score_report(f.stem, lab)
        src = "blind" if lab.get("blind") else "claude" if lab.get("prefill") == "claude" else "model"  # how the label started
        all_rows += [(f.stem, src, *r) for r in rows]
        per_report[f.stem] = summarise([r for r in rows if (r[0], r[1]) not in NOT_IN_SCHEMA and r[0] not in SEPARATE_TABLES])

    scored = [r for r in all_rows if (r[2], r[3]) not in NOT_IN_SCHEMA and r[2] not in SEPARATE_TABLES]
    by = lambda key: {k: summarise([r[2:] for r in scored if key(r) == k]) for k in sorted({key(r) for r in scored})}
    result = {
        "total": summarise([r[2:] for r in scored]),
        "by_table": by(lambda r: r[2]),
        "by_field": by(lambda r: f"{r[2]}.{r[3].split('@')[0]}"),
        "activity_by_source": by(lambda r: r[3] if r[3].startswith("activity@") else ""),
        "by_prefill": by(lambda r: r[1]),
        "by_group": by(lambda r: "D07" if r[0].startswith("D07") else "other"),
        "by_set_table": by(lambda r: f"{'blind' if r[1] == 'blind' else 'first15'}.{r[3] if r[3].startswith('activity@') else r[2]}"),
        "by_report": per_report,
        "not_in_schema": {f"{t}.{f}": summarise([r[2:] for r in all_rows if (r[2], r[3]) == (t, f)]) for t, f in NOT_IN_SCHEMA},
        "separate_tables": {t: summarise([r[2:] for r in all_rows if r[2] == t]) for t in SEPARATE_TABLES},
        "activity_confusion": dict(Counter(f"{r[5]} -> {r[6]}" for r in scored
                                           if r[3].startswith("activity") and r[4] == "wrong").most_common()),
        "errors": [{"report": r[0], "table": r[2], "field": r[3], "outcome": r[4], "label": r[5], "model": r[6]}
                   for r in scored if r[4] != "correct"],
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def line(name, s):
        print(f"{name:<28}{s['fields']:>6}{s['accuracy']:>8}{s['recall'] or '-':>8}{s['precision'] or '-':>8}"
              f"{s['wrong']:>7}{s['missed']:>7}{s['extra']:>7}")
    print(f"{'':<28}{'fields':>6}{'acc':>8}{'recall':>8}{'prec':>8}{'wrong':>7}{'missed':>7}{'extra':>7}")
    line("TOTAL", result["total"])
    result["activity_by_source"].pop("", None)
    for sect in ("by_group", "by_prefill", "by_table", "by_field", "activity_by_source"):
        print(f"-- {sect}")
        for k, s in result[sect].items():
            line(k, s)
    print("-- left out of totals")
    for k, s in (result["not_in_schema"] | result["separate_tables"]).items():
        line(k, s)
    print("-- activity confusion (label -> model)")
    for k, n in result["activity_confusion"].items():
        print(f"  {n:>3}  {k}")
    print(f"-> {OUT}")
