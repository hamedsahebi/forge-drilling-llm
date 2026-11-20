"""Score chart readings (any method) against the Pason sensor CSV.

Input: readings CSV(.gz) with columns method, page, time, variable, value, kind
  kind = reading (a value read off the chart) | at_max / at_min (curve pinned at the track edge) | text (printed label)
Truth: the 10 s sensor record nearest in time, plus the min/max within +-WINDOW (the chart is drawn from
decimated data and a curve is ~1 pt wide, so a reading inside that band counts as correct).
    python eval_charts.py                                   # scores outputs/chart_eval/readings_*.csv.gz
    python eval_charts.py dev                               # scores outputs/chart_eval/dev/readings_*.csv.gz
"""
from __future__ import annotations

import bisect
import csv
import gzip
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from chart_reader import SCALE
from sensor_truth import load

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "chart_eval"
WINDOW = timedelta(seconds=30)
DEPTH_TOL_FT = 5.0  # depth labels: printed to the foot, placed by text position


def truth_at(times, col, t: datetime):
    """(nearest value, window min, window max) or None when the sensor had no data."""
    i = bisect.bisect_left(times, t)
    near = [j for j in (i - 1, i) if 0 <= j < len(times) and abs(times[j] - t) <= timedelta(seconds=10)]
    if not near:
        return None
    j = min(near, key=lambda j: abs(times[j] - t))
    lo, hi = bisect.bisect_left(times, t - WINDOW), bisect.bisect_right(times, t + WINDOW)
    win = [v for v in col[lo:hi] if v is not None]
    return (col[j], min(win), max(win)) if col[j] is not None and win else None


def score(rows, times, cols):
    out = []
    for r in rows:
        tr = truth_at(times, cols[r["variable"]], datetime.fromisoformat(r["time"]))
        if tr is None:
            continue
        val, (near, lo, hi) = float(r["value"]), tr
        fs = SCALE.get(r["variable"])
        if r["kind"] == "text":
            ok = lo - DEPTH_TOL_FT <= val <= hi + DEPTH_TOL_FT
        elif r["kind"] == "at_max":  # claim: true value is at or above the scale limit
            ok = hi >= fs * 0.98
        elif r["kind"] == "at_min":
            ok = lo <= fs * 0.02
        else:
            ok = lo - 0.02 * fs <= val <= hi + 0.02 * fs
        out.append({**r, "truth": near, "error": val - near, "ok": ok,
                    "err_fs": abs(val - near) / fs if fs and r["kind"] == "reading" else None,
                    "beyond_scale": bool(fs) and near > fs})  # true value off the chart: a reading here is an axis limit
    return out


def summarize(scored):
    groups = defaultdict(list)
    for s in scored:
        groups[(s["method"], s["variable"], s["kind"])].append(s)
    summary = []
    for (m, v, k), g in sorted(groups.items()):
        errs = [abs(s["error"]) for s in g]
        fs = [s["err_fs"] for s in g if s["err_fs"] is not None]
        summary.append({"method": m, "variable": v, "kind": k, "n": len(g),
                        "correct_pct": round(100 * sum(s["ok"] for s in g) / len(g), 1),
                        "median_abs_error": round(statistics.median(errs), 3),
                        "median_error_pct_full_scale": round(100 * statistics.median(fs), 2) if fs else None,
                        "true_value_beyond_scale": sum(s["beyond_scale"] for s in g)})
    return summary


if __name__ == "__main__":
    folder = EVAL / sys.argv[1] if len(sys.argv) > 1 else EVAL  # e.g. "dev" for the prompt-variant pages
    times, cols = load()
    rows = []
    for p in sorted(folder.glob("readings_*.csv.gz")):
        with gzip.open(p, "rt", newline="") as f:
            rows += list(csv.DictReader(f))
    scored = score(rows, times, cols)
    summary = summarize(scored)
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(scored)} of {len(rows)} readings had sensor truth\n")
    print(f"{'method':8} {'variable':18} {'kind':8} {'n':>6} {'correct%':>9} {'med|err|':>9} {'med%FS':>7} {'>scale':>7}")
    for s in summary:
        print(f"{s['method']:8} {s['variable']:18} {s['kind']:8} {s['n']:>6} {s['correct_pct']:>9} "
              f"{s['median_abs_error']:>9} {'' if s['median_error_pct_full_scale'] is None else s['median_error_pct_full_scale']:>7} {s['true_value_beyond_scale']:>7}")
