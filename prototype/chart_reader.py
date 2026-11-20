"""Deterministic reader for the 78B-32 Pason graph (D09): vector curves + text layer -> timed readings.

The chart is a time log: time runs down the page (36 pt per 10 min), three tracks of three colour-coded curves,
bit depth labels on the left and TVD at each 10-min tick on the right. The legend (page 1 header) fixes each
curve's colour and scale, so values can be read exactly from the PDF drawing commands. This is the
"text-layer / vector parser" route that the vision model is compared against; its own error against the
sensor CSV shows how much of any remaining error comes from the chart itself (decimation, line width).
    python chart_reader.py                 # all pages -> outputs/chart_eval/readings_vector.csv.gz
"""
from __future__ import annotations

import bisect
import csv
import gzip
import re
from datetime import datetime, timedelta
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "forge" / "78B-32_Pason_graph.pdf"
OUT = ROOT / "outputs" / "chart_eval" / "readings_vector.csv.gz"
FIRST_DAY = datetime(2021, 6, 27)  # date of the first tick on page 1 ("23:55", label "Jun 27, 2021")
PT_PER_MIN = 3.6
LAST_TIME = datetime(2021, 8, 10, 16, 11)  # end of the sensor data. The block after it on page 246 is from 27 Oct
                                           # and has no day label, so its date cannot be read from the axis
TIME_RE = re.compile(r"^\d\d:\d\d$")
DAY_RE = re.compile(r"[A-Z][a-z]{2} \d\d, \d{4}")

# (track x0, x1), colour -> variable, scale max (all scales start at 0); from the page-1 legend
TRACKS = [(108.0, 219.0), (219.0, 330.0), (330.0, 441.0)]
CURVES = {
    (1.0, 0.0, 0.0): ("hook_load_klbs", 0, 150.0),
    (0.0, 0.0, 1.0): ("wob_klbs", 0, 30.0),
    (0.0, 1.0, 0.0): ("rpm", 0, 200.0),
    (1.0, 0.42, 0.0): ("torque_kftlb", 1, 10.0),
    (0.42, 0.0, 1.0): ("rop_ft_hr", 1, 100.0),
    (0.0, 0.8, 0.33): ("flow_pct", 1, 100.0),
    (0.8, 0.0, 0.33): ("on_bottom_hours", 2, 100.0),
    (0.0, 0.42, 1.0): ("circulating_hours", 2, 100.0),
    (0.33, 0.8, 0.0): ("mud_volume_bbl", 2, 200.0),
}
SCALE = {v: m for v, _, m in CURVES.values()}
EDGE = 0.6  # pt; a curve this close to the track edge is pinned at the scale limit


def page_axes(doc: fitz.Document) -> list[list[tuple[float, datetime]]]:
    """Per page, the (y of tick line, datetime) pairs. Ticks carry only HH:MM, so the date is carried across
    pages; when the clock goes backwards it takes the nearest printed day label ("Aug 05, 2021"), because
    the chart skips days without data, else the next day."""
    day, last, axes = FIRST_DAY, None, []
    for page in doc:
        majors = sorted({round(it[1].y, 2) for g in page.get_drawings() if g.get("color") == (0.5, 0.5, 0.5)
                         for it in g["items"] if it[0] == "l" and abs(it[1].y - it[2].y) < 0.01})
        words = page.get_text("words")
        labels = [(y0, datetime.strptime(m.group(), "%b %d, %Y"))
                  for x0, y0, _, _, w, *_ in words if 45 < x0 < 60 and w in ("Jun", "Jul", "Aug", "Sep", "Oct")
                  if (m := DAY_RE.search(" ".join(x[4] for x in sorted(words) if abs(x[1] - y0) < 1 and 45 < x[0] < 100)))]
        ticks = []
        for x0, y0, x1, y1, w, *_ in sorted(words, key=lambda w: w[1]):
            if x0 < 60 and TIME_RE.match(w):
                hm = datetime.strptime(w, "%H:%M").time()
                if last is not None and hm < last:
                    later = [lb for lb in labels if lb[1] > day]
                    day = min(later, key=lambda lb: abs(lb[0] - y0))[1] if later else day + timedelta(days=1)
                last = hm
                yc = (y0 + y1) / 2
                y = min(majors, key=lambda m: abs(m - yc)) if majors else yc
                ticks.append((y, datetime.combine(day.date(), hm)))
        axes.append(ticks)
    return axes


def y_to_time(ticks, y: float) -> datetime:
    y0, t0 = min(ticks, key=lambda t: abs(t[0] - y))
    return t0 + timedelta(minutes=(y - y0) / PT_PER_MIN)


def sample_points(ticks) -> list[tuple[float, datetime]]:
    """(y, time) every 5 min. Time is continuous only between ticks 10 min apart: the chart closes up long data
    gaps, so a page can jump days between two ticks. The point halfway across such a jump has no defined time."""
    half, pts = timedelta(minutes=5), [(ticks[0][0] - 5 * PT_PER_MIN, ticks[0][1] - timedelta(minutes=5))]
    for i, (y, t) in enumerate(ticks):
        pts.append((y, t))
        if i + 1 == len(ticks) or ticks[i + 1][1] - t == 2 * half:
            pts.append((y + 5 * PT_PER_MIN, t + half))
    return pts


def curve_points(page: fitz.Page) -> dict[str, tuple[list[float], list[float]]]:
    """Each curve as y-sorted (y, x) points from its line segments."""
    pts: dict[str, list] = {}
    for g in page.get_drawings():
        col = tuple(round(c, 2) for c in g.get("color") or ())
        if col in CURVES:
            for it in g["items"]:
                if it[0] == "l":
                    pts.setdefault(CURVES[col][0], []).extend([(it[1].y, it[1].x), (it[2].y, it[2].x)])
    out = {}
    for v, p in pts.items():
        p.sort()
        out[v] = ([y for y, _ in p], [x for _, x in p])
    return out


def read_curve(ys: list[float], xs: list[float], y: float) -> float | None:
    i = bisect.bisect_left(ys, y)
    if i == 0 or i == len(ys):
        return None
    (ya, xa), (yb, xb) = (ys[i - 1], xs[i - 1]), (ys[i], xs[i])
    if yb - ya > 3 * PT_PER_MIN:  # gap in the curve (no data), do not interpolate across it
        return None
    return xa if yb == ya else xa + (xb - xa) * (y - ya) / (yb - ya)


def page_readings(page: fitz.Page, ticks):
    """Rows (time, variable, value, kind). kind: reading | at_max | at_min (curve pinned at a track edge)."""
    rows = []
    curves = curve_points(page)
    top = 270.0 if page.number == 0 else 0.0
    for y, t in sample_points(ticks):
        if not top <= y <= page.rect.height:
            continue
        for col, (var, track, vmax) in CURVES.items():
            if var not in curves or (x := read_curve(*curves[var], y)) is None:
                continue
            x0, x1 = TRACKS[track]
            kind = "at_max" if x >= x1 - EDGE else "at_min" if x <= x0 + EDGE else "reading"
            rows.append((t, var, round((x - x0) / (x1 - x0) * vmax, 3), kind))
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        if not re.fullmatch(r"\d{2,5}(\.\d+)?", w) or (page.number == 0 and y0 < 262):
            continue
        tm = y_to_time(ticks, (y0 + y1) / 2)
        if 80 <= x0 < 108:
            rows.append((tm, "bit_depth_ft", float(w), "text"))
        elif 440 <= x0 < 480:
            rows.append((tm, "tvd_ft", float(w), "text"))
    return rows


if __name__ == "__main__":
    doc = fitz.open(PDF)
    axes = page_axes(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(OUT, "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "page", "time", "variable", "value", "kind"])
        for page, ticks in zip(doc, axes):
            if len(ticks) < 2:
                continue
            for t, var, val, kind in page_readings(page, ticks):
                if t > LAST_TIME:
                    continue
                wr.writerow(["vector", page.number + 1, t.isoformat(timespec="seconds"), var, val, kind])
                n += 1
    print(f"{n} readings from {len(doc)} pages -> {OUT}")
