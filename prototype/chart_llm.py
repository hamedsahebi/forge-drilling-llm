"""Vision-model route for the 78B-32 Pason graph (D09): Qwen2.5-VL reads chart pages -> timed readings.

Same output columns as chart_reader.py, so eval_charts.py scores both routes against the sensor CSV.
The model gets the page image plus the legend as text (the legend is printed only on page 1, so a page on its
own has unlabelled curves; a human reader would also carry it over). It is asked for every curve at a few
printed tick times, and for each value whether it is a reading or the curve pinned at the track edge.
Only the chart's own legend goes into the prompt, never values read from the evaluated pages.
Raw answers are cached under outputs/chart_eval/llm_raw/<variant>/, so the run can be resumed.
Variants (compared on 3 dev pages, --dev, kept out of the 20 scored pages):
  kind_first / value_first / position: whole page, all nine curves at six times in one call
  strip: text layer locates tick and track; one crop per track, curve position as a fraction of the width
  axis:  as strip, but one curve per call and the crop carries that curve's scale numbers (best on dev)
    python chart_llm.py --dev --variant strip      # dev pages -> outputs/chart_eval/dev/readings_llm_strip.csv.gz
    python chart_llm.py --variant axis             # 20 test pages -> outputs/chart_eval/readings_llm_axis.csv.gz
    python chart_llm.py --pages 61 95              # chosen pages (1-based), output under dev/
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

import fitz
from pydantic import BaseModel, Field, create_model

from chart_reader import CURVES, PDF, PT_PER_MIN, SCALE as SCALE_MAX, TRACKS, page_axes
from llm import VISION_MODEL, extract

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "chart_eval"
RAW = EVAL / "llm_raw"
VECTOR = EVAL / "readings_vector.csv.gz"
SAMPLE_PAGES = {1, 11, 61, 151}  # the 4 D09 pages of the prototype figure phase; kept out of this test
DPI = 130
TIMES_PER_PAGE = 6

TRACK_NAMES = ["left", "middle", "right"]
UNITS = {"hook_load_klbs": "klbs", "wob_klbs": "klbs", "rpm": "RPM", "torque_kftlb": "kft-lbf",
         "rop_ft_hr": "ft/hr", "flow_pct": "percent", "on_bottom_hours": "hrs", "circulating_hours": "hrs",
         "mud_volume_bbl": "barrels"}
COLOUR_NAMES = {"hook_load_klbs": "red", "wob_klbs": "dark blue", "rpm": "bright green", "torque_kftlb": "orange",
                "rop_ft_hr": "purple", "flow_pct": "teal green", "on_bottom_hours": "magenta",
                "circulating_hours": "light blue", "mud_volume_bbl": "olive green"}

KIND = (Literal["reading", "at_max", "at_min", "not_visible"],
        Field(description="reading = curve inside the track; at_max / at_min = curve pinned at the right / left "
                          "track edge (true value is off the scale); not_visible = no curve at this time"))
VALUE = (float | None, Field(description="value in the curve's units on its 0-to-max scale; empty if not_visible"))
POSITION = (float | None, Field(description="where the curve crosses the grid line: 0 = left edge of its track, "
                                            "1 = right edge; empty if not_visible"))
# kind_first: decide "pinned at the edge" before writing a number; value_first: the reverse;
# position: the model only locates the curve and the scale arithmetic is done in code
FIELDS = {"kind_first": {"kind": KIND, "value": VALUE},
          "value_first": {"value": VALUE, "kind": KIND},
          "position": {"position": POSITION, "kind": KIND}}


def page_schema(variant: str) -> type[BaseModel]:
    cv = create_model(f"Curve_{variant}", **FIELDS[variant])
    row = create_model(f"TimeRow_{variant}", time=(str, Field(description="the HH:MM tick label")),
                       **{v: (cv, ...) for v, _, _ in CURVES.values()})
    return create_model(f"ChartPage_{variant}", rows=(list[row], ...))


def legend() -> str:
    lines = []
    for var, track, vmax in CURVES.values():
        lines.append(f"- {var}: {COLOUR_NAMES[var]} curve in the {TRACK_NAMES[track]} track, "
                     f"scale 0 (left edge) to {vmax:g} {UNITS[var]} (right edge)")
    return "\n".join(lines)


SYSTEM = f"""You read values off a drilling time-log chart (Pason EDR log).
Time runs down the page; the HH:MM labels on the far left mark the major horizontal grid lines.
The blue numbers next to them are bit depth and the numbers right of the tracks are TVD; ignore both.
There are three tracks, each with three colour-coded curves on a linear scale. Legend:
{legend()}
For each requested time, find the grid line with that label and read where each curve crosses it:
position = horizontal place of the curve inside its track, as a fraction of the track width (0 to 1);
value = position x scale max.
If the curve runs along the right edge of its track, the true value is above the scale: kind = at_max.
If it runs along the left edge, kind = at_min. Do not report the scale limit as a reading."""


def pick_pages(n: int = 20, n_dev: int = 3) -> tuple[list[int], list[int]]:
    """(test, dev) pages with drilling activity (ROP and WOB in range at several times), spread evenly
    through the well; the dev pages sit between test pages and are used only to choose the prompt variant."""
    active = Counter()
    with gzip.open(VECTOR, "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["variable"] in ("rop_ft_hr", "wob_klbs") and r["kind"] == "reading" and float(r["value"]) > 1:
                active[int(r["page"])] += 1
    pages = sorted(p for p, c in active.items() if c >= 25 and p not in SAMPLE_PAGES)
    test = [pages[int(i * len(pages) / n)] for i in range(n)]
    rest = [p for p in pages if p not in test]
    return test, [rest[int((i + 0.5) * len(rest) / n_dev)] for i in range(n_dev)]


def ask_times(ticks) -> list[datetime]:
    step = max(1, len(ticks) // TIMES_PER_PAGE)
    return [t for _, t in ticks[1::step]][:TIMES_PER_PAGE]  # skip the first tick: often cut at the page top


def read_page(doc, pno: int, ticks, variant: str) -> dict:
    dst = RAW / variant / f"page_{pno:03d}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    times = ask_times(ticks)
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        doc[pno - 1].get_pixmap(dpi=DPI).save(img)
        labels = ", ".join(f"{t:%H:%M}" for t in times)
        out, meta = extract(page_schema(variant), SYSTEM, f"Read all nine curves at these times: {labels}.",
                            images=[str(img)], model=VISION_MODEL, num_ctx=8192, num_predict=3000)
    res = {"page": pno, "times": [t.isoformat() for t in times], "meta": meta,
           "result": out.model_dump() if out else None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))
    return res


STRIP_SYSTEM = """You read a narrow crop of one track of a drilling time-log chart.
The left edge of the image is 0 and the right edge is the scale maximum of every curve in it.
A dashed black horizontal line marks the requested time. For each named curve, give where it crosses the
dashed line, as a fraction of the image width (0 = left edge, 1 = right edge). Leave it empty if that curve
does not cross the dashed line."""
STRIP_MIN = 10  # crop +- this many minutes around the tick
EDGE_FRAC = 0.02  # position this close to an edge -> the value is typed as an axis limit in code


def read_strips(doc, pno: int, ticks) -> dict:
    """Hybrid route: the text layer locates each tick (y) and track (x), the model only places the curves."""
    dst = RAW / "strip" / f"page_{pno:03d}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    times = ask_times(ticks)
    strips, secs = [], 0.0
    y_of = {t: y for y, t in ticks}
    pad = STRIP_MIN * PT_PER_MIN
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            y = y_of[t]
            for track, (x0, x1) in enumerate(TRACKS):
                curves = [v for v, tr, _ in CURVES.values() if tr == track]
                one = fitz.open()  # the marker goes on a fresh copy so later crops do not show earlier lines
                one.insert_pdf(doc, from_page=pno - 1, to_page=pno - 1)
                one[0].draw_line((x0, y), (x1, y), color=(0, 0, 0), width=0.6, dashes="[2 2] 0")
                img = Path(tmp) / "strip.png"
                one[0].get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(x0, y - pad, x1, y + pad)).save(img)
                one.close()
                schema = create_model("Strip", **{v: (float | None, Field(description=f"{COLOUR_NAMES[v]} curve"))
                                                  for v in curves})
                ask = "Curves: " + "; ".join(f"{v} = {COLOUR_NAMES[v]}" for v in curves) + "."
                out, meta = extract(schema, STRIP_SYSTEM, ask, images=[str(img)], model=VISION_MODEL,
                                    num_ctx=4096, num_predict=300)
                secs += meta["seconds"]
                strips.append({"time": t.isoformat(), "track": track, "meta": meta,
                               "result": out.model_dump() if out else None})
    res = {"page": pno, "times": [t.isoformat() for t in times], "meta": {"seconds": secs}, "strips": strips}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))
    return res


AXIS_SYSTEM = """You read one value off a narrow crop of a drilling time-log chart.
The numbers printed along the top of the image label the vertical grid lines (the curve's own scale).
A dashed black horizontal line marks the requested time. Find where the named curve crosses the dashed line
and read its value against the printed scale, interpolating between grid lines. Leave it empty if the named
curve does not cross the dashed line."""


class AxisValue(BaseModel):
    value: float | None = Field(description="value where the curve crosses the dashed line, in the printed scale")


def read_axis(doc, pno: int, ticks) -> dict:
    """Hybrid route, one curve per call: the crop carries that curve's scale numbers on the grid lines."""
    dst = RAW / "axis" / f"page_{pno:03d}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    times = ask_times(ticks)
    strips, secs = [], 0.0
    y_of = {t: y for y, t in ticks}
    pad = STRIP_MIN * PT_PER_MIN
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            y = y_of[t]
            for var, track, vmax in CURVES.values():
                x0, x1 = TRACKS[track]
                one = fitz.open()
                one.insert_pdf(doc, from_page=pno - 1, to_page=pno - 1)
                pg = one[0]
                pg.draw_line((x0, y), (x1, y), color=(0, 0, 0), width=0.6, dashes="[2 2] 0")
                top = y - pad
                pg.draw_rect(fitz.Rect(x0 - 8, top, x1 + 8, top + 7), color=None, fill=(1, 1, 1))
                for k in range(0, 11, 2):  # labels on every second grid line (the track has 10 divisions)
                    lab = f"{vmax * k / 10:g}"
                    xc = x0 + (x1 - x0) * k / 10 - 1.4 * len(lab)
                    pg.insert_text((min(max(xc, x0 - 6), x1 + 6 - 2.8 * len(lab)), top + 5.5), lab, fontsize=5.5)
                img = Path(tmp) / "axis.png"
                pg.get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(x0 - 8, top, x1 + 8, y + pad)).save(img)
                one.close()
                out, meta = extract(AxisValue, AXIS_SYSTEM,
                                    f"Curve: the {COLOUR_NAMES[var]} line ({var}, {UNITS[var]}).",
                                    images=[str(img)], model=VISION_MODEL, num_ctx=4096, num_predict=100)
                secs += meta["seconds"]
                strips.append({"time": t.isoformat(), "track": track, "variable": var, "meta": meta,
                               "result": out.model_dump() if out else None})
    res = {"page": pno, "times": [t.isoformat() for t in times], "meta": {"seconds": secs}, "strips": strips}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))
    return res


def axis_rows(res: dict) -> tuple[list[tuple], Counter]:
    rows, stats = [], Counter()
    for s in res["strips"]:
        stats["asked"] += 1
        v = (s["result"] or {}).get("value")
        if v is None:
            stats["not_visible"] += 1
            continue
        stats["answered"] += 1
        vmax = SCALE_MAX[s["variable"]]
        v = min(max(v, 0.0), vmax)
        kind = "at_max" if v >= vmax * (1 - EDGE_FRAC) else "at_min" if v <= vmax * EDGE_FRAC else "reading"
        rows.append(("llm_axis", res["page"], s["time"][:19], s["variable"], round(v, 3), kind))
    return rows, stats


def strip_rows(res: dict) -> tuple[list[tuple], Counter]:
    rows, stats = [], Counter()
    for s in res["strips"]:
        for var, track, vmax in CURVES.values():
            if track != s["track"]:
                continue
            stats["asked"] += 1
            pos = (s["result"] or {}).get(var)
            if pos is None:
                stats["not_visible"] += 1
                continue
            stats["answered"] += 1
            pos = min(max(pos, 0.0), 1.0)
            kind = "at_max" if pos >= 1 - EDGE_FRAC else "at_min" if pos <= EDGE_FRAC else "reading"
            rows.append(("llm_strip", res["page"], s["time"][:19], var, round(pos * vmax, 3), kind))
    return rows, stats


def to_rows(res: dict, variant: str) -> tuple[list[tuple], Counter]:
    by_hm = {datetime.fromisoformat(t).strftime("%H:%M"): datetime.fromisoformat(t) for t in res["times"]}
    rows, stats = [], Counter()
    stats["asked"] = len(by_hm) * len(CURVES)
    for row in (res["result"] or {}).get("rows", []):
        t = by_hm.get(row["time"].strip()[:5])
        if t is None:
            stats["unknown_time"] += 1
            continue
        for var, _, vmax in CURVES.values():
            cv = row[var]
            value = cv["position"] * vmax if variant == "position" and cv["position"] is not None else cv.get("value")
            if cv["kind"] == "not_visible" or value is None:
                stats["not_visible"] += 1
                continue
            stats["answered"] += 1
            rows.append((f"llm_{variant}", res["page"], t.isoformat(timespec="seconds"), var, round(value, 3),
                         cv["kind"]))
    return rows, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, nargs="*", help="1-based page numbers (default: 20 drilling pages)")
    ap.add_argument("--dev", action="store_true", help="the 3 dev pages, output under outputs/chart_eval/dev/")
    ap.add_argument("--variant", choices=[*FIELDS, "strip", "axis"], default="axis")
    args = ap.parse_args()
    doc = fitz.open(PDF)
    axes = page_axes(doc)
    test, dev = pick_pages()
    pages = args.pages or (dev if args.dev else test)
    out_dir = EVAL / "dev" if args.dev or args.pages else EVAL
    print(f"{args.variant} on pages {pages}")
    all_rows, total = [], Counter()
    for pno in pages:
        if args.variant == "axis":
            res = read_axis(doc, pno, axes[pno - 1])
            rows, stats = axis_rows(res)
        elif args.variant == "strip":
            res = read_strips(doc, pno, axes[pno - 1])
            rows, stats = strip_rows(res)
        else:
            res = read_page(doc, pno, axes[pno - 1], args.variant)
            rows, stats = to_rows(res, args.variant)
        all_rows += rows
        total += stats
        print(f"page {pno:3d}: {stats['answered']:3d}/{stats['asked']} values, {res['meta']['seconds']:.0f} s"
              + ("" if res.get("result", 1) else f"  FAILED {res['meta'].get('error', '')[:80]}"), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / f"readings_llm_{args.variant}.csv.gz", "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "page", "time", "variable", "value", "kind"])
        wr.writerows(all_rows)
    print(f"{len(all_rows)} readings; coverage {total['answered']}/{total['asked']}; {dict(total)}")
