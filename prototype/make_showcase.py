"""Build outputs/showcase.html: source pages vs extracted data, figure gallery, cross-report charts, agent Q&A."""
from __future__ import annotations

import base64
import html
import json
import statistics
import sqlite3
from collections import defaultdict
from pathlib import Path

import fitz

from build_db import DB, OUTD, ROOT
from run_prototype import DOCS, RAW

DOC_LABEL = {
    "ddr": "Daily drilling report", "long_report": "End-of-Well Report", "plot_document": "Plot document",
}
ACT_ORDER = ["drilling", "tripping", "casing", "cementing", "circulating", "bop_testing"]  # rest -> other
SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"]


def jpeg_uri(pix: fitz.Pixmap, q: int = 60) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(pix.tobytes(output="jpg", jpg_quality=q)).decode()


def png_file_uri(path: Path, max_w: int = 520) -> str:
    pix = fitz.Pixmap(str(path))
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.width > max_w:
        pix.shrink(1 if pix.width < 2 * max_w else 2)
    return jpeg_uri(pix, 62)


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def rows(con, q, *a):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(q, a)]


def depth_chart(points: list[tuple[int, float, str]], well: str) -> str:
    """Days-vs-depth line (depth increases downward, as drillers plot it)."""
    if len(points) < 2:
        return ""
    W, H, L, R, T, B = 640, 300, 64, 16, 16, 40
    xs = [p[0] for p in points]
    ymax = max(p[1] for p in points)
    ymax = (int(ymax / 1000) + 1) * 1000
    x0, x1 = min(xs), max(xs)
    sx = lambda x: L + (x - x0) / max(1, x1 - x0) * (W - L - R)
    sy = lambda y: T + y / ymax * (H - T - B)
    grid = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
        f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:,}</text>'
        for v in range(0, ymax + 1, max(1000, ymax // 5 // 1000 * 1000)))
    xt = "".join(f'<text x="{sx(d):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{d}</text>'
                 for d in range(x0, x1 + 1, max(1, (x1 - x0) // 8)))
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(d):.1f},{sy(v):.1f}" for i, (d, v, _) in enumerate(points))
    dots = "".join(
        f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="4" class="dot" data-tip="Day {d} · {esc(dt)} · MD {v:,.0f} ft"/>'
        for d, v, dt in points)
    last = points[-1]
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Days versus depth for well {esc(well)}">
{grid}{xt}
<text x="{(L + W - R) / 2}" y="{H - 4}" class="axis" text-anchor="middle">Report day</text>
<text x="14" y="{(T + H - B) / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(T + H - B) / 2})">Measured depth (ft)</text>
<path d="{path}" class="line"/>{dots}
<text x="{sx(last[0]) - 6:.1f}" y="{sy(last[1]) - 10:.1f}" class="lbl" text-anchor="end">{last[1]:,.0f} ft</text>
</svg>'''


def activity_bars(con) -> str:
    data = defaultdict(lambda: defaultdict(float))
    for r in rows(con, "SELECT well, activity, SUM(hours) h FROM operations GROUP BY well, activity"):
        a = r["activity"] if r["activity"] in ACT_ORDER else "other"
        data[r["well"]][a] += r["h"] or 0
    cats = ACT_ORDER + ["other"]
    out = ['<div class="legend">' + "".join(
        f'<span><i style="background:var({SERIES[i]})"></i>{c.replace("_", " ")}</span>' for i, c in enumerate(cats)) + "</div>"]
    for well, d in sorted(data.items()):
        tot = sum(d.values()) or 1
        segs = "".join(
            f'<span class="seg" style="flex:{d[c]:.2f};background:var({SERIES[i]})" '
            f'data-tip="{esc(well)} · {c.replace("_", " ")}: {d[c]:.1f} h ({d[c] / tot:.0%})"></span>'
            for i, c in enumerate(cats) if d[c] > 0)
        out.append(f'<div class="barrow"><div class="barlab">{esc(well)}<small>{tot:.0f} h logged</small></div>'
                   f'<div class="bar">{segs}</div></div>')
    return "\n".join(out)


def build() -> Path:
    con = sqlite3.connect(DB)
    reports = rows(con, "SELECT * FROM reports ORDER BY well, report_date")
    n = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
         for t in ["reports", "operations", "events", "bits", "bha", "surveys", "mud", "casing", "figures"]}
    total_sec = (con.execute("SELECT SUM(seconds) FROM reports").fetchone()[0] or 0) + \
                (con.execute("SELECT SUM(seconds) FROM figures").fetchone()[0] or 0)

    # --- report viewer data (first page image + all extracted tables)
    viewer = []
    pdfs = {}
    for r in reports:
        doc = next(d for d in DOCS if d["id"] == r["doc_id"])
        pdf = pdfs.setdefault(doc["path"], fitz.open(RAW / doc["path"]))
        pages = json.loads(r["pages"])
        img = jpeg_uri(pdf[pages[0] - 1].get_pixmap(dpi=72), 55)
        data = json.loads((ROOT / "data/processed" / r["doc_id"] / "reports" / f"{r['report_id']}.json")
                          .read_text(encoding="utf-8"))
        viewer.append({"id": r["report_id"], "well": r["well"], "date": r["report_date"], "doc": r["doc_id"],
                       "pages": pages, "img": img, "sec": round(r["seconds"] or 0), "data": data["data"],
                       "flags": data["meta"].get("sanity_flags", [])})

    # --- figures
    figs = rows(con, "SELECT * FROM figures ORDER BY doc_id, page")
    for f in figs:
        f["img"] = png_file_uri(ROOT / f["image_path"])
        f["series"] = json.loads(f["series"] or "[]")
        f["key_values"] = json.loads(f["key_values"] or "[]")
    kinds = sorted({f["kind"] or "other" for f in figs})

    # --- charts
    charts = []
    for well in sorted({r["well"] for r in reports}):
        pts = [(r["report_number"], r["md_ft"], r["report_date"]) for r in reports
               if r["well"] == well and r["md_ft"] and r["report_number"]]
        pts = sorted({p[0]: p for p in pts}.values())
        if len(pts) >= 5:
            charts.append((well, len(pts), depth_chart(pts, well)))

    events = rows(con, "SELECT * FROM events ORDER BY well, report_date")
    npt = rows(con, "SELECT well, SUM(hours) h, COUNT(*) n, npt_source s FROM operations WHERE is_npt=1 GROUP BY well, s")
    agent = json.loads((OUTD / "agent_demo.json").read_text(encoding="utf-8")) if (OUTD / "agent_demo.json").exists() else []
    summary = json.loads((ROOT / "data/processed/D08/well_summary.json").read_text(encoding="utf-8")) \
        if (ROOT / "data/processed/D08/well_summary.json").exists() else {}

    doc_rows = []
    for d in DOCS:
        npages = len(fitz.open(RAW / d["path"]))
        used = len(d.get("pages", [])) or npages
        nrep = sum(1 for r in reports if r["doc_id"] == d["id"])
        nfig = sum(1 for f in figs if f["doc_id"] == d["id"])
        out = ", ".join(x for x in [f"{nrep} daily report{'s' * (nrep != 1)}" if nrep else "",
                                    f"{nfig} figures" if nfig else "",
                                    "well summary" if d["kind"] == "long_report" else ""] if x)
        doc_rows.append(f"<tr><td class='mono'>{d['id']}</td><td>{esc(d['well'])}</td><td>{DOC_LABEL[d['kind']]}</td>"
                        f"<td class='num'>{used if used == npages else f'{used} of {npages}'}</td><td>{esc(out)}</td></tr>")

    tpl = (Path(__file__).parent / "showcase_template.html").read_text(encoding="utf-8")
    page = (tpl
            .replace("/*DATA*/", "const REPORTS=" + json.dumps(viewer) + ";\nconst FIGS=" + json.dumps(figs)
                     + ";\nconst KINDS=" + json.dumps(kinds) + ";\nconst AGENT=" + json.dumps(agent, default=str)
                     + ";\nconst SUMMARY=" + json.dumps(summary) + ";")
            .replace("{{N_REPORTS}}", str(n["reports"]))
            .replace("{{N_FIGS}}", str(n["figures"]))
            .replace("{{N_ROWS}}", f"{sum(v for k, v in n.items() if k not in ('reports', 'figures', 'checks', 'trust')):,}")
            .replace("{{N_OPS}}", f"{n['operations']:,}")
            .replace("{{N_BHA}}", str(n["bha"]))
            .replace("{{N_SURV}}", str(n["surveys"]))
            .replace("{{N_BITS}}", str(n["bits"]))
            .replace("{{N_MUD}}", str(n["mud"]))
            .replace("{{N_CASING}}", str(n["casing"]))
            .replace("{{N_EVENTS}}", str(n["events"]))
            .replace("{{HOURS}}", f"{statistics.median(r['seconds'] for r in reports) / 60:.1f}")
            .replace("{{DOC_ROWS}}", "\n".join(doc_rows))
            .replace("{{DEPTH_CHARTS}}", "\n".join(
                f'<figure class="chartbox"><figcaption>{esc(w)} · {k} reports</figcaption>{svg}</figure>'
                for w, k, svg in charts))
            .replace("{{ACTIVITY}}", activity_bars(con))
            .replace("{{EVENTS}}", "\n".join(
                f"<tr><td>{esc(e['well'])}</td><td class='mono'>{esc(e['report_date'])}</td>"
                f"<td><span class='chip'>{esc(e['event_type']).replace('_', ' ')}</span></td>"
                f"<td class='num'>{'' if e['depth_ft'] is None else f'{e['depth_ft']:,.0f}'}</td>"
                f"<td>{esc(e['description'])}</td></tr>" for e in events) or
                "<tr><td colspan='5' class='muted'>No problem events found.</td></tr>")
            .replace("{{NPT}}", ", ".join(f"{esc(r['well'])}: {r['h']:.1f} h in {r['n']} rows ({'operator marks' if r['s'] == 'column' else 'model estimate'})" for r in npt) or "none flagged"))
    dst = OUTD / "showcase.html"
    dst.write_text(page, encoding="utf-8")
    return dst


if __name__ == "__main__":
    p = build()
    print(p, f"{p.stat().st_size / 1e6:.1f} MB")
