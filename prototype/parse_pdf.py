"""PDF -> layout-preserving text, report segments and figure crops (PyMuPDF only)."""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?)\s*\d+", re.I)
REPORT_START_RE = re.compile(r"Page:?\s*1\s+of\s+\d+|RPT NUM", re.I)
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
ELAPSED_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")
NPT_HEADERS = {"Non-Prod", "NPT"}
NPT_MARKS = {"X", "x"}


def page_text(page: fitz.Page, clip: fitz.Rect | None = None, cols: int = 170) -> str:
    """Monospace layout: each word is placed on a character grid by its x-position, so multi-row
    table headers stay vertically aligned with their values (form-style reports defeat table detectors)."""
    area = clip or page.rect
    cw = area.width / cols
    rows: dict[int, list] = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words", clip=clip, sort=True):
        rows.setdefault(round((y0 + y1) / 6), []).append((x0, w))
    out = []
    for k in sorted(rows):
        line = ""
        for x0, w in sorted(rows[k]):
            c = int((x0 - area.x0) / cw)
            if line and c <= len(line):
                c = len(line) + 1
            line = line.ljust(c) + w
        out.append(line.rstrip())
    return "\n".join(out)


def split_reports(doc: fitz.Document, pages: list[int]) -> list[list[int]]:
    """Group pages into individual daily reports (compilation PDFs hold many)."""
    groups: list[list[int]] = []
    for i in pages:
        head = doc[i].get_text()
        if not groups or REPORT_START_RE.search(head):
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def casing_rows(pdf: fitz.Document, pages: list[int]) -> list[dict]:
    """Casing/Tubular table, by word position: each value falls under the header word to its left."""
    names = ["size", "top_md", "top_tvd", "bottom_md", "bottom_tvd", "section", "hole", "weight", "grade", "mud"]
    for i in pages:
        words = pdf[i].get_text("words", sort=True)
        title = next((w for w in words if w[4] == "Casing/Tubular"), None)
        if not title:
            continue
        head = next((w for w in words if w[4] == "Type" and w[1] > title[1]), None)
        if not head:
            continue
        cols = sorted(w[0] for w in words if abs(w[1] - head[1]) < 4 and w[0] > head[0])
        if len(cols) != len(names):
            return []
        lines = defaultdict(list)
        for w in words:
            if w[1] > head[3]:
                lines[round(w[1])].append(w)
        rows = []
        for y in sorted(lines):
            row = {}
            for w in lines[y]:
                k = bisect_right(cols, w[0]) - 1
                if k >= 0:
                    row[names[k]] = w[4]
            if not re.fullmatch(r"\d+\.\d{3}", row.get("size", "")):
                break
            rows.append(row)
        return rows
    return []


MUD_COLS = {"Dens.": "density_ppg", "Vis": "funnel_viscosity_s", "PV": "pv_cp", "YP": "yp_lbf_100ft2",
            "pH/ES": "ph", "Cl": "chlorides_mg_l"}
NUM_RE = re.compile(r"^-?[\d,]*\.?\d+$")


def mud_rows(pdf: fitz.Document, pages: list[int]) -> list[dict]:
    """Mud checks by word position. Under the header line (Dens. Vis PV YP ... Cl) each check is an
    "at Depth N ft" line followed by a line of values, right-aligned under their headers."""
    for i in pages:
        words = pdf[i].get_text("words", sort=True)
        dens = next((w for w in words if w[4] == "Dens."), None)
        if not dens:
            continue
        head = [w for w in words if abs(w[1] - dens[1]) < 3]
        if not {"Vis", "PV", "YP"} <= {w[4] for w in head}:
            continue
        rights = [(w[2], w[4]) for w in head]
        lines: dict[int, list] = defaultdict(list)
        for w in words:
            if w[1] > dens[3]:
                lines[round(w[1])].append(w)
        rows, depth = [], None
        for y in sorted(lines):
            ln = sorted(lines[y], key=lambda w: w[0])
            texts = [w[4] for w in ln]
            if "Depth" in texts:
                k = texts.index("Depth")
                depth = texts[k + 1] if k + 1 < len(texts) and NUM_RE.match(texts[k + 1]) else None
                continue
            if depth is None or not NUM_RE.match(texts[0]):
                break
            row = {"depth_ft": depth}
            for w in ln:
                name = min(rights, key=lambda r: abs(r[0] - w[2]))[1]
                if name in MUD_COLS and NUM_RE.match(w[4]):
                    row[MUD_COLS[name]] = w[4]
            rows.append(row)
            depth = None
        return rows
    return []


BIT_COLS = {"No": "bit_number", "Make": "manufacturer", "Model": "bit_type", "Diam": "size_in", "In": "depth_in_ft",
            "Avg": "rop_ft_hr", "WOB": "wob_klbf", "RPM": "rpm"}


def bit_rows(pdf: fitz.Document, pages: list[int]) -> list[dict]:
    """Bit/BHA/Workstring table by word position. Each bit is a row under the header (No Run Make Model Diam In ...
    Avg Max WOB RPM ...); a value belongs to the header whose right edge is nearest. The "Jets:" line below a bit
    holds its depth out ("Out: N"). The table ends at the BHA line."""
    for i in pages:
        words = pdf[i].get_text("words", sort=True)
        title = next((w for w in words if w[4].startswith("Bit/BHA")), None)
        if not title:
            continue
        no = next((w for w in words if w[4] == "No" and w[1] > title[1] and any(
            v[4] == "Make" and abs(v[1] - w[1]) < 3 for v in words)), None)
        if not no:
            continue
        rights = [(w[2], w[4]) for w in words if abs(w[1] - no[1]) < 3]
        lines: dict[int, list] = defaultdict(list)
        for w in words:
            if w[1] > no[3]:
                lines[round(w[1])].append(w)
        rows = []
        for y in sorted(lines):
            ln = sorted(lines[y], key=lambda w: w[0])
            texts = [w[4] for w in ln]
            if texts[0] == "Jets:" and rows:
                if "Out:" in texts:
                    k = texts.index("Out:")
                    if k + 1 < len(texts) and NUM_RE.match(texts[k + 1]):
                        rows[-1]["depth_out_ft"] = texts[k + 1]
                continue
            if not NUM_RE.match(texts[0]) or ln[0][0] > no[2] + 5:
                break
            row = {}
            for w in ln:
                name = min(rights, key=lambda r: abs(r[0] - w[2]))[1]
                if name in BIT_COLS and BIT_COLS[name] not in row:
                    row[BIT_COLS[name]] = w[4]
            rows.append(row)
        return rows
    return []


def read_npt_column(doc: fitz.Document, pages: list[int]) -> list[dict] | None:
    """Time-log rows of one report with the operator's Non-Prod/NPT mark, read from word positions.
    A row is a From, To, Elapsed triple on one line. Returns None if the report has no such column."""
    col, rows = None, []
    for i in pages:
        words = doc[i].get_text("words", sort=True)
        head = next((w for w in words if w[4] in NPT_HEADERS and any(
            v[4].lower() == "from" and abs(v[1] - w[1]) < 3 for v in words)), None)
        if head:
            col = (head[0] - 2, head[2] + 2)
        if not col:
            continue
        top = head[3] if head else 0
        found = []
        for a, b, c in zip(words, words[1:], words[2:]):
            if (a[1] > top and abs(a[1] - b[1]) < 3 and abs(a[1] - c[1]) < 3 and TIME_RE.match(a[4])
                    and TIME_RE.match(b[4]) and ELAPSED_RE.match(c[4])):
                found.append({"page": i + 1, "y": a[1], "start": a[4], "end": b[4], "hours": float(c[4]), "marked": False})
        for w in words:
            if found and w[4] in NPT_MARKS and col[0] <= w[0] and w[2] <= col[1]:
                near = min(found, key=lambda r: abs(r["y"] - w[1]))
                if abs(near["y"] - w[1]) < 15:
                    near["marked"] = True
        rows += found
    return rows if col else None


def read_activity_codes(doc: fitz.Document, pages: list[int]) -> list[dict] | None:
    """Time-log rows of one report with the operator's activity code (the "Code" column), read from word positions.
    A row is a From, To, Elapsed triple on one line. Returns None if the report has no such column."""
    col, rows = None, []
    for i in pages:
        words = doc[i].get_text("words", sort=True)
        head = next((w for w in words if w[4] == "Code" and any(
            v[4].lower() == "from" and abs(v[1] - w[1]) < 3 for v in words)), None)
        if head:
            col = (head[0] - 15, head[2] + 15)
        if not col:
            continue
        top = head[3] if head else 0
        for a, b, c in zip(words, words[1:], words[2:]):
            if (a[1] > top and abs(a[1] - b[1]) < 3 and abs(a[1] - c[1]) < 3 and TIME_RE.match(a[4])
                    and TIME_RE.match(b[4]) and ELAPSED_RE.match(c[4])):
                code = " ".join(w[4] for w in words if abs(w[1] - a[1]) < 3 and col[0] <= w[0] <= col[1])
                rows.append({"page": i + 1, "start": a[4], "end": b[4], "code": code})
    return rows if col else None


@dataclass
class Figure:
    doc_id: str
    page: int
    bbox: tuple
    source: str              # raster | vector | full_page
    caption: str | None = None
    text_hint: str = ""      # text-layer words inside the figure (axis labels, values)
    image_path: str = ""
    extra: dict = field(default_factory=dict)


def _caption_near(page: fitz.Page, rect: fitz.Rect) -> str | None:
    best = None
    for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
        if CAPTION_RE.match(txt):
            d = min(abs(y0 - rect.y1), abs(rect.y0 - y1))
            if d < 60 and (best is None or d < best[0]):
                best = (d, " ".join(txt.split()))
    return best[1] if best else None


def find_figures(doc: fitz.Document, doc_id: str, pages: list[int], out_dir: Path,
                 full_page: bool = False, min_frac: float = 0.06, dpi: int = 150) -> list[Figure]:
    """Detect figures: big raster images + large vector-drawing clusters that are not tables.

    Images repeated on many pages (logos, letterheads) are ignored.
    full_page=True treats each page as one figure (plot-only documents such as mud logs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xref_count = Counter(info["xref"] for i in pages for info in doc[i].get_image_info(xrefs=True))
    repeated = {x for x, n in xref_count.items() if len(pages) > 3 and n > 0.3 * len(pages)}

    figs: list[Figure] = []
    for i in pages:
        page = doc[i]
        area = page.rect.width * page.rect.height
        cands: list[tuple[fitz.Rect, str]] = []
        if full_page:
            cands.append((page.rect, "full_page"))
        else:
            for info in page.get_image_info(xrefs=True):
                r = fitz.Rect(info["bbox"]) & page.rect
                if info["xref"] not in repeated and r.width * r.height > min_frac * area:
                    cands.append((r, "raster"))
            tables = [fitz.Rect(t.bbox) for t in page.find_tables().tables]
            for r in page.cluster_drawings():
                if r.width * r.height < min_frac * area:
                    continue
                if any((r & t).get_area() > 0.5 * r.get_area() for t in tables):
                    continue  # ruled table, not a figure
                if any((r & c).get_area() > 0.5 * r.get_area() for c, _ in cands):
                    continue
                cands.append((r, "vector"))
        for n, (r, src) in enumerate(cands):
            path = out_dir / f"{doc_id}_p{i + 1}_{n}.png"
            page.get_pixmap(clip=r, dpi=dpi).save(path)
            figs.append(Figure(doc_id, i + 1, tuple(round(v, 1) for v in r), src,
                               caption=_caption_near(page, r),
                               text_hint=page_text(page, clip=r)[:3000],
                               image_path=str(path)))
    return figs
