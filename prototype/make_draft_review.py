"""Review page for the question-set answers drafted by Claude from the End-of-Well Report (checked_by "draft").

For each draft question it renders the cited D08 pages with the answer's numbers (and a few key words) highlighted,
plus a link that opens the PDF at that page. Hamed marks each answer right / wrong / drop with a correction; the page
copies the verdicts as JSON to paste back to Claude. The PDF is not changed.

    python make_draft_review.py      # -> outputs/draft_review/index.html (git-ignored; needs data/raw/)
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "forge" / "56-32_EOWR.pdf"
QUESTIONS = ROOT / "data" / "questions" / "questions.json"
OUT = ROOT / "outputs" / "draft_review"
FIG_PAGE = {4: 17, 5: 19, 21: 46}   # figures cited without a page number
KEYWORDS = {"Q16": ["granite"], "Q17": ["60%", "180%", "MSE"], "Q18": ["silicate"], "Q19": ["core"],
            "Q38": ["hammer"], "Q44": ["losses", "lost circulation"], "Q45": ["silicate", "displace"],
            "Q48": ["oscillator", "shock sub", "new motor", "ream"], "Q36": ["days"], "Q22": ["S/No"]}
DPI = 110


def pages_of(source: str) -> list[int]:
    pages = []
    for m in re.finditer(r"Figure (\d+)(?: \(p(\d+)\))?|p(\d+)(?:-p?(\d+))?", source):
        if m.group(1):
            pages.append(int(m.group(2)) if m.group(2) else FIG_PAGE.get(int(m.group(1)), 0))
        else:
            a = int(m.group(3))
            pages += range(a, int(m.group(4) or a) + 1)
    return list(dict.fromkeys(p for p in pages if p))


def terms(q: dict) -> list[str]:
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", str(q["answer"]))
    out = []
    for n in nums:
        if len(n.replace(",", "")) < 2:
            continue
        out.append(n)
        if "," not in n and n.isdigit() and len(n) >= 4:
            out.append(f"{int(n):,}")  # 9145 is printed as 9,145
    return list(dict.fromkeys(out + KEYWORDS.get(q["id"], [])))


def main() -> None:
    qs = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["checked_by"] == "draft"]
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    pdf_url = PDF.resolve().as_uri()
    cards = []
    for q in qs:
        ts = terms(q)
        imgs = []
        for p in pages_of(q["source"]):
            page = doc[p - 1]
            found = []
            for t in ts:
                rects = page.search_for(t)
                if rects:
                    found.append(t)
                    for r in rects:
                        page.add_highlight_annot(r)
            name = f"{q['id']}_p{p}.png"
            page.get_pixmap(dpi=DPI, annots=True).save(OUT / "img" / name)
            note = f"highlighted: {', '.join(found)}" if found else "no answer text found on this page (image or other wording)"
            imgs.append(f'<figure><figcaption><a href="{pdf_url}#page={p}" target="_blank">page {p} - open in PDF</a>'
                        f' <span>{html.escape(note)}</span></figcaption><img loading="lazy" src="img/{name}" alt="D08 page {p}"></figure>')
        e = html.escape
        cards.append(f"""<section class="card" id="{q['id']}" data-id="{q['id']}">
<h2>{q['id']} <small>level {q['level']} - {e(q['category'])}</small></h2>
<p class="q">{e(q['question'])}</p>
<p class="a"><b>Draft answer:</b> {e(str(q['answer']))}{f" <i>(tolerance {q['tolerance']})</i>" if q['tolerance'] not in (None, 0) else ""}</p>
<p class="s">Source: {e(q['source'])}</p>
<div class="verdict"><label><input type="radio" name="v_{q['id']}" value="right"> right</label>
<label><input type="radio" name="v_{q['id']}" value="wrong"> wrong, correct answer:</label>
<label><input type="radio" name="v_{q['id']}" value="drop"> drop question</label>
<input type="text" class="fix" placeholder="correction or note"></div>
<div class="pages">{''.join(imgs)}</div></section>""")
    doc.close()  # highlights were only for rendering; the PDF file is untouched
    nav = " ".join(f'<a href="#{q["id"]}">{q["id"]}</a>' for q in qs)
    (OUT / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Draft Answer Review</title>
<style>
:root{{--bg:#f3f5f6;--sheet:#fff;--ink:#15212b;--muted:#5a6873;--rule:#d5dde2;--accent:#0d7a87}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f161c;--sheet:#151f27;--ink:#e3eaee;--muted:#93a2ad;--rule:#2b3a45;--accent:#3cc2cf}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--sheet);border-bottom:1px solid var(--rule);padding:10px 16px;z-index:2}}
header h1{{font-size:18px;margin:0 0 4px}} header nav a{{margin-right:8px;color:var(--accent)}}
#copy{{float:right;padding:6px 12px;font:inherit;cursor:pointer}} #status{{color:var(--muted);font-size:13px;margin-left:8px}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
.card{{background:var(--sheet);border:1px solid var(--rule);border-radius:6px;padding:14px 18px;margin-bottom:18px;scroll-margin-top:90px}}
.card h2{{margin:0;font-size:18px}} .card small{{color:var(--muted);font-weight:400}}
.q{{font-weight:600;margin:6px 0}} .s{{color:var(--muted);font-size:13px;margin:4px 0}}
.verdict{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:8px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}}
.fix{{flex:1 1 260px;font:inherit;padding:4px 8px}}
.pages{{display:grid;gap:12px;margin-top:10px}} figure{{margin:0}} figcaption{{font-size:13px;margin-bottom:4px}}
figcaption span{{color:var(--muted)}} figcaption a{{color:var(--accent)}}
img{{max-width:100%;border:1px solid var(--rule);background:#fff}}
.done{{border-left:4px solid var(--accent)}}
</style></head><body>
<header><button id="copy">Copy verdicts</button><h1>Draft answers from the 56-32 End-of-Well Report ({len(qs)})</h1>
<nav>{nav}</nav><span id="status"></span></header>
<main>{''.join(cards)}</main>
<script>
const KEY = "draft_review_v1";
let saved = {{}}; try {{ saved = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const cards = [...document.querySelectorAll(".card")];
function collect() {{
  const out = {{}};
  for (const c of cards) {{
    const v = c.querySelector("input[type=radio]:checked"), fix = c.querySelector(".fix").value.trim();
    if (v || fix) out[c.dataset.id] = {{ verdict: v ? v.value : null, note: fix }};
    c.classList.toggle("done", !!v);
  }}
  try {{ localStorage.setItem(KEY, JSON.stringify(out)); }} catch (e) {{}}
  document.getElementById("status").textContent = Object.values(out).filter(x => x.verdict).length + " of {len(qs)} marked";
  return out;
}}
for (const c of cards) {{
  const s = saved[c.dataset.id];
  if (s) {{ if (s.verdict) c.querySelector(`input[value=${{s.verdict}}]`).checked = true; c.querySelector(".fix").value = s.note || ""; }}
  c.addEventListener("input", collect);
}}
collect();
document.getElementById("copy").onclick = async () => {{
  const txt = JSON.stringify(collect(), null, 1);
  try {{ await navigator.clipboard.writeText(txt); document.getElementById("status").textContent += " - copied"; }}
  catch (e) {{ prompt("Copy this:", txt); }}
}};
</script></body></html>""", encoding="utf-8")
    print(f"{len(qs)} questions -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
