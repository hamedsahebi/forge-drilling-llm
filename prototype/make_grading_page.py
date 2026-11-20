"""Build the hand-grading page (outputs/grading_page.html) for the agent's answers to the question set (runbook F2).

Lists every answer that needs a hand grade (and the automatic number grades, so they can be overridden), with the
gold answer, the grading rule for its question type and the agent's tool calls. The page saves grades to the
artifact's database (collection `grades`, doc id = key with "|" written as "~"); read them back into
data/questions/hand_grades.json, then run grade_questions.py.
    python make_grading_page.py
"""
from __future__ import annotations

import json
from pathlib import Path

from grade_questions import HAND, NA_WORDS, auto_number, gold_hint, load_runs

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("grading_page_template.html")
OUT = ROOT / "outputs" / "grading_page.html"
CUT = 700  # characters of each tool result shown

# Short form of the special rules in docs/hand_grading.md, by question id
RULES = {
    **{q: "Not available: right only if the agent says it is not in the data. Any value, even a plausible one, is wrong."
       for q in ("Q50", "Q51", "Q52")},
    "Q43": "False premise: right only if the agent says no kick is recorded that day. Explaining a kick is wrong.",
    "Q44": "False premise: right only if the agent says there were no losses while drilling the 8-3/4 in. hole "
           "(the problems came during cementing).",
    "Q49": "Axis limit: right if the agent says the chart cannot show the value (pinned at the 200 bbl top / at_max). "
           "200 bbl or any chart number given as the real value is wrong.",
    **{q: "Level 3 judge: right = all rubric points (or 'any three' if the rubric says so), partial = about half, "
          "wrong = none or mostly invented. Points outside the rubric do not count." for q in ("Q45", "Q46", "Q47", "Q48")},
    "Q40": "Cross-source: needs both numbers and the conclusion (agree at 3,494 ft). Conclusion with wrong numbers is wrong.",
    "Q41": "Plan vs actual: needs the numbers and the conclusion (yes, 8,500 ft reached 16 July 20:00).",
    "Q42": "Plan vs actual: needs the numbers and the conclusion (about 400 ft deeper than planned).",
}
GENERAL = ("Grade content, not wording. Right: every part asked for is there and correct. Partial: some parts right, "
           "some missing or wrong; or a right value stated as fact when it is an unverified vision reading or a "
           "fail/conflict trust value; or a citation that does not hold the value. Wrong: main value wrong or missing, "
           "'not available' when it exists, invented, or a different question answered.")


def short(v, n: int = CUT) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f" ... ({len(s) - n} more characters)"


def main() -> None:
    hand = json.loads(HAND.read_text(encoding="utf-8")) if HAND.exists() else {}
    items = []
    for setting, rows in load_runs().items():
        for r in rows:
            variants = [("", r["agent_answer"])]
            if r.get("checks") and (r.get("first_answer") or "") != r["agent_answer"]:
                variants.append(("|first", r.get("first_answer") or ""))
            for suffix, text in variants:
                key = f"{setting}|{r['id']}{suffix}"
                auto = auto_number(text, r["answer"], r["tolerance"]) if r["type"] == "number" and r["level"] < 3 else None
                items.append({
                    "key": key, "setting": setting, "id": r["id"], "first": bool(suffix), "level": r["level"],
                    "category": r["category"], "type": r["type"], "checked_by": r["checked_by"],
                    "question": r["question"], "gold": str(r["answer"]), "source": r["source"],
                    "tolerance": r["tolerance"], "answer": (text or "").strip(), "auto": auto,
                    "file": hand.get(key), "hint": gold_hint(text, r["answer"]),
                    "says_na": bool(r["type"] == "not_available" and NA_WORDS.search(text or "")),
                    "rule": RULES.get(r["id"], ""),
                    "final": r["agent_answer"].strip() if suffix else None,
                    "trace": [{"tool": t.get("tool"), "args": short(t.get("args"), 400), "result": short(t.get("result"))}
                              for t in r.get("trace", [])],
                    "checks": short(r.get("checks"), 1500) if r.get("checks") else None,
                    "seconds": r.get("seconds"),
                })
    data = {"items": items, "general": GENERAL}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("{{DATA}}", blob), encoding="utf-8")
    todo = sum(i["auto"] is None and i["file"] is None for i in items)
    print(f"{len(items)} answers ({todo} still to grade by hand) -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
