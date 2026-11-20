"""Grade the agent's answers to the question set (runbook F2).

Reads outputs/agent_questions/<setting>.json (written by `agent.py --questions [--verify] [--think]`).

Grades, scored right = 1, partial = 0.5, wrong = 0:
  number         automatic: right if a number in the answer is within `tolerance` of the gold value
                 (lenient: any number in the answer counts; a hand grade overrides it)
  text, level 3, not_available
                 by hand, in data/questions/hand_grades.json: {"<setting>|<id>": "right" | "partial" | "wrong"};
                 for --verify runs the answer before the checker is graded as "<setting>|<id>|first"
Answers with checked_by "draft" (read by Claude from the End-of-Well Report, not yet checked by Hamed) are scored
but kept out of the headline until their checked_by is changed in make_questions.py.

    python grade_questions.py            # scores + outputs/agent_questions/grading_sheet.md for what is still ungraded
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "agent_questions"
HAND = ROOT / "data" / "questions" / "hand_grades.json"
QUESTIONS = ROOT / "data" / "questions" / "questions.json"
GOLD_KEYS = ("level", "category", "question", "answer", "type", "tolerance", "source", "checked_by")
SHEET = RUNS / "grading_sheet.md"
SCORES = RUNS / "scores.json"
POINTS = {"right": 1.0, "partial": 0.5, "wrong": 0.0}
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
NA_WORDS = re.compile(r"not available|no data|not (?:recorded|reported|found|in the|present)|no record|does not (?:contain|"
                      r"include|exist)|is not (?:given|listed|stated)|no (?:serial|mud check|chloride)|unavailable|"
                      r"cannot be (?:found|determined)", re.I)


def numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def auto_number(answer: str, gold: float, tol) -> str:
    tol = float(tol or 0)
    return "right" if any(abs(x - float(gold)) <= tol + 1e-6 for x in numbers(answer)) else "wrong"


def gold_hint(answer: str, gold) -> str:
    """For hand grading: how many numbers of the gold answer appear in the agent's answer."""
    g = {x for x in numbers(str(gold)) if not 1900 < x < 2100}
    if not g:
        return ""
    got = numbers(answer)
    hit = sum(any(abs(x - y) <= max(0.01, 0.005 * abs(y)) for x in got) for y in g)
    return f"gold numbers found {hit}/{len(g)}"


def grade(r: dict, text: str, hand: dict, key: str) -> tuple[str | None, str]:
    """(grade or None if it still needs a hand grade, how it was graded)."""
    if key in hand:
        return hand[key], "hand"
    if r["type"] == "number" and r["level"] < 3:
        return auto_number(text, r["answer"], r["tolerance"]), "auto"
    return None, "hand"


def summarise(items: list[tuple[str, float | None]]) -> dict:
    graded = [p for _, p in items if p is not None]
    return {"n": len(items), "graded": len(graded),
            "score": round(sum(graded) / len(graded), 3) if graded else None}


def load_runs() -> dict[str, list[dict]]:
    # gold comes from the question file: in the run files the agent's `answer` has replaced the gold one
    gold = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    return {p.stem: [{**r, **{k: gold[r["id"]][k] for k in GOLD_KEYS}, "agent_answer": r["answer"]}
                     for r in json.loads(p.read_text(encoding="utf-8")) if r["id"] in gold]
            for p in sorted(RUNS.glob("*.json")) if p.stem not in ("scores", "failure_modes")}


def main() -> None:
    hand = json.loads(HAND.read_text(encoding="utf-8")) if HAND.exists() else {}
    runs = load_runs()
    if not runs:
        print("no runs in", RUNS)
        return
    scores, todo = {}, []
    for setting, rows in runs.items():
        groups = defaultdict(list)
        per_q = {}
        for r in rows:
            variants = [("", r["agent_answer"])]
            if r.get("checks"):  # verify run: also grade the answer before the checker
                variants.append(("|first", r.get("first_answer") or ""))
            for suffix, text in variants:
                key = f"{setting}|{r['id']}{suffix}"
                same = bool(suffix) and text == r["agent_answer"]  # checker changed nothing: same grade as the final
                if same:
                    g, how = grade(r, r["agent_answer"], hand, f"{setting}|{r['id']}")
                else:
                    g, how = grade(r, text, hand, key)
                p = POINTS.get(g) if g else None
                if g is None and not same:
                    todo.append((setting, r, suffix, text))
                which = "first" if suffix else "final"
                per_q.setdefault(r["id"], {})[which] = g
                headline = r["checked_by"] != "draft"
                for name in ("all", f"level{r['level']}", f"by_source.{r['checked_by']}", f"category.{r['category']}"):
                    groups[f"{which}.{name}"].append((r["id"], p))
                if headline:
                    groups[f"{which}.headline"].append((r["id"], p))
                    groups[f"{which}.headline.level{r['level']}"].append((r["id"], p))
        tools = [len(r.get("trace", [])) for r in rows]
        scores[setting] = {
            "questions": len(rows),
            "mean_seconds": round(sum(r.get("seconds", 0) for r in rows) / max(1, len(rows)), 1),
            "mean_tool_calls": round(sum(tools) / max(1, len(tools)), 2),
            "no_tool_call": sum(t == 0 for t in tools),
            "step_limit": sum(r["agent_answer"] == "(step limit reached)" for r in rows),
            "checker_changed_answer": sum(bool(r.get("checks")) and r.get("first_answer") != r["agent_answer"] for r in rows),
            "groups": {k: summarise(v) for k, v in sorted(groups.items())},
            "per_question": per_q,
        }
    SCORES.write_text(json.dumps(scores, indent=1), encoding="utf-8")

    lines = ["# Agent answers still to grade",
             f"Write grades into `{HAND.relative_to(ROOT).as_posix()}` as `\"<key>\": \"right\" | \"partial\" | \"wrong\"`, "
             "then run `python grade_questions.py` again.", ""]
    for setting, r, suffix, text in todo:
        key = f"{setting}|{r['id']}{suffix}"
        hint = gold_hint(text, r["answer"])
        na = " · says not available" if r["type"] == "not_available" and NA_WORDS.search(text or "") else ""
        lines += [f"## `{key}`  (level {r['level']}, {r['category']}, {r['checked_by']})",
                  f"**Q:** {r['question']}", f"**Gold:** {r['answer']}  _(source: {r['source']})_",
                  f"**Agent{' (before checker)' if suffix else ''}:** {(text or '').strip()}",
                  f"_{', '.join(x for x in (hint, na.strip(' ·')) if x) or 'no hint'}_", ""]
    SHEET.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'setting':<20}{'graded':>8}{'all':>7}{'headline':>10}{'L1':>7}{'L2':>7}{'L3':>7}{'first':>7}  tools  s/q")
    for s, v in scores.items():
        g = v["groups"]
        f = lambda k: "-" if not g.get(k) or g[k]["score"] is None else f"{g[k]['score']:.2f}"
        print(f"{s:<20}{g['final.all']['graded']:>4}/{g['final.all']['n']:<3}{f('final.all'):>7}{f('final.headline'):>10}"
              f"{f('final.level1'):>7}{f('final.level2'):>7}{f('final.level3'):>7}{f('first.all'):>7}"
              f"  {v['mean_tool_calls']:>5} {v['mean_seconds']:>5}")
    print(f"{len(todo)} answers to grade by hand -> {SHEET}")
    print("->", SCORES)


if __name__ == "__main__":
    main()
