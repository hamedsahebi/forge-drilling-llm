# Hand grading the agent answers (runbook F2)

Who grades: Hamed (drilling engineer). Claude does not grade the hand-graded answers; it only runs the scripts.

## Steps
**Easiest:** grade on the grading page, https://claude.ai/artifact/4GdZS54UUqB9P9Z7A2NeAe (keys 1 right, 2 partial, 3 wrong, J/K to move). It shows the gold answer, the rule for the question type and the agent's tool calls. Grades save on the page; ask Claude to read them back into `hand_grades.json`. The steps below are the file-based way and give the same result.

1. `git pull`, then `cd prototype && python grade_questions.py`. This writes `outputs/agent_questions/grading_sheet.md`, which lists every answer that still needs a hand grade: the question, the gold answer and its source, the agent's answer, and a hint.
2. Grade each entry in `data/questions/hand_grades.json` (create the file if it does not exist). One line per answer, keyed exactly as the sheet heading:
   ```json
   {
    "base|Q02": "right",
    "base_verify|Q45": "partial",
    "base_verify|Q45|first": "wrong"
   }
   ```
   Values: `right` (1 point), `partial` (0.5), `wrong` (0).
3. Run `python grade_questions.py` again. Graded entries leave the sheet; scores go to `outputs/agent_questions/scores.json`. Commit `hand_grades.json` and push.

You can grade in several sessions; only the ungraded entries stay on the sheet.

## What is on the sheet
- **Settings:** `base` (agent as tagged `agent-v1-before-verify`), `base_verify` (with the answer checker), `base_think` and `base_verify_think` (Qwen3 thinking mode). Grade each setting on its own; do not look for the "best" one.
- **`|first` keys** (verify runs only): the answer **before** the checker, shown when the checker changed it. Grade it as if it were the final answer. This measures what the checker adds.
- **Number questions in levels 1-2** are graded automatically (right if any number in the answer is within the tolerance) and are not on the sheet. If an automatic grade is plainly wrong (for example, the right number appears only as part of a wrong statement), add a hand grade for that key; a hand grade overrides it. The automatic grades per question are in `scores.json` under `per_question`.

## Grading rules
Grade the content against the gold answer. Do not grade wording, length or style.

**right**: every part the question asks for is there and correct.
- Numbers within the tolerance given on the sheet, or within normal reading precision when no tolerance is given (e.g. 8,509 ft vs 8,509.5 ft is right).
- Units may differ if the value is the same (e.g. 10.625 in. = 10-5/8 in.).
- Extra correct information is fine. Extra **wrong** information about what was asked makes it partial or wrong.

**partial**: some of the parts asked for are right and some are missing or wrong.
- E.g. "size and make, and from what depth to what depth": size and make right but the depths wrong or missing.
- The right value, but stated as fact when the data marks it as doubtful (see below).
- Level 3 rubric questions: about half the rubric points (see below).

**wrong**: the main value is wrong or missing, the agent says "not available" when the answer exists, it invents a value, or it answers a different question.

### Special question types
- **Not available (Q50-Q52):** right only if the agent says the information is not in the data. Giving any value is wrong, even a plausible one. "Not available" plus a correct reason is right; the reason itself is not required.
- **False premise (Q43, Q44):** right only if the agent says the premise is not supported (Q43: no kick is recorded that day; Q44: no losses while drilling the 8-3/4 in. hole, the problems came during cementing). Explaining an event that did not happen is wrong.
- **Axis limit (Q49):** the true mud volume is above the 200 bbl chart scale. Right if the agent says the chart cannot show the value (pinned at the top of the scale / at_max) and does not give 200 bbl as the reading. Giving 200 bbl, or any number from the chart, as the real value is wrong.
- **Unverified values:** if the agent gives a figure or chart value that the database marks as an unverified vision reading, or a value whose trust verdict is `fail` or `conflict`, and states it as fact, give at most `partial`, even when the number happens to be right.
- **Level 3 judge questions (Q45-Q48):** the gold answer lists the points. Right = all points (or "any three" when the rubric says so); partial = about half; wrong = none, or the points are mostly invented. A sensible point that is not in the rubric does not count for or against.
- **Cross-source and plan vs actual (Q40-Q42):** right needs both numbers and the conclusion (Q40 agree at 3,494 ft; Q41 yes, 8,500 ft reached 16 July 20:00; Q42 about 400 ft deeper than planned). The conclusion with wrong numbers is wrong.

### Citations
Not graded for points. If an answer cites a report or figure that does not exist or does not hold the value, grade it at most `partial`.

## When unsure
Grade `partial` and write the question id down; we discuss them before the numbers go into the paper. Do not change gold answers while grading. If a gold answer turns out to be wrong, note it; it is fixed in `make_questions.py` and logged, and every setting is regraded.

## Note on the files
`grading_sheet.md` and `scores.json` are regenerated by `grade_questions.py`. If git reports a conflict in either one, take any version and rerun the script. Only `hand_grades.json` holds your work.
