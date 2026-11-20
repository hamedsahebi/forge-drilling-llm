"""Write the agent question set to data/questions/questions.json.

Levels, scored separately:
  1 lookup              - one value from one place
  2 compute or compare  - sums, differences, rates, ranking, a summary across rows or pages
  3 judge               - plan vs actual, consistency between sources, false premises, axis limits, "not available",
                          and recommendations; a drilling engineer grades these against `answer` as a rubric
Every answer comes from a source the extraction did not produce (`checked_by`):
  label   - the hand labels in data/labels/labels/ (blind ones marked)
  code    - computed here from the Pason 10 s sensor CSV (the agent sees only the chart readings) or from the operator's
            Non-Prod column, which is read by code from the PDF
  draft   - read by Claude from the End-of-Well Report text or figure image; Hamed checks it
  engineer - a draft that Hamed checked against the PDF pages (make_draft_review.py), 2026-11-11
No question uses the dev reports D01, D02 or D05.
    python make_questions.py
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENSOR = ROOT / "data" / "processed" / "sensor" / "78B-32_pason_10s_chart.csv.gz"
OUT = ROOT / "data" / "questions" / "questions.json"
L, E, S = "hand label", "End-of-Well Report D08", "Pason 10 s sensor CSV"
NP = "operator Non-Prod marks (D07, read by code)"


def load_sensor() -> list[dict]:
    with gzip.open(SENSOR, "rt", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def values(rows, var, a, b):
    return [x for r in rows if a <= r["time"] < b and (x := num(r[var])) is not None]


def at(rows, var, t):
    return next(num(r[var]) for r in rows if r["time"] >= t and num(r[var]) is not None)


def first_reach(rows, var, level):
    return next(r["time"] for r in rows if (num(r[var]) or 0) >= level)


def q(level, cat, question, answer, kind="number", tol=None, source="", check="label"):
    return {"level": level, "category": cat, "question": question, "answer": answer, "type": kind, "tolerance": tol,
            "source": source, "checked_by": check}


def questions(s: list[dict]) -> list[dict]:
    hd = values(s, "hole_depth_ft", "2021-07-11T00:00", "2021-07-11T06:00")
    hook = max((num(r["hook_load_klbs"]), r["time"]) for r in s
               if r["time"].startswith("2021-07-14") and num(r["hook_load_klbs"]) is not None)
    t8500 = first_reach(s, "hole_depth_ft", 8500)
    mean = lambda var, a, b: round(statistics.mean(values(s, var, a, b)), 1)
    return [
        # ---- level 1: lookup ----
        q(1, "table", "What is the shoe depth of the 7 in. production casing in well 78B-32?", 8509, tol=0,
          source=L + " D07_r029-r035 (D07_r026 printed 8,532 ft)"),
        q(1, "table", "What size and make was bit #17 on well 78B-32, and from what depth to what depth did it drill?",
          "5.75 in., REED, 8,555 ft to 9,500 ft", "text", source=L + " D07_r034 (blind)"),
        q(1, "table", "Which bit drilled from 3,651 ft to 5,761 ft on well 78B-32?", "bit #7, 10-5/8 in. NOV", "text",
          source=L + " D07_r011"),
        q(1, "table", "What were the measured depth and true vertical depth of well 78B-32 on 30 July 2021?",
          "MD 9,500 ft, TVD 9,497 ft", "text", source=L + " D07_r034 (blind)"),
        q(1, "table", "What size is the production casing in well 16A(78)-32, and where is its shoe?",
          "7 in., shoe at 10,787 ft", "text", source=L + " D03_r001"),
        q(1, "table", "Where is the 9-5/8 in. intermediate casing shoe in well 56-32 according to the daily report?", 3494,
          tol=0, source=L + " D06_r001"),
        q(1, "table", "What inclination and azimuth were surveyed at 9,500 ft MD on well 78B-32?", "3.91 deg, 235.5 deg",
          "text", source=L + " D07_r034 (blind)"),
        q(1, "table", "What mud weight was reported on well 78B-32 on 4 July 2021, and at what depth was it measured?",
          "8.7 ppg at 3,341 ft", "text", source=L + " D07_r008"),
        q(1, "timelog", "How many hours of non-productive time were there on well 78B-32 on 9 July 2021?", 24, tol=0,
          source=L + " D07_r013 (blind)"),
        q(1, "timelog", "How many NPT hours were there on well 78B-32 on 15 July 2021, and what caused them?",
          "8 h, waiting on a hydraulic hose (rig repair)", "text", source=L + " D07_r019 (blind)"),
        q(1, "timelog", "Why did operations stop between 15:00 and 16:00 on 18 July 2021 on well 78B-32?",
          "shut down for high winds and a thunderstorm", "text", source=L + " D07_r022 (blind)"),
        q(1, "timelog", "How many NPT hours were logged on well 56-32 on 6 February 2021?", 12, tol=0, source=L + " D04_r001"),
        q(1, "timelog", "What log was run on well 78B-32 on 30 July 2021, and what maximum bottom-hole temperature did it record?",
          "Triple-Combo, max BHT 416 F", "text", source=L + " D07_r034 (blind), time-log description"),
        q(1, "chart", "What was the rotary RPM on well 78B-32 at 15:30 on 13 July 2021?",
          at(s, "rpm", "2021-07-13T15:30"), tol=3, source=S, check="code"),
        q(1, "text", "To what total depth was well 56-32 drilled?", 9145, tol=0, source=E + " p8, Figure 5", check="engineer"),
        q(1, "text", "At about what depth did well 56-32 reach the top of the granite?", 3110, tol=0, source=E + " p8",
          check="engineer"),
        q(1, "text", "How much did ROP improve on well 56-32 compared with wells 16A-32 and 58-32, and why?",
          "more than 60% vs 16A-32 and 180% vs 58-32, from new PDC cutters and MSE", "text", source=E + " p8",
          check="engineer"),
        q(1, "text", "What lesson does the End-of-Well Report give about sodium silicate in the cement slurry?",
          "do not use it if no losses were seen while drilling, because it can make the slurry set early; cost avoided "
          "over $100K", "text", source=E + " p83", check="engineer"),
        q(1, "text", "Was core recovered on well 56-32?",
          "no: a sidewall core at 6,999 ft failed to break the hard formation", "text", source=E + " p62", check="engineer"),
        q(1, "figure", "According to the as-built schematic of well 56-32, what production casing was run and to what hole depth?",
          "5-1/2 in., 17 ppf, L-80, BTC; 8-3/4 in. hole to 9,133.5 ft", "text", source=E + " Figure 5 (p19)",
          check="engineer"),
        q(1, "figure", "Where is the float collar in the 56-32 as-built schematic, and how much cement was left at the bottom?",
          "8,997 ft; 135 ft of cement", "text", source=E + " Figure 5 (p19)", check="engineer"),
        q(1, "figure", "What was the serial number of the bit in BHA #5 on well 56-32, and what was the BHA length?",
          "S/No 5258442; 762.86 ft", "text", source=E + " Figure 21 (p46)", check="engineer"),
        q(1, "figure", "In the planned days-versus-depth plot for well 56-32, after how many days was the well planned to reach 9,000 ft?",
          33, tol=1, source=E + " Figure 4 (p17), read off the chart", check="engineer"),
        q(1, "figure", "At what depth was the 9-5/8 in. casing planned in the days-versus-depth plot for well 56-32?", 3100,
          tol=100, source=E + " Figure 4 (p17), read off the chart", check="engineer"),
        # ---- level 2: compute or compare ----
        q(2, "timelog", "How many hours of tripping were logged on well 78B-32 on 30 July 2021?", 3.75, tol=0,
          source=L + " D07_r034 (blind; the labeller classed laying down pipe as rig_up_down)"),
        q(2, "timelog", "How many NPT hours did the operator mark on well 78B-32 in total, and what was the main cause?",
          "84 h; most of it the SCR / drawworks traction-motor failure of 8-10 July (repair, then waiting on an SCR "
          "technician for 34 h)", "text", tol=0, source=NP + " and the time-log descriptions", check="code"),
        q(2, "timelog", "What was the longest single NPT event on well 78B-32, and how long did it last?",
          "waiting on the SCR technician, 9 July 00:00 to 10 July 10:00 (34 h; rig back on day rate at 13:00 on 10 July)",
          "text", source=NP, check="code"),
        q(2, "chart", "How many feet of hole were drilled on well 78B-32 between 00:00 and 06:00 on 11 July 2021?",
          round(hd[-1] - hd[0], 1), tol=10, source=f"{S}: hole depth {hd[0]} -> {hd[-1]} ft", check="code"),
        q(2, "chart", "What was the average rate of penetration on well 78B-32 between 00:00 and 06:00 on 11 July 2021 (feet drilled over elapsed time)?",
          round((hd[-1] - hd[0]) / 6, 1), tol=5, source=S + ": hole depth change / 6 h", check="code"),
        q(2, "chart", "What was the average weight on bit on well 78B-32 between 16:00 and 17:00 on 4 July 2021?",
          mean("wob_klbs", "2021-07-04T16:00", "2021-07-04T17:00"), tol=3, source=S + ", klbs", check="code"),
        q(2, "chart", "What was the average torque on well 78B-32 between 07:00 and 08:00 on 6 July 2021?",
          mean("torque_kftlb", "2021-07-06T07:00", "2021-07-06T08:00"), tol=1, source=S + ", kft-lb", check="code"),
        q(2, "chart", "What was the highest hook load on well 78B-32 on 14 July 2021, and when?",
          f"{hook[0]} klbs at {hook[1][11:16]}", "text", tol=10, source=S, check="code"),
        q(2, "chart", "When did the hole depth of well 78B-32 first reach 8,500 ft?",
          f"{t8500[:10]} at {t8500[11:16]}", "text", source=S, check="code"),
        q(2, "table", "How much hole was drilled on well 78B-32 between the 28 July and the 30 July 2021 reports?", 550,
          tol=0, source=L + " D07_r032 MD 8,950 ft, D07_r034 MD 9,500 ft"),
        q(2, "text", "Which 12-1/4 in. bit on well 56-32 had the best average ROP, and how much hole did it make?",
          "bit #3 (IADC M423): 301 ft/h, 2,919 ft", "text", source=E + " Table 10 (p37)", check="engineer"),
        q(2, "text", "How long did well 56-32 take compared with the plan, and what added time?",
          "28 days against 38 planned (TD 13 days ahead of plan); the failed 5-1/2 in. cement job added about 6 days",
          "text", source=E + " p21", check="engineer"),
        q(2, "text", "Did any 8-3/4 in. bit on well 56-32 drill more than 1,000 ft?",
          "yes, two of the last PDC bits: 1,089 ft and 1,234 ft", "text", source=E + " p21", check="engineer"),
        q(2, "text", "What went wrong with the mud hammer on well 56-32?",
          "it failed to fire or engage twice (7,620-7,627 ft and 7,627-7,662 ft), even after reaming with two roller "
          "reamers; drilling continued with PDC bits", "text", source=E + " p20, p81", check="engineer"),
        q(2, "cross_source", "How much open hole is left below the 5-1/2 in. casing shoe of well 56-32?",
          "about 40 ft (TD 9,145 ft, shoe 9,105 ft); Figure 5 gives the hole as 9,133.5 ft, which would be 28.5 ft",
          "text", source=E + " p8, p58, Figure 5", check="engineer"),
        # ---- level 3: judge ----
        q(3, "cross_source", "Do the daily report and the End-of-Well Report agree on the 9-5/8 in. casing shoe depth of well 56-32?",
          "yes, both give 3,494 ft", "text", source=L + " D06_r001; " + E + " p20 and Figure 5", check="engineer"),
        q(3, "cross_source", "The 16 July 2021 report on well 78B-32 gives an MD of 8,500 ft. Does the sensor record agree?",
          f"yes: the hole depth first reached 8,500 ft at {t8500[11:16]} on {t8500[:10]}", "text",
          source=L + " D07_r020; " + S, check="code"),
        q(3, "plan_vs_actual", "How did the 9-5/8 in. casing depth of well 56-32 compare with the plan?",
          "planned at about 3,100 ft (Figure 4), set at 3,494 ft, so about 400 ft deeper", "text",
          source=E + " Figure 4, p20", check="engineer"),
        q(3, "false_premise", "Why did well 78B-32 take a kick on 20 July 2021?",
          "no kick is recorded: that day was slip and cut, opening the core hole from 8,514 to 8,540 ft, a UBI log and "
          "a clean-out run", "text", source=L + " D07_r024 (blind)"),
        q(3, "false_premise", "What lost circulation was seen while drilling the 8-3/4 in. hole of well 56-32?",
          "none is reported while drilling (the lessons say no losses were detected); the problems came while cementing "
          "the 5-1/2 in. casing", "text", source=E + " p58, p83", check="engineer"),
        q(3, "judge", "Why could the 5-1/2 in. cement on well 56-32 not be displaced, and what should be done differently next time?",
          "pressure rose from 500 to 1,800 psi after 537 bbl and displacement stopped after 27 bbl at about 6,200 psi: "
          "the slurry set early (sodium silicate, bottom-hole temperature underestimated). Next time: no sodium silicate "
          "without losses, measure BHT first (circulate or Silixa profile), do not reciprocate casing with a fibre cable",
          "text", source=E + " p58, p83-84", check="engineer"),
        q(3, "judge", "Was the coring run on well 78B-32 on 18 July 2021 successful?",
          "only partly: about 10 ft was cored (8,530 to 8,540 ft), then the core barrel jammed and the assembly was "
          "pulled", "text", source=L + " D07_r022 (blind)"),
        q(3, "judge", "Was 9 July 2021 a productive day on well 78B-32? Explain.",
          "no: all 24 h were NPT waiting on an SCR technician, and the depth stayed at 5,761 ft", "text",
          source=L + " D07_r013 (blind)"),
        q(3, "judge", "Based on the 56-32 lessons learned, what would you change in mud-motor and BHA practice on the next well?",
          "rubric, any three: use only new motor components; test the motor without the bit and not inside casing; "
          "ream slides every 2,000-3,000 ft of directional work; do not use the CanRig oscillator while sliding; "
          "leave out the shock sub", "text", source=E + " p21, p64, p79-82", check="engineer"),
        q(3, "axis_limit", "What was the active mud volume on well 78B-32 at 08:00 on 20 July 2021?",
          at(s, "mud_volume_bbl", "2021-07-20T08:00"), tol=10,
          source=S + "; above the 200 bbl chart scale, so a right answer says the chart cannot show it", check="code"),
        q(3, "not_available", "What is the serial number of bit #17 on well 78B-32?",
          "not available: the daily reports print no serial number", "not_available",
          source=L + " (serial number empty in every labelled bit row)"),
        q(3, "not_available", "What was the mud chloride content on well 56-32 on 6 February 2021?",
          "not available: that daily report has no mud check", "not_available", source=L + " D04_r001 (no mud rows)"),
        q(3, "not_available", "What was the ROP on well 78B-32 on 15 August 2021?",
          f"not available: the chart and sensor records end on {s[-1]['time'][:10]}", "not_available", source=S,
          check="code"),
    ]


if __name__ == "__main__":
    qs = [{"id": f"Q{i:02d}", **x} for i, x in enumerate(questions(load_sensor()), 1)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(qs), "questions ->", OUT)
    for k in ("level", "category", "checked_by"):
        print(k, dict(Counter(x[k] for x in qs)))
