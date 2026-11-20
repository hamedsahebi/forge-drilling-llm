"""Confidence intervals for the agent question scores, from the saved grades (no model runs).

Reads outputs/agent_questions/scores.json (per_question final grades from grade_questions.py) and the question levels.
  - per setting: mean score with a 95% bootstrap interval over the questions (10,000 resamples), overall and per level
  - per pair of settings: mean paired difference on the same questions, its 95% bootstrap interval, and a two-sided
    paired permutation (sign-flip) p-value; also how many questions each side wins
Question sampling only: one run per setting, so run-to-run variation of the model is not included.
    python score_ci.py        # writes outputs/agent_questions/ci.json
"""
from __future__ import annotations

import json
import random
from itertools import combinations

from grade_questions import POINTS, QUESTIONS, RUNS, SCORES

N_BOOT = 10_000
SEED = 1
OUT = RUNS / "ci.json"


def boot_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def sign_flip_p(diffs: list[float], rng: random.Random) -> float:
    obs = abs(sum(diffs))
    nz = [d for d in diffs if d]
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nz)) >= obs - 1e-9 for _ in range(N_BOOT))
    return (hits + 1) / (N_BOOT + 1)


def main() -> None:
    rng = random.Random(SEED)
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    level = {q["id"]: q["level"] for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    pts = {s: {qid: POINTS[g["final"]] for qid, g in v["per_question"].items() if g.get("final") in POINTS}
           for s, v in scores.items()}
    ids = sorted(set.intersection(*(set(p) for p in pts.values())))
    out = {"note": "95% bootstrap over questions; one run per setting, so model run-to-run variation is not included",
           "questions": len(ids), "settings": {}, "pairs": {}}
    for s, p in pts.items():
        row = {}
        for name, sel in [("all", ids)] + [(f"level{l}", [i for i in ids if level[i] == l]) for l in (1, 2, 3)]:
            vals = [p[i] for i in sel]
            lo, hi = boot_ci(vals, rng)
            row[name] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 3), "ci95": [round(lo, 3), round(hi, 3)]}
        out["settings"][s] = row
    for a, b in combinations(pts, 2):
        d = [pts[b][i] - pts[a][i] for i in ids]
        lo, hi = boot_ci(d, rng)
        out["pairs"][f"{b} - {a}"] = {"mean_diff": round(sum(d) / len(d), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                      "p_sign_flip": round(sign_flip_p(d, rng), 4),
                                      "b_better": sum(x > 0 for x in d), "a_better": sum(x < 0 for x in d),
                                      "same": sum(x == 0 for x in d)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(ids)} questions; 95% bootstrap intervals (question sampling only)")
    for s, r in out["settings"].items():
        print(f"  {s:<19}" + "  ".join(f"{k} {v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" for k, v in r.items()))
    print("paired differences (later minus earlier):")
    for k, v in out["pairs"].items():
        print(f"  {k:<40} {v['mean_diff']:+.3f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  p={v['p_sign_flip']:.3f}"
              f"  wins {v['b_better']}-{v['a_better']} (same {v['same']})")
    print("->", OUT)


if __name__ == "__main__":
    main()
