"""Compare repeated extractions of the same reports field by field (repeatability test).

Each run is a directory laid out like data/processed (<doc>/reports/<report>.json). Every leaf value of `data`
is compared across runs at the same path (rows by index, so a dropped row shifts the rest and is counted
as a row-count difference). Reports with a different number of rows in a table are listed.

    python compare_runs.py ../outputs/repeat/D04_cold1 ../outputs/repeat/D04_cold2 ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def load(run: Path) -> dict[str, dict]:
    return {f.stem: json.loads(f.read_text(encoding="utf-8"))["data"] or {} for f in sorted(run.glob("*/reports/*.json"))}


def compare(runs: list[Path]) -> dict:
    data = [load(r) for r in runs]
    reports = sorted(set().union(*data))
    same, diff, by_table, row_counts, examples = 0, 0, Counter(), [], []
    for rid in reports:
        vals = [dict(leaves(d.get(rid, {}))) for d in data]
        for t in sorted({k for d in data for k, v in d.get(rid, {}).items() if isinstance(v, list)}):
            n = [len(d.get(rid, {}).get(t) or []) for d in data]
            if len(set(n)) > 1:
                row_counts.append({"report": rid, "table": t, "rows": n})
        for path in sorted(set().union(*vals)):
            got = [json.dumps(v.get(path)) for v in vals]
            if len(set(got)) == 1:
                same += 1
            else:
                diff += 1
                by_table[path.split("[")[0].split(".")[0]] += 1
                if len(examples) < 40:
                    examples.append({"report": rid, "path": path, "values": [v.get(path) for v in vals]})
    return {"runs": [str(r) for r in runs], "reports": len(reports), "fields": same + diff, "same": same,
            "differ": diff, "share_same": round(same / max(same + diff, 1), 4), "differ_by_table": dict(by_table),
            "row_count_differences": row_counts, "examples": examples}


if __name__ == "__main__":
    print(json.dumps(compare([Path(p) for p in sys.argv[1:]]), indent=1, default=str))
