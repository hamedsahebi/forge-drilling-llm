"""Score the LLM activity class on the dev set (D01, D02, D05) against data/labels/dev_activity.json.
Tuning only: the reference is Claude-written from the labelling rules, not hand-checked.
    python devscore.py ../outputs/repeat/C/dev_v1
"""
import json
import sys
from pathlib import Path

ref = json.loads((Path(__file__).resolve().parents[1] / "data" / "labels" / "dev_activity.json").read_text(encoding="utf-8"))
run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "processed"
hit = n = 0
for rid, classes in ref.items():
    if rid.startswith("_"):
        continue
    ops = json.loads((run / rid[:3] / "reports" / f"{rid}.json").read_text(encoding="utf-8"))["data"]["operations"]
    for i, (want, o) in enumerate(zip(classes, ops)):
        got = o.get("llm_activity") or o["activity"]
        n += 1
        hit += got == want
        if got != want:
            print(f"  {rid} row {i}: want {want}, got {got} | {(o.get('description') or '')[:90]}")
print(f"dev LLM activity: {hit}/{n}")
