"""Merge grades saved on the grading page into data/questions/hand_grades.json.

The page's `grades` collection is read with Claude's ArtifactData tool (`list`, out_dir = a scratch folder), which
writes one JSON file per grade under <dir>/grades/. Grades from the page replace the same key in the file; a page
entry with no grade (a note only, or cleared) is left out. Notes go to data/questions/hand_grade_notes.json.
    python pull_grades.py <dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from grade_questions import HAND, POINTS

NOTES = HAND.with_name("hand_grade_notes.json")


def main(folder: str) -> None:
    hand = json.loads(HAND.read_text(encoding="utf-8")) if HAND.exists() else {}
    notes = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}
    files = sorted(Path(folder).rglob("*.json"))
    n = 0
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        d = d.get("data", d)  # the saved file may wrap the document
        key = d.get("key")
        if not key:
            continue
        if d.get("grade") in POINTS:
            hand[key] = d["grade"]
            n += 1
        if d.get("note"):
            notes[key] = d["note"]
    HAND.write_text(json.dumps(dict(sorted(hand.items())), indent=1), encoding="utf-8")
    if notes:
        NOTES.write_text(json.dumps(dict(sorted(notes.items())), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{n} grades from {len(files)} page entries -> {HAND} ({len(hand)} in total), {len(notes)} notes")


if __name__ == "__main__":
    main(sys.argv[1])
