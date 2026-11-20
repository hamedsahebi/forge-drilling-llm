"""SQLite database -> one self-contained HTML page for browsing every table (outputs/db_viewer.html).

    python make_db_viewer.py
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

from build_db import DB, OUTD

TEMPLATE = Path(__file__).with_name("db_viewer_template.html")
OUT = OUTD / "db_viewer.html"


def table_data(con: sqlite3.Connection, name: str) -> dict:
    """Columns with few distinct values become an index into a list, which keeps chart_readings small."""
    cur = con.execute(f'SELECT * FROM "{name}"')
    cols, rows = [c[0] for c in cur.description], [list(r) for r in cur]
    dicts = []
    for i in range(len(cols)):
        values = sorted({r[i] for r in rows}, key=str)
        if rows and len(values) <= 300 and len(values) < len(rows) / 2:
            index = {v: k for k, v in enumerate(values)}
            for r in rows:
                r[i] = index[r[i]]
            dicts.append(values)
        else:
            dicts.append(None)
    return {"cols": cols, "dicts": dicts, "rows": rows}


if __name__ == "__main__":
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    data = json.dumps({n: table_data(con, n) for n in names}, separators=(",", ":"), default=str).replace("</", "<\\/")
    built = datetime.datetime.fromtimestamp(DB.stat().st_mtime).strftime("%d %b %Y")
    OUT.write_text(TEMPLATE.read_text(encoding="utf-8").replace("{{BUILT}}", built).replace("{{DATA}}", data), encoding="utf-8")
    print(f"{len(names)} tables -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
