"""Pason 10 s sensor CSV (762 MB, 387 columns) -> small committed extract with only the charted channels.

Ground truth for chart reading: every curve on the 78B-32 Pason graph (D09) has a CSV column of the same name.
    python data/download_forge.py --sensors
    python sensor_truth.py
"""
from __future__ import annotations

import csv
import gzip
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "forge" / "78B-32_pason_10s.csv"
DST = ROOT / "data" / "processed" / "sensor" / "78B-32_pason_10s_chart.csv.gz"
MISSING = "-999.25"  # Pason / LAS null value

# chart variable -> CSV column (names as printed in the chart header and in the CSV)
COLUMNS = {
    "bit_depth_ft": "Bit Depth (feet)",
    "hole_depth_ft": "Hole Depth (feet)",
    "tvd_ft": "True Vertical Depth (feet)",
    "hook_load_klbs": "Hook Load (klbs)",
    "wob_klbs": "Weight on Bit (klbs)",
    "rpm": "Rotary RPM (RPM)",
    "torque_kftlb": "Convertible Torque (kft_lb)",
    "rop_ft_hr": "On Bottom ROP (ft_per_hr)",
    "flow_pct": "Flow (flow_percent)",
    "on_bottom_hours": "On Bottom Hours (hrs)",
    "circulating_hours": "Circulating Hours (hrs)",
    "mud_volume_bbl": "Total Mud Volume (barrels)",
}


def load(path: Path = DST) -> tuple[list[datetime], dict[str, list[float | None]]]:
    """Read the extract: sorted timestamps and one value list per variable (None where the sensor had no data)."""
    times, cols = [], {k: [] for k in COLUMNS}
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            times.append(datetime.fromisoformat(row["time"]))
            for k in COLUMNS:
                cols[k].append(float(row[k]) if row[k] else None)
    return times, cols


if __name__ == "__main__":
    DST.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(SRC, newline="") as f, gzip.open(DST, "wt", newline="") as out:
        rd = csv.reader(f)
        head = rd.__next__()
        idx = {k: head.index(c) for k, c in COLUMNS.items()}
        di, ti = head.index("YYYY/MM/DD"), head.index("HH:MM:SS")
        wr = csv.writer(out)
        wr.writerow(["time", *COLUMNS])
        for row in rd:
            if len(row) < len(head):
                continue
            t = datetime.strptime(row[di] + " " + row[ti], "%Y/%m/%d %H:%M:%S")
            wr.writerow([t.isoformat(), *("" if row[i] == MISSING else row[i] for i in idx.values())])
            n += 1
    print(f"{n} rows -> {DST} ({DST.stat().st_size / 1e6:.1f} MB)")
