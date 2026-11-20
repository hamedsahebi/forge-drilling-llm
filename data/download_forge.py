"""Download the public Utah FORGE documents used by the prototype into data/raw/forge/.

Source: US DOE Geothermal Data Repository (gdr.openei.org), licensed CC BY 4.0.
    python data/download_forge.py              # report PDFs
    python data/download_forge.py --sensors    # also the Pason sensor CSVs for all three wells (~2.8 GB)
"""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DST = Path(__file__).resolve().parent / "raw" / "forge"
FILES = {
    "78B-32_DDR.pdf": "https://gdr.openei.org/files/1330/78B-32-DailyDrillingReports-6-27thru7-31.pdf",
    "78B-32_Pason_graph.pdf": "https://gdr.openei.org/files/1330/78B-32%20Pason%20graph.pdf",
    "78B-32_Mud_Log.pdf": "https://gdr.openei.org/files/1330/78B-32%20Mud%20Log.pdf",
    "56-32_EOWR.pdf": "https://gdr.openei.org/files/1295/56-32%20EOWR%20-%20Final%2022%20July%202021.pdf",
    "56-32_DDR.zip": "https://gdr.openei.org/files/1295/56-32%20Daily%20Drilling%20Reports.zip",
    "16A78-32_DDR.zip": "https://gdr.openei.org/files/1283/16A%2878%29-32_Daily_Reports.zip",
}
# Pason sensor data: ground truth for figure reading (78B-32, the D09 chart well) and for the depth and time-log
# checks. Too large for git, so only downloaded on request.
# Left out: the "Standard" copies (same data), the 56-32 full 1 s archive zip and the 16A .las zip.
SENSOR_FILES = {
    "78B-32_pason_1s.csv": "https://gdr.openei.org/files/1330/78B-32%201%20sec%20data%2027200701.csv",
    "78B-32_pason_10s.csv": "https://gdr.openei.org/files/1330/78B-32%2010%20sec%20data%2027200701.csv",
    "56-32_pason_1s.csv": "https://gdr.openei.org/files/1295/56-32%201sec%20data%2027029986.csv",
    "56-32_pason_10s.csv": "https://gdr.openei.org/files/1295/56-32%2010sec%20data%2027029986.csv",
    "16A78-32_pason_1s.csv": "https://gdr.openei.org/files/1283/16A%2878%29-32_time_data_1s_intervals.csv",
    "16A78-32_pason_10s.csv": "https://gdr.openei.org/files/1283/16A%2878%29-32_time_data_10s_intervals.csv",
}

def fetch(url: str, path: Path, tries: int = 20) -> None:
    """Download to <name>.part, resuming with an HTTP Range request after a dropped connection."""
    part = path.with_name(path.name + ".part")
    for attempt in range(tries):
        done = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if done and r.status != 206:  # server ignored the range: start again
                    done = 0
                with open(part, "ab" if done else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            part.replace(path)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and done:  # already complete
                part.replace(path)
                return
            raise
        except OSError as e:
            print(f"  {path.name}: {e}; resuming at {part.stat().st_size if part.exists() else 0:,} bytes")
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"{path.name}: gave up after {tries} tries")


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    files = {**FILES, **SENSOR_FILES} if "--sensors" in sys.argv else FILES
    for name, url in files.items():
        path = DST / name
        if not path.exists():
            print("downloading", name)
            fetch(url, path)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
                z.extractall(DST / path.stem, members)
    print("done ->", DST)
