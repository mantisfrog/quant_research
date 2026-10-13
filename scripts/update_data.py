"""Refresh all data layers in one command.

Step 1: scan the source CSVs; export a new immutable raw snapshot only when the
        source has advanced beyond the newest existing snapshot.
Step 2: rebuild the derived valuation panel (derived/valuation/) from the
        newest snapshot.

Usage:
    .venv/Scripts/python scripts/update_data.py                  # idempotent daily refresh
    .venv/Scripts/python scripts/update_data.py --force-rebuild  # re-export even if up to date
    .venv/Scripts/python scripts/update_data.py --skip-valuation # raw snapshot only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw"
DEFAULT_DATA_DIR = Path(r"C:\Users\87913\Desktop\Gongzuo\data")
START_DATE = "2011-01-01"

sys.path.insert(0, str(SCRIPTS))
from export_quant_snapshot import date_id, parse_iso_date, scan_daily_bounds  # noqa: E402


def latest_snapshot_dir() -> Path:
    candidates = sorted(
        path for path in RAW_ROOT.glob("quant_snapshot_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError("no quant_snapshot_* directory under raw/")
    return candidates[-1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force-rebuild", action="store_true",
                        help="delete and re-export the snapshot even if it is current")
    parser.add_argument("--skip-valuation", action="store_true",
                        help="only refresh the raw snapshot")
    args = parser.parse_args()

    started = time.perf_counter()
    start_id = date_id(parse_iso_date(START_DATE))

    print("[1/2] scanning source CSVs ...", flush=True)
    _, source_max = scan_daily_bounds(args.data_dir / "fact_daily.csv", start_id)
    name = f"quant_snapshot_{start_id}_{source_max}_v1"
    target = RAW_ROOT / name

    if target.exists() and not args.force_rebuild:
        print(f"[1/2] snapshot already current: {name} (source max date {source_max})", flush=True)
    else:
        if target.exists():
            shutil.rmtree(target)
        run([sys.executable, str(SCRIPTS / "export_quant_snapshot.py"),
             "--data-dir", str(args.data_dir)])

    snapshot = latest_snapshot_dir()
    if args.skip_valuation:
        print(f"[2/2] skipped; snapshot: {snapshot.name}", flush=True)
    else:
        print(f"[2/2] rebuilding valuation panel from {snapshot.name} ...", flush=True)
        run([sys.executable, str(SCRIPTS / "build_valuation_panel.py"),
             "--snapshot-dir", str(snapshot)])

    print(f"done in {time.perf_counter() - started:.1f}s; snapshot: {snapshot.name}", flush=True)


if __name__ == "__main__":
    main()
