"""Build the point-in-time valuation panel from the raw quant snapshot.

Outputs (build parameters and stats are recorded in _params.json):

    derived/valuation/year=YYYY/part-00000.parquet   daily panel
    derived/valuation/_params.json                   parameters + build stats
    derived/valuation_latest.parquet                 latest cross-section

Ratios (market_cap is stored in 万元, i.e. 10k CNY):

    pe          = market_cap / FN308 归母净利润TTM          (万元 / 万元)
    ps          = market_cap / FN319 营业总收入TTM          (万元 / 万元)
    pb          = market_cap / (FN271 归母股东权益 * 1e-4)   (元 -> 万元)
    pcf         = market_cap / (FN307 经营现金流净额TTM * 1e-4)

Point-in-time: on every trading day the denominator is the latest report with
announce_date_id <= date_id. Ratios with denominator <= 0 or value > RATIO_MAX
are NaN. Percentiles are per-stock expanding-window ranks of the valid ratio
history (share of observations <= today's value, 0-100, two decimals), NaN
until MIN_PERIODS valid observations exist, so they carry no look-ahead bias.

A full rebuild takes only minutes, so daily maintenance is simply a re-run.

Example:
    .venv/Scripts/python scripts/build_valuation_panel.py
    .venv/Scripts/python scripts/build_valuation_panel.py --codes 600519.SH --output-dir derived/valuation_smoke
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "derived" / "valuation"

RATIO_MAX = 10000.0
MIN_PERIODS = 250


def latest_snapshot_dir() -> Path:
    candidates = sorted(
        path for path in (ROOT / "raw").glob("quant_snapshot_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError("no quant_snapshot_* directory under raw/")
    return candidates[-1]

# output column -> (metric_code, factor converting metric units to 万元)
RATIO_SPECS: dict[str, tuple[str, float]] = {
    "pe": ("FN308", 1.0),
    "ps": ("FN319", 1.0),
    "pb": ("FN271", 1e-4),
    "pcf": ("FN307", 1e-4),
}
PCT_COLUMN = {col: f"{col.split('_')[0]}_pct" for col in RATIO_SPECS}


def load_market(snapshot: Path, codes: set[str] | None) -> pd.DataFrame:
    files = sorted((snapshot / "market").glob("year=*/part-00000.parquet"))
    if not files:
        raise FileNotFoundError(f"no market parquet files under {snapshot / 'market'}")
    df = pd.concat(
        (pd.read_parquet(f, columns=["date_id", "stock_code", "market_cap"]) for f in files),
        ignore_index=True,
    )
    df["market_cap"] = df["market_cap"].astype("float64")
    if codes is not None:
        df = df[df["stock_code"].isin(codes)]
    return df.sort_values("date_id", kind="stable").reset_index(drop=True)


def load_metric(snapshot: Path, metric_code: str, codes: set[str] | None) -> pd.DataFrame:
    dataset = ds.dataset(snapshot / "financials", format="parquet")
    df = dataset.to_table(
        columns=["stock_code", "report_period", "announce_date_id", "metric_value"],
        filter=ds.field("metric_code") == metric_code,
    ).to_pandas()
    df = df.dropna(subset=["announce_date_id"])
    df["metric_value"] = df["metric_value"].astype("float64")
    # zero is a missing-data placeholder in this source (e.g. 扣非 is not
    # disclosed in Q1/Q3 reports); drop so the asof falls back to the last
    # real disclosure. Genuine negatives (losses) are kept.
    df = df[df["metric_value"] != 0]
    if codes is not None:
        df = df[df["stock_code"].isin(codes)]
    # several report periods can share one announce date; keep the latest period
    df = df.sort_values(["stock_code", "announce_date_id", "report_period"], kind="stable")
    df = df.drop_duplicates(subset=["stock_code", "announce_date_id"], keep="last")
    return df.sort_values("announce_date_id", kind="stable").reset_index(drop=True)


def attach_ratio(panel: pd.DataFrame, snapshot: Path, codes: set[str] | None, col: str) -> None:
    metric_code, factor = RATIO_SPECS[col]
    metric = load_metric(snapshot, metric_code, codes)
    left = panel[["date_id", "stock_code"]].reset_index(names="_row")
    merged = pd.merge_asof(
        left,
        metric[["stock_code", "announce_date_id", "metric_value"]],
        left_on="date_id",
        right_on="announce_date_id",
        by="stock_code",
        direction="backward",
    )
    denom = merged.set_index("_row")["metric_value"].reindex(panel.index).to_numpy() * factor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = panel["market_cap"].to_numpy() / denom
    ratio[(denom <= 0) | (ratio > RATIO_MAX) | ~np.isfinite(ratio)] = np.nan
    panel[col] = ratio.astype("float32")


def expanding_pct(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    idx = np.flatnonzero(~np.isnan(values))
    m = idx.size
    if m < MIN_PERIODS:
        return out
    v = values[idx]
    le = np.tril(v[:, None] >= v[None, :]).sum(axis=1)
    p = le / np.arange(1, m + 1) * 100.0
    p[: MIN_PERIODS - 1] = np.nan
    out[idx] = p
    return out


def add_percentiles(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["stock_code", "date_id"], kind="stable").reset_index(drop=True)
    codes = panel["stock_code"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    for col in RATIO_SPECS:
        vals = panel[col].to_numpy(dtype="float64")
        pct = np.full(vals.shape, np.nan)
        for s, e in zip(bounds[:-1], bounds[1:]):
            pct[s:e] = expanding_pct(vals[s:e])
        panel[PCT_COLUMN[col]] = np.round(pct, 2).astype("float32")
    return panel


def write_panel(panel: pd.DataFrame, output_dir: Path, compression: str) -> None:
    panel = panel.sort_values(["date_id", "stock_code"], kind="stable").reset_index(drop=True)
    panel["stock_code"] = panel["stock_code"].astype("category")
    years = (panel["date_id"] // 10000).astype("int16")
    for year in sorted(years.unique()):
        sub = panel[years == year]
        part_dir = output_dir / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(table, part_dir / "part-00000.parquet", compression=compression)


def write_latest(panel: pd.DataFrame, snapshot: Path, path: Path) -> pd.DataFrame:
    last_date = int(panel["date_id"].max())
    latest = panel[panel["date_id"] == last_date].copy()
    latest["stock_code"] = latest["stock_code"].astype(str)
    stocks = pd.read_parquet(snapshot / "dimensions" / "stocks.parquet")
    latest = stocks.merge(latest, on="stock_code", how="inner")
    latest.to_parquet(path, index=False)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=None,
                        help="default: newest raw/quant_snapshot_* directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codes", type=str, default=None,
                        help="comma-separated stock codes for a smoke-test subset")
    parser.add_argument("--compression", type=str, default="zstd")
    args = parser.parse_args()

    snapshot = args.snapshot_dir if args.snapshot_dir is not None else latest_snapshot_dir()
    output_dir = args.output_dir
    codes = set(args.codes.split(",")) if args.codes else None

    print(f"snapshot: {snapshot}")
    print(f"output:   {output_dir}")
    panel = load_market(snapshot, codes)
    print(f"market rows: {len(panel):,}; stocks: {panel['stock_code'].nunique():,}")

    for col in RATIO_SPECS:
        attach_ratio(panel, snapshot, codes, col)
        print(f"ratio {col}: {panel[col].notna().sum():,} valid")

    panel = add_percentiles(panel)
    write_panel(panel, output_dir, args.compression)

    latest_path = output_dir.parent / f"{output_dir.name}_latest.parquet"
    latest = write_latest(panel, snapshot, latest_path)

    nan_rates = {c: round(float(panel[c].isna().mean()), 6)
                 for c in list(RATIO_SPECS) + list(PCT_COLUMN.values())}
    params = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": snapshot.name,
        "definitions": {
            "pe": "market_cap / 归母净利润TTM (FN308)",
            "ps": "market_cap / 营业总收入TTM (FN319)",
            "pb": "market_cap / 归母股东权益 (FN271, 元 converted to 万元)",
            "pcf": "market_cap / 经营活动现金流净额TTM (FN307, 元 converted to 万元)",
        },
        "market_cap_unit": "万元",
        "point_in_time": "denominator = latest report with announce_date_id <= date_id",
        "missing_placeholder": "metric_value == 0 treated as undisclosed (e.g. 扣非 in Q1/Q3 reports); "
                               "the previous real disclosure carries forward",
        "invalid_rule": f"denominator <= 0 or ratio > {RATIO_MAX:g} -> NaN; NaN excluded from percentiles",
        "percentile": f"per-stock expanding window, share of valid observations <= current * 100, "
                      f"2 decimals, NaN until {MIN_PERIODS} valid observations",
        "rows": len(panel),
        "stocks": int(panel["stock_code"].nunique()),
        "date_range": [int(panel["date_id"].min()), int(panel["date_id"].max())],
        "nan_rates": nan_rates,
        "latest_cross_section": str(latest_path),
        "latest_asof_date_id": int(panel["date_id"].max()),
        "latest_rows": len(latest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: params[k] for k in ("rows", "stocks", "date_range", "nan_rates")},
                     ensure_ascii=False, indent=2))
    print(f"latest cross-section: {latest_path} ({len(latest):,} rows)")


if __name__ == "__main__":
    main()
