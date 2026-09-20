"""Export an immutable Parquet snapshot from source CSV files (no database needed).

Recreates the snapshot layout under raw/ (see raw/quant_snapshot_*/manifest.json):

    raw/quant_snapshot_<start>_<end>_v1/
        market/year=YYYY/part-00000.parquet
        financials/report_year=YYYY/part-00000.parquet
        dimensions/{stocks,calendar,financial_metrics}.parquet
        README.md, manifest.json, checksums.sha256

Source CSVs (see --data-dir):
    fact_daily.csv, fact_adjustment_factor_period.csv,
    fact_financial_report.csv, fact_financial_value.csv,
    dim_date.csv, dim_stock.csv, dim_tdx_industry.csv

Example:
    .venv/Scripts/python scripts/export_quant_snapshot.py
    .venv/Scripts/python scripts/export_quant_snapshot.py --end-date 2026-08-07
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(r"C:\Users\87913\Desktop\Gongzuo\data")
DEFAULT_OUTPUT_ROOT = ROOT / "raw"

SCHEMA_VERSION = "1.0.0"
DEFAULT_START_DATE = date(2011, 1, 1)
DEFAULT_FINANCIAL_START_PERIOD = date(2010, 1, 1)
CHUNK_SIZE = 1_000_000

ADJ_SCALE = Decimal("0.00000001")

METRIC_GROUPS: dict[str, tuple[str, Sequence[tuple[str, str, str]]]] = {
    "balance_sheet": (
        "stock",
        (
            ("FN8", "货币资金", "元"), ("FN11", "应收账款", "元"),
            ("FN17", "存货", "元"), ("FN21", "流动资产合计", "元"),
            ("FN25", "长期股权投资", "元"), ("FN27", "固定资产", "元"),
            ("FN28", "在建工程", "元"), ("FN33", "无形资产", "元"),
            ("FN35", "商誉", "元"), ("FN40", "资产总计", "元"),
            ("FN41", "短期借款", "元"), ("FN44", "应付账款", "元"),
            ("FN54", "流动负债合计", "元"), ("FN55", "长期借款", "元"),
            ("FN56", "应付债券", "元"), ("FN62", "非流动负债合计", "元"),
            ("FN63", "负债合计", "元"), ("FN64", "实收资本（或股本）", "元"),
            ("FN68", "未分配利润", "元"), ("FN69", "少数股东权益", "元"),
            ("FN72", "所有者权益合计", "元"),
            ("FN271", "归属于母公司股东权益", "元"),
        ),
    ),
    "income_statement": (
        "ytd",
        (
            ("FN74", "营业收入", "元"), ("FN75", "营业成本", "元"),
            ("FN76", "营业税金及附加", "元"), ("FN77", "销售费用", "元"),
            ("FN78", "管理费用", "元"), ("FN80", "财务费用", "元"),
            ("FN81", "资产减值损失", "元"), ("FN82", "公允价值变动净收益", "元"),
            ("FN83", "投资收益", "元"), ("FN86", "营业利润", "元"),
            ("FN88", "营业外收入", "元"), ("FN89", "营业外支出", "元"),
            ("FN92", "利润总额", "元"), ("FN93", "所得税", "元"),
            ("FN95", "净利润", "元"), ("FN96", "归属于母公司所有者的净利润", "元"),
            ("FN304", "研发费用", "元"),
        ),
    ),
    "cash_flow_statement": (
        "ytd",
        (
            ("FN98", "销售商品、提供劳务收到的现金", "元"),
            ("FN107", "经营活动产生的现金流量净额", "元"),
            ("FN114", "购建固定资产、无形资产和其他长期资产支付的现金", "元"),
            ("FN115", "投资支付的现金", "元"),
            ("FN119", "投资活动产生的现金流量净额", "元"),
            ("FN124", "偿还债务支付的现金", "元"),
            ("FN125", "分配股利、利润或偿付利息支付的现金", "元"),
            ("FN128", "筹资活动产生的现金流量净额", "元"),
            ("FN131", "现金及现金等价物净增加额", "元"),
            ("FN133", "期末现金及现金等价物余额", "元"),
        ),
    ),
    "single_quarter": (
        "single_quarter",
        (
            ("FN230", "营业收入（单季度）", "元"),
            ("FN328", "营业成本（单季度）", "万元"),
            ("FN231", "营业利润（单季度）", "元"),
            ("FN324", "净利润（单季度）", "万元"),
            ("FN232", "归母净利润（单季度）", "元"),
            ("FN233", "扣非净利润（单季度）", "元"),
            ("FN234", "经营活动现金流净额（单季度）", "元"),
            ("FN235", "投资活动现金流净额（单季度）", "元"),
            ("FN236", "筹资活动现金流净额（单季度）", "元"),
        ),
    ),
    "ttm": (
        "ttm",
        (
            ("FN319", "营业总收入TTM", "万元"),
            ("FN338", "营业成本TTM-非金融类", "万元"),
            ("FN323", "营业利润TTM", "万元"), ("FN276", "净利润TTM", "元"),
            ("FN308", "归母净利润TTM", "万元"),
            ("FN309", "扣非净利润TTM", "万元"),
            ("FN307", "经营活动现金流净额TTM", "元"),
            ("FN316", "投资活动现金流净额TTM", "万元"),
            ("FN310", "现金净流量TTM", "万元"),
        ),
    ),
}

METRIC_ROWS = tuple(
    (code, name, category, value_kind, unit)
    for category, (value_kind, metrics) in METRIC_GROUPS.items()
    for code, name, unit in metrics
)
METRIC_CODES = tuple(row[0] for row in METRIC_ROWS)
METRIC_SET = frozenset(METRIC_CODES)

MARKET_SCHEMA = pa.schema(
    [
        ("date_id", pa.int32()), ("date", pa.date32()), ("stock_code", pa.string()),
        ("open", pa.decimal128(18, 2)), ("high", pa.decimal128(18, 2)),
        ("low", pa.decimal128(18, 2)), ("close", pa.decimal128(18, 2)),
        ("adjust_factor", pa.decimal128(28, 12)),
        ("adj_open", pa.decimal128(38, 8)), ("adj_high", pa.decimal128(38, 8)),
        ("adj_low", pa.decimal128(38, 8)), ("adj_close", pa.decimal128(38, 8)),
        ("vol", pa.int64()), ("amount", pa.decimal128(20, 4)),
        ("total_shares", pa.int64()), ("float_shares", pa.int64()),
        ("market_cap", pa.decimal128(24, 4)),
        ("float_market_cap", pa.decimal128(24, 4)),
    ]
)

FINANCIAL_SCHEMA = pa.schema(
    [
        ("report_id", pa.int64()), ("stock_code", pa.string()),
        ("report_period", pa.date32()), ("announce_date_id", pa.int32()),
        ("metric_code", pa.string()), ("metric_value", pa.decimal128(28, 6)),
    ]
)

STOCK_SCHEMA = pa.schema(
    [
        ("stock_code", pa.string()), ("stock_name", pa.string()),
        ("tdx_sector_code", pa.string()), ("tdx_sector_name", pa.string()),
    ]
)

CALENDAR_SCHEMA = pa.schema(
    [
        ("date_id", pa.int32()), ("date", pa.date32()), ("year_num", pa.int16()),
        ("quarter_num", pa.int8()), ("month_num", pa.int8()), ("day_week", pa.int8()),
        ("is_trade_day", pa.bool_()), ("trade_day_index", pa.int32()),
    ]
)

METRIC_SCHEMA = pa.schema(
    [
        ("metric_code", pa.string()), ("metric_name", pa.string()),
        ("statement_category", pa.string()), ("value_kind", pa.string()),
        ("unit", pa.string()), ("is_financial_sector_specific", pa.bool_()),
    ]
)


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from error


def date_id(value: date) -> int:
    return value.year * 10_000 + value.month * 100 + value.day


def id_date(value: int) -> date:
    text = str(value)
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", type=parse_iso_date, default=DEFAULT_START_DATE)
    parser.add_argument(
        "--financial-start-period",
        type=parse_iso_date,
        default=DEFAULT_FINANCIAL_START_PERIOD,
    )
    parser.add_argument("--end-date", type=parse_iso_date)
    return parser.parse_args()


def quantize8(value: Decimal) -> Decimal:
    # PostgreSQL NUMERIC rounds half away from zero; ROUND_HALF_UP matches
    # that behavior for the non-negative prices handled here.
    return value.quantize(ADJ_SCALE, rounding=ROUND_HALF_UP)


def is_na(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def to_decimal(value, scale: int | None = None) -> Decimal | None:
    if is_na(value):
        return None
    result = Decimal(value)
    if scale is not None:
        result = result.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)
    return result


def to_int(value) -> int | None:
    return None if is_na(value) else int(value)


class YearlyWriter:
    """Streams batches into one part-00000.parquet per partition year."""

    def __init__(self, root: Path, layout: str, schema: pa.Schema):
        self.root = root
        self.layout = layout
        self.schema = schema
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.counts: dict[int, int] = {}

    def write(self, year: int, columns: dict[str, list]) -> None:
        writer = self.writers.get(year)
        if writer is None:
            path = self.root / self.layout.format(year=year)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(path, self.schema, compression="zstd")
            self.writers[year] = writer
            self.counts[year] = 0
        rows = len(next(iter(columns.values())))
        if rows:
            writer.write_table(
                pa.Table.from_pydict(columns, schema=self.schema),
                row_group_size=CHUNK_SIZE,
            )
            self.counts[year] += rows

    def close(self) -> dict[int, int]:
        for writer in self.writers.values():
            writer.close()
        return self.counts


def scan_daily_bounds(daily_csv: Path, start_id: int) -> tuple[int | None, int]:
    minimum: int | None = None
    maximum = 0
    for chunk in pd.read_csv(daily_csv, usecols=["date_id"], chunksize=4_000_000):
        values = chunk["date_id"]
        maximum = max(maximum, int(values.max()))
        in_range = values[values >= start_id]
        if len(in_range):
            chunk_min = int(in_range.min())
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
    return minimum, maximum


def load_adjustment_periods(data_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    adj = pd.read_csv(data_dir / "fact_adjustment_factor_period.csv", dtype={"adjust_factor": str})
    adj = adj.sort_values(["stock_code", "valid_from_date_id"])
    periods: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for stock_code, group in adj.groupby("stock_code", sort=False):
        periods[stock_code] = (
            group["valid_from_date_id"].to_numpy(),
            group["valid_to_date_id"].to_numpy(),
            group["adjust_factor"].tolist(),
        )
    return periods


def load_date_map(data_dir: Path) -> dict[int, date]:
    dim = pd.read_csv(data_dir / "dim_date.csv", usecols=["date_id", "date"])
    return dict(zip(dim["date_id"], pd.to_datetime(dim["date"]).dt.date))


def iter_year_slices(years: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (year, indices) groups for arbitrarily ordered input."""
    for year in np.unique(years):
        yield int(year), np.flatnonzero(years == year)


def export_market(
    data_dir: Path,
    temp_root: Path,
    start_id: int,
    end_id: int,
    periods: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    dates: dict[int, date],
) -> tuple[dict[int, int], int]:
    writer = YearlyWriter(temp_root, "market/year={year}/part-00000.parquet", MARKET_SCHEMA)
    dropped = 0
    decimal_dtypes = {
        name: str for name in (
            "open", "high", "low", "close", "vol", "amount",
            "total_shares", "float_shares", "market_cap", "float_market_cap",
        )
    }
    reader = pd.read_csv(data_dir / "fact_daily.csv", dtype=decimal_dtypes, chunksize=CHUNK_SIZE)
    try:
        for chunk in reader:
            chunk = chunk[(chunk["date_id"] >= start_id) & (chunk["date_id"] <= end_id)]
            if not len(chunk):
                continue

            date_ids = chunk["date_id"].to_numpy()
            stock_codes = chunk["stock_code"].tolist()
            keys = list(zip(date_ids.tolist(), stock_codes))

            factors: list[Decimal | None] = []
            for date_value, stock_code in keys:
                period = periods.get(stock_code)
                factor: Decimal | None = None
                if period is not None:
                    valid_from, valid_to, factor_texts = period
                    index = int(np.searchsorted(valid_from, date_value, side="right")) - 1
                    if index >= 0 and date_value <= valid_to[index]:
                        factor = to_decimal(factor_texts[index], 12)
                factors.append(factor)

            keep = [i for i, factor in enumerate(factors) if factor is not None]
            dropped += len(factors) - len(keep)
            if not keep:
                continue

            opens = [to_decimal(chunk["open"].iat[i], 2) for i in keep]
            highs = [to_decimal(chunk["high"].iat[i], 2) for i in keep]
            lows = [to_decimal(chunk["low"].iat[i], 2) for i in keep]
            closes = [to_decimal(chunk["close"].iat[i], 2) for i in keep]
            kept_factors = [factors[i] for i in keep]
            columns = {
                "date_id": [int(date_ids[i]) for i in keep],
                "date": [dates[int(date_ids[i])] for i in keep],
                "stock_code": [stock_codes[i] for i in keep],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "adjust_factor": kept_factors,
                "adj_open": [
                    None if value is None else quantize8(value * factor)
                    for value, factor in zip(opens, kept_factors)
                ],
                "adj_high": [
                    None if value is None else quantize8(value * factor)
                    for value, factor in zip(highs, kept_factors)
                ],
                "adj_low": [
                    None if value is None else quantize8(value * factor)
                    for value, factor in zip(lows, kept_factors)
                ],
                "adj_close": [
                    None if value is None else quantize8(value * factor)
                    for value, factor in zip(closes, kept_factors)
                ],
                "vol": [to_int(chunk["vol"].iat[i]) for i in keep],
                "amount": [to_decimal(chunk["amount"].iat[i], 4) for i in keep],
                "total_shares": [to_int(chunk["total_shares"].iat[i]) for i in keep],
                "float_shares": [to_int(chunk["float_shares"].iat[i]) for i in keep],
                "market_cap": [to_decimal(chunk["market_cap"].iat[i], 4) for i in keep],
                "float_market_cap": [to_decimal(chunk["float_market_cap"].iat[i], 4) for i in keep],
            }
            year_values = np.array([value // 10_000 for value in columns["date_id"]])
            for year, indices in iter_year_slices(year_values):
                writer.write(
                    year, {name: [values[i] for i in indices] for name, values in columns.items()}
                )
    finally:
        counts = writer.close()
    return counts, dropped


def load_reports(
    data_dir: Path, financial_start: date, financial_end: date, end_id: int
) -> dict[int, tuple[str, date, int]]:
    reports = pd.read_csv(data_dir / "fact_financial_report.csv", encoding="utf-8-sig")
    mask = (
        (reports["report_period"] >= financial_start.isoformat())
        & (reports["report_period"] <= financial_end.isoformat())
        & (reports["announce_date_id"] <= end_id)
    )
    reports = reports[mask]
    return {
        int(row.report_id): (row.stock_code, date.fromisoformat(row.report_period), int(row.announce_date_id))
        for row in reports.itertuples()
    }


def export_financials(
    data_dir: Path,
    temp_root: Path,
    reports: dict[int, tuple[str, date, int]],
) -> dict[int, int]:
    writer = YearlyWriter(
        temp_root, "financials/report_year={year}/part-00000.parquet", FINANCIAL_SCHEMA
    )
    reader = pd.read_csv(
        data_dir / "fact_financial_value.csv",
        encoding="utf-8-sig",
        dtype={"metric_value": str},
        chunksize=CHUNK_SIZE,
    )
    try:
        for chunk in reader:
            mask = chunk["metric_code"].isin(METRIC_SET) & chunk["report_id"].isin(reports.keys())
            chunk = chunk[mask]
            if not len(chunk):
                continue
            report_ids = chunk["report_id"].tolist()
            metric_codes = chunk["metric_code"].tolist()
            metric_values = chunk["metric_value"].tolist()
            matched = [reports[report_id] for report_id in report_ids]
            columns = {
                "report_id": report_ids,
                "stock_code": [item[0] for item in matched],
                "report_period": [item[1] for item in matched],
                "announce_date_id": [item[2] for item in matched],
                "metric_code": metric_codes,
                "metric_value": [to_decimal(value, 6) for value in metric_values],
            }
            year_values = np.array([item[1].year for item in matched])
            for year, indices in iter_year_slices(year_values):
                writer.write(
                    year, {name: [values[i] for i in indices] for name, values in columns.items()}
                )
    finally:
        counts = writer.close()
    return counts


def export_stocks(data_dir: Path, path: Path) -> int:
    stocks = pd.read_csv(data_dir / "dim_stock.csv", encoding="utf-8-sig", dtype=str)
    industries = pd.read_csv(data_dir / "dim_tdx_industry.csv", encoding="utf-8-sig", dtype=str)
    merged = stocks.merge(industries, on="tdx_sector_code", how="left")
    merged = merged.sort_values("stock_code")
    columns = {
        name: [None if is_na(value) else str(value) for value in merged[name]]
        for name in ("stock_code", "stock_name", "tdx_sector_code", "tdx_sector_name")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=STOCK_SCHEMA), path, compression="zstd")
    return len(merged)


def export_calendar(data_dir: Path, path: Path, start_id: int, end_id: int) -> int:
    dim = pd.read_csv(data_dir / "dim_date.csv")
    dim = dim[(dim["date_id"] >= start_id) & (dim["date_id"] <= end_id)].sort_values("date_id")
    trade_day_index = [
        int(value) if pd.notna(value) else None for value in dim["trade_day_index"]
    ]
    columns = {
        "date_id": [int(value) for value in dim["date_id"]],
        "date": pd.to_datetime(dim["date"]).dt.date.tolist(),
        "year_num": [int(value) for value in dim["year_num"]],
        "quarter_num": [int(value) for value in dim["quarter_num"]],
        "month_num": [int(value) for value in dim["month_num"]],
        "day_week": [int(value) for value in dim["day_week"]],
        "is_trade_day": [bool(value) for value in dim["is_trade_day"]],
        "trade_day_index": trade_day_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=CALENDAR_SCHEMA), path, compression="zstd")
    return len(dim)


def export_metrics(path: Path) -> int:
    columns = {
        "metric_code": [row[0] for row in METRIC_ROWS],
        "metric_name": [row[1] for row in METRIC_ROWS],
        "statement_category": [row[2] for row in METRIC_ROWS],
        "value_kind": [row[3] for row in METRIC_ROWS],
        "unit": [row[4] for row in METRIC_ROWS],
        "is_financial_sector_specific": [False] * len(METRIC_ROWS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=METRIC_SCHEMA), path, compression="zstd")
    return len(METRIC_ROWS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def schema_record(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def validate_parquet(path: Path, expected_schema: pa.Schema, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(f"row count mismatch for {path}")
    if not parquet.schema_arrow.equals(expected_schema):
        raise RuntimeError(f"schema mismatch for {path}")


def write_readme(root: Path, start_id: int, end_id: int, financial_start: date) -> None:
    text = f"""# Quant research snapshot

- Schema version: `{SCHEMA_VERSION}`
- Market range: `{id_date(start_id).isoformat()}` through `{id_date(end_id).isoformat()}`
- Financial report periods: `{financial_start.isoformat()}` onward, with announcements through `{id_date(end_id).isoformat()}`
- Prices: raw OHLC and back-adjusted OHLC (`raw * adjust_factor`)
- Volume and amount: unadjusted actual values
- Financial availability: filter with `announce_date_id <= research_date_id`

```python
import pandas as pd

market = pd.read_parquet("market", filters=[("year", ">=", 2020)])
financials = pd.read_parquet("financials", filters=[("report_year", ">=", 2020)])
```

`stocks.parquet` contains current names and industries, not historical point-in-time names,
ST flags, or industries. Financial values retain source-native units; consult
`dimensions/financial_metrics.parquet` before combining metrics.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def export_snapshot(args: argparse.Namespace) -> Path:
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_start_id = date_id(args.start_date)
    actual_start_id, source_max_id = scan_daily_bounds(
        data_dir / "fact_daily.csv", requested_start_id
    )
    end_id = date_id(args.end_date) if args.end_date else source_max_id
    if end_id > source_max_id:
        raise ValueError(f"end date {end_id} exceeds source maximum {source_max_id}")
    if actual_start_id is None or actual_start_id > end_id:
        raise RuntimeError("no market rows in requested date range")

    snapshot_name = f"quant_snapshot_{requested_start_id}_{end_id}_v1"
    final_root = output_root / snapshot_name
    temp_root = output_root / f".{snapshot_name}.tmp-{os.getpid()}"
    if final_root.exists():
        raise FileExistsError(f"snapshot already exists: {final_root}")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir()

    try:
        files: list[dict[str, Any]] = []

        periods = load_adjustment_periods(data_dir)
        dates = load_date_map(data_dir)
        market_counts, dropped = export_market(
            data_dir, temp_root, actual_start_id, end_id, periods, dates
        )
        if dropped:
            print(f"market rows without adjustment factor (dropped): {dropped:,}", flush=True)
        for year in sorted(market_counts):
            path = temp_root / "market" / f"year={year}" / "part-00000.parquet"
            validate_parquet(path, MARKET_SCHEMA, market_counts[year])
            files.append(file_record(temp_root, path, market_counts[year]))
            print(f"market {year}: {market_counts[year]:,} rows", flush=True)
        market_total = sum(market_counts.values())

        financial_end = id_date(end_id)
        reports = load_reports(data_dir, args.financial_start_period, financial_end, end_id)
        financial_counts = export_financials(data_dir, temp_root, reports)
        for year in sorted(financial_counts):
            path = temp_root / "financials" / f"report_year={year}" / "part-00000.parquet"
            validate_parquet(path, FINANCIAL_SCHEMA, financial_counts[year])
            files.append(file_record(temp_root, path, financial_counts[year]))
            print(f"financial {year}: {financial_counts[year]:,} rows", flush=True)
        financial_total = sum(financial_counts.values())

        stocks_path = temp_root / "dimensions" / "stocks.parquet"
        stock_count = export_stocks(data_dir, stocks_path)
        validate_parquet(stocks_path, STOCK_SCHEMA, stock_count)
        files.append(file_record(temp_root, stocks_path, stock_count))

        calendar_path = temp_root / "dimensions" / "calendar.parquet"
        calendar_count = export_calendar(data_dir, calendar_path, requested_start_id, end_id)
        validate_parquet(calendar_path, CALENDAR_SCHEMA, calendar_count)
        files.append(file_record(temp_root, calendar_path, calendar_count))

        metrics_path = temp_root / "dimensions" / "financial_metrics.parquet"
        metric_count = export_metrics(metrics_path)
        validate_parquet(metrics_path, METRIC_SCHEMA, metric_count)
        files.append(file_record(temp_root, metrics_path, metric_count))

        write_readme(temp_root, actual_start_id, end_id, args.financial_start_period)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "CSV",
            "requested_market_start": args.start_date.isoformat(),
            "actual_market_start": id_date(actual_start_id).isoformat(),
            "market_end": id_date(end_id).isoformat(),
            "financial_start_period": args.financial_start_period.isoformat(),
            "metric_count": len(METRIC_CODES),
            "row_counts": {"market": market_total, "financials": financial_total},
            "schemas": {
                "market": schema_record(MARKET_SCHEMA),
                "financials": schema_record(FINANCIAL_SCHEMA),
                "stocks": schema_record(STOCK_SCHEMA),
                "calendar": schema_record(CALENDAR_SCHEMA),
                "financial_metrics": schema_record(METRIC_SCHEMA),
            },
            "files": files,
        }
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksum_paths = sorted(
            path for path in temp_root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
        )
        checksum_text = "".join(
            f"{sha256_file(path)}  {path.relative_to(temp_root).as_posix()}\n"
            for path in checksum_paths
        )
        (temp_root / "checksums.sha256").write_text(checksum_text, encoding="ascii")

        temp_root.rename(final_root)
        print(f"snapshot published: {final_root}", flush=True)
        return final_root
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def main() -> int:
    args = parse_args()
    export_snapshot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
