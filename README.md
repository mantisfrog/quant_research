# quant_research
解耦数据获取，清洗工作后的因子研究工作流

## 数据快照

`raw/` 下的行情/财务快照由 `scripts/export_quant_snapshot.py` 从源 CSV 文件导出
（默认数据目录 `C:\Users\87913\Desktop\Gongzuo\data`，包含 `fact_daily.csv`、
`fact_adjustment_factor_period.csv`、`fact_financial_report.csv`、
`fact_financial_value.csv`、`dim_date.csv`、`dim_stock.csv`、`dim_tdx_industry.csv`）：

```bash
# 全量导出（2011-01-01 起，截止到源数据最新交易日），输出到 raw/
.venv/Scripts/python scripts/export_quant_snapshot.py

# 自定义区间 / 数据源 / 输出目录
.venv/Scripts/python scripts/export_quant_snapshot.py \
    --start-date 2011-01-01 --end-date 2026-08-07 \
    --financial-start-period 2010-01-01 \
    --data-dir "C:\Users\87913\Desktop\Gongzuo\data" \
    --output-root raw
```

依赖：`pandas`、`pyarrow`。财务指标清单（67 个 FN 代码）内置于脚本中。

每个快照目录内含 `manifest.json`（schema、行数、逐文件 sha256）与
`checksums.sha256`，可用于校验数据完整性。
