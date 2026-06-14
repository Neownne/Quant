#!/usr/bin/env python
"""数据完整性检查脚本。

功能：
1. 检查各表最新日期、字段填充率、跨表一致性。
2. 按股票维度检测历史交易日缺口。
3. 输出报告到控制台和 data/data_integrity_report.json。

用法：
    python scripts/check_data_integrity.py
"""
import json
import sys
import os
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from loguru import logger

from data.db import get_engine
from config.settings import PROJECT_ROOT


REPORT_PATH = PROJECT_ROOT / "data" / "data_integrity_report.json"
COVERAGE_WINDOW_DAYS = 30


def _get_engine():
    return get_engine()


def check_table_status(engine) -> dict:
    """检查各表最新日期、记录数、代码数。"""
    table_checks = {
        "stock_daily": ("code", "trade_date"),
        "stock_daily_extra": ("code", "trade_date"),
        "index_daily": ("code", "trade_date"),
        "etf_daily": ("code", "trade_date"),
        "stock_basic": ("code", None),
        "etf_basic": ("code", None),
    }
    results = {}
    with engine.connect() as conn:
        for table, (id_col, date_col) in table_checks.items():
            entry = {"exists": False, "n_records": 0, "n_codes": 0, "latest_date": None}
            exists = conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=:t)"),
                {"t": table},
            ).scalar()
            if not exists:
                results[table] = entry
                continue
            entry["exists"] = True
            entry["n_records"] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            entry["n_codes"] = conn.execute(text(f"SELECT COUNT(DISTINCT {id_col}) FROM {table}")).scalar()
            if date_col:
                latest = conn.execute(text(f"SELECT MAX({date_col}) FROM {table}")).scalar()
                entry["latest_date"] = str(latest) if latest else None
            results[table] = entry
    return results


def check_extra_fill_rate(engine) -> dict:
    """检查 stock_daily_extra 各字段填充率。"""
    cols = ["market_cap", "pb", "pe", "float_market_cap", "total_share", "float_share"]
    sql = "SELECT COUNT(*) AS total, " + ", ".join(
        f"COUNT(*) FILTER (WHERE {c} IS NOT NULL) AS {c}" for c in cols
    ) + " FROM stock_daily_extra"
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().fetchone()
    total = row["total"]
    return {
        c: {"filled": row[c], "total": total, "rate": round(row[c] / total, 4) if total else 0.0}
        for c in cols
    }


def check_recent_coverage(engine, window_days: int = COVERAGE_WINDOW_DAYS) -> dict:
    """检查近 N 日单日覆盖率（股票日线）。"""
    with engine.connect() as conn:
        total_stocks = conn.execute(
            text("SELECT COUNT(*) FROM stock_basic WHERE is_st = FALSE")
        ).scalar()

        df = pd.read_sql(
            text("""
                SELECT trade_date, COUNT(DISTINCT code) AS n
                FROM stock_daily
                WHERE trade_date >= CURRENT_DATE - CAST(:days || ' days' AS INTERVAL)
                GROUP BY trade_date
                ORDER BY trade_date
            """),
            conn,
            params={"days": str(window_days)},
        )
    df["coverage"] = df["n"] / total_stocks if total_stocks else 0.0
    df["trade_date"] = df["trade_date"].astype(str)
    return {
        "total_stocks": total_stocks,
        "days": df.to_dict(orient="records"),
    }


def check_missing_stocks(engine, threshold: float = 0.9) -> dict:
    """按股票维度检测历史交易日缺口。

    以 stock_daily 的 MIN/MAX trade_date 为范围生成交易日历，
    计算每只股票实际交易日数 / 预期交易日数，返回覆盖率低于阈值的股票。
    """
    with engine.connect() as conn:
        min_max = conn.execute(
            text("SELECT MIN(trade_date), MAX(trade_date) FROM stock_daily")
        ).fetchone()
        min_date, max_date = min_max
        if not min_date or not max_date:
            return {"range": None, "threshold": threshold, "count": 0, "examples": []}

        # 生成预期交易日历（排除周末）
        calendar = pd.date_range(start=min_date, end=max_date, freq="B").date
        expected_days = len(calendar)

        # 每只股票实际交易日数
        actual = pd.read_sql(
            text("""
                SELECT sb.code, COUNT(DISTINCT sd.trade_date) AS actual_days
                FROM stock_basic sb
                LEFT JOIN stock_daily sd ON sd.code = sb.code
                WHERE sb.is_st = FALSE
                GROUP BY sb.code
            """),
            conn,
        )

    actual["expected_days"] = expected_days
    actual["coverage"] = actual["actual_days"] / expected_days
    missing = actual[actual["coverage"] < threshold].sort_values("coverage")
    return {
        "range": {"min": str(min_date), "max": str(max_date), "expected_days": expected_days},
        "threshold": threshold,
        "count": int(len(missing)),
        "examples": missing.head(20).to_dict(orient="records"),
    }


def check_cross_table_consistency(engine) -> dict:
    """检查跨表一致性。"""
    with engine.connect() as conn:
        # stock_daily vs stock_daily_extra 日期范围差异
        ranges = conn.execute(text("""
            SELECT
                'stock_daily' AS tbl, MIN(trade_date) AS min_date, MAX(trade_date) AS max_date
            FROM stock_daily
            UNION ALL
            SELECT
                'stock_daily_extra', MIN(trade_date), MAX(trade_date)
            FROM stock_daily_extra
        """)).mappings().all()
        ranges = {r["tbl"]: {"min_date": str(r["min_date"]), "max_date": str(r["max_date"])} for r in ranges}

        # stock_daily 有但 stock_daily_extra 没有的代码
        missing_in_extra = conn.execute(text("""
            SELECT COUNT(DISTINCT code) FROM stock_daily
            WHERE code NOT IN (SELECT DISTINCT code FROM stock_daily_extra)
        """)).scalar()

        # stock_daily_extra 有但 stock_daily 没有的代码
        missing_in_daily = conn.execute(text("""
            SELECT COUNT(DISTINCT code) FROM stock_daily_extra
            WHERE code NOT IN (SELECT DISTINCT code FROM stock_daily)
        """)).scalar()

    return {
        "date_ranges": ranges,
        "codes_in_daily_not_extra": int(missing_in_extra),
        "codes_in_extra_not_daily": int(missing_in_daily),
    }


def run_all_checks(engine) -> dict:
    """运行所有检查并返回报告。"""
    report = {
        "generated_at": date.today().isoformat(),
        "table_status": check_table_status(engine),
        "extra_fill_rate": check_extra_fill_rate(engine),
        "recent_coverage": check_recent_coverage(engine),
        "missing_stocks": check_missing_stocks(engine),
        "cross_table_consistency": check_cross_table_consistency(engine),
    }
    return report


def print_report(report: dict) -> None:
    """将报告打印到控制台。"""
    print("=" * 60)
    print("数据完整性报告")
    print("=" * 60)

    print("\n[表级状态]")
    for table, info in report["table_status"].items():
        if not info["exists"]:
            print(f"  {table}: 表不存在")
            continue
        print(
            f"  {table}: 记录 {info['n_records']:,}, 代码 {info['n_codes']:,}, "
            f"最新 {info['latest_date']}"
        )

    print("\n[extra 字段填充率]")
    for col, info in report["extra_fill_rate"].items():
        print(f"  {col}: {info['filled']:,}/{info['total']:,} ({info['rate']*100:.1f}%)")

    print("\n[近30日覆盖率]")
    rc = report["recent_coverage"]
    print(f"  股票总数: {rc['total_stocks']}")
    for row in rc["days"][-5:]:
        print(f"  {row['trade_date']}: {row['n']} 只 ({row['coverage']*100:.1f}%)")

    print("\n[缺失股票]")
    ms = report["missing_stocks"]
    print(f"  范围: {ms['range']['min']} ~ {ms['range']['max']}")
    print(f"  覆盖率 < {ms['threshold']*100:.0f}% 的股票: {ms['count']} 只")
    for ex in ms["examples"][:5]:
        print(f"    {ex['code']}: {ex['actual_days']}/{ex['expected_days']} ({ex['coverage']*100:.1f}%)")

    print("\n[跨表一致性]")
    ct = report["cross_table_consistency"]
    for tbl, rng in ct["date_ranges"].items():
        print(f"  {tbl}: {rng['min_date']} ~ {rng['max_date']}")
    print(f"  daily 有但 extra 无: {ct['codes_in_daily_not_extra']} 只")
    print(f"  extra 有但 daily 无: {ct['codes_in_extra_not_daily']} 只")

    print("=" * 60)


def main():
    engine = _get_engine()
    try:
        report = run_all_checks(engine)
        print_report(report)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {REPORT_PATH}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
