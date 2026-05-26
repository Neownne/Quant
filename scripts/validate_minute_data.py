#!/usr/bin/env python3
"""校验分钟数据：聚合日频 close 对比 stock_daily close。

用法:
    python3 scripts/validate_minute_data.py --sample 50
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from data.db import get_engine


def validate(codes: list[str]) -> dict:
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])

    # 分钟取每日最后 bar close
    minute_sql = f"""
        SELECT code, trade_time::date AS trade_date, close
        FROM stock_minute
        WHERE code IN ({code_list}) AND period = '60'
        ORDER BY code, trade_time
    """
    minute_df = pd.read_sql(minute_sql, engine)
    if minute_df.empty:
        engine.dispose()
        return {"error": "无分钟数据，请先运行 sync_minute_data.py"}

    minute_daily = minute_df.groupby(["code", "trade_date"])["close"].last().reset_index()
    minute_daily["trade_date"] = pd.to_datetime(minute_daily["trade_date"])

    # 日线 close
    daily_df = pd.read_sql(
        f"SELECT code, trade_date, close FROM stock_daily "
        f"WHERE code IN ({code_list}) ORDER BY code, trade_date",
        engine,
    )
    engine.dispose()
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])

    merged = minute_daily.merge(daily_df, on=["code", "trade_date"], suffixes=("_m", "_d"))
    if merged.empty:
        return {"error": "无重叠日期"}

    merged["deviation"] = (merged["close_m"] - merged["close_d"]).abs() / merged["close_d"]
    bad = merged[merged["deviation"] > 0.01]

    return {
        "n_overlap": len(merged),
        "max_deviation": merged["deviation"].max(),
        "mean_deviation": merged["deviation"].mean(),
        "n_bad": len(bad),
        "bad_samples": bad[["code", "trade_date", "close_m", "close_d", "deviation"]].head(20).to_string(),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args()

    engine = get_engine()
    codes = pd.read_sql(
        f"SELECT DISTINCT code FROM stock_minute WHERE period='60' LIMIT {args.sample}", engine
    )["code"].tolist()
    engine.dispose()

    if not codes:
        print("无分钟数据，请先运行 sync_minute_data.py")
        return

    r = validate(codes)
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return

    print(f"样本: {len(codes)} 只, 重叠日: {r['n_overlap']}")
    print(f"Close偏差: max={r['max_deviation']:.4%}, mean={r['mean_deviation']:.4%}, >1%= {r['n_bad']} 条")
    if r["n_bad"] > 0:
        print("异常:")
        print(r["bad_samples"])
    else:
        print("校验通过 ✓")


if __name__ == "__main__":
    main()
