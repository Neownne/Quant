#!/usr/bin/env python3
"""校验分钟数据：对比分钟聚合日收益率 vs stock_daily 日收益率。

前复权（分钟）和后复权（日线）绝对价格不同，但日收益率一致。
用法:
    python3 scripts/validate_minute_data.py --sample 50
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.db import get_engine


def validate(codes: list[str]) -> dict:
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])

    # 分钟取每日最后 bar close → 计算日收益率
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
    minute_daily = minute_daily.sort_values(["code", "trade_date"])
    minute_daily["ret_m"] = minute_daily.groupby("code")["close"].pct_change()

    # 日线 close → 日收益率
    daily_df = pd.read_sql(
        f"SELECT code, trade_date, close FROM stock_daily "
        f"WHERE code IN ({code_list}) ORDER BY code, trade_date",
        engine,
    )
    engine.dispose()
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
    daily_df = daily_df.sort_values(["code", "trade_date"])
    daily_df["ret_d"] = daily_df.groupby("code")["close"].pct_change()

    merged = minute_daily.merge(daily_df, on=["code", "trade_date"], suffixes=("_m", "_d"))
    merged = merged.dropna(subset=["ret_m", "ret_d"])
    if merged.empty:
        return {"error": "无重叠收益率数据"}

    merged["ret_dev"] = (merged["ret_m"] - merged["ret_d"]).abs()
    bad = merged[merged["ret_dev"] > 0.001]  # 日收益率偏差 > 0.1%

    return {
        "n_overlap": len(merged),
        "max_ret_dev": merged["ret_dev"].max(),
        "mean_ret_dev": merged["ret_dev"].mean(),
        "n_bad": len(bad),
        "bad_samples": bad[["code", "trade_date", "ret_m", "ret_d", "ret_dev"]].head(20).to_string(),
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
    print(f"日收益率偏差: max={r['max_ret_dev']:.6f}, mean={r['mean_ret_dev']:.6f}, >0.1%={r['n_bad']} 条")
    if r["n_bad"] > 0:
        print("偏差 > 0.1% 的样本:")
        print(r["bad_samples"])
    else:
        print("校验通过 ✓")


if __name__ == "__main__":
    main()
