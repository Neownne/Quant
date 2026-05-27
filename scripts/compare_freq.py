#!/usr/bin/env python3
"""日频 vs 分钟频 ML 回测对比。"""
import sys
import time
sys.path.insert(0, ".")

import pandas as pd
from data.db import get_engine
from sqlalchemy import text
# [架构重构] app.utils 已移除，此处需要重构
from app.utils.ml_backtest import run_ml_backtest  # noqa: F401 (removed in v2.0)


def get_codes(n: int | None = None) -> list[str]:
    engine = get_engine()
    sql = "SELECT DISTINCT code FROM stock_minute WHERE period='60' ORDER BY code"
    if n:
        sql += f" LIMIT {n}"
    with engine.connect() as conn:
        codes = pd.read_sql(text(sql), conn)["code"].tolist()
    engine.dispose()
    return codes


def main():
    codes = get_codes()

    # 取 2024-05 到 2026-05，分钟数据覆盖的区间
    start = "20240501"
    end = "20260526"

    config = {
        "name": "对比测试",
        "train_years": 1,
        "val_years": 1,
        "top_n": 15,
        "rebalance_mode": "ndrop",
        "ndrop_n": 2,
        "stop_loss_pct": 0.08,
        "max_dd_limit": 0.25,
    }

    print(f"股票池: {len(codes)} 只, 区间: {start}-{end}")
    print()

    # -- 日频 --
    print("=" * 50)
    print("运行日频回测 (daily)...")
    t0 = time.time()
    daily_config = {**config, "freq": "daily", "name": "daily-compare"}
    daily_result = run_ml_backtest(daily_config, codes, start, end, initial_cash=1_000_000)
    daily_time = time.time() - t0

    if "error" in daily_result:
        print(f"日频回测失败: {daily_result['error']}")
        return

    # -- 分钟频 --
    print("=" * 50)
    print("运行分钟频回测 (60min)...")
    t0 = time.time()
    minute_config = {**config, "freq": "60min", "name": "60min-compare"}
    minute_result = run_ml_backtest(minute_config, codes, start, end, initial_cash=1_000_000)
    minute_time = time.time() - t0

    if "error" in minute_result:
        print(f"分钟频回测失败: {minute_result['error']}")
        return

    # -- 对比 --
    print()
    print("=" * 50)
    print("对比结果:")
    print("=" * 50)
    dm = daily_result["metrics"]
    mm = minute_result["metrics"]

    rows = [
        ("年化收益率", f"{dm['annual_return']:.2%}", f"{mm['annual_return']:.2%}"),
        ("总收益率", f"{dm['total_return']:.2%}", f"{mm['total_return']:.2%}"),
        ("最大回撤", f"{dm['max_drawdown']:.1f}%", f"{mm['max_drawdown']:.1f}%"),
        ("夏普比率", f"{dm['sharpe_ratio']:.2f}", f"{mm['sharpe_ratio']:.2f}"),
        ("胜率", f"{dm['win_rate']:.2%}", f"{mm['win_rate']:.2%}"),
        ("交易次数", str(dm["n_trades"]), str(mm["n_trades"])),
        ("可用因子数", str(len(daily_result["results_json"]["active_factors"])),
                       str(len(minute_result["results_json"]["active_factors"]))),
        ("耗时", f"{daily_time:.0f}s", f"{minute_time:.0f}s"),
    ]

    print(f"{'指标':<16} {'日频(daily)':<16} {'分钟频(60min)':<16}")
    print("-" * 48)
    for label, d, m in rows:
        print(f"{label:<16} {d:<16} {m:<16}")

    # 保存权益曲线对比
    import json
    daily_eq = daily_result["equity_curve"].to_dict(orient="records")
    minute_eq = minute_result["equity_curve"].to_dict(orient="records")
    comparison = {
        "config": config,
        "n_codes": len(codes),
        "daily": {"metrics": dm, "equity_curve": daily_eq},
        "minute": {"metrics": mm, "equity_curve": minute_eq},
    }
    # Convert date/timestamp to string
    for series in [daily_eq, minute_eq]:
        for row in series:
            row["date"] = str(row["date"])
    with open("/tmp/backtest_compare.json", "w") as f:
        json.dump(comparison, f, default=str)
    print("\n对比数据已保存到 /tmp/backtest_compare.json")


if __name__ == "__main__":
    main()
