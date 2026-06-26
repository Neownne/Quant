#!/usr/bin/env python
"""涨停池信号生成器 —— 新规则，对齐管线 CSV 格式。

4条件: 市值30-500亿 + 股价5-100 + MA5>MA10 + 近20日涨停2-4次
不要求当日涨停。

用法:
    python scripts/gen_limit_up_signals.py --start 2020-01-01 --top-n 30
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd
from datetime import timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from config.settings import TradingConfig


def _hit_limit(high: float, prev_close: float, code: str) -> bool:
    """盘中是否摸过涨停价（用于统计 lu_20d，非收盘涨停判断）。"""
    if pd.isna(high) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return high >= TradingConfig.calc_limit_price(prev_close, code, is_up=True)


def parse_args():
    p = argparse.ArgumentParser(description="涨停池信号生成")
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top-n", type=int, default=30, help="每日最多信号数")
    p.add_argument("--out", type=str, default="data/signals/bt_signals_limit_up.csv")
    return p.parse_args()


def _infer_end_date(engine):
    last_two = pd.read_sql(
        text("SELECT trade_date, COUNT(*) AS n FROM stock_daily "
             "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2"), engine)
    if len(last_two) >= 2:
        n_today, n_yesterday = last_two.iloc[0]["n"], last_two.iloc[1]["n"]
        if n_today < n_yesterday * 0.8 and n_yesterday >= 2500:
            return str(last_two.iloc[1]["trade_date"])[:10]
    return str(last_two.iloc[0]["trade_date"])[:10]


def main():
    args = parse_args()
    engine = get_engine()
    t0 = time.time()

    end_date_str = args.end or _infer_end_date(engine)

    # ── 股票池（仅主板，上市≥120天）──
    min_list = (pd.Timestamp(args.start) - timedelta(days=120)).strftime("%Y-%m-%d")
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE is_st=FALSE AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"),
            conn, params={"ld": min_list})
    codes_df["code"] = codes_df["code"].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df["code"], codes_df["name"]))
    code_set = set(codes_df["code"].tolist())
    logger.info(f"股票池: {len(code_set)} 只")

    # ── 加载数据 ──
    pre_start = (pd.Timestamp(args.start) - timedelta(days=120)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, code_set, pre_start, end_date_str,
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])

    extra = load_mcap_data(engine, code_set, pre_start, end_date_str, use_proxy=True)
    if not extra.empty:
        extra["code"] = extra["code"].astype(str).str.zfill(6)
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])

    logger.info(f"日线: {len(daily)}行 | 市值: {len(extra)}行 ({time.time()-t0:.0f}s)")

    # ── 因子预计算（一次性）──
    logger.info("预计算因子...")
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    # is_lu=1: 盘中摸过涨停（high >= limit_price）
    # 回测引擎会用 is_at_limit_up(close, prev_close) 跳过收盘封死的
    daily["is_lu"] = daily.apply(
        lambda r: 1 if _hit_limit(r["high"], r["prev_close"], str(r["code"])) else 0,
        axis=1)
    daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    daily["lu_20d"] = daily.groupby("code")["is_lu"].transform(lambda x: x.rolling(20, min_periods=1).sum())

    # 预分组
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")} if not extra.empty else {}
    logger.info(f"因子完成 ({time.time()-t0:.0f}s)")

    # ── 逐日筛选 ──
    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates
                   if pd.Timestamp(args.start) <= d <= pd.Timestamp(end_date_str)]
    logger.info(f"交易日: {len(trade_dates)} 天")

    rows = []
    for today in trade_dates:
        td_df = daily_by_date.get(today)
        if td_df is None or td_df.empty:
            continue

        # 市值
        ex_td = extra_by_date.get(today)
        if ex_td is not None and not ex_td.empty:
            td_df = td_df.copy()
            td_df["mcap"] = ex_td.get("market_cap", np.nan)

        # 4条件筛选
        mask = (
            (td_df.get("mcap", pd.Series(np.nan, index=td_df.index)).between(30, 500)) &
            (td_df["close"].between(5, 100)) &
            (td_df["ma5"] > td_df["ma10"]) &
            (td_df["lu_20d"] >= 2) & (td_df["lu_20d"] <= 4) &
            (td_df["close"] > 0) & (td_df.index.isin(code_set))
        )
        sel = td_df[mask]
        if sel.empty:
            continue

        # 按今日涨幅排序
        sel = sel.copy()
        sel["ret_today"] = sel["close"] / sel["prev_close"] - 1
        sel = sel.nlargest(min(args.top_n, len(sel)), "ret_today")

        for rank, (code, r) in enumerate(sel.iterrows(), 1):
            rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "rank": rank,
                "code": str(code).zfill(6),
                "name": name_map.get(code, "?"),
                "score": round(float(r["ret_today"]) * 100, 2),
                "close": round(float(r["close"]), 2),
                "is_limit_up": bool(r["is_lu"] == 1),
                "is_limit_down": False,
            })

    engine.dispose()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.success(f"导出 {len(df)} 条 → {args.out} ({time.time()-t0:.0f}s)")
    logger.info(f"日期: {df['date'].min()} ~ {df['date'].max()}, {df['date'].nunique()}天, {df['code'].nunique()}只")


if __name__ == "__main__":
    main()
