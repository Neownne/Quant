#!/usr/bin/env python
"""ML 增强涨停信号生成 —— 复用 gen_signals 筛选逻辑 + XGBoost 回归排序。

用法:
    python scripts/gen_signals_ml.py --start 2025-01-01 --top-n 5
    python scripts/gen_signals_ml.py --start 2020-01-01 --top-n 20 --model data/models/signal_quality_xgb_reg.pkl
"""

from __future__ import annotations

import argparse, os, sys, pickle, time
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from strategies.limit_up.base import LimitUpParams, run_screening
from config.settings import TradingConfig
from scripts.validate_factors import compute_factors_fast, compute_cross_sectional_factors

DEFAULTS = dict(
    mcap_min=30, mcap_max=500, price_min=5, price_max=63,
    limit_up_lookback=20, limit_up_count=1,
    min_conditions=4, min_listed_days=120,
)

MODEL_PATH = "data/models/signal_quality_xgb_reg.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="ML增强涨停信号生成")
    p.add_argument("--start", type=str, default="2025-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--candidate-multiplier", type=int, default=4,
                   help="候选池倍数，候选=top_n*multiplier")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--mcap-proxy", action="store_true")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--out", type=str, default="data/signals/bt_signals_ml.csv")
    p.add_argument("--no-ml", action="store_true", help="跳过ML过滤（调试用）")
    return p.parse_args()


def load_model(model_path):
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行: python scripts/train_signal_quality.py --start 2020-01-01")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    t0 = time.time()

    engine = get_engine()

    # ── 加载模型 ──
    if not args.no_ml:
        bundle = load_model(args.model)
        model = bundle["model"]
        selected_factors = bundle["selected_factors"]
        logger.info(f"模型已加载: {len(selected_factors)} 因子 (回归模型，按预测收益率排序取 top-N)")
    else:
        model = None
        selected_factors = []
        logger.info("跳过 ML 过滤（--no-ml）")

    # ── 加载数据（复用 gen_signals.py 逻辑）──
    from sqlalchemy import text
    min_list = pd.Timestamp(args.start) - timedelta(days=args.min_listed_days)
    with engine.connect() as conn:
        name_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE is_st = FALSE AND list_date <= :d"),
            conn, params={"d": min_list.strftime("%Y-%m-%d")},
        )
    name_map = dict(zip(name_df["code"], name_df["name"]))
    code_set = set(name_df["code"])

    pre_start = (pd.Timestamp(args.start) - timedelta(days=args.limit_up_lookback + 30)).strftime("%Y-%m-%d")
    end_date = args.end or _infer_end_date(engine)

    logger.info(f"加载数据: {args.start} → {end_date}")
    daily = load_daily_data(engine, code_set, pre_start, end_date,
                            cols=["open", "high", "low", "close", "volume", "amount", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    extra = load_mcap_data(engine, code_set, pre_start, end_date, use_proxy=args.mcap_proxy)
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])
    engine.dispose()

    logger.info(f"日线: {len(daily)} 行, {daily['code'].nunique()} 只 | 市值: {len(extra)} 行")

    # ── 加载行业 ──
    engine2 = get_engine()
    with engine2.connect() as conn:
        basic = pd.read_sql(
            text("SELECT code, industry FROM stock_basic WHERE is_st = FALSE"),
            conn,
        )
    industry_map = dict(zip(basic["code"].astype(str).str.zfill(6), basic["industry"]))
    daily["industry"] = daily["code"].map(industry_map).fillna("其他")
    engine2.dispose()

    # ── 预分组 ──
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")}

    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates
                   if pd.Timestamp(args.start) <= d <= pd.Timestamp(end_date)]
    logger.info(f"交易日: {len(trade_dates)}")

    # ── 计算因子（全量 daily，供 ML 用）──
    if not args.no_ml:
        logger.info("计算 ML 因子 ...")
        daily = compute_factors_fast(daily)
        daily = compute_cross_sectional_factors(daily)
        daily["key"] = daily["trade_date"].astype(str) + "_" + daily["code"].astype(str)

    # ── 逐日生成信号 ──
    params = LimitUpParams(
        mcap_min=args.mcap_min, mcap_max=args.mcap_max,
        price_min=args.price_min, price_max=args.price_max,
        lu_pct=TradingConfig.LIMIT_UP_PCT,
        lu_lookback=args.limit_up_lookback, lu_count=args.limit_up_count,
        min_conditions=args.min_conditions,
    )

    rows = []
    total_candidates = 0
    total_passed = 0
    days_with_signals = 0

    # 前一天收盘价映射（判断涨跌停用）
    sorted_dates = sorted(daily_by_date.keys())

    for today in trade_dates:
        td = daily_by_date.get(today)
        if td is None or td.empty:
            continue

        # 前收盘 map
        prev_close_map = {}
        prev_idx = sorted_dates.index(today) - 1 if today in sorted_dates else -1
        if prev_idx >= 0:
            prev_day_df = daily_by_date[sorted_dates[prev_idx]]
            prev_close_map = prev_day_df["close"].to_dict()

        # 基础筛选（候选池更大）
        base_signals = run_screening(today, daily, extra, code_set, params,
                                     daily_by_date=daily_by_date)
        if not base_signals:
            continue

        # 按涨停次数排序，选 top-N * multiplier
        lookback_start = today - timedelta(days=args.limit_up_lookback + 5)
        lb = daily[(daily["trade_date"] >= lookback_start) & (daily["trade_date"] <= today)]
        lu_counts = lb[lb["ret"] >= TradingConfig.LIMIT_UP_PCT].groupby("code").size()
        base_signals.sort(key=lambda x: (lu_counts.get(x[0], 0),
                                          -(today - lb[lb["code"] == x[0]]["trade_date"].max()).days
                                          if not lb[lb["code"] == x[0]].empty else -99),
                          reverse=True)

        candidate_pool = base_signals[: args.top_n * args.candidate_multiplier]
        total_candidates += len(candidate_pool)

        if args.no_ml:
            # 不加 ML 过滤，直接输出 top-N
            for i, (code, lu_n, close_p) in enumerate(candidate_pool[: args.top_n]):
                prev_c = prev_close_map.get(code)
                is_lu = prev_c and prev_c > 0 and TradingConfig.is_at_limit_up(
                    float(close_p), prev_c, str(code))
                is_ld = prev_c and prev_c > 0 and TradingConfig.is_at_limit_down(
                    float(close_p), prev_c, str(code))
                rows.append({
                    "date": today.strftime("%Y-%m-%d"), "rank": i + 1,
                    "code": str(code).zfill(6), "name": name_map.get(code, "?"),
                    "score": float(lu_n), "close": float(close_p),
                    "is_limit_up": is_lu, "is_limit_down": is_ld,
                    "ml_score": 0.0,
                })
            days_with_signals += 1
            total_passed += min(len(candidate_pool), args.top_n)
            continue

        # ── ML 打分排序（回归：按预测收益率降序取 top-N）──
        scored = []
        for code, lu_n, close_p in candidate_pool:
            key = today.strftime("%Y-%m-%d") + "_" + str(code).zfill(6)
            factor_row = daily[daily["key"] == key]

            if factor_row.empty:
                continue

            # 提取因子值
            X = factor_row[selected_factors].fillna(0).values
            if len(X) == 0:
                continue

            ml_score = float(model.predict(X)[0])  # 回归：预测收益率
            scored.append((code, lu_n, close_p, ml_score))

        # 按预测收益率降序排列，取 top-N
        scored.sort(key=lambda x: x[3], reverse=True)
        for code, lu_n, close_p, ml_score in scored[: args.top_n]:
            prev_c = prev_close_map.get(code)
            is_lu = prev_c and prev_c > 0 and TradingConfig.is_at_limit_up(
                float(close_p), prev_c, str(code))
            is_ld = prev_c and prev_c > 0 and TradingConfig.is_at_limit_down(
                float(close_p), prev_c, str(code))
            rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "rank": len([r for r in rows if r["date"] == today.strftime("%Y-%m-%d")]) + 1,
                "code": str(code).zfill(6),
                "name": name_map.get(code, "?"),
                "score": float(lu_n),
                "close": float(close_p),
                "is_limit_up": is_lu,
                "is_limit_down": is_ld,
                "ml_score": round(ml_score, 4),
            })
            total_passed += 1

        if any(r["date"] == today.strftime("%Y-%m-%d") for r in rows[-len(candidate_pool):]):
            days_with_signals += 1

    # ── 输出 ──
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        logger.warning("无信号！检查模型或候筛选条件")
        df_out = pd.DataFrame(columns=["date", "rank", "code", "name", "score", "close",
                                       "is_limit_up", "is_limit_down", "ml_score"])
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    logger.info(f"信号导出: {args.out} ({len(df_out)} 条, {days_with_signals} 天有信号)")
    logger.info(f"候选={total_candidates}, 通过={total_passed} ({total_passed/max(total_candidates,1):.1%})")
    logger.info(f"完成 ({elapsed:.0f}s)")


def _infer_end_date(engine):
    """推断最新可用交易日。"""
    from sqlalchemy import text
    last_two = pd.read_sql(
        text("SELECT trade_date, COUNT(*) AS n FROM stock_daily "
             "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2"),
        engine,
    )
    if len(last_two) >= 2:
        n_today, n_yesterday = last_two.iloc[0]["n"], last_two.iloc[1]["n"]
        if n_today < n_yesterday * 0.8 and n_yesterday >= 2500:
            return str(last_two.iloc[1]["trade_date"])[:10]
    return str(last_two.iloc[0]["trade_date"])[:10]


if __name__ == "__main__":
    main()
