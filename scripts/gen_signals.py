#!/usr/bin/env python
"""涨停策略信号预计算 —— 提取筛选管线，输出到CSV供 backtrader 使用。

用法:
    python scripts/gen_signals.py --start 2020-01-01 --top-n 5
    python scripts/gen_signals.py --start 2025-01-01 --top-n 1
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from strategies.limit_up.base import LimitUpParams, run_screening
from config.settings import TradingConfig

# ── 默认参数（E4增强版）──
DEFAULTS = dict(
    mcap_min=30, mcap_max=500, price_min=5, price_max=63,
    limit_up_lookback=20, limit_up_count=1,
    min_conditions=4, min_listed_days=120,
)


def parse_args():
    p = argparse.ArgumentParser(description="涨停策略信号生成")
    p.add_argument("--start", type=str, default="2025-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top-n", type=int, default=5)
    # 筛选
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--mcap-proxy", action="store_true")
    p.add_argument("--no-mcap", action="store_true")
    # 评分
    p.add_argument("--lu-score", action="store_true")
    p.add_argument("--lu-decay", action="store_true")
    p.add_argument("--lu-quality", action="store_true")
    p.add_argument("--lu-streak", action="store_true")
    p.add_argument("--no-5day-streak", action="store_true")
    p.add_argument("--streak-lookback", type=int, default=7)
    # 择时
    p.add_argument("--trend-filter", action="store_true")
    # 输出
    p.add_argument("--out", type=str, default="data/signals/bt_signals.csv")
    return p.parse_args()


def _infer_end_date(engine):
    """根据最近两日数据完整度推断最新可用交易日。"""
    last_two = pd.read_sql(
        text("SELECT trade_date, COUNT(*) AS n FROM stock_daily "
             "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2"),
        engine,
    )
    if len(last_two) >= 2 and last_two.iloc[1]["n"] > last_two.iloc[0]["n"] * 0.8:
        return str(last_two.iloc[1]["trade_date"])[:10]
    return str(last_two.iloc[0]["trade_date"])[:10]


def _load_name_map(engine, min_list_date):
    df = pd.read_sql(
        text("SELECT code, name FROM stock_basic WHERE is_st = FALSE AND list_date <= :d"),
        engine,
        params={"d": min_list_date},
    )
    return dict(zip(df["code"], df["name"])), set(df["code"])


def _load_csi1k_trend(engine, start, end):
    df = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code='000852' "
             "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"),
        engine,
        params={"s": start, "e": end},
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ma60"] = df["close"].rolling(60, min_periods=30).mean()
    return dict(zip(df["trade_date"], df["close"] > df["ma60"]))


def compute_streak_map(daily, lu_pct):
    """预计算连板数 {(code, date): max_consecutive_lu}，O(n) 实现。"""
    smap = {}
    for code, grp in daily.groupby("code"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        dates = grp["trade_date"].tolist()
        is_lu = [(r >= lu_pct) for r in grp["ret"]]
        n = len(dates)
        for i in range(n):
            start_i = max(0, i - 19)
            streak = 0
            max_streak = 0
            for j in range(start_i, i + 1):
                if is_lu[j]:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            smap[(code, dates[i])] = max_streak
    return smap


def _apply_scoring(code, lu_n, lb, today, args):
    """在基础涨停次数上叠加评分增强。"""
    score = float(lu_n)
    if args.lu_score:
        if lu_n <= 3:      score = lu_n + 1.0
        elif lu_n <= 5:    score = lu_n + 2.0
        elif lu_n == 6:    score = lu_n + 1.0
        else:              score = lu_n - 2.0
    if args.lu_decay:
        code_lu_dates = lb[(lb["code"] == code) & (lb["ret"] >= TradingConfig.LIMIT_UP_PCT)]["trade_date"]
        if len(code_lu_dates) > 0:
            weights = [max(0.1, 1.0 - (today - d).days / 20)
                       for d in code_lu_dates if today >= d]
            score = sum(weights) if weights else lu_n
    if args.lu_quality:
        code_lu = lb[(lb["code"] == code) & (lb["ret"] >= TradingConfig.LIMIT_UP_PCT)]
        if len(code_lu) > 0:
            q_scores = [min(1.5, max(0.3, (r["close"] / r["high"])
                           if r["high"] > 0 else 1.0)) for _, r in code_lu.iterrows()]
            score *= np.mean(q_scores) if q_scores else 1.0
    if args.lu_streak:
        code_rets = lb[lb["code"] == code].sort_values("trade_date")["ret"]
        streak = 0
        max_streak = 0
        for ret in code_rets:
            if ret >= TradingConfig.LIMIT_UP_PCT:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        if max_streak >= 2:
            score += (max_streak - 1) * 1.0
    return score


def screen_day(today, daily, extra_df, implied_shares, code_set, csi1k_up,
               streak_map, args):
    """在 today 执行筛选+评分，返回 [(code, score, close, is_limit_up, is_limit_down)]"""
    params = LimitUpParams(
        mcap_min=args.mcap_min, mcap_max=args.mcap_max,
        price_min=args.price_min, price_max=args.price_max,
        lu_pct=TradingConfig.LIMIT_UP_PCT,
        lu_lookback=args.limit_up_lookback, lu_count=args.limit_up_count,
        min_conditions=args.min_conditions,
    )

    # 基础筛选（去跌停，4条件）
    base_signals = run_screening(today, daily, extra_df, code_set, params)
    if not base_signals:
        return []

    # 如果启用 mcap proxy，重新覆盖市值条件结果
    if args.mcap_proxy and implied_shares:
        filtered = []
        for code, lu_n, close_p in base_signals:
            if code in implied_shares:
                proxy = implied_shares[code] * close_p
                if args.mcap_min <= proxy <= args.mcap_max:
                    filtered.append((code, lu_n, close_p))
        base_signals = filtered

    if args.no_mcap:
        # 不限制市值：这里不额外过滤，但 base.run_screening 已用市值条件
        pass

    # 趋势过滤：CSI1000 < MA60 → 空仓
    if args.trend_filter and csi1k_up is not None:
        if not csi1k_up.get(today, True):
            return []

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    td = daily_by_date[today]
    lookback_start = today - timedelta(days=args.limit_up_lookback + 5)
    lb = daily[(daily["trade_date"] >= lookback_start) & (daily["trade_date"] <= today)]

    passed = []
    for code, lu_n, close_p in base_signals:
        # 过滤 5 连板
        if args.no_5day_streak:
            max_streak_recent = max(
                (streak_map.get((code, today - timedelta(days=d)), 0)
                 for d in range(args.streak_lookback)), default=0)
            if max_streak_recent >= 5:
                continue

        score = _apply_scoring(code, lu_n, lb, today, args)

        # 涨跌停标记（给 backtrader 判断流动性）
        prev_rows = daily[(daily["code"] == code) & (daily["trade_date"] < today)].tail(1)
        prev_cp = prev_rows["close"].values[0] if not prev_rows.empty else None
        is_limit_up = prev_cp and prev_cp > 0 and (close_p / prev_cp - 1) >= TradingConfig.LIMIT_UP_PCT
        is_limit_down = prev_cp and prev_cp > 0 and (close_p / prev_cp - 1) <= TradingConfig.LIMIT_DOWN_PCT

        passed.append((code, score, close_p, is_limit_up, is_limit_down))

    # 按评分降序；同分按最近涨停距今升序
    lu_dates_map = {}
    for code, _, _, _, _ in passed:
        code_lu = lb[(lb['code'] == code) & (lb['ret'] >= TradingConfig.LIMIT_UP_PCT)]
        lu_dates_map[code] = (today - code_lu['trade_date'].max()).days if not code_lu.empty else 99
    passed.sort(key=lambda x: (x[1], -lu_dates_map.get(x[0], 99)), reverse=True)

    return passed[:args.top_n]


def main():
    args = parse_args()
    engine = get_engine()

    end_date_str = args.end or _infer_end_date(engine)
    min_list = pd.Timestamp(end_date_str) - timedelta(days=args.min_listed_days)
    name_map, code_set = _load_name_map(engine, min_list)

    pre_start = pd.Timestamp(args.start) - timedelta(days=args.limit_up_lookback + 30)

    logger.info(f"加载数据: {args.start} → {end_date_str}")
    daily = load_daily_data(engine, code_set, pre_start, end_date_str, cols=["open", "high", "close"])
    extra = load_mcap_data(engine, code_set, pre_start, end_date_str)
    logger.info(f"日线: {len(daily)} 行, {daily['code'].nunique()} 只")

    # 隐含股本（mcap proxy 用）
    implied_shares = {}
    if args.mcap_proxy:
        last_close = daily.sort_values("trade_date").groupby("code").last()["close"]
        for code in code_set:
            extra_code = extra[extra["code"] == code]
            if extra_code.empty:
                continue
            valid = extra_code[extra_code["market_cap"].notna() & (extra_code["market_cap"] > 0)]
            if valid.empty:
                continue
            latest = valid.sort_values("trade_date").iloc[-1]
            if latest["market_cap"] > 0 and code in last_close.index and last_close[code] > 0:
                implied_shares[code] = latest["market_cap"] / last_close[code] / 1e8

    csi1k_up = _load_csi1k_trend(engine, args.start, end_date_str) if args.trend_filter else None

    logger.info("计算连板数据...")
    streak_map = compute_streak_map(daily, TradingConfig.LIMIT_UP_PCT)

    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates
                   if pd.Timestamp(args.start) <= d <= pd.Timestamp(end_date_str)]
    logger.info(f"生成信号: {len(trade_dates)} 个交易日")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for today in trade_dates:
        signals = screen_day(today, daily, extra, implied_shares,
                             code_set, csi1k_up, streak_map, args)
        for i, (code, score, close_p, is_lu, is_ld) in enumerate(signals):
            rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "rank": i + 1,
                "code": str(code).zfill(6),
                "name": name_map.get(code, "?"),
                "score": round(score, 2),
                "close": round(close_p, 2),
                "is_limit_up": is_lu,
                "is_limit_down": is_ld,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.info(f"信号导出: {args.out} ({len(df)} 条)")
    engine.dispose()


if __name__ == "__main__":
    main()
