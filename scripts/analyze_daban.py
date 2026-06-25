#!/usr/bin/env python
"""打板策略分市场状态分析 —— 无未来函数。

核心逻辑:
  T 日盘中: 触板（high ≥ 涨停价 且 open < 涨停价）→ 候选，排除一字板
  成交概率: base(前5日均换手率) × market_adj × streak_adj（全用T日前数据）
  卖出: 4种持有期对比

输出:
  1. 按市场状态 × 持有期 × 变体的收益矩阵
  2. 各变体胜率/盈亏比/最大回撤
  3. 成交概率调整后的期望收益

用法:
    python scripts/analyze_daban.py --start 2020-01-01
    python scripts/analyze_daban.py --start 2020-01-01 --end 2025-12-31
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
from factors.regime import detect_regime

# ── 涨停阈值（板别感知）──
_LIMIT_MULT = {"688": 1.19899, "8": 1.29899, "4": 1.29899, "300": 1.19899, "301": 1.19899}
_DEFAULT_MULT = 1.09899

OUT_DIR = "data/arsenal"


def _calc_limit_price(prev_close: float, code: str) -> float:
    """涨停价 = round(prev_close × multiplier, 4)"""
    mult = _DEFAULT_MULT
    for prefix, m in _LIMIT_MULT.items():
        if str(code).startswith(prefix):
            mult = m
            break
    return round(prev_close * mult, 4)


def _compute_fill_rate(avg_turnover_5d: float, regime: str, streak: int) -> float:
    """成交概率模型（全用T日前数据，无未来函数）。

    base:    前5日均换手率 → 基础成交率
    market:  市场状态调整
    streak:  连板数调整
    """
    # base
    if pd.isna(avg_turnover_5d) or avg_turnover_5d <= 0:
        base = 0.10
    elif avg_turnover_5d >= 10:
        base = 0.80
    elif avg_turnover_5d >= 5:
        base = 0.50
    elif avg_turnover_5d >= 2:
        base = 0.25
    else:
        base = 0.10

    # market
    market_adj_map = {
        "strong_bull": 1.2, "weak_bull": 1.0,
        "sideways": 0.9, "slow_bear": 0.8, "fast_bear": 0.7,
    }
    market_adj = market_adj_map.get(regime, 0.9)

    # streak（含当日，首板=1，2连板=2...）
    if streak <= 1:
        streak_adj = 1.0
    elif streak == 2:
        streak_adj = 0.8
    else:
        streak_adj = 0.6

    return min(base * market_adj * streak_adj, 1.0)


def _mcap_tier(mcap):
    """市值分档。"""
    if pd.isna(mcap):
        return "未知"
    if mcap < 30:
        return "<30亿"
    if mcap <= 100:
        return "30-100亿"
    if mcap <= 300:
        return "100-300亿"
    if mcap <= 500:
        return "300-500亿"
    return ">500亿"


def _report_section(title, rows, col_widths):
    """格式化输出表格截面。"""
    print(f"\n### {title}")
    header = "".join(f"{h:>{w}s}" for h, w in col_widths)
    sep = "-" * sum(w for _, w in col_widths)
    print(header)
    print(sep)
    for row in rows:
        line = "".join(f"{str(v):>{w}s}" for v, (_, w) in zip(row, col_widths))
        print(line)


def main():
    parser = argparse.ArgumentParser(description="打板策略分市场状态分析")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    engine = get_engine()
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════
    # 1. 加载指数 → 市场状态
    # ═══════════════════════════════════════════════════════════
    logger.info("加载指数数据...")
    index_df = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code = '000001' ORDER BY trade_date"),
        engine,
    )
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    regime_df = detect_regime(index_df)
    regime_map = dict(zip(regime_df["trade_date"], regime_df["regime"]))

    if args.end is None:
        args.end = str(index_df["trade_date"].max())[:10]

    logger.info(f"日期范围: {args.start} ~ {args.end}")

    # ═══════════════════════════════════════════════════════════
    # 2. 股票池（仅主板）
    # ═══════════════════════════════════════════════════════════
    min_list = pd.Timestamp(args.start) - timedelta(days=252)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text(
                "SELECT code, name FROM stock_basic "
                "WHERE is_st=FALSE AND list_date <= :ld "
                "AND code !~ '^(300|301|688|[48])'"
            ),
            conn,
            params={"ld": pd.Timestamp(args.end).strftime("%Y-%m-%d")},
        )
    codes_df["code"] = codes_df["code"].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df["code"], codes_df["name"]))
    code_set = set(codes_df["code"].tolist())
    logger.info(f"主板股票池: {len(code_set)} 只")

    # ═══════════════════════════════════════════════════════════
    # 3. 加载日线
    # ═══════════════════════════════════════════════════════════
    pre_start = (pd.Timestamp(args.start) - timedelta(days=120)).strftime("%Y-%m-%d")
    daily = load_daily_data(
        engine, code_set, pre_start, args.end,
        cols=["open", "high", "low", "close", "volume", "turnover"],
    )
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════
    # 4. 加载市值
    # ═══════════════════════════════════════════════════════════
    extra = load_mcap_data(engine, code_set, pre_start, args.end, use_proxy=True)
    if not extra.empty:
        extra["code"] = extra["code"].astype(str).str.zfill(6)
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])

    # ═══════════════════════════════════════════════════════════
    # 5. 加载概念板块（板块效应变体用）
    # ═══════════════════════════════════════════════════════════
    logger.info("加载概念板块归属...")
    concept_df = pd.read_sql(
        text("SELECT stock_code, board_code FROM concept_stock"), engine
    )
    concept_df["stock_code"] = concept_df["stock_code"].astype(str).str.zfill(6)
    stock_concepts = concept_df.groupby("stock_code")["board_code"].apply(set).to_dict()
    logger.info(f"  概念覆盖: {len(stock_concepts)} 只股票")

    engine.dispose()

    # ═══════════════════════════════════════════════════════════
    # 6. 预计算因子（全部向量化）
    # ═══════════════════════════════════════════════════════════
    logger.info("预计算因子...")
    t_pre = time.time()

    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["ret"] = daily.groupby("code")["close"].pct_change()

    # 涨停价（板别感知）
    daily["limit_price"] = daily.apply(
        lambda r: _calc_limit_price(r["prev_close"], str(r["code"]))
        if pd.notna(r["prev_close"]) and r["prev_close"] > 0
        else np.nan,
        axis=1,
    )

    # 涨停标记: 收盘 ≥ 涨停价 或 盘中摸过涨停价
    daily["is_lu"] = (
        daily["high"].notna()
        & daily["limit_price"].notna()
        & (daily["high"] >= daily["limit_price"])
    ).astype(int)

    # 前5日均换手率（shift 确保只用T-1及之前）
    daily["turnover_5d"] = daily.groupby("code")["turnover"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )

    # 前5日均量
    daily["vol_5d"] = daily.groupby("code")["volume"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    daily["vol_ratio"] = daily["volume"] / daily["vol_5d"]

    # 连板数（含当日，首板=1）
    daily["streak"] = 0
    for code, grp in daily.groupby("code"):
        grp = grp.sort_values("trade_date")
        is_lu_arr = grp["is_lu"].values
        streak_arr = np.zeros(len(grp), dtype=int)
        cnt = 0
        for i, v in enumerate(is_lu_arr):
            if v:
                cnt += 1
            else:
                cnt = 0
            streak_arr[i] = cnt
        daily.loc[grp.index, "streak"] = streak_arr

    logger.info(f"  因子完成 ({time.time() - t_pre:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 7. 构建快速查找结构
    # ═══════════════════════════════════════════════════════════
    all_dates = sorted(daily["trade_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # code → date → Series 的嵌套字典（只存需要的列）
    need_cols = ["open", "high", "low", "close", "limit_price", "is_lu",
                 "turnover_5d", "vol_ratio", "streak", "prev_close"]
    fast = {}
    for code, grp in daily.groupby("code"):
        grp_sorted = grp.sort_values("trade_date")
        fast[code] = {
            row["trade_date"]: row
            for _, row in grp_sorted[["trade_date"] + need_cols].iterrows()
        }

    # 市值快查: date → code → mcap
    mcap_fast = {}
    if not extra.empty:
        for _, row in extra.iterrows():
            d = row["trade_date"]
            if d not in mcap_fast:
                mcap_fast[d] = {}
            mcap_fast[d][row["code"]] = row.get("market_cap", np.nan)

    logger.info(f"  快查结构构建完成 ({time.time() - t_pre:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # 8. 逐日扫描触板候选
    # ═══════════════════════════════════════════════════════════
    logger.info("扫描触板候选（盘中触板 = open<涨停价 AND high≥涨停价）...")
    t_scan = time.time()

    trade_dates = [
        d for d in all_dates if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)
    ]

    candidates = []

    for di, today in enumerate(trade_dates):
        if di % 500 == 0 and di > 0:
            logger.info(f"  进度: {di}/{len(trade_dates)} 天, 已找到 {len(candidates)} 候选")

        regime = regime_map.get(today, "sideways")
        today_idx = date_to_idx.get(today, -1)

        # 预取后续5个交易日日期
        next_dates = {n: all_dates[today_idx + n] if today_idx + n < len(all_dates) else None
                      for n in [1, 2, 3, 4, 5]}

        # 当天概念涨停计数
        board_lu_count = {}
        today_codes_with_data = set()
        for c, cdata in fast.items():
            if today in cdata:
                today_codes_with_data.add(c)
                r = cdata[today]
                if r["is_lu"]:
                    for b in stock_concepts.get(c, set()):
                        board_lu_count[b] = board_lu_count.get(b, 0) + 1

        # 扫描每只股票
        for code in today_codes_with_data:
            r = fast[code][today]

            # 必须有有效涨停价
            if pd.isna(r["limit_price"]) or r["limit_price"] <= 0:
                continue

            # 排除一字板: open ≥ 涨停价 → 开盘就封死，买不到
            if r["open"] >= r["limit_price"]:
                continue

            # 盘中触板: high ≥ 涨停价
            if r["high"] < r["limit_price"]:
                continue

            # ── 候选成立！──
            prev_streak_before_today = r["streak"] - 1 if r["is_lu"] else r["streak"]
            current_streak = r["streak"]

            # 成交概率（用 T 日之前数据）
            fill_rate = _compute_fill_rate(
                r["turnover_5d"], regime, current_streak
            )

            # 封板/炸板（收盘可见，但只用于事后分析）
            is_sealed = r["close"] >= r["limit_price"]

            # 市值
            mcap = mcap_fast.get(today, {}).get(code, np.nan)
            mcap_tier = _mcap_tier(mcap)

            # 板块效应
            stock_boards = stock_concepts.get(code, set())
            max_board_lu = max(
                (board_lu_count.get(b, 0) for b in stock_boards), default=0
            )
            has_sector_effect = max_board_lu >= 3

            # 放量/缩量
            vol_ratio_val = r["vol_ratio"]
            if pd.isna(vol_ratio_val):
                vol_type = "未知"
            elif vol_ratio_val > 1.5:
                vol_type = "放量"
            else:
                vol_type = "缩量"

            # ── 计算4种持有期收益 ──
            t_limit = r["limit_price"]
            rets = {}

            # Helper: 获取 code 在 date 的数据
            def _get_day(c, d):
                if d is None:
                    return None
                return fast.get(c, {}).get(d)

            # 持有期 A: T+1 开盘卖
            day1 = _get_day(code, next_dates[1])
            ret_a = None
            if day1 is not None and not pd.isna(day1["open"]):
                ret_a = (day1["open"] - t_limit) / t_limit

            # 持有期 B: T+1 收盘卖（封板则续持到不封板那天）
            ret_b = None
            if day1 is not None and not pd.isna(day1["close"]):
                # 找第一个不封板的交易日
                sell_day = day1
                sell_n = 1
                for n in [1, 2, 3, 4, 5]:
                    dn = _get_day(code, next_dates[n])
                    if dn is None:
                        break
                    sell_day = dn
                    sell_n = n
                    # 封板 = close ≥ limit_price
                    limit_n = dn.get("limit_price", np.nan)
                    if pd.isna(limit_n):
                        break
                    if dn["close"] < limit_n:
                        # 不封板，卖在收盘
                        break
                ret_b = (sell_day["close"] - t_limit) / t_limit

            # 持有期 C: 持有3天，止损 -3%
            ret_c = None
            stop_c = -0.03
            found_c = False
            for n in [1, 2, 3]:
                dn = _get_day(code, next_dates[n])
                if dn is None:
                    continue
                # 日内最低价触发止损
                ret_low = (dn["low"] - t_limit) / t_limit
                if ret_low <= stop_c:
                    ret_c = stop_c
                    found_c = True
                    break
                if n == 3:
                    ret_c = (dn["close"] - t_limit) / t_limit
                    found_c = True
            if not found_c:
                ret_c = ret_b  # fallback

            # 持有期 D: 持有5天，移动止盈 5%，硬止损 -5%
            ret_d = None
            trail_pct = 0.05
            hard_stop = -0.05
            found_d = False
            trailing_high = t_limit
            for n in [1, 2, 3, 4, 5]:
                dn = _get_day(code, next_dates[n])
                if dn is None:
                    continue
                trailing_high = max(trailing_high, dn["high"])
                trail_stop = trailing_high * (1 - trail_pct)

                # 移动止盈触发
                if dn["low"] <= trail_stop:
                    ret_d = max((trail_stop - t_limit) / t_limit, hard_stop)
                    found_d = True
                    break
                # 硬止损触发
                ret_low = (dn["low"] - t_limit) / t_limit
                if ret_low <= hard_stop:
                    ret_d = hard_stop
                    found_d = True
                    break
                # 最后一天
                if n == 5:
                    ret_d = (dn["close"] - t_limit) / t_limit
                    found_d = True
            if not found_d:
                ret_d = ret_b

            candidates.append({
                "date": today,
                "code": code,
                "name": name_map.get(code, "?"),
                "regime": regime,
                "mcap": mcap,
                "mcap_tier": mcap_tier,
                "streak": current_streak,
                "is_first_board": current_streak == 1,
                "is_sealed": is_sealed,
                "fill_rate": round(fill_rate, 3),
                "turnover_5d": r["turnover_5d"] if pd.notna(r.get("turnover_5d", np.nan)) else np.nan,
                "vol_ratio": round(vol_ratio_val, 2) if pd.notna(vol_ratio_val) else np.nan,
                "vol_type": vol_type,
                "has_sector_effect": has_sector_effect,
                "max_board_lu": max_board_lu,
                "ret_A": ret_a,
                "ret_B": ret_b,
                "ret_C": ret_c,
                "ret_D": ret_d,
            })

    df = pd.DataFrame(candidates)
    logger.info(f"  扫描完成: {len(df)} 候选 ({time.time() - t_scan:.0f}s)")

    if df.empty:
        logger.error("无触板候选！检查数据范围或涨停阈值。")
        return

    # ═══════════════════════════════════════════════════════════
    # 9. 分析报告
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  打板策略分市场状态分析报告")
    print(f"  日期: {args.start} ~ {args.end}")
    print(f"  触板候选: {len(df)} 个 | 个股: {df['code'].nunique()} 只")
    print(f"  日均触板: {len(df) / df['date'].nunique():.1f} 个")
    print("=" * 90)

    regime_order = ["strong_bull", "weak_bull", "sideways", "slow_bear", "fast_bear"]
    regime_labels = {
        "strong_bull": "强牛", "weak_bull": "弱牛",
        "sideways": "震荡", "slow_bear": "慢熊", "fast_bear": "快熊",
    }
    hp_labels = {
        "A": "T+1开盘卖", "B": "T+1收盘/续持",
        "C": "持有3天(-3%止损)", "D": "持有5天(移动止盈5%)",
    }

    # ── 9a. 按市场状态 × 持有期 ──
    print("\n" + "─" * 90)
    print("  一、按市场状态 × 持有期")
    print("─" * 90)

    for hp in ["A", "B", "C", "D"]:
        col = f"ret_{hp}"
        print(f"\n【持有期 {hp}: {hp_labels[hp]}】")
        hdr = f"{'市场状态':<12s} {'信号':>6s} {'胜率':>7s} {'均收益':>8s} {'中位':>8s} {'盈亏比':>7s} {'成交后均':>9s} {'最大盈':>7s} {'最大亏':>7s}"
        print(hdr)
        print("-" * len(hdr))

        for reg in regime_order:
            sub = df[(df["regime"] == reg) & df[col].notna()]
            if len(sub) < 5:
                continue
            rets = sub[col]
            win_rate = (rets > 0).mean()
            avg_ret = rets.mean()
            med_ret = rets.median()
            wins = rets[rets > 0]
            losses = rets[rets < 0]
            pr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0
            adj_ret = (rets * sub["fill_rate"]).mean()

            print(
                f"{regime_labels.get(reg, reg):<12s} {len(sub):>6d} {win_rate:>6.1%} "
                f"{avg_ret:>7.2%} {med_ret:>7.2%} {pr:>6.1f} {adj_ret:>8.2%} "
                f"{rets.max():>6.2%} {rets.min():>6.2%}"
            )

    # ── 9b. 持有期对比汇总 ──
    print("\n" + "─" * 90)
    print("  二、持有期对比汇总（全市场状态）")
    print("─" * 90)
    hdr = f"{'持有期':<20s} {'样本':>6s} {'胜率':>7s} {'均收益':>8s} {'中位':>8s} {'盈亏比':>7s} {'成交后均':>9s} {'日均信号':>8s}"
    print(hdr)
    print("-" * len(hdr))

    for hp in ["A", "B", "C", "D"]:
        col = f"ret_{hp}"
        sub = df[df[col].notna()]
        if len(sub) == 0:
            continue
        rets = sub[col]
        win_rate = (rets > 0).mean()
        avg_ret = rets.mean()
        med_ret = rets.median()
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        pr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0
        adj_ret = (rets * sub["fill_rate"]).mean()

        print(
            f"{hp_labels[hp]:<20s} {len(sub):>6d} {win_rate:>6.1%} "
            f"{avg_ret:>7.2%} {med_ret:>7.2%} {pr:>6.1f} {adj_ret:>8.2%} "
            f"{len(sub)/df['date'].nunique():>7.1f}"
        )

    # ── 9c. 变体维度对比（持有期A）──
    print("\n" + "─" * 90)
    print("  三、变体维度对比（持有期A: T+1开盘卖）")
    print("─" * 90)

    vdf = df[df["ret_A"].notna()].copy()

    def print_variant(label, sub, key_col="ret_A"):
        """打印单个变体的统计。"""
        rets = sub[key_col]
        if len(sub) < 5:
            print(f"  {label:<25s} 样本不足({len(sub)})")
            return
        win_rate = (rets > 0).mean()
        avg_ret = rets.mean()
        med_ret = rets.median()
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        pr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0
        adj_ret = (rets * sub["fill_rate"]).mean()
        print(
            f"  {label:<25s} {len(sub):>6d} {win_rate:>6.1%} {avg_ret:>7.2%} "
            f"{med_ret:>7.2%} {pr:>6.1f} {adj_ret:>8.2%} {rets.max():>6.2%} {rets.min():>6.2%}"
        )

    # 封板 vs 炸板
    print("\n【封板 vs 炸板】")
    for sealed, label in [(True, "封板成功"), (False, "炸板（未封死）")]:
        print_variant(label, vdf[vdf["is_sealed"] == sealed])

    # 首板 vs 连板
    print("\n【首板 vs 连板】")
    for first, label in [(True, "首板"), (False, "连板(≥2)")]:
        print_variant(label, vdf[vdf["is_first_board"] == first])

    # 连板细分
    print("\n【连板数细分】")
    for s in [1, 2, 3, 4]:
        sub = vdf[vdf["streak"] == s]
        label = f"  {s}连板" if s > 1 else f"  {s}板(首板)"
        print_variant(label, sub)

    # 市值分档
    print("\n【市值分档】")
    for tier in ["<30亿", "30-100亿", "100-300亿", "300-500亿", ">500亿"]:
        print_variant(tier, vdf[vdf["mcap_tier"] == tier])

    # 板块效应
    print("\n【板块效应】")
    for has_eff, label in [(True, "同概念≥3只涨停"), (False, "无板块效应")]:
        print_variant(label, vdf[vdf["has_sector_effect"] == has_eff])

    # 量比
    print("\n【放量 vs 缩量】")
    for vt, label in [("放量", "放量板(量比>1.5)"), ("缩量", "缩量板(量比≤1.5)")]:
        print_variant(label, vdf[vdf["vol_type"] == vt])

    # ── 9d. 市场状态 × 首板/连板 × 封板/炸板 ──
    print("\n" + "─" * 90)
    print("  四、市场状态 × 首板/连板（持有期A）")
    print("─" * 90)

    for reg in regime_order:
        sub_reg = vdf[vdf["regime"] == reg]
        if len(sub_reg) < 10:
            continue
        print(f"\n  [{regime_labels.get(reg, reg)}] 总信号: {len(sub_reg)}")
        hdr_line = f"  {'':<16s} {'信号':>6s} {'胜率':>7s} {'均收益':>8s} {'中位':>8s} {'盈亏比':>7s} {'成交后均':>9s}"
        print(hdr_line)
        for is_first in [True, False]:
            sub = sub_reg[sub_reg["is_first_board"] == is_first]
            if len(sub) < 5:
                continue
            rets = sub["ret_A"]
            wr = (rets > 0).mean()
            avg = rets.mean()
            med = rets.median()
            wins = rets[rets > 0]
            losses = rets[rets < 0]
            pr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0
            adj = (rets * sub["fill_rate"]).mean()
            label = "  首板" if is_first else "  连板(≥2)"
            print(
                f"  {label:<16s} {len(sub):>6d} {wr:>6.1%} {avg:>7.2%} "
                f"{med:>7.2%} {pr:>6.1f} {adj:>8.2%}"
            )

    # ── 9e. 封板率分析 ──
    print("\n" + "─" * 90)
    print("  五、封板率分析")
    print("─" * 90)
    print(f"\n  {'维度':<20s} {'触板数':>7s} {'封板数':>7s} {'封板率':>7s}")
    print(f"  {'-'*41}")
    total_touch = len(df)
    total_sealed = df["is_sealed"].sum()
    print(f"  {'全部':<20s} {total_touch:>7d} {total_sealed:>7d} {total_sealed/total_touch:>6.1%}")

    for reg in regime_order:
        sub = df[df["regime"] == reg]
        if len(sub) < 5:
            continue
        se = sub["is_sealed"].sum()
        print(f"  {regime_labels.get(reg, reg):<20s} {len(sub):>7d} {se:>7d} {se/len(sub):>6.1%}")

    print()
    for first, label in [(True, "首板"), (False, "连板")]:
        sub = df[df["is_first_board"] == first]
        se = sub["is_sealed"].sum()
        print(f"  {label:<20s} {len(sub):>7d} {se:>7d} {se/len(sub):>6.1%}")

    # ── 9f. 年度统计 ──
    print("\n" + "─" * 90)
    print("  六、年度统计（持有期A）")
    print("─" * 90)

    vdf["year"] = pd.to_datetime(vdf["date"]).dt.year
    hdr = f"  {'年份':<8s} {'信号':>6s} {'胜率':>7s} {'均收益':>8s} {'中位':>8s} {'盈亏比':>7s} {'成交后均':>9s} {'封板率':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for yr in sorted(vdf["year"].unique()):
        sub = vdf[vdf["year"] == yr]
        if len(sub) < 5:
            continue
        rets = sub["ret_A"]
        wr = (rets > 0).mean()
        avg = rets.mean()
        med = rets.median()
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        pr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and len(wins) > 0 else 0
        adj = (rets * sub["fill_rate"]).mean()
        sr = sub["is_sealed"].mean()
        print(
            f"  {yr:<8d} {len(sub):>6d} {wr:>6.1%} {avg:>7.2%} "
            f"{med:>7.2%} {pr:>6.1f} {adj:>8.2%} {sr:>6.1%}"
        )

    # ═══════════════════════════════════════════════════════════
    # 10. 保存结果
    # ═══════════════════════════════════════════════════════════
    os.makedirs(OUT_DIR, exist_ok=True)
    date_tag = pd.Timestamp.now().strftime("%Y%m%d")

    # CSV 明细
    csv_path = f"{OUT_DIR}/daban_candidates_{date_tag}.csv"
    save_cols = [
        "date", "code", "name", "regime", "mcap", "mcap_tier",
        "streak", "is_first_board", "is_sealed", "fill_rate",
        "turnover_5d", "vol_ratio", "vol_type",
        "has_sector_effect", "max_board_lu",
        "ret_A", "ret_B", "ret_C", "ret_D",
    ]
    df_save = df[save_cols].copy()
    df_save["date"] = df_save["date"].astype(str)
    df_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.success(f"明细保存: {csv_path} ({len(df_save)} 行)")

    # JSON 摘要
    summary = {
        "date_range": f"{args.start} ~ {args.end}",
        "total_candidates": len(df),
        "unique_stocks": int(df["code"].nunique()),
        "trading_days": int(df["date"].nunique()),
        "avg_per_day": round(len(df) / df["date"].nunique(), 1),
        "regime_distribution": df["regime"].value_counts().to_dict(),
    }

    # 持有期对比
    for hp in ["A", "B", "C", "D"]:
        col = f"ret_{hp}"
        sub = df[df[col].notna()]
        if len(sub) == 0:
            continue
        rets = sub[col]
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        summary[f"hp_{hp}"] = {
            "label": hp_labels[hp],
            "n": len(sub),
            "win_rate": round(float((rets > 0).mean()), 4),
            "avg_return": round(float(rets.mean()), 4),
            "median_return": round(float(rets.median()), 4),
            "profit_factor": round(float(wins.mean() / abs(losses.mean())), 2) if len(losses) > 0 and len(wins) > 0 else None,
            "adj_return": round(float((rets * sub["fill_rate"]).mean()), 4),
        }

    json_path = f"{OUT_DIR}/daban_summary_{date_tag}.json"
    import json
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.success(f"摘要保存: {json_path}")

    print(f"\n  耗时: {time.time() - t0:.0f}s")
    print(f"  明细: {csv_path}")
    print(f"  摘要: {json_path}")
    print("\n  ⚠ 注意: 收益以涨停价买入计算，未扣除交易成本。")
    print("  '成交后均' = 收益 × 成交概率，粗略估算实际期望收益。")


if __name__ == "__main__":
    main()
