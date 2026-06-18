#!/usr/bin/env python
"""涨停策略优化版 —— 只买涨停日高分信号。

回测发现: 涨停日买入胜率47% vs 回调日35%。大赢14:1。
规则: is_limit_up=True + score>8 (今日涨幅>8%)。

用法:
    python scripts/bt_limit_up_optimized.py --start 2020-01-01 --top-n 5 --label optimized
"""

from __future__ import annotations
import argparse, os, sys, csv, time
import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig

REBALANCE_DAYS = 5
OUT_DIR = "data/backtest_trades"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="涨停策略优化版")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", default="optimized")
    p.add_argument("--signals", default="data/signals/bt_signals_limit_up_full.csv")
    p.add_argument("--min-score", type=float, default=8.0, help="最低score(今日涨幅pct)")
    p.add_argument("--require-lu", action="store_true", default=True, help="要求当日涨停")
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--trailing-stop", type=float, default=0.15)
    p.add_argument("--min-hold-days", type=int, default=5)
    return p.parse_args()


def _load_name_map(engine, min_list_date):
    df = pd.read_sql(
        text("SELECT code, name FROM stock_basic WHERE is_st=FALSE AND list_date <= :d"),
        engine, params={"d": min_list_date.strftime("%Y-%m-%d")})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return dict(zip(df["code"], df["name"]))


def run_backtest(args):
    engine = get_engine()

    # ── 读取+过滤信号 ──
    sig_df = pd.read_csv(args.signals)
    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df["code"] = sig_df["code"].astype(str).str.zfill(6)

    total_raw = len(sig_df)
    # 过滤：只保留涨停日+高分
    if args.require_lu:
        sig_df = sig_df[sig_df["is_limit_up"] == True]
    sig_df = sig_df[sig_df["score"] >= args.min_score]
    logger.info(f"信号: {total_raw}条 → {len(sig_df)}条 (涨停日+score≥{args.min_score})")

    if sig_df.empty:
        logger.error("无信号")
        return

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else sig_df["date"].max()
    name_map = _load_name_map(engine, start - timedelta(days=252))
    codes = sig_df["code"].unique().tolist()
    for c in codes:
        if c not in name_map:
            name_map[c] = "?"

    # ── 加载日线 ──
    pre_start = (start - timedelta(days=90)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, codes, pre_start, end.strftime("%Y-%m-%d"),
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["ma20"] = daily.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    engine.dispose()

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    all_dates = sorted(d for d in daily_by_date if start <= d <= end)
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # 信号按日期分组
    sig_by_date = {}
    for d, g in sig_df.groupby("date"):
        sig_by_date[d] = g.sort_values("score", ascending=False)

    # ── 回测 ──
    cash = args.cash
    positions = {}  # code -> {entry_price, shares, entry_date, hold_days, current_price, peak_price}
    equity, trade_log = [], []
    trade_count = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    peak_value, frozen, frozen_days = cash, False, 0

    for i, td in enumerate(all_dates):
        td_df = daily_by_date[td]
        px_map = td_df["close"].to_dict()
        prev_map = {c: r["prev_close"] for c, r in td_df.iterrows() if pd.notna(r.get("prev_close"))}
        ma20_map = td_df["ma20"].to_dict()

        # 更新持仓
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos["current_price"], pos["hold_days"] = cur_px, pos["hold_days"] + 1
            if cur_px > pos.get("peak_price", 0):
                pos["peak_price"] = cur_px

        pos_val = sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in positions.values())
        total = cash + pos_val
        equity.append({"date": td.strftime("%Y-%m-%d"), "value": round(total, 2), "cash": round(cash, 2)})

        # 熔断
        if total > peak_value:
            peak_value = total
        dd = (peak_value - total) / peak_value if peak_value > 0 else 0
        if dd > 0.35 and not frozen:
            frozen, frozen_days = True, 0
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] DD {dd:.1%} 熔断")
        if frozen:
            frozen_days += 1
        if frozen and frozen_days > 60:
            frozen, peak_value = False, total
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 熔断解除")

        # ── 退出 ──
        for code, pos in list(positions.items()):
            cur_px, sell_reason = pos["current_price"], None

            # 破MA20
            ma20 = ma20_map.get(code)
            if ma20 and cur_px < ma20 and pos["hold_days"] > 5:
                sell_reason = "破MA20"
            # 移动止盈
            elif (pos["hold_days"] >= args.min_hold_days and
                  pos.get("peak_price", 0) > pos["entry_price"] * 1.05):
                if cur_px < pos["peak_price"] * (1 - args.trailing_stop):
                    sell_reason = "移动止盈"
            # 硬止损
            if cur_px < pos["entry_price"] * (1 - args.stop_loss):
                # 检查是否跌停无法卖出
                prev_c = prev_map.get(code)
                if not (prev_c and TradingConfig.is_at_limit_down(cur_px, prev_c, code)):
                    sell_reason = f"止损({args.stop_loss:.0%})"

            if sell_reason:
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_down(cur_px, prev_c, code):
                    continue
                proceeds = pos["shares"] * cur_px * NET_SELL
                cash += proceeds
                pnl = (cur_px / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": f"卖出({sell_reason})",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": round(pos["entry_price"], 2), "当前价/出场价": round(cur_px, 2),
                    "盈亏%": round(pnl, 2), "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "持有天数": pos["hold_days"],
                    "总资产": round(cash, 2), "当前现金": round(cash, 2),
                })
                trade_count += 1
                del positions[code]

        # ── 调仓买入 ──
        if i % REBALANCE_DAYS != 0 or frozen:
            continue
        if td not in sig_by_date:
            continue

        available = args.top_n - len(positions)
        if available <= 0:
            continue

        today_sigs = sig_by_date[td]
        held = set(positions.keys())
        today_sigs = today_sigs[~today_sigs["code"].isin(held)]

        alloc = cash * 0.95 / available
        bought = 0
        for _, s in today_sigs.iterrows():
            if bought >= available:
                break
            code, px = s["code"], px_map.get(s["code"])
            if not px or px <= 0:
                continue
            # 涨停封死的不买
            prev_c = prev_map.get(code)
            if prev_c and TradingConfig.is_at_limit_up(px, prev_c, code):
                continue

            sz = int(alloc / px / 100) * 100
            if sz < 100:
                continue
            cost = sz * px * BUY_COST
            if cost > cash * 0.95:
                sz = int(cash * 0.9 / px / 100) * 100
                if sz < 100:
                    continue
                cost = sz * px * BUY_COST

            cash -= cost
            positions[code] = {
                "code": code, "entry_price": px, "shares": sz,
                "entry_date": td.strftime("%Y-%m-%d"),
                "hold_days": 0, "current_price": px, "peak_price": px,
            }
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": round(px, 2), "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                "总资产": round(cash + sum(p["shares"] * px_map.get(c, p["entry_price"])
                                           for c, p in positions.items()), 2), "当前现金": round(cash, 2),
            })
            trade_count += 1
            bought += 1

    # ── 统计 ──
    eq_values = [e["value"] for e in equity]
    fv = eq_values[-1] if eq_values else cash
    ret_total = (fv / args.cash - 1)
    n_years = max((all_dates[-1] - all_dates[0]).days / 365.25, 0.5)
    ret_annual = (fv / args.cash) ** (1 / n_years) - 1

    if len(eq_values) > 2:
        dret = [(eq_values[i] - eq_values[i-1]) / max(eq_values[i-1], 1)
                for i in range(1, len(eq_values))]
        sharpe = float(np.mean(dret) / np.std(dret) * np.sqrt(252)) if np.std(dret) > 0 else 0
    else:
        sharpe = 0

    peak = eq_values[0]
    mdd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        mdd = max(mdd, (peak - v) / peak)

    sells_list = [t for t in trade_log if "卖出" in str(t.get("操作", ""))]
    wins = [t for t in sells_list if float(str(t.get("盈亏%", "0")).replace("nan", "0") or 0) > 0]
    win_rate = len(wins) / max(len(sells_list), 1)

    # 按年统计
    yr_returns = {}
    for e in equity:
        yr = e["date"][:4]
        if yr not in yr_returns:
            yr_returns[yr] = {"start": e["value"]}
        yr_returns[yr]["end"] = e["value"]

    csv_path = f"{OUT_DIR}/trades_{args.label}_{args.top_n}_{args.start.replace('-','')}_{end.strftime('%Y%m%d')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                     "盈亏%", "股数", "入场日期", "持有天数", "总资产", "当前现金"])
        for t in trade_log:
            w.writerow([t.get("日期", ""), t.get("操作", ""), t.get("股票代码", ""),
                        t.get("股票名称", ""), t.get("入场价", ""), t.get("当前价/出场价", ""),
                        t.get("盈亏%", ""), t.get("股数", ""), t.get("入场日期", ""),
                        t.get("持有天数", ""), t.get("总资产", ""), t.get("当前现金", "")])

    buys = len([t for t in trade_log if t.get("操作") == "买入"])
    print(f"\n{'='*60}")
    print(f"  涨停优化策略 Top-{args.top_n} [涨停日+score≥{args.min_score}]")
    print(f"  {args.start} → {end.strftime('%Y-%m-%d')} | 本金 {args.cash:,.0f}")
    print(f"  终值 {fv:,.0f} | 收益 {ret_total:+.1%} | 年化 {ret_annual:+.1%}")
    print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {mdd:.1%}")
    print(f"  交易 {trade_count} 笔 | 胜率 {win_rate:.1%} | 买入 {buys} | 卖出 {len(sells_list)}")
    for yr, vals in sorted(yr_returns.items()):
        yr_ret = vals["end"] / vals["start"] - 1
        print(f"    {yr}: {yr_ret:+.1%}")
    print(f"  交割单: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_backtest(parse_args())
