#!/usr/bin/env python
"""小市值反转策略 —— 最小市值 + 近期超跌 + 周频调仓。

核心逻辑：
  1. 每周从全 A 股市值最小的 100 只中选 10 只近期跌最多的
  2. 过滤 ST、次新（<252天）、科创板
  3. 等权持仓，周频调仓，-8% 个股止损，-25% 组合熔断

用法:
    python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10
"""
from __future__ import annotations

import argparse, os, sys, json, csv, time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from config.settings import TradingConfig

# 默认参数
DEFAULTS = {
    "mcap_pool": 100,          # 候选池：市值最小的 N 只
    "lookback": 20,            # 回看窗口（天）
    "top_n": 10,               # 持仓数
    "rebalance": 5,            # 调仓频率（交易日）
    "stop_loss": 0.08,         # 个股止损
    "portfolio_dd_stop": 0.25, # 组合回撤熔断
    "min_listed_days": 252,    # 上市满一年
    "exclude_star": True,      # 排除科创板（688）
    "exclude_gem": True,       # 排除创业板（300/301）
    "exclude_bse": True,       # 排除北交所（4/8）
}


def load_universe(engine, trade_date, min_listed_days=252):
    """加载候选池：非 ST、上市满 N 天、仅主板。"""
    min_list = trade_date - timedelta(days=min_listed_days)
    df = pd.read_sql(text("""
        SELECT code, name FROM stock_basic
        WHERE is_st = FALSE AND list_date <= :ld
    """), engine, params={"ld": min_list.strftime("%Y-%m-%d")})
    codes = df["code"].tolist()
    name_map = dict(zip(df["code"], df["name"]))
    # 只保留主板：排除科创板(688)、创业板(300/301)、北交所(4/8)
    EXCLUDE_PREFIXES = []
    if DEFAULTS["exclude_star"]:
        EXCLUDE_PREFIXES.append("688")
    if DEFAULTS["exclude_gem"]:
        EXCLUDE_PREFIXES.extend(["300", "301"])
    if DEFAULTS["exclude_bse"]:
        EXCLUDE_PREFIXES.extend(["4", "8"])
    if EXCLUDE_PREFIXES:
        codes = [c for c in codes if not str(c).startswith(tuple(EXCLUDE_PREFIXES))]
    return codes, name_map


def run_backtest(args):
    engine = get_engine()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today())

    # 加载股票池
    all_codes, name_map = load_universe(engine, end)

    # 加载数据（从 start 前 60 天开始，给 lookback 预热）
    pre_start = (start - timedelta(days=args.lookback + 30)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, all_codes, pre_start, end.strftime("%Y-%m-%d"),
                            cols=["open", "high", "low", "close", "volume"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    extra = load_mcap_data(engine, all_codes, pre_start, end.strftime("%Y-%m-%d"), use_proxy=True)
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])
    engine.dispose()

    # 按日期分组
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")}

    all_dates = sorted(d for d in daily_by_date if start <= d <= end)
    logger.info(f"交易日: {len(all_dates)} | 股票池: {len(all_codes)} 只")

    # ── 回测主循环 ──
    cash = args.cash
    positions = {}              # {code: {"entry_price": px, "shares": sz, "entry_date": dt}}
    equity = []
    trade_log = []
    trade_count = 0
    frozen = False              # 组合熔断标记

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    peak_value = cash

    for i, td in enumerate(all_dates):
        td_df = daily_by_date.get(td)
        if td_df is None or td_df.empty:
            continue
        px_map = td_df["close"].to_dict()

        # ── 估值：计算当前持仓市值 ──
        pos_val = 0.0
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos_val += pos["shares"] * cur_px

        total = cash + pos_val
        equity.append({"date": td.strftime("%Y-%m-%d"), "value": round(total, 2),
                       "cash": round(cash, 2)})

        # ── 组合熔断检查 ──
        if total > peak_value:
            peak_value = total
        dd = (peak_value - total) / peak_value if peak_value > 0 else 0
        if dd > args.portfolio_dd_stop:
            frozen = True
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 组合回撤 {dd:.1%} > {args.portfolio_dd_stop:.0%}，熔断")
        if dd < args.portfolio_dd_stop * 0.8:
            frozen = False

        # ── 个股止损 ──
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            if cur_px < pos["entry_price"] * (1 - args.stop_loss):
                proceeds = pos["shares"] * cur_px * NET_SELL
                cash += proceeds
                pnl = (cur_px / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": "卖出(止损)",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": pos["entry_price"], "当前价/出场价": cur_px,
                    "盈亏%": round(pnl, 2), "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "总资产": round(cash, 2),
                    "当前现金": round(cash, 2),
                })
                trade_count += 1
                del positions[code]

        # ── 调仓日 ──
        if i % args.rebalance != 0 or frozen:
            # 非调仓日只记录持仓快照
            for code, pos in positions.items():
                cur_px = px_map.get(code, pos["entry_price"])
                pnl = round((cur_px / pos["entry_price"] - 1) * 100, 2)
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": "持仓",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": pos["entry_price"], "当前价/出场价": cur_px,
                    "盈亏%": pnl, "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "总资产": round(total, 2),
                    "当前现金": round(cash, 2),
                })
            continue

        # ── 调仓：清仓 → 选股 → 买入 ──
        # 卖出
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            proceeds = pos["shares"] * cur_px * NET_SELL
            cash += proceeds
            pnl = round((cur_px / pos["entry_price"] - 1) * 100, 2)
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "卖出(调仓)",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": pos["entry_price"], "当前价/出场价": cur_px,
                "盈亏%": pnl, "股数": pos["shares"],
                "入场日期": pos["entry_date"], "总资产": round(cash, 2),
                "当前现金": round(cash, 2),
            })
            trade_count += 1
        positions.clear()

        # 选股：市值最小的 pool 只 + 近期跌最多的 top_n
        extra_td = extra_by_date.get(td)
        if extra_td is None or extra_td.empty:
            continue

        # 市值排序
        mcap_td = extra_td.copy()
        mcap_td = mcap_td[mcap_td.index.isin(px_map.keys())]
        mcap_td = mcap_td[mcap_td["market_cap"] > 0].dropna(subset=["market_cap"])
        smallest = mcap_td.nsmallest(args.mcap_pool, "market_cap")

        if smallest.empty:
            continue

        # 计算回看期收益（跌最多的）
        lb_start = td - timedelta(days=args.lookback + 5)
        lb_df = daily[(daily["trade_date"] >= lb_start) & (daily["trade_date"] <= td)]
        lb_df = lb_df[lb_df["code"].isin(smallest.index)]

        # 回看期累计收益
        rets = {}
        for code in smallest.index:
            code_data = lb_df[lb_df["code"] == code].sort_values("trade_date")
            if len(code_data) >= 5:
                first_close = code_data["close"].iloc[0]
                last_close = code_data["close"].iloc[-1]
                if first_close > 0:
                    rets[code] = (last_close / first_close - 1)

        if not rets:
            continue

        # 选跌最多的 top_n（反转）
        sorted_stocks = sorted(rets.items(), key=lambda x: x[1])  # 跌最多的排前面
        picks = sorted_stocks[:args.top_n]

        # 等权买入
        alloc = total / max(len(picks), 1)
        for code, ret in picks:
            px = px_map.get(code)
            if not px or px <= 0:
                continue
            sz = int(alloc / px / 100) * 100
            if sz <= 0:
                continue
            cost = sz * px * BUY_COST
            if cost > cash:
                sz = int(cash * 0.95 / px / 100) * 100
                if sz <= 0:
                    continue
                cost = sz * px * BUY_COST
            cash -= cost
            positions[code] = {"entry_price": px, "shares": sz,
                              "entry_date": td.strftime("%Y-%m-%d")}
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": px, "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": td.strftime("%Y-%m-%d"),
                "总资产": round(cash + sum(p["shares"] * px_map.get(c, p["entry_price"])
                                          for c, p in positions.items()), 2),
                "当前现金": round(cash, 2),
            })
            trade_count += 1

        # 持仓快照
        for code, pos in positions.items():
            cur_px = px_map.get(code, pos["entry_price"])
            pnl = round((cur_px / pos["entry_price"] - 1) * 100, 2)
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "持仓",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": pos["entry_price"], "当前价/出场价": cur_px,
                "盈亏%": pnl, "股数": pos["shares"],
                "入场日期": pos["entry_date"], "总资产": round(total, 2),
                "当前现金": round(cash, 2),
            })

    # ── 输出指标 ──
    eq_values = [e["value"] for e in equity]
    fv = eq_values[-1] if eq_values else cash
    ret_total = (fv / args.cash - 1)
    # 年化
    n_years = (all_dates[-1] - all_dates[0]).days / 365.25
    ret_annual = (fv / args.cash) ** (1 / max(n_years, 0.5)) - 1

    # Sharpe
    if len(eq_values) > 2:
        dret = [(eq_values[i] - eq_values[i-1]) / max(eq_values[i-1], 1)
                for i in range(1, len(eq_values))]
        avg_dret = np.mean(dret) if dret else 0
        std_dret = np.std(dret) if dret else 1
        sharpe = float(avg_dret / std_dret * np.sqrt(252)) if std_dret > 0 else 0
    else:
        sharpe = 0

    # MDD
    peak = eq_values[0] if eq_values else cash
    mdd = 0.0
    for v in eq_values:
        if v > peak: peak = v
        mdd = max(mdd, (peak - v) / peak)

    # 输出
    trades_dir = "data/backtest_trades"
    os.makedirs(trades_dir, exist_ok=True)
    date_tag = f"{args.start.replace('-','')}_{end.strftime('%Y%m%d')}"
    csv_path = f"{trades_dir}/trades_sc_{args.top_n}_{date_tag}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                     "盈亏%", "股数", "入场日期", "总资产", "当前现金"])
        for t in trade_log:
            w.writerow([t["日期"], t["操作"], t["股票代码"], t["股票名称"],
                        t["入场价"], t["当前价/出场价"], t["盈亏%"],
                        t["股数"], t["入场日期"], t["总资产"], t["当前现金"]])

    print(f"\n{'='*60}")
    print(f"  小市值反转 Top-{args.top_n}")
    print(f"  {args.start} → {end.strftime('%Y-%m-%d')} | 本金 {args.cash:,.0f}")
    print(f"  终值 {fv:,.0f} | 收益 {ret_total:+.1%} | 年化 {ret_annual:+.1%}")
    print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {mdd:.1%}")
    print(f"  交易 {trade_count} 笔")
    print(f"  交割单: {csv_path}")
    print(f"{'='*60}")

    return {"final_value": fv, "return": ret_total, "sharpe": sharpe, "mdd": mdd,
            "trades": trade_count, "csv": csv_path, "equity": equity}


def parse_args():
    p = argparse.ArgumentParser(description="小市值反转策略")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--mcap-pool", type=int, default=100)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--rebalance", type=int, default=5)
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--portfolio-dd-stop", type=float, default=0.25)
    p.add_argument("--cash", type=float, default=1_000_000)
    return p.parse_args()


if __name__ == "__main__":
    run_backtest(parse_args())
