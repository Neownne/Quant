#!/usr/bin/env python
"""ML信号质量策略 — 向量化回测。

买入持有风格：仅在ML信号日入场，持有至止损/移动止盈，不因排名变化调仓。

用法:
    python scripts/bt_ml_signals.py --start 2020-01-01 --top-n 5
    python scripts/bt_ml_signals.py --start 2020-01-01 --signals data/signals/bt_signals_ml.csv
"""

from __future__ import annotations

import argparse, os, sys, csv, time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from config.settings import TradingConfig


def parse_args():
    p = argparse.ArgumentParser(description="ML信号质量策略回测")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--signals", default="data/signals/bt_signals_ml.csv")
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--trailing-stop", type=float, default=0.12)
    p.add_argument("--position-pct", type=float, default=0.20,
                   help="单只股票占总资产比例 (0.2=20%%)")
    p.add_argument("--portfolio-dd-stop", type=float, default=0.35,
                   help="组合回撤熔断阈值（默认35%%）")
    p.add_argument("--min-hold-days", type=int, default=3,
                   help="最短持有天数（避免T+1就卖）")
    p.add_argument("--no-trailing", action="store_true", help="禁用移动止盈")
    return p.parse_args()


def run_backtest(args):
    engine = get_engine()

    # ── 加载信号 ──
    sig = pd.read_csv(args.signals)
    sig = sig.rename(columns={"date": "trade_date"}) if "date" in sig.columns else sig
    sig["trade_date"] = pd.to_datetime(sig["trade_date"])
    sig["code"] = sig["code"].astype(str).str.zfill(6)
    sig = sig[(sig["trade_date"] >= pd.Timestamp(args.start))]
    if args.end:
        sig = sig[(sig["trade_date"] <= pd.Timestamp(args.end))]

    # 提取 ml_score（如果有）
    has_ml = "ml_score" in sig.columns

    signal_by_date = {}
    for d, g in sig.groupby("trade_date"):
        signal_by_date[d] = g.sort_values("ml_score", ascending=False) if has_ml else g

    all_signal_codes = sorted(sig["code"].unique().tolist())

    # ── 加载股票名称 ──
    from sqlalchemy import text
    with engine.connect() as conn:
        name_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE code = ANY(:codes)"),
            conn, params={"codes": all_signal_codes},
        )
    name_map = dict(zip(name_df["code"].astype(str).str.zfill(6), name_df["name"]))

    logger.info(f"信号: {len(sig)} 条, {len(all_signal_codes)} 只, {len(signal_by_date)} 天有信号")

    # ── 加载日线 ──
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today())
    pre_start = (start - timedelta(days=90)).strftime("%Y-%m-%d")

    # 加载所有有信号的股票 + 指数（如果后续需要）
    daily = load_daily_data(engine, all_signal_codes, pre_start, end.strftime("%Y-%m-%d"),
                            cols=["open", "high", "low", "close", "volume"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    engine.dispose()

    # 前收盘（涨跌停判断用）
    daily = daily.sort_values(["code", "trade_date"])
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    all_dates = sorted(d for d in daily_by_date if start <= d <= end)
    logger.info(f"交易日: {len(all_dates)}")

    # ── 回测主循环 ──
    cash = args.cash
    positions = {}        # {code: {entry_price, shares, entry_date, peak_price, hold_days}}
    equity = []
    trade_log = []
    trade_count = 0
    frozen = False
    frozen_days = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    peak_value = cash

    for i, td in enumerate(all_dates):
        td_df = daily_by_date.get(td)
        if td_df is None or td_df.empty:
            continue
        px_map = td_df["close"].to_dict()
        prev_map = {c: r["prev_close"] for c, r in td_df.iterrows() if pd.notna(r.get("prev_close"))}

        # ── 估值 ──
        pos_val = 0.0
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos_val += pos["shares"] * cur_px
            pos["current_price"] = cur_px
            pos["hold_days"] += 1
            if cur_px > pos.get("peak_price", 0):
                pos["peak_price"] = cur_px

        total = cash + pos_val
        equity.append({
            "date": td.strftime("%Y-%m-%d"), "value": round(total, 2),
            "cash": round(cash, 2),
        })

        # ── 组合熔断：锁利润，定时重置 ──
        if total > peak_value:
            peak_value = total
        dd = (peak_value - total) / peak_value if peak_value > 0 else 0

        if dd > args.portfolio_dd_stop and not frozen:
            frozen = True
            frozen_days = 0
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 组合回撤 {dd:.1%} > {args.portfolio_dd_stop:.0%}，熔断")

        if frozen:
            frozen_days += 1

        # 解锁：DD恢复 或 60天自动解
        if frozen and (dd < args.portfolio_dd_stop * 0.6 or frozen_days > 60):
            frozen = False
            peak_value = total
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 熔断解除 (DD={dd:.1%}, 冻结{frozen_days}天)")

        # ── 止损/移动止盈 ──
        for code, pos in list(positions.items()):
            cur_px = pos["current_price"]
            sell_reason = None

            # 硬止损
            if cur_px < pos["entry_price"] * (1 - args.stop_loss):
                sell_reason = "止损"
            # 移动止盈（持有 > min_hold_days 后才触发）
            elif (not args.no_trailing and pos["hold_days"] >= args.min_hold_days
                  and pos.get("peak_price", 0) > pos["entry_price"] * 1.05
                  and cur_px < pos["peak_price"] * (1 - args.trailing_stop)):
                sell_reason = "移动止盈"

            # 跌停检查：跌停封死无法卖出
            if sell_reason:
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_down(cur_px, prev_c, code):
                    sell_reason = None  # 今天卖不掉，等下一天

            if sell_reason:
                proceeds = pos["shares"] * cur_px * NET_SELL
                cash += proceeds
                pnl = (cur_px / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": f"卖出({sell_reason})",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": pos["entry_price"],
                    "当前价/出场价": cur_px, "盈亏%": round(pnl, 2),
                    "股数": pos["shares"], "入场日期": pos["entry_date"],
                    "持有天数": pos["hold_days"],
                    "总资产": round(cash, 2), "当前现金": round(cash, 2),
                })
                trade_count += 1
                del positions[code]

        # ── 入场（仅信号日）──
        if not frozen and td in signal_by_date:
            today_signals = signal_by_date[td]

            # 计算可用仓位
            available_slots = args.top_n - len(positions)
            if available_slots <= 0:
                # 满仓时记录持仓快照
                for code, pos in positions.items():
                    cur_px = pos["current_price"]
                    pnl = round((cur_px / pos["entry_price"] - 1) * 100, 2)
                    trade_log.append({
                        "日期": td.strftime("%Y-%m-%d"), "操作": "持仓",
                        "股票代码": code, "股票名称": name_map.get(code, ""),
                        "入场价": pos["entry_price"],
                        "当前价/出场价": cur_px, "盈亏%": pnl,
                        "股数": pos["shares"], "入场日期": pos["entry_date"],
                        "持有天数": pos["hold_days"],
                        "总资产": round(total, 2), "当前现金": round(cash, 2),
                    })
                continue

            # 按总资产比例分配，但不能超过可用现金
            alloc = total * args.position_pct
            # 总仓位上限控制（不能超过现金）
            max_total_cost = cash * 0.98  # 留2%缓冲
            if alloc > max_total_cost / max(available_slots, 1):
                alloc = max_total_cost / max(available_slots, 1)

            bought_today = 0
            for _, s in today_signals.iterrows():
                if bought_today >= available_slots:
                    break
                code = str(s["code"]).zfill(6)

                # 已持有则跳过
                if code in positions:
                    continue

                px = px_map.get(code)
                if not px or px <= 0:
                    continue

                # 涨停封板跳过（买不到，顺延到下一个候选）
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_up(px, prev_c, code):
                    continue

                sz = int(alloc / px / 100) * 100
                if sz < 100:
                    continue

                cost = sz * px * BUY_COST
                if cost > cash * 0.98:
                    sz = int(cash * 0.95 / px / 100) * 100
                    if sz < 100:
                        continue
                    cost = sz * px * BUY_COST

                cash -= cost
                positions[code] = {
                    "entry_price": px,
                    "shares": sz,
                    "entry_date": td.strftime("%Y-%m-%d"),
                    "peak_price": px,
                    "hold_days": 0,
                    "current_price": px,
                }

                pos_val_after = sum(
                    p["shares"] * px_map.get(c, p["entry_price"])
                    for c, p in positions.items()
                )
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": px, "当前价/出场价": "",
                    "盈亏%": "", "股数": sz,
                    "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                    "总资产": round(cash + pos_val_after, 2),
                    "当前现金": round(cash, 2),
                })
                trade_count += 1
                bought_today += 1

        # ── 记录持仓快照（非信号日且有持仓时）──
        if td not in signal_by_date:
            for code, pos in positions.items():
                cur_px = pos["current_price"]
                pnl = round((cur_px / pos["entry_price"] - 1) * 100, 2)
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": "持仓",
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": pos["entry_price"],
                    "当前价/出场价": cur_px, "盈亏%": pnl,
                    "股数": pos["shares"], "入场日期": pos["entry_date"],
                    "持有天数": pos["hold_days"],
                    "总资产": round(total, 2), "当前现金": round(cash, 2),
                })

    # ── 输出指标 ──
    eq_values = [e["value"] for e in equity]
    fv = eq_values[-1] if eq_values else cash
    ret_total = (fv / args.cash - 1)
    n_years = (all_dates[-1] - all_dates[0]).days / 365.25 if all_dates else 1
    ret_annual = (fv / args.cash) ** (1 / max(n_years, 0.5)) - 1

    if len(eq_values) > 2:
        dret = [(eq_values[i] - eq_values[i-1]) / max(eq_values[i-1], 1)
                for i in range(1, len(eq_values))]
        avg_dret = np.mean(dret) if dret else 0
        std_dret = np.std(dret) if dret else 1
        sharpe = float(avg_dret / std_dret * np.sqrt(252)) if std_dret > 0 else 0
    else:
        sharpe = 0

    peak = eq_values[0] if eq_values else cash
    mdd = 0.0
    for v in eq_values:
        if v > peak:
            peak = v
        mdd = max(mdd, (peak - v) / peak)

    # 胜率
    sell_trades = [t for t in trade_log if "卖出" in str(t.get("操作", ""))]
    wins = [t for t in sell_trades if float(str(t.get("盈亏%", "0")).replace("nan", "0") or 0) > 0]
    win_rate = len(wins) / max(len(sell_trades), 1)

    # ── 输出 ──
    trades_dir = "data/backtest_trades"
    os.makedirs(trades_dir, exist_ok=True)
    end_str = args.end or date.today().strftime("%Y%m%d")
    date_tag = f"{args.start.replace('-', '')}_{end_str.replace('-', '')}" if args.end else f"{args.start.replace('-', '')}_{date.today().strftime('%Y%m%d')}"
    csv_path = f"{trades_dir}/trades_ml_{args.top_n}_{date_tag}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                     "盈亏%", "股数", "入场日期", "持有天数", "总资产", "当前现金"])
        for t in trade_log:
            w.writerow([t.get("日期", ""), t.get("操作", ""), t.get("股票代码", ""),
                        t.get("股票名称", ""),
                        t.get("入场价", ""), t.get("当前价/出场价", ""),
                        t.get("盈亏%", ""), t.get("股数", ""), t.get("入场日期", ""),
                        t.get("持有天数", ""), t.get("总资产", ""), t.get("当前现金", "")])

    print(f"\n{'='*60}")
    print(f"  ML 信号质量策略 Top-{args.top_n}")
    print(f"  {args.start} → {end.strftime('%Y-%m-%d')} | 本金 {args.cash:,.0f}")
    print(f"  终值 {fv:,.0f} | 收益 {ret_total:+.1%} | 年化 {ret_annual:+.1%}")
    print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {mdd:.1%}")
    print(f"  交易 {trade_count} 笔 | 胜率 {win_rate:.1%}")
    print(f"  交割单: {csv_path}")
    print(f"{'='*60}")

    return {
        "final_value": fv, "return": ret_total, "sharpe": sharpe,
        "mdd": mdd, "trades": trade_count, "win_rate": win_rate,
        "csv": csv_path, "equity": equity,
    }


if __name__ == "__main__":
    run_backtest(parse_args())
