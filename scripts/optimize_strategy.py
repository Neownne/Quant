#!/usr/bin/env python
"""策略优化：趋势过滤 + 信号交集 + 动态仓位 + 自适应止损。

在涨停信号基础上叠加优化，2020-2026全量回测找最优组合。
"""

from __future__ import annotations
import sys, os, csv, time, json
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


def load_signals():
    """加载三策略信号并标记交集。"""
    lu = pd.read_csv("data/signals/bt_signals_limit_up_full.csv")
    yg = pd.read_csv("data/signals/bt_signals_yaogu_full.csv")
    bl = pd.read_csv("data/signals/bt_signals_bull_full.csv")

    for df in [lu, yg, bl]:
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str).str.zfill(6)

    # 交集标记
    yg_codes = yg.groupby("date")["code"].apply(set).to_dict()
    bl_codes = bl.groupby("date")["code"].apply(set).to_dict()

    lu["in_yaogu"] = lu.apply(lambda r: r["code"] in yg_codes.get(r["date"], set()), axis=1)
    lu["in_bull"] = lu.apply(lambda r: r["code"] in bl_codes.get(r["date"], set()), axis=1)
    lu["in_both"] = lu["in_yaogu"] & lu["in_bull"]

    logger.info(f"涨停: {len(lu)} | 涨停∩妖股: {lu['in_yaogu'].sum()} | "
                f"涨停∩牛股: {lu['in_bull'].sum()} | 三交集: {lu['in_both'].sum()}")
    return lu


def load_csi_trend(engine, start, end):
    """CSI1000 趋势过滤。"""
    df = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code='000852' "
             "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"),
        engine, params={"s": (start - timedelta(days=90)).strftime("%Y-%m-%d"),
                         "e": end.strftime("%Y-%m-%d")})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ma60"] = df["close"].rolling(60, min_periods=30).mean()
    df["trend_up"] = df["close"] > df["ma60"]
    return dict(zip(df["trade_date"], df["trend_up"]))


def run_optimized(sig_df, engine, config):
    """优化版回测。

    config = {
        "top_n": 5,
        "trend_filter": True,        # CSI1000>MA60才交易
        "require_yaogu": False,       # 要求涨停∩妖股
        "require_bull": False,        # 要求涨停∩牛股
        "dynamic_sizing": True,       # 熊市减仓
        "adaptive_stop": True,        # 波动率自适应止损
        "trailing_stop": 0.12,
        "min_hold_days": 5,
    }
    """
    cash = 1_000_000
    positions = {}
    equity, trade_log = [], []
    trade_count = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    start = sig_df["date"].min()
    end = sig_df["date"].max()

    # 加载日线
    codes = sig_df["code"].unique().tolist()
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

    csi_trend = load_csi_trend(engine, start, end) if config.get("trend_filter") else None

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    all_dates = sorted(d for d in daily_by_date if start <= d <= end)

    # 信号按日期分组
    sig_by_date = {}
    for d, g in sig_df.groupby("date"):
        # 信号交集过滤
        if config.get("require_yaogu"):
            g = g[g["in_yaogu"]]
        if config.get("require_bull"):
            g = g[g["in_bull"]]
        if not g.empty:
            sig_by_date[d] = g.sort_values("score", ascending=False)

    peak_value, frozen, frozen_days = cash, False, 0
    bear_mode = False

    for i, td in enumerate(all_dates):
        td_df = daily_by_date[td]
        px_map = td_df["close"].to_dict()
        prev_map = {c: r["prev_close"] for c, r in td_df.iterrows() if pd.notna(r.get("prev_close"))}
        ma20_map = td_df["ma20"].to_dict()

        # 趋势判断
        if csi_trend is not None:
            trend_up = csi_trend.get(td, True)
            bear_mode = not trend_up

        # 更新持仓
        for pos in positions.values():
            cur_px = px_map.get(pos["code"], pos["entry_price"])
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
        if frozen:
            frozen_days += 1
        if frozen and frozen_days > 60:
            frozen, peak_value = False, total

        # ── 退出检查 ──
        for pos in list(positions.values()):
            code = pos["code"]
            cur_px, sell_reason = pos["current_price"], None

            ma20 = ma20_map.get(code)
            if ma20 and cur_px < ma20 and pos["hold_days"] > 5:
                sell_reason = "破MA20"
            elif (pos["hold_days"] >= config["min_hold_days"] and
                  pos.get("peak_price", 0) > pos["entry_price"] * 1.05):
                if cur_px < pos["peak_price"] * (1 - config["trailing_stop"]):
                    sell_reason = "移动止盈"

            # 自适应止损
            if config.get("adaptive_stop"):
                code_ret = daily[(daily["code"] == code) & (daily["trade_date"] <= td)]
                stock_vol = code_ret.tail(20)["ret"].std() if len(code_ret) >= 10 else 0
                stop_pct = max(0.08, stock_vol * 2) if pd.notna(stock_vol) and stock_vol > 0 else 0.08
                # 熊市收紧止损
                if bear_mode:
                    stop_pct = max(0.05, stock_vol * 1.5) if pd.notna(stock_vol) and stock_vol > 0 else 0.08
            else:
                stop_pct = 0.08

            if cur_px < pos["entry_price"] * (1 - stop_pct):
                sell_reason = f"止损({stop_pct:.0%})"

            if sell_reason:
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_down(cur_px, prev_c, code):
                    continue
                proceeds = pos["shares"] * cur_px * NET_SELL
                cash += proceeds
                pnl = (cur_px / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": f"卖出({sell_reason})",
                    "股票代码": code, "股票名称": "",
                    "入场价": round(pos["entry_price"], 2), "当前价/出场价": round(cur_px, 2),
                    "盈亏%": round(pnl, 2), "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "持有天数": pos["hold_days"],
                    "总资产": round(cash, 2), "当前现金": round(cash, 2),
                })
                trade_count += 1
                positions.remove(pos)

        # ── 调仓买入 ──
        if i % REBALANCE_DAYS != 0 or frozen:
            continue
        if td not in sig_by_date:
            continue
        if config.get("trend_filter") and bear_mode:
            continue  # 熊市不买

        available = config["top_n"] - len(positions)
        if available <= 0:
            continue

        today_sigs = sig_by_date[td]
        held = {p["code"] for p in positions}
        today_sigs = today_sigs[~today_sigs["code"].isin(held)]

        # 动态仓位：熊市仓位减半
        position_cap = available if not bear_mode else max(1, available // 2)
        alloc = cash * 0.95 / position_cap

        bought = 0
        for _, s in today_sigs.iterrows():
            if bought >= position_cap:
                break
            code, px = s["code"], px_map.get(s["code"])
            if not px or px <= 0:
                continue
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
            positions.append({
                "code": code, "entry_price": px, "shares": sz,
                "entry_date": td.strftime("%Y-%m-%d"),
                "hold_days": 0, "current_price": px, "peak_price": px,
            })
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                "股票代码": code, "股票名称": "",
                "入场价": round(px, 2), "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                "总资产": round(cash + sum(p["shares"] * px_map.get(p["code"], p["entry_price"])
                                           for p in positions), 2), "当前现金": round(cash, 2),
            })
            trade_count += 1
            bought += 1

    engine.dispose()

    # ── 统计 ──
    eq_values = [e["value"] for e in equity]
    fv = eq_values[-1] if eq_values else cash
    ret_total = (fv / 1_000_000 - 1)
    n_years = max((all_dates[-1] - all_dates[0]).days / 365.25, 0.5)
    ret_annual = (fv / 1_000_000) ** (1 / n_years) - 1

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

    sells = [t for t in trade_log if "卖出" in str(t.get("操作", ""))]
    wins = [t for t in sells if float(str(t.get("盈亏%", "0")).replace("nan", "0") or 0) > 0]
    win_rate = len(wins) / max(len(sells), 1)

    return {
        "fv": fv, "ret_total": ret_total, "ret_annual": ret_annual,
        "sharpe": round(sharpe, 2), "mdd": round(mdd * 100, 1),
        "trades": trade_count, "win_rate": round(win_rate * 100, 1),
        "buys": len([t for t in trade_log if t.get("操作") == "买入"]),
        "sells": len(sells),
        "trade_log": trade_log,
    }


def main():
    engine = get_engine()
    sig_df = load_signals()

    configs = [
        # (name, config)
        ("baseline", {"top_n": 5, "trend_filter": False, "require_yaogu": False,
                      "require_bull": False, "dynamic_sizing": False,
                      "adaptive_stop": False, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("trend_filter", {"top_n": 5, "trend_filter": True, "require_yaogu": False,
                          "require_bull": False, "dynamic_sizing": False,
                          "adaptive_stop": False, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("yaogu_overlap", {"top_n": 5, "trend_filter": False, "require_yaogu": True,
                           "require_bull": False, "dynamic_sizing": False,
                           "adaptive_stop": False, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("adaptive_stop", {"top_n": 5, "trend_filter": False, "require_yaogu": False,
                           "require_bull": False, "dynamic_sizing": False,
                           "adaptive_stop": True, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("dynamic_sizing", {"top_n": 5, "trend_filter": True, "require_yaogu": False,
                            "require_bull": False, "dynamic_sizing": True,
                            "adaptive_stop": False, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("all_in", {"top_n": 5, "trend_filter": True, "require_yaogu": False,
                    "require_bull": False, "dynamic_sizing": True,
                    "adaptive_stop": True, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("all_yaogu", {"top_n": 5, "trend_filter": True, "require_yaogu": True,
                       "require_bull": False, "dynamic_sizing": True,
                       "adaptive_stop": True, "trailing_stop": 0.12, "min_hold_days": 5}),
        ("top3_tight", {"top_n": 3, "trend_filter": True, "require_yaogu": False,
                        "require_bull": False, "dynamic_sizing": True,
                        "adaptive_stop": True, "trailing_stop": 0.08, "min_hold_days": 3}),
    ]

    results = []
    best_name, best_ret, best_sharpe, best_mdd = "", -999, -999, 999

    for name, cfg in configs:
        logger.info(f"测试: {name} ...")
        t0 = time.time()
        try:
            r = run_optimized(sig_df, engine, cfg)
            r["name"] = name
            results.append(r)
            elapsed = time.time() - t0
            logger.info(f"  {name}: 年化{r['ret_annual']:+.1%} Sharpe{r['sharpe']:.1f} "
                        f"MDD{r['mdd']:.1f}% 胜率{r['win_rate']:.0f}% ({elapsed:.0f}s)")

            # 综合评分：Sharpe优先，MDD惩罚
            score = r["sharpe"] - r["mdd"] / 100 * 0.5
            if score > best_sharpe - best_mdd / 100 * 0.5:
                best_name, best_ret, best_sharpe, best_mdd = name, r["ret_annual"], r["sharpe"], r["mdd"]
        except Exception as e:
            logger.error(f"  {name} 失败: {e}")

    # ── 打印对比 ──
    print(f"\n{'='*80}")
    print(f"  优化结果对比 (2020-01-01 → 2026-06-17)")
    print(f"{'='*80}")
    print(f"{'配置':<20s} {'年化':>8s} {'Sharpe':>8s} {'MDD':>8s} {'胜率':>7s} {'交易':>6s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
    for r in sorted(results, key=lambda x: x["sharpe"] - x["mdd"] / 100 * 0.5, reverse=True):
        print(f"{r['name']:<20s} {r['ret_annual']:>+7.1%} {r['sharpe']:>8.1f} {r['mdd']:>7.1f}% "
              f"{r['win_rate']:>6.0f}% {r['trades']:>5d}")

    # ── 最佳配置详细 ──
    print(f"\n🏆 最佳: {best_name}")
    print(f"   年化: {best_ret:+.1%} | Sharpe: {best_sharpe:.1f} | MDD: {best_mdd:.1f}%")

    # 保存最佳配置的交割单
    for r in results:
        if r["name"] == best_name:
            csv_path = f"{OUT_DIR}/trades_optimized_{best_name}_20200101_20260617.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                             "盈亏%", "股数", "入场日期", "持有天数", "总资产", "当前现金"])
                for t in r["trade_log"]:
                    w.writerow([t.get("日期", ""), t.get("操作", ""), t.get("股票代码", ""),
                                t.get("股票名称", ""), t.get("入场价", ""), t.get("当前价/出场价", ""),
                                t.get("盈亏%", ""), t.get("股数", ""), t.get("入场日期", ""),
                                t.get("持有天数", ""), t.get("总资产", ""), t.get("当前现金", "")])
            print(f"   交割单: {csv_path}")
            break


if __name__ == "__main__":
    main()
