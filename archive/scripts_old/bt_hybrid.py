#!/usr/bin/env python
"""主力+妖股 混合策略 —— 板块资金流定方向，涨停规则选个股。

用法:
    python scripts/bt_hybrid.py --start 2020-01-01 --top-n 5 --label v1
    python scripts/bt_hybrid.py --start 2024-07-01 --label val
    python scripts/bt_hybrid.py --start 2025-07-01 --label test
"""

from __future__ import annotations

import argparse, os, sys, csv, time
from datetime import date, timedelta
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig

REBALANCE_DAYS = 5  # 周频调仓

# ── 涨停阈值（板别感知）──
_LIMIT_MAP = {"688": 0.20, "8": 0.30, "4": 0.30, "300": 0.20, "301": 0.20}
_DEFAULT_LIMIT = 0.10

def _get_limit(code: str) -> float:
    for prefix, limit in _LIMIT_MAP.items():
        if str(code).startswith(prefix):
            return limit
    return _DEFAULT_LIMIT

def parse_args():
    p = argparse.ArgumentParser(description="主力+妖股 混合策略")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", default="hybrid")
    p.add_argument("--features-csv", default="data/signals/bt_signals_features.csv")
    return p.parse_args()


def load_universe(engine, trade_date, min_listed_days=252):
    """非ST/上市>252天/排除科创北交。"""
    min_list = trade_date - timedelta(days=min_listed_days)
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT code, name, industry FROM stock_basic
            WHERE is_st = FALSE AND list_date <= :ld
        """), conn, params={"ld": min_list.strftime("%Y-%m-%d")})
    df["code"] = df["code"].astype(str).str.zfill(6)
    codes = [c for c in df["code"] if not str(c).startswith(("688","8","4"))]
    name_map = dict(zip(df["code"], df["name"]))
    ind_map = dict(zip(df["code"], df["industry"].fillna("其他")))
    return codes, name_map, ind_map


def load_yaogu_signals(features_csv, start_date):
    """加载预计算的妖股特征，评分 ≥6，生成买入信号（等待首次非涨停日）。"""
    feat = pd.read_csv(features_csv)
    feat["date"] = pd.to_datetime(feat["date"])

    # 评分
    feat["rule_score"] = 0
    feat["rule_score"] += np.where(feat["lu_is_yiziban"].fillna(0) > 0, 3, 0)
    feat["rule_score"] += np.where(feat["lu_amplitude"].fillna(1) < 0.08, 2, 0)
    feat["rule_score"] += np.where(feat["lu_vol_intensity"].fillna(99) < 1.5, 1, 0)
    feat["rule_score"] += np.where(feat["lu_volume_climax"].fillna(99) < 0.8, 1, 0)
    feat["rule_score"] += np.where(feat["lu_streak"].fillna(0) >= 2, 1, 0)
    feat["rule_score"] += np.where(feat["low_vol_streak"].fillna(0) >= 1, 1, 0)

    high = feat[feat["rule_score"] >= 6].copy()
    return high


def wait_for_buyable(high_signals, daily, all_dates, date_idx):
    """T日高分信号 → 找首个非涨停非跌停日 → T+N日买入。"""
    daily_map = {}
    for d, g in daily.groupby("trade_date"):
        daily_map[d] = g.set_index("code")

    rows = []
    for _, sig in high_signals.iterrows():
        sig_date = sig["date"]
        code = str(sig["code"]).zfill(6)
        idx = date_idx.get(sig_date)
        if idx is None:
            continue
        for offset in range(1, 11):
            nxt = idx + offset
            if nxt >= len(all_dates):
                break
            nd = all_dates[nxt]
            ndf = daily_map.get(nd)
            if ndf is None or code not in ndf.index:
                continue
            r = ndf.loc[code]
            px = r["close"]
            prev_c = r.get("prev_close")
            if pd.notna(prev_c) and prev_c > 0:
                if TradingConfig.is_at_limit_up(px, prev_c, code):
                    continue
                if TradingConfig.is_at_limit_down(px, prev_c, code):
                    continue
            rows.append({
                "date": nd, "code": code,
                "score": int(sig["rule_score"]),
                "close": float(px),
                "signal_date": sig_date,
                "wait_days": offset,
            })
            break
    return pd.DataFrame(rows)


def compute_sector_flow(daily, ind_map, all_dates, current_idx):
    """板块资金流: 持续放量+动量+宽度（只用 ≤T 数据）。"""
    today = all_dates[current_idx]
    # 看最近 10 天
    lookback = all_dates[max(0, current_idx - 10):current_idx + 1]
    # 对比前 10 天
    prev_lookback = all_dates[max(0, current_idx - 20):max(0, current_idx - 10)]

    wd = daily[(daily["trade_date"].isin(lookback))].copy()
    if wd.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    wd["industry"] = wd["code"].map(ind_map).fillna("其他")
    wd["ret"] = wd.groupby("code")["close"].pct_change()

    if "turnover" in wd.columns:
        wd["turnover_ma"] = wd.groupby("code")["turnover"].transform(lambda x: x.rolling(10, min_periods=3).mean())
        wd["vol_ratio"] = wd["turnover"] / wd["turnover_ma"].replace(0, np.nan)
    else:
        wd["vol_ratio"] = 1.0

    # 今天
    today_data = wd[wd["trade_date"] == today]

    # 1. 板块动量（10日）
    sector_mom = today_data.groupby("industry")["ret"].mean()

    # 2. 板块放量持续度（连续放量天数 × 量比）
    sector_vol = today_data.groupby("industry")["vol_ratio"].mean()

    # 3. 板块涨跌比
    sector_adv = today_data.groupby("industry")["ret"].apply(
        lambda x: (x > 0).mean() if len(x) > 0 else 0.5)

    # 4. 板块内涨停家数占比（板别感知）
    today_data["is_lu"] = today_data.apply(
        lambda r: 1 if pd.notna(r["ret"]) and r["ret"] >= _get_limit(str(r["code"])) * 0.98 else 0, axis=1
    )
    sector_lu = today_data.groupby("industry")["is_lu"].apply(
        lambda x: x.mean() if len(x) > 0 else 0)

    # 综合（动量最重要，放量确认，宽度辅助）
    score = (sector_mom.fillna(0) * 0.35 +
             sector_vol.fillna(1.0) * 0.25 +
             sector_adv.fillna(0.5) * 0.25 +
             sector_lu.fillna(0) * 0.15)

    sector_n = today_data.groupby("industry").size()
    return score, sector_n


def select_sector_leaders(daily, ind_map, all_dates, current_idx, top_sectors, n_per_sector):
    """在指定行业内选放量趋势股。"""
    today = all_dates[current_idx]
    td_df = daily[daily["trade_date"] == today].copy()
    if td_df.empty:
        return []

    td_df["industry"] = td_df["code"].map(ind_map).fillna("其他")
    td_df = td_df[td_df["industry"].isin(top_sectors)]

    # 趋势+量能指标
    td_df["ret"] = td_df.groupby("code")["close"].pct_change()
    td_df["mom_10"] = td_df.groupby("code")["close"].transform(lambda x: x.pct_change(10))
    td_df["up_day_ratio"] = td_df.groupby("code")["ret"].transform(
        lambda x: (x > 0).rolling(20, min_periods=5).mean())

    if "turnover" in td_df.columns:
        td_df["vol_ratio"] = td_df.groupby("code")["turnover"].transform(
            lambda x: x / x.rolling(20, min_periods=5).mean().replace(0, np.nan))
    else:
        td_df["vol_ratio"] = 1.0

    # 综合趋势分: 动量 + 趋势质量 + 量能确认
    td_df["trend_score"] = (
        td_df["mom_10"].fillna(0) * 0.35 +
        td_df["up_day_ratio"].fillna(0.5) * 0.30 +
        td_df["vol_ratio"].clip(0.5, 3.0).fillna(1.0) * 0.20 +
        td_df["ret"].fillna(0) * 0.15
    )

    leaders = []
    for sector in top_sectors:
        sec_stocks = td_df[td_df["industry"] == sector].dropna(subset=["trend_score"])
        if sec_stocks.empty:
            continue
        top = sec_stocks.nlargest(n_per_sector, "trend_score")
        for _, r in top.iterrows():
            leaders.append({
                "code": r["code"], "industry": sector,
                "close": r["close"], "trend_score": r["trend_score"],
            })
    return leaders


def run_backtest(args):
    engine = get_engine()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today())

    # ── 股票池 + 行业映射 ──
    codes, name_map, ind_map = load_universe(engine, end)
    logger.info(f"股票池: {len(codes)} 只")

    # ── 加载日线 ──
    pre_start = (start - timedelta(days=90)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, codes, pre_start, end.strftime("%Y-%m-%d"),
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    daily["ma20"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    # CSI1000 择时
    csi = pd.read_sql(text(
        "SELECT trade_date, close FROM index_daily WHERE code='000852' "
        "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"
    ), engine, params={"s": pre_start, "e": end.strftime("%Y-%m-%d")})
    csi["trade_date"] = pd.to_datetime(csi["trade_date"])
    csi["ma60"] = csi["close"].rolling(60, min_periods=30).mean()
    csi_up = dict(zip(csi["trade_date"], csi["close"] > csi["ma60"]))
    engine.dispose()

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    all_dates = sorted(d for d in daily_by_date if start <= d <= end)
    date_idx = {d: i for i, d in enumerate(all_dates)}
    logger.info(f"交易日: {len(all_dates)}")

    # ── 妖股信号预计算 ──
    logger.info("加载妖股信号...")
    yaogu_high = load_yaogu_signals(args.features_csv, start)
    yaogu_signals = wait_for_buyable(yaogu_high, daily, all_dates, date_idx)
    yaogu_by_date = {}
    if len(yaogu_signals) > 0:
        for d, g in yaogu_signals.groupby("date"):
            yaogu_by_date[d] = g.sort_values("score", ascending=False)
    logger.info(f"妖股买入信号: {len(yaogu_signals)}条, {len(yaogu_by_date)}天")

    # ── 回测主循环 ──
    cash = args.cash
    positions = {}
    equity = []
    trade_log = []
    trade_count = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    peak_value = args.cash
    frozen = False
    frozen_days = 0

    for i, td in enumerate(all_dates):
        td_df = daily_by_date.get(td)
        if td_df is None:
            continue
        px_map = td_df["close"].to_dict()
        prev_map = {c: r["prev_close"] for c, r in td_df.iterrows() if pd.notna(r.get("prev_close"))}
        ma20_map = td_df["ma20"].to_dict()

        # ── 估值 ──
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos["current_price"] = cur_px
            pos["hold_days"] += 1

        pos_val = sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in positions.values())
        total = cash + pos_val
        equity.append({"date": td.strftime("%Y-%m-%d"), "value": round(total, 2), "cash": round(cash, 2)})

        # ── 熔断 ──
        if total > peak_value:
            peak_value = total
        dd = (peak_value - total) / peak_value if peak_value > 0 else 0
        if dd > 0.35 and not frozen:
            frozen = True; frozen_days = 0
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 组合回撤 {dd:.1%} > 35%，熔断")
        if frozen:
            frozen_days += 1
        if frozen and frozen_days > 60:
            frozen = False; peak_value = total
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] 熔断解除")

        # ── 退出 ──
        for code, pos in list(positions.items()):
            cur_px = pos["current_price"]
            sell_reason = None

            ma20 = ma20_map.get(code)
            if ma20 and cur_px < ma20 and pos["hold_days"] > 5:
                sell_reason = "趋势破MA20"

            stock_vol = daily[(daily["code"] == code) & (daily["trade_date"] <= td)].tail(20)["ret"].std()
            stop_pct = max(0.08, stock_vol * 2) if pd.notna(stock_vol) else 0.08
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
                    "股票代码": code, "股票名称": name_map.get(code, ""),
                    "入场价": pos["entry_price"], "当前价/出场价": cur_px,
                    "盈亏%": round(pnl, 2), "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "持有天数": pos["hold_days"],
                    "总资产": round(cash, 2), "当前现金": round(cash, 2),
                    "类型": pos.get("type", ""),
                })
                trade_count += 1
                del positions[code]

        # ── 调仓日 ──
        if i % REBALANCE_DAYS != 0 or frozen:
            continue

        # 择时
        if not csi_up.get(td, True):
            continue

        available_slots = args.top_n - len(positions)
        if available_slots <= 0:
            continue

        held_codes = set(positions.keys())

        # ── 信号收集 ──
        buys = []  # [{code, close, source, score}]

        # 妖股信号
        if td in yaogu_by_date:
            yg_today = yaogu_by_date[td]
            yg_today = yg_today[~yg_today["code"].isin(held_codes)]
            for _, r in yg_today.iterrows():
                buys.append({
                    "code": r["code"], "close": r["close"],
                    "source": "妖股", "score": r["score"],
                })

        # 主力板块选股
        sector_score, sector_n = compute_sector_flow(daily, ind_map, all_dates, i)
        if not sector_score.empty:
            top3 = sector_score.nlargest(3).index.tolist()
            sector_leaders = select_sector_leaders(
                daily, ind_map, all_dates, i, top3, 3)
            for sl in sector_leaders:
                if sl["code"] not in held_codes:
                    buys.append({
                        "code": sl["code"], "close": sl["close"],
                        "source": f"主力({sl['industry']})",
                        "score": sl["trend_score"],
                    })

        if not buys:
            continue

        # ── 仓位分配 ──
        yaogu_buys = [b for b in buys if b["source"] == "妖股"]
        sector_buys = [b for b in buys if b["source"] != "妖股"]

        # 仓位分配：主力最多占 3 只，始终留 2 只给妖股
        yaogu_slots = min(3, available_slots) if len(yaogu_buys) >= 1 else 0
        sector_slots = min(3, available_slots - yaogu_slots)

        picks = yaogu_buys[:yaogu_slots] + sector_buys[:sector_slots]

        if not picks:
            continue

        # ── 买入 ──
        alloc = cash * 0.95 / len(picks)
        for p in picks:
            code = p["code"]
            px = px_map.get(code, p["close"])
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
                if sz < 100: continue
                cost = sz * px * BUY_COST

            cash -= cost
            positions[code] = {
                "entry_price": px, "shares": sz,
                "entry_date": td.strftime("%Y-%m-%d"),
                "hold_days": 0, "current_price": px,
                "type": p["source"],
            }

            pos_val_after = sum(pp["shares"] * px_map.get(c, pp["entry_price"])
                                for c, pp in positions.items())
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": px, "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                "总资产": round(cash + pos_val_after, 2), "当前现金": round(cash, 2),
                "类型": p["source"],
            })
            trade_count += 1

    # ── 输出 ──
    eq_values = [e["value"] for e in equity]
    fv = eq_values[-1] if eq_values else cash
    ret_total = (fv / args.cash - 1)
    n_years = (all_dates[-1] - all_dates[0]).days / 365.25 if all_dates else 1
    ret_annual = (fv / args.cash) ** (1 / max(n_years, 0.5)) - 1

    if len(eq_values) > 2:
        dret = [(eq_values[i] - eq_values[i-1]) / max(eq_values[i-1], 1) for i in range(1, len(eq_values))]
        sharpe = float(np.mean(dret) / np.std(dret) * np.sqrt(252)) if np.std(dret) > 0 else 0
    else:
        sharpe = 0

    peak = eq_values[0] if eq_values else cash
    mdd = 0.0
    for v in eq_values:
        if v > peak: peak = v
        mdd = max(mdd, (peak - v) / peak)

    sell_trades = [t for t in trade_log if "卖出" in str(t.get("操作", ""))]
    wins = [t for t in sell_trades if float(str(t.get("盈亏%", "0")).replace("nan", "0") or 0) > 0]
    win_rate = len(wins) / max(len(sell_trades), 1)

    # ── 保存（永不覆盖）──
    trades_dir = "data/backtest_trades"
    os.makedirs(trades_dir, exist_ok=True)
    end_str = (args.end or date.today().strftime("%Y%m%d")).replace("-", "")
    start_str = args.start.replace("-", "")
    label = args.label.replace("/", "_").replace(" ", "_")
    csv_path = f"{trades_dir}/trades_hybrid_{args.top_n}_{start_str}_{end_str}_{label}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                     "盈亏%", "股数", "入场日期", "持有天数", "总资产", "当前现金", "类型"])
        for t in trade_log:
            w.writerow([t.get("日期", ""), t.get("操作", ""), t.get("股票代码", ""),
                        t.get("股票名称", ""), t.get("入场价", ""), t.get("当前价/出场价", ""),
                        t.get("盈亏%", ""), t.get("股数", ""), t.get("入场日期", ""),
                        t.get("持有天数", ""), t.get("总资产", ""), t.get("当前现金", ""),
                        t.get("类型", "")])

    print(f"\n{'='*60}")
    print(f"  主力+妖股 混合策略 Top-{args.top_n}")
    print(f"  {args.start} → {end.strftime('%Y-%m-%d')} | 本金 {args.cash:,.0f}")
    print(f"  终值 {fv:,.0f} | 收益 {ret_total:+.1%} | 年化 {ret_annual:+.1%}")
    print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {mdd:.1%}")
    print(f"  交易 {trade_count} 笔 | 胜率 {win_rate:.1%}")
    print(f"  妖股买入: {len([t for t in trade_log if t.get('类型')=='妖股' and t.get('操作')=='买入'])}")
    print(f"  主力买入: {len([t for t in trade_log if '主力' in str(t.get('类型','')) and t.get('操作')=='买入'])}")
    print(f"  交割单: {csv_path}")
    print(f"{'='*60}")

    return {"final_value": fv, "return": ret_total, "sharpe": sharpe, "mdd": mdd,
            "trades": trade_count, "win_rate": win_rate, "csv": csv_path}


if __name__ == "__main__":
    run_backtest(parse_args())
