#!/usr/bin/env python
"""自适应趋势策略 —— 滚动IC选因子 + 动态仓位 + 多指标择时。

核心: 每20个交易日重算因子IC → 自动切换评分公式 → 适应当前市场风格。

用法:
    python scripts/bt_trend_adaptive.py --start 2020-01-01 --top-n 5
"""

from __future__ import annotations

import argparse, os, sys, csv, time
from datetime import date, timedelta
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig


# ── 候选因子池（固定）──
CANDIDATE_FACTORS = [
    'mom_20', 'mom_60', 'mom_120', 'price_accel',
    'ema_ratio_5_20', 'price_position',
    'vol_20', 'vol_60',
    'up_day_ratio', 'max_dd_20', 'rsi_14',
    'rev_5', 'rev_20',
]

FWD_DAYS = 40        # 预测窗口
IC_WINDOW = 120      # IC滚动窗口
IC_STEP = 20         # 重算间隔
N_SELECT = 6         # 每期选几个因子
REBALANCE_DAYS = 10  # 调仓频率（降低，避免过度交易）


def parse_args():
    p = argparse.ArgumentParser(description="自适应趋势策略")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--stop-loss", type=float, default=0.08)
    return p.parse_args()


def load_universe(engine, trade_date, min_listed_days=252):
    """加载股票池: 非ST, 上市>252天, 排除科创/北交。"""
    min_list = trade_date - timedelta(days=min_listed_days)
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT code, name, industry FROM stock_basic
            WHERE is_st = FALSE AND list_date <= :ld
        """), conn, params={"ld": min_list.strftime("%Y-%m-%d")})
    codes = df["code"].tolist()
    name_map = dict(zip(df["code"], df["name"]))
    ind_map = dict(zip(df["code"], df["industry"].fillna("其他")))
    # 排除 688/8/4 开头
    codes = [c for c in codes if not str(c).startswith(("688", "8", "4"))]
    return codes, name_map, ind_map


def compute_factors(daily):
    """向量化计算所有候选因子。"""
    df = daily.sort_values(["code", "trade_date"]).copy()
    df['ret'] = df.groupby('code')['close'].pct_change()

    # 动量
    df['mom_20'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(20))
    df['mom_60'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(60))
    df['mom_120'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(120))
    df['price_accel'] = df['mom_20'] - df['mom_60']

    # 均线
    ema5 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    ema20 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['ema_ratio_5_20'] = ema5 / ema20 - 1
    ma60 = df.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())
    h60 = df.groupby('code')['close'].transform(lambda x: x.rolling(60).max())
    l60 = df.groupby('code')['close'].transform(lambda x: x.rolling(60).min())
    df['price_position'] = (df['close'] - ma60) / (h60 - l60).replace(0, np.nan)

    # 波动
    df['vol_20'] = df.groupby('code')['ret'].transform(lambda x: x.rolling(20).std())
    df['vol_60'] = df.groupby('code')['ret'].transform(lambda x: x.rolling(60).std())

    # 趋势质量
    df['up_day_ratio'] = df.groupby('code')['ret'].transform(lambda x: (x > 0).rolling(20).mean())
    df['max_dd_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20).apply(lambda y: (y.max()-y.min())/y.max() if y.max()>0 else 0))

    # RSI
    delta = df.groupby('code')['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df['code']).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    avg_loss = loss.groupby(df['code']).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - 100 / (1 + rs)

    # 反转
    df['rev_5'] = -df.groupby('code')['close'].pct_change(5)
    df['rev_20'] = -df.groupby('code')['close'].pct_change(20)

    # MA 用于退出
    df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())

    return df


def compute_rolling_ic(daily, all_dates, current_idx):
    """在当前日期，用前IC_WINDOW天数据计算每个因子的RankIC。"""
    w_start = all_dates[max(0, current_idx - IC_WINDOW)]
    w_end = all_dates[current_idx]
    wd = daily[(daily['trade_date'] >= w_start) & (daily['trade_date'] <= w_end)]

    ic_means = {}
    for f in CANDIDATE_FACTORS:
        ics = []
        for dt, g in wd.groupby('trade_date'):
            valid = g[[f, f'ret_fwd']].dropna()
            if len(valid) >= 30:
                ic, _ = spearmanr(valid[f], valid[f'ret_fwd'])
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            ic_means[f] = np.mean(ics)

    # 选 |IC| 最大的 N_SELECT 个
    if ic_means:
        sorted_factors = sorted(ic_means, key=lambda x: abs(ic_means[x]), reverse=True)
        selected = sorted_factors[:N_SELECT]
        weights = {f: abs(ic_means[f]) for f in selected}
        w_sum = sum(weights.values())
        weights = {f: w / w_sum for f, w in weights.items()}
        return selected, weights, ic_means
    return CANDIDATE_FACTORS[:N_SELECT], {f: 1/N_SELECT for f in CANDIDATE_FACTORS[:N_SELECT]}, {}


def compute_market_votes(daily, all_dates, current_idx, csi_data):
    """多指标市场择时投票。"""
    today = all_dates[current_idx]
    votes = 0
    details = {}

    # 1. CSI1000趋势（自适应MA）
    if csi_data is not None and len(csi_data) > 0:
        cs = csi_data[csi_data['trade_date'] <= today].tail(60)
        if len(cs) >= 30:
            # 自适应N: 最近60日波动率越高 → MA越短
            cs_vol = cs['close'].pct_change().std()
            n_days = 20 if cs_vol > 0.02 else (40 if cs_vol > 0.015 else 60)
            cs_ma = cs['close'].rolling(n_days, min_periods=10).mean().iloc[-1]
            if cs['close'].iloc[-1] > cs_ma:
                votes += 1
                details['csi_trend'] = True

    # 2. 近20日收益 > 0
    if current_idx >= 20:
        prev_20 = all_dates[current_idx - 20]
        wd = daily[(daily['trade_date'] >= prev_20) & (daily['trade_date'] <= today)]
        mkt_ret = wd.groupby('trade_date')['ret'].mean().mean()
        if mkt_ret > 0:
            votes += 1
            details['mkt_ret_20'] = True

    # 3. 涨跌比 > 0.5
    wd_recent = daily[(daily['trade_date'] >= all_dates[max(0, current_idx-5)]) & (daily['trade_date'] <= today)]
    adv_ratio = (wd_recent.groupby('trade_date')['ret'].apply(lambda x: (x > 0).mean())).mean()
    if adv_ratio > 0.5:
        votes += 1
        details['adv_ratio'] = True

    # 4. 涨停家数趋势（用涨幅>9.5%代理）
    wd_lu = daily[(daily['trade_date'] >= all_dates[max(0, current_idx-20)]) & (daily['trade_date'] <= today)]
    lu_count = wd_lu[wd_lu['ret'] > 0.095].groupby('trade_date').size()
    if len(lu_count) >= 10:
        if lu_count.iloc[-1] > lu_count.mean():
            votes += 1
            details['lu_trend'] = True

    return votes, details


def compute_adaptive_stop_loss(daily, code, today):
    """自适应止损: max(8%, 个股近20日波动率×2)。"""
    stock_data = daily[(daily['code'] == code) & (daily['trade_date'] <= today)].tail(20)
    if len(stock_data) >= 10:
        vol = stock_data['ret'].std()
        return max(0.08, vol * 2)
    return 0.08


def run_backtest(args):
    engine = get_engine()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today())

    # ── 股票池 ──
    codes, name_map, ind_map = load_universe(engine, end)
    logger.info(f"股票池: {len(codes)} 只")

    # ── 加载数据 ──
    pre_start = (start - timedelta(days=180)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, codes, pre_start, end.strftime("%Y-%m-%d"),
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])

    # CSI1000
    csi = pd.read_sql(text(
        "SELECT trade_date, close FROM index_daily WHERE code='000852' "
        "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"
    ), engine, params={"s": pre_start, "e": end.strftime("%Y-%m-%d")})
    csi["trade_date"] = pd.to_datetime(csi["trade_date"])
    engine.dispose()

    logger.info(f"日线: {len(daily)} 行")

    # ── 计算因子 ──
    logger.info("计算因子...")
    daily = compute_factors(daily)
    daily['ret_fwd'] = daily.groupby('code')['close'].transform(lambda x: x.shift(-FWD_DAYS) / x - 1)
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}

    all_dates = sorted(d for d in daily_by_date if start <= d <= end)
    logger.info(f"交易日: {len(all_dates)}")

    # ── 回测主循环 ──
    cash = args.cash
    positions = {}
    equity = []
    trade_log = []
    trade_count = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE

    # 自适应状态
    active_factors = CANDIDATE_FACTORS[:N_SELECT]
    factor_weights = {f: 1/N_SELECT for f in active_factors}
    ic_info = {f: -0.05 for f in active_factors}  # 初始默认负IC
    adaptive_position_pct = 0.20
    last_ic_update = -999

    peak_value = cash
    prev_perf = []

    for i, td in enumerate(all_dates):
        td_df = daily_by_date.get(td)
        if td_df is None or td_df.empty:
            continue

        px_map = td_df["close"].to_dict()
        ma20_map = td_df["ma20"].to_dict() if "ma20" in td_df.columns else {}
        ma60_map = td_df["ma60"].to_dict() if "ma60" in td_df.columns else {}

        # ── 估值 ──
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos["current_price"] = cur_px
            pos["hold_days"] += 1

        pos_val = sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in positions.values())
        total = cash + pos_val
        equity.append({"date": td.strftime("%Y-%m-%d"), "value": round(total, 2), "cash": round(cash, 2)})

        # ── 自适应参数更新（每 IC_STEP 天）──
        if i - last_ic_update >= IC_STEP and i >= IC_WINDOW:
            active_factors, factor_weights, ic_info = compute_rolling_ic(daily, all_dates, i)
            last_ic_update = i
            if ic_info:
                top3 = sorted(ic_info, key=lambda x: abs(ic_info[x]), reverse=True)[:3]
                logger.info(f"  [{td.strftime('%Y-%m-%d')}] 更新因子: {', '.join(f'{t}({ic_info[t]:+.3f})' for t in top3)}")

        # ── 仓位自适应 ──
        if len(equity) >= 60:
            recent_peak = max(e["value"] for e in equity[-60:])
            recent_dd = (recent_peak - total) / recent_peak
            if recent_dd < 0.05:
                adaptive_position_pct = 0.20
            elif recent_dd < 0.15:
                adaptive_position_pct = 0.12
            else:
                adaptive_position_pct = 0.06

        # ── 退出检查 ──
        for code, pos in list(positions.items()):
            cur_px = pos["current_price"]
            sell_reason = None

            # 趋势破坏: 收盘 < MA20 或 MA60
            ma20 = ma20_map.get(code)
            ma60 = ma60_map.get(code)
            if ma20 and cur_px < ma20 and pos["hold_days"] > 5:
                sell_reason = "趋势破MA20"
            elif ma60 and cur_px < ma60 and pos["hold_days"] > 10:
                sell_reason = "趋势破MA60"

            # 个股止损
            stop = compute_adaptive_stop_loss(daily, code, td)
            if cur_px < pos["entry_price"] * (1 - stop):
                sell_reason = f"止损({stop:.0%})"

            if sell_reason:
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
                })
                trade_count += 1
                del positions[code]

        # ── 调仓日 ──
        if i % REBALANCE_DAYS != 0:
            continue

        # 择时投票
        votes, vote_details = compute_market_votes(daily, all_dates, i, csi)
        if votes < 2:
            continue  # 市场环境不好，不开仓

        # 仓位控制
        available_slots = args.top_n - len(positions)
        if available_slots <= 0:
            continue

        # 获取已有持仓的代码
        held_codes = set(positions.keys())

        # ── 选股: 评分 = Σ (权重 × 因子Z-score) ──
        td_data = td_df.copy()
        # 排除已持仓
        td_data = td_data[~td_data.index.isin(held_codes)]
        # 排除上市不足252天（通过股票池已过滤）
        # 排除无因子数据的
        td_data = td_data.dropna(subset=active_factors, how='any')

        if len(td_data) < available_slots:
            continue

        # Z-score归一化（注意IC符号：负IC→因子值越低越好）
        scores = pd.Series(0.0, index=td_data.index)
        for f in active_factors:
            vals = td_data[f]
            z = (vals - vals.mean()) / (vals.std() if vals.std() > 0 else 1)
            # IC方向：正IC→高分=正向，负IC→高分=反向
            ic_sign = 1 if ic_info.get(f, 0) > 0 else -1
            scores += z * factor_weights.get(f, 0) * ic_sign

        # 选 Top-N
        top_stocks = scores.nlargest(available_slots)

        # ── 买入 ──
        alloc_per_stock = cash * adaptive_position_pct / max(available_slots, 1)
        for code in top_stocks.index:
            px = px_map.get(code)
            if not px or px <= 0:
                continue

            sz = int(alloc_per_stock / px / 100) * 100
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
                "entry_price": px, "shares": sz,
                "entry_date": td.strftime("%Y-%m-%d"),
                "hold_days": 0, "current_price": px,
            }

            pos_val_after = sum(p["shares"] * px_map.get(c, p["entry_price"]) for c, p in positions.items())
            trade_log.append({
                "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                "股票代码": code, "股票名称": name_map.get(code, ""),
                "入场价": px, "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                "总资产": round(cash + pos_val_after, 2), "当前现金": round(cash, 2),
            })
            trade_count += 1

    # ── 输出指标 ──
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

    # ── 保存交割单 ──
    trades_dir = "data/backtest_trades"
    os.makedirs(trades_dir, exist_ok=True)
    end_str = args.end or date.today().strftime("%Y%m%d")
    date_tag = f"{args.start.replace('-','')}_{end_str.replace('-','')}" if args.end else f"{args.start.replace('-','')}_{date.today().strftime('%Y%m%d')}"
    csv_path = f"{trades_dir}/trades_adaptive_{args.top_n}_{date_tag}.csv"

    if trade_log:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价",
                         "盈亏%", "股数", "入场日期", "持有天数", "总资产", "当前现金"])
            for t in trade_log:
                w.writerow([t.get("日期", ""), t.get("操作", ""), t.get("股票代码", ""),
                            t.get("股票名称", ""), t.get("入场价", ""), t.get("当前价/出场价", ""),
                            t.get("盈亏%", ""), t.get("股数", ""), t.get("入场日期", ""),
                            t.get("持有天数", ""), t.get("总资产", ""), t.get("当前现金", "")])

    print(f"\n{'='*60}")
    print(f"  自适应趋势策略 Top-{args.top_n}")
    print(f"  {args.start} → {end.strftime('%Y-%m-%d')} | 本金 {args.cash:,.0f}")
    print(f"  终值 {fv:,.0f} | 收益 {ret_total:+.1%} | 年化 {ret_annual:+.1%}")
    print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {mdd:.1%}")
    print(f"  交易 {trade_count} 笔 | 胜率 {win_rate:.1%}")
    print(f"  交割单: {csv_path}")
    print(f"{'='*60}")

    return {"final_value": fv, "return": ret_total, "sharpe": sharpe, "mdd": mdd,
            "trades": trade_count, "win_rate": win_rate, "csv": csv_path}


if __name__ == "__main__":
    run_backtest(parse_args())
