#!/usr/bin/env python
"""大小票 v4.0 每日模拟盘。

CSI1000 vs MA60 → 动态分配：
  强势（CSI1000 > MA60）→ 涨停侧 70% + 小票侧 30%
  弱势（CSI1000 < MA60）→ 小票侧 90% + 涨停侧 10%

涨停侧：5条件规则筛选，日频调仓
小票侧：20日反转选股（简化版，替代ML重训），日频调仓

用法:
    python scripts/run_daily_paper_switch.py
    python scripts/run_daily_paper_switch.py --date 2026-06-05 --no-sync
    python scripts/run_daily_paper_switch.py --dry-run
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from data.sync import sync_stock_daily, sync_daily_extra
from config.settings import TradingConfig

# ── 参数 ──
ACCOUNT_ID = 2
RUN_ID = 2
LU_TOP_N = 5
SC_TOP_N = 5
MCAP_MIN, MCAP_MAX = 50.0, 300.0
PRICE_MIN, PRICE_MAX = 5.0, 50.0
LU_PCT, LU_LOOKBACK, LU_COUNT = 0.099, 20, 1
LD_PCT, LD_LOOKBACK = -0.099, 10
LU_MIN_COND = 4
MIN_LISTED_DAYS = 120


def sync_data(engine):
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"同步数据至 {today} ...")
    sync_stock_daily(engine, start_date=today, workers=8)
    sync_daily_extra(engine, start_date=today, workers=8)


def get_csi1k_deviation(engine, trade_date):
    """CSI1000 偏离 MA60 的程度。"""
    row = pd.read_sql(
        "SELECT close FROM index_daily WHERE code='000852' AND trade_date <= %s "
        "ORDER BY trade_date DESC LIMIT 1",
        engine, params=(trade_date,),
    )
    if row.empty:
        return 0.0, True
    c = float(row.iloc[0]["close"])

    ma = pd.read_sql(
        "SELECT AVG(close) as ma60 FROM ("
        "SELECT close FROM index_daily WHERE code='000852' AND trade_date <= %s "
        "ORDER BY trade_date DESC LIMIT 60) t",
        engine, params=(trade_date,),
    )
    m = float(ma.iloc[0]["ma60"]) if not ma.empty else c
    return (c / m - 1), (c > m)


def load_daily(engine, codes, pre_start, end_date):
    cl = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(
        f"SELECT code, trade_date, open, close, volume, amount "
        f"FROM stock_daily WHERE code IN ({cl}) "
        f"AND trade_date BETWEEN %s AND %s ORDER BY code, trade_date",
        engine, params=(pre_start, end_date),
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["ma5"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(10, min_periods=10).mean())
    df["ret_20"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change(20))
    df["ret_oc"] = df["close"] / df["open"] - 1
    return df


def load_extra(engine, codes, pre_start, end_date):
    cl = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(
        f"SELECT code, trade_date, market_cap FROM stock_daily_extra "
        f"WHERE code IN ({cl}) AND trade_date BETWEEN %s AND %s",
        engine, params=(pre_start, end_date),
    )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def lu_screening(today, daily, extra_df, code_set, min_cond=LU_MIN_COND):
    """涨停侧筛选。"""
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    if today not in daily_by_date:
        return []
    td = daily_by_date[today]

    lb_start = today - timedelta(days=max(LU_LOOKBACK, LD_LOOKBACK) + 5)
    lb = daily[(daily["trade_date"] >= lb_start) & (daily["trade_date"] <= today)]
    lu_counts = lb[lb["ret"] >= LU_PCT].groupby("code").size()
    ld_codes = set(lb[lb["ret"] <= LD_PCT]["code"].unique())

    extra_by_date = {pd.Timestamp(d): g.set_index("code")
                     for d, g in extra_df.groupby("trade_date")} if not extra_df.empty else {}
    avail = sorted([d for d in extra_by_date if d <= today], reverse=True)
    mcap_s = extra_by_date[avail[0]]["market_cap"] if avail else None

    passed = []
    for code in td.index:
        if code not in code_set:
            continue
        r = td.loc[code]
        close_p = r["close"]
        if pd.isna(close_p) or close_p <= 0:
            continue
        ma5, ma10 = r.get("ma5"), r.get("ma10")

        c1 = (mcap_s is not None and code in mcap_s.index and
              not pd.isna(mcap_s.loc[code]) and
              MCAP_MIN <= mcap_s.loc[code] <= MCAP_MAX)
        c2 = PRICE_MIN <= close_p <= PRICE_MAX
        c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
        c4 = int(lu_counts.get(code, 0)) > LU_COUNT
        c5 = code not in ld_codes

        if sum([c1, c2, c3, c4, c5]) >= min_cond:
            passed.append((code, int(lu_counts.get(code, 0)), float(close_p)))

    passed.sort(key=lambda x: x[1], reverse=True)
    return passed


def sc_screening(today, daily, code_set):
    """小票侧：20日反转选股（跌得最多 + 非ST + 非跌停）。"""
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    if today not in daily_by_date:
        return []
    td = daily_by_date[today]

    # 排除近5日有跌停的
    lb = daily[(daily["trade_date"] >= today - timedelta(days=LD_LOOKBACK)) &
               (daily["trade_date"] <= today)]
    ld_codes = set(lb[lb["ret"] <= LD_PCT]["code"].unique())

    candidates = []
    for code in td.index:
        if code not in code_set:
            continue
        r = td.loc[code]
        close_p = r["close"]
        ret20 = r.get("ret_20")
        if pd.isna(close_p) or close_p <= 0 or pd.isna(ret20):
            continue
        if code in ld_codes:
            continue
        if close_p < 3 or close_p > 100:
            continue
        # 选20日跌幅最大的（反转）
        candidates.append((code, ret20, float(close_p)))

    candidates.sort(key=lambda x: x[1])  # 跌幅最大排前面
    return candidates


def get_positions(engine):
    rows = pd.read_sql(
        "SELECT stock_code, entry_date, entry_price, quantity FROM paper_positions "
        "WHERE run_id = %s AND exit_date IS NULL", engine, params=(RUN_ID,),
    )
    return {r["stock_code"]: {"entry_date": r["entry_date"],
                               "entry_price": r["entry_price"],
                               "quantity": r["quantity"]}
            for _, r in rows.iterrows()}


def get_cash(engine):
    return float(pd.read_sql(
        "SELECT cash FROM paper_account WHERE id = %s",
        engine, params=(ACCOUNT_ID,)).iloc[0]["cash"])


def get_next_trading_day(engine, trade_date):
    """找到 trade_date 之后的下一个实际交易日。"""
    row = pd.read_sql(
        "SELECT MIN(trade_date) FROM stock_daily WHERE trade_date > %s",
        engine, params=(trade_date,),
    )
    if row.empty or row.iloc[0, 0] is None or pd.isna(row.iloc[0, 0]):
        return trade_date + timedelta(days=1)
    return pd.Timestamp(row.iloc[0, 0])


def execute(engine, trade_date, lu_picks, sc_picks, lu_weight, dry_run=False):
    """执行调仓。"""
    cash = get_cash(engine)
    positions = get_positions(engine)
    initial = 1_000_000

    lu_target = set(s[0] for s in lu_picks[:LU_TOP_N])
    sc_target = set(s[0] for s in sc_picks[:SC_TOP_N])
    all_target = lu_target | sc_target
    current = set(positions.keys())

    to_buy = all_target - current
    to_sell = current - all_target
    next_date = get_next_trading_day(engine, trade_date)

    # 次日无数据 → 只记录信号不执行
    next_data = pd.read_sql(
        "SELECT COUNT(*) as n FROM stock_daily WHERE trade_date=%s",
        engine, params=(next_date,),
    )
    if next_data.iloc[0,0] == 0:
        logger.info(f"  次日{str(next_date)[:10]}无交易数据，跳过执行（仅记录信号）")
        return

    if len(to_buy) == 0 and len(to_sell) == 0:
        logger.info(f"  无变动（持{len(current)}只）")
        return

    # 卖出释放现金
    sell_cash = sum(positions[c]["entry_price"] * positions[c]["quantity"]
                    for c in to_sell)
    total_cash = cash + sell_cash
    n_buy = len(to_buy)
    per_stock = total_cash / n_buy if n_buy > 0 else 0

    if dry_run:
        logger.info(f"  [DRY RUN] lu_w={lu_weight:.0%} 买{len(to_buy)}卖{len(to_sell)}")
        for c in to_buy:
            logger.info(f"    BUY {c}")
        for c in to_sell:
            logger.info(f"    SELL {c}")
        return

    with engine.begin() as conn:
        for code in to_sell:
            pos = positions[code]
            # 取 T+1 开盘价
            op = pd.read_sql(
                "SELECT open FROM stock_daily WHERE code=%s AND trade_date=%s",
                engine, params=(code, next_date),
            )
            exit_px = float(op.iloc[0]["open"]) if not op.empty else pos["entry_price"]
            pnl = (exit_px - pos["entry_price"]) * pos["quantity"]
            conn.execute(text("""
                UPDATE paper_positions SET exit_date=:ed, exit_price=:ep, pnl=:pnl
                WHERE run_id=:rid AND stock_code=:code AND exit_date IS NULL
            """), {"ed": next_date, "ep": exit_px, "pnl": pnl,
                   "rid": RUN_ID, "code": code})

        for code in to_buy:
            op = pd.read_sql(
                "SELECT open FROM stock_daily WHERE code=%s AND trade_date=%s",
                engine, params=(code, next_date),
            )
            price = float(op.iloc[0]["open"]) if not op.empty else 0
            if price <= 0:
                continue
            qty = int(per_stock / price / 100) * 100
            if qty <= 0:
                continue
            conn.execute(text("""
                INSERT INTO paper_positions (run_id, stock_code, entry_date, entry_price, quantity)
                VALUES (:rid, :code, :ed, :ep, :qty)
            """), {"rid": RUN_ID, "code": code, "ed": next_date,
                   "ep": price, "qty": qty})

        # 更新现金
        total_buy = n_buy * per_stock
        new_cash = cash + sell_cash - total_buy
        conn.execute(text("UPDATE paper_account SET cash=:c WHERE id=:aid"),
                     {"c": max(0, new_cash), "aid": ACCOUNT_ID})

        # 写信号 (ON CONFLICT UPDATE 避免重复跑导致 rank 冲突)
        for i, s in enumerate(lu_picks[:LU_TOP_N]):
            conn.execute(text("""
                INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                VALUES (:rid, :sd, :code, :score, :rank)
                ON CONFLICT (run_id, signal_date, stock_code) DO UPDATE SET
                    predicted_score = :score2, rank = :rank2
            """), {"rid": RUN_ID, "sd": trade_date, "code": s[0],
                   "score": float(s[1]), "rank": i+1,
                   "score2": float(s[1]), "rank2": i+1})
        for i, s in enumerate(sc_picks[:SC_TOP_N]):
            conn.execute(text("""
                INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                VALUES (:rid, :sd, :code, :score, :rank)
                ON CONFLICT (run_id, signal_date, stock_code) DO UPDATE SET
                    predicted_score = :score2, rank = :rank2
            """), {"rid": RUN_ID, "sd": trade_date, "code": s[0],
                   "score": float(round(s[1], 4)), "rank": LU_TOP_N + i + 1,
                   "score2": float(round(s[1], 4)), "rank2": LU_TOP_N + i + 1})

    logger.info(f"  执行: 买{len(to_buy)}卖{len(to_sell)} lu_w={lu_weight:.0%}")


def update_daily_pnl(engine, trade_date):
    positions = get_positions(engine)
    cash = get_cash(engine)
    initial = 1_000_000

    position_value = 0
    if positions:
        codes = list(positions.keys())
        cl = ",".join([f"'{c}'" for c in codes])
        prices = pd.read_sql(
            f"SELECT code, close FROM stock_daily WHERE code IN ({cl}) AND trade_date=%s",
            engine, params=(trade_date,),
        )
        pm = dict(zip(prices["code"], prices["close"]))
        for c, pos in positions.items():
            position_value += pos["quantity"] * pm.get(c, pos["entry_price"])

    total_value = cash + position_value

    # 日收益
    prev = pd.read_sql(
        "SELECT total_value FROM paper_daily_pnl WHERE account_id=%s ORDER BY trade_date DESC LIMIT 1",
        engine, params=(ACCOUNT_ID,),
    )
    prev_tv = float(prev.iloc[0]["total_value"]) if not prev.empty else initial
    daily_ret = (total_value / prev_tv - 1) if prev_tv > 0 else 0

    # 回撤
    peak = pd.read_sql(
        "SELECT MAX(total_value) as peak FROM paper_daily_pnl WHERE account_id=%s",
        engine, params=(ACCOUNT_ID,),
    ).iloc[0]["peak"]
    peak_val = float(peak) if peak and not pd.isna(peak) else initial
    peak_val = max(peak_val, initial, total_value)
    drawdown = total_value / peak_val - 1

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_daily_pnl (account_id, trade_date, cash, position_value,
                total_value, daily_return, drawdown)
            VALUES (:aid, :td, :cash, :pv, :tv, :dr, :dd)
            ON CONFLICT (account_id, trade_date) DO UPDATE SET
                total_value=:tv, daily_return=:dr, drawdown=:dd
        """), {"aid": ACCOUNT_ID, "td": trade_date, "cash": cash,
               "pv": position_value, "tv": total_value,
               "dr": daily_ret, "dd": drawdown})

    logger.info(f"  估值: 总{total_value:,.0f} 现金{cash:,.0f} 持仓{position_value:,.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    engine = get_engine()

    if not args.no_sync:
        sync_data(engine)

    # ── 交易日 ──
    trade_date = pd.Timestamp(args.date) if args.date else pd.read_sql(
        "SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0]
    trade_date = pd.Timestamp(trade_date)
    logger.info(f"交易日: {trade_date.date()}")

    # ── CSI1000 趋势 ──
    ma_dev, trend_up = get_csi1k_deviation(engine, trade_date)
    lu_weight = np.clip(0.3 + ma_dev * 10, 0.1, 0.7)
    logger.info(f"CSI1000偏离MA60: {ma_dev*100:+.1f}% → lu_w={lu_weight:.0%} sc_w={1-lu_weight:.0%} {'↑强势' if trend_up else '↓弱势'}")

    # ── 涨停侧候选池（成交额 Top-1000）──
    pre_start = trade_date - timedelta(days=max(LU_LOOKBACK, LD_LOOKBACK) + 60)
    lu_codes = set(pd.read_sql(f"""
        SELECT code FROM (
            SELECT code, ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rn
            FROM stock_daily WHERE trade_date BETWEEN %s AND %s
            GROUP BY code
        ) t WHERE rn <= 1000
    """, engine, params=(pre_start, trade_date)).iloc[:, 0].tolist())

    # ── 小票侧候选池（成交额 1000-3000）──
    sc_codes = set(pd.read_sql(f"""
        SELECT code FROM (
            SELECT code, ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rn
            FROM stock_daily WHERE trade_date BETWEEN %s AND %s
            GROUP BY code
        ) t WHERE rn BETWEEN 1000 AND 3000
    """, engine, params=(pre_start, trade_date)).iloc[:, 0].tolist())

    all_codes = lu_codes | sc_codes

    # ── 非ST过滤 ──
    st_set = set(pd.read_sql("SELECT code FROM stock_basic WHERE is_st=TRUE", engine)["code"].tolist())
    sc_codes -= st_set
    lu_codes -= st_set
    logger.info(f"涨停候选: {len(lu_codes)}只 | 小票候选: {len(sc_codes)}只")

    # ── 加载数据 ──
    daily = load_daily(engine, all_codes, pre_start, trade_date)
    extra = load_extra(engine, lu_codes, pre_start, trade_date)
    logger.info(f"日线: {len(daily)}行 | 市值: {len(extra)}行")

    # ── 涨停侧筛选 ──
    lu_picks = lu_screening(trade_date, daily, extra, lu_codes)
    logger.info(f"涨停侧: {len(lu_picks)}只通过")

    # ── 小票侧筛选 ──
    sc_daily = daily[daily["code"].isin(sc_codes)].copy()
    sc_picks = sc_screening(trade_date, sc_daily, sc_codes)
    logger.info(f"小票侧: {len(sc_picks)}只候选")

    # ── 估值（有持仓时才记录，首日跳过）──
    positions = get_positions(engine)
    has_history = len(pd.read_sql(
        "SELECT 1 FROM paper_daily_pnl WHERE account_id=%s LIMIT 1",
        engine, params=(ACCOUNT_ID,))) > 0
    if not args.dry_run and (positions or has_history):
        update_daily_pnl(engine, trade_date)

    # ── 执行（T+1生效）──
    execute(engine, trade_date, lu_picks, sc_picks, lu_weight, dry_run=args.dry_run)

    engine.dispose()


if __name__ == "__main__":
    main()
