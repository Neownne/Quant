#!/usr/bin/env python
"""涨停 Top-5 每日模拟盘。

每日收盘后运行：
  1. 同步最新日线数据
  2. 5条件筛选 → 取Top-5
  3. 对比当前持仓，生成买卖信号
  4. T+1 开盘执行 → 写入 paper_* 表

用法:
    python scripts/run_daily_paper_lu.py                # 今日
    python scripts/run_daily_paper_lu.py --date 2026-06-05  # 指定日期
    python scripts/run_daily_paper_lu.py --dry-run       # 试运行
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from data.sync import sync_stock_daily, sync_daily_extra
from config.settings import TradingConfig

# ── 策略参数 ──
ACCOUNT_ID = 1
RUN_ID = 1
TOP_N = 5
MCAP_MIN, MCAP_MAX = 50.0, 300.0
PRICE_MIN, PRICE_MAX = 5.0, 50.0
LU_PCT, LU_LOOKBACK, LU_COUNT = 0.099, 20, 1
LD_PCT, LD_LOOKBACK = -0.099, 10
MIN_CONDITIONS = 5
MIN_LISTED_DAYS = 120
COMMISSION = TradingConfig.COMMISSION
STAMP_DUTY = TradingConfig.STAMP_DUTY
SLIPPAGE = TradingConfig.SLIPPAGE


def sync_data(engine):
    """增量同步日线 + 市值数据。"""
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"同步数据至 {today} ...")
    sync_stock_daily(engine, start_date=today, workers=8)
    sync_daily_extra(engine, start_date=today, workers=8)


def get_latest_date(engine):
    return pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0]


def load_daily_data(engine, codes, pre_start, end_date):
    """加载日线 + 计算均线和收益。"""
    cl = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(
        f"SELECT code, trade_date, open, close FROM stock_daily "
        f"WHERE code IN ({cl}) AND trade_date BETWEEN %s AND %s "
        f"ORDER BY code, trade_date",
        engine, params=(pre_start, end_date),
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["ma5"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(10, min_periods=10).mean())
    return df


def load_mcap_data(engine, codes, pre_start, end_date):
    cl = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(
        f"SELECT code, trade_date, market_cap FROM stock_daily_extra "
        f"WHERE code IN ({cl}) AND trade_date BETWEEN %s AND %s",
        engine, params=(pre_start, end_date),
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def run_screening(today, daily, extra_df, code_set):
    """在 today 执行5条件筛选，返回 [(code, lu_count), ...]。"""
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    if today not in daily_by_date:
        return []

    td = daily_by_date[today]
    lookback_start = today - timedelta(days=max(LU_LOOKBACK, LD_LOOKBACK) + 5)
    lb = daily[(daily["trade_date"] >= lookback_start) & (daily["trade_date"] <= today)]

    lu_counts = lb[lb["ret"] >= LU_PCT].groupby("code").size()
    ld_codes = set(lb[lb["ret"] <= LD_PCT]["code"].unique())

    # 市值
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
              not pd.isna(mcap_s.loc[code]) and MCAP_MIN <= mcap_s.loc[code] <= MCAP_MAX)
        c2 = PRICE_MIN <= close_p <= PRICE_MAX
        c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
        c4 = int(lu_counts.get(code, 0)) > LU_COUNT
        c5 = code not in ld_codes

        if sum([c1, c2, c3, c4, c5]) >= MIN_CONDITIONS:
            passed.append((code, int(lu_counts.get(code, 0)), float(close_p)))

    passed.sort(key=lambda x: x[1], reverse=True)
    return passed[:TOP_N]


def get_current_positions(engine):
    """获取当前持仓 (未平仓的)。"""
    rows = pd.read_sql(
        "SELECT stock_code, entry_date, entry_price, quantity FROM paper_positions "
        "WHERE run_id = %s AND exit_date IS NULL", engine, params=(RUN_ID,)
    )
    return {r["stock_code"]: {"entry_date": r["entry_date"],
                               "entry_price": r["entry_price"],
                               "quantity": r["quantity"]}
            for _, r in rows.iterrows()}


def get_account(engine):
    row = pd.read_sql(
        "SELECT cash, initial_capital FROM paper_account WHERE id = %s",
        engine, params=(ACCOUNT_ID,)
    ).iloc[0]
    return float(row["cash"]), float(row["initial_capital"])


def execute(engine, trade_date, signals, positions, dry_run=False):
    """
    signals: [(code, lu_count, close), ...]  — 今日筛选出的Top-5
    positions: {code: {entry_date, entry_price, quantity}}  — 当前持仓
    """
    tc = TradingConfig()
    cash, _ = get_account(engine)

    target_set = set(s[0] for s in signals)
    current_set = set(positions.keys())

    to_buy = target_set - current_set
    to_sell = current_set - target_set
    to_hold = target_set & current_set

    orders = []
    # 找最近交易日 → T+1执行（跳过周末/假日）
    next_row = pd.read_sql(
        "SELECT MIN(trade_date) FROM stock_daily WHERE trade_date > %s",
        engine, params=(trade_date,),
    )
    if next_row.empty or next_row.iloc[0, 0] is None or pd.isna(next_row.iloc[0, 0]):
        logger.warning("无后续交易日，跳过执行")
        return
    next_date = pd.Timestamp(next_row.iloc[0, 0])

    # ── 卖出 ──
    for code in to_sell:
        pos = positions[code]
        # 取T+1开盘价（如果还没到就用收盘价）
        open_price = _get_next_open(engine, code, next_date)
        sell_price = open_price if open_price else pos["entry_price"]
        qty = pos["quantity"]
        pnl = (sell_price - pos["entry_price"]) * qty
        cost = sell_price * qty * (COMMISSION + STAMP_DUTY + SLIPPAGE)
        pnl_net = pnl - cost

        orders.append({
            "code": code, "direction": "SELL",
            "price": sell_price, "quantity": qty,
            "entry_date": pos["entry_date"], "exit_date": next_date,
            "pnl": pnl_net,
        })

    # ── 买入 ──
    n_buy = len(to_buy)
    if n_buy > 0:
        # 用当日收盘价（更接近T+1开盘）估算卖出释放的现金
        sell_est = 0
        for c in to_sell:
            pos = positions[c]
            # 用筛选时的收盘价估算，比 entry_price 更接近实际
            cp_row = pd.read_sql("SELECT close FROM stock_daily WHERE code=%s AND trade_date=%s",
                                 engine, params=(c, trade_date))
            est_price = float(cp_row.iloc[0]["close"]) if not cp_row.empty else pos["entry_price"]
            sell_est += est_price * pos["quantity"]
        total_cash = cash + sell_est
        per_stock_cash = total_cash / n_buy

    for code in to_buy:
        open_price = _get_next_open(engine, code, next_date)
        buy_price = open_price if open_price else signals[[s[0] for s in signals].index(code)][2]
        # 按手取整（100股）
        qty = int(per_stock_cash / buy_price / 100) * 100 if n_buy > 0 else 0
        if qty <= 0:
            continue
        cost = buy_price * qty * (COMMISSION + SLIPPAGE)
        orders.append({
            "code": code, "direction": "BUY",
            "price": buy_price, "quantity": qty,
            "entry_date": next_date, "exit_date": None,
            "pnl": -cost,
        })

    if dry_run:
        logger.info(f"  [DRY RUN] 买入{len(to_buy)}只 卖出{len(to_sell)}只 持有{len(to_hold)}只")
        for o in orders:
            logger.info(f"    {o['direction']} {o['code']} @ {o['price']:.2f} x {o['quantity']}股")
        return

    # ── 写入 DB ──
    with engine.begin() as conn:
        # 更新持仓
        for code in to_sell:
            pos = positions[code]
            o = [x for x in orders if x["code"] == code and x["direction"] == "SELL"][0]
            conn.execute(text("""
                UPDATE paper_positions SET exit_date = :ed, exit_price = :ep,
                pnl = :pnl, pnl_pct = :pct
                WHERE run_id = :rid AND stock_code = :code AND exit_date IS NULL
            """), {
                "ed": next_date, "ep": o["price"], "pnl": o["pnl"],
                "pct": o["pnl"] / (pos["entry_price"] * pos["quantity"]) if pos["quantity"] > 0 else 0,
                "rid": RUN_ID, "code": code,
            })

        for code in to_buy:
            o = [x for x in orders if x["code"] == code and x["direction"] == "BUY"][0]
            conn.execute(text("""
                INSERT INTO paper_positions (run_id, stock_code,
                    entry_date, entry_price, quantity, pnl, pnl_pct)
                VALUES (:rid, :code, :ed, :ep, :qty, 0, 0)
            """), {
                "rid": RUN_ID, "code": code, "ed": next_date,
                "ep": o["price"], "qty": o["quantity"],
            })

        # 更新现金
        total_buy = sum(o["price"] * o["quantity"] for o in orders if o["direction"] == "BUY")
        total_sell = sum(o["price"] * o["quantity"] for o in orders if o["direction"] == "SELL")
        new_cash = cash - total_buy + total_sell
        conn.execute(text("UPDATE paper_account SET cash = :c WHERE id = :aid"),
                     {"c": new_cash, "aid": ACCOUNT_ID})

        # 写订单日志
        for o in orders:
            conn.execute(text("""
                INSERT INTO paper_orders (account_id, code, direction, price, volume, status)
                VALUES (:aid, :code, :dir, :price, :qty, 'filled')
            """), {
                "aid": ACCOUNT_ID, "code": o["code"], "dir": o["direction"],
                "price": o["price"], "qty": o["quantity"],
            })

        # 写信号
        for s in signals:
            conn.execute(text("""
                INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                VALUES (1, :sd, :code, :score, :rank)
            """), {
                "sd": trade_date, "code": s[0], "score": s[1], "rank": signals.index(s) + 1,
            })

    logger.info(f"  执行完成: 买{len(to_buy)}只 卖{len(to_sell)}只 持{len(to_hold)}只")


def update_daily_pnl(engine, trade_date):
    """更新当日估值。"""
    positions = get_current_positions(engine)
    cash, initial = get_account(engine)

    # 取最新收盘价
    codes = list(positions.keys())
    position_value = 0
    if not codes:
        total_value = cash
    else:
        cl = ",".join([f"'{c}'" for c in codes])
        prices = pd.read_sql(
            f"SELECT code, close FROM stock_daily "
            f"WHERE code IN ({cl}) AND trade_date = %s",
            engine, params=(trade_date,),
        )
        price_map = dict(zip(prices["code"], prices["close"]))
        position_value = sum(
            positions[c]["quantity"] * price_map.get(c, positions[c]["entry_price"])
            for c in codes
        )
        total_value = cash + position_value

    # 日收益：相对前一日的变动
    prev_tv = pd.read_sql(
        "SELECT total_value FROM paper_daily_pnl WHERE account_id = %s ORDER BY trade_date DESC LIMIT 1",
        engine, params=(ACCOUNT_ID,),
    )
    prev_total = float(prev_tv.iloc[0]["total_value"]) if not prev_tv.empty else initial
    daily_ret = (total_value / prev_total - 1) if prev_total > 0 else 0

    # 查历史最高净值算回撤
    peak = pd.read_sql(
        "SELECT MAX(total_value) as peak FROM paper_daily_pnl WHERE account_id = %s",
        engine, params=(ACCOUNT_ID,),
    ).iloc[0]["peak"]
    peak_value = float(peak) if peak and not pd.isna(peak) else initial
    peak_value = max(peak_value, initial, total_value)
    drawdown = total_value / peak_value - 1

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_daily_pnl (account_id, trade_date, cash, position_value,
                total_value, daily_return, drawdown)
            VALUES (:aid, :td, :cash, :pv, :tv, :dr, :dd)
            ON CONFLICT (account_id, trade_date) DO NOTHING
        """), {
            "aid": ACCOUNT_ID, "td": trade_date,
            "cash": cash, "pv": position_value,
            "tv": total_value, "dr": daily_ret, "dd": drawdown,
        })

    logger.info(f"  估值更新: 总资产 {total_value:,.0f} | 现金 {cash:,.0f}")


def _get_next_open(engine, code, next_date):
    """获取 T+1 开盘价，如果还没发生则返回 None。"""
    try:
        row = pd.read_sql(
            "SELECT open FROM stock_daily WHERE code = %s AND trade_date = %s",
            engine, params=(code, next_date),
        )
        return float(row.iloc[0]["open"]) if not row.empty else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="回测日期 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    tc = TradingConfig()

    # ── 1. 同步数据 ──
    if not args.no_sync:
        sync_data(engine)

    # ── 2. 确定交易日 ──
    trade_date = pd.Timestamp(args.date) if args.date else get_latest_date(engine)
    trade_date = pd.Timestamp(trade_date)
    logger.info(f"交易日: {trade_date.date()}")

    # ── 3. 候选池 ──
    code_set = set(pd.read_sql(
        f"SELECT code FROM stock_basic WHERE is_st = FALSE AND list_date <= %s",
        engine, params=(trade_date - timedelta(days=MIN_LISTED_DAYS),),
    )["code"].tolist())
    logger.info(f"候选池: {len(code_set)} 只")

    # ── 4. 加载数据 ──
    pre_start = trade_date - timedelta(days=max(LU_LOOKBACK, LD_LOOKBACK) + 30)
    daily = load_daily_data(engine, code_set, pre_start, trade_date)
    extra = load_mcap_data(engine, code_set, pre_start, trade_date)
    logger.info(f"日线: {len(daily)} 行 | 市值: {len(extra)} 行")

    # ── 5. 筛选 ──
    signals = run_screening(trade_date, daily, extra, code_set)
    if not signals:
        logger.warning("无股票通过筛选，检查市场状态。")
        return

    logger.info(f"筛选结果: {len(signals)} 只")
    for s in signals:
        logger.info(f"  {s[0]} 涨停{s[1]}次 收盘{s[2]:.2f}")

    # ── 6. 估值（先记今日净值，再执行明日交易）──
    positions = get_current_positions(engine)
    if not args.dry_run and positions:
        update_daily_pnl(engine, trade_date)

    # ── 7. 执行（T+1生效）──
    execute(engine, trade_date, signals, positions, dry_run=args.dry_run)
    # 首日建仓后也记录：执行完立即用今日收盘价估值新持仓
    if not args.dry_run and not positions:
        new_positions = get_current_positions(engine)
        if new_positions:
            update_daily_pnl(engine, trade_date)

    engine.dispose()


if __name__ == "__main__":
    main()
