"""涨停策略模拟盘日净值与回撤计算。"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text

from data.loader import get_position_values


def _get_latest_close(engine, code, before_date):
    """获取某日期前最近一个交易日的收盘价（停牌/数据缺失时回退）。"""
    row = pd.read_sql(
        text("SELECT close FROM stock_daily WHERE code = :code AND trade_date <= :date "
             "ORDER BY trade_date DESC LIMIT 1"),
        engine,
        params={"code": code, "date": before_date},
    )
    return float(row.iloc[0]["close"]) if not row.empty else None


def get_account(engine, account_id):
    """获取账户现金和初始本金。"""
    row = pd.read_sql(
        text("SELECT cash, initial_capital FROM paper_account WHERE id = :aid"),
        engine,
        params={"aid": account_id},
    ).iloc[0]
    return float(row["cash"]), float(row["initial_capital"])


def get_current_positions(engine, run_id):
    """获取当前未平仓持仓。"""
    rows = pd.read_sql(
        text("SELECT stock_code, entry_date, entry_price, quantity FROM paper_positions "
             "WHERE run_id = :rid AND exit_date IS NULL"),
        engine,
        params={"rid": run_id},
    )
    return {
        r["stock_code"]: {"entry_date": r["entry_date"], "entry_price": r["entry_price"], "quantity": r["quantity"]}
        for _, r in rows.iterrows()
    }


def update_daily_pnl(engine, account_id, run_id, trade_date, initial_capital=None):
    """更新当日账户估值、日收益、回撤。

    使用 paper_daily_pnl 表中上一交易日的 total_value 计算日收益。
    """
    positions = get_current_positions(engine, run_id)
    cash, initial = get_account(engine, account_id)
    if initial_capital is not None:
        initial = initial_capital

    codes = list(positions.keys())
    price_map = get_position_values(engine, codes, trade_date)

    position_value = 0.0
    for c in codes:
        qty = positions[c]["quantity"]
        if c in price_map:
            position_value += qty * price_map[c]
        else:
            # 停牌/缺数据：回退到最近收盘价，再回退到入场价
            latest = _get_latest_close(engine, c, trade_date)
            px = latest if latest else positions[c]["entry_price"]
            position_value += qty * px
    total_value = cash + position_value

    # 上一交易日净值
    prev_row = pd.read_sql(
        text("SELECT trade_date, total_value FROM paper_daily_pnl WHERE account_id = :aid "
             "ORDER BY trade_date DESC LIMIT 1"),
        engine,
        params={"aid": account_id},
    )
    prev_total = float(prev_row.iloc[0]["total_value"]) if not prev_row.empty else initial
    daily_ret = (total_value / prev_total - 1) if prev_total > 0 else 0

    # 跳空告警
    if not prev_row.empty:
        prev_date = pd.Timestamp(prev_row.iloc[0]["trade_date"])
        gap_days = (pd.Timestamp(trade_date) - prev_date).days
        if gap_days > 3:
            logger.warning(f"距上次 PnL 记录 {prev_date.date()} 已 {gap_days} 天，"
                           f"daily_return 反映多日累积")

    # 历史峰值
    peak_row = pd.read_sql(
        text("SELECT MAX(total_value) AS peak FROM paper_daily_pnl WHERE account_id = :aid"),
        engine,
        params={"aid": account_id},
    ).iloc[0]["peak"]
    peak_value = float(peak_row) if peak_row and not pd.isna(peak_row) else initial
    peak_value = max(peak_value, initial, total_value)
    drawdown = total_value / peak_value - 1

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_daily_pnl (account_id, trade_date, cash, position_value,
                total_value, daily_return, drawdown)
            VALUES (:aid, :td, :cash, :pv, :tv, :dr, :dd)
            ON CONFLICT (account_id, trade_date) DO UPDATE SET
                cash = :cash2, position_value = :pv2, total_value = :tv2,
                daily_return = :dr2, drawdown = :dd2
        """), {
            "aid": account_id, "td": trade_date,
            "cash": cash, "pv": position_value, "tv": total_value,
            "dr": daily_ret, "dd": drawdown,
            "cash2": cash, "pv2": position_value, "tv2": total_value,
            "dr2": daily_ret, "dd2": drawdown,
        })

    return {"cash": cash, "position_value": position_value, "total_value": total_value,
            "daily_return": daily_ret, "drawdown": drawdown}
