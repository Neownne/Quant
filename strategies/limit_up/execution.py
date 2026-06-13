"""涨停策略 T+1 交易执行引擎。"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text

from config.settings import TradingConfig
from data.loader import get_next_open, get_next_trading_date, get_prev_close, get_today_close


LU_PCT = TradingConfig.LIMIT_UP_PCT
LD_PCT = TradingConfig.LIMIT_DOWN_PCT
COMMISSION = TradingConfig.COMMISSION
STAMP_DUTY = TradingConfig.STAMP_DUTY
SLIPPAGE = TradingConfig.SLIPPAGE


def _get_account(engine, account_id):
    row = pd.read_sql(
        text("SELECT cash, initial_capital FROM paper_account WHERE id = :aid"),
        engine,
        params={"aid": account_id},
    ).iloc[0]
    return float(row["cash"]), float(row["initial_capital"])


def _calc_trade_cost(amount: float, direction: str) -> float:
    """计算单笔交易成本。"""
    if direction == "BUY":
        return amount * (COMMISSION + SLIPPAGE)
    return amount * (COMMISSION + STAMP_DUTY + SLIPPAGE)


def execute(
    engine,
    account_id: int,
    run_id: int,
    trade_date,
    signals: list[tuple[str, int, float]],
    positions: dict,
    top_n: int = 5,
    stop_loss_pct: float = 0.08,
    dry_run: bool = False,
):
    """T+1 开盘执行涨停策略调仓。

    Parameters
    ----------
    engine
    account_id, run_id: 账户与运行ID
    trade_date: 信号生成日（T日）
    signals: [(code, score/lu_count, close), ...] 已排序的目标持仓
    positions: {code: {entry_date, entry_price, quantity}} 当前持仓
    top_n: 目标持仓数
    stop_loss_pct: 个股止损比例
    dry_run: 为 True 时只打印不写入

    Returns
    -------
    dict
    """
    trade_date = pd.Timestamp(trade_date)
    cash, _ = _get_account(engine, account_id)

    target_set = set(s[0] for s in signals[:top_n])
    current_set = set(positions.keys())

    # ── 个股止损：T日收盘跌破成本 stop_loss_pct 强制卖出 ──
    stop_loss = set()
    for code in current_set:
        pos = positions[code]
        today_close = get_today_close(engine, code, trade_date)
        if today_close is not None and today_close < pos["entry_price"] * (1 - stop_loss_pct):
            stop_loss.add(code)
            logger.info(f"  止损 {code}: 入场{pos['entry_price']:.2f}→现价{today_close:.2f} "
                        f"({today_close/pos['entry_price']-1:+.1%})")

    to_buy = target_set - current_set
    to_sell = (current_set - target_set) | stop_loss
    to_hold = (target_set & current_set) - stop_loss

    # ── T+1 执行日 ──
    next_date = get_next_trading_date(engine, trade_date)
    if next_date is None:
        logger.warning("无后续交易日，跳过执行（等次日数据就绪）")
        return {"executed": False}

    orders = []
    to_sell_delayed = []  # 跌停封死顺延

    # ── 卖出 ──
    for code in to_sell:
        pos = positions[code]
        prev_close = get_prev_close(engine, code, trade_date)
        today_close = get_today_close(engine, code, trade_date)

        # 跌停判断：当日收盘相对前日收盘跌幅 >= |LD_PCT|
        if prev_close and today_close and (today_close / prev_close - 1) <= LD_PCT:
            logger.info(f"  {code} 跌停封死无法卖出，顺延到次日")
            to_sell_delayed.append(code)
            continue

        sell_price = get_next_open(engine, code, next_date)
        if sell_price is None:
            sell_price = today_close if today_close is not None else pos["entry_price"]
            logger.info(f"  {code} T+1开盘价不可用，用今日收盘价 {sell_price:.2f}")

        qty = pos["quantity"]
        gross = (sell_price - pos["entry_price"]) * qty
        cost = _calc_trade_cost(sell_price * qty, "SELL")
        pnl_net = gross - cost

        orders.append({
            "code": code, "direction": "SELL",
            "price": sell_price, "quantity": qty,
            "entry_date": pos["entry_date"], "exit_date": next_date,
            "pnl": pnl_net, "cost": cost,
        })

    # ── 买入（先卖后买）──
    # 跌停顺延的卖出金额不计入可用现金
    to_sell_actual = [c for c in to_sell if c not in to_sell_delayed]
    actual_sell_proceeds = sum(
        next((o["price"] * o["quantity"] for o in orders if o["code"] == c and o["direction"] == "SELL"), 0)
        for c in to_sell_actual
    )

    sig_map = {s[0]: s[2] for s in signals}
    n_buy = len(to_buy)
    per_stock_cash = 0.0
    if n_buy > 0 and (cash + actual_sell_proceeds) > 0:
        # 组合净值近似 = 现金 + 持仓市值（按成本）
        portfolio_value = cash + sum(p["entry_price"] * p["quantity"] for p in positions.values())
        target_per_stock = portfolio_value / max(top_n, 1)
        max_per_stock = target_per_stock * 1.5
        raw_per_stock = (cash + actual_sell_proceeds) / n_buy
        per_stock_cash = min(raw_per_stock, max_per_stock) * 0.98  # 留2%缓冲

    no_open_warned = set()
    for code in to_buy:
        buy_price = get_next_open(engine, code, next_date)
        if buy_price is None:
            buy_price = sig_map.get(code, 0)
            if code not in no_open_warned:
                logger.info(f"  {code} T+1开盘价不可用，用收盘价 {buy_price:.2f}")
                no_open_warned.add(code)

        # 涨停判断：T+1开盘价相对T日收盘价涨幅 >= LU_PCT 则跳过
        sig_close = sig_map.get(code, 0)
        if sig_close and buy_price and (buy_price / sig_close - 1) >= LU_PCT:
            logger.info(f"  {code} 开盘涨停(+{buy_price/sig_close-1:.1%})无法买入，顺延")
            continue

        qty = int(per_stock_cash / buy_price / 100) * 100 if buy_price > 0 else 0
        if qty <= 0:
            logger.warning(f"  {code} qty=0 (price={buy_price:.2f} per_stock={per_stock_cash:.0f})，跳过买入")
            continue

        cost = _calc_trade_cost(buy_price * qty, "BUY")
        orders.append({
            "code": code, "direction": "BUY",
            "price": buy_price, "quantity": qty,
            "entry_date": next_date, "exit_date": None,
            "pnl": -cost, "cost": cost,
        })

    if dry_run:
        delay_info = f" (跌停延后{len(to_sell_delayed)}只)" if to_sell_delayed else ""
        logger.info(f"  [DRY RUN] 买入{len([o for o in orders if o['direction']=='BUY'])}只 "
                    f"卖出{len([o for o in orders if o['direction']=='SELL'])}只 "
                    f"持有{len(to_hold)}只{delay_info}")
        for o in orders:
            logger.info(f"    {o['direction']} {o['code']} @ {o['price']:.2f} x {o['quantity']}股")
        return {"executed": False, "orders": orders, "to_sell_delayed": to_sell_delayed}

    # ── 写入 DB ──
    buy_orders = [o for o in orders if o["direction"] == "BUY"]
    sell_orders = [o for o in orders if o["direction"] == "SELL"]

    total_buy = sum(o["price"] * o["quantity"] for o in buy_orders)
    total_sell = sum(o["price"] * o["quantity"] for o in sell_orders)
    total_cost = sum(o["cost"] for o in orders)
    new_cash = cash + total_sell - total_buy - total_cost

    with engine.begin() as conn:
        for o in sell_orders:
            pos = positions[o["code"]]
            conn.execute(text("""
                UPDATE paper_positions SET exit_date = :ed, exit_price = :ep,
                    pnl = :pnl, pnl_pct = :pct
                WHERE run_id = :rid AND stock_code = :code AND exit_date IS NULL
            """), {
                "ed": next_date, "ep": o["price"], "pnl": o["pnl"],
                "pct": o["pnl"] / (pos["entry_price"] * pos["quantity"]) if pos["quantity"] > 0 else 0,
                "rid": run_id, "code": o["code"],
            })

        for o in buy_orders:
            conn.execute(text("""
                INSERT INTO paper_positions (run_id, stock_code, entry_date, entry_price, quantity, pnl, pnl_pct)
                VALUES (:rid, :code, :ed, :ep, :qty, 0, 0)
            """), {
                "rid": run_id, "code": o["code"], "ed": next_date,
                "ep": o["price"], "qty": o["quantity"],
            })

        conn.execute(text("UPDATE paper_account SET cash = :c WHERE id = :aid"),
                     {"c": new_cash, "aid": account_id})

        for o in orders:
            conn.execute(text("""
                INSERT INTO paper_orders (account_id, code, direction, price, volume, status)
                VALUES (:aid, :code, :dir, :price, :qty, 'filled')
            """), {
                "aid": account_id, "code": o["code"], "dir": o["direction"],
                "price": o["price"], "qty": o["quantity"],
            })

    delay_info = f" 跌停延后{len(to_sell_delayed)}只" if to_sell_delayed else ""
    logger.info(f"  执行完成: 买{len(buy_orders)}只 卖{len(sell_orders)}只 持{len(to_hold)}只{delay_info}")
    return {
        "executed": True,
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "to_sell_delayed": to_sell_delayed,
        "new_cash": new_cash,
        "total_cost": total_cost,
    }
