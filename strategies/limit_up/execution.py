"""涨停策略 T 日收盘执行引擎。

主路径：T 日收盘价成交（模拟 14:50 实时行情 → 收盘下单）。
顺延路径：涨跌停封板时顺延到 T+1 开盘价判断。
"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text

from config.settings import TradingConfig
from data.loader import (
    get_next_open,
    get_next_trading_date,
    get_position_values,
    get_prev_close,
    get_today_close,
)

LU_PCT = TradingConfig.LIMIT_UP_PCT              # 回退值
LD_PCT = TradingConfig.LIMIT_DOWN_PCT            # 回退值
COMMISSION = TradingConfig.COMMISSION
STAMP_DUTY = TradingConfig.STAMP_DUTY
SLIPPAGE = TradingConfig.SLIPPAGE
BUY_COST_RATIO = COMMISSION + SLIPPAGE           # 买入侧成本率
SELL_COST_RATIO = COMMISSION + STAMP_DUTY + SLIPPAGE  # 卖出侧成本率
IS_LIMIT_UP = TradingConfig.is_at_limit_up        # 板别感知涨停
IS_LIMIT_DOWN = TradingConfig.is_at_limit_down    # 板别感知跌停


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

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
        return amount * BUY_COST_RATIO
    return amount * SELL_COST_RATIO


def _compute_weights(
    target_codes: list[str],
    signals: list[tuple[str, int, float]],
    mode: str = "equal",
    daily_df: pd.DataFrame | None = None,
    lookback_days: int = 20,
) -> dict[str, float]:
    """计算目标仓位权重。

    Parameters
    ----------
    target_codes : 目标股票代码列表
    signals : [(code, score, close), ...] 按评分降序排列
    mode : "equal" | "volatility_inverse" | "score_weighted"
    daily_df : 日线数据（volatility_inverse 模式需要）
    lookback_days : 波动率回溯窗口

    Returns
    -------
    dict[str, float]  code -> weight（和为 1.0）
    """
    n = len(target_codes)
    if n == 0:
        return {}

    if mode == "equal":
        return {c: 1.0 / n for c in target_codes}

    if mode == "score_weighted":
        sig_map = {s[0]: max(float(s[1]), 0.01) for s in signals if s[0] in target_codes}
        total = sum(sig_map.values())
        if total <= 0:
            return {c: 1.0 / n for c in target_codes}
        return {c: sig_map.get(c, 0.01) / total for c in target_codes}

    if mode == "volatility_inverse":
        if daily_df is None or daily_df.empty:
            logger.warning("volatility_inverse 模式需要 daily_df，回退到等权")
            return {c: 1.0 / n for c in target_codes}
        vols = {}
        for code in target_codes:
            code_data = daily_df[daily_df["code"] == code]
            if len(code_data) < 5:
                vols[code] = 1.0
                continue
            rets = code_data.sort_values("trade_date")["close"].pct_change().dropna()
            if len(rets) < 2:
                vols[code] = 1.0
                continue
            vols[code] = rets.tail(lookback_days).std()
        inv_vols = {c: 1.0 / max(v, 0.001) for c, v in vols.items()}
        total = sum(inv_vols.values())
        if total <= 0:
            return {c: 1.0 / n for c in target_codes}
        return {c: inv_vols[c] / total for c in target_codes}

    return {c: 1.0 / n for c in target_codes}


# ═══════════════════════════════════════════════════════════════
#  顺延订单持久化
# ═══════════════════════════════════════════════════════════════

def _get_pending_orders(engine, account_id: int) -> list[dict]:
    """获取前日未处理的涨跌停顺延订单。"""
    rows = pd.read_sql(
        text("""
            SELECT id, code, direction, price, volume
            FROM paper_orders
            WHERE account_id = :aid AND status = 'pending_limit'
            ORDER BY id
        """),
        engine,
        params={"aid": account_id},
    )
    return rows.to_dict(orient="records") if not rows.empty else []


def _resolve_pending_orders(engine, account_id: int, run_id: int, trade_date, positions: dict):
    """处理前日顺延订单：用 T 日收盘价判断是否仍封板，未封板则执行。

    返回 (resolved_orders, still_pending)，其中 resolved_orders 可直接写入 DB。
    """
    pending = _get_pending_orders(engine, account_id)
    if not pending:
        return [], []

    logger.info(f"处理前日顺延订单 {len(pending)} 笔")
    resolved = []
    still_pending = []

    for po in pending:
        code, direction, ref_price = po["code"], po["direction"], float(po["price"])
        today_close = get_today_close(engine, code, trade_date)
        prev_close = get_prev_close(engine, code, trade_date)

        if direction == "SELL":
            if code not in positions:
                # 已无持仓，标记取消
                _cancel_pending(engine, po["id"], "已无持仓")
                continue

            pos = positions[code]
            sell_price = today_close if today_close else pos["entry_price"]
            # T 日仍跌停？
            if prev_close and sell_price and IS_LIMIT_DOWN(sell_price, prev_close, code):
                still_pending.append(po)
                continue

            qty = pos["quantity"]
            gross = (sell_price - pos["entry_price"]) * qty
            cost = _calc_trade_cost(sell_price * qty, "SELL")
            pnl_net = gross - cost - (pos["entry_price"] * qty * BUY_COST_RATIO)
            resolved.append({
                "code": code, "direction": "SELL", "price": sell_price,
                "quantity": qty, "entry_date": pos["entry_date"],
                "exit_date": trade_date, "pnl": pnl_net, "cost": cost,
                "pending_id": po["id"],
            })

        elif direction == "BUY":
            # 仍涨停？
            if today_close and prev_close and IS_LIMIT_UP(today_close, prev_close, code):
                still_pending.append(po)
                continue
            # 顺延买入不再执行（原信号已过期，由当日新信号覆盖）
            _cancel_pending(engine, po["id"], "信号过期")

    return resolved, still_pending


def _cancel_pending(engine, order_id: int, reason: str):
    """取消顺延订单。"""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE paper_orders SET status = 'cancelled', note = :note WHERE id = :id"),
            {"note": reason, "id": order_id},
        )


def _write_pending_order(engine, account_id: int, code: str, direction: str,
                         price: float, quantity: int, note: str):
    """持久化涨跌停顺延订单。"""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_orders (account_id, code, direction, price, volume, status, note)
            VALUES (:aid, :code, :dir, :price, :qty, 'pending_limit', :note)
        """), {
            "aid": account_id, "code": code, "dir": direction,
            "price": price, "qty": quantity or 0,
            "note": note,
        })


# ═══════════════════════════════════════════════════════════════
#  执行路径
# ═══════════════════════════════════════════════════════════════

def _execute_at_close(
    engine, trade_date, to_sell: set,
    signals: list, positions: dict, cash: float,
    top_n: int, weight_mode: str, daily_df,
) -> tuple[list[dict], list[dict]]:
    """主路径：T 日收盘价执行。买入按信号排名顺延，直到凑满 top_n。

    Returns
    -------
    (orders, deferred)
        orders: 已成交订单列表
        deferred: 封板顺延项列表 [{code, direction, pos?, t_close, prev_close}]
    """
    orders = []
    deferred = []
    sig_map = {s[0]: s[2] for s in signals}

    # ── 卖出 ──
    actual_sell_proceeds = 0.0
    for code in to_sell:
        pos = positions[code]
        sell_price = get_today_close(engine, code, trade_date)
        if sell_price is None or sell_price <= 0:
            sell_price = pos["entry_price"]

        prev_close = get_prev_close(engine, code, trade_date)
        # 跌停检查（T 日收盘）
        if prev_close and sell_price and IS_LIMIT_DOWN(sell_price, prev_close, code):
            logger.info(f"  {code} T日收盘跌停，顺延到T+1")
            deferred.append({"code": code, "direction": "SELL",
                             "pos": pos, "t_close": sell_price, "prev_close": prev_close})
            continue

        qty = pos["quantity"]
        gross = (sell_price - pos["entry_price"]) * qty
        cost = _calc_trade_cost(sell_price * qty, "SELL")
        # PnL 含买入侧成本
        pnl_net = gross - cost - (pos["entry_price"] * qty * BUY_COST_RATIO)
        actual_sell_proceeds += sell_price * qty
        orders.append({
            "code": code, "direction": "SELL",
            "price": round(sell_price, 2), "quantity": qty,
            "entry_date": pos["entry_date"], "exit_date": trade_date,
            "pnl": pnl_net, "cost": cost,
        })

    # ── 买入：从排名列表按序买入，跳过涨停/已持，凑满 top_n ──
    held_codes = set(positions.keys())
    held_after_sell = held_codes - to_sell  # 卖出后剩余持仓
    available_cash = (cash + actual_sell_proceeds) * 0.98

    # 组合估值（T 日收盘市价）
    current_prices = get_position_values(engine, list(held_codes), trade_date) if held_codes else {}
    position_value = sum(
        current_prices.get(c, positions[c]["entry_price"]) * positions[c]["quantity"]
        for c in held_codes
    )
    portfolio_value = cash + position_value
    target_per_stock = portfolio_value / max(top_n, 1)
    max_per_stock = target_per_stock * 1.5

    buy_count = 0
    for code, score, close_p in signals:
        # 跳过已持仓（含卖出后仍持有的）
        if code in held_after_sell:
            continue
        # 已经买够了
        if len(held_after_sell) + buy_count >= top_n:
            break

        buy_price = get_today_close(engine, code, trade_date)
        if buy_price is None or buy_price <= 0:
            continue

        prev_close = get_prev_close(engine, code, trade_date)
        # 涨停检查（T 日收盘）
        if prev_close and buy_price and IS_LIMIT_UP(buy_price, prev_close, code):
            logger.info(f"  {code} T日收盘涨停，顺延")
            continue  # 涨停跳过，试下一个

        qty = int(available_cash / max(top_n - len(held_after_sell), 1) / buy_price / 100) * 100 if buy_price > 0 else 0
        per_stock_cash = qty * buy_price
        if per_stock_cash > max_per_stock:
            qty = int(max_per_stock / buy_price / 100) * 100
        if qty <= 0:
            continue

        cost = _calc_trade_cost(buy_price * qty, "BUY")
        orders.append({
            "code": code, "direction": "BUY",
            "price": round(buy_price, 2), "quantity": qty,
            "entry_date": trade_date, "exit_date": None,
            "pnl": -cost, "cost": cost,
        })
        buy_count += 1

    return orders, deferred


def _execute_deferred_at_open(
    engine, trade_date, next_date, deferred_items: list[dict], signals: list,
) -> list[dict]:
    """顺延路径：T+1 开盘价执行封板股票。

    Returns
    -------
    list[dict]  已成交订单（封板仍不成交的写入 pending）
    """
    orders = []
    account_id = 1  # 与 execute() 的 account_id 一致

    for item in deferred_items:
        code = item["code"]
        direction = item["direction"]
        open_price = get_next_open(engine, code, next_date)

        if direction == "SELL":
            pos = item["pos"]
            prev_close = item["prev_close"]
            sell_price = open_price if open_price else item["t_close"]

            # T+1 开盘跌停检查
            if prev_close and sell_price and IS_LIMIT_DOWN(sell_price, prev_close, code):
                logger.info(f"  {code} T+1开盘仍跌停，写入pending")
                _write_pending_order(engine, account_id, code, "SELL",
                                     sell_price, pos["quantity"],
                                     f"跌停顺延-T{str(trade_date)[:10]}")
                continue

            qty = pos["quantity"]
            gross = (sell_price - pos["entry_price"]) * qty
            cost = _calc_trade_cost(sell_price * qty, "SELL")
            pnl_net = gross - cost - (pos["entry_price"] * qty * BUY_COST_RATIO)
            orders.append({
                "code": code, "direction": "SELL",
                "price": round(sell_price, 2), "quantity": qty,
                "entry_date": pos["entry_date"], "exit_date": next_date,
                "pnl": pnl_net, "cost": cost,
            })

        elif direction == "BUY":
            t_close = item["t_close"]
            buy_price = open_price if open_price else t_close

            # T+1 开盘涨停检查
            if t_close and buy_price and IS_LIMIT_UP(buy_price, t_close, code):
                logger.info(f"  {code} T+1开盘仍涨停，写入pending")
                _write_pending_order(engine, account_id, code, "BUY",
                                     buy_price, 0,
                                     f"涨停顺延-T{str(trade_date)[:10]}")
                continue

            # 买入（使用 T+1 开盘价，但仓位计算仍基于 T 日组合估值）
            # 简化：顺延买入用固定金额，取 T 日信号的 close 作为参考
            qty = 0  # 顺延买入无原始信号仓位信息，跳过
            logger.info(f"  {code} 顺延买入（T+1开={buy_price:.2f}），待信号重新触发")
            # 顺延买入暂不执行——由次日新信号覆盖

    return orders


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

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
    weight_mode: str = "equal",
    daily_df: pd.DataFrame | None = None,
):
    """T 日收盘执行涨停策略调仓。

    主路径 = T 日收盘价成交。涨跌停封板的顺延到 T+1 开盘价判断。
    不再依赖 get_next_trading_date 返回非 None。

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
    weight_mode: "equal" | "volatility_inverse" | "score_weighted"
    daily_df: 日线数据（volatility_inverse 需要）

    Returns
    -------
    dict
    """
    trade_date = pd.Timestamp(trade_date)
    cash, _ = _get_account(engine, account_id)

    # ── 处理前日顺延订单 ──
    pending_resolved, _ = _resolve_pending_orders(
        engine, account_id, run_id, trade_date, positions,
    )

    target_set = set(s[0] for s in signals[:top_n])
    current_set = set(positions.keys())

    # ── 个股止损：T 日收盘价判断 ──
    stop_loss = set()
    for code in current_set:
        pos = positions[code]
        today_close = get_today_close(engine, code, trade_date)
        if today_close is not None and today_close < pos["entry_price"] * (1 - stop_loss_pct):
            stop_loss.add(code)
            logger.info(f"  止损 {code}: 入场{pos['entry_price']:.2f}→现价{today_close:.2f} "
                        f"({today_close/pos['entry_price']-1:+.1%})")

    to_sell = (current_set - target_set) | stop_loss
    to_hold = (target_set & current_set) - stop_loss

    # ── 主路径：T 日收盘价执行 ──
    primary_orders, deferred_items = _execute_at_close(
        engine, trade_date, to_sell, signals, positions, cash,
        top_n, weight_mode, daily_df,
    )

    # ── 顺延路径：T+1 开盘价（仅封板股票）──
    next_date = get_next_trading_date(engine, trade_date)
    deferred_orders = []
    if next_date is not None and deferred_items:
        deferred_orders = _execute_deferred_at_open(
            engine, trade_date, next_date, deferred_items, signals,
        )
    elif next_date is None and deferred_items:
        logger.warning(f"无后续交易日，{len(deferred_items)} 笔顺延将于次日处理")
        # 将顺延项写入 pending（下次脚本运行时 _resolve_pending_orders 处理）
        for item in deferred_items:
            if item["direction"] == "SELL":
                _write_pending_order(engine, account_id, item["code"], "SELL",
                                     item["t_close"], item["pos"]["quantity"],
                                     f"跌停顺延-T{str(trade_date)[:10]}")
            elif item["direction"] == "BUY":
                _write_pending_order(engine, account_id, item["code"], "BUY",
                                     item["t_close"], 0,
                                     f"涨停顺延-T{str(trade_date)[:10]}")

    orders = pending_resolved + primary_orders + deferred_orders

    if dry_run:
        n_buy = len([o for o in orders if o["direction"] == "BUY"])
        n_sell = len([o for o in orders if o["direction"] == "SELL"])
        logger.info(f"  [DRY RUN] 买入{n_buy}只 卖出{n_sell}只 持有{len(to_hold)}只"
                    f" (顺延{len(deferred_items)}只)")
        for o in orders:
            logger.info(f"    {o['direction']} {o['code']} @ {o['price']:.2f} x {o['quantity']}股")
        return {"executed": False, "orders": orders, "deferred_count": len(deferred_items)}

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
            exit_date = o.get("exit_date", trade_date)
            conn.execute(text("""
                UPDATE paper_positions SET exit_date = :ed, exit_price = :ep,
                    pnl = :pnl, pnl_pct = :pct
                WHERE run_id = :rid AND stock_code = :code AND exit_date IS NULL
            """), {
                "ed": exit_date, "ep": o["price"], "pnl": o["pnl"],
                "pct": o["pnl"] / (pos["entry_price"] * pos["quantity"]) if pos["quantity"] > 0 else 0,
                "rid": run_id, "code": o["code"],
            })

            # 清理对应 pending 订单
            if "pending_id" in o:
                conn.execute(
                    text("UPDATE paper_orders SET status = 'filled' WHERE id = :id"),
                    {"id": o["pending_id"]},
                )

        for o in buy_orders:
            entry_date = o.get("entry_date", trade_date)
            conn.execute(text("""
                INSERT INTO paper_positions (run_id, stock_code, entry_date, entry_price, quantity, pnl, pnl_pct)
                VALUES (:rid, :code, :ed, :ep, :qty, 0, 0)
            """), {
                "rid": run_id, "code": o["code"], "ed": entry_date,
                "ep": o["price"], "qty": o["quantity"],
            })

        conn.execute(text("UPDATE paper_account SET cash = :c WHERE id = :aid"),
                     {"c": new_cash, "aid": account_id})

        for o in orders:
            status = o.get("status", "filled")
            conn.execute(text("""
                INSERT INTO paper_orders (account_id, code, direction, price, volume, status)
                VALUES (:aid, :code, :dir, :price, :qty, :status)
            """), {
                "aid": account_id, "code": o["code"], "dir": o["direction"],
                "price": o["price"], "qty": o["quantity"], "status": status,
            })

    logger.info(f"  执行完成: 买{len(buy_orders)}只 卖{len(sell_orders)}只 持{len(to_hold)}只"
                f" (顺延{len(deferred_items)}只)")
    return {
        "executed": True,
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "deferred_count": len(deferred_items),
        "new_cash": new_cash,
        "total_cost": total_cost,
    }
