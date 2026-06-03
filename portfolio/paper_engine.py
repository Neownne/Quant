"""PaperEngine：ML 选股日频模拟盘引擎（V2 Schema）。

每日流程:
1. 加载当前持仓（V2 paper_positions）和现金（paper_account）
2. 过滤候选池（ST/停牌/涨跌停/次新）
3. 预测打分 → 选股 → 等权分配 → 仓位上限
4. 止损检查 → 组合回撤检查
5. 生成买卖单 → 写入 V2 paper_positions/paper_signals/paper_orders
6. 扣除交易成本（佣金+印花税+滑点）

V2 Schema:
    paper_runs(id, strategy_id, version_id, start_date, status, ...)
    paper_signals(id, run_id, signal_date, stock_code, predicted_score, rank)
    paper_positions(id, run_id, signal_id, stock_code, entry_date, entry_price,
                    exit_date, exit_price, quantity, pnl, pnl_pct)
    signal_factors(id, signal_id, factor_name, value)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from datetime import date

from data.db import get_engine
from config.settings import TradingConfig
from models.regime import REGIME_PARAMS
from portfolio.selector import select_top_n, select_topk_ndrop, filter_stocks
from portfolio.allocator import equal_weight, apply_position_limits
from portfolio.risk import (
    apply_atr_stop_loss, portfolio_stop_reduce, check_drawdown_limit, compute_atr,
    check_index_crash,
)


class PaperEngine:
    """日频 ML 选股模拟盘引擎（V2 Schema）。"""

    def __init__(
        self,
        account_id: int,
        run_id: int,
        predictor,
        factor_names: list[str] | None = None,
        top_n: int | None = None,
        rebalance_mode: str = "ndrop",
        ndrop_n: int | None = None,
        max_single: float | None = None,
        max_industry: float | None = None,
        stop_loss_pct: float | None = None,
        atr_multiplier: float = 1.5,
        atr_period: int = 20,
        portfolio_dd_threshold: float | None = None,
        portfolio_dd_reduce_to: float = 0.50,
        max_dd_limit: float | None = None,
        signal_only: bool = False,
        index_crash_lookback: int | None = None,
        index_crash_threshold: float | None = None,
    ):
        self.account_id = account_id
        self.run_id = run_id
        self.predictor = predictor
        self.factor_names = factor_names or []
        # 默认值从 TradingConfig 读取
        self.top_n = top_n if top_n is not None else TradingConfig.TOP_N
        self.rebalance_mode = rebalance_mode
        self.ndrop_n = ndrop_n if ndrop_n is not None else TradingConfig.NDROP_N
        self.max_single = max_single if max_single is not None else TradingConfig.MAX_SINGLE_WEIGHT
        self.max_industry = max_industry if max_industry is not None else TradingConfig.MAX_INDUSTRY_WEIGHT
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else TradingConfig.STOP_LOSS_PCT
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.portfolio_dd_threshold = portfolio_dd_threshold if portfolio_dd_threshold is not None else TradingConfig.PORTFOLIO_DD_THRESHOLD
        self.portfolio_dd_reduce_to = portfolio_dd_reduce_to
        self.max_dd_limit = max_dd_limit if max_dd_limit is not None else TradingConfig.MAX_DD_LIMIT
        self.signal_only = signal_only
        self.index_crash_lookback = index_crash_lookback if index_crash_lookback is not None else TradingConfig.INDEX_CRASH_LOOKBACK
        self.index_crash_threshold = index_crash_threshold if index_crash_threshold is not None else TradingConfig.INDEX_CRASH_THRESHOLD
        self.peak_value: float | None = None
        # 交易成本
        self.commission = TradingConfig.COMMISSION
        self.stamp_duty = TradingConfig.STAMP_DUTY
        self.slippage = TradingConfig.SLIPPAGE  # 与回测一致(0.1%)

    # ── 公开接口 ──────────────────────────────────────────

    def run_daily(
        self,
        trade_date: pd.Timestamp,
        factor_df: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        industry_map: dict[str, str] | None = None,
        index_ohlcv: pd.DataFrame | None = None,
        regime: str = "sideways",
    ) -> dict | None:
        """执行单日完整交易周期。regime 为当前5状态标签，用于自适应调参。"""
        if factor_df.empty:
            return None

        # ── 按市场状态覆写参数 ──
        reg_params = REGIME_PARAMS.get(regime, REGIME_PARAMS["sideways"])
        self.top_n = reg_params["top_n"]
        self.stop_loss_pct = reg_params["stop_loss_pct"]
        pos_ratio = reg_params.get("position_ratio", 1.0)
        self._pos_ratio = pos_ratio  # 在选股分配时使用

        ohlcv_data = ohlcv_data.copy()
        ohlcv_data["trade_date"] = pd.to_datetime(ohlcv_data["trade_date"])
        if index_ohlcv is not None and not index_ohlcv.empty:
            index_ohlcv = index_ohlcv.copy()
            index_ohlcv["trade_date"] = pd.to_datetime(index_ohlcv["trade_date"])

        # 0. 指数大跌检查
        if index_ohlcv is not None and not index_ohlcv.empty:
            idx_hist = index_ohlcv[index_ohlcv["trade_date"] <= trade_date]
            if not idx_hist.empty:
                idx_close = idx_hist.sort_values("trade_date")["close"].values
                if check_index_crash(idx_close, self.index_crash_lookback, self.index_crash_threshold):
                    return self._handle_crash(trade_date, ohlcv_data)

        # 1. 加载状态
        cash, positions, peak = self._load_state()
        if self.signal_only and cash <= 0:
            cash = TradingConfig.INITIAL_CASH
        if peak is not None:
            self.peak_value = max(self.peak_value or 0, peak)
        if self.peak_value is None:
            self.peak_value = cash

        # 2. 过滤候选池
        day_ohlcv = ohlcv_data[ohlcv_data["trade_date"] == trade_date]
        candidates = self._filter_candidates(factor_df, ohlcv_data, trade_date)
        if candidates.empty:
            return self._skip_day(trade_date, cash, positions, day_ohlcv)

        # 3. 预测 → 选股 → 分配
        current_holdings = set(positions.keys())
        target, signal_scores = self._select_and_allocate(
            factor_df, candidates, industry_map, cash, current_holdings)
        if target.empty:
            return self._skip_day(trade_date, cash, positions, day_ohlcv)

        # 4. 获取今日价格（T+1执行用开盘价）
        today_prices = self._get_price_map(day_ohlcv, use_open=True)

        # 5. 风控检查
        stop_codes, portfolio_reduced, reduced_positions = self._check_risk(
            positions, today_prices, ohlcv_data, cash)

        # 6. 止损平仓
        risk_orders = []
        for code in stop_codes:
            if code in positions:
                p = today_prices.get(code, 0)
                if p > 0:
                    cash += positions[code]["shares"] * p
                risk_orders.append({
                    "code": code, "direction": "SELL",
                    "price": p if p > 0 else positions[code]["cost_basis"],
                    "volume": positions[code]["shares"],
                    "reason": "stop_loss",
                })
                del positions[code]

        # 7. 组合减仓
        if portfolio_reduced and reduced_positions:
            for code, rp in reduced_positions.items():
                if code in positions:
                    diff = positions[code]["shares"] - rp["shares"]
                    if diff > 0:
                        p = today_prices.get(code, 0)
                        if p > 0:
                            cash += diff * p
                        risk_orders.append({
                            "code": code, "direction": "SELL",
                            "price": p if p > 0 else positions[code]["cost_basis"],
                            "volume": diff,
                            "reason": "portfolio_reduce",
                        })
                        positions[code] = rp

        # 8. 生成信号订单
        buy_orders, sell_orders = self._generate_orders(
            target, positions, today_prices, cash)

        # 9. 扣除交易成本
        cost = self._calc_cost(buy_orders, sell_orders, risk_orders, today_prices)
        cash -= cost

        # 10. 执行订单（写 DB）
        all_orders = risk_orders + sell_orders + buy_orders
        daily_signals = []  # for paper_signals
        if not self.signal_only:
            daily_signals = self._execute_orders_v2(
                buy_orders, sell_orders, risk_orders,
                signal_scores, trade_date)

        # 11. 内存更新持仓（先卖后买，现金已在 _generate_orders 中验证）
        for so in sell_orders:
            code = so["code"]
            if code in positions:
                p = today_prices.get(code, 0)
                if p > 0:
                    cash += positions[code]["shares"] * p
                del positions[code]
        for bo in buy_orders:
            code = bo["code"]
            price = bo["price"]
            shares = bo["volume"]
            cash -= shares * price
            positions[code] = {"shares": shares, "cost_basis": price}

        # 12. 计算当日净值
        position_value = sum(
            pos["shares"] * today_prices.get(code, pos["cost_basis"])
            for code, pos in positions.items()
        )
        total_value = cash + position_value
        # 日收益 = 今日/昨日 - 1（从 paper_daily_pnl 取前一日总值）
        prev_total = self._get_prev_total(trade_date)
        daily_return = (total_value / prev_total - 1) if prev_total > 0 else 0
        drawdown = ((self.peak_value - total_value) / self.peak_value) if self.peak_value and self.peak_value > 0 else 0
        self.peak_value = max(self.peak_value or 0, total_value)

        # 13. 记录净值
        if not self.signal_only:
            self._record_daily_pnl(trade_date, cash, position_value, total_value, daily_return, drawdown)

        return {
            "date": trade_date,
            "n_candidates": len(candidates),
            "n_selected": len(target),
            "n_buy_orders": len(buy_orders),
            "n_sell_orders": len(sell_orders) + len(risk_orders),
            "stop_losses": stop_codes,
            "portfolio_reduced": portfolio_reduced,
            "total_value": total_value,
            "cash": cash,
            "cost": cost,
            "orders": all_orders,
            "signals": daily_signals,
        }

    # ── 内部方法 ──────────────────────────────────────────

    def _load_state(self) -> tuple[float, dict[str, dict], float | None]:
        """从 DB 加载现金、V2 持仓和峰值。"""
        engine = get_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT cash, initial_capital FROM paper_account WHERE id = :aid"),
                    {"aid": self.account_id},
                ).fetchone()
                cash = float(row[0]) if row else 0.0

                peak_row = conn.execute(
                    text("SELECT COALESCE(MAX(total_value), 0) FROM paper_daily_pnl WHERE account_id = :aid"),
                    {"aid": self.account_id},
                ).fetchone()
                peak = float(peak_row[0]) if peak_row and peak_row[0] > 0 else None

                # V2: 查询未平仓持仓（exit_date IS NULL），按 stock_code 聚合
                pos_rows = conn.execute(text("""
                    SELECT stock_code, SUM(quantity) AS total_qty,
                           SUM(entry_price * quantity) / NULLIF(SUM(quantity), 0) AS avg_cost
                    FROM paper_positions
                    WHERE run_id = :rid AND exit_date IS NULL
                    GROUP BY stock_code
                """), {"rid": self.run_id}).fetchall()

                positions = {}
                for r in pos_rows:
                    qty = int(r[1]) if r[1] else 0
                    if qty > 0:
                        positions[str(r[0])] = {"shares": qty, "cost_basis": float(r[2])}
        finally:
            engine.dispose()
        return cash, positions, peak

    def _filter_candidates(
        self, factor_df: pd.DataFrame, ohlcv_data: pd.DataFrame, trade_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """串联所有过滤器。"""
        stocks = factor_df[["code"]].drop_duplicates().copy()
        if "name" in factor_df.columns:
            names = factor_df[["code", "name"]].drop_duplicates(subset="code")
            stocks = stocks.merge(names, on="code", how="left")

        result = filter_stocks(stocks, ref_date=trade_date, exclude_st=True, min_list_days=60)

        ohlcv_lookup: dict[str, pd.DataFrame] = {}
        for code in result["code"].unique():
            hist = ohlcv_data[ohlcv_data["code"] == code].sort_values("trade_date")
            if not hist.empty:
                ohlcv_lookup[code] = hist

        from portfolio.selector import filter_suspended, filter_limit_up_down
        result = filter_suspended(result, ohlcv_lookup, trade_date)

        prev_close_map: dict[str, float] = {}
        cur_price_map = {}
        for code in result["code"].unique():
            hist = ohlcv_lookup.get(code)
            if hist is not None:
                prev_rows = hist[hist["trade_date"] < trade_date].tail(1)
                if not prev_rows.empty:
                    prev_close_map[code] = float(prev_rows["close"].iloc[0])
            day_data = ohlcv_data[
                (ohlcv_data["code"] == code) & (ohlcv_data["trade_date"] == trade_date)
            ]
            if not day_data.empty:
                # T+1执行用开盘价
                cur_price_map[code] = float(day_data["open"].iloc[0]) if "open" in day_data.columns else float(day_data["close"].iloc[0])
                if code not in prev_close_map:
                    prev_close_map[code] = cur_price_map[code]

        result["price"] = result["code"].map(cur_price_map)
        result = filter_limit_up_down(result, prev_close_map)
        valid_codes = set(factor_df["code"].unique())
        result = result[result["code"].isin(valid_codes)]
        return result

    def _select_and_allocate(
        self, factor_df: pd.DataFrame, candidates: pd.DataFrame,
        industry_map: dict[str, str] | None, cash: float,
        current_holdings: set[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """预测 → 选 top-N/NDrop → 等权 → 上限约束。返回 (target_df, {code: score})。"""
        valid_codes = set(candidates["code"].unique())
        factor_slice = factor_df[factor_df["code"].isin(valid_codes)]

        try:
            scores = self.predictor.predict(factor_slice)
        except Exception as e:
            logger.warning(f"预测失败: {e}")
            return pd.DataFrame(), {}

        score_map = dict(zip(scores["code"], scores["score"]))

        if self.rebalance_mode == "ndrop" and current_holdings is not None:
            score_series = pd.Series(score_map).sort_values(ascending=False)
            new_holdings, to_buy, to_sell = select_topk_ndrop(
                score_series, current_holdings=current_holdings,
                K=self.top_n, N=self.ndrop_n,
            )
            codes = list(new_holdings)
        else:
            selected = select_top_n(scores, n=self.top_n)
            if selected.empty:
                return pd.DataFrame(), score_map
            codes = selected["code"].tolist()

        if not codes:
            return pd.DataFrame(), score_map

        alloc = equal_weight(codes, cash)
        result = apply_position_limits(alloc, industry_map or {}, self.max_single, self.max_industry)

        price_map = dict(zip(candidates["code"], candidates.get("price", pd.Series())))
        weight_per_code = dict(zip(result["code"], result["weight"]))
        for i, row in result.iterrows():
            code = row["code"]
            price = float(price_map.get(code, 0))
            if price > 0:
                target_value = cash * weight_per_code[code] * getattr(self, '_pos_ratio', 1.0)
                shares = int(target_value / price / 100) * 100
                result.at[i, "target_shares"] = shares
                result.at[i, "price"] = price
            else:
                result.at[i, "target_shares"] = 0
                result.at[i, "price"] = 0

        result = result[result["target_shares"] > 0]
        return result, score_map

    def _check_risk(
        self, positions: dict[str, dict], today_prices: dict[str, float],
        ohlcv_data: pd.DataFrame, cash: float,
    ) -> tuple[list[str], bool, dict[str, dict] | None]:
        """个股止损 + 组合回撤。"""
        stop_codes: list[str] = []

        if positions:
            pos_df = pd.DataFrame([
                {"code": c, "shares": p["shares"], "cost_basis": p["cost_basis"]}
                for c, p in positions.items()
            ])
            atr_values: dict[str, float] = {}
            for code in positions:
                hist = ohlcv_data[ohlcv_data["code"] == code].sort_values("trade_date")
                if len(hist) >= self.atr_period:
                    atr_values[code] = compute_atr(
                        hist["high"].values, hist["low"].values,
                        hist["close"].values, self.atr_period)

            cost_basis = {c: p["cost_basis"] for c, p in positions.items()}
            stop_df = apply_atr_stop_loss(
                pos_df, today_prices, cost_basis, atr_values,
                self.stop_loss_pct, self.atr_multiplier)
            if not stop_df.empty:
                stop_codes = stop_df["code"].tolist()

        position_value_est = sum(
            pos["shares"] * today_prices.get(code, pos["cost_basis"])
            for code, pos in positions.items() if code not in stop_codes
        )
        total_est = cash + position_value_est

        if check_drawdown_limit(total_est, self.peak_value or total_est, self.max_dd_limit):
            logger.warning(f"最大回撤触发 (>{self.max_dd_limit:.0%})，全部清仓")
            stop_codes = list(positions.keys())
            return stop_codes, False, None

        reduced, new_pos = portfolio_stop_reduce(
            positions, total_est, self.peak_value or total_est,
            self.portfolio_dd_threshold, self.portfolio_dd_reduce_to)
        return stop_codes, reduced, new_pos

    def _get_price_map(self, day_ohlcv: pd.DataFrame, use_open: bool = True) -> dict[str, float]:
        """获取当日价格。T+1执行用开盘价(模拟次日开盘买入)，回测用收盘价。"""
        if day_ohlcv.empty:
            return {}
        col = "open" if use_open and "open" in day_ohlcv.columns else "close"
        return dict(zip(day_ohlcv["code"].astype(str), day_ohlcv[col].astype(float)))

    def _generate_orders(
        self, target: pd.DataFrame, positions: dict[str, dict],
        today_prices: dict[str, float], cash: float,
    ) -> tuple[list[dict], list[dict]]:
        """对比目标持仓和当前持仓，生成买卖单。"""
        buy_orders, sell_orders = [], []
        target_map = {}
        for _, row in target.iterrows():
            target_map[row["code"]] = {
                "shares": int(row["target_shares"]), "price": float(row["price"])}

        current_codes = set(positions.keys())
        target_codes = set(target_map.keys())

        for code in current_codes - target_codes:
            p = today_prices.get(code, positions[code]["cost_basis"])
            sell_orders.append({
                "code": code, "direction": "SELL",
                "price": p, "volume": positions[code]["shares"], "reason": "signal"})

        # 计算可用现金（现有现金 + 待卖出回笼）
        sell_proceeds = sum(
            today_prices.get(code, positions[code]["cost_basis"]) * positions[code]["shares"]
            for code in current_codes - target_codes
        )
        available_cash = cash + sell_proceeds

        for code in target_codes:
            t_shares = target_map[code]["shares"]
            t_price = target_map[code]["price"]
            c_shares = positions[code]["shares"] if code in positions else 0

            if t_shares > c_shares:
                diff = int((t_shares - c_shares) / 100) * 100
                cost_needed = diff * t_price
                if diff > 0 and cost_needed <= available_cash:
                    buy_orders.append({
                        "code": code, "direction": "BUY",
                        "price": t_price, "volume": diff, "reason": "signal"})
                    available_cash -= cost_needed
                elif diff > 0:
                    logger.warning(f"现金不足: {code} 需{cost_needed:.0f} 可用{available_cash:.0f}")
            elif t_shares < c_shares:
                diff = int((c_shares - t_shares) / 100) * 100
                if diff > 0:
                    sell_orders.append({
                        "code": code, "direction": "SELL",
                        "price": t_price, "volume": diff, "reason": "signal"})
                    available_cash += diff * t_price

        return buy_orders, sell_orders

    def _calc_cost(
        self, buy_orders: list[dict], sell_orders: list[dict],
        risk_orders: list[dict], prices: dict[str, float],
    ) -> float:
        """计算所有订单的交易成本。"""
        total_cost = 0.0
        all_sells = sell_orders + [o for o in risk_orders if o["direction"] == "SELL"]
        all_buys = buy_orders + [o for o in risk_orders if o["direction"] == "BUY"]

        for o in all_buys:
            amt = o["price"] * o["volume"]
            total_cost += amt * (self.commission + self.slippage)

        for o in all_sells:
            amt = o["price"] * o["volume"]
            total_cost += amt * (self.commission + self.stamp_duty + self.slippage)

        return total_cost

    def _execute_orders_v2(
        self, buy_orders: list[dict], sell_orders: list[dict],
        risk_orders: list[dict], signal_scores: dict[str, float],
        trade_date: pd.Timestamp,
    ) -> list[dict]:
        """V2: 写入 paper_signals + paper_positions（lot 级）+ paper_orders。"""
        engine = get_engine()
        dt = trade_date.date() if hasattr(trade_date, 'date') else trade_date
        all_orders = risk_orders + sell_orders + buy_orders
        daily_signals = []

        try:
            with engine.begin() as conn:
                # 1. 写入信号（paper_signals）
                ranked = sorted(signal_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (code, score) in enumerate(ranked[:self.top_n], 1):
                    r = conn.execute(text("""
                        INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                        VALUES (:rid, :sd, :sc, :ps, :rk)
                        ON CONFLICT (run_id, signal_date, stock_code) DO UPDATE SET predicted_score=:ps2, rank=:rk2
                        RETURNING id
                    """), {
                        "rid": self.run_id, "sd": dt,
                        "sc": code, "ps": float(score), "rk": rank,
                        "ps2": float(score), "rk2": rank,
                    }).fetchone()
                    signal_id = r[0]
                    daily_signals.append({
                        "signal_id": signal_id, "code": code, "score": float(score), "rank": rank})

                # 2. 写订单
                for o in all_orders:
                    conn.execute(text("""
                        INSERT INTO paper_orders (account_id, code, direction, price, volume, amount, status, note)
                        VALUES (:aid, :code, :dir, :price, :vol, :amt, 'filled', :note)
                    """), {
                        "aid": self.account_id, "code": o["code"],
                        "dir": o["direction"], "price": o["price"],
                        "vol": o["volume"], "amt": o["price"] * o["volume"],
                        "note": o.get("reason", ""),
                    })

                # 3. 处理卖单：关闭对应 V2 持仓（按 entry_date 升序平仓）
                for o in sell_orders + [r for r in risk_orders if r["direction"] == "SELL"]:
                    remaining = o["volume"]
                    # FIFO 平仓
                    lots = conn.execute(text("""
                        SELECT id, quantity, entry_price FROM paper_positions
                        WHERE run_id = :rid AND stock_code = :code AND exit_date IS NULL
                        ORDER BY entry_date ASC
                    """), {"rid": self.run_id, "code": o["code"]}).fetchall()

                    for lot in lots:
                        if remaining <= 0:
                            break
                        lot_qty = int(lot[1])
                        close_qty = min(remaining, lot_qty)
                        entry_p = float(lot[2])
                        exit_p = o["price"]
                        # PnL = 价差 - 买入成本 - 卖出成本
                        gross = close_qty * (exit_p - entry_p)
                        buy_cost = close_qty * entry_p * (self.commission + self.slippage)
                        sell_cost = close_qty * exit_p * (self.commission + self.stamp_duty + self.slippage)
                        pnl = gross - buy_cost - sell_cost
                        pnl_pct = pnl / (close_qty * entry_p) if entry_p > 0 else 0
                        if close_qty >= lot_qty:
                            conn.execute(text("""
                                UPDATE paper_positions
                                SET exit_date = :ed, exit_price = :ep, pnl = :pnl, pnl_pct = :pp
                                WHERE id = :lid
                            """), {
                                "ed": dt, "ep": o["price"],
                                "pnl": pnl, "pp": pnl_pct, "lid": lot[0],
                            })
                        else:
                            # 部分平仓：原 lot 减量，建新已平仓 lot
                            conn.execute(text("""
                                UPDATE paper_positions SET quantity = quantity - :cq
                                WHERE id = :lid
                            """), {"cq": close_qty, "lid": lot[0]})
                            conn.execute(text("""
                                INSERT INTO paper_positions
                                    (run_id, stock_code, entry_date, entry_price,
                                     exit_date, exit_price, quantity, pnl, pnl_pct)
                                VALUES (:rid, :sc, (SELECT entry_date FROM paper_positions WHERE id = :lid),
                                        :ep_in, :ed, :ep_out, :qty, :pnl, :pp)
                            """), {
                                "rid": self.run_id, "sc": o["code"],
                                "lid": lot[0], "ep_in": float(lot[2]),
                                "ed": dt, "ep_out": o["price"],
                                "qty": close_qty, "pnl": pnl, "pp": pnl_pct,
                            })
                        remaining -= close_qty

                # 4. 处理买单：新建 V2 持仓 lot
                for o in buy_orders + [r for r in risk_orders if r["direction"] == "BUY"]:
                    conn.execute(text("""
                        INSERT INTO paper_positions
                            (run_id, stock_code, entry_date, entry_price, quantity)
                        VALUES (:rid, :sc, :ed, :ep, :qty)
                    """), {
                        "rid": self.run_id, "sc": o["code"],
                        "ed": dt, "ep": o["price"], "qty": o["volume"],
                    })

                # 5. 更新现金（含交易成本）
                buy_total = sum(o["price"] * o["volume"] for o in buy_orders)
                sell_total = sum(o["price"] * o["volume"] for o in sell_orders + [r for r in risk_orders if r["direction"] == "SELL"])
                cost = (
                    buy_total * (self.commission + self.slippage) +
                    sell_total * (self.commission + self.stamp_duty + self.slippage)
                )
                conn.execute(text(
                    "UPDATE paper_account SET cash = cash + :sell - :buy - :cost WHERE id = :aid"
                ), {"sell": sell_total, "buy": buy_total, "cost": cost, "aid": self.account_id})

        finally:
            engine.dispose()

        return daily_signals

    def _get_prev_total(self, trade_date) -> float:
        """获取前一交易日的总资产。"""
        from data.db import get_engine
        engine = get_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT total_value FROM paper_daily_pnl WHERE account_id=:aid AND trade_date < :d ORDER BY trade_date DESC LIMIT 1"
                ), {"aid": self.account_id, "d": trade_date.date() if hasattr(trade_date, 'date') else trade_date}).fetchone()
                return float(row[0]) if row else TradingConfig.INITIAL_CASH
        finally:
            engine.dispose()

    def _record_daily_pnl(
        self, trade_date, cash: float, position_value: float,
        total_value: float, daily_return: float, drawdown: float,
    ) -> None:
        engine = get_engine()
        dt = trade_date.date() if hasattr(trade_date, 'date') else trade_date
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO paper_daily_pnl
                        (account_id, trade_date, cash, position_value, total_value, daily_return, drawdown)
                    VALUES (:aid, :d, :c, :pv, :tv, :dr, :dd)
                    ON CONFLICT (account_id, trade_date) DO UPDATE SET
                        cash = :c2, position_value = :pv2, total_value = :tv2,
                        daily_return = :dr2, drawdown = :dd2
                """), {
                    "aid": self.account_id, "d": dt,
                    "c": cash, "pv": position_value, "tv": total_value,
                    "dr": daily_return, "dd": drawdown,
                    "c2": cash, "pv2": position_value, "tv2": total_value,
                    "dr2": daily_return, "dd2": drawdown,
                })
        finally:
            engine.dispose()

    def _skip_day(
        self, trade_date: pd.Timestamp, cash: float,
        positions: dict[str, dict], day_ohlcv: pd.DataFrame,
    ) -> dict:
        today_prices = self._get_price_map(day_ohlcv, use_open=False)
        position_value = sum(
            pos["shares"] * today_prices.get(code, pos["cost_basis"])
            for code, pos in positions.items()
        )
        total_value = cash + position_value
        self.peak_value = max(self.peak_value or total_value, total_value)

        if not self.signal_only:
            prev_total = self.peak_value or total_value
            daily_ret = (total_value / prev_total - 1) if prev_total > 0 else 0
            drawdown = ((self.peak_value - total_value) / self.peak_value) if self.peak_value and self.peak_value > 0 else 0
            self._record_daily_pnl(trade_date, cash, position_value, total_value, daily_ret, drawdown)

        return {
            "date": trade_date, "n_candidates": 0, "n_selected": 0,
            "n_buy_orders": 0, "n_sell_orders": 0,
            "stop_losses": [], "portfolio_reduced": False,
            "total_value": total_value, "cash": cash, "cost": 0, "orders": [], "signals": [],
        }

    def _handle_crash(self, trade_date: pd.Timestamp, ohlcv_data: pd.DataFrame) -> dict:
        """指数大跌 → 全部清仓。"""
        logger.warning(f"{trade_date.date()} 指数大跌触发，空仓")
        cash, positions, _ = self._load_state()
        day_ohlcv = ohlcv_data[ohlcv_data["trade_date"] == trade_date]
        today_prices = self._get_price_map(day_ohlcv, use_open=False)

        crash_orders = []
        for code in list(positions.keys()):
            p = today_prices.get(code, 0)
            if p > 0:
                cash += positions[code]["shares"] * p
            crash_orders.append({
                "code": code, "direction": "SELL",
                "price": p if p > 0 else positions[code].get("cost_basis", 0),
                "volume": positions[code]["shares"], "reason": "index_crash",
            })
            del positions[code]

        if not self.signal_only:
            self._execute_orders_v2([], [], crash_orders, {}, trade_date)
            self._record_daily_pnl(trade_date, cash, 0, cash, 0, 0)

        return {
            "date": trade_date, "n_candidates": 0, "n_selected": 0,
            "n_buy_orders": 0, "n_sell_orders": len(crash_orders),
            "stop_losses": [], "portfolio_reduced": False,
            "crash_warning": True, "total_value": cash, "cash": cash,
            "cost": 0, "orders": crash_orders, "signals": [],
        }


def _save_signal_factors(signal_id: int, factor_values: dict):
    """Write per-signal factor values for attribution analysis."""
    if not factor_values:
        return
    engine = get_engine()
    try:
        with engine.begin() as conn:
            for factor_name, value in factor_values.items():
                conn.execute(text("""
                    INSERT INTO signal_factors (signal_id, factor_name, value)
                    VALUES (:sid, :name, :val)
                    ON CONFLICT (signal_id, factor_name) DO NOTHING
                """), {"sid": signal_id, "name": factor_name, "val": float(value)})
    except Exception:
        pass
