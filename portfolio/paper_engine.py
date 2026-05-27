"""PaperEngine：ML 选股日频模拟盘引擎。

每日流程:
1. 加载当前持仓和现金
2. 过滤候选池 (ST/停牌/涨跌停/次新)
3. 预测打分 → 选股 → 等权分配 → 仓位上限
4. 止损检查 → 组合回撤检查
5. 生成买卖单 → 写入数据库
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from portfolio.selector import select_top_n, select_topk_ndrop, filter_stocks
from portfolio.allocator import equal_weight, apply_position_limits
from portfolio.risk import (
    apply_atr_stop_loss, portfolio_stop_reduce, check_drawdown_limit, compute_atr,
    check_index_crash,
)


class PaperEngine:
    """日频 ML 选股模拟盘引擎。"""

    def __init__(
        self,
        account_id: int,
        predictor,
        factor_names: list[str],
        top_n: int = 20,
        rebalance_mode: str = "full",
        ndrop_n: int = 2,
        max_single: float = 0.10,
        max_industry: float = 0.30,
        stop_loss_pct: float = 0.08,
        atr_multiplier: float = 1.5,
        atr_period: int = 20,
        portfolio_dd_threshold: float = 0.20,
        portfolio_dd_reduce_to: float = 0.50,
        max_dd_limit: float = 0.25,
        signal_only: bool = False,
        index_crash_lookback: int = 15,
        index_crash_threshold: float = -0.12,
    ):
        self.account_id = account_id
        self.predictor = predictor
        self.factor_names = factor_names
        self.top_n = top_n
        self.rebalance_mode = rebalance_mode
        self.ndrop_n = ndrop_n
        self.max_single = max_single
        self.max_industry = max_industry
        self.stop_loss_pct = stop_loss_pct
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.portfolio_dd_threshold = portfolio_dd_threshold
        self.portfolio_dd_reduce_to = portfolio_dd_reduce_to
        self.max_dd_limit = max_dd_limit
        self.signal_only = signal_only
        self.index_crash_lookback = index_crash_lookback
        self.index_crash_threshold = index_crash_threshold
        self.peak_value: float | None = None

    # ── 公开接口 ──────────────────────────────────────────

    def run_daily(
        self,
        trade_date: pd.Timestamp,
        factor_df: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        industry_map: dict[str, str] | None = None,
        index_ohlcv: pd.DataFrame | None = None,
    ) -> dict | None:
        """执行单日完整交易周期。

        参数
        ----
        trade_date : 交易日
        factor_df : 当日因子截面 (含 code, trade_date, 所有因子列, 可选的 name/close 列)
        ohlcv_data : OHLCV 历史数据 (含 code, trade_date, open, high, low, close, volume)
        industry_map : {code: industry_label}
        index_ohlcv : 指数 OHLCV 数据 (含 trade_date, close)，用于大跌过滤器

        返回
        ----
        dict 或 None（交易日无数据时）
        """
        if factor_df.empty:
            return None

        # 防御性类型转换：DB 加载的 trade_date 可能是 date 对象
        ohlcv_data = ohlcv_data.copy()
        ohlcv_data["trade_date"] = pd.to_datetime(ohlcv_data["trade_date"])
        if index_ohlcv is not None and not index_ohlcv.empty:
            index_ohlcv = index_ohlcv.copy()
            index_ohlcv["trade_date"] = pd.to_datetime(index_ohlcv["trade_date"])

        # 0. 指数大跌检查
        crash_warning = False
        if index_ohlcv is not None and not index_ohlcv.empty:
            idx_hist = index_ohlcv[index_ohlcv["trade_date"] <= trade_date]
            if not idx_hist.empty:
                idx_close = idx_hist.sort_values("trade_date")["close"].values
                if check_index_crash(idx_close, self.index_crash_lookback, self.index_crash_threshold):
                    logger.warning(f"{trade_date.date()} 指数大跌触发，空仓")
                    crash_warning = True
                    cash, positions, peak = self._load_state()
                    day_ohlcv = ohlcv_data[ohlcv_data["trade_date"] == trade_date]
                    today_prices = self._get_price_map(day_ohlcv)
                    # 清仓
                    crash_orders = []
                    for code in list(positions.keys()):
                        p = today_prices.get(code, 0)
                        if p > 0:
                            cash += positions[code]["shares"] * p
                        crash_orders.append({
                            "code": code, "direction": "SELL",
                            "price": p if p > 0 else positions[code].get("cost_basis", 0),
                            "volume": positions[code]["shares"],
                            "reason": "index_crash",
                        })
                        del positions[code]
                    if not self.signal_only:
                        self._record_daily_pnl(trade_date, cash, 0, cash, 0, 0)
                    return {
                        "date": trade_date,
                        "n_candidates": 0, "n_selected": 0,
                        "n_buy_orders": 0, "n_sell_orders": len(crash_orders),
                        "stop_losses": [],
                        "portfolio_reduced": False,
                        "crash_warning": True,
                        "total_value": cash,
                        "orders": crash_orders,
                    }

        # 1. 加载状态
        cash, positions, peak = self._load_state()
        if self.signal_only and cash <= 0:
            cash = 1_000_000.0
        if peak is not None:
            self.peak_value = max(self.peak_value or 0, peak)
        if self.peak_value is None:
            self.peak_value = cash

        # 2. 过滤候选池
        day_ohlcv = ohlcv_data[ohlcv_data["trade_date"] == trade_date]
        candidates = self._filter_candidates(factor_df, ohlcv_data, trade_date)
        if candidates.empty:
            logger.debug(f"{trade_date.date()}: 候选池为空")
            return self._skip_day(trade_date, cash, positions, day_ohlcv)

        # 3. 预测 → 选股 → 分配
        current_holdings = set(positions.keys())
        target = self._select_and_allocate(factor_df, candidates, industry_map, cash, current_holdings)
        if target.empty:
            return self._skip_day(trade_date, cash, positions, day_ohlcv)

        # 4. 获取今日价格
        today_prices = self._get_price_map(day_ohlcv)

        # 5. 风控检查（当前持仓）
        stop_codes, portfolio_reduced, reduced_positions = self._check_risk(
            positions, today_prices, ohlcv_data, cash,
        )

        # 收集风控触发的订单
        risk_orders = []

        # 6. 执行止损平仓
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

        # 8. 生成订单
        buy_orders, sell_orders = self._generate_orders(
            target, positions, today_prices, cash, reason="signal",
        )

        # 9. 执行订单（含风控订单）
        all_orders = risk_orders + sell_orders + buy_orders
        if not self.signal_only:
            self._execute_orders(buy_orders, sell_orders, trade_date)

        # 10. 更新持仓（模拟执行）
        for so in sell_orders:
            code = so["code"]
            if code in positions:
                p = today_prices.get(code, 0)
                cash += positions[code]["shares"] * p
                del positions[code]
        for bo in buy_orders:
            code = bo["code"]
            price = bo["price"]
            shares = bo["volume"]
            cost = shares * price
            if cost <= cash:
                cash -= cost
                positions[code] = {"shares": shares, "cost_basis": price}

        # 11. 计算当日市值
        position_value = 0.0
        for code, pos in positions.items():
            p = today_prices.get(code, pos["cost_basis"])
            position_value += pos["shares"] * (p if p > 0 else pos["cost_basis"])
        total_value = cash + position_value
        prev_total = self.peak_value  # 近似：用 peak 作为上个参考点
        daily_return = (total_value / max(prev_total, 1) - 1) if prev_total > 0 else 0
        drawdown = (max(self.peak_value, total_value) - total_value) / max(self.peak_value, total_value) if max(self.peak_value, total_value) > 0 else 0
        self.peak_value = max(self.peak_value, total_value)

        # 12. 记录净值
        if not self.signal_only:
            self._record_daily_pnl(trade_date, cash, position_value, total_value, daily_return, drawdown)

        return {
            "date": trade_date,
            "n_candidates": len(candidates),
            "n_selected": len(target),
            "n_buy_orders": len(buy_orders),
            "n_sell_orders": len(sell_orders),
            "stop_losses": stop_codes,
            "portfolio_reduced": portfolio_reduced,
            "total_value": total_value,
            "orders": all_orders,
        }

    # ── 内部方法 ──────────────────────────────────────────

    def _load_state(self) -> tuple[float, dict[str, dict], float | None]:
        """从 DB 加载现金、持仓和峰值。"""
        engine = get_engine()
        try:
            with engine.connect() as conn:
                # 现金
                row = conn.execute(
                    text("SELECT cash, initial_capital FROM paper_account WHERE id = :aid"),
                    {"aid": self.account_id},
                ).fetchone()
                cash = float(row[0]) if row else 0.0

                # 峰值：从 paper_daily_pnl 取历史最高 total_value
                peak_row = conn.execute(
                    text("SELECT COALESCE(MAX(total_value), 0) FROM paper_daily_pnl WHERE account_id = :aid"),
                    {"aid": self.account_id},
                ).fetchone()
                peak = float(peak_row[0]) if peak_row and peak_row[0] > 0 else None

                # 持仓
                pos_rows = conn.execute(
                    text("SELECT code, volume, avg_cost FROM paper_positions WHERE account_id = :aid"),
                    {"aid": self.account_id},
                ).fetchall()
                positions = {}
                for r in pos_rows:
                    if int(r[1]) > 0:
                        positions[str(r[0])] = {"shares": int(r[1]), "cost_basis": float(r[2])}
        finally:
            engine.dispose()
        return cash, positions, peak

    def _filter_candidates(
        self, factor_df: pd.DataFrame, ohlcv_data: pd.DataFrame, trade_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """串联所有过滤器。"""
        stocks = factor_df[["code"]].drop_duplicates().copy()
        # 合并名称（如有）
        if "name" in factor_df.columns:
            names = factor_df[["code", "name"]].drop_duplicates(subset="code")
            stocks = stocks.merge(names, on="code", how="left")

        # ST + 次新过滤
        result = filter_stocks(stocks, ref_date=trade_date, exclude_st=True, min_list_days=60)

        # 停牌过滤: 构建 ohlcv_lookup
        ohlcv_lookup: dict[str, pd.DataFrame] = {}
        for code in result["code"].unique():
            hist = ohlcv_data[ohlcv_data["code"] == code].sort_values("trade_date")
            if not hist.empty:
                ohlcv_lookup[code] = hist

        from portfolio.selector import filter_suspended
        result = filter_suspended(result, ohlcv_lookup, trade_date)

        # 涨跌停过滤: 构建 prev_close_map
        prev_date = trade_date - pd.Timedelta(days=1)
        prev_close_map: dict[str, float] = {}
        for code in result["code"].unique():
            hist = ohlcv_lookup.get(code)
            if hist is not None:
                prev_rows = hist[hist["trade_date"] <= trade_date].tail(1)
                if not prev_rows.empty:
                    prev_close_map[code] = float(prev_rows["close"].iloc[0])
            # 如果查不到前日价，从 ohlcv_data 当天取 close 作为近似
            if code not in prev_close_map:
                day_data = ohlcv_data[
                    (ohlcv_data["code"] == code) & (ohlcv_data["trade_date"] == trade_date)
                ]
                if not day_data.empty:
                    prev_close_map[code] = float(day_data["close"].iloc[0])

        # 把当天 close 作为价格装入 stocks
        close_map = {}
        for code in result["code"].unique():
            day_data = ohlcv_data[
                (ohlcv_data["code"] == code) & (ohlcv_data["trade_date"] == trade_date)
            ]
            if not day_data.empty:
                close_map[code] = float(day_data["close"].iloc[0])
        result["close"] = result["code"].map(close_map)

        from portfolio.selector import filter_limit_up_down
        result = filter_limit_up_down(result, prev_close_map)

        # 只保留 factor_df 中有的股票
        valid_codes = set(factor_df["code"].unique())
        result = result[result["code"].isin(valid_codes)]

        return result

    def _select_and_allocate(
        self, factor_df: pd.DataFrame, candidates: pd.DataFrame,
        industry_map: dict[str, str] | None, cash: float,
        current_holdings: set[str] | None = None,
    ) -> pd.DataFrame:
        """预测 → 选 top-N/NDrop → 等权 → 上限约束。"""
        valid_codes = set(candidates["code"].unique())
        factor_slice = factor_df[factor_df["code"].isin(valid_codes)]

        try:
            scores = self.predictor.predict(factor_slice)
        except Exception as e:
            logger.warning(f"预测失败: {e}")
            return pd.DataFrame()

        # NDrop 增量调仓 vs 全量换仓
        if self.rebalance_mode == "ndrop" and current_holdings is not None:
            score_series = pd.Series(
                scores.set_index("code")["score"].to_dict()
            ).sort_values(ascending=False)
            new_holdings, to_buy, to_sell = select_topk_ndrop(
                score_series, current_holdings=current_holdings,
                K=self.top_n, N=self.ndrop_n,
            )
            codes = list(new_holdings)
        else:
            selected = select_top_n(scores, n=self.top_n)
            if selected.empty:
                return pd.DataFrame()
            codes = selected["code"].tolist()

        if not codes:
            return pd.DataFrame()

        alloc = equal_weight(codes, cash)
        result = apply_position_limits(alloc, industry_map or {}, self.max_single, self.max_industry)

        # 计算目标股数（整手）
        weight_per_code = dict(zip(result["code"], result["weight"]))
        for i, row in result.iterrows():
            code = row["code"]
            price_row = candidates[candidates["code"] == code]
            price = float(price_row["close"].iloc[0]) if not price_row.empty and "close" in price_row.columns else 0
            if price > 0:
                target_value = cash * weight_per_code[code]
                shares = int(target_value / price / 100) * 100
                result.at[i, "target_shares"] = shares
                result.at[i, "price"] = price
            else:
                result.at[i, "target_shares"] = 0
                result.at[i, "price"] = 0

        result = result[result["target_shares"] > 0]
        return result

    def _check_risk(
        self,
        positions: dict[str, dict],
        today_prices: dict[str, float],
        ohlcv_data: pd.DataFrame,
        cash: float,
    ) -> tuple[list[str], bool, dict[str, dict] | None]:
        """执行风控检查。

        返回 (stop_loss_codes, portfolio_reduced, reduced_positions)
        """
        stop_codes: list[str] = []

        # 个股止损
        if positions:
            pos_df = pd.DataFrame([
                {"code": c, "shares": p["shares"], "cost_basis": p["cost_basis"]}
                for c, p in positions.items()
            ])
            # ATR
            atr_values: dict[str, float] = {}
            for code in positions:
                hist = ohlcv_data[ohlcv_data["code"] == code].sort_values("trade_date")
                if len(hist) >= self.atr_period:
                    atr = compute_atr(
                        hist["high"].values, hist["low"].values,
                        hist["close"].values, self.atr_period,
                    )
                    atr_values[code] = atr

            cost_basis = {c: p["cost_basis"] for c, p in positions.items()}
            stop_df = apply_atr_stop_loss(
                pos_df, today_prices, cost_basis, atr_values,
                self.stop_loss_pct, self.atr_multiplier,
            )
            if not stop_df.empty:
                stop_codes = stop_df["code"].tolist()

        # 组合回撤（在止损执行后计算）
        position_value_est = 0.0
        for code, pos in positions.items():
            if code not in stop_codes:
                p = today_prices.get(code, pos["cost_basis"])
                position_value_est += pos["shares"] * (p if p > 0 else pos["cost_basis"])
        total_est = cash + position_value_est

        # 最大回撤 → 清仓
        if check_drawdown_limit(total_est, self.peak_value or total_est, self.max_dd_limit):
            logger.warning(f"最大回撤触发 (>{self.max_dd_limit:.0%})，全部清仓")
            stop_codes = list(positions.keys())
            return stop_codes, False, None

        # 组合回撤 → 减仓
        reduced, new_pos = portfolio_stop_reduce(
            positions, total_est, self.peak_value or total_est,
            self.portfolio_dd_threshold, self.portfolio_dd_reduce_to,
        )
        return stop_codes, reduced, new_pos

    def _get_price_map(self, day_ohlcv: pd.DataFrame) -> dict[str, float]:
        """当日收盘价映射。"""
        if day_ohlcv.empty:
            return {}
        return dict(zip(
            day_ohlcv["code"].astype(str),
            day_ohlcv["close"].astype(float),
        ))

    def _generate_orders(
        self,
        target: pd.DataFrame,
        positions: dict[str, dict],
        today_prices: dict[str, float],
        cash: float,
        reason: str = "signal",
    ) -> tuple[list[dict], list[dict]]:
        """对比目标持仓和当前持仓，生成买卖单。"""
        buy_orders: list[dict] = []
        sell_orders: list[dict] = []

        target_map = {}
        for _, row in target.iterrows():
            target_map[row["code"]] = {
                "shares": int(row["target_shares"]),
                "price": float(row["price"]),
            }

        current_codes = set(positions.keys())
        target_codes = set(target_map.keys())

        # 卖出: 当前有但目标没有的
        for code in current_codes - target_codes:
            p = today_prices.get(code, positions[code]["cost_basis"])
            sell_orders.append({
                "code": code, "direction": "SELL",
                "price": p, "volume": positions[code]["shares"],
                "reason": reason,
            })

        # 买入/调整
        for code in target_codes:
            t_shares = target_map[code]["shares"]
            t_price = target_map[code]["price"]
            c_shares = positions[code]["shares"] if code in positions else 0

            if t_shares > c_shares:
                diff = t_shares - c_shares
                # 整手
                diff = int(diff / 100) * 100
                if diff > 0 and diff * t_price <= cash + sum(
                    (today_prices.get(c2, 0) * positions[c2]["shares"])
                    for c2 in current_codes - target_codes
                ):
                    buy_orders.append({
                        "code": code, "direction": "BUY",
                        "price": t_price, "volume": diff,
                        "reason": reason,
                    })
            elif t_shares < c_shares:
                diff = c_shares - t_shares
                diff = int(diff / 100) * 100
                if diff > 0:
                    sell_orders.append({
                        "code": code, "direction": "SELL",
                        "price": t_price, "volume": diff,
                        "reason": reason,
                    })

        return buy_orders, sell_orders

    def _execute_orders(
        self, buy_orders: list[dict], sell_orders: list[dict], trade_date: pd.Timestamp,
    ) -> None:
        """写入 paper_orders 并更新 paper_positions。"""
        engine = get_engine()
        try:
            with engine.begin() as conn:
                # 写订单
                for o in sell_orders + buy_orders:
                    conn.execute(text("""
                        INSERT INTO paper_orders (account_id, code, direction, price, volume, amount, status, note)
                        VALUES (:aid, :code, :dir, :price, :vol, :amt, 'filled', :note)
                    """), {
                        "aid": self.account_id,
                        "code": o["code"],
                        "dir": o["direction"],
                        "price": o["price"],
                        "vol": o["volume"],
                        "amt": o["price"] * o["volume"],
                        "note": o.get("reason", ""),
                    })

                # 更新持仓
                for o in sell_orders:
                    existing = conn.execute(text(
                        "SELECT volume FROM paper_positions WHERE account_id = :aid AND code = :code"
                    ), {"aid": self.account_id, "code": o["code"]}).fetchone()
                    if existing and int(existing[0]) > o["volume"]:
                        conn.execute(text(
                            "UPDATE paper_positions SET volume = volume - :v WHERE account_id = :aid AND code = :code"
                        ), {"v": o["volume"], "aid": self.account_id, "code": o["code"]})
                    else:
                        conn.execute(text(
                            "DELETE FROM paper_positions WHERE account_id = :aid AND code = :code"
                        ), {"aid": self.account_id, "code": o["code"]})

                for o in buy_orders:
                    conn.execute(text("""
                        INSERT INTO paper_positions (account_id, code, volume, avg_cost)
                        VALUES (:aid, :code, :vol, :price)
                        ON CONFLICT (account_id, code)
                        DO UPDATE SET volume = paper_positions.volume + :vol2,
                                      avg_cost = (paper_positions.avg_cost * paper_positions.volume + :price * :vol3)
                                               / (paper_positions.volume + :vol4)
                    """), {
                        "aid": self.account_id, "code": o["code"],
                        "vol": o["volume"], "price": o["price"],
                        "vol2": o["volume"], "vol3": o["volume"], "vol4": o["volume"],
                    })

                # 更新现金
                buy_total = sum(o["price"] * o["volume"] for o in buy_orders)
                sell_total = sum(o["price"] * o["volume"] for o in sell_orders)
                conn.execute(text(
                    "UPDATE paper_account SET cash = cash + :sell - :buy WHERE id = :aid"
                ), {"sell": sell_total, "buy": buy_total, "aid": self.account_id})

                # TODO: 在此处写入 paper_signals + 调用 _save_signal_factors()
                # 当 `run_daily` 中的 factor_df（因子值）和 predicted_score 可传入时，
                # 为每个买入信号插入 paper_signals 记录，然后调用 _save_signal_factors(signal_id, factor_values)。
                # factor_values 应从 factor_df 中提取当前 stock_code 对应的因子列值。
        finally:
            engine.dispose()

    def _record_daily_pnl(
        self, trade_date: pd.Timestamp, cash: float, position_value: float,
        total_value: float, daily_return: float, drawdown: float,
    ) -> None:
        """写入每日净值记录。"""
        engine = get_engine()
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
                    "aid": self.account_id, "d": trade_date.date(),
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
        """无交易的跳过日。"""
        today_prices = self._get_price_map(day_ohlcv)
        position_value = 0.0
        for code, pos in positions.items():
            p = today_prices.get(code, pos["cost_basis"])
            position_value += pos["shares"] * (p if p > 0 else pos["cost_basis"])
        total_value = cash + position_value
        self.peak_value = max(self.peak_value or total_value, total_value)

        if not self.signal_only:
            drawdown = (
                (self.peak_value - total_value) / self.peak_value
            ) if self.peak_value > 0 else 0
            self._record_daily_pnl(trade_date, cash, position_value, total_value, 0.0, drawdown)

        return {
            "date": trade_date,
            "n_candidates": 0,
            "n_selected": 0,
            "n_buy_orders": 0,
            "n_sell_orders": 0,
            "stop_losses": [],
            "portfolio_reduced": False,
            "total_value": total_value,
            "orders": [],
        }


class StaticPaperEngine:
    """非ML策略模拟盘引擎：用 backtrader 回测回放生成买卖信号。

    将 backtrader 的逐笔信号写入 paper_orders/paper_positions，
    让静态策略（SMA/MACD/RSI）也能在模拟盘中展示持仓和交易。"""

    def __init__(self, account_id: int, strategy_class, strategy_params: dict,
                 codes: list[str],
                 initial_cash: float = 1_000_000,
                 commission: float = 0.00009,
                 stamp_duty: float = 0.0005,
                 slippage: float = 0.01,
                 use_market_filter: bool = True):
        self.account_id = account_id
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.codes = codes
        self.initial_cash = initial_cash
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.use_market_filter = use_market_filter

    def run_replay(self, start_date: str, end_date: str) -> dict:
        """对每只股票运行 backtrader 回测，将交易信号写入 paper 表。

        返回汇总 dict。"""
        # [架构重构] app.utils 已移除，此处需要重构
        from app.utils.backtest_runner import run_backtest, load_index_data  # noqa: F401 (removed in v2.0)
        from app.utils.data_loader import load_ohlcv  # noqa: F401 (removed in v2.0)

        engine = get_engine()

        # 加载指数数据
        index_df = None
        if self.use_market_filter:
            try:
                index_df = load_index_data(start_date, end_date)
            except Exception:
                pass

        # 清空现有记录，从头回放
        with engine.begin() as conn:
            conn.execute(text(
                "DELETE FROM paper_orders WHERE account_id = :aid"), {"aid": self.account_id})
            conn.execute(text(
                "DELETE FROM paper_positions WHERE account_id = :aid"), {"aid": self.account_id})
            conn.execute(text(
                "DELETE FROM paper_daily_pnl WHERE account_id = :aid"), {"aid": self.account_id})
            conn.execute(text(
                "UPDATE paper_account SET cash = :cash WHERE id = :aid"),
                {"cash": self.initial_cash, "aid": self.account_id})

        total_trades = 0
        all_signals: list[dict] = []

        for code in self.codes:
            df = load_ohlcv(code, start_date, end_date)
            if df.empty or len(df) < 50:
                continue

            try:
                result = run_backtest(
                    strategy_class=self.strategy_class,
                    df=df,
                    strategy_params=self.strategy_params,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                    stamp_duty=self.stamp_duty,
                    slippage=self.slippage,
                    index_df=index_df,
                    batch_mode=True,
                )
            except Exception:
                continue

            signals = result.get("signals", [])
            for s in signals:
                s["code"] = code
            all_signals.extend(signals)
            total_trades += len(result.get("trades", []))

        # 按日期排序信号
        all_signals.sort(key=lambda x: x.get("date", pd.Timestamp.min))

        # 逐信号执行，更新持仓和现金
        positions: dict[str, dict] = {}  # code -> {shares, cost_basis}
        cash = self.initial_cash

        for sig in all_signals:
            code = sig["code"]
            price = sig["price"]
            size = sig["size"]
            trade_date = sig["date"]
            if isinstance(trade_date, pd.Timestamp):
                trade_date = trade_date.to_pydatetime()

            if sig["direction"] == "BUY":
                cost = price * size
                if cost <= cash:
                    cash -= cost
                    if code in positions:
                        old_shares = positions[code]["shares"]
                        old_cost = positions[code]["cost_basis"]
                        new_shares = old_shares + size
                        new_cost = (old_cost * old_shares + price * size) / new_shares
                        positions[code] = {"shares": new_shares, "cost_basis": new_cost}
                    else:
                        positions[code] = {"shares": size, "cost_basis": price}
            else:
                if code in positions:
                    cash += price * min(size, positions[code]["shares"])
                    remaining = positions[code]["shares"] - size
                    if remaining <= 0:
                        del positions[code]
                    else:
                        positions[code]["shares"] = remaining

        # 写入信号到 paper_orders
        with engine.begin() as conn:
            for sig in all_signals:
                conn.execute(text("""
                    INSERT INTO paper_orders (account_id, code, direction, price, volume, amount, status, note)
                    VALUES (:aid, :code, :dir, :price, :vol, :amt, 'filled', 'static_replay')
                """), {
                    "aid": self.account_id,
                    "code": sig["code"],
                    "dir": sig["direction"],
                    "price": sig["price"],
                    "vol": sig["size"],
                    "amt": sig["price"] * sig["size"],
                })

            # 写入最终持仓
            for code, pos in positions.items():
                conn.execute(text("""
                    INSERT INTO paper_positions (account_id, code, volume, avg_cost)
                    VALUES (:aid, :code, :vol, :cost)
                    ON CONFLICT (account_id, code)
                    DO UPDATE SET volume = :vol2, avg_cost = :cost2
                """), {
                    "aid": self.account_id, "code": code,
                    "vol": pos["shares"], "cost": pos["cost_basis"],
                    "vol2": pos["shares"], "cost2": pos["cost_basis"],
                })

            # 更新现金
            conn.execute(text(
                "UPDATE paper_account SET cash = :cash WHERE id = :aid"
            ), {"cash": cash, "aid": self.account_id})

        engine.dispose()
        return {
            "n_stocks": len(self.codes),
            "n_signals": len(all_signals),
            "n_trades": total_trades,
            "n_positions": len(positions),
            "final_cash": cash,
        }

    def run_daily(self, trade_date, factor_df, ohlcv_data,
                  industry_map=None, index_ohlcv=None) -> dict | None:
        """静态策略每日运行：调用 run_replay 回放全部历史。"""
        return None


# ══════════════════════════════════════════════════════════
# PaperEngine batch-mode skip (restored)
# ══════════════════════════════════════════════════════════


# ── 信号因子写入 ──────────────────────────────────────────


def _save_signal_factors(signal_id: int, factor_values: dict):
    """Write per-signal factor values for attribution analysis.

    Parameters
    ----------
    signal_id : paper_signals 表的主键 ID
    factor_values : {factor_name: value} 字典，来自 run_daily 中的 factor_df

    调用时机：在 paper_signals 行写入后立即调用。
    在 PaperEngine._execute_orders() 中集成，
    传入从 factor_df 提取的对应股票因子值。
    """
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
