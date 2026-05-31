"""SmallCapEngine: 小市值事件驱动引擎（低开反转 + 线性止盈）。

每日流程:
1. 构建候选池（最小市值N只, 排除688/300/4/8/ST/次新<375天）
2. 扫描入场信号（低开反转7条件）
3. 检查持仓出场（止损/移动止盈/破开盘价）
4. 生成买卖单 + 写DB
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from datetime import date
from data.db import get_engine
from config.settings import TradingConfig


class SmallCapEngine:
    def __init__(self, account_id: int = 0, run_id: int = 0,
                 stock_num: int = 10, universe_size: int = 300):
        self.account_id = account_id
        self.run_id = run_id
        self.stock_num = stock_num
        self.universe_size = universe_size
        self.commission = TradingConfig.COMMISSION
        self.stamp_duty = TradingConfig.STAMP_DUTY
        self.slippage = 0.0015  # 小市值模拟盘滑点更保守

    @staticmethod
    def seasonal_skip(dt) -> bool:
        """季节空仓: 1月/4月/12.20-31/3.20-31"""
        m, d = dt.month, dt.day
        return m in (1, 4) or (m == 12 and d >= 20) or (m == 3 and d >= 20)

    @staticmethod
    def build_universe(engine, ref_date: str, universe_size: int = 300) -> list[str]:
        """小市值候选池: 排除科创/创业/北交/ST/次新,按code排序"""
        sql = f"""
            SELECT code FROM stock_basic WHERE is_st = FALSE
            AND list_date <= '{ref_date}'::date - INTERVAL '375 days'
            AND code NOT LIKE '688%' AND code NOT LIKE '300%'
            AND code NOT LIKE '4%' AND code NOT LIKE '8%'
            ORDER BY code LIMIT {universe_size}
        """
        return pd.read_sql(text(sql), engine)["code"].tolist()

    def run_daily(self, trade_date, prev_date, today_data, prev_data,
                  positions: dict, cash: float, peak_value: float,
                  lookback_data: dict, industry_map: dict) -> dict:
        """执行单日完整周期。today_data/prev_data 是 DataFrame, indexed by code"""
        dt = pd.Timestamp(trade_date)
        if today_data is None or (hasattr(today_data, 'empty') and today_data.empty):
            return self._empty_result(positions, cash, peak_value)

        # 季节空仓 → 清仓
        if self.seasonal_skip(dt):
            return self._seasonal_clear(today_data, positions, cash, peak_value)

        today_open_map = {}
        if hasattr(today_data, 'index'):
            for code in today_data.index:
                today_open_map[code] = float(today_data.loc[code, "open"])
        else:
            # DataFrame with 'code' column
            for _, row in today_data.iterrows():
                today_open_map[row["code"]] = float(row["open"])

        # 1. 出场检查
        sells = self._check_exits(today_data, positions, today_open_map)
        sell_orders = []
        for code, reason, price in sells:
            pos = positions.pop(code, None)
            if pos:
                revenue = pos["shares"] * price * (1 - self.stamp_duty - self.commission - self.slippage)
                cash += revenue
                sell_orders.append({"code": code, "direction": "SELL",
                                    "price": price, "volume": pos["shares"], "reason": reason})

        # 2. 入场扫描
        buys, cash = self._scan_entries(today_data, prev_data, positions,
                                         lookback_data, industry_map, cash)
        buy_orders = []
        for b in buys:
            positions[b["code"]] = {"shares": b["shares"], "cost": b["price"],
                                     "today_high": b["price"],
                                     "today_open": today_open_map.get(b["code"], b["price"])}
            buy_orders.append({"code": b["code"], "direction": "BUY",
                               "price": b["price"], "volume": b["shares"], "reason": "signal"})

        # 3. 净值
        position_value = self._calc_position_value(today_data, positions)
        total_value = cash + position_value
        peak_value = max(peak_value, total_value)

        # 4. 写DB
        if self.account_id > 0 and self.run_id > 0:
            self._write_db(sell_orders, buy_orders, cash, position_value, total_value)

        return {"positions": positions, "cash": cash, "peak_value": peak_value,
                "orders": sell_orders + buy_orders, "total_value": total_value,
                "n_buys": len(buy_orders), "n_sells": len(sell_orders)}

    def _get_row(self, data, code):
        """兼容DataFrame index和column的取值"""
        if hasattr(data, 'index') and code in data.index:
            return data.loc[code]
        elif hasattr(data, 'loc') and 'code' in data.columns:
            rows = data[data["code"] == code]
            if not rows.empty:
                return rows.iloc[0]
        return None

    def _check_exits(self, today_data, positions, today_open_map):
        to_sell = []
        for code, pos in list(positions.items()):
            row = self._get_row(today_data, code)
            if row is None:
                continue
            close_p = float(row["close"])
            high_p = float(row.get("high", close_p))
            open_p = today_open_map.get(code, pos["cost"])

            pos["today_high"] = max(pos.get("today_high", open_p), high_p)

            # 跌停跳过
            is_3xx = str(code).startswith('3')
            limit_down_pct = 0.80 if is_3xx else 0.90
            if close_p <= open_p * limit_down_pct * 0.995:
                continue

            profit_pct = (close_p - pos["cost"]) / pos["cost"]

            # 硬止损 -3%
            if profit_pct <= -0.03:
                to_sell.append((code, "hard_stop", close_p))
                continue

            # 破开盘价
            if close_p < open_p:
                to_sell.append((code, "below_open", close_p))
                continue

            # 线性移动止盈
            day_gain = (close_p - open_p) / open_p if open_p > 0 else 0
            if day_gain > 0.01:
                today_high = pos.get("today_high", close_p)
                max_dd = min(0.02, 0.005 + (day_gain - 0.01) * (0.015 / 0.09))
                max_dd = max(0.003, max_dd)
                if close_p < today_high * (1 - max_dd):
                    to_sell.append((code, "trailing_stop", close_p))
                    continue
        return to_sell

    def _scan_entries(self, today_data, prev_data, positions, lookback_data, industry_map, cash):
        buy_orders = []
        if len(positions) >= self.stock_num or prev_data is None:
            return buy_orders, cash

        # 获取公共code集合
        today_codes = set()
        if hasattr(today_data, 'index'):
            today_codes = set(today_data.index)
        else:
            today_codes = set(today_data["code"].unique())

        prev_codes = set()
        if hasattr(prev_data, 'index'):
            prev_codes = set(prev_data.index)
        else:
            prev_codes = set(prev_data["code"].unique())

        common = today_codes & prev_codes
        candidates = []
        for code in common:
            if code in positions:
                continue
            t_row = self._get_row(today_data, code)
            p_row = self._get_row(prev_data, code)
            if t_row is None or p_row is None:
                continue
            today_open = float(t_row["open"])
            prev_close = float(p_row["close"])
            prev_low = float(p_row["low"])
            prev_high = float(p_row.get("high", prev_close))

            # 条件1: 低开 > 0.75%
            gap = (today_open - prev_close) / prev_close if prev_close > 0 else 0
            if not (today_open < prev_low and gap < -0.0075):
                continue

            # 条件2: 近4日振幅 < 10%
            lb = lookback_data.get(code, [])
            if len(lb) < 3:
                continue
            lb_closes = [r["close"] if isinstance(r, dict) else float(r) for r in lb[-4:]]
            lb_high = max(lb_closes)
            lb_low = min(lb_closes)
            if lb_low > 0 and (lb_high - lb_low) / lb_low > 0.10:
                continue

            # 条件3: 昨收 > 5日均线
            if len(lb) >= 5:
                ma5 = np.mean([r["close"] if isinstance(r, dict) else float(r) for r in lb[-5:]])
            else:
                ma5 = np.mean(lb_closes)
            if prev_close <= ma5:
                continue

            # 条件4: 昨日非涨停
            if prev_high > 0 and prev_close >= prev_high * 0.995:
                continue

            # 条件5: 行业分散
            ind = industry_map.get(code, "未知")
            existing_inds = {industry_map.get(c, "未知") for c in positions}
            if ind in existing_inds:
                continue

            candidates.append((code, gap))

        candidates.sort(key=lambda x: x[1])
        slots = self.stock_num - len(positions)
        for code, gap in candidates[:slots]:
            t_row = self._get_row(today_data, code)
            if t_row is None:
                continue
            price = float(t_row["close"])
            per_stock = cash / max(slots, 1)
            shares = int(per_stock / price / 100) * 100
            cost = shares * price * (1 + self.commission + self.slippage)
            if shares >= 100 and cost <= cash:
                cash -= cost
                buy_orders.append({"code": code, "price": price, "shares": shares, "gap": gap})
        return buy_orders, cash

    def _calc_position_value(self, today_data, positions):
        pv = 0.0
        for code, pos in positions.items():
            row = self._get_row(today_data, code)
            p = float(row["close"]) if row is not None else pos["cost"]
            pv += pos["shares"] * p
        return pv

    def _write_db(self, sell_orders, buy_orders, cash, position_value, total_value):
        engine = get_engine()
        try:
            with engine.begin() as conn:
                # 卖单: 更新 exit_date
                for o in sell_orders:
                    conn.execute(text("""
                        INSERT INTO paper_orders (account_id, code, direction, price, volume, amount, status, note)
                        VALUES (:aid, :code, :dir, :price, :vol, :amt, 'filled', :note)
                    """), {"aid": self.account_id, "code": o["code"], "dir": o["direction"],
                           "price": o["price"], "vol": o["volume"],
                           "amt": o["price"] * o["volume"], "note": o.get("reason", "")})
                    conn.execute(text("""
                        UPDATE paper_positions SET exit_date = CURRENT_DATE, exit_price = :ep
                        WHERE run_id = :rid AND stock_code = :sc AND exit_date IS NULL
                    """), {"rid": self.run_id, "sc": o["code"], "ep": o["price"]})
                # 买单
                for o in buy_orders:
                    conn.execute(text("""
                        INSERT INTO paper_orders (account_id, code, direction, price, volume, amount, status, note)
                        VALUES (:aid, :code, :dir, :price, :vol, :amt, 'filled', :note)
                    """), {"aid": self.account_id, "code": o["code"], "dir": o["direction"],
                           "price": o["price"], "vol": o["volume"],
                           "amt": o["price"] * o["volume"], "note": o.get("reason", "")})
                    conn.execute(text("""
                        INSERT INTO paper_positions (run_id, stock_code, entry_date, entry_price, quantity)
                        VALUES (:rid, :sc, CURRENT_DATE, :ep, :qty)
                    """), {"rid": self.run_id, "sc": o["code"], "ep": o["price"], "qty": o["volume"]})
                # 现金 + 净值
                conn.execute(text("UPDATE paper_account SET cash = :c WHERE id = :aid"),
                             {"c": cash, "aid": self.account_id})
                conn.execute(text("""
                    INSERT INTO paper_daily_pnl (account_id, trade_date, cash, position_value, total_value)
                    VALUES (:aid, CURRENT_DATE, :cash, :pv, :tv)
                    ON CONFLICT (account_id, trade_date) DO UPDATE SET total_value = :tv2
                """), {"aid": self.account_id, "cash": cash, "pv": position_value,
                       "tv": total_value, "tv2": total_value})
        finally:
            engine.dispose()

    def _empty_result(self, positions, cash, peak_value):
        return {"positions": positions, "cash": cash, "peak_value": peak_value,
                "orders": [], "total_value": cash, "n_buys": 0, "n_sells": 0}

    def _seasonal_clear(self, today_data, positions, cash, peak_value):
        for code in list(positions.keys()):
            row = self._get_row(today_data, code)
            p = float(row["close"]) if row is not None else positions[code]["cost"]
            cash += positions[code]["shares"] * p * (1 - self.stamp_duty - self.commission - self.slippage)
            del positions[code]
        return {"positions": {}, "cash": cash, "peak_value": max(peak_value, cash),
                "orders": [], "total_value": cash, "n_buys": 0, "n_sells": 0}
