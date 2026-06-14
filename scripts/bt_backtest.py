#!/usr/bin/env python
"""backtrader 涨停策略回测 —— 原生 Broker 管理现金/佣金/仓位，输出权益曲线+交易明细。

用法:
    python scripts/gen_signals.py --start 2025-01-01 --top-n 5 ...
    python scripts/bt_backtest.py --start 2025-01-01 --top-n 5 --signals data/signals/bt_signals.csv
"""
import sys, os, argparse, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import pandas as pd
from loguru import logger

from sqlalchemy import text
import backtrader as bt
from data.db import get_engine
from config.settings import TradingConfig


# ── A股佣金+印花税 ──
class CNStockComm(bt.CommInfoBase):
    params = (("commission", 0.00009), ("stamp_duty", 0.0005),
              ("stocklike", True), ("commtype", bt.CommInfoBase.COMM_PERC), ("percabs", True))

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission

    def getoperationcost(self, size, price):
        cost = abs(size) * price * self.p.commission
        if size < 0:
            cost += abs(size) * price * self.p.stamp_duty
        return cost


# ── 策略 ──
class LuStrategy(bt.Strategy):
    params = dict(
        top_n=5, signals_csv="", exit_stop=0.08, exec_close=True,
        # 移动止盈
        trailing_stop=0.0,            # >0 启用，从最高点回落 X% 卖出
        # 金字塔加仓
        pyramid_threshold=0.0,        # >0 启用，盈利>X% 触发
        pyramid_ratio=0.5,            # 加原仓位 X 倍
        # 入场冷却
        cooling_days=0,               # >0 启用，卖出后 N 天不买回
        require_positive_day=False,   # 买入当天收阳才买
    )

    def __init__(self):
        df = pd.read_csv(self.p.signals_csv)
        df["date"] = pd.to_datetime(df["date"])
        self.signals = {d: g for d, g in df.groupby("date")}
        self.sig_dates = set(self.signals.keys())
        self._trade_log = []       # 自记录交割单
        self._open_trades = {}     # {code: {entry_date, entry_price, quantity, highest_price}}
        self._bought_today = {}    # 本轮买入 {code: price}
        self._sold_today = {}      # 本轮卖出 {code: (price, reason)}
        self._today_signals = {}   # 今日信号 {code: close}
        self._today_skip_reasons = {}  # 跳过的候选 {code: reason}
        self._last_signal_date = None  # 上一个有信号的交易日
        self._recently_sold = {}   # {code: sell_date} 冷却期追踪
        self._manual_cash = None   # 手工跟踪现金（coc 延迟时用）

    def prenext(self):
        """委托给 next()，避免因部分 feed 无数据而阻塞整个回测。"""
        self.next()

    def next(self):
        # 确认上日挂单成交（coc 在 bar 之间结算）
        self._confirm_settled_buys()

        today = pd.Timestamp(self.datas[0].datetime.date(0))
        if today not in self.sig_dates:
            # 非信号日只同步持仓快照
            self._record_holdings()
            self._validate()
            return

        sigs = self.signals[today]
        self._last_signal_date = today
        targets = set(str(s.code).zfill(6) for _, s in sigs.iterrows())

        # ── 止损（支持固定止损 / 移动止盈）──
        stopped_out = set()
        stop_proceeds = 0.0
        NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
        for d in self.datas:
            if len(d) == 0:
                continue
            pos = self.getposition(d)
            if pos.size <= 0:
                continue

            # 确定止损价：移动止盈 vs 固定止损
            if self.p.trailing_stop > 0:
                highest = self._open_trades.get(d._name, {}).get("highest_price", pos.price)
                stop_price = highest * (1 - self.p.trailing_stop)
                triggered = d.close[0] < stop_price
            else:
                stop_price = pos.price * (1 - self.p.exit_stop)
                triggered = d.close[0] < stop_price

            if not triggered:
                continue

            # 跌停检查（与调仓卖出一致）
            prev_c = d.close[-1] if len(d) > 1 else d.close[0]
            if prev_c > 0 and TradingConfig.is_at_limit_down(d.close[0], prev_c, d._name):
                continue  # 跌停卖不掉

            reason = "移动止盈" if self.p.trailing_stop > 0 else "止损"
            self._record_exit(d._name, d.close[0], pos.size, pos.price, reason)
            self._sold_today[d._name] = (d.close[0], reason)
            stop_proceeds += pos.size * d.close[0] * NET_SELL
            self.close(data=d)
            stopped_out.add(d._name)
            # 记入冷却列表
            if self.p.cooling_days > 0:
                self._recently_sold[d._name] = pd.Timestamp(self.datas[0].datetime.date(0))

        # ── 调仓 ──
        held = {d._name for d in self.datas if len(d) > 0 and self.getposition(d).size > 0}
        to_sell = held - targets - stopped_out  # 排除已止损的

        # 计算同日卖出预期回款（含止损 + 调仓）
        sell_proceeds = stop_proceeds
        sold_codes = stopped_out.copy()  # 实际已卖出（含止损）
        for d in self.datas:
            if len(d) == 0:
                continue
            if d._name in to_sell:
                prev_c = d.close[-1] if len(d) > 1 else d.close[0]
                if prev_c > 0 and TradingConfig.is_at_limit_down(d.close[0], prev_c, d._name):
                    continue  # 跌停卖不掉
                pos = self.getposition(d)
                if pos.size > 0:
                    sell_proceeds += pos.size * d.close[0] * NET_SELL
                self._record_exit(d._name, d.close[0], pos.size if pos.size > 0 else 0, pos.price, "调仓")
                self._sold_today[d._name] = (d.close[0], "调仓")
                self.close(data=d)
                sold_codes.add(d._name)
                # 记入冷却列表
                if self.p.cooling_days > 0:
                    self._recently_sold[d._name] = pd.Timestamp(self.datas[0].datetime.date(0))

        portfolio_value = self.broker.getvalue()  # 已含所有持仓市值
        available_cash = self.broker.getcash() + sell_proceeds

        # ── 金字塔加仓（盈利>阈值时追加现有仓位）──
        if self.p.pyramid_threshold > 0:
            for d in self.datas:
                if len(d) == 0:
                    continue
                pos = self.getposition(d)
                if pos.size <= 0 or d._name in self._sold_today:
                    continue
                profit_pct = (d.close[0] - pos.price) / pos.price if pos.price > 0 else 0
                if profit_pct < self.p.pyramid_threshold:
                    continue
                # 加仓金额不超过原始仓位的 pyramid_ratio 倍
                add_val = pos.size * pos.price * self.p.pyramid_ratio
                add_sz = int(add_val / d.close[0] / 100) * 100
                if add_sz <= 0:
                    continue
                add_cost = add_sz * d.close[0] * (1 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE)
                if add_cost > available_cash * 0.3:  # 单次加仓不超过可用现金30%
                    add_sz = int(available_cash * 0.3 / d.close[0] / 100) * 100
                    if add_sz <= 0:
                        continue
                self.buy(data=d, size=add_sz)
                available_cash -= add_sz * d.close[0] * (1 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE)
                today_str = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
                self._trade_log.append({
                    "日期": today_str, "操作": "加仓", "股票代码": d._name, "股票名称": "",
                    "入场价": round(d.close[0], 2), "当前价/出场价": "",
                    "盈亏%": f"{profit_pct:+.1%}", "股数": add_sz, "入场日期": "",
                    "总资产": round(self.broker.getvalue(), 2),
                    "当前现金": round(self.broker.getcash(), 2),
                })
                logger.info(f"  [金字塔] {d._name} 加仓 {add_sz}股 @ {d.close[0]:.2f} (盈利{profit_pct:.1%})")

        available_feeds = {d._name for d in self.datas if len(d) > 0 and d._name != "bench"}
        # 从排名列表中按序买入，跳过涨停的，凑满 top_n 只
        target_ps = portfolio_value / max(self.p.top_n, 1)
        max_per_stock = target_ps * 1.5

        # 记录今日信号
        self._today_signals = {str(s.code).zfill(6): s.close for _, s in sigs.iterrows()}
        self._today_skip_reasons = {}

        for _, s in sigs.iterrows():
            code = str(s.code).zfill(6)
            # 跳过已持有或本轮已卖出（防 T+0）
            if code in self._open_trades or code in self._sold_today:
                self._today_skip_reasons[code] = "已持有"
                continue
            # ── 冷却期检查 ──
            if self.p.cooling_days > 0 and code in self._recently_sold:
                sell_dt = self._recently_sold[code]
                if (today - sell_dt).days <= self.p.cooling_days:
                    self._today_skip_reasons[code] = f"冷却期({(today-sell_dt).days}d)"
                    continue
            if code not in available_feeds:
                self._today_skip_reasons[code] = "无数据"
                continue
            d = next((x for x in self.datas if x._name == code), None)
            if d is None or len(d) == 0:
                self._today_skip_reasons[code] = "无数据"
                continue
            px = d.close[0]
            prev_c = d.close[-1] if len(d) > 1 else px
            # ── 收阳确认 ──
            if self.p.require_positive_day and px <= prev_c:
                self._today_skip_reasons[code] = "未收阳"
                continue
            if prev_c > 0 and TradingConfig.is_at_limit_up(px, prev_c, code):
                self._today_skip_reasons[code] = "涨停"
                continue

            slots_left = self.p.top_n - len(self._open_trades)
            if slots_left <= 0:
                self._today_skip_reasons[code] = "仓位已满"
                break
            est_cash_per_stock = available_cash / max(slots_left, 1)
            per_stock = min(est_cash_per_stock, max_per_stock)
            sz = int(per_stock / px / 100) * 100
            if sz <= 0:
                self._today_skip_reasons[code] = f"资金不足(qty=0)"
                continue
            self.buy(data=d, size=sz)
            self._bought_today[code] = px
            # 立即记入 _open_trades + 买入日志
            today_str = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
            self._open_trades[code] = {"entry_date": today_str, "entry_price": px, "quantity": sz}
            buy_cost = sz * px * (1 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE)
            cash_after = available_cash - buy_cost
            # 总资产 = 买入前资产 + 新持仓市值 - 交易成本
            total_after = round(portfolio_value + sz * px - buy_cost, 2)
            self._trade_log.append({
                "日期": today_str, "操作": "买入", "股票代码": code, "股票名称": "",
                "入场价": round(px, 2), "当前价/出场价": "", "盈亏%": "",
                "股数": sz, "入场日期": "",
                "总资产": total_after, "当前现金": round(cash_after, 2),
            })
            available_cash = cash_after

        # 手工现金（coc 下 broker.getcash() 当天不变，用此变量）
        self._manual_cash = available_cash

        # 日终持仓快照 + 校验
        self._record_holdings()
        self._validate()

    def _validate(self):
        """日终全面校验：资金 / 持仓 / 买卖逻辑 / 止损 / 顺延 / 信号。"""
        today = self.datas[0].datetime.date(0)
        cash = self.broker.getcash()

        # ── 1. 资金 ──
        assert cash >= -0.01, f"[{today}] 现金为负: {cash:.2f}"

        bt_pv = 0.0
        bt_held = {}
        for d in self.datas:
            if len(d) == 0: continue
            sz = float(self.getposition(d).size)
            if sz > 0:
                bt_pv += sz * d.close[0]
                bt_held[d._name] = (sz, self.getposition(d).price, d.close[0])
        nav = cash + bt_pv
        broker_nav = self.broker.getvalue()
        drift = abs(nav - broker_nav) / max(broker_nav, 1)
        assert drift < 0.01, \
            f"[{today}] 总资产不匹配: nav={nav:.0f} broker={broker_nav:.0f} drift={drift:.2%}"

        # ── 2. 持仓上限 ──
        assert len(self._open_trades) <= self.p.top_n, \
            f"[{today}] 持仓 {len(self._open_trades)} > top_n={self.p.top_n}"

        # ── 3. 买入必须来自当日信号 ──
        for code in self._bought_today:
            assert code in self._today_signals, \
                f"[{today}] 买入 {code} 不在当日信号中! signals={list(self._today_signals.keys())[:10]}"

        # ── 4. 买入不能是涨停板 ──
        for code, px in self._bought_today.items():
            d = next((x for x in self.datas if x._name == code), None)
            if d is None or len(d) < 2: continue
            prev_c = d.close[-1]
            assert not TradingConfig.is_at_limit_up(px, prev_c, code), \
                f"[{today}] 买入 {code} 涨停封板! close={px} prev={prev_c}"

        # ── 5. 卖出不能是跌停板 ──
        for code, (px, reason) in self._sold_today.items():
            d = next((x for x in self.datas if x._name == code), None)
            if d is None or len(d) < 2: continue
            prev_c = d.close[-1]
            assert not TradingConfig.is_at_limit_down(px, prev_c, code), \
                f"[{today}] 卖出 {code}({reason}) 跌停封板! close={px} prev={prev_c}"

        # ── 6. T+0 禁止：同日不能又买又卖同一只 ──
        t0 = set(self._bought_today) & set(self._sold_today)
        assert len(t0) == 0, f"[{today}] T+0 交易: {t0}"

        # ── 7. 止损：持仓收盘价 < 入场价*0.92 必须已触发卖出 ──
        if today in self.sig_dates:
            for code, entry in self._open_trades.items():
                d = next((x for x in self.datas if x._name == code), None)
                if d is None or len(d) == 0: continue
                close_p = d.close[0]
                stop_line = entry["entry_price"] * (1 - self.p.exit_stop)
                if close_p < stop_line and code not in self._sold_today:
                    # 检查是否跌停封死（无法卖）
                    prev_c = d.close[-1] if len(d) > 1 else close_p
                    if not TradingConfig.is_at_limit_down(close_p, prev_c, code):
                        assert False, \
                            f"[{today}] {code} 止损未触发! close={close_p:.2f} entry={entry['entry_price']:.2f} stop={stop_line:.2f}"

        # ── 8. 当持有不到 top_n 且有未涨停候选时，必须继续买入 ──
        if today in self.sig_dates and len(self._open_trades) < self.p.top_n:
            remaining = self.p.top_n - len(self._open_trades)
            skipped_buyable = 0
            for _, s in self.signals[today].iterrows():
                code = str(s.code).zfill(6)
                if code in self._open_trades: continue
                reason = self._today_skip_reasons.get(code, "")
                if reason in ("仓位已满",): continue  # 正常的停止原因
                if reason in ("涨停",): continue  # 涨停不能买
                # 还能买但跳过了
                skipped_buyable += 1
            # 只告警不中断：可能因为资金不足
            if skipped_buyable > 0:
                logger.debug(f"[{today}] 尚有 {remaining} 个空位但跳过了 {skipped_buyable} 个可买候选")

        # ── 9. 持仓股不在信号也不在卖出 → 可能跌停封死或遗漏 ──
        if today in self.sig_dates:
            targets = set(str(s.code).zfill(6) for _, s in self.signals[today].iterrows())
            for code in self._open_trades:
                if code not in targets and code not in self._sold_today:
                    # 检查是否因跌停无法卖出
                    d = next((x for x in self.datas if x._name == code), None)
                    is_ld = False
                    if d and len(d) >= 2:
                        is_ld = TradingConfig.is_at_limit_down(d.close[0], d.close[-1], code)
                    if not is_ld:
                        logger.warning(f"[{today}] {code} 不在信号也不在卖出列表，疑似调仓遗漏")

        # ── 10. 卖出必须有原因 ──
        VALID_REASONS = ("止损", "调仓", "移动止盈")
        for code, (_, reason) in self._sold_today.items():
            assert reason in VALID_REASONS, \
                f"[{today}] {code} 卖出原因异常: {reason}"

        # 清理当日追踪
        self._bought_today.clear()
        self._sold_today.clear()
        self._today_signals.clear()
        self._today_skip_reasons.clear()
        """买入时暂不记录——等 _record_holdings 确认 backtrader 有仓位再说。"""
        pass

    def _record_exit(self, code, exit_price, size, entry_price, reason):
        """记录卖出并从 _open_trades 移除。"""
        entry = self._open_trades.pop(code, {})
        entry_px = entry.get("entry_price", entry_price)
        entry_qty = entry.get("quantity", size)
        today = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        pnl_pct = round((exit_price / entry_px - 1) * 100, 2) if entry_px > 0 else 0
        total = round(self.broker.getvalue(), 2)
        self._trade_log.append({
            "日期": today, "操作": f"卖出({reason})", "股票代码": code, "股票名称": "",
            "入场价": round(entry_px, 2), "当前价/出场价": round(exit_price, 2),
            "盈亏%": pnl_pct, "股数": entry_qty, "入场日期": "", "总资产": total, "当前现金": round(self.broker.getcash(), 2),
        })

    def _confirm_settled_buys(self):
        """确认 backtrader 已结算买入——只同步数量，入场价保持下单价不变。"""
        for d in self.datas:
            if len(d) == 0: continue
            pos = self.getposition(d)
            if pos.size > 0 and pos.price > 0 and d._name in self._open_trades:
                # 只更新数量（coc 模式下应与下单数量一致），入场价固定
                self._open_trades[d._name]["quantity"] = int(pos.size)

    def _record_holdings(self):
        """记录当日持仓快照。同步 _open_trades 与 backtrader（清理已结算卖出）。"""
        today = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        # 按当天收盘算持仓总市值（用于总资产 = 市值 + 现金）
        pos_val = 0.0
        for code, entry in self._open_trades.items():
            d = next((x for x in self.datas if x._name == code), None)
            if d is not None and len(d) > 0 and d.close[0] > 0:
                pos_val += entry["quantity"] * d.close[0]
            else:
                pos_val += entry["quantity"] * entry["entry_price"]

        # 手工总资产优先（coc 日 broker 未结算），否则回退 broker
        if self._manual_cash is not None:
            total = round(self._manual_cash + pos_val, 2)
            calc_cash = round(self._manual_cash, 2)
        else:
            total = round(self.broker.getvalue(), 2)
            calc_cash = round(total - pos_val, 2)

        # 清理 backtrader 已结算的卖出
        bt_held = set()
        for d in self.datas:
            if len(d) == 0: continue
            pos = self.getposition(d)
            if pos.size > 0 and pos.price > 0:
                bt_held.add(d._name)
        for code in list(self._open_trades.keys()):
            if code not in bt_held and code not in self._bought_today:
                del self._open_trades[code]

        # 记录所有持仓
        for code, entry in self._open_trades.items():
            d = next((x for x in self.datas if x._name == code), None)
            if d is None or len(d) == 0: continue
            cur_price = d.close[0]
            prev_high = entry.get("highest_price", entry["entry_price"])
            entry["highest_price"] = max(prev_high, cur_price)
            pnl_pct = round((cur_price / entry["entry_price"] - 1) * 100, 2) if entry["entry_price"] > 0 else 0
            self._trade_log.append({
                "日期": today, "操作": "持仓", "股票代码": code, "股票名称": "",
                "入场价": round(entry["entry_price"], 2), "当前价/出场价": round(cur_price, 2),
                "盈亏%": pnl_pct, "股数": entry["quantity"], "入场日期": "",
                "总资产": total, "当前现金": calc_cash,
            })

# ── 权益曲线 Observer ──
class EquityObserver(bt.Observer):
    """记录每日 (date, value, cash) 到列表，跑完后写入 JSON"""
    lines = ("value", "cash")
    plotinfo = dict(plot=False)

    def __init__(self):
        self._history = []

    def prenext(self):
        self.next()

    def next(self):
        dt = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        self._history.append({
            "date": dt,
            "value": round(self._owner.broker.getvalue(), 2),
            "cash": round(self._owner.broker.getcash(), 2),
        })


# ── 回测主函数 ──
def run_bt(args):
    cerebro = bt.Cerebro()
    tc = TradingConfig()

    # Broker
    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(
        CNStockComm(commission=tc.COMMISSION, stamp_duty=tc.STAMP_DUTY))
    cerebro.broker.set_slippage_perc(tc.SLIPPAGE)

    # Cheat-On-Close：--exec-close 时订单在当日收盘价成交（模拟 14:50 行情→收盘执行）
    if args.exec_close:
        cerebro.broker.set_coc(True)

    # 信号
    sdf = pd.read_csv(args.signals)
    sdf["date"] = pd.to_datetime(sdf["date"])
    codes = [str(c).zfill(6) for c in sorted(sdf["code"].unique())]
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end) if args.end else sdf["date"].max()

    # 加载日线
    engine = get_engine()
    daily = pd.read_sql(
        text("SELECT code, trade_date, open, high, low, close, volume FROM stock_daily "
             "WHERE code = ANY(:codes) AND trade_date BETWEEN :s AND :e "
             "ORDER BY code, trade_date"),
        engine,
        params={
            "codes": codes,
            "s": (start_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "e": (end_dt + timedelta(days=5)).strftime("%Y-%m-%d"),
        },
    )
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    engine.dispose()

    added = 0
    for c, g in daily.groupby("code"):
        g = g.sort_values("trade_date").set_index("trade_date")
        if len(g) < 30:
            continue
        d = bt.feeds.PandasData(dataname=g, datetime=None, open="open", high="high",
                                low="low", close="close", volume="volume", name=c, plot=False)
        cerebro.adddata(d, name=c)
        added += 1

    # 策略 + Observer + Analyzer
    py_thresh, py_ratio = args.pyramid if args.pyramid else (0.0, 0.5)
    cerebro.addstrategy(LuStrategy, top_n=args.top_n, signals_csv=args.signals,
                         exit_stop=args.exit_stop, exec_close=args.exec_close,
                         trailing_stop=args.trailing_stop or 0.0,
                         pyramid_threshold=py_thresh,
                         pyramid_ratio=py_ratio,
                         cooling_days=args.cooling_days if args.cooling else 0,
                         require_positive_day=args.require_positive_day)
    cerebro.addobserver(EquityObserver)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="ann")

    logger.info(f"回测 {args.start} → {end_dt.strftime('%Y-%m-%d')}, "
                f"{added}只, 本金{args.cash:,.0f}")
    results = cerebro.run()
    strat = results[0]
    fv = cerebro.broker.getvalue()
    ret = (fv - args.cash) / args.cash

    # ── 权益曲线提取（先于终端输出，供自行计算 MDD）──
    obs = [o for o in strat.getobservers() if hasattr(o, '_history')][0]
    equity = obs._history

    # 自行计算最大回撤（峰-谷算法），比 backtrader 内置 analyzer 更可靠
    equity_values = [e["value"] for e in equity]
    mdd = 0.0
    if equity_values:
        peak = equity_values[0]
        for v in equity_values:
            if v > peak:
                peak = v
            dd_pct = (peak - v) / peak if peak > 0 else 0.0
            mdd = max(mdd, dd_pct)

    # ── 终端输出 ──
    print(f"\n{'='*60}")
    print(f"  backtrader 涨停策略 Top-{args.top_n}")
    print(f"  本金 {args.cash:,.0f} | 终值 {fv:,.0f} | 收益 {ret:+.1%}")
    print(f"  成本: 买{tc.COMMISSION+tc.SLIPPAGE:.4%} 卖{tc.COMMISSION+tc.STAMP_DUTY+tc.SLIPPAGE:.4%}")
    print(f"  执行: {'收盘价 (coc)' if args.exec_close else 'T+1开盘价'}")
    sh = strat.analyzers.sharpe.get_analysis()
    if sh and sh.get("sharperatio") is not None:
        print(f"  Sharpe: {sh['sharperatio']:.2f}")
    print(f"  最大回撤: {mdd:.1%}")
    ann = strat.analyzers.ann.get_analysis()
    if ann:
        print("  年度:")
        for y, r in sorted(ann.items()):
            print(f"    {y}: {r:+.1%}")
    tr = strat.analyzers.trades.get_analysis()
    n_tr = 0
    if tr and hasattr(tr, "total") and hasattr(tr.total, "total"):
        n_tr = int(tr.total.total)
    print(f"  交易 {n_tr}笔")
    print(f"{'='*60}\n")

    # ── 权益曲线导出 ──
    os.makedirs("data/backtest_trades", exist_ok=True)
    equity_path = f"data/backtest_trades/equity_top{args.top_n}.json"
    with open(equity_path, "w") as f:
        json.dump({
            "params": {"start": args.start, "end": str(end_dt.date()),
                       "top_n": args.top_n, "cash": args.cash,
                       "final_value": round(fv, 2), "return": round(ret, 6)},
            "equity": equity,
        }, f, ensure_ascii=False)
    logger.info(f"权益曲线: {equity_path} ({len(equity)}天)")

    # ── 交易明细导出 ──
    # 加载股票名称
    name_map = {}
    try:
        sdf_names = pd.read_csv(args.signals)
        name_map = dict(zip(sdf_names["code"].astype(str).str.zfill(6), sdf_names["name"]))
    except Exception:
        pass

    trades_path = f"data/backtest_trades/trades_top{args.top_n}.csv"
    trade_log = getattr(strat, "_trade_log", [])
    with open(trades_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日期", "操作", "股票代码", "股票名称", "入场价", "当前价/出场价", "盈亏%", "股数", "入场日期", "总资产", "当前现金"])
        # 按日期排序，同一天内买入→持仓→卖出
        def _sort_key(t):
            op = t["操作"]
            if op.startswith("卖出"): order = 0   # 先卖
            elif op == "买入": order = 1           # 后买
            else: order = 2                        # 持仓
            return (t["日期"], order)
        trade_log.sort(key=_sort_key)
        for t in trade_log:
            code = str(t["股票代码"]).zfill(6)
            t["股票名称"] = name_map.get(code, "")
            w.writerow([t["日期"], t["操作"], code, t["股票名称"],
                        t["入场价"], t["当前价/出场价"],
                        t["盈亏%"], t["股数"], t["入场日期"], t["总资产"], t.get("当前现金", "")])
    logger.info(f"交易明细: {trades_path} ({len(trade_log)} 笔)")

    # ── DB 写入 backtest_results（Web 展示用）──
    eq_dict = {e["date"]: e["value"] for e in equity}
    metrics = {
        "start": args.start, "end": str(end_dt.date()), "top_n": args.top_n,
        "cash": args.cash, "final_value": round(fv, 2), "return": round(ret, 6),
        "sharpe": round(sh["sharperatio"], 2) if sh and sh.get("sharperatio") is not None else None,
        "max_drawdown": round(mdd, 4), "trades": n_tr, "exec_mode": "close" if args.exec_close else "T+1_open",
    }
    dr_dict = {}  # daily returns can be computed from equity if needed
    # ── 确定 version_id ──
    try:
        eng = get_engine()
        if args.variant_label:
            version_id = _ensure_strategy_version(eng, args.variant_label)
        elif args.version_id:
            version_id = args.version_id
        else:
            version_id = 49  # 兼容旧调用
        with eng.begin() as conn:
            conn.execute(text("""
                INSERT INTO backtest_results (version_id, start_date, end_date,
                    quality, metrics_json, equity_curve_json, daily_returns_json)
                VALUES (:v, :s, :e, 'valid', :m, :eq, :dr)
            """), {"v": version_id, "s": args.start, "e": str(end_dt.date()),
                   "m": json.dumps(metrics, ensure_ascii=False),
                   "eq": json.dumps(eq_dict), "dr": json.dumps(dr_dict)})
        eng.dispose()
        logger.info(f"回测结果已写入 backtest_results (version_id={version_id})")
    except Exception as e:
        logger.warning(f"DB 写入失败（Web 无法展示本次回测）: {e}")

    return {"final_value": fv, "return": ret, "equity": equity}


def _ensure_strategy_version(engine, label: str) -> int:
    """确保 strategy_configs + strategy_versions 有对应行，返回 version_id。"""
    with engine.begin() as conn:
        # 确保 strategy_configs 存在（先查后插，避免 RETURNING 在 ON CONFLICT DO NOTHING 返回空）
        sid = conn.execute(
            text("SELECT id FROM strategy_configs WHERE name = 'limit_up'")
        ).scalar()
        if not sid:
            conn.execute(text("""
                INSERT INTO strategy_configs (name, type, description)
                VALUES ('limit_up', 'static', '涨停动量策略（规则型）')
                ON CONFLICT (name) DO NOTHING
            """))
            sid = conn.execute(
                text("SELECT id FROM strategy_configs WHERE name = 'limit_up'")
            ).scalar()
        if not sid:
            raise RuntimeError("无法创建 strategy_configs 行")

        # 确保 strategy_versions 存在
        conn.execute(text("""
            INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version)
            VALUES (:sid, :ver, 'rule_based', 'v1')
            ON CONFLICT (strategy_id, version) DO NOTHING
        """), {"sid": sid, "ver": label})
        vid = conn.execute(
            text("SELECT id FROM strategy_versions WHERE strategy_id = :sid AND version = :ver"),
            {"sid": sid, "ver": label},
        ).scalar()
        if not vid:
            raise RuntimeError(f"无法创建 strategy_versions 行: sid={sid}, ver={label}")
        return vid


def parse_args():
    p = argparse.ArgumentParser(description="backtrader 涨停策略回测")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=TradingConfig.INITIAL_CASH)
    p.add_argument("--signals", default="data/signals/bt_signals.csv")
    p.add_argument("--exec-close", action="store_true")
    p.add_argument("--exit-stop", type=float, default=0.08)
    p.add_argument("--variant-label", type=str, default=None,
                   help="策略变体标签，自动注册 strategy_configs/versions")
    p.add_argument("--version-id", type=int, default=None,
                   help="直接指定 strategy_versions.id（覆盖 variant-label）")
    # 移动止盈
    p.add_argument("--trailing-stop", type=float, default=0.0,
                   help=">0 启用移动止盈，从最高点回落 X 卖出")
    # 金字塔加仓
    p.add_argument("--pyramid", nargs=2, type=float, default=None,
                   metavar=("THRESHOLD", "RATIO"),
                   help="金字塔加仓: 盈利阈值 加仓比例")
    # 入场冷却
    p.add_argument("--cooling", action="store_true", help="启用入场冷却期")
    p.add_argument("--cooling-days", type=int, default=3, help="冷却天数")
    p.add_argument("--require-positive-day", action="store_true", help="买入当天必须收阳")
    return p.parse_args()


if __name__ == "__main__":
    run_bt(parse_args())
