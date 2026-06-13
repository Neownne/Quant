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
    params = dict(top_n=5, signals_csv="", exit_stop=0.08, exec_close=True)

    def __init__(self):
        df = pd.read_csv(self.p.signals_csv)
        df["date"] = pd.to_datetime(df["date"])
        self.signals = {d: g for d, g in df.groupby("date")}
        self.sig_dates = set(self.signals.keys())

    def next(self):
        today = pd.Timestamp(self.datas[0].datetime.date(0))
        if today not in self.sig_dates:
            return

        sigs = self.signals[today]
        targets = set(str(s.code) for _, s in sigs.iterrows())

        # ── 止损 ──
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size <= 0 or d.close[0] >= pos.price * (1 - self.p.exit_stop):
                continue
            self.close(data=d)

        # ── 调仓 ──
        held = {d._name for d in self.datas if self.getposition(d).size > 0}
        to_sell = held - targets

        for d in self.datas:
            if d._name in to_sell:
                prev_c = d.close[-1] if len(d) > 1 else d.close[0]
                if prev_c > 0 and d.close[0] / prev_c - 1 <= TradingConfig.LIMIT_DOWN_PCT:
                    continue  # 跌停卖不掉
                self.close(data=d)

        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        to_buy = targets - held
        n = len(to_buy)
        if n == 0:
            return

        target_ps = portfolio_value / max(self.p.top_n, 1)
        per_stock = min(cash / n, target_ps * 1.5)

        for d in self.datas:
            code = d._name
            if code not in to_buy:
                continue
            px = d.close[0] if self.p.exec_close else d.open[0]
            if self.p.exec_close:
                prev_c = d.close[-1] if len(d) > 1 else px
                if prev_c > 0 and px / prev_c - 1 >= TradingConfig.LIMIT_UP_PCT:
                    continue  # 涨停买不到
            sz = int(per_stock / px / 100) * 100
            if sz <= 0:
                continue
            self.buy(data=d, size=sz)


# ── 权益曲线 Observer ──
class EquityObserver(bt.Observer):
    """记录每日 (date, value, cash) 到列表，跑完后写入 JSON"""
    lines = ("value", "cash")
    plotinfo = dict(plot=False)

    def __init__(self):
        self._history = []

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

    # 信号
    sdf = pd.read_csv(args.signals)
    sdf["date"] = pd.to_datetime(sdf["date"])
    codes = [str(int(c)) for c in sorted(sdf["code"].unique())]
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
    cerebro.addstrategy(LuStrategy, top_n=args.top_n, signals_csv=args.signals,
                         exit_stop=args.exit_stop, exec_close=args.exec_close)
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

    # ── 终端输出 ──
    print(f"\n{'='*60}")
    print(f"  backtrader 涨停策略 Top-{args.top_n}")
    print(f"  本金 {args.cash:,.0f} | 终值 {fv:,.0f} | 收益 {ret:+.1%}")
    print(f"  成本: 买{tc.COMMISSION+tc.SLIPPAGE:.4%} 卖{tc.COMMISSION+tc.STAMP_DUTY+tc.SLIPPAGE:.4%}")
    sh = strat.analyzers.sharpe.get_analysis()
    if sh and sh.get("sharperatio") is not None:
        print(f"  Sharpe: {sh['sharperatio']:.2f}")
    dd = strat.analyzers.dd.get_analysis()
    if dd and "max" in dd:
        print(f"  最大回撤: {dd['max']['drawdown']:.1%}")
    ann = strat.analyzers.ann.get_analysis()
    if ann:
        print("  年度:")
        for y, r in sorted(ann.items()):
            print(f"    {y}: {r:+.1%}")
    tr = strat.analyzers.trades.get_analysis()
    n_tr = 0
    for data_key, trade_list in tr.items():
        if not isinstance(trade_list, list):
            continue
        n_tr += len(trade_list)
    print(f"  交易 {n_tr}笔")
    print(f"{'='*60}\n")

    # ── 权益曲线导出 ──
    obs = [o for o in strat.getobservers() if hasattr(o, '_history')][0]
    equity = obs._history
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
    trades_path = f"data/backtest_trades/trades_top{args.top_n}.csv"
    with open(trades_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["入场日期", "出场日期", "股票代码", "入场价", "出场价", "盈亏%", "盈亏额", "出场原因"])
        for data_key, trade_list in tr.items():
            if not isinstance(trade_list, list):
                continue
            for tk in trade_list:
                if not hasattr(tk, "history") or len(tk.history) < 2:
                    continue
                e = tk.history[0].status
                x = tk.history[-1].status
                ep, xp = e.price, x.price
                pnl_pct = (xp - ep) / ep if ep else 0
                w.writerow([
                    str(e.dt)[:10] if e.dt else "",
                    str(x.dt)[:10] if x.dt else "",
                    str(data_key), round(ep, 2), round(xp, 2),
                    round(pnl_pct * 100, 2), round(tk.pnl, 2), ""])
    logger.info(f"交易明细: {trades_path}")

    # ── DB 写入（略，backtrader 结果通过 JSON 文件服务）──
    return {"final_value": fv, "return": ret, "equity": equity}


def parse_args():
    p = argparse.ArgumentParser(description="backtrader 涨停策略回测")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=TradingConfig.INITIAL_CASH)
    p.add_argument("--signals", default="data/signals/bt_signals.csv")
    p.add_argument("--exec-close", action="store_true")
    p.add_argument("--exit-stop", type=float, default=0.08)
    return p.parse_args()


if __name__ == "__main__":
    run_bt(parse_args())
