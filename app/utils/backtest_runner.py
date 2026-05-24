import io
from datetime import datetime, timedelta
from typing import Any

import backtrader as bt
import pandas as pd


def _mpl_to_datetime(mpl_float: float) -> pd.Timestamp:
    """matplotlib date number → pandas Timestamp。"""
    return pd.Timestamp(datetime(1, 1, 1) + timedelta(days=mpl_float))


class PgDataFeed(bt.feeds.PandasData):
    """自定义数据源：映射 DB 查询结果的字段名到 backtrader。"""

    params = (
        ("datetime", "trade_date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class TradeRecorder(bt.Analyzer):
    """通过 notify_order 记录每笔已完成交易的详情。"""

    def __init__(self):
        self.trades = []
        self._pending_entries = []  # [(price, size, dt)]

    def notify_order(self, order):
        if order.status not in [order.Completed]:
            return
        if order.isbuy():
            self._pending_entries.append((
                order.executed.price,
                order.executed.size,
                order.executed.dt,
            ))
        else:
            # 按 FIFO 匹配
            exit_size = abs(order.executed.size)
            remaining = exit_size
            while remaining > 0 and self._pending_entries:
                entry_price, entry_size, entry_dt = self._pending_entries.pop(0)
                matched = min(entry_size, remaining)
                exit_price = order.executed.price
                pnl = (exit_price - entry_price) * matched
                cost = entry_price * matched
                self.trades.append({
                    "entry_date": _mpl_to_datetime(entry_dt),
                    "exit_date": _mpl_to_datetime(order.executed.dt),
                    "direction": "买入",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "size": matched,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / cost * 100, 2) if cost else 0,
                })
                if entry_size > matched:
                    self._pending_entries.insert(0, (entry_price, entry_size - matched, entry_dt))
                remaining -= matched

    def get_analysis(self):
        return self.trades


def run_backtest(
    strategy_class: type,
    df: pd.DataFrame,
    strategy_params: dict[str, Any] | None = None,
    initial_cash: float = 100000,
    commission: float = 0.0003,
    slippage: float = 0.01,
) -> dict:
    """
    运行回测，返回 {metrics, equity_curve, trades}。

    参数
    ----
    strategy_class : backtrader.Strategy 子类
    df : 需包含 trade_date, open, high, low, close, volume
    strategy_params : 策略参数字典
    initial_cash : 初始资金
    commission : 佣金费率
    slippage : 固定滑点（元）
    """
    cerebro = bt.Cerebro()

    # 数据
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    data = PgDataFeed(dataname=df)
    cerebro.adddata(data)

    # 策略
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)

    # 仓位管理：每次买入用 95% 可用资金
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # 资金 & 费用
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_fixed(slippage)

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(TradeRecorder, _name="traderecorder")

    # 权益记录
    cerebro.addwriter(bt.WriterFile, csv=True, out=io.StringIO())

    results = cerebro.run()
    strat = results[0]

    # ---- 提取指标 ----
    metrics = {}

    sharpe = strat.analyzers.sharpe.get_analysis()
    metrics["sharpe_ratio"] = sharpe.get("sharperatio", 0) or 0

    dd = strat.analyzers.drawdown.get_analysis()
    metrics["max_drawdown"] = dd.get("max", {}).get("drawdown", 0) or 0

    returns = strat.analyzers.returns.get_analysis()
    metrics["total_return"] = returns.get("rtot", 0) or 0
    # 年化收益
    rtn_dict = returns or {}
    years = (df["trade_date"].max() - df["trade_date"].min()).days / 365.25
    if years > 0 and rtn_dict.get("rtot"):
        metrics["annual_return"] = (1 + rtn_dict["rtot"]) ** (1 / years) - 1
    else:
        metrics["annual_return"] = 0

    ta = strat.analyzers.trades.get_analysis()
    total = ta.get("total", {}) or {}
    won = ta.get("won", {}) or {}
    lost = ta.get("lost", {}) or {}
    metrics["total_trades"] = total.get("total", 0) or total.get("closed", 0)
    metrics["won_trades"] = won.get("total", 0)
    metrics["lost_trades"] = lost.get("total", 0)
    if metrics["total_trades"] > 0:
        metrics["win_rate"] = metrics["won_trades"] / metrics["total_trades"]

    metrics["final_value"] = cerebro.broker.getvalue()

    # ---- 权益曲线 ----
    eq_dict = strat.analyzers.timereturn.get_analysis()
    equity_curve = pd.DataFrame(
        {"date": list(eq_dict.keys()), "return": list(eq_dict.values())}
    )
    if not equity_curve.empty:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        equity_curve["equity"] = initial_cash * (1 + equity_curve["return"]).cumprod()

    # ---- 交易明细 ----
    raw_trades = strat.analyzers.traderecorder.get_analysis() or []
    trades_df = pd.DataFrame(raw_trades)

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades_df,
    }
