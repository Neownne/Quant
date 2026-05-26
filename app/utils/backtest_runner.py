import io
from datetime import datetime, timedelta
from typing import Any

import backtrader as bt
import pandas as pd


def _mpl_to_datetime(mpl_float: float) -> pd.Timestamp:
    """matplotlib date number → pandas Timestamp。"""
    return pd.Timestamp(datetime(1, 1, 1) + timedelta(days=mpl_float))


# 模块级缓存：每个 Worker 进程只加载一次上证指数数据
_index_df_cache: pd.DataFrame | None = None


def _cached_index_df(start: str = "20150101", end: str = "20300101") -> pd.DataFrame:
    global _index_df_cache
    if _index_df_cache is not None:
        return _index_df_cache
    _index_df_cache = load_index_data(start, end)
    return _index_df_cache


class AShareCommission(bt.CommInfoBase):
    """A 股交易费用：佣金（买卖双向）+ 印花税（卖出单向 0.05%）。

    买方费用 = 成交额 × 佣金费率
    卖方费用 = 成交额 × 佣金费率 + 成交额 × 印花税率
    """

    params = (
        ("commission", 0.00009),   # 佣金 万分之 0.9
        ("stamp_duty", 0.0005),    # 印花税 万分之 5（仅卖出）
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        value = abs(size) * price
        comm = value * self.p.commission
        if size < 0:
            comm += value * self.p.stamp_duty
        return comm


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


class MarketAwareSizer(bt.Sizer):
    """动态仓位管理：大盘在 200 日均线上方用全仓，下方自动降至防守仓位。

    参数
    ----
    percents : 牛市仓位百分比（默认 95%）
    bear_percent : 熊市仓位百分比（默认 40%）
    ma_period : 均线周期（默认 200）
    """

    params = (
        ("percents", 95),
        ("bear_percent", 40),
        ("ma_period", 200),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        pct = self.p.percents
        strat = self.strategy

        # 如果传入了指数数据（data1），用年线判断牛熊
        if strat and hasattr(strat, "datas") and len(strat.datas) > 1:
            idx = strat.datas[1]
            period = self.p.ma_period
            if len(idx) >= period:
                sma = sum(idx.close[-i] for i in range(period)) / period
                if idx.close[0] < sma:
                    pct = self.p.bear_percent

        size = cash * (pct / 100) / data.close[0]
        return size


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
                    "买入日期": _mpl_to_datetime(entry_dt),
                    "卖出日期": _mpl_to_datetime(order.executed.dt),
                    "买入价": round(entry_price, 2),
                    "卖出价": round(exit_price, 2),
                    "数量(股)": matched,
                    "盈亏(元)": round(pnl, 2),
                    "盈亏%": round(pnl / cost * 100, 2) if cost else 0,
                })
                if entry_size > matched:
                    self._pending_entries.insert(0, (entry_price, entry_size - matched, entry_dt))
                remaining -= matched

    def get_analysis(self):
        return self.trades


class SignalRecorder(bt.Analyzer):
    """记录每笔买卖信号（含日期、方向、价格、数量），用于模拟盘回放。"""

    def __init__(self):
        self.signals: list[dict] = []

    def notify_order(self, order):
        if order.status not in [order.Completed]:
            return
        dt = _mpl_to_datetime(order.executed.dt) if hasattr(order.executed, 'dt') else None
        if dt is None:
            dt = order.executed.dt
        self.signals.append({
            "date": dt,
            "direction": "BUY" if order.isbuy() else "SELL",
            "price": round(order.executed.price, 2),
            "size": int(abs(order.executed.size)),
        })

    def get_analysis(self):
        return self.signals


def load_index_data(start: str = "20150101", end: str = "20300101") -> pd.DataFrame:
    """加载上证指数（000001）日线数据，用于大盘过滤器。"""
    from sqlalchemy import text
    from data.db import get_engine
    engine = get_engine()
    sql = """
        SELECT trade_date, open, high, low, close, volume
        FROM index_daily
        WHERE code = '000001'
          AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params={"start": start, "end": end})
    engine.dispose()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# -- 基准指数定义 --
BENCHMARK_INDICES: dict[str, str] = {
    "000001": "上证指数",
    "399001": "深证成指",
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
}


def load_benchmark_indices(start: str, end: str) -> dict[str, pd.DataFrame]:
    """批量加载基准指数日线数据。

    Returns: {code: DataFrame(trade_date, close)}
    """
    from sqlalchemy import text
    from data.db import get_engine
    engine = get_engine()
    codes = list(BENCHMARK_INDICES.keys())
    placeholders = ",".join([f":c{i}" for i in range(len(codes))])
    sql = f"""
        SELECT code, trade_date, close
        FROM index_daily
        WHERE code IN ({placeholders})
          AND trade_date BETWEEN :start AND :end
        ORDER BY code, trade_date
    """
    params = {f"c{i}": c for i, c in enumerate(codes)}
    params["start"] = start
    params["end"] = end
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    engine.dispose()
    if df.empty:
        return {}
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return {code: group.drop(columns=["code"]) for code, group in df.groupby("code")}


def compute_benchmark_returns(
    benchmarks: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """计算各基准指数在指定区间内的买入持有收益率。"""
    result = {}
    for code, df in benchmarks.items():
        if df.empty or len(df) < 2:
            result[code] = 0.0
            continue
        first = df[df["trade_date"] >= pd.Timestamp(start_date)]
        last = df[df["trade_date"] <= pd.Timestamp(end_date)]
        if first.empty or last.empty:
            result[code] = 0.0
            continue
        start_close = float(first.iloc[0]["close"])
        end_close = float(last.iloc[-1]["close"])
        result[code] = (end_close / start_close - 1) if start_close > 0 else 0.0
    return result


def run_backtest(
    strategy_class: type,
    df: pd.DataFrame,
    strategy_params: dict[str, Any] | None = None,
    initial_cash: float = 1_000_000,
    commission: float = 0.00009,
    stamp_duty: float = 0.0005,
    slippage: float = 0.01,
    index_df: pd.DataFrame | None = None,
    batch_mode: bool = False,
) -> dict:
    """
    运行回测，返回 {metrics, equity_curve, trades}。

    参数
    ----
    strategy_class : backtrader.Strategy 子类
    df : 需包含 trade_date, open, high, low, close, volume
    strategy_params : 策略参数字典
    initial_cash : 初始资金（默认 100 万）
    commission : 佣金费率（默认万 0.9，买卖双向）
    stamp_duty : 印花税率（默认万 5，仅卖出）
    slippage : 固定滑点（元）
    index_df : 大盘指数 OHLCV（可选，用于动态仓位管理）
    batch_mode : 跳过权益曲线和 CSV 输出，加快批量回测
    """
    cerebro = bt.Cerebro()

    # 数据
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    data = PgDataFeed(dataname=df)
    cerebro.adddata(data)

    # 大盘指数数据（data1），用于 MarketAwareSizer 判断牛熊
    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx["trade_date"] = pd.to_datetime(idx["trade_date"])
        cerebro.adddata(PgDataFeed(dataname=idx))

    # 策略
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)

    # 动态仓位管理：牛市 95%，熊市自动降至 40%
    cerebro.addsizer(MarketAwareSizer, percents=95, bear_percent=40)

    # 资金 & A 股费用（佣金 + 印花税）
    cerebro.broker.setcash(initial_cash)
    comm_info = AShareCommission(commission=commission, stamp_duty=stamp_duty)
    cerebro.broker.addcommissioninfo(comm_info)
    cerebro.broker.set_slippage_fixed(slippage)

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(TradeRecorder, _name="traderecorder")
    cerebro.addanalyzer(SignalRecorder, _name="signalrecorder")

    if batch_mode:
        # 批量模式：跳过 TradeAnalyzer（重）、TimeReturn（重）、CSV 写入
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    else:
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
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
    total_ret = rtn_dict.get("rtot", 0) or 0
    if years > 0 and total_ret > -1:
        metrics["annual_return"] = (1 + total_ret) ** (1 / years) - 1
    else:
        metrics["annual_return"] = 0

    # 从 TradeRecorder 计算胜率
    raw_trades = strat.analyzers.traderecorder.get_analysis() or []
    raw_signals = strat.analyzers.signalrecorder.get_analysis() or []
    metrics["total_trades"] = len(raw_trades)
    if raw_trades:
        won = sum(1 for t in raw_trades if t["pnl"] > 0)
        metrics["won_trades"] = won
        metrics["lost_trades"] = len(raw_trades) - won
        metrics["win_rate"] = won / len(raw_trades)
    else:
        metrics["won_trades"] = 0
        metrics["lost_trades"] = 0
        metrics["win_rate"] = 0.0

    metrics["final_value"] = cerebro.broker.getvalue()

    if batch_mode:
        return {
            "metrics": metrics,
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(raw_trades),
            "signals": raw_signals,
        }

    # ---- 权益曲线 ----
    eq_dict = strat.analyzers.timereturn.get_analysis()
    equity_curve = pd.DataFrame(
        {"date": list(eq_dict.keys()), "return": list(eq_dict.values())}
    )
    if not equity_curve.empty:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        equity_curve["equity"] = initial_cash * (1 + equity_curve["return"]).cumprod()

    # ---- 交易明细 ----
    trades_df = pd.DataFrame(raw_trades)

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "signals": raw_signals,
    }
