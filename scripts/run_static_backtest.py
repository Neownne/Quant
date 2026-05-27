#!/usr/bin/env python
"""静态策略回测（替代已删除的 app/utils/backtest_runner.py）。

用法:
    python scripts/run_static_backtest.py --strategy "双均线交叉" --codes 000001,600519
    python scripts/run_static_backtest.py --strategy "震荡网格(高抛低吸)" --top-n 30
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine
from config.settings import TradingConfig
from scripts.overfit_check import OverfitChecker


class AShareCommInfo(bt.CommInfoBase):
    """A股真实交易成本：引用 TradingConfig"""
    params = (
        ('commission', TradingConfig.COMMISSION),
        ('stamp_duty', TradingConfig.STAMP_DUTY),
        ('slip_perc', TradingConfig.SLIPPAGE),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        comm = value * (self.p.commission + self.p.slip_perc)
        if size < 0:  # 卖出：额外加印花税
            comm += value * self.p.stamp_duty
        return comm


class EquityRecorder(bt.Analyzer):
    """记录每日组合净值。"""
    def __init__(self):
        self.equity = []
    def next(self):
        self.equity.append((
            self.datas[0].datetime.date(0).isoformat(),
            round(self.strategy.broker.getvalue(), 2),
        ))
    def get_analysis(self):
        return self.equity


def load_ohlcv(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY trade_date
    """
    df = pd.read_sql(sql, engine)
    engine.dispose()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return {c: g.drop(columns=["code"]) for c, g in df.groupby("code")}


def run_single_backtest(strategy_cls, df: pd.DataFrame, **params) -> dict:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, **params)

    data = bt.feeds.PandasData(
        dataname=df.set_index("trade_date"),
        open="open", high="high", low="low", close="close", volume="volume",
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(TradingConfig.INITIAL_CASH)
    # A股真实成本：佣金万0.9（买卖双向）+ 印花税万5（卖出单向）+ 滑点0.1%
    cerebro.broker.addcommissioninfo(AShareCommInfo())

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
        timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
    cerebro.addanalyzer(EquityRecorder, _name='equity_recorder')

    results = cerebro.run()
    strat = results[0]

    # Extract from analyzers
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get('sharperatio', None)
    # SharpeRatio may be None (no trades) or list (monthly)
    if sharpe is None:
        sharpe = 0.0
    elif isinstance(shpe := sharpe, list):
        sharpe = float(np.mean(shpe)) if shpe else 0.0
    else:
        sharpe = float(sharpe)

    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get('drawdown', 0.0)
    if isinstance(max_dd, (int, float)) and max_dd < 0:
        max_dd = abs(max_dd) / 100.0

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_closed = trade_analysis.get('total', {}).get('total', 0)
    won_total = trade_analysis.get('won', {}).get('total', 0)
    lost_total = trade_analysis.get('lost', {}).get('total', 0)
    pnl_net = trade_analysis.get('pnl', {}).get('net', {}).get('total', 0.0)

    # Equity curve from recorder
    equity = strat.analyzers.equity_recorder.get_analysis()

    return {
        "equity": equity,
        "sharpe": sharpe,
        "max_drawdown": max_dd if isinstance(max_dd, float) else 0.0,
        "n_trades": int(total_closed),
        "n_wins": int(won_total),
        "n_losses": int(lost_total),
        "pnl_net": float(pnl_net) if pnl_net else 0.0,
        "n_days": len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, help="策略显示名称")
    parser.add_argument("--codes", default="", help="逗号分隔股票代码，不指定则取市值前N")
    parser.add_argument("--top-n", type=int, default=30, help="按成交额取前N只")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260501")
    args = parser.parse_args()

    from strategies import get_all_strategies
    all_s = get_all_strategies()
    if args.strategy not in all_s:
        print(f"未知策略: {args.strategy}, 可选: {list(all_s.keys())}")
        sys.exit(1)
    strategy_cls = all_s[args.strategy]

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        engine = get_engine()
        codes = pd.read_sql(
            f"SELECT code FROM stock_daily WHERE trade_date >= '{args.start}' AND trade_date <= '{args.end}' GROUP BY code ORDER BY SUM(amount) DESC LIMIT {args.top_n}",
            engine,
        )["code"].tolist()
        engine.dispose()

    print(f"策略: {args.strategy}, 标的: {len(codes)} 只, 区间: {args.start}~{args.end}")

    ohlcv_map = load_ohlcv(codes, args.start, args.end)
    print(f"数据: {sum(len(v) for v in ohlcv_map.values())} 行")

    # Run per-stock backtests
    stock_results = []
    all_equity_dfs = []
    for code, df in ohlcv_map.items():
        if len(df) < 100:
            continue
        try:
            r = run_single_backtest(strategy_cls, df)
            r["code"] = code
            stock_results.append(r)
            # Collect equity curve
            if r["equity"]:
                eq_df = pd.DataFrame(r["equity"], columns=["date", code]).set_index("date")
                eq_df[code] = eq_df[code] / eq_df[code].iloc[0]  # Normalize to 1.0
                all_equity_dfs.append(eq_df)
        except Exception as e:
            print(f"  {code} 回测失败: {e}")

    if not stock_results:
        print("无有效回测结果")
        return

    # Aggregate per-stock totals
    df_r = pd.DataFrame(stock_results)
    win_rate = (df_r["pnl_net"] > 0).mean() if "pnl_net" in df_r else float((df_r["n_wins"].sum() / max(df_r["n_trades"].sum(), 1)))
    total_trades = int(df_r["n_trades"].sum())
    total_wins = int(df_r["n_wins"].sum())
    total_losses = int(df_r["n_losses"].sum())

    # Compute from aggregate equity curve (more accurate than averaging per-stock)
    total_return = 0.0
    annual_return = 0.0
    computed_sharpe = 0.0
    computed_mdd = 0.0

    # Build equal-weighted aggregate equity curve
    equity_curve = {}
    daily_returns = {}
    if all_equity_dfs:
        combined = all_equity_dfs[0]
        for eq_df in all_equity_dfs[1:]:
            combined = combined.join(eq_df, how="outer")
        combined = combined.ffill().fillna(1.0)
        eq_nav = combined.mean(axis=1)
        eq_nav = eq_nav.sort_index()

        # Total & annual return from aggregate curve
        total_return = float(eq_nav.iloc[-1] / eq_nav.iloc[0] - 1)
        n_days = len(eq_nav)
        years = max(n_days / 252, 0.2)
        annual_return = float((1 + total_return) ** (1 / years) - 1)

        # Build equity curve dict
        for dt, v in eq_nav.items():
            equity_curve[str(dt)[:10]] = round(float(v), 6)

        # Daily returns
        eq_vals = eq_nav.values
        prev_dates = eq_nav.index.tolist()
        daily_ret_list = []
        for i in range(1, len(prev_dates)):
            prev = eq_vals[i - 1]
            curr = eq_vals[i]
            ret = float(curr / prev - 1) if prev > 0 else 0.0
            daily_returns[str(prev_dates[i])[:10]] = round(ret, 6)
            daily_ret_list.append(ret)

        # Compute Sharpe from aggregate daily returns
        if daily_ret_list and np.std(daily_ret_list) > 0:
            computed_sharpe = float(np.mean(daily_ret_list) / np.std(daily_ret_list) * np.sqrt(252))

        # Compute Max Drawdown from aggregate equity curve
        peak = eq_vals[0]
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > computed_mdd:
                computed_mdd = dd
    else:
        annual_return = 0.0

    # Strategy parameters
    try:
        strategy_params = {}
        for k in dir(strategy_cls.params):
            if k.startswith('_'):
                continue
            v = getattr(strategy_cls.params, k)
            if callable(v):
                continue
            strategy_params[k] = v
    except Exception:
        strategy_params = {}

    n_params = len(strategy_params)

    print(f"\n=== {args.strategy} 回测汇总 ===")
    print(f"股票数: {len(stock_results)}, 总交易: {total_trades}")
    print(f"年化收益: {annual_return:.2%}, Sharpe: {computed_sharpe:.2f}, 胜率: {win_rate:.1%}")
    print(f"最大回撤: {computed_mdd:.2%}")

    # Overfit check
    checker = OverfitChecker(min_trades=10, min_regimes=1)
    result = checker.check(
        {
            "train_sharpe": 0,
            "val_sharpe": 0,
            "test_sharpe": computed_sharpe,
            "n_trades": total_trades,
            "n_params": n_params,
            "start_date": args.start,
            "end_date": args.end,
            "win_rate": float(win_rate),
        },
        regime_count=1,
        sensitivity_stable=True,
    )

    # Find version_id
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT sv.id FROM strategy_versions sv JOIN strategy_configs sc ON sv.strategy_id=sc.id WHERE sc.name=:n AND sv.version='1.10'"),
            {"n": args.strategy},
        ).fetchone()
        version_id = row[0] if row else 0

        metrics_full = {
            "win_rate": float(win_rate),
            "annual_return": float(annual_return),
            "total_return": float(total_return),
            "sharpe": float(computed_sharpe),
            "max_drawdown": float(computed_mdd),
            "n_trades": int(total_trades),
            "n_wins": int(total_wins),
            "n_losses": int(total_losses),
            "n_params": n_params,
            "n_stocks": len(stock_results),
            "n_days": len(equity_curve),
            "adjusted_sharpe": result["adjusted_sharpe"],
            "start_date": args.start,
            "end_date": args.end,
            "strategy_params": strategy_params,
        }

        with engine.begin() as conn2:
            conn2.execute(text("""
                INSERT INTO backtest_results
                    (version_id, start_date, end_date, quality, quality_flags,
                     metrics_json, equity_curve_json, daily_returns_json)
                VALUES (:vid, :start, :end, :quality, :flags, :metrics, :equity, :returns)
            """), {
                "vid": version_id,
                "start": args.start,
                "end": args.end,
                "quality": result["quality"],
                "flags": result["flags"],
                "metrics": json.dumps(metrics_full, default=str),
                "equity": json.dumps(equity_curve, default=str),
                "returns": json.dumps(daily_returns, default=str),
            })

    engine.dispose()
    print(f"\n结果: quality={result['quality']}, flags={result['flags']}")
    print(f"已写入 backtest_results (version_id={version_id})")


if __name__ == "__main__":
    main()
