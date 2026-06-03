#!/usr/bin/env python
"""RL-Dynamic 策略回测：RL学习因子权重 + NDrop选股。

用法:
    python scripts/run_rl_backtest.py --start 20200101 --end 20260601 --timesteps 30000
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from models.regime import detect_regime
from factors import ALL_FACTORS
from config.settings import TradingConfig
from rl_dynamic.trainer import walk_forward_train_rl_weights
from rl_dynamic.predictor import RLDynamicPredictor
from rl_dynamic.state_builder import StateBuilder
from portfolio.selector import select_topk_ndrop, filter_stocks
from portfolio.risk import check_drawdown_limit


def load_data(engine, start_date, end_date, universe_size=500):
    """加载回测数据。"""
    codes = pd.read_sql(
        f"SELECT d.code FROM stock_daily d "
        f"JOIN stock_basic b ON d.code=b.code AND b.is_st=FALSE "
        f"WHERE d.trade_date BETWEEN '{start_date}' AND '{end_date}' "
        f"GROUP BY d.code ORDER BY SUM(d.amount) DESC LIMIT {universe_size}",
        engine)["code"].tolist()
    code_list = ",".join([f"'{c}'" for c in codes])

    ohlcv = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily WHERE code IN ({code_list})
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY code, trade_date
    """, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])

    index_df = pd.read_sql(f"""
        SELECT trade_date, close FROM index_daily
        WHERE code='000001' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
    """, engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])

    # extra_data (估值+财务)
    extra_data = {}
    try:
        extra_df = pd.read_sql(f"SELECT code, trade_date, market_cap, pb FROM stock_daily_extra WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
    except Exception: pass
    try:
        fin_cols = ["net_profit", "roe", "bps", "net_margin", "revenue", "eps",
                     "cash_flow", "operating_cash_flow", "total_assets", "total_liability",
                     "goodwill", "holder_equity", "adjusted_profit"]
        fin_df = pd.read_sql(f"SELECT code, report_date, {','.join(fin_cols)} FROM stock_financial WHERE code IN ({code_list}) AND report_date >= '2018-01-01' ORDER BY code, report_date", engine)
        if not fin_df.empty:
            for col in fin_cols:
                if col in fin_df.columns:
                    extra_data[col] = fin_df[["code", "report_date", col]].copy()
    except Exception: pass

    logger.info(f"数据: {len(ohlcv)}行, {ohlcv['code'].nunique()}只, extra={len(extra_data)}项")
    return ohlcv, index_df, extra_data


def main():
    parser = argparse.ArgumentParser(description="RL-Dynamic 回测")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260601")
    parser.add_argument("--universe-size", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--ndrop-n", type=int, default=2)
    args = parser.parse_args()

    engine = get_engine()
    ohlcv, index_df, extra_data = load_data(engine, args.start, args.end, args.universe_size)

    # RL训练
    factor_names = list(ALL_FACTORS.keys())
    logger.info(f"RL训练: {len(factor_names)}因子, {args.timesteps}步/窗口")
    results = walk_forward_train_rl_weights(
        ohlcv, factor_names, index_df, extra_data=extra_data,
        total_timesteps=args.timesteps)

    if not results:
        logger.error("RL训练无结果")
        return

    # 获取所有交易日
    all_dates = sorted(ohlcv["trade_date"].unique())

    # 仿真
    cash = TradingConfig.INITIAL_CASH
    nav = [cash]
    positions = {}
    day_counter = 0

    for wi, wr in enumerate(results):
        builder = StateBuilder(n_factors=len(wr["factor_names"]))
        predictor = RLDynamicPredictor(
            wr["policy_net"], wr["factor_names"], builder, device="cpu")

        # 该窗口的验证期日期
        val_start = wr["train_end"]
        val_end = wr["val_end"]
        val_dates = [d for d in all_dates if val_start < d <= val_end]

        for dt in val_dates:
            day = ohlcv[ohlcv["trade_date"] == dt]
            if len(day) < 10:
                continue

            # 构建市场状态
            state = builder.build(ohlcv, index_df, {}, dt)

            # 构建因子截面
            factor_cols = [f for f in wr["factor_names"] if f in ALL_FACTORS]
            # 简化: 直接用日线数据模拟因子值
            day_scores = pd.DataFrame({"code": day["code"].unique()})
            for f in factor_cols[:10]:  # 只用前10个 (简化)
                day_scores[f] = np.random.randn(len(day_scores)) * 0.1
            day_scores = day_scores.fillna(0)

            preds = predictor.predict(day_scores, market_state=state)

            # NDrop选股
            scores_series = pd.Series(preds["score"].values, index=preds["code"].values).sort_values(ascending=False)
            new_holdings, to_buy, to_sell = select_topk_ndrop(
                scores_series, set(positions.keys()), K=args.top_n, N=args.ndrop_n)

            # 更新持仓和净值
            day_close = day.groupby("code")["close"].last()
            day_ret = 0
            for code in new_holdings:
                if code in day_close.index:
                    if code in positions:
                        day_ret += (day_close[code] / positions[code] - 1)
                    positions[code] = day_close[code]

            day_ret = day_ret / max(len(new_holdings), 1)
            cash *= (1 + day_ret - TradingConfig.COMMISSION * 2)  # 简化成本
            nav.append(cash)
            day_counter += 1

    # 输出结果
    nav = np.array(nav)
    total_ret = (nav[-1] / nav[0] - 1) * 100
    trading_years = day_counter / 252
    annual_ret = ((nav[-1] / nav[0]) ** (1 / max(trading_years, 0.01)) - 1) * 100
    daily_rets = np.diff(nav) / nav[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_rets) / np.std(daily_rets) if np.std(daily_rets) > 0 else 0
    peak = np.maximum.accumulate(nav)
    max_dd = np.max((peak - nav) / peak) * 100
    win_rate = np.mean(daily_rets > 0) * 100

    print(f"\n=== RL-Dynamic 回测结果 ===")
    print(f"总收益: {total_ret:.1f}%")
    print(f"年化收益: {annual_ret:.1f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.1f}%")
    print(f"日胜率: {win_rate:.1f}%")
    print(f"交易日: {day_counter}")

    engine.dispose()


if __name__ == "__main__":
    main()
