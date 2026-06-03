#!/usr/bin/env python
"""RL-Dynamic 回测：PPO学习因子权重 + NDrop选股 + NAV仿真。

用法:
    python scripts/run_rl_backtest.py --start 20200101 --end 20260601 --timesteps 5000
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from factors import ALL_FACTORS
from config.settings import TradingConfig
from rl_dynamic.trainer import walk_forward_train_rl_weights
from rl_dynamic.predictor import RLDynamicPredictor
from rl_dynamic.state_builder import StateBuilder
from rl_dynamic.factor_pool import FactorPool
from portfolio.selector import select_topk_ndrop


def load_data(engine, start_date, end_date, universe_size=500):
    """加载 OHLCV + 指数 + extra_data。"""
    # 只用首年成交额选宇宙（无前视偏差）
    first_year = f"{int(start_date[:4])+1}{start_date[4:]}"
    codes = pd.read_sql(
        f"SELECT code FROM stock_daily "
        f"WHERE trade_date >= '{start_date}' AND trade_date <= '{first_year}' "
        f"GROUP BY code ORDER BY SUM(amount) DESC LIMIT {universe_size}",
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

    # extra_data
    extra = {}
    try:
        edf = pd.read_sql(f"SELECT code,trade_date,market_cap,pb FROM stock_daily_extra "
                          f"WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not edf.empty:
            edf["log_mcap"] = np.log(edf["market_cap"].replace(0, np.nan))
            extra["log_mcap"] = edf[["code", "trade_date", "log_mcap"]]
            extra["pb"] = edf[["code", "trade_date", "pb"]]
    except Exception: pass
    try:
        fcols = ["net_profit","roe","bps","net_margin","revenue","eps","cash_flow",
                 "operating_cash_flow","total_assets","total_liability","goodwill",
                 "holder_equity","adjusted_profit"]
        fdf = pd.read_sql(f"SELECT code,report_date,{','.join(fcols)} FROM stock_financial "
                          f"WHERE code IN ({code_list}) AND report_date>='2018-01-01' ORDER BY code,report_date", engine)
        if not fdf.empty:
            for c in fcols:
                if c in fdf.columns: extra[c] = fdf[["code","report_date",c]].copy()
    except Exception: pass

    logger.info(f"数据: {len(ohlcv)}行, {ohlcv['code'].nunique()}只")
    return ohlcv, index_df, extra


def main():
    parser = argparse.ArgumentParser(description="RL-Dynamic 回测")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260601")
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--ndrop-n", type=int, default=2)
    args = parser.parse_args()

    engine = get_engine()
    ohlcv, index_df, extra = load_data(engine, args.start, args.end, args.universe_size)

    # ── 因子列表（不含基本面，避免太多 NaN）──
    fn = [f for f in ALL_FACTORS.keys() if not f.startswith('fin_') and f in ALL_FACTORS]
    logger.info(f"RL因子: {len(fn)}个")

    # ── Walk-Forward RL 训练 ──
    train_results = walk_forward_train_rl_weights(
        ohlcv, fn, index_df, extra_data=extra, total_timesteps=args.timesteps)
    if not train_results:
        logger.error("RL训练无结果")
        return
    logger.info(f"RL训练: {len(train_results)}窗口")

    # ── 构建因子数据集（用于仿真）──
    pool = FactorPool(fn)
    ds = pool.compute_factors(ohlcv, extra)
    ds["trade_date"] = pd.to_datetime(ds["trade_date"])
    factor_cols = [c for c in fn if c in ds.columns]
    all_dates = sorted(ds["trade_date"].unique())

    # 股价索引（ohlcv 中的 close）
    close_map = ohlcv.pivot_table(index="trade_date", columns="code", values="close", aggfunc="last")

    INIT = TradingConfig.INITIAL_CASH
    COMM = TradingConfig.COMMISSION
    SLIP = TradingConfig.SLIPPAGE
    STAMP = TradingConfig.STAMP_DUTY

    # ── 仿真（仅用最后一个窗口的模型，覆盖其验证期）──
    nav = INIT
    positions = {}
    nav_history = [nav]
    daily_rets = []

    wr = train_results[-1]
    builder = wr["builder"]
    predictor = RLDynamicPredictor(wr["ppo_model"], wr["factor_names"], builder)
    val_dates = [d for d in all_dates if wr["train_end"] < d <= wr["val_end"]]

    for dt in val_dates:
        day = ds[ds["trade_date"] == dt]
        if len(day) < 10:
            continue

        state = builder.build(ohlcv, index_df, {}, dt)
        day_factor = day[["code"] + factor_cols].fillna(0).replace([np.inf, -np.inf], 0)
        preds = predictor.predict(day_factor, market_state=state)
        scores = pd.Series(preds["score"].values, index=preds["code"].values).sort_values(ascending=False)
        new_holdings, to_buy, to_sell = select_topk_ndrop(
            scores, set(positions.keys()), K=args.top_n, N=args.ndrop_n)

        day_ret = 0.0
        cost = 0.0
        if dt in close_map.index:
            for code in new_holdings:
                if code in close_map.columns:
                    p = close_map.loc[dt, code]
                    if not np.isnan(p) and p > 0:
                        if code in positions:
                            day_ret += (p / positions[code] - 1)
                        positions[code] = p
                        cost += p * (COMM + SLIP)

        day_ret = day_ret / max(len(new_holdings), 1) if new_holdings else 0
        cost += sum(close_map.loc[dt, c] * (STAMP + COMM + SLIP)
                    for c in to_sell if dt in close_map.index and c in close_map.columns
                    and not np.isnan(close_map.loc[dt, c]))
        cost_ratio = cost / max(INIT, 1)
        net_ret = day_ret - cost_ratio

        nav *= (1 + net_ret)
        nav_history.append(nav)
        daily_rets.append(net_ret)

    # ── 指标 ──
    nav = np.array(nav_history)
    td = len(nav) - 1
    if td < 10:
        print("数据不足，无法计算指标")
        return

    total_ret = (nav[-1] / nav[0] - 1) * 100
    years = max(td / 252, 0.2)
    cagr = ((nav[-1] / nav[0]) ** (1 / years) - 1) * 100
    dr = np.array(daily_rets)
    sh = float(np.sqrt(252) * np.mean(dr) / np.std(dr)) if np.std(dr) > 0 else 0
    pk = np.maximum.accumulate(nav)
    mdd = float(np.max((pk - nav) / pk) * 100)

    print(f"\n{'='*50}")
    print(f"RL-Dynamic 回测结果 (qfq前复权)")
    print(f"{'='*50}")
    print(f"总收益:    {total_ret:.1f}%")
    print(f"年化收益:  {cagr:.1f}%")
    print(f"Sharpe:    {sh:.2f}")
    print(f"最大回撤:  {mdd:.1f}%")
    print(f"交易天数:  {td}")
    print(f"仿真区间:  {val_start.date()} ~ {val_end.date()}")

    engine.dispose()


if __name__ == "__main__":
    main()
