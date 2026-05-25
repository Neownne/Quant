#!/usr/bin/env python
"""ML 选股每日模拟回测：完整交易管线验证。

链路: DailyPredictor → select_top_n → equal_weight → apply_stop_loss → P&L

用法:
    python scripts/run_simulation.py                      # 默认参数
    python scripts/run_simulation.py --top-n 10           # 持仓 10 只
    python scripts/run_simulation.py --no-orthogonal      # 跳过正交筛选
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_ensemble, walk_forward_train_by_regime
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from models.regime import detect_regime
from portfolio.selector import select_top_n
from portfolio.allocator import equal_weight
from portfolio.risk import apply_stop_loss, check_drawdown_limit


def run_window_simulation(ensemble, factor_df, price_map, top_n, init_cash=1_000_000):
    """在单个验证窗口上运行每日模拟交易。

    参数
    ----
    ensemble : EnsemblePredictor
    factor_df : DataFrame，含 code, trade_date 和所有因子列，按 trade_date 排序
    price_map : DataFrame，含 code, trade_date, close
    top_n : 每日持仓数
    init_cash : 初始资金

    返回
    ----
    dict: {daily_returns, total_return, sharpe, max_drawdown, win_rate, n_days}
    """
    dates = sorted(factor_df["trade_date"].unique())
    if len(dates) < 2:
        return None

    cash = init_cash
    peak_value = init_cash
    positions = {}  # code -> {shares, cost_basis}
    daily_values = [init_cash]
    daily_returns = []

    for i, today in enumerate(dates[:-1]):
        tomorrow = dates[i + 1]

        # 收盘前：获取今日因子截面
        today_factors = factor_df[factor_df["trade_date"] == today].copy()
        if len(today_factors) < top_n:
            daily_values.append(cash + sum(
                positions[c]["shares"] * _get_price(price_map, c, today)
                for c in positions
            ))
            daily_returns.append(0)
            continue

        # 预测得分
        try:
            scores = ensemble.predict(today_factors)
        except Exception:
            scores = today_factors[["code"]].copy()
            scores["score"] = 0.5
            scores["rank"] = range(1, len(scores) + 1)

        # 选股 + 分配
        selected = select_top_n(scores, n=top_n)
        target_codes = selected["code"].tolist()

        # 今日收盘价
        today_prices = {}
        for code in target_codes:
            p = _get_price(price_map, code, today)
            if p and p > 0:
                today_prices[code] = p

        if not today_prices:
            daily_values.append(cash + sum(
                positions[c]["shares"] * _get_price(price_map, c, today)
                for c in positions
            ))
            daily_returns.append(0)
            continue

        # 卖出所有现有持仓（按今日收盘价）
        for code, pos in list(positions.items()):
            p = _get_price(price_map, code, today)
            if p and p > 0:
                cash += pos["shares"] * p
        positions.clear()

        # 等权买入目标股票
        alloc = equal_weight(list(today_prices.keys()), cash)
        cash_after = 0  # 全仓买入
        for _, row in alloc.iterrows():
            code = row["code"]
            price = today_prices[code]
            shares = int(row["value"] / price / 100) * 100  # A 股整手
            cost = shares * price
            if shares > 0 and cost <= cash:
                cash -= cost
                positions[code] = {"shares": shares, "cost_basis": price}

        # 次日：计算持仓收益
        tomorrow_value = cash
        for code, pos in list(positions.items()):
            p_tom = _get_price(price_map, code, tomorrow)
            if p_tom and p_tom > 0:
                # 止损检查
                loss = (p_tom - pos["cost_basis"]) / pos["cost_basis"]
                if loss <= -0.08:
                    cash += pos["shares"] * p_tom
                    del positions[code]
                    continue
                tomorrow_value += pos["shares"] * p_tom
            else:
                # 停牌，按昨日价估值
                p_prev = _get_price(price_map, code, today)
                if p_prev and p_prev > 0:
                    tomorrow_value += pos["shares"] * p_prev

        daily_values.append(tomorrow_value)
        ret = (tomorrow_value / daily_values[-2]) - 1 if daily_values[-2] > 0 else 0
        daily_returns.append(ret)

        # 回撤检查（使用更新前的 peak）
        if check_drawdown_limit(tomorrow_value, peak_value, limit=0.25):
            logger.warning(f"  触发回撤上限 {today.date()}，清仓")
            for code, pos in positions.items():
                p = _get_price(price_map, code, tomorrow)
                if p and p > 0:
                    cash += pos["shares"] * p
            positions.clear()
            tomorrow_value = cash
        peak_value = max(peak_value, tomorrow_value)

    # 汇总指标
    rets = pd.Series(daily_returns)
    total_return = (daily_values[-1] / init_cash) - 1
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak_idx = np.argmax(daily_values)
    max_dd = float(min(
        (daily_values[i] - daily_values[peak_idx]) / daily_values[peak_idx]
        for i in range(len(daily_values))
    )) if daily_values else 0
    win_rate = float((rets > 0).mean()) if len(rets) > 0 else 0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_days": len(rets),
        "final_value": daily_values[-1],
    }


def _get_price(price_map, code, trade_date):
    """获取某只股票某日的收盘价。"""
    subset = price_map[(price_map["code"] == code) & (price_map["trade_date"] == trade_date)]
    if subset.empty:
        return None
    return float(subset["close"].iloc[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default="all")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260101")
    parser.add_argument("--codes", default="")
    parser.add_argument("--no-orthogonal", action="store_true")
    parser.add_argument("--no-ic-gate", action="store_true")
    parser.add_argument("--regime", action="store_true", help="启用分市场状态训练")
    parser.add_argument("--optuna", action="store_true", help="启用 Optuna 超参优化")
    parser.add_argument("--init-cash", type=float, default=1_000_000)
    args = parser.parse_args()

    if args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = [f.strip() for f in args.factors.split(",")]

    logger.info(f"{len(factor_names)} 个因子, top-{args.top_n}, 初始资金 {args.init_cash:,.0f}")

    engine = get_engine()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        codes = pd.read_sql("SELECT code FROM stock_basic LIMIT 200", engine)["code"].tolist()

    code_list = ",".join([f"'{c}'" for c in codes])
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 加载指数数据（regime 模式需要）
    regime_df = None
    if args.regime:
        try:
            idx_sql = f"""
                SELECT trade_date, close FROM index_daily
                WHERE code = '000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}'
                ORDER BY trade_date
            """
            index_df = pd.read_sql(idx_sql, engine)
            if not index_df.empty:
                regime_df = detect_regime(index_df)
                logger.info(f"指数数据: {len(index_df)} 行")
        except Exception as e:
            logger.warning(f"指数数据加载失败: {e}, 回退到非 regime 模式")
            args.regime = False

    # extra_data
    extra_data = {}
    try:
        extra_sql = f"""
            SELECT code, trade_date, market_cap, pb
            FROM stock_daily_extra
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        """
        extra_df = pd.read_sql(extra_sql, engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
    except Exception as e:
        logger.warning(f"估值数据: {e}")

    try:
        sh_sql = f"""
            SELECT code, end_date AS trade_date, shareholder_count
            FROM stock_shareholder
            WHERE code IN ({code_list})
              AND end_date BETWEEN '{args.start}' AND '{args.end}'
        """
        sh_df = pd.read_sql(sh_sql, engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
    except Exception as e:
        logger.warning(f"股东数据: {e}")

    engine.dispose()

    # 因子数据集（含 ret_1d 供 IC 计算）
    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary",
        extra_data=extra_data if extra_data else None,
    )

    # 1. IC 门禁
    if not args.no_ic_gate:
        factor_names = filter_factors_by_ic(dataset, factor_names, ret_col="ret_1d")

    # 2. 正交筛选
    if not args.no_orthogonal:
        factor_cols = select_orthogonal_factors(dataset, factor_names, threshold=0.7)
    else:
        factor_cols = factor_names

    logger.info(f"筛选后: {len(factor_cols)} 因子")

    # 3. Walk-forward 训练
    if args.regime and regime_df is not None:
        logger.info("使用分市场状态集成训练")
        results = walk_forward_train_by_regime(
            dataset, factor_cols, regime_df,
            train_years=args.train_years, val_years=args.val_years,
        )
    else:
        results = walk_forward_train_ensemble(
            dataset, factor_cols,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna,
        )
    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 价格映射表（用于模拟交易）
    price_map = ohlcv[["code", "trade_date", "close"]].copy()
    price_map["trade_date"] = pd.to_datetime(price_map["trade_date"])

    # 统一日期类型：dataset 中 trade_date 可能为 date 对象，转为 Timestamp
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])

    # 每日模拟
    print("\n=== 每日模拟回测结果 ===\n")
    all_sim = []
    for i, r in enumerate(results):
        logger.info(f"模拟窗口 {i+1}: {r['val_start'].date()}~{r['val_end'].date()}")

        val_mask = (
            (dataset["trade_date"] >= r["val_start"]) &
            (dataset["trade_date"] <= r["val_end"])
        )
        val_factors = dataset[val_mask].copy()

        sim = run_window_simulation(
            r["ensemble"], val_factors, price_map,
            top_n=args.top_n, init_cash=args.init_cash,
        )
        if sim is None:
            continue

        all_sim.append(sim)
        print(
            f"窗口 {i+1} ({r['val_start'].date()}~{r['val_end'].date()}): "
            f"收益={sim['total_return']:.1%}, 夏普={sim['sharpe']:.2f}, "
            f"回撤={sim['max_drawdown']:.1%}, 胜率={sim['win_rate']:.1%}, "
            f"交易日={sim['n_days']}"
        )

    if all_sim:
        summary = pd.DataFrame(all_sim)
        print(f"\n--- 汇总 ---")
        print(f"平均年化收益: {summary['total_return'].mean():.1%}")
        print(f"平均夏普比率: {summary['sharpe'].mean():.2f}")
        print(f"平均最大回撤: {summary['max_drawdown'].mean():.1%}")
        print(f"平均日胜率:   {summary['win_rate'].mean():.1%}")
        print(f"累计终值(均): {summary['final_value'].mean():,.0f}")


if __name__ == "__main__":
    main()
