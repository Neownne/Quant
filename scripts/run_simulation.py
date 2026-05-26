#!/usr/bin/env python
"""ML 选股每日模拟回测：完整交易管线验证。

链路: 基本面排雷 → 量价因子ML排序 → NDrop增量调仓 → 等权/上限 → 止损/回撤

用法:
    python scripts/run_simulation.py                      # 默认参数
    python scripts/run_simulation.py --top-n 10           # 持仓 10 只
    python scripts/run_simulation.py --ndrop              # 启用 NDrop 增量调仓
    python scripts/run_simulation.py --no-quality         # 跳过基本面筛选
    python scripts/run_simulation.py --save-results out.json  # 保存信号
"""
import argparse
import json
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
from models.dual_period import DualPeriodModel
from portfolio.selector import select_top_n, select_topk_ndrop
from portfolio.allocator import equal_weight
from portfolio.risk import apply_stop_loss, check_drawdown_limit, check_index_crash


def _get_price(price_map, code, trade_date):
    subset = price_map[(price_map["code"] == code) & (price_map["trade_date"] == trade_date)]
    if subset.empty:
        return None
    return float(subset["close"].iloc[0])


def _is_month_first(dates, today):
    """判断 today 是否当月首个交易日。"""
    month = today.strftime("%Y-%m")
    month_dates = [d for d in dates if d.strftime("%Y-%m") == month]
    if not month_dates:
        return False
    return today == min(month_dates)


def run_window_simulation(
    ensemble, factor_df, price_map, top_n, ndrop_n, index_ohlcv,
    quality_model, ohlcv_data, extra_data, init_cash=1_000_000,
    use_ndrop=False, use_quality=True,
):
    """在单个验证窗口上运行每日模拟交易。"""
    dates = sorted(factor_df["trade_date"].unique())
    if len(dates) < 2:
        return None, []

    cash = init_cash
    peak_value = init_cash
    positions = {}
    daily_values = [init_cash]
    daily_returns = []
    daily_signals = []

    for i, today in enumerate(dates[:-1]):
        tomorrow = dates[i + 1]
        today_ts = pd.Timestamp(today)

        # ---- 指数大跌 → 空仓 ----
        daily_trades = []
        if index_ohlcv is not None:
            idx_hist = index_ohlcv[
                pd.to_datetime(index_ohlcv["trade_date"]) <= today_ts
            ]
            if not idx_hist.empty:
                idx_close = idx_hist.sort_values("trade_date")["close"].values
                if check_index_crash(idx_close):
                    logger.warning(f"  {today.date()} 指数大跌，空仓")
                    for code, pos in positions.items():
                        p = _get_price(price_map, code, today)
                        if p and p > 0:
                            cash += pos["shares"] * p
                            daily_trades.append({
                                "code": code, "action": "SELL",
                                "price": p, "shares": pos["shares"],
                                "amount": pos["shares"] * p,
                                "reason": "index_crash",
                            })
                    positions.clear()
                    daily_signals.append({
                        "date": str(today.date()), "crash": True,
                        "trades": daily_trades, "value": cash,
                    })
                    daily_values.append(cash)
                    daily_returns.append(0)
                    continue

        # ---- 候选池 ----
        today_factors = factor_df[factor_df["trade_date"] == today].copy()
        if len(today_factors) < top_n:
            _record_skip(daily_values, daily_returns, daily_signals, today, cash, positions, price_map)
            continue

        # 基本面质量筛选（月频）
        is_month_first = _is_month_first(dates, today_ts)
        if use_quality and quality_model is not None:
            quality_codes = quality_model.quality_screen(
                ohlcv_data, extra_data or {}, today_ts,
            )
            if quality_codes:
                today_factors = today_factors[today_factors["code"].isin(quality_codes)]

        if len(today_factors) < top_n:
            _record_skip(daily_values, daily_returns, daily_signals, today, cash, positions, price_map)
            continue

        # ML 预测
        try:
            scores = ensemble.predict(today_factors)
        except Exception:
            scores = today_factors[["code"]].copy()
            scores["score"] = 0.5
            scores["rank"] = range(1, len(scores) + 1)

        # 选股
        if use_ndrop:
            holding_codes = set(positions.keys())
            score_series = pd.Series(
                scores.set_index("code")["score"].to_dict()
            ).sort_values(ascending=False)
            new_holdings, to_buy, to_sell = select_topk_ndrop(
                score_series, current_holdings=holding_codes,
                K=top_n, N=ndrop_n,
            )
            target_codes = list(new_holdings)
        else:
            selected = select_top_n(scores, n=top_n)
            target_codes = selected["code"].tolist()

        # 当日价格
        today_prices = {}
        for code in target_codes:
            p = _get_price(price_map, code, today)
            if p and p > 0:
                today_prices[code] = p

        if not today_prices:
            _record_skip(daily_values, daily_returns, daily_signals, today, cash, positions, price_map)
            continue

        # 卖出非目标持仓
        for code in list(positions.keys()):
            if code not in target_codes:
                p = _get_price(price_map, code, today)
                if p and p > 0:
                    pos_shares = positions[code]["shares"]
                    cash += pos_shares * p
                    daily_trades.append({
                        "code": code, "action": "SELL",
                        "price": p, "shares": pos_shares,
                        "amount": pos_shares * p,
                        "reason": "rebalance",
                    })
                    del positions[code]

        # 等权买入目标
        n_buy = len([c for c in target_codes if c not in positions])
        alloc_cash = cash / max(n_buy, 1)
        for code in target_codes:
            if code in positions:
                continue
            price = today_prices.get(code, 0)
            if price > 0:
                shares = int(alloc_cash / price / 100) * 100
                cost = shares * price
                if shares > 0 and cost <= cash:
                    cash -= cost
                    positions[code] = {"shares": shares, "cost_basis": price}
                    daily_trades.append({
                        "code": code, "action": "BUY",
                        "price": price, "shares": shares,
                        "amount": cost,
                        "reason": "signal",
                    })

        # 次日估值
        tomorrow_value = cash
        tomorrow_prices = {}
        for code in list(positions.keys()):
            p_tom = _get_price(price_map, code, tomorrow)
            if p_tom and p_tom > 0:
                tomorrow_prices[code] = p_tom
                tomorrow_value += positions[code]["shares"] * p_tom
            else:
                p_prev = _get_price(price_map, code, today)
                if p_prev and p_prev > 0:
                    tomorrow_value += positions[code]["shares"] * p_prev

        # 个股止损
        pos_list = [
            {"code": c, "shares": p["shares"], "cost_basis": p["cost_basis"]}
            for c, p in positions.items()
        ]
        pos_df = pd.DataFrame(pos_list) if pos_list else pd.DataFrame(columns=["code", "shares", "cost_basis"])
        if not pos_df.empty and tomorrow_prices:
            cost_basis_map = {c: p["cost_basis"] for c, p in positions.items()}
            stopped = apply_stop_loss(pos_df, tomorrow_prices, cost_basis_map, stop_pct=0.08)
            for code in stopped["code"].tolist():
                exit_price = tomorrow_prices[code]
                pos_shares = positions[code]["shares"]
                cash += pos_shares * exit_price
                tomorrow_value -= pos_shares * exit_price
                daily_trades.append({
                    "code": code, "action": "SELL",
                    "price": exit_price, "shares": pos_shares,
                    "amount": pos_shares * exit_price,
                    "reason": "stop_loss",
                })
                del positions[code]

        daily_values.append(tomorrow_value)
        ret = (tomorrow_value / daily_values[-2]) - 1 if daily_values[-2] > 0 else 0
        daily_returns.append(ret)

        # 回撤检查
        if check_drawdown_limit(tomorrow_value, peak_value, limit=0.25):
            logger.warning(f"  触发回撤上限 {today.date()}，清仓")
            for code in list(positions.keys()):
                p = _get_price(price_map, code, tomorrow)
                if p and p > 0:
                    pos_shares = positions[code]["shares"]
                    cash += pos_shares * p
                    daily_trades.append({
                        "code": code, "action": "SELL",
                        "price": p, "shares": pos_shares,
                        "amount": pos_shares * p,
                        "reason": "drawdown_clear",
                    })
            positions.clear()
            tomorrow_value = cash
            peak_value = tomorrow_value  # 重置峰值

        # 记录信号
        daily_signals.append({
            "date": str(today.date()),
            "crash": False,
            "n_candidates": len(today_factors),
            "n_holdings": len(positions),
            "value": tomorrow_value,
            "trades": daily_trades,
        })

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
        "total_return": total_return, "sharpe": sharpe,
        "max_drawdown": max_dd, "win_rate": win_rate,
        "n_days": len(rets), "final_value": daily_values[-1],
    }, daily_signals


def _record_skip(daily_values, daily_returns, daily_signals, today, cash, positions, price_map):
    val = cash + sum(
        positions[c]["shares"] * (_get_price(price_map, c, today) or 0)
        for c in positions
    )
    daily_values.append(val)
    daily_returns.append(0)
    daily_signals.append({"date": str(today.date()), "crash": False, "n_holdings": len(positions), "value": val, "trades": []})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default="all")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--ndrop", action="store_true", help="启用 NDrop 增量调仓")
    parser.add_argument("--ndrop-n", type=int, default=2)
    parser.add_argument("--no-quality", action="store_true", help="跳过基本面排雷")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260101")
    parser.add_argument("--codes", default="")
    parser.add_argument("--no-orthogonal", action="store_true")
    parser.add_argument("--no-ic-gate", action="store_true")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--init-cash", type=float, default=1_000_000)
    parser.add_argument("--save-results", default="", help="保存每日信号 JSON")
    args = parser.parse_args()

    if args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = [f.strip() for f in args.factors.split(",")]

    # 分离量价因子和基本面因子
    FUND_NAMES = {
        "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv", "fin_net_margin",
        "fin_bps_growth", "fin_revenue_stability", "fin_eps_growth",
        "fin_debt_ratio", "fin_goodwill_ratio", "fin_pledge_risk", "fin_audit_score",
    }
    pv_factor_names = [f for f in factor_names if f not in FUND_NAMES]
    fund_factor_names = [f for f in factor_names if f in FUND_NAMES]

    logger.info(f"{len(pv_factor_names)} 量价因子 + {len(fund_factor_names)} 基本面因子, top-{args.top_n}")

    engine = get_engine()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()

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

    # 指数数据（大跌过滤器 + regime）
    index_ohlcv = None
    regime_df = None
    try:
        idx_sql = f"""
            SELECT trade_date, close FROM index_daily
            WHERE code = '000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            ORDER BY trade_date
        """
        index_ohlcv = pd.read_sql(idx_sql, engine)
        if not index_ohlcv.empty:
            logger.info(f"指数数据: {len(index_ohlcv)} 行")
    except Exception as e:
        logger.warning(f"指数数据加载失败: {e}")

    if args.regime and index_ohlcv is not None:
        regime_df = detect_regime(index_ohlcv)

    # extra_data（估值 + 财务 + 质押）
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

    # 财务数据
    if not args.no_quality and fund_factor_names:
        try:
            fin_sql = f"""
                SELECT code, report_date, net_profit, cash_flow, roe, bps,
                       net_margin, revenue, eps, total_assets, total_liability,
                       goodwill, holder_equity, operating_cash_flow, adjusted_profit
                FROM stock_financial
                WHERE code IN ({code_list})
                ORDER BY code, report_date
            """
            fin_df = pd.read_sql(fin_sql, engine)
            if not fin_df.empty:
                fin_cols = [c for c in fin_df.columns if c not in ("code", "report_date")]
                for col in fin_cols:
                    extra_data[col] = fin_df[["code", "report_date", col]]
                logger.info(f"财务数据: {len(fin_df)} 行, {len(fin_cols)} 列")
        except Exception as e:
            logger.warning(f"财务数据: {e}")

    # 质押数据
    if not args.no_quality:
        try:
            pledge_sql = f"""
                SELECT code, trade_date, pledge_ratio
                FROM stock_pledge
                WHERE code IN ({code_list})
                  AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            """
            pledge_df = pd.read_sql(pledge_sql, engine)
            if not pledge_df.empty:
                extra_data["pledge_ratio"] = pledge_df[["code", "trade_date", "pledge_ratio"]]
                logger.info(f"质押数据: {len(pledge_df)} 行")
        except Exception as e:
            logger.warning(f"质押数据: {e}")

    engine.dispose()

    # 因子数据集
    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary",
        extra_data=extra_data if extra_data else None,
    )

    # IC 门禁 + 正交筛选（仅量价因子）
    pv_active = [f for f in pv_factor_names if f in dataset.columns]
    if not args.no_ic_gate:
        pv_active = filter_factors_by_ic(dataset, pv_active, ret_col="ret_1d")
    if not args.no_orthogonal:
        pv_active = select_orthogonal_factors(dataset, pv_active, threshold=0.7)

    # 基本面因子仅用于月频质量筛选，不参与 ML 训练（日频覆盖率太低）
    fund_active = [f for f in fund_factor_names if f in dataset.columns and dataset[f].notna().mean() > 0.10]
    if fund_active:
        logger.info(f"基本面因子有效（仅质量筛选）: {fund_active}")
    else:
        logger.warning("基本面因子全部无数据，跳过质量筛选")

    # ML 训练仅用量价因子
    all_active = pv_active
    logger.info(f"筛选后: {len(pv_active)} 量价因子 (ML训练) + {len(fund_active)} 基本面因子 (质量筛选)")

    # Walk-forward 训练
    if args.regime and regime_df is not None:
        results = walk_forward_train_by_regime(
            dataset, all_active, regime_df,
            train_years=args.train_years, val_years=args.val_years,
        )
    else:
        results = walk_forward_train_ensemble(
            dataset, all_active,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna,
        )
    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 初始化双周期模型（用于月频质量筛选）
    quality_model = None
    if not args.no_quality and fund_active:
        from models.trainer import EnsemblePredictor
        dummy_pred = EnsemblePredictor(None, None, pv_active) if results else None
        if results:
            quality_model = DualPeriodModel(
                predictor=results[0]["ensemble"],
                pv_factor_names=pv_active,
                fund_factor_names=fund_active,
                top_n=args.top_n,
                ndrop_n=args.ndrop_n,
            )
            logger.info("双周期模型: 月频基本面排雷 + 日频量价ML")

    # 每日模拟
    price_map = ohlcv[["code", "trade_date", "close"]].copy()
    price_map["trade_date"] = pd.to_datetime(price_map["trade_date"])
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])

    print("\n=== 每日模拟回测结果 ===\n")
    all_sim = []
    all_signals = []

    for i, r in enumerate(results):
        logger.info(f"模拟窗口 {i+1}: {r['val_start'].date()}~{r['val_end'].date()}")

        val_mask = (
            (dataset["trade_date"] >= r["val_start"]) &
            (dataset["trade_date"] <= r["val_end"])
        )
        val_factors = dataset[val_mask].copy()

        sim, signals = run_window_simulation(
            r["ensemble"], val_factors, price_map,
            top_n=args.top_n, ndrop_n=args.ndrop_n,
            index_ohlcv=index_ohlcv,
            quality_model=quality_model,
            ohlcv_data=ohlcv, extra_data=extra_data,
            init_cash=args.init_cash,
            use_ndrop=args.ndrop,
            use_quality=not args.no_quality,
        )
        if sim is None:
            continue

        all_sim.append(sim)
        all_signals.extend(signals)
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

        if args.save_results:
            out = {
                "config": {
                    "top_n": args.top_n, "ndrop": args.ndrop,
                    "pv_factors": pv_active, "fund_factors": fund_active,
                    "start": args.start, "end": args.end,
                },
                "summary": summary.to_dict(orient="records"),
                "daily_signals": all_signals,
            }
            for rec in out["summary"]:
                for k, v in rec.items():
                    if hasattr(v, "item"):
                        rec[k] = v.item()
            with open(args.save_results, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"\n结果已保存至: {args.save_results}")


if __name__ == "__main__":
    main()
