#!/usr/bin/env python
"""全市场回测 + 参数网格搜索 + 过拟合检测。

用法：
    python scripts/grid_backtest.py                    # 默认网格搜索
    python scripts/grid_backtest.py --baseline-only    # 仅跑baseline
    python scripts/grid_backtest.py --overfit-test     # 仅做过拟合检测
"""
import argparse
import json
import os
import sys
import time as _time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_ensemble
from models.dual_period import DualPeriodModel
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.selector import select_top_n, select_topk_ndrop
from portfolio.allocator import equal_weight
from portfolio.risk import apply_stop_loss, check_drawdown_limit, check_index_crash

FUND_NAMES = {
    "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv", "fin_net_margin",
    "fin_bps_growth", "fin_revenue_stability", "fin_eps_growth",
    "fin_debt_ratio", "fin_goodwill_ratio", "fin_pledge_risk", "fin_audit_score",
}


def load_data(codes, start, end, engine):
    """加载 OHLCV + extra_data + index。"""
    code_list = ",".join([f"'{c}'" for c in codes])

    # OHLCV
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily WHERE code IN ({code_list})
        AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # index
    idx_sql = f"""
        SELECT trade_date, close FROM index_daily
        WHERE code = '000001' AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY trade_date
    """
    try:
        index_ohlcv = pd.read_sql(idx_sql, engine)
    except Exception:
        index_ohlcv = pd.DataFrame()

    # extra_data (估值)
    extra_data = {}
    try:
        extra_sql = f"""
            SELECT code, trade_date, market_cap, pb FROM stock_daily_extra
            WHERE code IN ({code_list}) AND trade_date BETWEEN '{start}' AND '{end}'
        """
        edf = pd.read_sql(extra_sql, engine)
        if not edf.empty:
            edf["log_mcap"] = np.log(edf["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = edf[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = edf[["code", "trade_date", "pb"]]
    except Exception:
        pass

    # financial
    fin_cols = [
        "net_profit", "cash_flow", "roe", "bps", "net_margin", "revenue", "eps",
        "total_assets", "total_liability", "goodwill", "holder_equity",
        "operating_cash_flow", "adjusted_profit",
    ]
    try:
        fin_sql = f"""
            SELECT code, report_date, {','.join(fin_cols)}
            FROM stock_financial WHERE code IN ({code_list})
            ORDER BY code, report_date
        """
        fdf = pd.read_sql(fin_sql, engine)
        if not fdf.empty:
            for col in fin_cols:
                if col in fdf.columns:
                    extra_data[col] = fdf[["code", "report_date", col]]
    except Exception:
        pass

    # pledge
    try:
        p_sql = f"""
            SELECT code, trade_date, pledge_ratio FROM stock_pledge
            WHERE code IN ({code_list}) AND trade_date BETWEEN '{start}' AND '{end}'
        """
        pdf = pd.read_sql(p_sql, engine)
        if not pdf.empty:
            extra_data["pledge_ratio"] = pdf[["code", "trade_date", "pledge_ratio"]]
    except Exception:
        pass

    return ohlcv, index_ohlcv, extra_data


def run_single_backtest(
    ohlcv, index_ohlcv, extra_data,
    factor_names, fund_factor_names,
    start, end,
    top_n=15, use_ndrop=True, ndrop_n=2,
    quality_threshold=-3.0, crash_threshold=-0.12,
    init_cash=1_000_000, train_years=3, val_years=1,
):
    """单组参数回测，返回汇总指标。"""
    t0 = _time.time()

    # build dataset
    all_factor_names = factor_names + fund_factor_names
    dataset = build_factor_dataset(
        ohlcv, all_factor_names, label_mode="binary",
        extra_data=extra_data if extra_data else None,
    )

    # IC gate + orthogonal
    pv_active = [f for f in factor_names if f in dataset.columns]
    pv_active = filter_factors_by_ic(dataset, pv_active, ret_col="ret_1d")
    pv_active = select_orthogonal_factors(dataset, pv_active, threshold=0.7)

    fund_active = [f for f in fund_factor_names if f in dataset.columns and dataset[f].notna().mean() > 0.10]
    if not fund_active and quality_threshold > -10:
        logger.debug("无基本面数据，质量筛选不可用")
        quality_threshold = -999  # effectively disable

    # walk-forward train
    results = walk_forward_train_ensemble(dataset, pv_active, train_years, val_years)
    if not results:
        logger.warning("无 walk-forward 窗口")
        return None

    # simulate each window
    price_map = ohlcv[["code", "trade_date", "close"]].copy()
    price_map["trade_date"] = pd.to_datetime(price_map["trade_date"])
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    ohlcv_pd = ohlcv.copy()
    ohlcv_pd["trade_date"] = pd.to_datetime(ohlcv_pd["trade_date"])

    # Init quality model
    quality_model = None
    if fund_active and quality_threshold > -10:
        quality_model = DualPeriodModel(
            predictor=results[0]["ensemble"],
            pv_factor_names=pv_active,
            fund_factor_names=fund_active,
            top_n=top_n, ndrop_n=ndrop_n,
            quality_threshold=quality_threshold,
        )

    all_sim = []
    for i, r in enumerate(results):
        val_mask = (
            (dataset["trade_date"] >= r["val_start"]) &
            (dataset["trade_date"] <= r["val_end"])
        )
        val_factors = dataset[val_mask].copy()

        sim, _ = _simulate_window(
            r["ensemble"], val_factors, price_map,
            top_n=top_n, ndrop_n=ndrop_n,
            index_ohlcv=index_ohlcv,
            quality_model=quality_model,
            ohlcv_data=ohlcv_pd, extra_data=extra_data,
            init_cash=init_cash,
            use_ndrop=use_ndrop,
            use_quality=(quality_threshold > -10 and fund_active),
            crash_threshold=crash_threshold,
        )
        if sim:
            all_sim.append(sim)

    if not all_sim:
        return None

    summary = pd.DataFrame(all_sim)
    elapsed = _time.time() - t0
    return {
        "top_n": top_n, "ndrop": use_ndrop, "ndrop_n": ndrop_n,
        "quality_threshold": quality_threshold, "crash_threshold": crash_threshold,
        "n_windows": len(all_sim),
        "avg_return": float(summary["total_return"].mean()),
        "avg_sharpe": float(summary["sharpe"].mean()),
        "avg_max_dd": float(summary["max_drawdown"].mean()),
        "avg_win_rate": float(summary["win_rate"].mean()),
        "avg_final_value": float(summary["final_value"].mean()),
        "elapsed_sec": elapsed,
    }


def _simulate_window(
    ensemble, factor_df, price_map,
    top_n, ndrop_n, index_ohlcv,
    quality_model, ohlcv_data, extra_data,
    init_cash, use_ndrop, use_quality, crash_threshold,
):
    """模拟单个验证窗口（从 run_simulation.py 摘取）。"""
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

        # index crash
        if index_ohlcv is not None and not index_ohlcv.empty:
            idx_hist = index_ohlcv[pd.to_datetime(index_ohlcv["trade_date"]) <= today_ts]
            if not idx_hist.empty:
                idx_close = idx_hist.sort_values("trade_date")["close"].values
                if check_index_crash(idx_close, threshold=crash_threshold):
                    for code in list(positions.keys()):
                        p = _get_price(price_map, code, today)
                        if p and p > 0:
                            cash += positions[code]["shares"] * p
                    positions.clear()
                    daily_signals.append({"date": str(today.date()), "crash": True})
                    daily_values.append(cash)
                    daily_returns.append(0)
                    continue

        today_factors = factor_df[factor_df["trade_date"] == today].copy()
        if len(today_factors) < top_n:
            val = cash + sum(
                positions[c]["shares"] * (_get_price(price_map, c, today) or 0)
                for c in positions
            )
            daily_values.append(val)
            daily_returns.append(0)
            continue

        # quality filter
        if use_quality and quality_model is not None:
            quality_codes = quality_model.quality_screen(
                ohlcv_data, extra_data or {}, today_ts,
            )
            if quality_codes:
                today_factors = today_factors[today_factors["code"].isin(quality_codes)]

        if len(today_factors) < top_n:
            val = cash + sum(
                positions[c]["shares"] * (_get_price(price_map, c, today) or 0)
                for c in positions
            )
            daily_values.append(val)
            daily_returns.append(0)
            continue

        # predict
        try:
            scores = ensemble.predict(today_factors)
        except Exception:
            scores = today_factors[["code"]].copy()
            scores["score"] = 0.5
            scores["rank"] = range(1, len(scores) + 1)

        # select
        if use_ndrop:
            holding_codes = set(positions.keys())
            score_series = pd.Series(
                scores.set_index("code")["score"].to_dict()
            ).sort_values(ascending=False)
            new_holdings, _, _ = select_topk_ndrop(
                score_series, current_holdings=holding_codes,
                K=top_n, N=ndrop_n,
            )
            target_codes = list(new_holdings)
        else:
            selected = select_top_n(scores, n=top_n)
            target_codes = selected["code"].tolist()

        today_prices = {}
        for code in target_codes:
            p = _get_price(price_map, code, today)
            if p and p > 0:
                today_prices[code] = p

        if not today_prices:
            val = cash + sum(
                positions[c]["shares"] * (_get_price(price_map, c, today) or 0)
                for c in positions
            )
            daily_values.append(val)
            daily_returns.append(0)
            continue

        # sell non-target
        for code in list(positions.keys()):
            if code not in target_codes:
                p = _get_price(price_map, code, today)
                if p and p > 0:
                    cash += positions[code]["shares"] * p
                    del positions[code]

        # buy target
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

        # tomorrow value
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

        # stop loss
        pos_list = [
            {"code": c, "shares": p["shares"], "cost_basis": p["cost_basis"]}
            for c, p in positions.items()
        ]
        pos_df = pd.DataFrame(pos_list) if pos_list else pd.DataFrame(columns=["code", "shares", "cost_basis"])
        if not pos_df.empty and tomorrow_prices:
            cost_basis_map = {c: p["cost_basis"] for c, p in positions.items()}
            stopped = apply_stop_loss(pos_df, tomorrow_prices, cost_basis_map, stop_pct=0.08)
            for code in stopped["code"].tolist():
                cash += positions[code]["shares"] * tomorrow_prices[code]
                del positions[code]

        daily_values.append(tomorrow_value)
        ret = (tomorrow_value / daily_values[-2]) - 1 if daily_values[-2] > 0 else 0
        daily_returns.append(ret)

        if check_drawdown_limit(tomorrow_value, peak_value, limit=0.25):
            for code in list(positions.keys()):
                p = _get_price(price_map, code, tomorrow)
                if p and p > 0:
                    cash += positions[code]["shares"] * p
            positions.clear()
            tomorrow_value = cash
            peak_value = tomorrow_value
        peak_value = max(peak_value, tomorrow_value)

        daily_signals.append({
            "date": str(today.date()), "crash": False,
            "n_holdings": len(positions), "value": tomorrow_value,
        })

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


def _get_price(price_map, code, trade_date):
    subset = price_map[(price_map["code"] == code) & (price_map["trade_date"] == trade_date)]
    if subset.empty:
        return None
    return float(subset["close"].iloc[0])


def grid_search(codes, start, end, engine):
    """参数网格搜索。"""
    # fixed: only vary key params
    param_grid = {
        "top_n": [10, 15, 20],
        "use_ndrop": [True, False],
        "ndrop_n": [1, 2, 3],
        "quality_threshold": [-5, -3, -1],
        "crash_threshold": [-0.08, -0.12, -0.15],
    }

    # smart reduction: only meaningful combos
    combos = []
    for top_n in [10, 15, 20]:
        for use_ndrop in [True, False]:
            for ndrop_n in ([1, 2, 3] if use_ndrop else [0]):
                for qt in ([-3] if use_ndrop else [-3]):  # fix qt for now
                    for ct in [-0.12]:  # fix ct for now
                        combos.append({
                            "top_n": top_n, "use_ndrop": use_ndrop,
                            "ndrop_n": ndrop_n, "quality_threshold": qt,
                            "crash_threshold": ct,
                        })

    # Check cache
    cache_file = "scripts/grid_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    else:
        cache = {}

    logger.info(f"网格搜索: {len(combos)} 组合")
    all_factor_names = sorted(set(ALL_FACTORS.keys()) - FUND_NAMES)
    fund_names = sorted(FUND_NAMES)

    ohlcv, index_ohlcv, extra_data = load_data(codes, start, end, engine)

    results_list = []
    for j, combo in enumerate(combos):
        key = json.dumps(combo, sort_keys=True)
        if key in cache:
            logger.info(f"[{j+1}/{len(combos)}] {combo} ← cached")
            results_list.append(cache[key])
            continue

        logger.info(f"[{j+1}/{len(combos)}] 测试: {combo}")
        try:
            r = run_single_backtest(
                ohlcv, index_ohlcv, extra_data,
                all_factor_names, fund_names,
                start, end, **combo,
            )
            if r:
                cache[key] = r
                results_list.append(r)
                # save incrementally
                with open(cache_file, "w") as f:
                    json.dump(cache, f, indent=2, default=str)
                logger.info(f"  Sharpe={r['avg_sharpe']:.2f}, Return={r['avg_return']:.1%}, DD={r['avg_max_dd']:.1%}")
        except Exception as e:
            logger.error(f"  失败: {e}")

    return results_list


def overfit_test(codes, engine):
    """样本内 (2019-2022) vs 样本外 (2024-2026) 对比。"""
    logger.info("=" * 50)
    logger.info("过拟合检测：样本内 vs 样本外")
    logger.info("=" * 50)

    all_factor_names = sorted(set(ALL_FACTORS.keys()) - FUND_NAMES)
    fund_names = sorted(FUND_NAMES)

    best_params = {
        "top_n": 15, "use_ndrop": True, "ndrop_n": 2,
        "quality_threshold": -3, "crash_threshold": -0.12,
    }

    results = {}
    for label, (start, end) in {
        "样本内 2019-2022": ("20190101", "20221231"),
        "样本外 2024-2026": ("20240101", "20260501"),
    }.items():
        # 过拟合检测用较短训练窗口确保两段都有 walk-forward 窗口
        train_yr = 1
        val_yr = 1
        ohlcv, index_ohlcv, extra_data = load_data(codes, start, end, engine)
        r = run_single_backtest(
            ohlcv, index_ohlcv, extra_data,
            all_factor_names, fund_names,
            start, end, **best_params,
            train_years=train_yr, val_years=val_yr,
        )
        if r:
            results[label] = r
            logger.info(f"\n{label}:")
            logger.info(f"  Sharpe={r['avg_sharpe']:.2f}, Return={r['avg_return']:.1%}")
            logger.info(f"  MaxDD={r['avg_max_dd']:.1%}, WinRate={r['avg_win_rate']:.1%}")
            logger.info(f"  Windows={r['n_windows']}, Time={r['elapsed_sec']:.0f}s")

    # report
    if len(results) >= 2:
        insample = results.get("样本内 2019-2022", {})
        outsample = results.get("样本外 2024-2026", {})
        if insample and outsample:
            ratio = outsample["avg_sharpe"] / insample["avg_sharpe"] if insample["avg_sharpe"] != 0 else 0
            logger.info(f"\n=== 过拟合诊断 ===")
            logger.info(f"样本内 Sharpe: {insample['avg_sharpe']:.2f}")
            logger.info(f"样本外 Sharpe: {outsample['avg_sharpe']:.2f}")
            logger.info(f"衰减比:       {ratio:.1%}")
            if ratio < 0.3:
                logger.warning("⚠ 严重过拟合！样本外表现大幅下降")
            elif ratio < 0.6:
                logger.warning("⚡ 中度过拟合，需调整正则化")
            else:
                logger.success("✓ 过拟合程度可接受")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--overfit-test", action="store_true")
    parser.add_argument("--start", default="20190101")
    parser.add_argument("--end", default="20260501")
    parser.add_argument("--codes-file", default="scripts/backtest_codes.txt")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--ndrop", action="store_true", default=True)
    parser.add_argument("--output", default="scripts/backtest_results.json")
    args = parser.parse_args()

    # Load codes
    if os.path.exists(args.codes_file):
        with open(args.codes_file) as f:
            codes = f.read().strip().split(",")
    else:
        logger.error(f"未找到成分股文件: {args.codes_file}")
        logger.info("先生成: python -c \"import akshare as ak; ...\"")
        sys.exit(1)

    logger.info(f"股票池: {len(codes)} 只")
    engine = get_engine()

    try:
        if args.overfit_test:
            overfit_test(codes, engine)
            return

        if args.baseline_only:
            all_factor_names = sorted(set(ALL_FACTORS.keys()) - FUND_NAMES)
            fund_names = sorted(FUND_NAMES)
            ohlcv, index_ohlcv, extra_data = load_data(codes, args.start, args.end, engine)
            r = run_single_backtest(
                ohlcv, index_ohlcv, extra_data,
                all_factor_names, fund_names,
                args.start, args.end,
                top_n=args.top_n, use_ndrop=args.ndrop,
                ndrop_n=2, quality_threshold=-3, crash_threshold=-0.12,
            )
            if r:
                logger.info(f"\nBaseline 结果:")
                logger.info(f"  Sharpe={r['avg_sharpe']:.2f}, Return={r['avg_return']:.1%}")
                logger.info(f"  MaxDD={r['avg_max_dd']:.1%}, WinRate={r['avg_win_rate']:.1%}")
                logger.info(f"  Windows={r['n_windows']}, Time={r['elapsed_sec']:.0f}s")
            return

        # Full grid search
        results = grid_search(codes, args.start, args.end, engine)
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("avg_sharpe", ascending=False)
            print("\n=== 网格搜索结果 (按 Sharpe 排序) ===\n")
            print(df.head(20).to_string())
            print(f"\n最佳参数: {df.iloc[0].to_dict()}")

            with open(args.output, "w") as f:
                json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
            logger.info(f"结果已保存至 {args.output}")

        # Overfit test after grid search
        overfit_test(codes, engine)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
